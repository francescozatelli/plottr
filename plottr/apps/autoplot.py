"""
plottr/apps/autoplot.py : tools for simple automatic plotting.
"""

import logging
import os
import time
import argparse
from typing import Union, Tuple, Optional, Type, List, Any, Type
import numpy as np
from packaging import version

from .. import QtCore, Flowchart, Signal, Slot, QtWidgets, QtGui, config_entry
from .. import log as plottrlog
from ..data.datadict import DataDictBase
from ..data.datadict_storage import DDH5Loader
from ..data.qcodes_dataset import QCodesDSLoader
from ..gui import PlotWindow
from ..gui.widgets import MonitorIntervalInput, SnapshotWidget
from ..node.data_selector import DataSelector
from ..node.dim_reducer import XYSelector
from ..node.dim_reducer import ReductionMethod
from ..node.filter.correct_offset import SubtractAverage
from ..node.scaleunits import ScaleUnits
from ..node.grid import DataGridder, GridOption
from ..node.tools import linearFlowchart
from ..node.node import Node
from ..node.histogram import Histogrammer
from ..plot import PlotNode, makeFlowchartWithPlot, PlotWidget
from ..plot.mpl.autoplot import AutoPlot as MPLAutoPlot
from ..plot.pyqtgraph.autoplot import AutoPlot as PGAutoPlot
from ..utils.misc import unwrap_optional

__author__ = 'Wolfgang Pfaff'
__license__ = 'MIT'


# TODO: * separate logging window

LOGGER = logging.getLogger('plottr.apps.autoplot')


def _axis_unique_count(data: DataDictBase, axis: str) -> int:
    """Estimate axis size from unique finite coordinate values."""
    try:
        vals = np.asarray(data.data_vals(axis), dtype=float).reshape(-1)
    except Exception:
        return 0

    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0
    return int(np.unique(np.round(vals, decimals=12)).size)


def autoplot(inputData: Union[None, DataDictBase] = None,
             plotWidgetClass: Optional[Type[PlotWidget]] = None) \
        -> Tuple[Flowchart, 'AutoPlotMainWindow']:
    """
    Sets up a simple flowchart consisting of a data selector, gridder,
    an xy-axes selector, and creates a GUI together with an autoplot
    widget.

    :returns: the flowchart object and the dialog widget
    """

    nodes: List[Tuple[str, Type[Node]]] = [
        ('Data selection', DataSelector),
        ('Grid', DataGridder),
        ('Dimension assignment', XYSelector),
    ]

    widgetOptions = {
        "Data selection": dict(visible=True,
                               dockArea=QtCore.Qt.TopDockWidgetArea),
        "Dimension assignment": dict(visible=True,
                                     dockArea=QtCore.Qt.TopDockWidgetArea),
    }

    fc = makeFlowchartWithPlot(nodes)
    win = AutoPlotMainWindow(fc, widgetOptions=widgetOptions,
                             plotWidgetClass=plotWidgetClass)
    win.show()

    if inputData is not None:
        win.setInput(data=inputData)

    return fc, win


class UpdateToolBar(QtWidgets.QToolBar):
    """
    A very simple toolbar to enable monitoring or triggering based on a timer.
    Contains a timer whose interval can be set.
    The toolbar will then emit a signal each interval.
    """

    #: Signal emitted after each trigger interval
    trigger = Signal()

    def __init__(self, name: str, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(name, parent)

        self.monitorInput = MonitorIntervalInput()
        self.monitorInput.spin.setMaximum(1000)
        self.monitorInput.setToolTip('Set to 0 for disabling triggering')
        self.monitorInput.intervalChanged.connect(self.setMonitorInterval)
        self.addWidget(self.monitorInput)

        self.monitor = QtCore.QTimer()
        self.monitor.timeout.connect(self.monitorTriggered)

    @Slot()
    def monitorTriggered(self) -> None:
        """
        Is called whenever the monitor timer triggers, and emit the
        :attr:`trigger` Signal.
        """
        LOGGER.debug('Emit trigger')
        self.trigger.emit()

    @Slot(float)
    def setMonitorInterval(self, val: float) -> None:
        """
        Start a background timer that is triggered every `val' seconds.

        :param val: trigger interval in seconds
        """
        self.monitor.stop()
        if val > 0:
            self.monitor.start(int(val * 1000))

        self.monitorInput.spin.setValue(val)

    @Slot()
    def stop(self) -> None:
        """
        Stop the timer.
        """
        self.monitor.stop()


class AutoPlotMainWindow(PlotWindow):

    def __init__(self, fc: Flowchart,
                 parent: Optional[QtWidgets.QMainWindow] = None,
                 monitor: bool = False,
                 monitorInterval: Union[float, None] = None,
                 loaderName: Optional[str] = None,
                 plotWidgetClass: Optional[Type[PlotWidget]] = None,
                 **kwargs: Any):

        super().__init__(parent, fc=fc, plotWidgetClass=plotWidgetClass,
                         **kwargs)

        self.fc = fc
        self.loaderNode: Optional[Node] = None
        if loaderName is not None:
            self.loaderNode = fc.nodes()[loaderName]

        # a flag we use to set reasonable defaults when the first data
        # is processed
        self._initialized = False

        windowTitle = "Plottr | Autoplot"
        self.setWindowTitle(windowTitle)

        # status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # menu bar
        self.menu = self.menuBar()
        self.fileMenu = self.menu.addMenu('&Data')

        if self.loaderNode is not None:
            refreshAction = QtWidgets.QAction('&Refresh', self)
            refreshAction.setShortcut('R')
            refreshAction.triggered.connect(self.refreshData)
            self.fileMenu.addAction(refreshAction)

        # add monitor if needed
        self._userMonitorIntervalSec: float = 0.0
        self._effectiveMonitorIntervalSec: float = 0.0
        if monitor:
            self.monitorToolBar: Optional[UpdateToolBar] = UpdateToolBar('Monitor data')
            self.addToolBar(self.monitorToolBar)
            self.monitorToolBar.trigger.connect(self.refreshData)
            if monitorInterval is not None:
                self.setMonitorInterval(monitorInterval)
        else:
            self.monitorToolBar = None

        # set some sane defaults any time the data is significantly altered.
        if self.loaderNode is not None:
            self.loaderNode.dataFieldsChanged.connect(self.onChangedLoaderData)

    def setMonitorInterval(self, val: float) -> None:
        self._userMonitorIntervalSec = float(max(0.0, val))
        self._applyEffectiveMonitorInterval()

    def _currentGuardMode(self) -> str:
        if self.loaderNode is None:
            return 'normal'
        try:
            data = self.loaderNode.outputValues().get('dataOut')
            if data is None:
                return 'normal'
            mode = str(data.meta_val('plottr_memory_guard_mode') or 'normal').strip().lower()
            if mode in ['', 'off']:
                return 'normal'
            return mode
        except Exception:
            return 'normal'

    def _applyEffectiveMonitorInterval(self) -> None:
        if self.monitorToolBar is None:
            return
        base = float(max(0.0, self._userMonitorIntervalSec))
        mode = self._currentGuardMode()
        factor = float(config_entry('main', 'qcodes', 'refresh_slowdown_factor_emergency', default=3.0))

        effective = base
        if base > 0 and mode == 'emergency' and factor > 1.0:
            effective = base * factor

        self._effectiveMonitorIntervalSec = effective
        self.monitorToolBar.setMonitorInterval(effective)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        When closing the inspectr window, do some house keeping:
        * stop the monitor, if running
        """
        if self.monitorToolBar is not None:
            self.monitorToolBar.stop()
        return super().closeEvent(event)

    def showTime(self) -> None:
        """
        Displays current time and DS info in the status bar.
        """
        tstamp = time.strftime("%Y-%m-%d %H:%M:%S")
        extras = []
        if self.loaderNode is not None:
            try:
                data = self.loaderNode.outputValues().get('dataOut')
                if data is not None:
                    mode = data.meta_val('plottr_memory_guard_mode')
                    stride = data.meta_val('plottr_decimation_stride')
                    if mode not in [None, '', 'off', 'normal']:
                        extras.append(f"guard:{mode}")
                    if stride is not None and int(stride) > 1:
                        extras.append(f"decim x{int(stride)}")
            except Exception:
                pass
        if self._effectiveMonitorIntervalSec > 0:
            extras.append(f"refresh {self._effectiveMonitorIntervalSec:.1f}s")
        suffix = f" | {' | '.join(extras)}" if len(extras) > 0 else ""
        self.status.showMessage(f"loaded: {tstamp}{suffix}")

    @Slot()
    def onChangedLoaderData(self) -> None:
        assert self.loaderNode is not None
        data = self.loaderNode.outputValues()['dataOut']
        if data is not None:
            self.setDefaults(self.loaderNode.outputValues()['dataOut'])

    @Slot()
    def refreshData(self) -> None:
        """
        Refresh the dataset by calling `update' on the dataset loader node.
        """
        if self.loaderNode is not None:

            self.loaderNode.update()
            self.showTime()
            self._applyEffectiveMonitorInterval()

            if not self._initialized and self.loaderNode.nLoadedRecords > 0:
                self.onChangedLoaderData()
                self._initialized = True

    def setInput(self, data: DataDictBase, resetDefaults: bool = True) -> None:
        """
        Set input to the flowchart. Can only be used when no loader node is
        defined.
        """
        if self.loaderNode is not None:
            LOGGER.warning("A loader node is defined. Use that for inserting data.")
        else:
            self.fc.setInput(dataIn=data)
            if resetDefaults or not self._initialized:
                self.setDefaults(data)
                self._initialized = True

    def setDefaults(self, data: DataDictBase) -> None:
        """
        try to set some reasonable defaults so there's a plot right away.
        """
        selected = data.dependents()
        if len(selected) > 0:
            selected = selected[:1]

        axes = data.axes(selected)
        drs = dict()
        if len(axes) >= 2:
            # Prefer fastest-changing axis on x and a non-singleton axis on y.
            sizes = {ax: _axis_unique_count(data, ax) for ax in axes}
            x_ax = axes[-1]
            y_candidates = [ax for ax in axes[:-1] if sizes.get(ax, 0) > 1]
            y_ax = y_candidates[-1] if len(y_candidates) > 0 else axes[0]

            drs = {x_ax: 'x-axis', y_ax: 'y-axis'}

            # Coupled-sweep default: if one extra axis has effective size 1,
            # use it as the default secondary axis.
            singleton_extras = [
                ax for ax in axes
                if ax not in [x_ax, y_ax] and sizes.get(ax, 0) == 1
            ]
            if len(singleton_extras) > 0:
                drs[singleton_extras[0]] = 'y-secondary'

            for ax in axes:
                if ax in drs:
                    continue
                # For 3D+ data, default extra dimensions to averaging so a
                # meaningful plot appears immediately instead of an arbitrary
                # first-slice that can be empty or unrepresentative.
                drs[ax] = (ReductionMethod.average, [], {})
        if len(axes) == 1:
            drs = {axes[0]: 'x-axis'}

        try:
            self.fc.nodes()['Data selection'].selectedData = selected
            self.fc.nodes()['Grid'].grid = GridOption.guessShape, {}
            self.fc.nodes()['Dimension assignment'].dimensionRoles = drs
        # FIXME: this is maybe a bit excessive, but trying to set all the defaults
        #   like this can result in many types of errors.
        #   a better approach would be to inspect the data better and make sure
        #   we can set defaults reliably.
        except:
            pass
        unwrap_optional(self.plotWidget).update()


class QCAutoPlotMainWindow(AutoPlotMainWindow):
    """
    Main Window for autoplotting a qcodes dataset.

    Comes with menu options for refreshing the loaded dataset,
    and a toolbar for enabling live-monitoring/refreshing the loaded
    dataset.
    """

    def __init__(self, fc: Flowchart,
                 parent: Optional[QtWidgets.QMainWindow] = None,
                 pathAndId: Optional[Tuple[str, int]] = None, **kw: Any):

        super().__init__(fc, parent, **kw)

        windowTitle = "Plottr | QCoDeS autoplot"
        if pathAndId is not None:
            path = os.path.abspath(pathAndId[0])
            windowTitle += f" | {os.path.split(path)[1]} [{pathAndId[1]}]"
            pathAndId = path, pathAndId[1]
        self.setWindowTitle(windowTitle)

        if pathAndId is not None and self.loaderNode is not None:
            self.loaderNode.pathAndId = pathAndId

        if self.loaderNode is not None and self.loaderNode.nLoadedRecords > 0:
            self.setDefaults(self.loaderNode.outputValues()['dataOut'])
            self._initialized = True

    def setDefaults(self, data: DataDictBase) -> None:
        super().setDefaults(data)
        import qcodes as qc
        qcodes_support = (version.parse(qc.__version__) >=
                          version.parse("0.20.0"))
        if data.meta_val('qcodes_shape') is not None and qcodes_support:
            self.fc.nodes()['Grid'].grid = GridOption.metadataShape, {}
        else:
            self.fc.nodes()['Grid'].grid = GridOption.guessShape, {}


def autoplotQcodesDataset(log: bool = False,
              pathAndId: Union[Tuple[str, int], None] = None,
              parent: Optional[QtWidgets.QWidget] = None,
              monitor: bool = True,
              showWindow: bool = True,
              widgetOptions: Optional[dict] = None) \
    -> Tuple[Flowchart, QCAutoPlotMainWindow]:
    """
    Sets up a simple flowchart consisting of a data selector,
    an xy-axes selector, and creates a GUI together with an autoplot
    widget.

    returns the flowchart object and the mainwindow widget
    """

    fc = linearFlowchart(
        ('Data loader', QCodesDSLoader),
        ('Data selection', DataSelector),
        ('Grid', DataGridder),
        ('Dimension assignment', XYSelector),
        ('Subtract average', SubtractAverage),
        ('Scale units', ScaleUnits),
        ('plot', PlotNode)
    )

    if widgetOptions is None:
        widgetOptions = {
            "Data selection": dict(visible=True,
                                   dockArea=QtCore.Qt.TopDockWidgetArea),
            "Dimension assignment": dict(visible=True,
                                         dockArea=QtCore.Qt.TopDockWidgetArea),
        }

    win = QCAutoPlotMainWindow(fc, parent=parent, pathAndId=pathAndId,
                               widgetOptions=widgetOptions,
                               monitor=monitor,
                               loaderName='Data loader',
                               plotWidgetClass=PGAutoPlot)
    if showWindow:
        win.show()

    return fc, win


def autoplotDDH5(filepath: str = '',
                 groupname: str = 'data',
                 plotWidgetClass: Optional[Type[PlotWidget]] = None) \
        -> Tuple[Flowchart, AutoPlotMainWindow]:

    fc = linearFlowchart(
        ('Data loader', DDH5Loader),
        ('Data selection', DataSelector),
        ('Grid', DataGridder),
        ('Histogram', Histogrammer),
        ('Dimension assignment', XYSelector),
        ('plot', PlotNode)
    )

    widgetOptions = {
        "Data selection": dict(visible=True,
                               dockArea=QtCore.Qt.TopDockWidgetArea),
        "Histogram": dict(visible=False,
                          dockArea=QtCore.Qt.TopDockWidgetArea),
        "Dimension assignment": dict(visible=True,
                                     dockArea=QtCore.Qt.TopDockWidgetArea),
    }

    win = AutoPlotMainWindow(fc, loaderName='Data loader',
                             widgetOptions=widgetOptions,
                             monitor=True,
                             monitorInterval=0.0,
                             plotWidgetClass=plotWidgetClass)
    win.show()

    fc.nodes()['Data loader'].filepath = filepath
    fc.nodes()['Data loader'].groupname = groupname
    win.refreshData()

    return fc, win


def autoplotDDH5App(*args: Any) -> Tuple[Flowchart, AutoPlotMainWindow]:
    filepath = args[0][0]
    groupname = args[0][1]
    if len(args[0]) > 2 and args[0][2] == "matplotlib":
        return autoplotDDH5(filepath, groupname, MPLAutoPlot)
    elif len(args[0]) > 2 and args[0][2] == "pyqtgraph":
        return autoplotDDH5(filepath, groupname, PGAutoPlot)
    else:
        return autoplotDDH5(filepath, groupname)  # use default backend


def main(f: str, g: str) -> int:
    app = QtWidgets.QApplication([])
    fc, win = autoplotDDH5(f, g)

    return app.exec_()


def script() -> None:
    parser = argparse.ArgumentParser(
        description='plottr autoplot .dd.h5 files.'
    )
    parser.add_argument('--filepath', help='path to .dd.h5 file',
                        default='')
    parser.add_argument('--groupname', help='group in the hdf5 file',
                        default='data')
    args = parser.parse_args()

    main(args.filepath, args.groupname)
