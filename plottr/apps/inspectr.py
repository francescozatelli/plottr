"""
plottr/apps/inspectr.py -- tool for browsing qcodes data.

This module provides a GUI tool to browsing qcodes .db files.
You can drap/drop .db files into the inspectr window, then browse through
datasets by date. The inspectr itself shows some elementary information
about each dataset and you can launch a plotting window that allows visualizing
the data in it.

Note that this tool is essentially only visualizing some basic structure of the
runs contained in the database. It does not to any handling or loading of
data. it relies on the public qcodes API to get its information.
"""

import os
import time
import sys
import argparse
import logging
from typing import Optional, Sequence, List, Dict, Iterable, Union, cast, Tuple, Mapping, Set

from typing_extensions import TypedDict

from numpy import rint
import pandas

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None

from plottr import QtCore, QtWidgets, Signal, Slot, QtGui, Flowchart, config_entry

from .. import log as plottrlog
from ..data.qcodes_dataset import (get_runs_from_db_as_dataframe,
                                   get_runs_from_db_as_dataframe_filtered,
                                   get_ds_structure, load_dataset_from,
                                   ds_to_datadict)
from plottr.gui.widgets import MonitorIntervalInput, FormLayoutWrapper, dictToTreeWidgetItems

from .autoplot import autoplotQcodesDataset, QCAutoPlotMainWindow


__author__ = 'Wolfgang Pfaff'
__license__ = 'MIT'

LOGGER = plottrlog.getLogger('plottr.apps.inspectr')


### Database inspector tool

class DateList(QtWidgets.QListWidget):
    """Displays a list of dates for which there are runs in the database."""

    datesSelected = Signal(list)
    fileDropped = Signal(str)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.setAcceptDrops(True)
        self.setDefaultDropAction(QtCore.Qt.CopyAction)

        self.setSelectionMode(QtWidgets.QListView.ExtendedSelection)
        self.itemSelectionChanged.connect(self.sendSelectedDates)

    @Slot(list)
    def updateDates(self, dates: Sequence[str]) -> None:
        for d in dates:
            if len(self.findItems(d, QtCore.Qt.MatchExactly)) == 0:
                self.insertItem(0, d)

        i = 0
        while i < self.count():
            elem = self.item(i)
            if elem is not None and elem.text() not in dates:
                item = self.takeItem(i)
                del item
            else:
                i += 1

            if i >= self.count():
                break

        self.sortItems(QtCore.Qt.DescendingOrder)

    @Slot()
    def sendSelectedDates(self) -> None:
        selection = [item.text() for item in self.selectedItems()]
        self.datesSelected.emit(selection)

    ### Drag/drop handling
    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if len(urls) == 1:
                url = urls[0]
                if url.isLocalFile():
                    event.accept()
            else:
                event.ignore()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        url = event.mimeData().urls()[0].toLocalFile()
        self.fileDropped.emit(url)

    def mimeTypes(self) -> List[str]:
        return ([
            'text/uri-list',
            'application/x-qabstractitemmodeldatalist',
    ])


class SortableTreeWidgetItem(QtWidgets.QTreeWidgetItem):
    """
    QTreeWidgetItem with an overridden comparator that sorts numerical values
    as numbers instead of sorting them alphabetically.
    """
    def __init__(self, strings: Iterable[str]):
        super().__init__(strings)

    def __lt__(self, other: QtWidgets.QTreeWidgetItem) -> bool:
        col = self.treeWidget().sortColumn()
        text1 = self.text(col)
        text2 = other.text(col)
        try:
            return float(text1) < float(text2)
        except ValueError:
            return text1 < text2


class RunList(QtWidgets.QTreeWidget):
    """Shows the list of runs for a given date selection."""

    cols = ['Run ID', 'Tag', 'Experiment', 'Sample', 'Name', 'Started', 'Completed', 'Records', 'GUID']
    tag_dict = {'': '', 'star': '⭐', 'cross': '❌'}

    runSelected = Signal(int)
    runActivated = Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.setColumnCount(len(self.cols))
        self.setHeaderLabels(self.cols)

        self.itemSelectionChanged.connect(self.selectRun)
        self.itemActivated.connect(self.activateRun)

        self.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.showContextMenu)

    @Slot(QtCore.QPoint)
    def showContextMenu(self, position: QtCore.QPoint) -> None:
        model_index = self.indexAt(position)
        item = self.itemFromIndex(model_index)
        assert item is not None
        current_tag_char = item.text(1)

        menu = QtWidgets.QMenu()

        copy_icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogSaveButton)
        copy_action = menu.addAction(copy_icon, "Copy")

        window = cast(QCodesDBInspector, self.window())
        starAction: QtWidgets.QAction = window.starAction # type: ignore[has-type]

        starAction.setText('Star' if current_tag_char != self.tag_dict['star'] else 'Unstar')
        menu.addAction(starAction)

        crossAction: QtWidgets.QAction = window.crossAction # type: ignore[has-type]
        crossAction.setText('Cross' if current_tag_char != self.tag_dict['cross'] else 'Uncross')
        menu.addAction(crossAction)

        action = menu.exec_(self.mapToGlobal(position))
        if action == copy_action:
            QtWidgets.QApplication.clipboard().setText(item.text(
                model_index.column()))

    def addRun(self, runId: int, **vals: str) -> None:
        lst = [str(runId)]
        tag = vals.get('inspectr_tag', '')
        lst.append(self.tag_dict.get(tag, tag))  # if the tag is not in tag_dict, display in text
        lst.append(vals.get('experiment', ''))
        lst.append(vals.get('sample', ''))
        lst.append(vals.get('name', ''))
        lst.append(vals.get('started_date', '') + ' ' + vals.get('started_time', ''))
        lst.append(vals.get('completed_date', '') + ' ' + vals.get('completed_time', ''))
        lst.append(str(vals.get('records', '')))
        lst.append(vals.get('guid', ''))

        item = SortableTreeWidgetItem(lst)
        self.addTopLevelItem(item)

    def setRuns(self, selection: Mapping[int, Mapping[str, str]], show_only_star: bool, show_also_cross: bool) -> None:
        self.clear()

        # disable sorting before inserting values to avoid performance hit
        self.setSortingEnabled(False)

        for runId, record in selection.items():
            tag = record.get('inspectr_tag', '')
            if show_only_star and tag == '':
                continue
            elif show_also_cross or tag != 'cross':
                self.addRun(runId, **record)

        self.setSortingEnabled(True)

        for i in range(len(self.cols)):
            self.resizeColumnToContents(i)

    def updateRuns(self, selection: Mapping[int, Mapping[str, str]]) -> None:

        run_added = False
        for runId, record in selection.items():
            item = self.findItems(str(runId), QtCore.Qt.MatchExactly)
            if len(item) == 0:
                self.setSortingEnabled(False)
                self.addRun(runId, **record)
                run_added = True
            elif len(item) == 1:
                completed = record.get('completed_date', '') + ' ' + record.get(
                    'completed_time', '')
                if completed != item[0].text(6):
                    item[0].setText(6, completed)

                num_records = str(record.get('records', ''))
                if num_records != item[0].text(7):
                    item[0].setText(7, num_records)
            else:
                raise RuntimeError(f"More than one runs found with runId: "
                                   f"{runId}")

        if run_added:
            self.setSortingEnabled(True)
            for i in range(len(self.cols)):
                self.resizeColumnToContents(i)

    @Slot()
    def selectRun(self) -> None:
        selection = self.selectedItems()
        if len(selection) == 0:
            return

        runId = int(selection[0].text(0))
        self.runSelected.emit(runId)

    @Slot(QtWidgets.QTreeWidgetItem, int)
    def activateRun(self, item: QtWidgets.QTreeWidgetItem, column: int) -> None:
        runId = int(item.text(0))
        self.runActivated.emit(runId)


class RunInfo(QtWidgets.QTreeWidget):
    """widget that shows some more details on a selected run.

    When sending information in form of a dictionary, it will create
    a tree view of that dictionary and display that.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.setHeaderLabels(['Key', 'Value'])
        self.setColumnCount(2)

    @Slot(dict)
    def setInfo(self, infoDict: Dict[str, Union[dict, str]]) -> None:
        self.clear()

        items = dictToTreeWidgetItems(infoDict)
        for item in items:
            self.addTopLevelItem(item)
            item.setExpanded(True)

        self.expandAll()
        for i in range(2):
            self.resizeColumnToContents(i)


class LoadDBProcess(QtCore.QObject):
    """
    Worker object for getting a qcodes db overview as pandas dataframe.
    It's good to have this in a separate thread because it can be a bit slow
    for large databases.
    """
    dbdfLoaded = Signal(object)
    pathSet = Signal()

    def setPath(self, path: str) -> None:
        self.path = path
        self.pathSet.emit()

    def loadDB(self) -> None:
        dbdf = get_runs_from_db_as_dataframe(self.path)
        self.dbdfLoaded.emit(dbdf)


class QCodesDBInspector(QtWidgets.QMainWindow):
    """
    Main window of the inspectr tool.
    """

    #: `Signal ()` -- Emitted when when there's an update to the internally
    #: cached data (the *data base data frame* :)).
    dbdfUpdated = Signal()

    #: Signal (`dict`) -- emitted to communicate information about a given
    #: run to the widget that displays the information
    _sendInfo = Signal(dict)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None,
                 dbPath: Optional[str] = None):
        """Constructor for :class:`QCodesDBInspector`."""
        super().__init__(parent)

        self._plotWindows: Dict[int, WindowDict] = {}
        self._embeddedFlowchart: Optional[Flowchart] = None
        self._embeddedPlotWindow: Optional[QCAutoPlotMainWindow] = None
        self._plottedRunId: Optional[int] = None
        self._suppressSelectionPlot: bool = False

        self.filepath = dbPath
        self.dbdf: Optional[pandas.DataFrame] = None
        self.monitor = QtCore.QTimer()
        self.dbWatcher = QtCore.QFileSystemWatcher(self)
        self.refreshDebounce = QtCore.QTimer(self)
        self.refreshDebounce.setSingleShot(True)
        self._userMonitorIntervalSec: float = 0.0
        self._effectiveMonitorIntervalSec: float = 0.0
        self._lastEmbeddedRedrawTs: float = 0.0

        # flag for determining what has been loaded so far.
        # * None: nothing opened yet.
        # * -1: empty DS open.
        # * any value > 0: run ID from the most recent loading.
        self.latestRunId: Optional[int] = None

        self.setWindowTitle('Plottr | QCoDeS dataset inspectr')

        ### GUI elements

        # Main Selection widgets
        self.dateList = DateList()
        self._selected_dates: Tuple[str, ...] = ()
        self.runList = RunList()
        self.runInfo = RunInfo()
        self.plotPanel = QtWidgets.QWidget(self)
        self.plotPanelLayout = QtWidgets.QVBoxLayout(self.plotPanel)
        self.plotPanelLayout.setContentsMargins(0, 0, 0, 0)
        self.plotPanelLayout.setSpacing(0)
        self.plotPlaceholder = QtWidgets.QLabel('Select a run to plot')
        self.plotPlaceholder.setAlignment(QtCore.Qt.AlignCenter)
        self.plotPanelLayout.addWidget(self.plotPlaceholder)

        browserRightSplitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        browserRightSplitter.addWidget(self.runList)
        browserRightSplitter.addWidget(self.runInfo)
        browserRightSplitter.setSizes([450, 150])

        browserSplitter = QtWidgets.QSplitter()
        browserSplitter.addWidget(self.dateList)
        browserSplitter.addWidget(browserRightSplitter)
        browserSplitter.setSizes([120, 380])

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(browserSplitter)
        splitter.addWidget(self.plotPanel)
        splitter.setSizes([420, 980])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

        # status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

        # toolbar
        self.toolbar = self.addToolBar('Data monitoring')

        # toolbar item: monitor interval
        self.monitorInput = MonitorIntervalInput()
        self.monitorInput.setToolTip('Set to 0 for disabling')
        self.monitorInput.intervalChanged.connect(self.setMonitorInterval)
        self.toolbar.addWidget(self.monitorInput)

        self.toolbar.addSeparator()

        # toolbar item: auto-launch plotting
        self.autoLaunchPlots = FormLayoutWrapper([
            ('Auto-plot new', QtWidgets.QCheckBox())
        ])
        tt = "If checked, and automatic refresh is running, "
        tt += " select and plot new datasets automatically in the right panel."
        self.autoLaunchPlots.setToolTip(tt)
        self.toolbar.addWidget(self.autoLaunchPlots)

        self.showOnlyStarAction = self.toolbar.addAction(RunList.tag_dict['star'])
        self.showOnlyStarAction.setToolTip('Show only starred runs')
        self.showOnlyStarAction.setCheckable(True)
        self.showOnlyStarAction.triggered.connect(self.updateRunList)
        self.showAlsoCrossAction = self.toolbar.addAction(RunList.tag_dict['cross'])
        self.showAlsoCrossAction.setToolTip('Show also crossed runs')
        self.showAlsoCrossAction.setCheckable(True)
        self.showAlsoCrossAction.triggered.connect(self.updateRunList)

        # menu bar
        menu = self.menuBar()
        fileMenu = menu.addMenu('&File')

        # action: load db file
        loadAction = QtWidgets.QAction('&Load', self)
        loadAction.setShortcut('Ctrl+L')
        loadAction.triggered.connect(self.loadDB)
        fileMenu.addAction(loadAction)

        # action: updates from the db file
        refreshAction = QtWidgets.QAction('&Refresh', self)
        refreshAction.setShortcut('R')
        refreshAction.triggered.connect(self.refreshDB)
        fileMenu.addAction(refreshAction)

        # action: star/unstar the selected run
        self.starAction = QtWidgets.QAction()
        self.starAction.setShortcut('Ctrl+Alt+S')
        self.starAction.triggered.connect(self.starSelectedRun)
        self.addAction(self.starAction)

        # action: cross/uncross the selected run
        self.crossAction = QtWidgets.QAction()
        self.crossAction.setShortcut('Ctrl+Alt+X')
        self.crossAction.triggered.connect(self.crossSelectedRun)
        self.addAction(self.crossAction)

        # sizing
        scaledSize = int(640 * rint(self.logicalDpiX() / 96.0))
        self.resize(scaledSize, scaledSize)

        ### Thread workers

        # DB loading. can be slow, so nice to have in a thread.
        self.loadDBProcess = LoadDBProcess()
        self.loadDBThread = QtCore.QThread()
        self.loadDBProcess.moveToThread(self.loadDBThread)
        self.loadDBProcess.pathSet.connect(self.loadDBThread.start)
        self.loadDBProcess.dbdfLoaded.connect(self.DBLoaded)
        self.loadDBProcess.dbdfLoaded.connect(self.loadDBThread.quit)
        self.loadDBThread.started.connect(self.loadDBProcess.loadDB)

        ### connect signals/slots

        self.dbdfUpdated.connect(self.updateDates)
        self.dbdfUpdated.connect(self.showDBPath)

        self.dateList.datesSelected.connect(self.setDateSelection)
        self.dateList.fileDropped.connect(self.loadFullDB)
        self.runList.runSelected.connect(self.setRunSelection)
        self.runList.runSelected.connect(self._plotSelectedRun)
        self.runList.runActivated.connect(self.plotRun)
        self._sendInfo.connect(self.runInfo.setInfo)
        self.monitor.timeout.connect(self.monitorTriggered)
        self.dbWatcher.fileChanged.connect(self.onDBFileChanged)
        self.refreshDebounce.timeout.connect(self.refreshDB)

        if self.filepath is not None:
            self.loadFullDB(self.filepath)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        """
        When closing the inspectr window, do some house keeping:
        * stop the monitor, if running
        * close all plot windows
        """

        if self.monitor.isActive():
            self.monitor.stop()

        if self.refreshDebounce.isActive():
            self.refreshDebounce.stop()

        for _runId, info in self._plotWindows.items():
            info['window'].close()
        self._plotWindows.clear()

        if self._embeddedPlotWindow is not None:
            self._embeddedPlotWindow.close()
            self._embeddedPlotWindow.deleteLater()
            self._embeddedPlotWindow = None
            self._embeddedFlowchart = None

    @Slot()
    def showDBPath(self) -> None:
        tstamp = time.strftime("%Y-%m-%d %H:%M:%S")
        assert self.filepath is not None
        path = os.path.abspath(self.filepath)
        extras = self._statusExtras()
        suffix = f" | {extras}" if extras else ""
        self.status.showMessage(f"{path} (loaded: {tstamp}){suffix}")

    def _statusExtras(self) -> str:
        extras: List[str] = []
        if psutil is not None:
            try:
                rss_mb = psutil.Process(os.getpid()).memory_info().rss / (1024.0 * 1024.0)
                extras.append(f"RSS {rss_mb:.0f} MB")
            except Exception:
                pass

        if self._embeddedPlotWindow is not None and self._embeddedPlotWindow.loaderNode is not None:
            try:
                data = self._embeddedPlotWindow.loaderNode.outputValues().get('dataOut')
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

        return " | ".join(extras)

    def _currentGuardMode(self) -> str:
        if self._embeddedPlotWindow is None or self._embeddedPlotWindow.loaderNode is None:
            return 'normal'

        try:
            data = self._embeddedPlotWindow.loaderNode.outputValues().get('dataOut')
            if data is None:
                return 'normal'
            mode = str(data.meta_val('plottr_memory_guard_mode') or 'normal').strip().lower()
            if mode in ['', 'off']:
                return 'normal'
            return mode
        except Exception:
            return 'normal'

    def _applyEffectiveMonitorInterval(self) -> None:
        base = float(max(0.0, self._userMonitorIntervalSec))
        mode = self._currentGuardMode()
        factor = float(config_entry('main', 'qcodes', 'refresh_slowdown_factor_emergency', default=3.0))

        effective = base
        if base > 0 and mode == 'emergency' and factor > 1.0:
            effective = base * factor

        self._effectiveMonitorIntervalSec = effective

        self.monitor.stop()
        if effective > 0:
            self.monitor.start(int(effective * 1000))

    ### loading the DB and populating the widgets
    @Slot()
    def loadDB(self) -> None:
        """
        Open a file dialog that allows selecting a .db file for loading.
        If a file is selected, opens the db.
        """
        if self.filepath is not None:
            curdir = os.path.split(self.filepath)[0]
        else:
            curdir = os.getcwd()

        path, _fltr = QtWidgets.QFileDialog.getOpenFileName(
            self,
            'Open qcodes .db file',
            curdir,
            'qcodes .db files (*.db);;all files (*.*)',
            )

        if path:
            LOGGER.info(f"Opening: {path}")
            self.loadFullDB(path=path)

    def loadFullDB(self, path: Optional[str] = None) -> None:
        if path is not None and path != self.filepath:
            self.filepath = path

            # makes sure we treat a newly loaded file fresh and not as a
            # refreshed one.
            self.latestRunId = None

        self._watchCurrentDBFile()

        if self.filepath is not None:
            if not self.loadDBThread.isRunning():
                self.loadDBProcess.setPath(self.filepath)

    def _watchCurrentDBFile(self) -> None:
        if self.filepath is None:
            return

        fpath = os.path.abspath(self.filepath)
        old_paths = list(self.dbWatcher.files())
        for old in old_paths:
            if os.path.abspath(old) != fpath:
                self.dbWatcher.removePath(old)

        if os.path.isfile(fpath) and fpath not in self.dbWatcher.files():
            self.dbWatcher.addPath(fpath)

    def _showNewRuns(self, new_run_ids: Iterable[int]) -> None:
        if not self.autoLaunchPlots.elements['Auto-plot new'].isChecked():
            return

        new_ids = list(new_run_ids)
        if len(new_ids) == 0:
            return

        newest = int(max(new_ids))

        # Ensure the newest run date is part of the active date filter so the
        # item becomes selectable in the list immediately.
        if self.dbdf is not None and newest in self.dbdf.index:
            started_date = str(self.dbdf.at[newest, 'started_date'])
            if started_date not in self._selected_dates:
                self.setDateSelection(tuple(sorted(set(self._selected_dates + (started_date,)))))

        if not self._selectRunInList(newest):
            self.plotRun(newest)

    def DBLoaded(self, dbdf: pandas.DataFrame) -> None:
        if self.dbdf is not None and dbdf.equals(self.dbdf):
            LOGGER.debug('DB reloaded with no changes. Skipping update')
            return None
        self.dbdf = dbdf
        self.dbdfUpdated.emit()
        self.dateList.sendSelectedDates()
        LOGGER.debug('DB reloaded')

        if self.latestRunId is not None:
            idxs = self.dbdf.index.values
            newIdxs = idxs[idxs > self.latestRunId]
            self._showNewRuns(newIdxs)

    @Slot()
    def updateDates(self) -> None:
        assert self.dbdf is not None
        if self.dbdf.size > 0:
            dates = list(self.dbdf.groupby('started_date').indices.keys())
            self.dateList.updateDates(dates)

    ### reloading the db
    @Slot()
    def refreshDB(self) -> None:
        if self.filepath is None:
            return

        if self.loadDBThread.isRunning():
            return

        if self.dbdf is None or self.dbdf.size == 0:
            self.latestRunId = -1
            self.loadFullDB()
            return

        latest_run_id = int(self.dbdf.index.values.max())
        self.latestRunId = latest_run_id

        active_run_ids: Set[int] = set()
        if self._plottedRunId is not None:
            active_run_ids.add(self._plottedRunId)
        selected_items = self.runList.selectedItems()
        if len(selected_items) > 0:
            active_run_ids.add(int(selected_items[0].text(0)))

        dbdf_delta = get_runs_from_db_as_dataframe_filtered(
            self.filepath,
            min_run_id=latest_run_id,
            run_ids=active_run_ids,
        )

        if dbdf_delta.size == 0:
            # Even when the overview table does not change, the currently
            # plotted run may have new data points appended. Refresh the
            # embedded plot in place so live plotting stays reactive.
            self._refreshEmbeddedPlot()
            self._applyEffectiveMonitorInterval()
            self.showDBPath()
            return

        self.dbdf = pandas.concat([
            self.dbdf.loc[~self.dbdf.index.isin(dbdf_delta.index)],
            dbdf_delta,
        ]).sort_index()

        self._suppressSelectionPlot = True
        try:
            self.dbdfUpdated.emit()
            self.dateList.sendSelectedDates()
        finally:
            self._suppressSelectionPlot = False

        new_ids = dbdf_delta.index.values
        new_ids = new_ids[new_ids > latest_run_id]
        self._showNewRuns(new_ids)

        # Keep the currently plotted run live-updated without requiring
        # reselection from the run list.
        self._refreshEmbeddedPlot(updated_ids=set(int(i) for i in dbdf_delta.index.values))
        self._applyEffectiveMonitorInterval()
        self.showDBPath()

    @Slot(float)
    def setMonitorInterval(self, val: float) -> None:
        self._userMonitorIntervalSec = float(max(0.0, val))
        self._applyEffectiveMonitorInterval()

        if abs(float(self.monitorInput.spin.value()) - float(val)) > 1e-9:
            self.monitorInput.spin.blockSignals(True)
            self.monitorInput.spin.setValue(val)
            self.monitorInput.spin.blockSignals(False)

    @Slot()
    def monitorTriggered(self) -> None:
        LOGGER.debug('Refreshing DB')
        self.refreshDB()
        self._applyEffectiveMonitorInterval()

    @Slot(str)
    def onDBFileChanged(self, path: str) -> None:
        if self.filepath is None:
            return

        watched = os.path.abspath(self.filepath)
        changed = os.path.abspath(path)
        if watched != changed:
            return

        self._watchCurrentDBFile()

        # Debounce bursty DB write events to keep refresh responsive.
        self.refreshDebounce.start(300)

    @Slot()
    def updateRunList(self) -> None:
        if self.dbdf is None:
            return
        selection = self.dbdf.loc[self.dbdf['started_date'].isin(self._selected_dates)].sort_index(ascending=False)
        show_only_star = self.showOnlyStarAction.isChecked()
        show_also_cross = self.showAlsoCrossAction.isChecked()
        # Pandas types cannot infer that this dataframe will be
        # using int as index and Dict[str, str] as keys
        selection_dict = cast(Dict[int, Dict[str,str]], selection.to_dict(orient='index'))
        self.runList.setRuns(selection_dict, show_only_star, show_also_cross)

    @Slot(int)
    def _plotSelectedRun(self, runId: int) -> None:
        if self._suppressSelectionPlot:
            return
        self.plotRun(runId)

    ### handling user selections
    @Slot(list)
    def setDateSelection(self, dates: Sequence[str]) -> None:
        if len(dates) > 0:
            assert self.dbdf is not None
            selection = self.dbdf.loc[self.dbdf['started_date'].isin(dates)].sort_index(ascending=False)
            old_dates = self._selected_dates
            prev_selected_run_id: Optional[int] = None
            selected_items = self.runList.selectedItems()
            if len(selected_items) > 0:
                prev_selected_run_id = int(selected_items[0].text(0))

            # Internal run-list refreshes should not trigger re-plot via
            # runSelected. Re-enable signals after list update is stable.
            signals_were_blocked = self.runList.blockSignals(True)
            # Pandas types cannot infer that this dataframe will be
            # using int as index and Dict[str, str] as keys
            selection_dict = cast(Dict[int, Dict[str,str]], selection.to_dict(orient='index'))
            if not all(date in old_dates for date in dates):
                show_only_star = self.showOnlyStarAction.isChecked()
                show_also_cross = self.showAlsoCrossAction.isChecked()
                self.runList.setRuns(selection_dict, show_only_star, show_also_cross)
            else:
                self.runList.updateRuns(selection_dict)

            if prev_selected_run_id is not None:
                existing = self.runList.findItems(str(prev_selected_run_id), QtCore.Qt.MatchExactly)
                if len(existing) > 0:
                    self.runList.setCurrentItem(existing[0])

            self.runList.blockSignals(signals_were_blocked)
            self._selected_dates = tuple(dates)
        else:
            self._selected_dates = ()
            self.runList.clear()

    @Slot(int)
    def setRunSelection(self, runId: int) -> None:
        assert self.filepath is not None
        ds = load_dataset_from(self.filepath, runId)
        snap = None
        if hasattr(ds, 'snapshot'):
            snap = ds.snapshot

        structure = cast(Dict[str, dict], get_ds_structure(ds))
        # cast away typed dict so we can pop a key
        for k, v in structure.items():
            v.pop('values')
        contentInfo = {'Data structure': structure,
                       'Metadata': ds.metadata,
                       'QCoDeS Snapshot': snap}
        self._sendInfo.emit(contentInfo)

    def _ensureEmbeddedPlotWindow(self, runId: int) -> QCAutoPlotMainWindow:
        assert self.filepath is not None

        if self._embeddedPlotWindow is None:
            # Keep monitor disabled here; inspectr handles refresh cycles.
            fc, win = autoplotQcodesDataset(
                pathAndId=(self.filepath, runId),
                parent=self.plotPanel,
                monitor=False,
                showWindow=False,
                widgetOptions={
                    "Data selection": dict(visible=True),
                    "Dimension assignment": dict(visible=True),
                },
            )
            win.setWindowFlags(QtCore.Qt.Widget)
            win.menuBar().hide()

            self._embeddedFlowchart = fc
            self._embeddedPlotWindow = win
            self.plotPanelLayout.removeWidget(self.plotPlaceholder)
            self.plotPlaceholder.hide()
            self.plotPanelLayout.addWidget(win)
            win.show()

        return self._embeddedPlotWindow

    def _selectRunInList(self, runId: int) -> bool:
        items = self.runList.findItems(str(runId), QtCore.Qt.MatchExactly)
        if len(items) == 0:
            return False

        self.runList.setCurrentItem(items[0])
        self.runList.scrollToItem(items[0], QtWidgets.QAbstractItemView.PositionAtCenter)
        return True

    def _refreshEmbeddedPlot(self, updated_ids: Optional[Set[int]] = None) -> None:
        """Refresh currently embedded plot in place for live updates.

        If ``updated_ids`` is provided, only refresh when the plotted run
        appears in that set.
        """
        if self._embeddedPlotWindow is None or self._plottedRunId is None:
            return

        if updated_ids is not None and self._plottedRunId not in updated_ids:
            return

        max_fps = float(config_entry('main', 'qcodes', 'plot_refresh_max_fps', default=8.0))
        if max_fps > 0:
            now = time.monotonic()
            min_dt = 1.0 / max_fps
            if (now - self._lastEmbeddedRedrawTs) < min_dt:
                return
            self._lastEmbeddedRedrawTs = now

        win = self._embeddedPlotWindow
        if win.loaderNode is None:
            return

        # Do not reset defaults or selected variables; just pull newly appended
        # records for the currently visible run.
        win.refreshData()
        self.showDBPath()

    @Slot(int)
    def plotRun(self, runId: int) -> None:
        assert self.filepath is not None
        win = self._ensureEmbeddedPlotWindow(runId)
        if win.loaderNode is None:
            return

        try:
            win.fc.nodes()['Dimension assignment'].dimensionRoles = {}
            win.fc.nodes()['Data selection'].selectedData = []
        except Exception:
            pass

        # When switching runs, force a fresh dataset load so data-field
        # dependent UIs (data selection and dimension assignment) are rebuilt.
        if self._plottedRunId != runId:
            win.loaderNode.nLoadedRecords = 0
            win.loaderNode._dataset = None

        win.loaderNode.pathAndId = (self.filepath, runId)
        win._initialized = False
        win.refreshData()
        data_out = win.loaderNode.outputValues().get('dataOut')

        # Fallback for cases where loader update does not emit new data due to
        # record-count semantics; still rebuild selector options for this run.
        if data_out is None and win.loaderNode._dataset is not None:
            data_out = ds_to_datadict(win.loaderNode._dataset)

        if data_out is not None:
            shapes = data_out.shapes()
            dtype = type(data_out)
            try:
                data_sel_node = win.fc.nodes()['Data selection']
                if data_sel_node.ui is not None:
                    data_sel_node.ui.setData(data_out, shapes, dtype)
            except Exception:
                pass
            try:
                dim_node = win.fc.nodes()['Dimension assignment']
                if dim_node.ui is not None:
                    dim_node.ui.setData(data_out, shapes, dtype)
            except Exception:
                pass

            win.setDefaults(data_out)
            win._initialized = True
        win.showTime()

        self._plottedRunId = runId
        self.showDBPath()

    def setTag(self, item: QtWidgets.QTreeWidgetItem, tag: str) -> None:
        # set tag in the database
        assert self.filepath is not None
        runId = int(item.text(0))
        ds = load_dataset_from(self.filepath, runId)
        ds.add_metadata('inspectr_tag', tag)

        # set tag in self.dbdf
        assert self.dbdf is not None
        self.dbdf.at[runId, 'inspectr_tag'] = tag

        # set tag in the GUI
        tag_char = self.runList.tag_dict[tag]
        item.setText(1, tag_char)

        # refresh the RunInfo widget
        self.setRunSelection(runId)

    def tagSelectedRun(self, tag: str) -> None:
        for item in self.runList.selectedItems():
            current_tag_char = item.text(1)
            tag_char = self.runList.tag_dict[tag]
            if current_tag_char == tag_char:  # if already tagged
                self.setTag(item, '')  # clear tag
            else:  # if not tagged
                self.setTag(item, tag)  # set tag

    @Slot()
    def starSelectedRun(self) -> None:
        self.tagSelectedRun('star')

    @Slot()
    def crossSelectedRun(self) -> None:
        self.tagSelectedRun('cross')


class WindowDict(TypedDict):
    flowchart: Flowchart
    window: QCAutoPlotMainWindow


def inspectr(dbPath: Optional[str] = None) -> QCodesDBInspector:
    win = QCodesDBInspector(dbPath=dbPath)
    return win


def main(dbPath: Optional[str], log_level: Union[int, str] = logging.WARNING) -> None:
    app = QtWidgets.QApplication([])
    plottrlog.enableStreamHandler(True, log_level)

    win = inspectr(dbPath=dbPath)
    win.show()

    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        appinstance = QtWidgets.QApplication.instance()
        assert appinstance is not None
        appinstance.exec_()


def script() -> None:
    parser = argparse.ArgumentParser(description='inspectr -- sifting through qcodes data.')
    parser.add_argument('--dbpath', help='path to qcodes .db file',
                        default=None)
    parser.add_argument("--console-log-level",
                        choices=("ERROR", "WARNING", "INFO", "DEBUG"),
                        default="WARNING")
    args = parser.parse_args()
    main(args.dbpath, args.console_log_level)
