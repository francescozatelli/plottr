"""Convenience tools for generating ``pyqtgraph`` plots that can
be used in plottr's automatic plotting framework."""

from typing import Optional, Tuple, NoReturn

import numpy as np
import pyqtgraph as pg

from plottr import QtCore, QtWidgets, Signal, config_entry

__all__ = ['PlotBase', 'Plot']


class PlotBase(QtWidgets.QWidget):
    """A simple convenience widget class as container for ``pyqtgraph`` plots.

    The widget contains a layout that contains a ``GraphicsLayoutWidget``.
    This is handy because a plot may contain multiple elements (like an image
    and a colorbar).

    This base class should be inherited to use.
    """

    #: emitted when cursor moves within plot view coordinates
    cursorPositionChanged = Signal(object, float, float, object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        #: central layout of the widget. only contains a graphics layout.
        layout = QtWidgets.QHBoxLayout(self)
        #: ``pyqtgraph`` graphics layout
        self.graphicsLayout = pg.GraphicsLayoutWidget(self)

        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setLayout(layout)
        layout.addWidget(self.graphicsLayout)

        #: ``pyqtgraph`` plot item
        self.plot: pg.PlotItem = self.graphicsLayout.addPlot()
        self.plot.scene().sigMouseMoved.connect(self._onMouseMoved)

    def _onMouseMoved(self, pos: QtCore.QPointF) -> None:
        if not self.plot.vb.sceneBoundingRect().contains(pos):
            return
        mouse_point = self.plot.vb.mapSceneToView(pos)
        self.cursorPositionChanged.emit(self, float(mouse_point.x()), float(mouse_point.y()), None)

    def clearPlot(self) -> None:
        """Clear all plot contents (but do not delete plot elements, like axis
        spines, insets, etc).

        To be implemented by inheriting classes."""
        raise NotImplementedError


class Plot(PlotBase):
    """A simple plot with a single ``PlotItem``."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        legend = self.plot.addLegend(offset=(5, 5), pen='#999',
                                     brush=(255, 255, 255, 150))
        legend.layout.setContentsMargins(0, 0, 0, 0)
        self.plot.showGrid(True, True)

    def clearPlot(self) -> None:
        """Clear the plot item."""
        self.plot.clear()


class PlotWithColorbar(PlotBase):
    """Plot containing a plot item and a colorbar item.

    Plot is suited for either an image plot (:meth:`.setImage`) or a color
    scatter plot (:meth:`.setScatter2D`).
    The color scale is displayed in an interactive colorbar.
    """
    #: colorbar
    colorbar: pg.ColorBarItem

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)

        cmap_name = config_entry('main', 'pyqtgraph', 'default_colormap')
        cmap = self._getColormap(cmap_name)

        self.colorbar: pg.ColorBarItem = pg.ColorBarItem(interactive=True, values=(0, 1),
                                                         colorMap=cmap, width=15)
        self.graphicsLayout.addItem(self.colorbar)

        self.img: Optional[pg.ImageItem] = None
        self.scatter: Optional[pg.ScatterPlotItem] = None
        self.scatterZVals: Optional[np.ndarray] = None
        self.scatterXVals: Optional[np.ndarray] = None
        self.scatterYVals: Optional[np.ndarray] = None
        self.imageXVals: Optional[np.ndarray] = None
        self.imageYVals: Optional[np.ndarray] = None
        self.imageZVals: Optional[np.ndarray] = None
        self.imageRect: Optional[QtCore.QRectF] = None

    def _getColormap(self, name: str) -> pg.ColorMap:
        # Prefer pyqtgraph-native maps, then matplotlib (for names like magma_r),
        # and finally a vivid fallback to avoid accidental grayscale defaults.
        try:
            return pg.colormap.get(name)
        except Exception:
            pass

        try:
            return pg.colormap.getFromMatplotlib(name)
        except Exception:
            pass

        try:
            import matplotlib.cm as mpl_cm
            mpl_cmap = mpl_cm.get_cmap(name)
            sample = np.linspace(0.0, 1.0, 256)
            rgba = mpl_cmap(sample)
            return pg.ColorMap(sample, rgba)
        except Exception:
            pass

        try:
            return pg.colormap.get('magma')
        except Exception:
            return pg.colormap.get('viridis')

    def setColormap(self, name: str) -> None:
        self.colorbar.setColorMap(self._getColormap(name))

    def clearPlot(self) -> None:
        """Clear the content of the plot."""
        self.img = None
        self.scatter = None
        self.scatterZVals = None
        self.scatterXVals = None
        self.scatterYVals = None
        self.imageXVals = None
        self.imageYVals = None
        self.imageZVals = None
        self.imageRect = None
        self.plot.clear()
        try:
            self.colorbar.sigLevelsChanged.disconnect(self._colorScatterPoints)
        except TypeError:
            pass

    def setImage(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Set data to be plotted as image.

        Clears the plot before creating a new image item that gets places in the
        plot and linked to the colorscale.

        :param x: x coordinates (as 2D meshgrid)
        :param y: y coordinates (as 2D meshgrid)
        :param z: data values (as 2D meshgrid)
        :return: None
        """
        self.clearPlot()

        self.img = pg.ImageItem()
        self.plot.addItem(self.img)
        self.img.setImage(z)
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        x_finite = x_arr[np.isfinite(x_arr)]
        y_finite = y_arr[np.isfinite(y_arr)]
        if x_finite.size > 0 and y_finite.size > 0:
            xmin, xmax = float(x_finite.min()), float(x_finite.max())
            ymin, ymax = float(y_finite.min()), float(y_finite.max())
            rect = QtCore.QRectF(xmin, ymin, xmax - xmin, ymax - ymin)
            self.img.setRect(rect)
            self.imageRect = rect

        self.imageZVals = np.asarray(z)
        if np.asarray(x).ndim == 2:
            self.imageXVals = np.asarray(x)[0, :]
        else:
            self.imageXVals = np.asarray(x)

        if np.asarray(y).ndim == 2:
            self.imageYVals = np.asarray(y)[:, 0]
        else:
            self.imageYVals = np.asarray(y)

        self.colorbar.setImageItem(self.img)
        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            z_min, z_max = 0.0, 1.0
        self.colorbar.rounding = (z_max - z_min) * 1e-2
        self.colorbar.setLevels((z_min, z_max))
        self.plot.enableAutoRange(axis='xy', enable=True)
        self.plot.autoRange()

    def setScatter2d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
        """Set data to be plotted as image.

        Clears the plot before creating a new scatter item (based on flattened
        input data) that gets placed in the plot and linked to the colorscale.

        :param x: x coordinates
        :param y: y coordinates
        :param z: data values
        :return: None
        """
        self.clearPlot()

        x_flat = x.flatten()
        y_flat = y.flatten()
        z_flat = z.flatten()

        max_points = config_entry('main', 'pyqtgraph',
                                  'adaptive_color_scatter_max_points',
                                  default=80000)
        if x_flat.size > max_points:
            stride = int(np.ceil(x_flat.size / max_points))
            x_flat = x_flat[::stride]
            y_flat = y_flat[::stride]
            z_flat = z_flat[::stride]

        self.scatter = pg.ScatterPlotItem()
        self.scatter.setData(x=x_flat, y=y_flat, symbol='o', size=10, pen=None)
        self.plot.addItem(self.scatter)
        self.scatterXVals = x_flat
        self.scatterYVals = y_flat
        self.scatterZVals = z_flat

        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            z_min, z_max = 0.0, 1.0
        self.colorbar.setLevels((z_min, z_max))
        self.colorbar.rounding = (z_max - z_min) * 1e-2
        self._colorScatterPoints(self.colorbar)

        self.colorbar.sigLevelsChanged.connect(self._colorScatterPoints)
        self.plot.enableAutoRange(axis='xy', enable=True)
        self.plot.autoRange()

    # TODO: this seems crazy slow.
    def _colorScatterPoints(self, cbar: pg.ColorBarItem) -> None:
        if self.scatter is not None and self.scatterZVals is not None:
            z_norm = self._normalizeColors(self.scatterZVals, cbar.levels())
            cmap = None
            if hasattr(self.colorbar, 'colorMap'):
                cmap = self.colorbar.colorMap()
            elif hasattr(self.colorbar, 'cmap'):
                cmap = self.colorbar.cmap

            if cmap is None:
                cmap = self._getColormap(config_entry('main', 'pyqtgraph', 'default_colormap'))

            colors = cmap.mapToQColor(z_norm)
            self.scatter.setBrush(colors)

    def _normalizeColors(self, z: np.ndarray, levels: Tuple[float, float]) -> np.ndarray:
        scale = levels[1] - levels[0]
        if scale > 0:
            return (z - levels[0]) / scale
        else:
            return np.ones(z.size) * 0.5

    def _onMouseMoved(self, pos: QtCore.QPointF) -> None:
        if not self.plot.vb.sceneBoundingRect().contains(pos):
            return

        mouse_point = self.plot.vb.mapSceneToView(pos)
        x = float(mouse_point.x())
        y = float(mouse_point.y())
        z_val: Optional[float] = None

        if self.imageZVals is not None and self.imageXVals is not None and self.imageYVals is not None:
            if self.img is not None and self.imageRect is not None and self.imageZVals.ndim == 2:
                rect = self.imageRect
                if rect.width() > 0 and rect.height() > 0 and rect.contains(QtCore.QPointF(x, y)):
                    nx = self.imageZVals.shape[1]
                    ny = self.imageZVals.shape[0]
                    tx = (x - rect.left()) / rect.width()
                    ty = (y - rect.top()) / rect.height()
                    ix = int(np.clip(np.rint(tx * (nx - 1)), 0, nx - 1))
                    iy = int(np.clip(np.rint(ty * (ny - 1)), 0, ny - 1))
                    z = self.imageZVals[iy, ix]
                    if np.isfinite(z):
                        z_val = float(z)

        elif self.scatterZVals is not None and self.scatterXVals is not None and self.scatterYVals is not None:
            if self.scatterXVals.size > 0:
                x_min = float(np.nanmin(self.scatterXVals))
                x_max = float(np.nanmax(self.scatterXVals))
                y_min = float(np.nanmin(self.scatterYVals))
                y_max = float(np.nanmax(self.scatterYVals))
                if x_min <= x <= x_max and y_min <= y <= y_max:
                    x_span = max(x_max - x_min, 1e-15)
                    y_span = max(y_max - y_min, 1e-15)
                    dx = (self.scatterXVals - x) / x_span
                    dy = (self.scatterYVals - y) / y_span
                    dist2 = dx * dx + dy * dy

                    k = min(8, dist2.size)
                    idx = np.argpartition(dist2, k - 1)[:k]
                    d = np.sqrt(dist2[idx])
                    z_near = self.scatterZVals[idx]
                    finite = np.isfinite(z_near)
                    if np.any(finite):
                        d = d[finite]
                        z_near = z_near[finite]
                        w = 1.0 / (d + 1e-12)
                        z_interp = float(np.sum(w * z_near) / np.sum(w))
                        if np.isfinite(z_interp):
                            z_val = z_interp

        self.cursorPositionChanged.emit(self, x, y, z_val)
