"""Convenience tools for generating ``pyqtgraph`` plots that can
be used in plottr's automatic plotting framework."""

from typing import Optional, Tuple, NoReturn

import numpy as np
import pyqtgraph as pg

from plottr import QtCore, QtGui, QtWidgets, Signal, config_entry

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
        self._applyDefaultTextStyle()
        self.plot.scene().sigMouseMoved.connect(self._onMouseMoved)

    def _applyDefaultTextStyle(self) -> None:
        tick_pt = int(config_entry('main', 'pyqtgraph', 'axis_font_size_pt', default=11))
        label_pt = int(config_entry('main', 'pyqtgraph', 'axis_label_size_pt', default=12))
        font = QtGui.QFont()
        font.setPointSize(max(7, tick_pt))

        for side in ['left', 'bottom', 'right', 'top']:
            axis = self.plot.getAxis(side)
            axis.setStyle(tickFont=font, autoExpandTextSpace=True)
            try:
                axis.label.setAttr('size', f'{max(8, label_pt)}pt')
            except Exception:
                pass

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
        self._colorbarWidth = int(config_entry('main', 'pyqtgraph', 'colorbar_base_width', default=18))

        self.colorbar: pg.ColorBarItem = pg.ColorBarItem(interactive=True, values=(0, 1),
                                                         colorMap=cmap, width=self._colorbarWidth)
        self.graphicsLayout.addItem(self.colorbar)
        self._applyColorbarTextStyle()

        self.img: Optional[pg.ImageItem] = None
        self.scatter: Optional[pg.ScatterPlotItem] = None
        self.scatterZVals: Optional[np.ndarray] = None
        self.scatterXVals: Optional[np.ndarray] = None
        self.scatterYVals: Optional[np.ndarray] = None
        self.imageXVals: Optional[np.ndarray] = None
        self.imageYVals: Optional[np.ndarray] = None
        self.imageZVals: Optional[np.ndarray] = None
        self.imageXGrid: Optional[np.ndarray] = None
        self.imageYGrid: Optional[np.ndarray] = None
        self.imageRect: Optional[QtCore.QRectF] = None

    def _applyColorbarTextStyle(self) -> None:
        tick_pt = int(config_entry('main', 'pyqtgraph', 'axis_font_size_pt', default=11))
        label_pt = int(config_entry('main', 'pyqtgraph', 'axis_label_size_pt', default=12))
        font = QtGui.QFont()
        font.setPointSize(max(7, tick_pt))
        self.colorbar.axis.setStyle(tickFont=font, autoExpandTextSpace=True)
        try:
            self.colorbar.axis.label.setAttr('size', f'{max(8, label_pt)}pt')
        except Exception:
            pass

    def setColorbarWidth(self, width: int) -> None:
        width = max(4, int(width))
        if width == self._colorbarWidth:
            return

        self._colorbarWidth = width

        levels = (0.0, 1.0)
        cmap = self._getColormap(config_entry('main', 'pyqtgraph', 'default_colormap'))
        try:
            levels = tuple(self.colorbar.levels())
        except Exception:
            pass
        try:
            if hasattr(self.colorbar, 'colorMap'):
                cmap = self.colorbar.colorMap()
            elif hasattr(self.colorbar, 'cmap'):
                cmap = self.colorbar.cmap
        except Exception:
            pass

        try:
            self.graphicsLayout.removeItem(self.colorbar)
        except Exception:
            pass

        self.colorbar = pg.ColorBarItem(interactive=True, values=levels,
                                        colorMap=cmap, width=width)
        self.graphicsLayout.addItem(self.colorbar)
        self._applyColorbarTextStyle()

        # Re-link data source and scatter callbacks on the new colorbar.
        try:
            self.colorbar.setLevels(levels)
            if self.img is not None:
                self.colorbar.setImageItem(self.img)
            if self.scatter is not None:
                self.colorbar.sigLevelsChanged.connect(self._colorScatterPoints)
                self._colorScatterPoints(self.colorbar)
        except Exception:
            pass

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
        self.imageXGrid = None
        self.imageYGrid = None
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
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        z_arr = np.asarray(z)

        reuse_img = (
            self.img is not None and
            self.scatter is None and
            self.imageZVals is not None and
            self.imageZVals.shape == z_arr.shape
        )

        if not reuse_img:
            self.clearPlot()
            self.img = pg.ImageItem()
            self.plot.addItem(self.img)
            self.colorbar.setImageItem(self.img)

        self.img.setImage(z_arr)
        x_finite = x_arr[np.isfinite(x_arr)]
        y_finite = y_arr[np.isfinite(y_arr)]
        if x_finite.size > 0 and y_finite.size > 0:
            rect = QtCore.QRectF(
                float(x_finite.min()),
                float(y_finite.min()),
                float(x_finite.max() - x_finite.min()),
                float(y_finite.max() - y_finite.min())
            )
            self.img.setRect(rect)
            self.imageRect = rect

        # Keep hover caches memory-efficient: full 2D x/y grids are only
        # retained for moderately sized images where irregular-grid hover
        # benefits outweigh memory cost.
        self.imageZVals = self.img.image
        max_hover_cells = int(config_entry('main', 'pyqtgraph', 'cursor_full_grid_cache_max_cells', default=300000))
        if (
            x_arr.ndim == 2 and y_arr.ndim == 2 and
            x_arr.shape == z_arr.shape and y_arr.shape == z_arr.shape and
            z_arr.size <= max_hover_cells
        ):
            self.imageXGrid = np.asarray(x_arr, dtype=float)
            self.imageYGrid = np.asarray(y_arr, dtype=float)
        else:
            self.imageXGrid = None
            self.imageYGrid = None

        if x_arr.ndim == 2 and x_arr.shape == z_arr.shape:
            self.imageXVals = np.asarray(np.nanmedian(x_arr, axis=0), dtype=float)
        else:
            self.imageXVals = np.asarray(x_arr, dtype=float).reshape(-1)

        if y_arr.ndim == 2 and y_arr.shape == z_arr.shape:
            self.imageYVals = np.asarray(np.nanmedian(y_arr, axis=1), dtype=float)
        else:
            self.imageYVals = np.asarray(y_arr, dtype=float).reshape(-1)

        z_min = float(np.nanmin(z))
        z_max = float(np.nanmax(z))
        if not np.isfinite(z_min) or not np.isfinite(z_max) or z_min == z_max:
            z_min, z_max = 0.0, 1.0
        self.colorbar.rounding = (z_max - z_min) * 1e-2
        self.colorbar.setLevels((z_min, z_max))
        if not reuse_img:
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

        reuse_scatter = (self.scatter is not None and self.img is None)
        if not reuse_scatter:
            self.clearPlot()
            self.scatter = pg.ScatterPlotItem()
            self.plot.addItem(self.scatter)
            self.colorbar.sigLevelsChanged.connect(self._colorScatterPoints)

        self.scatter.setData(x=x_flat, y=y_flat, symbol='o', size=10, pen=None)
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

        if not reuse_scatter:
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
            if self.imageZVals.ndim == 2:
                if self.imageXGrid is not None and self.imageYGrid is not None:
                    xg = self.imageXGrid
                    yg = self.imageYGrid
                    zg = self.imageZVals
                    finite = np.isfinite(xg) & np.isfinite(yg) & np.isfinite(zg)
                    if np.any(finite):
                        x_min = float(np.nanmin(xg[finite]))
                        x_max = float(np.nanmax(xg[finite]))
                        y_min = float(np.nanmin(yg[finite]))
                        y_max = float(np.nanmax(yg[finite]))
                        if x_min <= x <= x_max and y_min <= y <= y_max:
                            dx = xg[finite] - x
                            dy = yg[finite] - y
                            dist2 = dx * dx + dy * dy
                            idx = int(np.argmin(dist2))
                            z = zg[finite][idx]
                            if np.isfinite(z):
                                z_val = float(z)
                else:
                    x_vals = np.asarray(self.imageXVals, dtype=float).reshape(-1)
                    y_vals = np.asarray(self.imageYVals, dtype=float).reshape(-1)
                    if x_vals.size > 0 and y_vals.size > 0:
                        x_min = float(np.nanmin(x_vals))
                        x_max = float(np.nanmax(x_vals))
                        y_min = float(np.nanmin(y_vals))
                        y_max = float(np.nanmax(y_vals))
                        if x_min <= x <= x_max and y_min <= y <= y_max:
                            ix = int(np.nanargmin(np.abs(x_vals - x)))
                            iy = int(np.nanargmin(np.abs(y_vals - y)))
                            if 0 <= iy < self.imageZVals.shape[0] and 0 <= ix < self.imageZVals.shape[1]:
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
