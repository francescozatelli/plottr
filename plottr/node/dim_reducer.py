"""dim_reducer.py

nodes and widgets for reducing data dimensionality.
"""
from typing import Dict, Any, Tuple, Type, Optional, List, Union, cast, Callable
from enum import Enum, unique

from typing_extensions import TypedDict
import numpy as np

from .node import Node, updateOption, NodeWidget
from ..data.datadict import MeshgridDataDict, DataDict, DataDictBase, datadict_to_meshgrid
from .. import QtCore, QtWidgets, Signal, Slot
from plottr.icons import get_xySelectIcon

__author__ = 'Wolfgang Pfaff'
__license__ = 'MIT'


# Some helpful reduction functions

def sliceAxis(arr: np.ndarray, sliceObj: slice, axis: int) -> np.ndarray:
    """
    return the array where the axis with the given index is sliced
    with the given slice object.

    :param arr: input array
    :param sliceObj: slice object to use on selected dimension
    :param axis: dimension of the array to apply slice to
    :return: array after slicing
    """
    slices = [np.s_[::] for i in arr.shape]
    slices[axis] = sliceObj
    return arr[tuple(slices)]


def selectAxisElement(arr: np.ndarray, index: int, axis: int) -> np.ndarray:
    """
    return the squeezed array where the given axis has been reduced to its
    value with the given index.

    :param arr: input array
    :param index: index of the element to keep
    :param axis: dimension on which to perform the reduction
    :return: reduced array
    """
    return np.squeeze(sliceAxis(arr, np.s_[index:index+1:], axis), axis=axis)


def averageAxis(arr: np.ndarray, axis: int,
                start: Optional[int] = None,
                stop: Optional[int] = None) -> np.ndarray:
    """
    Average along a selected axis, optionally restricted to an index range.

    :param arr: input array
    :param axis: dimension on which to perform the averaging
    :param start: first index (inclusive)
    :param stop: last index (inclusive)
    :return: reduced array
    """
    if start is None and stop is None:
        return np.mean(arr, axis=axis)

    naxvals = arr.shape[axis]
    lo = 0 if start is None else int(start)
    hi = (naxvals - 1) if stop is None else int(stop)
    lo = max(0, min(lo, naxvals - 1))
    hi = max(0, min(hi, naxvals - 1))
    if lo > hi:
        lo, hi = hi, lo

    sliced = sliceAxis(arr, np.s_[lo:hi+1:], axis)
    return np.mean(sliced, axis=axis)


# Translation between reduction functions and convenient naming
@unique
class ReductionMethod(Enum):
    """Built-in reduction methods"""
    elementSelection = 'select element'
    average = 'average'


#: mapping from reduction method Enum to functions
reductionFunc: Dict[ReductionMethod, Callable[..., Any]] = {
    ReductionMethod.elementSelection: selectAxisElement,
    ReductionMethod.average: averageAxis,
}


ReductionType = Tuple[ReductionMethod, List[Any], Dict[str, int]]
RoleOptionsDict = dict


class RoleDict(TypedDict):
    role: Optional[str]
    options: RoleOptionsDict


class DimensionAssignmentWidget(QtWidgets.QTreeWidget):
    """
    A Widget that allows to assign options ('roles') to dimensions of a
    dataset.
    In this base version, there are no options included.
    This needs to be done by inheriting classes.
    """

    rolesChanged = Signal(object)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.setColumnCount(4)
        self.setHeaderLabels(['Dimension', 'Role', 'Options', 'Info'])

        self._dataStructure: Optional[DataDictBase] = None
        self._dataShapes: Optional[Dict[str, Dict[int, int]]] = None
        self._dataType: Optional[Type[DataDictBase]] = None
        self._currentRoles: Dict[str, RoleDict] = {}

        #: This is a flag to control whether we need to emit signals when
        #: a role has changed. broadly speaking, this is only desired when
        #: the change comes from the user interacting with the UI, otherwise
        #: it might lead to undesired recursion.
        self.emitRoleChangeSignal = True

        self.choices: Dict[str, dict] = {}
        self.availableChoices: Dict[Type[DataDictBase], List[str]] = {
            DataDictBase: ['None', ],
            DataDict: [],
            MeshgridDataDict: [],
        }

    def clear(self) -> None:
        """
        Clear the widget, delete all accessory widgets.
        """
        super().clear()

        for n, opts in self.choices.items():
            opts['roleSelectionWidget'].deleteLater()
            del opts['roleSelectionWidget']

            if 'optionsWidget' in opts:
                if opts['optionsWidget'] is not None:
                    opts['optionsWidget'].deleteLater()
                del opts['optionsWidget']

        self._dataStructure = None
        self._dataShapes = None
        self._dataType = None
        self.choices = {}

    def updateSizes(self) -> None:
        """update column widths to fit content."""
        for i in range(4):
            self.resizeColumnToContents(i)

    def setData(self, structure: DataDictBase,
                shapes: Dict[str, Dict[int, int]], dtype: Type[DataDictBase]) -> None:
        """
        set data: add all dimensions to the list, and populate choices.

        :param data: DataDict object
        """
        if structure is None:
            self.clear()
            return
        if (self._dataStructure is not None
                and DataDictBase.same_structure(structure, self._dataStructure)
                and shapes == self._dataShapes
                and dtype == self._dataType):
            return

        self.clear()
        self._dataType = dtype
        self._dataShapes = shapes
        self._dataStructure = structure
        self._currentRoles = {}

        for ax in self._dataStructure.axes():
            self.addDimension(ax)

    def addDimension(self, name: str) -> None:
        """
        add a new dimension.

        :param name: name of the dimension.
        """
        assert self._dataType is not None
        item = QtWidgets.QTreeWidgetItem([name, '', '', ''])
        self.addTopLevelItem(item)

        combo = QtWidgets.QComboBox()
        for t, opts in self.availableChoices.items():
            if t == self._dataType or issubclass(self._dataType, t):
                for o in opts:
                    combo.addItem(o)

        scaling = int(np.rint(self.logicalDpiX() / 96.0))
        combo.setMinimumSize(50*scaling, 22*scaling)
        combo.setMaximumHeight(22 * scaling)
        self.setItemWidget(item, 1, combo)
        self.updateSizes()

        self.choices[name] = {
            'roleSelectionWidget': combo,
            'optionsWidget': None,
        }
        combo.currentTextChanged.connect(
            lambda x: self.processSelectionChange(name, x)
        )
        self.setDimInfo(name, '')

    def processSelectionChange(self, name: str, val: str) -> None:
        """
        Call to notify that a dimension's role should be changed.
        any specific actions should be implemented in :func:`setRole`.

        :param name: name of the dimension
        :param val: new role name
        """

        # we need a flag here to not emit signals when we recursively change
        # roles. Sometimes we do, because roles are not independent for the
        # dims.
        if self.emitRoleChangeSignal:
            self.emitRoleChangeSignal = False
            self.setRole(name, val)
            self.rolesChanged.emit(self.getRoles())
            self.emitRoleChangeSignal = True

    def setRole(self, dim: str, role: Optional[str] = None) -> None:
        """
        Set the role for a dimension, including options.

        :param dim: name of the dimension
        :param role: name of the role
        """
        curRole = self._currentRoles.get(dim, None)
        if curRole is None or curRole['role'] != role:
            self.choices[dim]['roleSelectionWidget'].setCurrentText(role)
            item = self.findItems(dim, QtCore.Qt.MatchExactly, 0)[0]

            if 'optionsWidget' in self.choices[dim]:
                w = self.choices[dim]['optionsWidget']
                if w is not None:
                    w.deleteLater()
                    self.choices[dim]['optionsWidget'] = None

            self.setItemWidget(item, 2, QtWidgets.QWidget())
            self.setDimInfo(dim, '')

            self._currentRoles[dim] = RoleDict(
                role=role,
                options={},
            )

    def getRole(self, name: str) -> Tuple[str, RoleOptionsDict]:
        """
        Get the current role and its options for a dimension.
        :param name: 
        :return: 
        """
        role = self.choices[name]['roleSelectionWidget'].currentText()
        opts: Dict = {}
        return role, opts

    def getRoles(self) -> Dict[str, RoleDict]:
        """
        Get all roles as set in the UI.
        :return: Dictionary with information on all current roles/options.
        """
        ret = {}
        for name, val in self.choices.items():
            role, opts = self.getRole(name)
            ret[name] = RoleDict(
                role=role,
                options=opts,
            )
        return ret

    @Slot(str, str)
    def setDimInfo(self, dim: str, info: str = '') -> None:
        try:
            item = self.findItems(dim, QtCore.Qt.MatchExactly, 0)[0]
            item.setText(3, info)
        except IndexError:
            pass

    @Slot(dict)
    def setDimInfos(self, infos: Dict[str, str]) -> None:
        for ax, info in infos.items():
            self.setDimInfo(ax, info)

    @Slot(dict)
    def setShapes(self, shapes: Dict[str, Dict[int, int]]) -> None:
        self._dataShapes = shapes

class DimensionReductionAssignmentWidget(DimensionAssignmentWidget):

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.availableChoices[DataDict] += [
            ReductionMethod.average.value,
            ReductionMethod.elementSelection.value,
        ]
        self.availableChoices[MeshgridDataDict] += [
            ReductionMethod.average.value,
            ReductionMethod.elementSelection.value,
        ]

    def getRole(self, name: str) -> Tuple[str, Dict]:
        role, opts = super().getRole(name)

        if role == ReductionMethod.elementSelection.value:
            w = self.choices[name]['optionsWidget']
            if w is not None:
                opts['index'] = w.value()
        elif role == ReductionMethod.average.value:
            w = self.choices[name]['optionsWidget']
            if w is not None:
                start, stop = w.values()
                opts['start'] = int(start)
                opts['stop'] = int(stop)

        return role, opts

    def setRole(self, dim: str, role: Optional[str] = None, **kw: Any) -> None:
        super().setRole(dim, role)

        # at this point, we've already populated the dropdown.
        # this is now for additional options
        item = self.findItems(dim, QtCore.Qt.MatchExactly, 0)[0]

        if role == ReductionMethod.elementSelection.value:

            # only create the slider widget if it doesn't exist yet
            if (self.itemWidget(item, 2) is None) or \
                    (self.choices[dim]['optionsWidget'] is None):
                self.setNewElementSelector(dim, item, **kw)
        elif role == ReductionMethod.average.value:
            if (self.itemWidget(item, 2) is None) or \
                    (self.choices[dim]['optionsWidget'] is None):
                self.setNewAverageSelector(dim, item, **kw)

    def setShapes(self, shapes: Dict[str, Dict[int, int]]) -> None:
        oldShapes = self._dataShapes
        super().setShapes(shapes)
        for dim, values in shapes.items():
            if dim in self.choices:
                role = self.getRole(dim)[0]
                if role in [ReductionMethod.elementSelection.value, ReductionMethod.average.value]:
                    # If this shape did not change, don't update controls.
                    if oldShapes is not None and values == oldShapes[dim]:
                        continue
                    item = self.findItems(dim, QtCore.Qt.MatchExactly, 0)[0]
                    if role == ReductionMethod.elementSelection.value:
                        self.setNewElementSelector(dim, item)
                    elif role == ReductionMethod.average.value:
                        self.setNewAverageSelector(dim, item)

    def _axisValues(self, dim: str) -> np.ndarray:
        """Return 1D coordinate values for a dimension in index order."""
        assert self._dataStructure is not None
        assert self._dataShapes is not None
        axidx = self._dataStructure.axes().index(dim)
        naxvals = self._dataShapes[dim][axidx]

        vals = np.asarray(self._dataStructure.data_vals(dim))
        if vals.ndim == 0:
            return np.array([vals.item()])

        if vals.ndim > 1:
            index = [0] * vals.ndim
            index[axidx] = slice(None)
            vals = np.asarray(vals[tuple(index)])

        vals = np.asarray(vals).reshape(-1)
        if vals.size != naxvals:
            vals = np.linspace(0, naxvals - 1, naxvals)
        return vals

    def setNewElementSelector(self, dim: str, item: QtWidgets.QTreeWidgetItem,
                              **kw: Any) -> None:
        assert self._dataStructure is not None
        assert self._dataShapes is not None
        # get the number of elements in this dimension
        axidx = self._dataStructure.axes().index(dim)
        naxvals = self._dataShapes[dim][axidx]
        axvals = self._axisValues(dim)

        previousSlider = self.itemWidget(item, 2)
        sliderValue = int(kw.get('index', 0))
        if hasattr(previousSlider, 'value'):
            sliderValue = previousSlider.value()
        sliderValue = max(0, min(sliderValue, naxvals - 1))

        w = self.elementSelectionControl(nvals=naxvals,
                                         values=axvals,
                                         value=sliderValue)
        w.valueChanged.connect(lambda x: self.elementSelectionSliderChange(dim))

        scaling = int(np.rint(self.logicalDpiX() / 96.0))
        width = 220 + 60 * (scaling - 1)
        height = 28 * scaling
        w.setMinimumSize(width, height)
        w.setMaximumHeight(height)

        self.choices[dim]['optionsWidget'] = w
        self.setItemWidget(item, 2, w)
        self.setElementSelectionInfo(dim, item.text(3).split(' (')[0])
        self.updateSizes()

    def setNewAverageSelector(self, dim: str, item: QtWidgets.QTreeWidgetItem,
                              **kw: Any) -> None:
        assert self._dataStructure is not None
        assert self._dataShapes is not None
        axidx = self._dataStructure.axes().index(dim)
        naxvals = self._dataShapes[dim][axidx]
        axvals = self._axisValues(dim)

        previous = self.itemWidget(item, 2)
        start = int(kw.get('start', 0))
        stop = int(kw.get('stop', naxvals - 1))
        if hasattr(previous, 'values'):
            pstart, pstop = previous.values()
            start, stop = int(pstart), int(pstop)

        start = max(0, min(start, naxvals - 1))
        stop = max(0, min(stop, naxvals - 1))

        w = self.averageRangeControl(nvals=naxvals,
                                     values=axvals,
                                     start=start,
                                     stop=stop)
        w.rangeChanged.connect(lambda a, b: self.averageRangeChange(dim))

        scaling = int(np.rint(self.logicalDpiX() / 96.0))
        width = 260 + 80 * (scaling - 1)
        height = 52 * scaling
        w.setMinimumSize(width, height)
        w.setMaximumHeight(height)

        self.choices[dim]['optionsWidget'] = w
        self.setItemWidget(item, 2, w)
        self.setAverageSelectionInfo(dim)
        self.updateSizes()

    def elementSelectionControl(self, nvals: int,
                                values: np.ndarray,
                                value: int = 0) -> QtWidgets.QWidget:
        class _ElementSelectionControl(QtWidgets.QWidget):
            valueChanged = Signal(int)

            def __init__(self, n: int, vals: np.ndarray, v: int,
                         parent: Optional[QtWidgets.QWidget] = None):
                super().__init__(parent)
                self._vals = vals
                self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
                self.slider.setMinimum(0)
                self.slider.setMaximum(max(0, n - 1))
                self.slider.setSingleStep(1)
                self.slider.setPageStep(1)
                self.slider.setTickInterval(max(1, n // 10))
                self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)

                self.label = QtWidgets.QLabel(self)
                self.label.setMinimumWidth(80)

                layout = QtWidgets.QHBoxLayout(self)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addWidget(self.slider)
                layout.addWidget(self.label)

                self.slider.valueChanged.connect(self._updateLabel)
                self.slider.valueChanged.connect(self.valueChanged.emit)
                self.slider.setValue(v)
                self._updateLabel(self.slider.value())

            def _updateLabel(self, idx: int) -> None:
                idx = int(max(0, min(idx, self._vals.size - 1)))
                self.label.setText(f"{self._vals[idx]:.6g}")

            def value(self) -> int:
                return int(self.slider.value())

        return _ElementSelectionControl(nvals, values, value)

    def averageRangeControl(self, nvals: int,
                            values: np.ndarray,
                            start: int = 0,
                            stop: Optional[int] = None) -> QtWidgets.QWidget:
        class _AverageRangeControl(QtWidgets.QWidget):
            rangeChanged = Signal(int, int)

            def __init__(self, n: int, vals: np.ndarray,
                         s: int, e: int,
                         parent: Optional[QtWidgets.QWidget] = None):
                super().__init__(parent)
                self._vals = vals

                self.startSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
                self.stopSlider = QtWidgets.QSlider(QtCore.Qt.Horizontal, self)
                for sl in [self.startSlider, self.stopSlider]:
                    sl.setMinimum(0)
                    sl.setMaximum(max(0, n - 1))
                    sl.setSingleStep(1)
                    sl.setPageStep(1)
                    sl.setTickInterval(max(1, n // 10))
                    sl.setTickPosition(QtWidgets.QSlider.TicksBelow)

                self.infoLabel = QtWidgets.QLabel(self)

                layout = QtWidgets.QVBoxLayout(self)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(2)
                layout.addWidget(self.startSlider)
                layout.addWidget(self.stopSlider)
                layout.addWidget(self.infoLabel)

                self.startSlider.valueChanged.connect(self._updateRange)
                self.stopSlider.valueChanged.connect(self._updateRange)

                if e is None:
                    e = n - 1
                self.startSlider.setValue(max(0, min(s, n - 1)))
                self.stopSlider.setValue(max(0, min(e, n - 1)))
                self._updateRange()

            def _updateRange(self) -> None:
                lo = min(self.startSlider.value(), self.stopSlider.value())
                hi = max(self.startSlider.value(), self.stopSlider.value())
                self.infoLabel.setText(
                    f"avg [{lo}:{hi}] = {self._vals[lo]:.6g} .. {self._vals[hi]:.6g}"
                )
                self.rangeChanged.emit(lo, hi)

            def values(self) -> Tuple[int, int]:
                lo = min(self.startSlider.value(), self.stopSlider.value())
                hi = max(self.startSlider.value(), self.stopSlider.value())
                return int(lo), int(hi)

        return _AverageRangeControl(nvals, values, start, stop)

    def elementSelectionSliderChange(self, dim: str) -> None:
        self.setElementSelectionInfo(dim)
        roles = self.getRoles()
        self.rolesChanged.emit(roles)

    def averageRangeChange(self, dim: str) -> None:
        self.setAverageSelectionInfo(dim)
        roles = self.getRoles()
        self.rolesChanged.emit(roles)

    def setElementSelectionInfo(self, dim: str, value: Optional[str] = None) -> None:
        # get the number of elements in this dimension
        assert self._dataStructure is not None
        assert self._dataShapes is not None
        roles = self.getRoles()
        axidx = self._dataStructure.axes().index(dim)
        naxvals = self._dataShapes[dim][axidx]
        text = ''
        if 'index' in roles[dim]['options']:
            idx = roles[dim]['options']['index']
            text = f"({idx + 1}/{naxvals})"
            if value is not None:
                text = value + ' ' + text
        self.setDimInfo(dim, text)

    def setAverageSelectionInfo(self, dim: str) -> None:
        assert self._dataStructure is not None
        assert self._dataShapes is not None
        roles = self.getRoles()
        axidx = self._dataStructure.axes().index(dim)
        naxvals = self._dataShapes[dim][axidx]
        text = ''
        start = roles[dim]['options'].get('start', 0)
        stop = roles[dim]['options'].get('stop', naxvals - 1)
        if naxvals > 0:
            text = f"({start + 1}:{stop + 1}/{naxvals})"
        self.setDimInfo(dim, text)


class XYSelectionWidget(DimensionReductionAssignmentWidget):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.availableChoices[DataDictBase] += ["x-axis", "y-axis"]
        self._secondaryCandidates: Dict[str, List[str]] = {'x': [], 'y': []}

    def setSecondaryCandidates(self, xCandidates: List[str], yCandidates: List[str]) -> None:
        self._secondaryCandidates = {
            'x': list(xCandidates),
            'y': list(yCandidates),
        }
        self._refreshRoleChoices()

    def setData(self, structure: DataDictBase,
                shapes: Dict[str, Dict[int, int]], dtype: Type[DataDictBase]) -> None:
        super().setData(structure, shapes, dtype)
        self._refreshRoleChoices()

    def _refreshRoleChoices(self) -> None:
        for dim, opts in self.choices.items():
            combo = opts['roleSelectionWidget']
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()

            entries: List[str] = ['None']
            entries += [ReductionMethod.average.value, ReductionMethod.elementSelection.value]
            entries += ['x-axis', 'y-axis']
            if dim in self._secondaryCandidates.get('x', []):
                entries += ['x-secondary']
            if dim in self._secondaryCandidates.get('y', []):
                entries += ['y-secondary']

            for e in entries:
                combo.addItem(e)

            if current in entries:
                combo.setCurrentText(current)
            else:
                combo.setCurrentText('None')
            combo.blockSignals(False)

        self.updateSizes()

    def setRole(self, dim: str, role: Optional[str] = None, **kw: Any) -> None:
        super().setRole(dim, role, **kw)

        # there can only be one x and y axis element.
        if role in ['x-axis', 'y-axis', 'x-secondary', 'y-secondary']:
            allRoles = self.getRoles()
            for d, r in allRoles.items():
                if d == dim:
                    continue
                if r['role'] == role:
                    self.setRole(d, 'None')


class DimensionReducerNodeWidget(NodeWidget[DimensionReductionAssignmentWidget]):

    def __init__(self, node: Optional[Node] = None):
        super().__init__(embedWidgetClass=DimensionReductionAssignmentWidget)
        # Not clear how to type that self.widget is not None
        # iff embedWidgetClass is not None
        assert self.widget is not None
        self.optSetters = {
            'reductions': self.setReductions,
        }
        self.optGetters = {
            'reductions': self.getReductions,
        }
        self.widget.rolesChanged.connect(
            lambda x: self.signalOption('reductions'))

    def getReductions(self) -> Dict[str, Optional[ReductionType]]:
        assert self.widget is not None
        roles = self.widget.getRoles()
        reductions: Dict[str, Optional[ReductionType]] = {}
        for dimName, rolesOptions in roles.items():
            role = rolesOptions['role']
            opts = rolesOptions['options']
            if role in [e.value for e in ReductionMethod]:
                method = ReductionMethod(role)
                reductions[dimName] = method, [], opts

        return reductions

    def setReductions(self, reductions: Dict[str, Optional[ReductionType]]) -> None:
        assert self.widget is not None
        for dimName, reduction in reductions.items():
            assert reduction is not None
            (method, arg, kw) = reduction
            role = method.value
            self.widget.setRole(dimName, role, **kw)

        for dimName, _ in self.widget.getRoles().items():
            if dimName not in reductions.keys():
                self.widget.setRole(dimName, 'None')

    def setData(self,
                structure: DataDictBase,
                shapes: Dict[str, Dict[int, int]],
                dtype: Type[DataDictBase]) -> None:
        assert self.widget is not None
        self.widget.setData(structure, shapes, dtype)

    def setShapes(self, shapes: Dict[str, Dict[int, int]]) -> None:
        assert self.widget is not None
        self.widget.setShapes(shapes)

class DimensionReducer(Node):
    """
    A Node that allows the user to reduce the dimensionality of input data.

    Each axis can be assigned an arbitrary reduction function that will reduce
    the axis to a single value. For each assigned reduction the dimension
    shrinks by 1.

    If the input is not GridData, data is just passed through, but we delete the
    axes present in reductions.

    If the output contains invalid entries, they will be masked.

    Properties are:

    :targetNames: ``List[str]`` or ``None``.
        reductions affect all dependents that are given. If None, will apply
        to all dependents.
    :reductions: ``Dict[str, (callable, *args, **kwargs)]``
        reduction functions. Keys are the axis names the reductions are applied
        to; values are tuples of the reduction function, and optional
        arguments and kw-arguments.
        The function can also be via :class:`ReductionMethod`.
        The function must accept an ``axis = <int>`` keyword, and must return
        an array of dimensionality that is reduced by one compared to its
        input.
    :reductionValues: ``Dict[str, float]``
        Holds the value of the currently selected axis index.
        This only happen when the role is ReductionMethod.elementSelection.
        Dictionary with the axis as keys and the selected values as values.

    """

    nodeName = 'DimensionReducer'
    uiClass: Type["NodeWidget"] = DimensionReducerNodeWidget

    #: A signal that emits (structure, shapes, type) when data structure has
    #: changed.
    newDataStructure = Signal(object, object, object)

    #: A signal that emits (shapes) when the shapes have changed.
    dataShapesChanged = Signal(dict)

    def __init__(self,  name: str):
        self._reductions: Dict[str, Optional[ReductionType]] = {}
        self._reductionValues: Dict[str, float] = {}
        self._targetNames: Optional[List[str]] = None
        self._dataStructure = None

        super().__init__(name)

    # Properties

    @property
    def reductions(self) -> Dict[str, Optional[ReductionType]]:
        return self._reductions

    @reductions.setter
    @updateOption('reductions')
    def reductions(self, val: Dict[str, Optional[ReductionType]]) -> None:
        self._reductions = val

    @property
    def targetNames(self) -> Optional[List[str]]:
        return self._targetNames

    @targetNames.setter
    @updateOption()
    def targetNames(self, val: Optional[List[str]]) -> None:
        self._targetNames = val

    @property
    def reductionValues(self) -> Dict[str, float]:
        return self._reductionValues

    @reductionValues.setter
    @updateOption('reductionValues')
    def reductionValues(self, val: Dict[str, float]) -> None:
        self._reductionValues = val

    # Data processing

    def _applyDimReductions(self, data: DataDictBase) -> Optional[DataDictBase]:
        """Apply the reductions"""
        reductionValues: Dict[str, float] = {}  # Holds the temporary reduction values before saving them.

        # QCoDeS data often arrives as DataDict even when it is logically gridded.
        # For built-in reduction methods, try to make a meshgrid first so linecuts
        # and averaging can be applied consistently from the UI.
        if not isinstance(data, MeshgridDataDict):
            has_builtin_reductions = any(
                isinstance(reduction[0], ReductionMethod)
                for reduction in self._reductions.values()
                if reduction is not None
            )
            if has_builtin_reductions and isinstance(data, DataDict):
                try:
                    data = datadict_to_meshgrid(data)
                except Exception as exc:
                    self.node_logger.warning(
                        f"Could not convert DataDict to meshgrid for reductions: {exc}. "
                        "Falling back to axis-removal behavior."
                    )

        if self._targetNames is not None:
            dnames = self._targetNames
        else:
            dnames = data.dependents()

        if not isinstance(data, MeshgridDataDict):
            self.node_logger.debug(f"Data is not on a grid. "
                                f"Reduction functions are ignored, "
                                f"axes will simply be removed.")

        for n in dnames:
            for ax, reduction in self._reductions.items():
                fun: Optional[ReductionMethod]
                if reduction is not None:
                    fun, arg, kw = reduction
                else:
                    fun, arg, kw = None, [], {}

                try:
                    idx = data[n]['axes'].index(ax)
                except IndexError:
                    self.node_logger.info(f'{ax} specified for reduction, '
                                       f'but not present in data; ignore.')

                kw['axis'] = idx

                # actual operation is only done if the data is on a grid.
                if isinstance(data, MeshgridDataDict):

                    # check that the new shape is actually correct
                    # get target shape by removing the right axis
                    targetShape_list = list(data[n]['values'].shape)
                    del targetShape_list[idx]
                    targetShape = tuple(targetShape_list)

                    # support for both pre-defined and custom functions
                    if isinstance(fun, ReductionMethod):
                        funCall: Optional[Callable[..., Any]] = reductionFunc[fun]
                    else:
                        funCall = fun

                    if funCall is None:
                        raise RuntimeError("Reduction function is None")
                    newvals = funCall(data[n]['values'], *arg, **kw)
                    if newvals.shape != targetShape:
                        self.node_logger.error(
                            f'Reduction on axis {ax} did not result in the '
                            f'right data shape. ' +
                            f'Expected {targetShape} but got {newvals.shape}.'
                            )
                        return None
                    data[n]['values'] = newvals

                    # since we are on a meshgrid, we also need to reduce
                    # the dimensions of the coordinate meshes
                    for ax in data[n]['axes']:
                        axdata = data.data_vals(ax)
                        if len(axdata.shape) > len(targetShape):
                            newaxvals = funCall(data[ax]['values'], *arg, **kw)
                            data[ax]['values'] = newaxvals
                            if ax in self._reductions:
                                reductionValues[ax] = newaxvals.flat[0]

                del data[n]['axes'][idx]

        data = data.sanitize()
        data.validate()
        if self.reductionValues != reductionValues:
            self.reductionValues = reductionValues
        return data

    def validateOptions(self, data: DataDictBase) -> bool:
        """
        Checks performed:
        * each item in reduction must be of the form (fun, [*arg], {**kw}),
          with arg and kw being optional; if the tuple is has length 2,
          the second element is taken as the arg-list.
          The function can be of type :class:`.ReductionMethod`.
        """
        delete = []
        for ax, reduction in self._reductions.items():

            if ax not in data.axes():
                self.node_logger.warning(f"{ax} is not a known dimension. Removing.")
                delete.append(ax)
                continue

            if reduction is None:
                if isinstance(data, MeshgridDataDict):
                    self.node_logger.warning(f'Reduction for axis {ax} is None. '
                                          f'Removing.')
                    delete.append(ax)
                else:
                    pass
                continue

            try:
                fun = reduction[0]
                if len(reduction) == 1:
                    arg = []
                    kw: Dict[str, int] = {}
                elif len(reduction) == 2:
                    arg = reduction[1]
                    kw = {}
                else:
                    arg = reduction[1]
                    kw = reduction[2]
            except:
                self.node_logger.warning(
                    f'Reduction for axis {ax} not in the right format.'
                )
                return False

            if not callable(fun) and not isinstance(fun, ReductionMethod):
                self.node_logger.error(
                    f'Invalid reduction method for axis {ax}. '
                    f'Needs to be callable or a ReductionMethod type.'
                )
                return False

            # set the reduction in the correct format.
            self._reductions[ax] = (fun, arg, kw)

        for ax in delete:
            del self._reductions[ax]

        return True

    def process(
            self,
            dataIn: Optional[DataDictBase] = None) -> \
            Optional[Dict[str, Optional[DataDictBase]]]:
        if dataIn is None:
            return None

        data = super().process(dataIn=dataIn)

        if data is None:
            return None

        dataout = data['dataOut']
        assert dataout is not None
        data = dataout.copy()
        data = data.mask_invalid()
        data = self._applyDimReductions(data)

        return dict(dataOut=data)

    def setupUi(self) -> None:
        super().setupUi()
        assert self.ui is not None
        self.newDataStructure.connect(self.ui.setData)
        self.dataShapesChanged.connect(self.ui.setShapes)


class XYSelectorNodeWidget(NodeWidget[XYSelectionWidget]):

    def __init__(self, node: Optional[Node] = None):
        self.icon = get_xySelectIcon()
        super().__init__(embedWidgetClass=XYSelectionWidget)
        assert self.widget is not None

        self.optSetters = {
            'dimensionRoles': self.setRoles,
            'reductionValues': self.setReductionValues,
            'secondaryRoleCandidates': self.setSecondaryRoleCandidates,
        }
        self.optGetters = {
            'dimensionRoles': self.getRoles,
        }

        self.widget.rolesChanged.connect(
            lambda x: self.signalOption('dimensionRoles')
        )

    def getRoles(self) -> Dict[str, Union[str, Tuple[ReductionMethod, List[Any], Dict[str, Any]]]]:
        assert self.widget is not None
        widgetRoles = self.widget.getRoles()
        roles: Dict[str, Union[str, Tuple[ReductionMethod, List[Any], Dict[str, Any]]]] = {}
        for dimName, rolesOptions in widgetRoles.items():
            role = rolesOptions['role']
            opts = rolesOptions['options']

            if role in ['x-axis', 'y-axis', 'x-secondary', 'y-secondary']:
                roles[dimName] = role

            elif role in [e.value for e in ReductionMethod]:
                method = ReductionMethod(role)
                if method is not None:
                    roles[dimName] = method, [], opts

        return roles

    def setRoles(self, roles: Dict[str, str]) -> None:
        assert self.widget is not None
        # when this is called, we do not want the UI to signal changes.
        self.widget.emitRoleChangeSignal = False

        for dimName, role in roles.items():
            if role in ['x-axis', 'y-axis', 'x-secondary', 'y-secondary']:
                self.widget.setRole(dimName, role)
            elif isinstance(role, tuple):
                method, arg, kw = role
                methodName = method.value
                self.widget.setRole(dimName, methodName, **kw)
            elif role is None:
                self.widget.setRole(dimName, 'None')

        for dimName, _ in self.widget.getRoles().items():
            if dimName not in roles.keys():
                self.widget.setRole(dimName, 'None')

        self.widget.emitRoleChangeSignal = True

    def setSecondaryRoleCandidates(self, val: Dict[str, List[str]]) -> None:
        assert self.widget is not None
        xCandidates = val.get('x', []) if isinstance(val, dict) else []
        yCandidates = val.get('y', []) if isinstance(val, dict) else []
        self.widget.setSecondaryCandidates(xCandidates, yCandidates)

    def setReductionValues(self, val: Dict[str, float]) -> None:
        if self.widget is not None:
            for dim, value in val.items():
                self.widget.setElementSelectionInfo(dim, str(value))

    def setData(self,
                structure: DataDictBase,
                shapes: Dict[str, Dict[int, int]],
                dtype: Type[DataDictBase]) -> None:
        assert self.widget is not None
        self.widget.setData(structure, shapes, dtype)

    def setShapes(self, shapes: Dict[str, Dict[int, int]]) -> None:
        assert self.widget is not None
        self.widget.setShapes(shapes)


class XYSelector(DimensionReducer):

    nodeName = 'XYSelector'
    uiClass = XYSelectorNodeWidget

    def __init__(self, name: str):
        self._xyAxes: Tuple[Optional[str], Optional[str]] = (None, None)
        self._secondaryAxes: Tuple[Optional[str], Optional[str]] = (None, None)
        self._secondaryRoleCandidates: Dict[str, List[str]] = {'x': [], 'y': []}
        super().__init__(name)

    @property
    def secondaryRoleCandidates(self) -> Dict[str, List[str]]:
        return self._secondaryRoleCandidates

    @secondaryRoleCandidates.setter
    @updateOption('secondaryRoleCandidates')
    def secondaryRoleCandidates(self, val: Dict[str, List[str]]) -> None:
        self._secondaryRoleCandidates = {
            'x': list(val.get('x', [])),
            'y': list(val.get('y', [])),
        }

    def _as_meshgrid(self, data: DataDictBase) -> Optional[MeshgridDataDict]:
        if isinstance(data, MeshgridDataDict):
            return data
        if isinstance(data, DataDict):
            try:
                return datadict_to_meshgrid(data)
            except Exception:
                return None
        return None

    def _secondary_mapping_for_pair(
            self,
            mesh: MeshgridDataDict,
            primary_axis: str,
            secondary_axis: str,
            side: str,
    ) -> Optional[Dict[str, Any]]:
        if primary_axis not in mesh.axes() or secondary_axis not in mesh.axes():
            return None
        p = np.asarray(mesh.data_vals(primary_axis), dtype=float).reshape(-1)
        s = np.asarray(mesh.data_vals(secondary_axis), dtype=float).reshape(-1)
        if p.size < 8:
            return None

        finite = np.isfinite(p) & np.isfinite(s)
        if np.count_nonzero(finite) < 8:
            return None

        pp = p[finite]
        ss = s[finite]
        p_std = float(np.std(pp))
        s_std = float(np.std(ss))
        if p_std == 0.0 or s_std == 0.0:
            return None

        p_round = np.round(pp, decimals=12)
        p_unique = np.unique(p_round)
        if p_unique.size < 4:
            return None

        sec_vals = np.empty_like(p_unique, dtype=float)
        for i, pv in enumerate(p_unique):
            m = p_round == pv
            sec_vals[i] = float(np.nanmedian(ss[m]))

        finite_map = np.isfinite(sec_vals)
        if np.count_nonzero(finite_map) < 4:
            return None

        p_unique = p_unique[finite_map]
        sec_vals = sec_vals[finite_map]
        order = np.argsort(p_unique)
        p_sorted = p_unique[order]
        s_sorted = sec_vals[order]

        # Coupled sweeps can be noisy/nonlinear. Require mostly monotonic
        # mapped trend instead of near-perfect linear correlation.
        sdiff = np.diff(s_sorted)
        nz = sdiff[np.abs(sdiff) > 0]
        if nz.size < 3:
            return None
        pos = np.count_nonzero(nz > 0)
        neg = np.count_nonzero(nz < 0)
        monotonic_score = max(pos, neg) / nz.size
        if monotonic_score < 0.75:
            return None

        return {
            'primary_axis': primary_axis,
            'secondary_axis': secondary_axis,
            'primary_unit': str(mesh[primary_axis].get('unit', '')),
            'secondary_unit': str(mesh[secondary_axis].get('unit', '')),
            'primary_values': p_sorted.tolist(),
            'secondary_values': s_sorted.tolist(),
            'side': side,
        }

    def _axis_size_in_mesh(self, data: DataDictBase, axis: str) -> Optional[int]:
        if not isinstance(data, MeshgridDataDict):
            return None
        deps = data.dependents()
        if len(deps) == 0:
            return None
        dep = deps[0]
        axes = data[dep].get('axes', [])
        if axis not in axes:
            return None
        idx = axes.index(axis)
        vals = np.asarray(data[dep]['values'])
        if idx < 0 or idx >= vals.ndim:
            return None
        return int(vals.shape[idx])

    def _pick_default_secondary(self, data: DataDictBase, side: str, candidates: List[str], already_used: List[str]) -> Optional[str]:
        if len(candidates) == 0:
            return None

        valid = [c for c in candidates if c not in already_used]
        if len(valid) == 0:
            return None

        size1 = [c for c in valid if self._axis_size_in_mesh(data, c) == 1]
        if len(size1) > 0:
            return size1[0]

        # Stable fallback: pick the first candidate in deterministic order.
        return valid[0]

    def _detect_secondary_role_candidates(self, data: DataDictBase) -> Dict[str, List[str]]:
        mesh = self._as_meshgrid(data)
        if mesh is None:
            return {'x': [], 'y': []}

        x_axis, y_axis = self._xyAxes
        reduced_axes = [ax for ax in mesh.axes() if ax in self._reductions and self._reductions[ax] is not None]
        x_candidates: List[str] = []
        y_candidates: List[str] = []
        for ax in reduced_axes:
            if x_axis is not None:
                if self._secondary_mapping_for_pair(mesh, x_axis, ax, 'top') is not None:
                    x_candidates.append(ax)
            if y_axis is not None:
                if self._secondary_mapping_for_pair(mesh, y_axis, ax, 'right') is not None:
                    y_candidates.append(ax)
        return {'x': x_candidates, 'y': y_candidates}

    def _build_secondary_axis_info(self, data: DataDictBase) -> Optional[List[Dict[str, Any]]]:
        mesh = self._as_meshgrid(data)
        if mesh is None:
            return None

        infos: List[Dict[str, Any]] = []
        x_axis, y_axis = self._xyAxes
        x_sec, y_sec = self._secondaryAxes
        if x_axis is not None and x_sec is not None:
            info = self._secondary_mapping_for_pair(mesh, x_axis, x_sec, 'top')
            if info is not None:
                infos.append(info)
        if y_axis is not None and y_sec is not None:
            info = self._secondary_mapping_for_pair(mesh, y_axis, y_sec, 'right')
            if info is not None:
                infos.append(info)
        return infos or None

    @property
    def xyAxes(self) -> Tuple[Optional[str], Optional[str]]:
        return self._xyAxes

    @xyAxes.setter
    @updateOption('xyAxes')
    def xyAxes(self, val: Tuple[Optional[str], Optional[str]]) -> None:
        self._xyAxes = val

    @property
    def dimensionRoles(self) -> Dict[str, Union[str, ReductionType, None]]:
        dr: Dict[str, Union[str, ReductionType, None]] = {}
        if self.xyAxes[0] is not None:
            dr[self.xyAxes[0]] = 'x-axis'
        if self.xyAxes[1] is not None:
            dr[self.xyAxes[1]] = 'y-axis'
        if self._secondaryAxes[0] is not None:
            dr[self._secondaryAxes[0]] = 'x-secondary'
        if self._secondaryAxes[1] is not None:
            dr[self._secondaryAxes[1]] = 'y-secondary'
        for dim, red in self.reductions.items():
            if dim not in dr:
                dr[dim] = red
        return dr

    @dimensionRoles.setter
    @updateOption('dimensionRoles')
    def dimensionRoles(self, val: Dict[str, str]) -> None:
        x = None
        y = None
        xsec = None
        ysec = None
        self._reductions = {}
        for dimName, role in val.items():
            if role == 'x-axis':
                x = dimName
            elif role == 'y-axis':
                y = dimName
            elif role == 'x-secondary':
                xsec = dimName
            elif role == 'y-secondary':
                ysec = dimName
            else:
                self._reductions[dimName] = cast(Optional[ReductionType], role)
        self._xyAxes = (x, y)
        self._secondaryAxes = (xsec, ysec)

    def validateOptions(self, data: DataDictBase) -> bool:
        """
        Checks performed:
        * values for xAxis and yAxis must be axes that exist for the input
        data.
        * x/y axes cannot be the same
        * x/y axes cannot be reduced (will be removed from reductions)
        * all axes that are not x/y must be reduced (defaulting to
        selection of the first element)
        """

        if not super().validateOptions(data):
            return False
        availableAxes = data.axes()

        if len(availableAxes) > 0:
            if self._xyAxes[0] is None:
                self.node_logger.debug(
                    f'x-Axis is None. this will result in empty output data.')
                return False
            elif self._xyAxes[0] not in availableAxes:
                self.node_logger.warning(
                    f'x-Axis {self._xyAxes[0]} not present in data')
                return False

            if self._xyAxes[1] is None:
                self.node_logger.debug(f'y-Axis is None; result will be 1D')
            elif self._xyAxes[1] not in availableAxes:
                self.node_logger.warning(
                    f'y-Axis {self._xyAxes[1]} not present in data')
                return False
            elif self._xyAxes[1] == self._xyAxes[0]:
                self.node_logger.warning(f"y-Axis cannot be equal to x-Axis.")
                return False

        # below we actually mess with the reduction options, but
        # without using the decorated property.
        # make sure we emit the right signal at the end.
        reductionsChanged = False

        # Check: an axis marked as x/y cannot be also reduced.
        delete = []
        for n, _ in self._reductions.items():
            if n in self._xyAxes:
                self.node_logger.debug(
                    f"{n} has been selected as axis, cannot be reduced.")
                delete.append(n)
            if n in self._secondaryAxes:
                # keep reduction on secondary-axis dimensions
                continue
        for n in delete:
            del self._reductions[n]
            reductionsChanged = True

        # check: axes not marked as x/y should all be reduced.
        for ax in availableAxes:
            if ax not in self._xyAxes:
                if ax not in self._reductions:
                    self.node_logger.debug(
                        f"{ax} must be reduced. "
                        f"Default to selecting first element.")

                    # reductions are only supported on GridData
                    if isinstance(data, MeshgridDataDict):
                        red: Optional[ReductionType] = (ReductionMethod.elementSelection, [],
                               dict(index=0))
                    else:
                        red = None

                    self._reductions[ax] = red
                    reductionsChanged = True

        for ax in self._secondaryAxes:
            if ax is not None and ax in self._xyAxes:
                if ax == self._secondaryAxes[0]:
                    self._secondaryAxes = (None, self._secondaryAxes[1])
                else:
                    self._secondaryAxes = (self._secondaryAxes[0], None)
                reductionsChanged = True

        candidates = self._detect_secondary_role_candidates(data)
        if candidates != self.secondaryRoleCandidates:
            self.secondaryRoleCandidates = candidates

        xsec, ysec = self._secondaryAxes
        if xsec is not None and xsec not in candidates.get('x', []):
            self._secondaryAxes = (None, self._secondaryAxes[1])
            reductionsChanged = True
        if ysec is not None and ysec not in candidates.get('y', []):
            self._secondaryAxes = (self._secondaryAxes[0], None)
            reductionsChanged = True

        # Default behavior for coupled sweeps: if a coupled candidate exists,
        # auto-select one secondary axis. Prefer the candidate on an axis of
        # length 1 in the gridded data (common for coupled sweeps).
        xsec, ysec = self._secondaryAxes
        used = [ax for ax in [xsec, ysec] if ax is not None]

        if ysec is None and self._xyAxes[1] is not None:
            ypick = self._pick_default_secondary(data, 'y', candidates.get('y', []), used)
            if ypick is not None:
                self._secondaryAxes = (self._secondaryAxes[0], ypick)
                used.append(ypick)
                reductionsChanged = True

        xsec, ysec = self._secondaryAxes
        used = [ax for ax in [xsec, ysec] if ax is not None]
        if xsec is None and self._xyAxes[0] is not None:
            xpick = self._pick_default_secondary(data, 'x', candidates.get('x', []), used)
            if xpick is not None:
                self._secondaryAxes = (xpick, self._secondaryAxes[1])
                reductionsChanged = True

        # emit signal that we've changed things
        if reductionsChanged:
            self.optionChangeNotification.emit(
                {'dimensionRoles': self.dimensionRoles}
            )
        return True

    def process(
            self,
            dataIn: Optional[DataDictBase] = None
    ) -> Optional[Dict[str, Optional[DataDictBase]]]:
        if dataIn is None:
            return None

        secondary_axis_info = self._build_secondary_axis_info(dataIn)

        data = super().process(dataIn=dataIn)
        if data is None:
            return None
        dataout = data['dataOut']
        assert dataout is not None
        data = dataout.copy()

        if self._xyAxes[0] is not None and self._xyAxes[1] is not None:
            _kw = {self._xyAxes[0]: 0, self._xyAxes[1]: 1}
            data = data.reorder_axes(None, **_kw)

        if secondary_axis_info is not None:
            data.add_meta('coupled_secondary_axis', secondary_axis_info)

        # it is possible that UI options have been re-generated, while the
        # options in the node have not been changed. to make sure everything
        # is in sync, we simply set the UI options again here.
        if self.ui is not None:
            self.ui.setRoles(self.dimensionRoles)

        return dict(dataOut=data)
