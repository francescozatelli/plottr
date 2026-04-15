from enum import Enum, unique
from typing import Optional, Dict, Any

import numpy as np

# Use plottr's bundled implementation to avoid deprecated qcodes plotting APIs.
from plottr.utils.find_scale_and_prefix import find_scale_and_prefix

from plottr import QtWidgets, Signal, Slot
from plottr.node import Node, NodeWidget, updateOption
from plottr.data.datadict import DataDictBase


@unique
class ScaleUnitsOption(Enum):
    """Options for how to scale units."""

    #: do not scale units
    never = 0

    #: all units
    always = 1


class ScaleUnitOptionWidget(QtWidgets.QWidget):
    """A widget that allows the user to specify if units should be scaled."""

    unit_scale_selected = Signal(ScaleUnitsOption)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)

        self.buttons = {
            ScaleUnitsOption.never: QtWidgets.QRadioButton('Never'),
            ScaleUnitsOption.always: QtWidgets.QRadioButton('Always'),
        }
        btnLayout = QtWidgets.QVBoxLayout()
        self.btnGroup = QtWidgets.QButtonGroup(self)

        for opt in ScaleUnitsOption:
            btn = self.buttons[opt]
            self.btnGroup.addButton(btn, opt.value)
            btnLayout.addWidget(btn)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(btnLayout)
        layout.addStretch()
        self.setLayout(layout)
        self.buttons[ScaleUnitsOption.always].setChecked(True)

        self.btnGroup.buttonToggled.connect(self.unitscale_button_selected)

    @Slot(QtWidgets.QAbstractButton, bool)
    def unitscale_button_selected(
            self,
            btn: QtWidgets.QAbstractButton,
            checked: bool
    ) -> None:
        if checked:
            self.unit_scale_selected.emit(ScaleUnitsOption(self.btnGroup.id(btn)))


class ScaleUnitsWidget(NodeWidget):
    """Node widget for :class:`ScaleUnits`."""

    def __init__(self, node: Optional[Node] = None):
        super().__init__(node=node, embedWidgetClass=ScaleUnitOptionWidget)
        assert self.widget is not None
        self.widget.unit_scale_selected.connect(
            lambda x: self.signalOption('scale_unit_option')
        )

        self.optSetters = {
            'scale_unit_option': self.set_scale_unit_option,
        }

        self.optGetters = {
            'scale_unit_option': self.get_scale_units_option,
        }

    def get_scale_units_option(self) -> ScaleUnitsOption:
        assert self.widget is not None
        activeBtn = self.widget.btnGroup.checkedButton()
        activeId = self.widget.btnGroup.id(activeBtn)
        return ScaleUnitsOption(activeId)

    def set_scale_unit_option(self, option: ScaleUnitsOption) -> None:
        assert self.widget is not None
        self._emitUpdate = False
        for k, btn in self.widget.buttons.items():
            if k == option:
                btn.setChecked(True)
        self._emitUpdate = True


class ScaleUnits(Node):
    """
    A Node that automatically scales the units and values such that
    for example 1e-9 V will be rendered as 1 nV. The logic for how to scale units
    in inherited from QCoDeS. Basically pure SI units are scaled with enginering prefixes
    while more complex units are scaled by adding a power of 10 prefix to the unit
    e.g (1*10**3 complexunit)
    """
    useUi = True
    nodeName = "ScaleUnits"
    uiClass = ScaleUnitsWidget

    def __init__(self, name: str):
        super().__init__(name)
        self._scale_unit_option: ScaleUnitsOption = ScaleUnitsOption.always

    @property
    def scale_unit_option(self) -> ScaleUnitsOption:
        return self._scale_unit_option

    @scale_unit_option.setter
    @updateOption('scale_unit_option')
    def scale_unit_option(self, value: ScaleUnitsOption) -> None:
        self._scale_unit_option = value

    def process(self, dataIn: Optional[DataDictBase] = None) -> Optional[Dict[str, Optional[DataDictBase]]]:
        if super().process(dataIn=dataIn) is None:
            return None
        assert dataIn is not None
        data = dataIn.copy()

        axis_scales: Dict[str, int] = {}
        axis_scale_info: Dict[str, Dict[str, Any]] = {}

        if self.scale_unit_option != ScaleUnitsOption.never:
            for name, data_item in data.data_items():
                raw_unit = str(data_item.get("unit", ""))
                prefix, selected_scale = find_scale_and_prefix(
                    data_item['values'],
                    data_item["unit"]
                )
                data_item["unit"] = prefix + data_item["unit"]
                data_item['values'] = data_item['values'] * 10**(-selected_scale)
                axis_scales[name] = selected_scale
                axis_scale_info[name] = {
                    'scale': int(selected_scale),
                    'unit_raw': raw_unit,
                    'unit_scaled': str(data_item.get('unit', '')),
                }

            # Keep coupled secondary-axis metadata in sync with scaled axes.
            if data.has_meta('coupled_secondary_axis'):
                info: Any = data.meta_val('coupled_secondary_axis')
                info_list = info if isinstance(info, list) else [info]
                updated: list[Dict[str, Any]] = []

                def _scale_vals(axis_name: Any, vals: Any, unit: str) -> tuple[np.ndarray, int, str]:
                    arr = np.asarray(vals, dtype=float)
                    if arr.size == 0:
                        return arr, 0, unit

                    if isinstance(axis_name, str) and axis_name in axis_scales:
                        scale = axis_scales[axis_name]
                        return arr * (10 ** (-scale)), scale, unit

                    prefix, selected_scale = find_scale_and_prefix(arr, unit)
                    return arr * (10 ** (-selected_scale)), selected_scale, prefix + unit

                for ent in info_list:
                    if not isinstance(ent, dict):
                        continue

                    ent2 = dict(ent)
                    primary_axis = ent2.get('primary_axis')
                    secondary_axis = ent2.get('secondary_axis')

                    p_unit = str(ent2.get('primary_unit', ''))
                    s_unit = str(ent2.get('secondary_unit', ''))

                    p_vals, p_scale, p_unit_scaled = _scale_vals(
                        primary_axis,
                        ent2.get('primary_values', []),
                        p_unit,
                    )
                    s_vals, s_scale, s_unit_scaled = _scale_vals(
                        secondary_axis,
                        ent2.get('secondary_values', []),
                        s_unit,
                    )

                    ent2['primary_values'] = p_vals.tolist()
                    ent2['secondary_values'] = s_vals.tolist()
                    ent2['primary_scale'] = int(p_scale)
                    ent2['secondary_scale'] = int(s_scale)
                    ent2['primary_unit'] = p_unit_scaled
                    ent2['secondary_unit'] = s_unit_scaled

                    updated.append(ent2)

                if isinstance(info, list):
                    data.add_meta('coupled_secondary_axis', updated)
                elif len(updated) > 0:
                    data.add_meta('coupled_secondary_axis', updated[0])

        if len(axis_scale_info) > 0:
            data.add_meta('axis_scale_info', axis_scale_info)

        return dict(dataOut=data)
