"""Clipboard helpers for plot images."""

import base64
from typing import Any

from plottr import QtCore, QtWidgets


def copy_image_to_clipboard(image: Any, dpi: float = 180.0) -> None:
    """Copy an image at a smaller paste size without downsampling pixels."""
    dpi = max(96.0, float(dpi))
    dots_per_meter = int(round(dpi / 0.0254))
    image.setDotsPerMeterX(dots_per_meter)
    image.setDotsPerMeterY(dots_per_meter)

    mime_data = QtCore.QMimeData()

    png_bytes = QtCore.QByteArray()
    buffer = QtCore.QBuffer(png_bytes)
    write_only = getattr(QtCore.QIODevice, 'WriteOnly', None)
    if write_only is None:
        write_only = QtCore.QIODevice.OpenModeFlag.WriteOnly

    if buffer.open(write_only):
        image.save(buffer, 'PNG')
        buffer.close()

        display_width = max(1, int(round(image.width() * 96.0 / dpi)))
        encoded = base64.b64encode(bytes(png_bytes)).decode('ascii')
        mime_data.setHtml(
            '<html><body>'
            f'<img src="data:image/png;base64,{encoded}" '
            f'width="{display_width}" '
            f'style="width:{display_width}px;height:auto;" />'
            '</body></html>'
        )
        mime_data.setData('image/png', png_bytes)
    else:
        mime_data.setImageData(image)

    QtWidgets.QApplication.clipboard().setMimeData(mime_data)
