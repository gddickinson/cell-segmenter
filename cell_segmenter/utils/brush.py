"""Circular paint brush for training labels (Qt-free, used by gui/paint_tool.py and mirrored by the web app).

The brush is a disk of radius brush_size / 2 inside a (2*brush_size + 1)^2 window centred on (row, col), clipped to
the image so painting at or beyond any edge is safe.
"""
import numpy as np


def brush_disk(brush_size: int) -> np.ndarray:
    """Boolean (2*brush_size+1, 2*brush_size+1) disk: True where distance from the centre <= brush_size / 2."""
    y, x = np.ogrid[-brush_size:brush_size + 1, -brush_size:brush_size + 1]
    return np.sqrt(x * x + y * y) <= brush_size / 2


def stamp_brush(mask: np.ndarray, row: int, col: int, brush_size: int) -> np.ndarray:
    """OR the brush disk centred on (row, col) into mask in place (clipped at the edges); returns mask."""
    brush = brush_disk(brush_size)
    h, w = mask.shape
    r0, r1 = row - brush_size, row + brush_size + 1
    c0, c1 = col - brush_size, col + brush_size + 1
    rr0, rr1 = max(0, r0), min(h, r1)
    cc0, cc1 = max(0, c0), min(w, c1)
    if rr0 >= rr1 or cc0 >= cc1:
        return mask
    mask[rr0:rr1, cc0:cc1] |= brush[rr0 - r0:rr1 - r0, cc0 - c0:cc1 - c0]
    return mask
