"""Regression tests for the paint brush (cell_segmenter/utils/brush.py).

Painting near the bottom or right edge used to raise "operands could not be broadcast together"
because the brush was not cropped at the far edge (see cell_segmentation.log, 2025-04-14).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cell_segmenter.utils.brush import brush_disk, stamp_brush

ROOT = Path(__file__).resolve().parent.parent



def test_disk_shape():
    d = brush_disk(5)
    assert d.shape == (11, 11) and d[5, 5] and d.sum() == np.count_nonzero(
        np.hypot(*np.ogrid[-5:6, -5:6]) <= 2.5)


@pytest.mark.parametrize("row,col", [(0, 0), (54, 54), (54, 0), (0, 54), (53, 27), (27, 53), (27, 27)])
def test_stamp_inside_and_at_edges(row, col):
    mask = np.zeros((55, 55), bool)
    stamp_brush(mask, row, col, 5)
    assert mask[row, col]
    expected = np.hypot(np.arange(55)[:, None] - row, np.arange(55)[None, :] - col) <= 2.5
    assert np.array_equal(mask, expected)


def test_stamp_outside_is_noop():
    mask = np.zeros((20, 30), bool)
    for r, c in [(-20, 5), (5, -20), (40, 5), (5, 60)]:
        stamp_brush(mask, r, c, 5)
    assert not mask.any()


def test_stamp_merges():
    mask = np.zeros((20, 20), bool)
    mask[0, 0] = True
    stamp_brush(mask, 10, 10, 3)
    assert mask[0, 0] and mask[10, 10]


PAINT_CODE = """
from types import SimpleNamespace
from PyQt6.QtWidgets import QApplication
import pyqtgraph as pg
from cell_segmenter.gui.paint_tool import PaintTool
app = QApplication.instance() or QApplication([])
tool = PaintTool(pg.ImageView())
tool.brush_size = 5
tool.image_shape = (55, 55)
label = SimpleNamespace(masks={})
tool.parent_window = SimpleNamespace(current_frame=0, active_label=label, update_overlay=lambda: None)
tool.paint_at_pos(54, 54)   # x, y
tool.paint_at_pos(0, 54)
mask = label.masks[0]
print('PAINTED', bool(mask[54, 54]), bool(mask[54, 0]))
"""


@pytest.mark.skipif(not all(importlib.util.find_spec(m) for m in ("PyQt6", "pyqtgraph")), reason="needs PyQt6")
def test_paint_tool_paints_at_bottom_right_edge():
    """The GUI's PaintTool must paint (not log a broadcast error) at the image edge.

    Runs in a subprocess with a fresh Qt: other packages in a shared environment (e.g. PySide6) can load a different
    QtCore into the pytest process first.
    """
    r = subprocess.run([sys.executable, "-c", PAINT_CODE], cwd=ROOT, capture_output=True, text=True, timeout=120,
                       env={"QT_QPA_PLATFORM": "offscreen", "PYQTGRAPH_QT_LIB": "PyQt6", "PATH": "/usr/bin:/bin"})
    assert "PAINTED True True" in r.stdout, (r.stdout + r.stderr)[-2000:]
