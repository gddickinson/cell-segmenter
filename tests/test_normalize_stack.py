"""Regression tests for normalize_stack (used by the GUI's Load TIFF Stack).

The GUI divided by (max - min) per frame, so a constant frame (e.g. a blank first frame) became all NaN,
and a single-page TIFF (2-D array) was silently ignored.
"""
import numpy as np

from cell_segmenter.utils.image_utils import normalize_stack


def test_per_frame_minmax():
    stack = np.stack([np.arange(12, dtype=np.uint16).reshape(3, 4) * k + k for k in (1, 3)])
    out = normalize_stack(stack)
    assert out.dtype == np.float64 and out.shape == (2, 3, 4)
    for f in out:
        assert f.min() == 0.0 and f.max() == 1.0


def test_constant_frame_is_zero_not_nan():
    stack = np.zeros((2, 4, 4), np.uint16)
    stack[1] = np.arange(16).reshape(4, 4)
    stack[0] = 7
    out = normalize_stack(stack)
    assert np.isfinite(out).all() and (out[0] == 0).all()


def test_single_page_becomes_one_frame_stack():
    out = normalize_stack(np.arange(6.0).reshape(2, 3))
    assert out.shape == (1, 2, 3) and out.max() == 1.0
