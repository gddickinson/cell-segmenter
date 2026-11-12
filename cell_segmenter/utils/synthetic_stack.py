"""Synthetic DIC-like TIFF stack with known cell masks and example training strokes (numpy only, seeded).

Used as the demo data of the web app and in tests. It is NOT real microscopy: cells are smooth elliptical phase objects
with a few granules, drifting slowly; the DIC look comes from the directional derivative of the phase image along a
45 degree shear, plus a smooth illumination gradient and Gaussian camera noise, stored as uint16.

make_synthetic_stack() returns (stack uint16 (T, H, W), cell_masks bool (T, H, W), strokes) where strokes maps a label
name ("Background", "Cell") to a bool (H, W) mask of painted training pixels on frame 0 - the same kind of input the
desktop app's paint tool produces.
"""
import numpy as np
from scipy import ndimage

from .brush import stamp_brush

DEFAULTS = dict(n_frames=6, shape=(160, 160), n_cells=9, seed=7)


def _cells(rng, n_cells, shape):
    h, w = shape
    cells = []
    for _ in range(200000):
        if len(cells) == n_cells:
            break
        a, b = rng.uniform(9, 16), rng.uniform(7, 12)
        cy, cx = rng.uniform(a + 6, h - a - 6), rng.uniform(a + 6, w - a - 6)
        if all(np.hypot(cy - c["cy"], cx - c["cx"]) > a + c["a"] + 4 for c in cells):
            granules = [(rng.uniform(-0.6, 0.6), rng.uniform(-0.6, 0.6), rng.uniform(1.0, 2.2)) for _ in range(5)]
            cells.append(dict(cy=cy, cx=cx, a=a, b=b, theta=rng.uniform(0, np.pi),
                              vy=rng.normal(0, 0.6), vx=rng.normal(0, 0.6), granules=granules))
    if len(cells) < n_cells:
        raise ValueError(f"Could not place {n_cells} non-overlapping cells in a {h}x{w} image; use a larger shape")
    return cells


def _frame(cells, t, shape, rng):
    h, w = shape
    yy, xx = np.mgrid[:h, :w].astype(float)
    phase = np.zeros(shape)
    mask = np.zeros(shape, bool)
    for c in cells:
        cy, cx = c["cy"] + c["vy"] * t, c["cx"] + c["vx"] * t
        ct, st = np.cos(c["theta"] + 0.05 * t), np.sin(c["theta"] + 0.05 * t)
        u = ((xx - cx) * ct + (yy - cy) * st) / c["a"]
        v = (-(xx - cx) * st + (yy - cy) * ct) / c["b"]
        r2 = u * u + v * v
        inside = r2 <= 1.0
        mask |= inside
        phase += np.clip(1.0 - r2, 0, None) ** 0.5            # thicker in the middle, soft rim
        for gu, gv, gs in c["granules"]:
            gy = cy + gu * c["a"] * st + gv * c["b"] * ct
            gx = cx + gu * c["a"] * ct - gv * c["b"] * st
            phase += 0.35 * np.exp(-((yy - gy) ** 2 + (xx - gx) ** 2) / (2 * gs * gs)) * inside
    phase = ndimage.gaussian_filter(phase, 1.0)
    dic = (np.roll(phase, -1, axis=0) + np.roll(phase, -1, axis=1) - np.roll(phase, 1, axis=0)
           - np.roll(phase, 1, axis=1)) / 2.0                 # shear along the diagonal
    illum = 1.0 + 0.15 * (xx / w) - 0.1 * (yy / h)
    img = 9000.0 * illum + 5200.0 * dic + rng.normal(0, 260.0, shape)
    return np.clip(img, 0, 65535).astype(np.uint16), mask


def _line(mask, p0, p1, brush_size):
    n = int(max(abs(p1[0] - p0[0]), abs(p1[1] - p0[1]))) + 1
    for s in np.linspace(0, 1, n):
        stamp_brush(mask, int(round(p0[0] + s * (p1[0] - p0[0]))), int(round(p0[1] + s * (p1[1] - p0[1]))), brush_size)


def _strokes(cells, mask0, shape, brush_size=5):
    """Example training strokes on frame 0: a short line inside six of the cells, and lines across the background."""
    h, w = shape
    cell = np.zeros(shape, bool)
    for c in cells[:6]:
        ct, st = np.cos(c["theta"]), np.sin(c["theta"])
        d = 0.5 * c["a"]
        _line(cell, (c["cy"] - d * st, c["cx"] - d * ct), (c["cy"] + d * st, c["cx"] + d * ct), brush_size)
    cell &= ndimage.binary_erosion(mask0, iterations=2)
    far = ~ndimage.binary_dilation(mask0, iterations=5)
    bg = np.zeros(shape, bool)
    for frac in (0.12, 0.5, 0.88):
        _line(bg, (frac * h, 4), (frac * h, w - 5), brush_size)
        _line(bg, (4, frac * w), (h - 5, frac * w), brush_size)
    bg &= far
    return {"Background": bg, "Cell": cell}


def make_synthetic_stack(n_frames=None, shape=None, n_cells=None, seed=None):
    """Return (stack uint16 (T,H,W), cell_masks bool (T,H,W), strokes {label: bool (H,W)} on frame 0)."""
    n_frames = DEFAULTS["n_frames"] if n_frames is None else n_frames
    shape = tuple(DEFAULTS["shape"] if shape is None else shape)
    n_cells = DEFAULTS["n_cells"] if n_cells is None else n_cells
    seed = DEFAULTS["seed"] if seed is None else seed
    rng = np.random.default_rng(seed)
    cells = _cells(rng, n_cells, shape)
    frames, masks = zip(*(_frame(cells, t, shape, rng) for t in range(n_frames)))
    stack, cell_masks = np.stack(frames), np.stack(masks)
    return stack, cell_masks, _strokes(cells, cell_masks[0], shape)
