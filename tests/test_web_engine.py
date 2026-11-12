"""The Python-vs-JS fixtures are current, the JS TIFF writer is readable by tifffile, and the Qt-free helpers behave.

Run: python -m pytest -q tests/test_web_engine.py   (regenerate fixtures: python scripts/make_web_fixtures.py)
"""
import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
FIX = ROOT / "tests" / "fixtures"
STATIC = ROOT / "cell_segmenter" / "web" / "static"


def _f64(s):
    return np.frombuffer(base64.b64decode(s), "<f8")


@pytest.fixture(scope="module")
def fresh():
    import make_web_fixtures
    return make_web_fixtures.build()


def test_features_fixture_current(fresh):
    old = json.loads((FIX / "features_case.json").read_text())
    for a, b in zip(old["cases"], fresh["features_case.json"]["cases"]):
        assert a["shape"] == b["shape"]
        np.testing.assert_allclose(_f64(a["image"]), _f64(b["image"]), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(_f64(a["features"]), _f64(b["features"]), rtol=1e-9, atol=1e-9)


def test_forest_fixture_current(fresh):
    old, new = json.loads((FIX / "forest_case.json").read_text()), fresh["forest_case.json"]
    for k in ("y", "sk_pred", "y_test", "sk_pred_test", "n_train", "crop_shape", "crop_pred", "crop_gt"):
        assert old[k] == new[k], k
    assert [len(t["left"]) for t in old["forest"]["trees"]] == [len(t["left"]) for t in new["forest"]["trees"]]


def test_objects_and_tiff_fixtures_current(fresh):
    old = json.loads((FIX / "objects_case.json").read_text())
    new = json.loads(json.dumps(fresh["objects_case.json"]))
    assert old["mask"] == new["mask"]
    for k in ("stats", "stats_um"):
        for a, b in zip(old[k]["objects"], new[k]["objects"]):
            assert a.keys() == b.keys()
            np.testing.assert_allclose([a[c] for c in a], [b[c] for c in a], rtol=1e-9)
    t_old = json.loads((FIX / "tiff_case.json").read_text())["files"]
    assert [(f["label"], f["expected"]) for f in t_old] == [(f["label"], f["expected"]) for f in fresh["tiff_case.json"]["files"]]


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_tiff_writer_read_by_tifffile(tmp_path):
    import tifffile
    out = tmp_path / "js.tif"
    r = subprocess.run(["node", str(STATIC / "tiff.test.mjs"), str(out)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    with tifffile.TiffFile(out) as tf:
        a = tf.asarray()
        assert tf.is_imagej and a.shape == (3, 5, 7) and a.dtype == np.uint16
        xres = tf.pages[0].tags["XResolution"].value
    expect = np.array([[(i * 911 + k * 7) % 65536 for i in range(35)] for k in range(3)]).reshape(3, 5, 7)
    assert np.array_equal(a, expect)
    assert abs(xres[1] / xres[0] - 0.25) < 1e-9  # µm per pixel


def test_synthetic_stack_is_seeded():
    from cell_segmenter.utils.synthetic_stack import make_synthetic_stack
    a, ga, sa = make_synthetic_stack(n_frames=2, shape=(80, 96), n_cells=2, seed=3)
    b, gb, sb = make_synthetic_stack(n_frames=2, shape=(80, 96), n_cells=2, seed=3)
    assert a.dtype == np.uint16 and a.shape == (2, 80, 96) and np.array_equal(a, b) and np.array_equal(ga, gb)
    assert sa["Cell"].any() and sa["Background"].any()
    assert not (sa["Cell"] & ~ga[0]).any() and not (sa["Background"] & ga[0]).any()


def test_object_stats_units():
    from cell_segmenter.utils.object_stats import object_stats
    m = np.zeros((10, 10), bool)
    m[1:4, 1:4] = True
    m[7:9, 6:9] = True
    s = object_stats(m)
    assert s["units"].startswith("px") and [o["area_px"] for o in s["objects"]] == [9, 6]
    s = object_stats(m, um_per_px=0.5, min_area_px=7)
    assert s["units"] == "um" and len(s["objects"]) == 1 and s["objects"][0]["area_um2"] == 9 * 0.25


def test_synthetic_stack_too_small_raises():
    from cell_segmenter.utils.synthetic_stack import make_synthetic_stack
    with pytest.raises(ValueError):
        make_synthetic_stack(n_frames=1, shape=(30, 30), n_cells=5, seed=1)
