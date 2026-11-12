"""Web interface tests: every exported endpoint answers, the static build is complete, and the JS engine tests pass.

Run: python -m pytest -q tests/test_web.py
"""
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cell_segmenter.web.app import STATIC, STATIC_ENDPOINTS, app  # noqa: E402

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200 and "static/main.js" in r.text


@pytest.mark.parametrize("path", STATIC_ENDPOINTS)
def test_endpoint(path):
    r = client.get("/api" + path)
    assert r.status_code == 200, r.text
    assert r.json() is not None


def test_static_build(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import build_static
    build_static.build(tmp_path / "site")
    assert (tmp_path / "site" / "index.html").read_text().count("APP_STATIC") == 1
    for path in STATIC_ENDPOINTS:
        assert (tmp_path / "site" / "api" / (path.strip("/") + ".json")).exists()


def test_static_assets_referenced_exist():
    html = (STATIC / "index.html").read_text()
    for name in ("style.css", "main.js"):
        assert f"static/{name}" in html and (STATIC / name).exists()


@pytest.mark.skipif(not shutil.which("node"), reason="node not installed")
def test_js_engine():
    tests = sorted(STATIC.glob("*.test.mjs"))
    for t in tests:
        r = subprocess.run(["node", str(t)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr


# --- project-specific API tests ---------------------------------------------------------------------------------------
import io  # noqa: E402

import numpy as np  # noqa: E402

from cell_segmenter.web.segment_service import b64, pack_mask, unb64, unpack_mask  # noqa: E402


def test_info_defaults_come_from_config():
    from cell_segmenter import config
    j = client.get("/api/info").json()
    assert j["defaults"] == {"n_estimators": config.RF_N_ESTIMATORS, "max_depth": config.RF_MAX_DEPTH,
                             "random_state": config.RF_RANDOM_STATE, "brush_size": config.DEFAULT_BRUSH_SIZE, "overlay_opacity": config.OVERLAY_OPACITY}
    assert j["features"]["count"] == 36 and len(j["demo"]["accuracy"]) == j["demo"]["frames"]


def test_demo_payload_is_consistent():
    j = client.get("/api/demo").json()
    s = j["summary"]
    t, h, w = s["frames"], s["height"], s["width"]
    frames = unb64(j["frames"], np.uint16, (t, h, w))
    pred = unb64(j["python_pred"], np.uint8, (t, h, w))
    truth = np.stack([unpack_mask(m, (h, w)) for m in j["truth"]])
    assert frames.std() > 0 and set(np.unique(pred)) <= {0, 1}
    acc = [float(((pred[f] > 0) == truth[f]).mean()) for f in range(t)]
    assert np.allclose(acc, s["accuracy"]) and min(acc) > 0.8
    assert [l["name"] for l in j["labels"]] == ["Background", "Cell"]
    assert len(j["forest"]["trees"]) == s["n_estimators"] and j["forest"]["n_features"] == 36


def _upload(arr, name="t.tif"):
    import tifffile
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr)
    return client.post("/api/stack", files={"file": (name, buf.getvalue(), "image/tiff")})


def test_upload_and_segment_with_python():
    from cell_segmenter.utils.synthetic_stack import make_synthetic_stack
    st, gt, strokes = make_synthetic_stack(n_frames=2, shape=(64, 72), n_cells=3, seed=2)
    r = _upload(st)
    assert r.status_code == 200 and r.json()["shape"] == [2, 64, 72]
    body = {"stack": r.json()["id"], "train_frame": 0, "frames": [0, 1], "seed": 0,
            "labels": [{"name": k, "mask": pack_mask(strokes[k])} for k in ("Background", "Cell")]}
    j = client.post("/api/segment", json=body).json()
    assert j["engine"] == "python" and j["classes"] == [0, 1] and len(j["importances"]) == 36
    p = unb64(j["pred"]["1"], np.uint8, (64, 72))
    assert ((p > 0) == gt[1]).mean() > 0.7


def test_segment_errors():
    assert client.post("/api/segment", json={"stack": "nope", "labels": [], "frames": [0]}).status_code == 404
    one = {"name": "a", "mask": pack_mask(np.ones((160, 160), bool))}
    assert client.post("/api/segment", json={"stack": "demo", "labels": [one], "frames": [0]}).status_code == 400
    assert client.post("/api/segment", json={"stack": "demo", "labels": [one, one], "frames": [99]}).status_code == 400
    assert client.post("/api/stack", files={"file": ("x.tif", b"not a tiff", "image/tiff")}).status_code == 400


def test_single_page_upload_becomes_one_frame():
    assert _upload(np.arange(30, dtype=np.uint16).reshape(5, 6)).json()["shape"] == [1, 5, 6]


def test_b64_roundtrip():
    a = np.random.default_rng(0).random((3, 4))
    assert np.array_equal(unb64(b64(a, np.float64), np.float64, (3, 4)), a)
    m = a > 0.5
    assert np.array_equal(unpack_mask(pack_mask(m), m.shape), m)
