"""FastAPI web app for Cell Segmenter: serves the front end in static/ and the JSON API under /api.

Run with `python run_web.py` (http://127.0.0.1:8146, local only). `scripts/build_static.py` exports every path in
STATIC_ENDPOINTS to JSON for the GitHub Pages site, so keep that list in step with the GET routes below. The POST
routes (upload a TIFF, train/predict with the repo's RandomForestModel) exist only in the local app; on the static site
the browser runs the JS port instead (static_api.js).
"""
import uuid
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from cell_segmenter import config
from cell_segmenter.models.features import FeatureExtractor
from cell_segmenter.web.demo import demo_data, demo_payload, demo_summary
from cell_segmenter.web.segment_service import (b64, normalize_stack, quiet_logs, read_tiff_bytes, train_and_predict,
                                                unpack_mask)

STATIC = Path(__file__).resolve().parent / "static"
MAX_STACKS = 4
MAX_UPLOAD_MB = 512

quiet_logs()
app = FastAPI(title="Cell Segmenter")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
_stacks: "OrderedDict[str, np.ndarray]" = OrderedDict()  # uploaded stacks, normalised as the GUI does


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    """Project summary, the GUI defaults from config.py, the feature extractor's parameters and the demo summary."""
    fe = FeatureExtractor()
    return {
        "title": "Cell Segmenter", "version": "0.1.0",
        "defaults": {"n_estimators": config.RF_N_ESTIMATORS, "max_depth": config.RF_MAX_DEPTH,
                     "random_state": config.RF_RANDOM_STATE,
                     "brush_size": config.DEFAULT_BRUSH_SIZE, "overlay_opacity": config.OVERLAY_OPACITY},
        "features": {"sigmas": fe.sigmas, "hessian_sigmas": fe.hessian_sigmas,
                     "gabor_frequencies": fe.gabor_frequencies, "gabor_orientations": fe.gabor_orientations,
                     "entropy_radius": fe.entropy_radius, "structure_sigma": fe.structure_sigma, "count": 36},
        "demo": demo_summary(),
    }


@app.get("/api/demo")
def demo():
    """Demo stack, example strokes, known masks, Python predictions and the trained forest (for the JS port)."""
    return demo_payload()


@app.post("/api/stack")
async def upload_stack(file: UploadFile = File(...)):
    """Read a TIFF with tifffile and normalise it as the GUI does; keep it in memory for /api/segment."""
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1e6:
        raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB")
    try:
        raw = read_tiff_bytes(data)
        norm = normalize_stack(raw)
    except Exception as e:  # tifffile / shape errors -> a readable message
        raise HTTPException(400, f"Could not read this TIFF: {e}")
    sid = uuid.uuid4().hex[:12]
    _stacks[sid] = norm
    while len(_stacks) > MAX_STACKS:
        _stacks.popitem(last=False)
    return {"id": sid, "shape": list(norm.shape), "dtype": str(raw.dtype), "name": file.filename}


class LabelIn(BaseModel):
    name: str
    mask: str  # base64 np.packbits of the strokes on the training frame


class SegmentIn(BaseModel):
    stack: str  # "demo" or an id from /api/stack
    train_frame: int = 0
    labels: List[LabelIn]
    frames: List[int]
    seed: Optional[int] = None  # None = a new forest every run (the page pre-fills config.RF_RANDOM_STATE)


@app.post("/api/segment")
def segment(body: SegmentIn):
    """Train RandomForestModel on the strokes of one frame and predict the requested frames (label index per pixel)."""
    norm = demo_data()["norm"] if body.stack == "demo" else _stacks.get(body.stack)
    if norm is None:
        raise HTTPException(404, "Unknown stack: upload it again")
    t, h, w = norm.shape
    if not 0 <= body.train_frame < t or any(not 0 <= f < t for f in body.frames):
        raise HTTPException(400, f"Frames must be between 0 and {t - 1}")
    strokes = {lab.name: unpack_mask(lab.mask, (h, w)) for lab in body.labels}
    classes = {i for i, m in enumerate(strokes.values()) if m.any()}
    if len(classes) < 2:
        raise HTTPException(400, "Paint strokes for at least two labels on the training frame")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # skimage: LBP on float images
        r = train_and_predict(norm, body.train_frame, strokes, body.frames, seed=body.seed)
    return {
        "engine": "python", "frames": body.frames, "classes": [int(c) for c in r["model"].classifier.classes_],
        "pred": {str(f): b64(p, np.uint8) for f, p in r["preds"].items()},
        "importances": r["importances"], "train_s": r["train_s"], "predict_s": r["predict_s"],
        "n_train": int(sum(int(m.sum()) for m in strokes.values())),
    }


# GET paths (without the /api prefix) written to the static site as api/<path>.json.
STATIC_ENDPOINTS = ["/info", "/demo"]
