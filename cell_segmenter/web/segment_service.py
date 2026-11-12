"""Segmentation for the web app, using the repo's own code: utils.image_utils.normalize_stack (as the GUI's Load TIFF
Stack), models.random_forest.RandomForestModel (train on painted strokes of one frame, predict frames) and
utils.object_stats. Also the array <-> base64 helpers shared with the front end and the forest export for the JS port.
"""
import base64
import io
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from cell_segmenter.models.random_forest import RandomForestModel
from cell_segmenter.utils.image_utils import normalize_stack
from cell_segmenter.utils.object_stats import object_stats


def quiet_logs():
    """The package logs every step at DEBUG to stdout and a file; keep the web server's output readable."""
    for name in list(logging.root.manager.loggerDict):
        if name.startswith("cell_segmenter"):
            logging.getLogger(name).setLevel(logging.WARNING)


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.dtype(dtype).newbyteorder("<")).tobytes()).decode()


def unb64(s: str, dtype, shape=None) -> np.ndarray:
    a = np.frombuffer(base64.b64decode(s), dtype=np.dtype(dtype).newbyteorder("<")).astype(dtype)
    return a.reshape(shape) if shape is not None else a


def pack_mask(m: np.ndarray) -> str:
    """Boolean mask -> base64 of np.packbits (row-major, MSB first)."""
    return base64.b64encode(np.packbits(np.asarray(m, bool).ravel()).tobytes()).decode()


def unpack_mask(s: str, shape) -> np.ndarray:
    bits = np.unpackbits(np.frombuffer(base64.b64decode(s), np.uint8))[: int(np.prod(shape))]
    return bits.astype(bool).reshape(shape)


def export_forest(classifier) -> dict:
    """sklearn RandomForestClassifier -> plain JSON (per-node feature, threshold, children, class fractions)."""
    trees = []
    for est in classifier.estimators_:
        t = est.tree_
        v = t.value[:, 0, :]
        v = v / v.sum(axis=1, keepdims=True)
        trees.append(dict(feature=t.feature.tolist(), threshold=[float(x) for x in t.threshold],
                          left=t.children_left.tolist(), right=t.children_right.tolist(),
                          value=[float(x) for x in v.ravel()]))
    return dict(classes=[int(c) for c in classifier.classes_], n_features=int(classifier.n_features_in_), trees=trees)


def read_tiff_bytes(data: bytes) -> np.ndarray:
    """Raw TIFF array (2-D or 3-D) as tifffile reads it in the GUI."""
    import tifffile
    return tifffile.imread(io.BytesIO(data))


def train_and_predict(norm: np.ndarray, training: List[Tuple[int, Dict[str, np.ndarray]]], frames: List[int],
                      seed="config", n_estimators: Optional[int] = None,
                      max_depth: Optional[int] = None) -> dict:
    """Train RandomForestModel on painted strokes and predict the given frames.

    training is [(frame, {label name: bool mask})] (utils.training_data.gather_training_data): one frame trains with
    RandomForestModel.train, several with train_frames ("Use training data from all frames"). Label names are in label
    order in every dict; the predicted value of a pixel is the index of its label,
    as in the GUI (which shows 'segmentation > 0', i.e. every label except the first, as the result mask).
    seed: "config" (config.RF_RANDOM_STATE, the desktop default), an int, or None for a new forest every run.
    """
    kw = {k: v for k, v in dict(n_estimators=n_estimators, max_depth=max_depth).items() if v is not None}
    if seed != "config":
        kw["random_state"] = seed
    model = RandomForestModel(**kw)
    t0 = time.perf_counter()
    if len(training) == 1:
        model.train(norm[training[0][0]], training[0][1])
    else:
        model.train_frames([norm[f] for f, _ in training], [d for _, d in training])
    t1 = time.perf_counter()
    preds = {f: model.predict(norm[f]).astype(np.uint8) for f in frames}
    t2 = time.perf_counter()
    return dict(model=model, preds=preds, train_s=t1 - t0, predict_s=t2 - t1,
                importances=[float(x) for x in model.classifier.feature_importances_])


def stats_for(pred: np.ndarray, image: np.ndarray, um_per_px=None, min_area_px=0) -> dict:
    """Object statistics of the GUI's result mask (pred > 0) on one frame."""
    return object_stats(pred > 0, image, um_per_px=um_per_px, min_area_px=min_area_px)


__all__ = ["b64", "unb64", "pack_mask", "unpack_mask", "export_forest", "read_tiff_bytes", "normalize_stack",
           "train_and_predict", "stats_for", "quiet_logs"]
