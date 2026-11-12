"""RandomForestModel is reproducible by default (config.RF_RANDOM_STATE) and reads config when constructed.

Before: no random_state (a different forest, and slightly different segmentation, on every run), and the defaults
were bound when random_forest.py was imported, so Settings > Apply did not reach new models.
"""
import warnings

import numpy as np
import pytest

from cell_segmenter import config
from cell_segmenter.models.random_forest import RandomForestModel
from cell_segmenter.utils.image_utils import normalize_stack
from cell_segmenter.utils.synthetic_stack import make_synthetic_stack


@pytest.fixture(scope="module")
def data():
    st, gt, strokes = make_synthetic_stack(n_frames=2, shape=(64, 72), n_cells=3, seed=2)
    return normalize_stack(st), strokes


def fit_predict(norm, strokes, **kw):
    m = RandomForestModel(**kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.train(norm[0], strokes)
        return m, m.predict(norm[1])


def test_default_seed_from_config(data):
    m, p1 = fit_predict(*data, n_estimators=15)
    assert m.random_state == config.RF_RANDOM_STATE == m.classifier.random_state
    assert config.RF_RANDOM_STATE is not None
    _, p2 = fit_predict(*data, n_estimators=15)
    assert np.array_equal(p1, p2)


def test_explicit_seed_and_none(data):
    m, _ = fit_predict(*data, n_estimators=5, random_state=None)
    assert m.classifier.random_state is None
    a = [e.tree_.threshold.tolist() for e in fit_predict(*data, n_estimators=5, random_state=3)[0].classifier.estimators_]
    b = [e.tree_.threshold.tolist() for e in fit_predict(*data, n_estimators=5, random_state=3)[0].classifier.estimators_]
    assert a == b


def test_config_read_at_construction(monkeypatch):
    monkeypatch.setattr(config, "RF_N_ESTIMATORS", 7)
    monkeypatch.setattr(config, "RF_MAX_DEPTH", 3)
    monkeypatch.setattr(config, "RF_RANDOM_STATE", 11)
    m = RandomForestModel()
    assert (m.classifier.n_estimators, m.classifier.max_depth, m.classifier.random_state) == (7, 3, 11)


def test_save_load_keeps_seed(tmp_path, data):
    m, _ = fit_predict(*data, n_estimators=5, random_state=4)
    m.save_model(str(tmp_path / "rf.joblib"))
    m2 = RandomForestModel()
    m2.load_model(str(tmp_path / "rf.joblib"))
    assert m2.random_state == 4 and m2.is_trained
