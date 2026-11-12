"""The training panel's feature selection reaches the model: only the selected feature maps are used for training and
prediction (before, the selection was passed to train() and ignored, so all 36 maps were always used)."""
import warnings

import numpy as np
import pytest

from cell_segmenter.models.features import FEATURE_GROUPS, feature_indices
from cell_segmenter.models.random_forest import RandomForestModel
from cell_segmenter.utils.image_utils import normalize_stack
from cell_segmenter.utils.synthetic_stack import make_synthetic_stack


@pytest.fixture(scope="module")
def data():
    st, gt, strokes = make_synthetic_stack(n_frames=2, shape=(64, 72), n_cells=3, seed=2)
    return normalize_stack(st), strokes


def test_groups_cover_all_36_once():
    rows = sorted(i for idx in FEATURE_GROUPS.values() for i in idx)
    assert rows == list(range(36))
    assert feature_indices(None) is None and feature_indices({}) is None and feature_indices(list(FEATURE_GROUPS)) is None
    assert feature_indices({"Gabor Features": True, "LBP Features": True, "Edge Features": False}) == [14] + list(range(21, 33))
    with pytest.raises(ValueError):
        feature_indices(["Nope"])


@pytest.mark.parametrize("selected,width", [(None, 36), ({"Original Intensity": True}, 1),
                                            (["Gaussian Features", "Structure Tensor"], 11),
                                            (["Gabor Features", "LBP Features"], 13)])
def test_selection_changes_feature_matrix_width(data, selected, width):
    norm, strokes = data
    m = RandomForestModel(n_estimators=5)
    m.select_features(selected)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X, y = m.prepare_training_data(norm[0], strokes)
        assert X.shape == (len(y), width)
        m.train(norm[0], strokes, selected_features=selected)
        pred = m.predict(norm[1])
    assert m.classifier.n_features_in_ == width and pred.shape == norm[1].shape


def test_selected_columns_are_the_right_maps(data):
    norm, strokes = data
    full, sub = RandomForestModel(), RandomForestModel()
    sub.select_features(["Hessian Features"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        Xf, _ = full.prepare_training_data(norm[0], strokes)
        Xs, _ = sub.prepare_training_data(norm[0], strokes)
    assert np.array_equal(Xs, Xf[:, 15:21])


def test_saved_model_keeps_selection(tmp_path, data):
    norm, strokes = data
    m = RandomForestModel(n_estimators=5)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m.train(norm[0], strokes, selected_features=["Original Intensity", "Entropy Features"])
        m.save_model(str(tmp_path / "m.joblib"))
        m2 = RandomForestModel()
        m2.load_model(str(tmp_path / "m.joblib"))
        assert m2.feature_indices == [0, 33] and np.array_equal(m2.predict(norm[1]), m.predict(norm[1]))
