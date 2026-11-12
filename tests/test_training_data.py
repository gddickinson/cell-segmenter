"""gather_training_data: current frame vs all frames, result labels excluded, class order kept."""
from types import SimpleNamespace as NS

import numpy as np

from cell_segmenter.utils.training_data import gather_training_data, n_classes_painted


def m(*idx):
    a = np.zeros((4, 5), bool)
    for i in idx:
        a.flat[i] = True
    return a


LABELS = [NS(name="bg", masks={0: m(0, 1), 2: m(3)}, is_result=False),
          NS(name="cell", masks={2: m(7, 8)}, is_result=False),
          NS(name="Segmentation_2", masks={0: m(*range(20)), 1: m(5)}, is_result=True)]


def test_current_frame_only():
    data = gather_training_data(LABELS, 0, False, (4, 5))
    assert [f for f, _ in data] == [0] and list(data[0][1]) == ["bg", "cell"]
    assert data[0][1]["cell"].sum() == 0 and n_classes_painted(data) == 1


def test_all_frames():
    data = gather_training_data(LABELS, 0, True, (4, 5))
    assert [f for f, _ in data] == [0, 2]          # frame 1 holds only a result -> not used
    assert all(list(d) == ["bg", "cell"] for _, d in data)
    assert n_classes_painted(data) == 2
    assert sum(int(x.sum()) for _, d in data for x in d.values()) == 5


def test_random_forest_train_frames():
    import warnings
    from cell_segmenter.models.random_forest import RandomForestModel
    from cell_segmenter.utils.image_utils import normalize_stack
    from cell_segmenter.utils.synthetic_stack import make_synthetic_stack
    st, gt, strokes = make_synthetic_stack(n_frames=2, shape=(64, 72), n_cells=3, seed=2)
    norm = normalize_stack(st)
    empty = np.zeros_like(strokes["Cell"])
    model = RandomForestModel(n_estimators=10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.train_frames([norm[0], norm[1]], [{"bg": strokes["Background"], "cell": empty},
                                                {"bg": empty, "cell": strokes["Cell"]}])
    assert list(model.classifier.classes_) == [0, 1]
    assert model.n_training_samples == strokes["Background"].sum() + strokes["Cell"].sum()
