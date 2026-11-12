"""Which painted strokes the model trains on (Qt-free; used by gui/main_window_model.py).

Labels are objects with .name, .masks ({frame: bool mask}) and .is_result (segmentation results, which are never
training data). Every returned labels_dict has all training labels in their list order, so a label keeps the same
class index on every frame (its position, as models/base.py::prepare_training_data numbers them).
"""
from typing import Dict, List, Sequence, Tuple

import numpy as np


def training_labels(labels: Sequence) -> list:
    """The labels that are training classes (not segmentation results), in order."""
    return [lab for lab in labels if not getattr(lab, "is_result", False)]


def gather_training_data(labels: Sequence, current_frame: int, all_frames: bool,
                         shape: Tuple[int, int]) -> List[Tuple[int, Dict[str, np.ndarray]]]:
    """[(frame, {label name: bool mask})] for the current frame, or for every frame with strokes when all_frames."""
    train = training_labels(labels)
    if all_frames:
        frames = sorted({f for lab in train for f, m in lab.masks.items() if m is not None and np.any(m)})
    else:
        frames = [current_frame]
    empty = np.zeros(shape, dtype=bool)
    return [(f, {lab.name: np.asarray(lab.masks.get(f, empty), bool) for lab in train}) for f in frames]


def n_classes_painted(data: List[Tuple[int, Dict[str, np.ndarray]]]) -> int:
    """Number of labels with at least one painted pixel in the gathered frames."""
    names = set()
    for _, d in data:
        names |= {k for k, m in d.items() if m.any()}
    return len(names)
