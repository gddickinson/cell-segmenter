"""Segmentation models and feature extraction.

Contents: SegmentationModel (base.py), RandomForestModel (random_forest.py), CNNModel and CNNArchitecture (cnn.py,
needs PyTorch), FeatureExtractor (features.py).

CNNModel/CNNArchitecture are imported lazily, so the feature extractor and the Random Forest work without PyTorch.
"""
from .base import SegmentationModel
from .random_forest import RandomForestModel
from .features import FeatureExtractor

__all__ = [
    'SegmentationModel',
    'RandomForestModel',
    'CNNModel',
    'CNNArchitecture',
    'FeatureExtractor'
]


def __getattr__(name):
    if name in ('CNNModel', 'CNNArchitecture'):
        from . import cnn
        return getattr(cnn, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
