"""
Smoke tests for cell segmenter models and features.

Tests feature extraction, random forest model, and label management.
"""

import sys
import os
import unittest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cell_segmenter.models.features import FeatureExtractor
from cell_segmenter.data.label import Label


class TestFeatureExtractor(unittest.TestCase):
    """Tests for FeatureExtractor."""

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_init(self):
        """Test extractor can be instantiated."""
        self.assertIsNotNone(self.extractor)
        self.assertIsInstance(self.extractor.sigmas, list)

    def test_extract_features_shape(self):
        """Test feature extraction produces correct output shape."""
        image = np.random.rand(64, 64).astype(np.float32)
        features = self.extractor.extract_features(image)
        self.assertIsNotNone(features)
        # Features should be 3D: (n_features, height, width)
        self.assertEqual(features.ndim, 3)
        self.assertEqual(features.shape[1], 64)
        self.assertEqual(features.shape[2], 64)

    def test_extract_features_nonzero(self):
        """Test that feature extraction produces non-trivial output."""
        # Use 0-1 range for float images (required by skimage)
        image = np.random.rand(32, 32).astype(np.float32)
        features = self.extractor.extract_features(image)
        # At least some features should be non-zero
        self.assertGreater(np.count_nonzero(features), 0)


class TestLabel(unittest.TestCase):
    """Tests for Label data class."""

    def _make_color(self):
        """Create a mock QColor-like object for testing without Qt."""
        try:
            from PyQt6.QtGui import QColor
            return QColor(255, 0, 0)
        except ImportError:
            # If Qt not available, skip
            self.skipTest("PyQt6 not available")

    def test_label_creation(self):
        """Test label can be created."""
        color = self._make_color()
        label = Label(name="test", color=color)
        self.assertEqual(label.name, "test")
        self.assertEqual(len(label.masks), 0)

    def test_add_mask(self):
        """Test adding a mask to a label."""
        color = self._make_color()
        label = Label(name="test", color=color)
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True
        label.add_mask(0, mask)
        self.assertIn(0, label.masks)
        self.assertTrue(np.array_equal(label.masks[0], mask))

    def test_get_mask(self):
        """Test retrieving a mask."""
        color = self._make_color()
        label = Label(name="test", color=color)
        mask = np.ones((5, 5), dtype=bool)
        label.add_mask(0, mask)
        result = label.get_mask(0)
        self.assertIsNotNone(result)
        self.assertTrue(np.array_equal(result, mask))

    def test_get_mask_missing(self):
        """Test retrieving a non-existent mask returns None."""
        color = self._make_color()
        label = Label(name="test", color=color)
        result = label.get_mask(99)
        self.assertIsNone(result)

    def test_merge_mask(self):
        """Test merging masks."""
        color = self._make_color()
        label = Label(name="test", color=color)
        mask1 = np.zeros((10, 10), dtype=bool)
        mask1[0:5, 0:5] = True
        label.add_mask(0, mask1)

        mask2 = np.zeros((10, 10), dtype=bool)
        mask2[3:8, 3:8] = True
        label.merge_mask(0, mask2)

        result = label.get_mask(0)
        # Should be union of both masks
        expected = np.logical_or(mask1, mask2)
        self.assertTrue(np.array_equal(result, expected))

    def test_clear_mask(self):
        """Test clearing a mask."""
        color = self._make_color()
        label = Label(name="test", color=color)
        mask = np.ones((5, 5), dtype=bool)
        label.add_mask(0, mask)
        label.clear_mask(0)
        self.assertNotIn(0, label.masks)

    def test_add_mask_type_conversion(self):
        """Test that non-bool masks are converted."""
        color = self._make_color()
        label = Label(name="test", color=color)
        mask = np.ones((5, 5), dtype=np.uint8)
        label.add_mask(0, mask)
        self.assertEqual(label.masks[0].dtype, bool)

    def test_add_mask_invalid_type(self):
        """Test that non-array raises ValueError."""
        color = self._make_color()
        label = Label(name="test", color=color)
        with self.assertRaises(ValueError):
            label.add_mask(0, "not an array")


class TestImports(unittest.TestCase):
    """Test that all modules can be imported."""

    def test_import_features(self):
        """Test features module imports."""
        from cell_segmenter.models.features import FeatureExtractor
        self.assertIsNotNone(FeatureExtractor)

    def test_import_random_forest(self):
        """Test random forest module imports."""
        from cell_segmenter.models.random_forest import RandomForestModel
        self.assertIsNotNone(RandomForestModel)

    def test_import_label(self):
        """Test label module imports."""
        from cell_segmenter.data.label import Label
        self.assertIsNotNone(Label)

    def test_import_config(self):
        """Test config module imports."""
        from cell_segmenter import config
        self.assertIsNotNone(config.RF_N_ESTIMATORS)

    def test_import_utils(self):
        """Test utils imports."""
        from cell_segmenter.utils.image_utils import normalize_image
        self.assertIsNotNone(normalize_image)


if __name__ == '__main__':
    unittest.main()
