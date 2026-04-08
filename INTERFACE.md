# Cell Segmenter -- Interface Map

## Entry Point
- `main.py` -- Application entry point, creates QApplication and MainWindow

## cell_segmenter/ Package
- `__init__.py` -- Package metadata
- `config.py` -- Configuration constants: paths, model params, GUI params

### cell_segmenter/gui/
- `__init__.py` -- Re-exports MainWindow, PaintTool
- `main_window.py` -- `MainWindow(QMainWindow)`: main GUI with image display, label management, training controls
- `paint_tool.py` -- `PaintTool`: brush-based painting on image for labeling
- `widgets.py` -- Custom widgets for the GUI
- `training_panel.py` -- `TrainingPanel`: model training UI with progress indicators
- `feature_viewer.py` -- `FeatureViewer`: visualize extracted features
- `settings_dialog.py` -- `SettingsDialog`: application settings

### cell_segmenter/models/
- `__init__.py` -- Re-exports SegmentationModel, RandomForestModel, CNNModel, FeatureExtractor
- `base.py` -- `SegmentationModel`: abstract base class for segmentation models
- `random_forest.py` -- `RandomForestModel(SegmentationModel)`: scikit-learn Random Forest classifier
- `cnn.py` -- `CNNModel(SegmentationModel)`: PyTorch CNN classifier
- `features.py` -- `FeatureExtractor`: multi-scale feature extraction (Gaussian, edge, LBP, Hessian, Gabor)

### cell_segmenter/data/
- `__init__.py` -- Re-exports Label
- `label.py` -- `Label` dataclass: name, color, frame-indexed masks with merge/clear ops

### cell_segmenter/utils/
- `__init__.py` -- Re-exports image utilities and logger
- `image_utils.py` -- Image manipulation: normalize, convert, enhance
- `logger.py` -- `setup_logger()`: logging configuration

## tests/
- `test_models.py` -- Tests for FeatureExtractor, Label, and module imports

## Key Class Relationships
MainWindow manages a list of Labels and uses PaintTool for labeling.
RandomForestModel and CNNModel both inherit from SegmentationModel.
FeatureExtractor is used by both model types for feature computation.
Config module provides constants used throughout the package.
