# Cell Segmentation Tool -- Roadmap

## Current State
A PyQt6-based interactive tool for segmenting microscopy images using Random Forest and CNN classifiers. Well-organized package in `cell_segmenter/` with subpackages: `gui/` (6 modules: main window, paint tool, widgets, training panel, feature viewer, settings), `models/` (4 modules: base, random forest, CNN, features), `data/` (label management), and `utils/` (image utils, logger). Has `setup.py`, `pyproject.toml`, and `requirements.txt`. Proper package with `__init__.py` files throughout.

## Short-term Improvements
- [x] Add unit tests for `models/features.py` (feature extraction) and `models/random_forest.py`
- [x] Add input validation in `data/label.py` for label name conflicts and color collisions
- [ ] Improve error handling in `gui/paint_tool.py` for edge cases (painting outside image bounds)
- [ ] Document the feature extraction pipeline in `models/features.py` with mathematical descriptions
- [ ] Add progress indicators for CNN training in `gui/training_panel.py`
- [ ] Validate TIFF bit depth on load and warn about unsupported formats

## Feature Enhancements
- [ ] Add model persistence -- save/load trained Random Forest and CNN models for reuse
- [ ] Implement semi-automatic labeling: model suggests labels, user corrects
- [ ] Add watershed post-processing to separate touching cells after classification
- [ ] Support multi-channel TIFF stacks (e.g., DIC + fluorescence as separate channels)
- [ ] Add comparison view showing original, labels, and segmentation side-by-side
- [ ] Implement magic wand / flood fill tool as alternative to brush painting

## Long-term Vision
- [ ] Add Cellpose and StarDist as additional segmentation backends in `models/`
- [ ] Implement transfer learning for CNN using pre-trained bioimage encoders
- [ ] Add napari plugin export for integration with the napari ecosystem
- [ ] Support 3D segmentation for z-stack data with 3D paint tools
- [ ] Create headless CLI for batch segmentation using trained models

## Technical Debt
- [x] Large `.tif` file in project root should be in `data/` with `.gitignore`
- [ ] `project-structure.txt` is a static file that will go stale -- consider auto-generation
- [x] `setup.py` should migrate to `pyproject.toml`
- [ ] `gui/feature_viewer.py` and `gui/settings_dialog.py` may be underused -- audit usage
- [ ] CNN model in `models/cnn.py` depends on PyTorch while rest uses scikit-learn -- document mixed ML stack
- [ ] No CI pipeline -- add automated tests and linting
