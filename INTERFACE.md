# Cell Segmenter -- Interface Map

## Entry Point
- `main.py` -- Application entry point: calls `cell_segmenter.gui.main_window.main` (also the `cell-segmenter` command)

## cell_segmenter/ Package
- `__init__.py` -- Package metadata
- `config.py` -- Configuration constants: paths, model params (incl. `RF_RANDOM_STATE`, default 0), GUI params
- `config_io.py` -- `current_settings`, `render_config(changes)`: the config.py text written by Settings > Save

### cell_segmenter/gui/
- `__init__.py` -- Re-exports MainWindow, PaintTool, widgets (lazily, so paint_tool imports without PyTorch)
- `main_window.py` -- `MainWindow`: state, load TIFF, brush/mode, frame change, overlay, close; `main()` entry point. Composed of mixins:
  - `main_window_layout.py` -- `LayoutMixin`: widgets (image panel, file/label/brush/mode/model groups) and menu bar
  - `main_window_labels.py` -- `LabelActionsMixin`: add/select/rename/recolour/remove labels, undo, clear training
  - `main_window_model.py` -- `ModelActionsMixin`: select model, train (current frame or all frames), segment frame / all frames
  - `main_window_tools.py` -- `ToolsMixin`: settings, about, feature viewer, configuration reload
- `paint_tool.py` -- `PaintTool`: brush-based painting on image for labeling (uses `utils/brush.py::stamp_brush`)
- `widgets.py` -- Custom widgets for the GUI
- `training_panel.py` -- `TrainingPanel`: model training UI with progress indicators
- `feature_viewer.py` -- `FeatureVisualizerDialog`: visualize extracted features (features and statistics tabs)
  - `feature_viewer_params.py` -- `FeatureParamsMixin`: parameters tab, save/load/preset, apply to the extractor
  - `feature_viewer_widgets.py` -- `CollapsibleGroupBox`, `FeatureParameters`
- `settings_dialog.py` -- `SettingsDialog`: application settings (saves via `config_io.render_config`)

### cell_segmenter/models/
- `__init__.py` -- Re-exports SegmentationModel, RandomForestModel, CNNModel, FeatureExtractor (CNN lazily: torch only needed for the CNN)
- `base.py` -- `SegmentationModel`: abstract base class; `select_features`, `extract_selected_features`, `prepare_training_data`, `prepare_training_data_frames`
- `random_forest.py` -- `RandomForestModel(SegmentationModel)`: scikit-learn Random Forest; defaults read from config at construction (trees, depth, seed); `train`, `train_frames`, `predict`, save/load
- `cnn.py` -- `CNNModel(SegmentationModel)`: PyTorch CNN classifier
- `features.py` -- `FeatureExtractor`: multi-scale feature extraction (36 maps); `FEATURE_GROUPS` (training-panel names -> rows), `feature_indices(selected)`

### cell_segmenter/data/
- `__init__.py` -- Re-exports Label
- `label.py` -- `Label` dataclass: name, color, frame-indexed masks with merge/clear ops, `is_result` (segmentation results are not training data)

### cell_segmenter/utils/
- `__init__.py` -- Re-exports image utilities and logger
- `image_utils.py` -- Image manipulation: normalize, `normalize_stack` (per-frame [0,1], used by Load TIFF Stack and the web app), enhance
- `brush.py` -- `brush_disk`, `stamp_brush(mask, row, col, size)`: the paint brush, clipped at the edges (Qt-free)
- `synthetic_stack.py` -- `make_synthetic_stack()`: seeded synthetic DIC-like uint16 stack + known cell masks + example strokes (web demo, tests)
- `training_data.py` -- `gather_training_data(labels, frame, all_frames, shape)`, `n_classes_painted`: strokes to train on (Qt-free)
- `object_stats.py` -- `label_objects`, `object_stats(mask, image, um_per_px, min_area_px)`: 8-connected objects and regionprops (Qt-free)
- `logger.py` -- `setup_logger()`: logging configuration

### cell_segmenter/web/ (web interface; `__WEBMODULE__` = `cell_segmenter.web.app`)
- `app.py` -- FastAPI: `/api/info`, `/api/demo` (exported to the static site via `STATIC_ENDPOINTS`), `POST /api/stack` (upload TIFF), `POST /api/segment` (train/predict with RandomForestModel)
- `segment_service.py` -- base64/packbits helpers, `export_forest` (sklearn -> JSON), `train_and_predict`, `stats_for`, `quiet_logs`
- `demo.py` -- `demo_data()` (synthetic stack segmented by RandomForestModel, seed 0, cached), `demo_summary`, `demo_payload`
- `static/` -- ES-module front end, no build step:
  - `index.html`, `style.css` (shared kit theme), `style_app.css`, `icon.svg`, `core.js` (kit: api, router, helpers), `main.js` (pages, background)
  - `static_api.js` -- static site: answers GETs from exported JSON
  - `filters.js` -- scipy.ndimage-style correlate1d / convolve / gaussian (+derivatives) / edge filters
  - `features.js` -- JS port of `FeatureExtractor.extract_features` (36 maps, same order), `FEATURE_NAMES`
  - `forest.js` -- `predictForest` (exact sklearn predict for exported forests), `trainForest` (JS random forest), `forestFromJson`
  - `objects.js` -- port of `utils/object_stats.py`, `objectsCsv`
  - `tiff.js` -- `readTiff` (multi-page grayscale; none/LZW/Deflate/PackBits; OME/ImageJ pixel size), `writeTiff` (ImageJ stack)
  - `seg_worker.js` + `worker_client.js` -- features/training/prediction in a Web Worker
  - `state.js` -- session state (stack, labels, strokes, result), `loadDemo`, `loadFile`, `segment` (Python or JS engine)
  - `viewer.js` -- canvas composite (image, result, strokes, known outlines), brush, undo
  - `segment.js` (+ `results.js`: object table, exports), `features_page.js`, `home.js`, `docs.js` (method, about) -- pages
  - `*.test.mjs` -- node tests against `tests/fixtures/*.json`

## Root scripts
- `run_web.py` -- starts the web app on 127.0.0.1:8146
- `scripts/build_static.py` -- static GitHub Pages export (`site/`); `.github/workflows/pages.yml` publishes it
- `scripts/screenshots.py` -- Playwright smoke test + `docs/img/*.png`
- `scripts/make_web_fixtures.py` -- writes `tests/fixtures/{features,forest,objects,tiff}_case.json` from the Python code

## tests/
- `test_models.py` -- Tests for FeatureExtractor, Label, and module imports
- `test_imports_without_torch.py`, `test_brush.py`, `test_normalize_stack.py`, `test_config_io.py`, `test_random_forest_seed.py`,
  `test_training_data.py`, `test_packaging.py`, `test_feature_selection.py` -- regression tests for the 2026-11 fixes
- `test_gui_offscreen.py` (+ `gui_scripts/gui_smoke.py`) -- MainWindow, feature viewer, all-frames training offscreen (PyTorch stubbed)
- `test_web.py` -- API endpoints, static build, node engine tests
- `test_web_engine.py` -- fixtures current, JS TIFF writer readable by tifffile, synthetic stack, object stats

## Key Class Relationships
MainWindow manages a list of Labels and uses PaintTool for labeling.
RandomForestModel and CNNModel both inherit from SegmentationModel.
FeatureExtractor is used by both model types for feature computation.
Config module provides constants used throughout the package (the web app reads its defaults from it too).
The web app imports the Qt-free modules only (models, utils); it never imports gui/ or cnn.py.
