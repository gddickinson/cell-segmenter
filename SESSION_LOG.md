# Session log

## 2026-11-11/12: web interface and static site (Claude, branch `web-interface`)

**Added**
- Web interface (`cell_segmenter/web/`, `run_web.py`, port 8146) and static GitHub Pages site
  (`scripts/build_static.py`, `.github/workflows/pages.yml`). Pages: Home, Segment (load TIFF/demo, labels, paint,
  train, segment frame/all, objects, PNG/TIFF/CSV export), Feature maps, Method, About.
- JS ports tested against Python with node: the 36-feature extractor (max relative difference < 1e-13), sklearn forest
  prediction (identical labels), a JS random forest for in-browser training (99.4-99.5% pixel agreement with the Python
  forest on the demo stack, three seeds), object statistics (= skimage regionprops), a TIFF reader/writer (checked
  against tifffile both ways).
- Qt-free helpers the web app and GUI share: `utils/brush.py`, `utils/object_stats.py`, `utils/synthetic_stack.py`
  (synthetic demo data: not real microscopy), `utils/image_utils.normalize_stack`.
- Tests: `tests/test_web.py`, `tests/test_web_engine.py`, fixtures in `tests/fixtures/` (`scripts/make_web_fixtures.py`).

**Fixed (separate commits, regression tests first)**
- Importing FeatureExtractor / RandomForestModel / PaintTool (and collecting the test suite) required PyTorch: CNN and
  GUI classes are now imported lazily (`tests/test_imports_without_torch.py`).
- Painting within brush_size pixels of the bottom/right edge failed with a broadcast error and painted nothing
  (seen in the committed log): brush moved to `utils/brush.py`, clipped on all sides (`tests/test_brush.py`).
- Load TIFF Stack turned a constant frame into NaN and silently ignored single-page TIFFs: now `normalize_stack`
  (`tests/test_normalize_stack.py`).
- The rotating log files in `cell_segmenter/` are no longer tracked (every run modified them).
- README: corrected the clone URL, "Undo/redo" (only undo, which clears the label's strokes on the frame) and
  multi-frame training (the "Use training data from all frames" checkbox is not wired up).

## 2026-11-12: follow-up fixes (George: "fix ... any other problems")
- "Use training data from all frames" now works (Random Forest; `utils/training_data.py`, `train_frames`), segmentation
  result labels are no longer fed back as training classes, and training needs strokes for two labels. Web page has
  the same checkbox (JS and Python engines).
- Random Forest seeded by default (`config.RF_RANDOM_STATE = 0`, Settings > Model > Random Seed, saved with the model);
  RandomForestModel reads config when constructed, so Settings > Apply reaches new models.
- `main_window.py` and `feature_viewer.py` split into mixins under 500 lines, methods moved verbatim; the offscreen GUI
  run (`tests/test_gui_offscreen.py`, PyTorch stubbed) gives the same methods, menus and workflow before and after.
- Also fixed: Train Model refused to run until the Feature Viewer had been opened; Settings > Save wrote a config.py
  without LOG_FILE (the app then failed to start); `pip install -e .` failed (non-existent build backend) and the
  `cell-segmenter` command pointed at a missing `main`; the synthetic generator could loop forever.

- The training panel's feature selection now reaches the models (`FEATURE_GROUPS`, `select_features`; only the ticked
  groups' maps are used for training and prediction, saved with the model); the web page has the same feature-group
  checkboxes (JS and Python engines). GUI tests now run Qt in subprocesses (PySide6 in the shared venv broke them).

**Open / for George**
- The CNN still trains on one frame (all-frames training is Random Forest only); the CNN was not run (no PyTorch here).
