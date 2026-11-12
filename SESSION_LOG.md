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

**Open / for George**
- Wire up "Use training data from all frames" (`main_window.py` creates the checkbox but `train_model` never reads it)?
  It would change results when ticked, so left as is.
- `gui/main_window.py` (941 lines) and `gui/feature_viewer.py` (652) exceed the 500-line budget (pre-existing); not
  split here because the GUI cannot be exercised without PyTorch in this environment.
- `RandomForestModel` has no random seed, so desktop results vary run to run; the web demo sets `random_state=0`.
