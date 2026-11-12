"""Web interface (FastAPI + static ES-module front end) and the static GitHub Pages export.

Contents: app.py (routes, STATIC_ENDPOINTS), segment_service.py (normalise, train/predict with RandomForestModel,
object stats, base64 helpers, forest export), demo.py (the synthetic demo stack and its Python segmentation),
static/ (front end, JS ports of the feature extractor, forest, object stats and a TIFF reader/writer).
"""
