"""GUI components for the application.

Contents: MainWindow (main_window.py), PaintTool (paint_tool.py), FrameNavigator, IntensityAdjuster,
ModelParametersWidget (widgets.py); also training_panel.py, feature_viewer.py, settings_dialog.py.

Names are imported lazily, so importing one GUI module (e.g. paint_tool) does not pull in main_window and PyTorch.
"""
import importlib

_EXPORTS = {
    'MainWindow': '.main_window',
    'PaintTool': '.paint_tool',
    'FrameNavigator': '.widgets',
    'IntensityAdjuster': '.widgets',
    'ModelParametersWidget': '.widgets',
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(_EXPORTS[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
