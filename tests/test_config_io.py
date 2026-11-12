"""Settings > Save writes a config.py that still defines every setting (it used to drop LOG_FILE, so the logger
failed with AttributeError on the next start, and anything not on the dialog, such as RF_RANDOM_STATE)."""
import types

from cell_segmenter import config
from cell_segmenter.config_io import current_settings, render_config


def _exec(text, tmp_path):
    p = tmp_path / "config.py"
    p.write_text(text)
    mod = types.ModuleType("cfg")
    mod.__file__ = str(p)
    exec(compile(text, str(p), "exec"), mod.__dict__)
    return mod


def test_render_keeps_every_setting(tmp_path):
    mod = _exec(render_config({}), tmp_path)
    for k, v in current_settings().items():
        if k in ("PROJECT_ROOT", "DATA_DIR", "MODELS_DIR", "LOG_FILE"):  # same place relative to config.py
            assert getattr(mod, k).relative_to(mod.PROJECT_ROOT) == v.relative_to(config.PROJECT_ROOT), k
        else:
            assert getattr(mod, k) == v, k


def test_render_applies_changes(tmp_path):
    mod = _exec(render_config({"RF_N_ESTIMATORS": 250, "RF_RANDOM_STATE": None, "LOG_FORMAT": 'a "quoted" %(message)s'}), tmp_path)
    assert mod.RF_N_ESTIMATORS == 250 and mod.RF_RANDOM_STATE is None and mod.LOG_FORMAT == 'a "quoted" %(message)s'
    assert mod.LOG_FILE.name == config.LOG_FILE.name and mod.RF_MAX_DEPTH == config.RF_MAX_DEPTH
