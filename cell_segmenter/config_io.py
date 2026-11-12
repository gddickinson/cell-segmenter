"""Read and write config.py for the Settings dialog (Qt-free).

render_config(changes) returns the text of a new config.py: every setting currently in config, with `changes` applied.
Paths keep their expressions (relative to PROJECT_ROOT), other values are written with repr(), so strings with quotes
and None survive a save.
"""
from . import config

PATH_EXPRESSIONS = {
    "PROJECT_ROOT": "Path(__file__).parent",
    "DATA_DIR": 'PROJECT_ROOT / "data"',
    "MODELS_DIR": 'PROJECT_ROOT / "models"',
    "LOG_FILE": 'PROJECT_ROOT / "cell_segmentation.log"',
}


def current_settings() -> dict:
    """Upper-case settings of the loaded config module, in definition order."""
    return {k: v for k, v in vars(config).items() if k.isupper()}


def render_config(changes: dict) -> str:
    settings = {**current_settings(), **changes}
    out = ['"""Configuration settings for the cell segmentation application."""',
           "import os", "from pathlib import Path", "", "# Project paths"]
    out += [f"{k} = {PATH_EXPRESSIONS[k]}" for k in ("PROJECT_ROOT", "DATA_DIR", "MODELS_DIR")]
    out += ["", "# Create directories if they don't exist", "os.makedirs(DATA_DIR, exist_ok=True)",
            "os.makedirs(MODELS_DIR, exist_ok=True)", "", "# Settings"]
    for k, v in settings.items():
        if k in ("PROJECT_ROOT", "DATA_DIR", "MODELS_DIR"):
            continue
        out.append(f"{k} = {PATH_EXPRESSIONS[k]}" if k in PATH_EXPRESSIONS else f"{k} = {v!r}")
    return "\n".join(out) + "\n"
