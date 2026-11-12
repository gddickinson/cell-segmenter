"""pip install works and the 'cell-segmenter' command points at a real function.

pyproject.toml named a build backend that does not exist (setuptools.backends._legacy), so `pip install -e .` from
the README failed, and the console script pointed at cell_segmenter.gui.main_window:main, which was not defined.
"""
import ast
import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_build_backend_importable():
    backend = PY["build-system"]["build-backend"]
    importlib.import_module(backend.split(":")[0])


def test_console_script_target_defined():
    for target in PY["project"]["scripts"].values():
        module, func = target.split(":")
        src = (ROOT / (module.replace(".", "/") + ".py")).read_text()
        assert func in {n.name for n in ast.parse(src).body if isinstance(n, ast.FunctionDef)}, target
