"""Regression test: the feature extractor, Random Forest and paint tool import without PyTorch.

models/__init__.py and gui/__init__.py used to import cnn.py (and so torch) eagerly, so the whole test suite failed
to collect on a machine without PyTorch even though only the CNN needs it.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE = """
import sys

class NoTorch:  # any 'import torch' now raises ImportError, as on a machine without PyTorch
    def find_spec(self, name, path=None, target=None):
        if name == 'torch' or name.startswith('torch.'):
            raise ImportError('torch blocked for this test')

sys.meta_path.insert(0, NoTorch())
from cell_segmenter.models.features import FeatureExtractor
from cell_segmenter.models import RandomForestModel
import cell_segmenter.gui.paint_tool
print('ok')
"""


def test_import_without_torch():
    r = subprocess.run([sys.executable, "-c", CODE], cwd=ROOT, capture_output=True, text=True,
                       env={"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-800:]
