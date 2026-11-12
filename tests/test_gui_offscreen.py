"""The desktop MainWindow constructs and runs the train / segment workflow offscreen (PyTorch stubbed if absent)."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "gui_scripts" / "gui_smoke.py"
pytest.importorskip("PyQt6")
pytest.importorskip("pyqtgraph")


def run(mode):
    r = subprocess.run([sys.executable, str(SCRIPT), mode], capture_output=True, text=True, timeout=300,
                       env={"QT_QPA_PLATFORM": "offscreen", "PATH": "/usr/bin:/bin"})
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT ")]
    assert r.returncode == 0 and line, (r.stdout + r.stderr)[-3000:]
    return json.loads(line[0][7:])


def test_main_window_workflow():
    out = run("basic")
    for m in ("load_tiff", "add_label", "train_model", "segment_current_frame", "segment_all_frames", "update_overlay",
              "rename_label", "remove_label", "clear_all_training", "show_settings", "show_feature_viewer"):
        assert m in out["methods"]
    assert out["train1_classes"] == [0, 1]
    assert out["result_frames"] == [0, 1, 2] and min(out["result_agreement"]) > 0.6
    assert not [m for m in out["messages"] if m[0] == "critical"], out["messages"]


def test_feature_viewer_and_training_panel():
    out = run("viewer")
    assert len(out["viewer_checkboxes"]) >= 8 and out["panel_features"]
    assert out["train1_classes"] == [0, 1]
    assert not [m for m in out["messages"] if m[0] == "critical"], out["messages"]


def test_train_on_all_frames_and_seed():
    """'Use training data from all frames' gathers strokes from every frame; results are not training data."""
    out = run("extended")
    assert out["random_state"] == out["config_random_state"] == out["settings_seed"] == 0
    # frame 0 has only Background strokes: training on it alone is refused with a message
    assert out["frame0_only_messages"] and out["frame0_only_messages"][0][0] == "warning"
    assert "two labels" in out["frame0_only_messages"][0][2]
    # ticked: Background (frames 0, 1) and Cell (frame 1) are all used, the segmentation results are not
    assert out["all_frames_classes"] == [0, 1]
    assert out["all_frames_n_train"] == out["expected_n_train"]
    assert out["result_labels_flagged"] == [False, False, True, True]
    assert out["reproducible"]
