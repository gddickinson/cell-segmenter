"""Offscreen run of the desktop MainWindow (with a stub PyTorch if it is not installed); prints a JSON summary.

Run by tests/test_gui_offscreen.py in a subprocess so the stub never leaks into other tests. Modal message boxes are
replaced by a recorder (the skill rule: never trigger a modal dialog from automation).
"""
import json
import os
import sys
import types
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import torch  # noqa: F401
    STUB = False
except ImportError:  # minimal stand-in so cnn.py imports; the CNN is not trained here
    STUB = True
    torch = types.ModuleType("torch")
    torch.Tensor = type("Tensor", (), {})
    nn = types.ModuleType("torch.nn")
    nn.Module = type("Module", (), {"__init__": lambda self, *a, **k: None})
    optim = types.ModuleType("torch.optim")
    torch.nn, torch.optim = nn, optim
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.device = lambda *a, **k: "cpu"
    sys.modules.update({"torch": torch, "torch.nn": nn, "torch.optim": optim})

import warnings  # noqa: E402

import numpy as np  # noqa: E402
from PyQt6.QtGui import QColor  # noqa: E402
from PyQt6.QtWidgets import QApplication, QListWidgetItem, QMessageBox  # noqa: E402

warnings.simplefilter("ignore")
messages = []
for kind in ("information", "warning", "critical"):
    setattr(QMessageBox, kind, staticmethod(lambda parent, title, text, *a, _k=kind: messages.append((_k, title, text))))

from cell_segmenter.data.label import Label  # noqa: E402
from cell_segmenter.gui.main_window import MainWindow  # noqa: E402
from cell_segmenter.utils.image_utils import normalize_stack  # noqa: E402
from cell_segmenter.utils.synthetic_stack import make_synthetic_stack  # noqa: E402

app = QApplication.instance() or QApplication([])
w = MainWindow()
out = {"stub_torch": STUB, "methods": sorted(m for m in dir(w) if not m.startswith("_") and callable(getattr(type(w), m, None))
                                          and (getattr(getattr(type(w), m), "__module__", None) or "").startswith("cell_segmenter"))}

stack, gt, strokes = make_synthetic_stack(n_frames=3, shape=(80, 96), n_cells=3, seed=2)
w.image_data = normalize_stack(stack)
w.image_view.setImage(w.image_data)
for name, col, frame_masks in (("Background", QColor(0, 0, 255), {0: strokes["Background"], 1: strokes["Background"]}),
                               ("Cell", QColor(255, 0, 0), {1: strokes["Cell"]})):
    lab = Label(name, col)
    lab.masks.update({k: v.copy() for k, v in frame_masks.items()})
    w.labels.append(lab)
    item = QListWidgetItem(name)
    w.label_list.addItem(item)
w.select_label(w.label_list.item(1))
w.set_tool_mode("paint")
w.paint_tool.paint_at_pos(95, 79)            # bottom-right corner (regression of the edge fix)
w.set_tool_mode("navigate")

mode = sys.argv[1] if len(sys.argv) > 1 else "basic"
if mode == "viewer":
    w.show_feature_viewer()
    fv = w.feature_viewer
    fv.update_features()
    out["viewer_checkboxes"] = sorted(fv.feature_checkboxes)
    out["viewer_params"] = sorted(fv.feature_params.get_values()) if hasattr(fv, "feature_params") else None
    fv.update_feature_extractor()
    w.training_panel.update_feature_list()
    out["panel_features"] = sorted(w.training_panel.get_selected_features())
w.current_frame = 1
w.train_model()
out["train1_classes"] = [int(c) for c in w.current_model.classifier.classes_]
w.segment_current_frame()
w.segment_all_frames()
out["n_labels_after_segment"] = len(w.labels)
res = w.labels[-1].masks
out["result_frames"] = sorted(int(k) for k in res)
out["result_agreement"] = [float((res[f] == gt[f]).mean()) for f in sorted(res)]
w.update_overlay()

if mode == "extended":
    from cell_segmenter import config
    out["random_state"] = w.current_model.classifier.random_state
    out["config_random_state"] = config.RF_RANDOM_STATE
    # current frame 0 has only Background strokes: training on it alone cannot learn two classes
    w.current_frame = 0
    w.use_all_frames.setChecked(False)
    w.train_model()
    out["frame0_only_messages"] = [m for m in messages if m[0] != "information"][-1:]
    w.use_all_frames.setChecked(True)
    messages.clear()
    w.train_model()
    m = w.current_model
    out["all_frames_classes"] = [int(c) for c in m.classifier.classes_]
    out["all_frames_n_train"] = int(m.n_training_samples)
    out["expected_n_train"] = int(strokes["Background"].sum() * 2 + strokes["Cell"].sum() + w.labels[1].masks[0].sum())
    out["result_labels_excluded"] = all(lab.is_result for lab in w.labels[2:])
    # reproducible: two trainings with the default seed give the same prediction
    p1 = m.predict(w.image_data[2])
    w.train_model()
    out["reproducible"] = bool(np.array_equal(p1, w.current_model.predict(w.image_data[2])))

w.undo_last_training()
w.remove_selected_label()
out["messages"] = [m[:2] for m in messages]
out["menus"] = [a.text() for a in w.menuBar().actions()]
w.close()
print("RESULT " + json.dumps(out))
