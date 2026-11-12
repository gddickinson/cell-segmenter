"""Write the Python-vs-JS fixtures in tests/fixtures/ (read by cell_segmenter/web/static/*.test.mjs).

  features_case.json  FeatureExtractor.extract_features on a crop of the demo stack and on a tiny random image
  forest_case.json    a seeded RandomForestModel (10 trees) trained on demo strokes: exported trees, float32 features
                      and sklearn's predictions, plus the Python prediction for a whole crop
  objects_case.json   utils.object_stats on a mask with touching/diagonal objects
  tiff_case.json      TIFF files written by tifffile (uint8/uint16/float32, none/zlib/LZW+predictor/PackBits, big-endian)

Run: python scripts/make_web_fixtures.py   (tests/test_web_engine.py checks the fixtures are current)
"""
import io
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "tests" / "fixtures"

from cell_segmenter.models.features import FeatureExtractor  # noqa: E402
from cell_segmenter.utils.image_utils import normalize_stack  # noqa: E402
from cell_segmenter.utils.object_stats import object_stats  # noqa: E402
from cell_segmenter.utils.synthetic_stack import make_synthetic_stack  # noqa: E402
from cell_segmenter.web.segment_service import b64, export_forest, pack_mask, quiet_logs, train_and_predict  # noqa: E402

CROP = (slice(36, 60), slice(52, 80))  # 24 x 28 pixels of the demo stack (part of a cell and background)


def features_case():
    import warnings
    stack, _, _ = make_synthetic_stack()
    crop = normalize_stack(stack[:1, CROP[0], CROP[1]])[0]
    tiny = np.random.default_rng(3).random((7, 9))
    cases = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # skimage warns that LBP on float images is sensitive to ties
        for name, img in (("demo_crop", crop), ("tiny_random", tiny)):
            f = FeatureExtractor().extract_features(img)
            cases.append(dict(name=name, shape=list(img.shape), image=b64(img, np.float64), features=b64(f, np.float64)))
    return dict(n_features=36, cases=cases)


def forest_case():
    import warnings
    stack, gt, strokes = make_synthetic_stack()
    norm = normalize_stack(stack)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = train_and_predict(norm, [(0, strokes)], [0], seed=0, n_estimators=10)
        model = r["model"]
        X, y = model.prepare_training_data(norm[0], strokes)
        crop = norm[1][CROP[0], CROP[1]]
        crop_pred = model.predict(crop)
    X32 = X.astype(np.float32)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(X32))
    pick, test = perm[:500], perm[500:1000]
    return dict(forest=export_forest(model.classifier), n_train=int(len(y)),
                X=b64(X32[pick].T, np.float32), y=[int(v) for v in y[pick]],
                sk_pred=[int(v) for v in model.classifier.predict(X32[pick])],
                X_test=b64(X32[test].T, np.float32), y_test=[int(v) for v in y[test]],
                sk_pred_test=[int(v) for v in model.classifier.predict(X32[test])],
                crop_shape=list(crop.shape), crop_image=b64(crop, np.float64), crop_pred=b64(crop_pred, np.uint8),
                crop_gt=pack_mask(gt[1][CROP[0], CROP[1]]))


def objects_case():
    m = np.zeros((14, 17), bool)
    m[1:4, 1:6] = True          # rectangle
    m[4, 6] = True              # diagonal neighbour -> same object (8-connectivity)
    m[8:13, 2:5] = True
    m[9, 1] = True
    m[2:12, 10:12] = True       # tall bar
    m[6, 12:16] = True
    m[13, 16] = True            # single pixel in the corner
    img = np.random.default_rng(5).random(m.shape)
    return dict(shape=list(m.shape), mask=pack_mask(m), image=b64(img, np.float64),
                stats=object_stats(m, img), stats_um=object_stats(m, img, um_per_px=0.16, min_area_px=2))


def tiff_case():
    import tifffile
    rng = np.random.default_rng(11)
    arrays = {
        "u8_stack": rng.integers(0, 256, (3, 5, 7)).astype(np.uint8),
        "u16_stack": rng.integers(0, 65536, (4, 6, 5)).astype(np.uint16),
        "f32_single": rng.random((6, 8)).astype(np.float32),
    }
    files = []
    for name, opts in (("u8_stack", {}), ("u8_stack", {"photometric": "minisblack"}), ("u16_stack", {}), ("u16_stack", {"compression": "zlib"}),
                       ("u16_stack", {"compression": "lzw", "predictor": True}), ("u8_stack", {"compression": "packbits"}),
                       ("u16_stack", {"byteorder": ">"}), ("f32_single", {}), ("u16_stack", {"rowsperstrip": 2})):
        buf = io.BytesIO()
        tifffile.imwrite(buf, arrays[name], **opts)
        a = arrays[name]
        files.append(dict(label=f"{name} {opts}", array=name, dtype=str(a.dtype), shape=list(a.shape),
                          data=__import__("base64").b64encode(buf.getvalue()).decode(),
                          expected=a.ravel().tolist() if a.dtype != np.float32 else [float(v) for v in a.ravel()]))
    return dict(files=files)


def build():
    quiet_logs()
    return {"features_case.json": features_case(), "forest_case.json": forest_case(),
            "objects_case.json": objects_case(), "tiff_case.json": tiff_case()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, data in build().items():
        (OUT / name).write_text(json.dumps(data, separators=(",", ":")) + "\n")
        print(f"wrote tests/fixtures/{name} ({(OUT / name).stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
