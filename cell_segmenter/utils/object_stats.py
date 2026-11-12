"""Per-object statistics for a segmentation mask (Qt-free; mirrored by the web app's objects.js).

Objects are the 8-connected components of a boolean mask (skimage.measure.label, connectivity 2), numbered in raster
order. Pixel coordinates are pixel centres with the origin at the top-left pixel (row, col), as numpy/skimage index
them. Physical sizes need the recording's pixel size: pass um_per_px, otherwise only pixel units are reported and
'units' says so.
"""
from typing import Optional

import numpy as np
from skimage import measure

COLUMNS = ["label", "area_px", "centroid_row", "centroid_col", "bbox_min_row", "bbox_min_col", "bbox_max_row",
           "bbox_max_col", "equivalent_diameter_px", "eccentricity", "mean_intensity"]


def label_objects(mask: np.ndarray) -> np.ndarray:
    """Integer label image (0 = background) of the 8-connected components of mask."""
    return measure.label(np.asarray(mask, bool), connectivity=2)


def object_stats(mask: np.ndarray, image: Optional[np.ndarray] = None, um_per_px: Optional[float] = None,
                 min_area_px: int = 0) -> dict:
    """Return {'units', 'um_per_px', 'objects': [row dicts], 'summary'} for the objects in mask.

    bbox_max_* are exclusive (skimage convention). mean_intensity is over the object's pixels of `image` (None if no
    image). Objects smaller than min_area_px are dropped. With um_per_px, area_um2 and equivalent_diameter_um are added.
    """
    labels = label_objects(mask)
    props = measure.regionprops(labels, intensity_image=None if image is None else np.asarray(image, float))
    rows = []
    for p in props:
        if p.area < min_area_px:
            continue
        r = dict(label=int(p.label), area_px=int(p.area), centroid_row=float(p.centroid[0]),
                 centroid_col=float(p.centroid[1]), bbox_min_row=int(p.bbox[0]), bbox_min_col=int(p.bbox[1]),
                 bbox_max_row=int(p.bbox[2]), bbox_max_col=int(p.bbox[3]),
                 equivalent_diameter_px=float(p.equivalent_diameter_area), eccentricity=float(p.eccentricity),
                 mean_intensity=None if image is None else float(p.intensity_mean))
        if um_per_px:
            r["area_um2"] = r["area_px"] * um_per_px ** 2
            r["equivalent_diameter_um"] = r["equivalent_diameter_px"] * um_per_px
        rows.append(r)
    areas = np.array([r["area_px"] for r in rows], float)
    summary = dict(n_objects=len(rows), mask_fraction=float(np.mean(np.asarray(mask, bool))),
                   mean_area_px=float(areas.mean()) if len(areas) else None,
                   median_area_px=float(np.median(areas)) if len(areas) else None)
    return dict(units="um" if um_per_px else "px (no pixel size given)", um_per_px=um_per_px, objects=rows,
                summary=summary)
