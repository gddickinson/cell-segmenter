// JS port of cell_segmenter/utils/object_stats.py: 8-connected components numbered in raster order (as
// skimage.measure.label) and per-object area, centroid, bounding box, equivalent diameter, eccentricity and mean
// intensity (as skimage.measure.regionprops). Tested against Python in objects.test.mjs.

export const COLUMNS = ['label', 'area_px', 'centroid_row', 'centroid_col', 'bbox_min_row', 'bbox_min_col', 'bbox_max_row',
  'bbox_max_col', 'equivalent_diameter_px', 'eccentricity', 'mean_intensity'];

// Label image (Int32Array, 0 = background) and the number of objects.
export function labelObjects(mask, H, W) {
  const lab = new Int32Array(H * W), stack = new Int32Array(H * W);
  let n = 0;
  for (let s = 0; s < H * W; s++) {
    if (!mask[s] || lab[s]) continue;
    n++;
    let top = 0;
    stack[top++] = s; lab[s] = n;
    while (top) {
      const p = stack[--top], r = (p / W) | 0, c = p - r * W;
      for (let dr = -1; dr <= 1; dr++) for (let dc = -1; dc <= 1; dc++) {
        const rr = r + dr, cc = c + dc;
        if (rr < 0 || cc < 0 || rr >= H || cc >= W) continue;
        const q = rr * W + cc;
        if (mask[q] && !lab[q]) { lab[q] = n; stack[top++] = q; }
      }
    }
  }
  return { labels: lab, n };
}

// Same result shape as object_stats() in Python.
export function objectStats(mask, H, W, { image = null, umPerPx = null, minAreaPx = 0 } = {}) {
  const { labels, n } = labelObjects(mask, H, W);
  const acc = Array.from({ length: n + 1 }, () => ({ m: 0, sr: 0, sc: 0, r0: H, c0: W, r1: -1, c1: -1, si: 0, pix: [] }));
  for (let i = 0; i < H * W; i++) {
    const l = labels[i];
    if (!l) continue;
    const a = acc[l], r = (i / W) | 0, c = i - r * W;
    a.m++; a.sr += r; a.sc += c; a.pix.push(i);
    if (r < a.r0) a.r0 = r; if (r > a.r1) a.r1 = r; if (c < a.c0) a.c0 = c; if (c > a.c1) a.c1 = c;
    if (image) a.si += image[i];
  }
  const rows = [];
  for (let l = 1; l <= n; l++) {
    const a = acc[l];
    if (a.m < minAreaPx) continue;
    const cr = a.sr / a.m, cc = a.sc / a.m;
    let mu20 = 0, mu02 = 0, mu11 = 0;
    for (const i of a.pix) { const r = ((i / W) | 0) - cr, c = (i % W) - cc; mu20 += r * r; mu02 += c * c; mu11 += r * c; }
    // inertia tensor [[mu02, -mu11], [-mu11, mu20]] / m; eigenvalues descending, clipped at 0
    const t0 = mu02 / a.m, t1 = -mu11 / a.m, t2 = mu20 / a.m;
    const half = (t0 + t2) / 2, d = Math.sqrt(((t0 - t2) / 2) ** 2 + t1 * t1);
    const l1 = Math.max(half + d, 0), l2 = Math.max(half - d, 0);
    const row = {
      label: l, area_px: a.m, centroid_row: cr, centroid_col: cc, bbox_min_row: a.r0, bbox_min_col: a.c0,
      bbox_max_row: a.r1 + 1, bbox_max_col: a.c1 + 1, equivalent_diameter_px: Math.sqrt(4 * a.m / Math.PI),
      eccentricity: l1 === 0 ? 0 : Math.sqrt(1 - l2 / l1), mean_intensity: image ? a.si / a.m : null,
    };
    if (umPerPx) { row.area_um2 = row.area_px * umPerPx ** 2; row.equivalent_diameter_um = row.equivalent_diameter_px * umPerPx; }
    rows.push(row);
  }
  const areas = rows.map(r => r.area_px).sort((x, y) => x - y);
  let on = 0;
  for (let i = 0; i < H * W; i++) if (mask[i]) on++;
  const med = areas.length ? (areas.length % 2 ? areas[(areas.length - 1) / 2] : (areas[areas.length / 2 - 1] + areas[areas.length / 2]) / 2) : null;
  return {
    units: umPerPx ? 'um' : 'px (no pixel size given)', um_per_px: umPerPx || null, objects: rows, labels,
    summary: { n_objects: rows.length, mask_fraction: on / (H * W),
      mean_area_px: areas.length ? areas.reduce((s, v) => s + v, 0) / areas.length : null, median_area_px: med },
  };
}

// CSV of the object table (pixel units always; µm columns when a pixel size was given).
export function objectsCsv(stats, frame) {
  const cols = ['frame', ...COLUMNS, ...(stats.um_per_px ? ['area_um2', 'equivalent_diameter_um'] : [])];
  const fmt = v => (v === null || v === undefined ? '' : Number.isInteger(v) ? String(v) : (+v).toPrecision(8));
  return [cols.join(','), ...stats.objects.map(r => cols.map(c => fmt(c === 'frame' ? frame : r[c])).join(','))].join('\n') + '\n';
}
