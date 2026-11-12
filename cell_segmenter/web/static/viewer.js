// Canvas view of the current frame with stroke and result overlays, and the paint/erase brush.
// The brush is a port of utils/brush.py::stamp_brush (disk of radius size/2, clipped at the image edges).
import { S, emit } from './state.js';

const hex = c => [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16));

// Display range of a raw frame (min-max, as the GUI shows the normalised stack).
const rangeCache = new WeakMap();
function range(fr) {
  if (!rangeCache.has(fr)) {
    let lo = Infinity, hi = -Infinity;
    for (const v of fr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    rangeCache.set(fr, [lo, hi]);
  }
  return rangeCache.get(fr);
}

// Render frame f into an ImageData: gray image, result overlay (labels after the first, at S.opacity), strokes,
// known-mask outline (demo). opts: { strokes, result, truth }.
export function compose(f, opts) {
  const { H, W, frames } = S.stack, fr = frames[f], [lo, hi] = range(fr), k = hi > lo ? 255 / (hi - lo) : 0;
  const img = new ImageData(W, H), d = img.data;
  for (let i = 0; i < H * W; i++) { const g = (fr[i] - lo) * k; d[4 * i] = d[4 * i + 1] = d[4 * i + 2] = g; d[4 * i + 3] = 255; }
  const pred = opts.result && S.result?.pred.get(f);
  if (pred) {
    const cols = S.labels.map(l => hex(l.color)), a = S.opacity;
    for (let i = 0; i < H * W; i++) {
      const c = pred[i];
      if (c > 0 && cols[c]) for (let j = 0; j < 3; j++) d[4 * i + j] = d[4 * i + j] * (1 - a) + cols[c][j] * a;
    }
  }
  if (opts.strokes) {
    S.labels.forEach(l => {
      const m = l.masks.get(f);
      if (!m) return;
      const c = hex(l.color);
      for (let i = 0; i < H * W; i++) if (m[i]) for (let j = 0; j < 3; j++) d[4 * i + j] = d[4 * i + j] * 0.25 + c[j] * 0.75;
    });
  }
  const truth = opts.truth && S.stack.truth?.[f];
  if (truth) {
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const i = r * W + c;
      if (!truth[i]) continue;
      const edge = r === 0 || c === 0 || r === H - 1 || c === W - 1 || !truth[i - 1] || !truth[i + 1] || !truth[i - W] || !truth[i + W];
      if (edge) { d[4 * i] = 255; d[4 * i + 1] = 255; d[4 * i + 2] = 255; }
    }
  }
  return img;
}

export function stamp(mask, H, W, row, col, size, value) {
  const r2 = (size / 2) ** 2;
  for (let dr = -size; dr <= size; dr++) for (let dc = -size; dc <= size; dc++) {
    const r = row + dr, c = col + dc;
    if (r < 0 || c < 0 || r >= H || c >= W || dr * dr + dc * dc > r2) continue;
    mask[r * W + c] = value;
  }
}

// Attach painting to a canvas. getMode() -> 'paint' | 'erase' | 'view'; redraw() repaints the canvas.
export function attachBrush(canvas, getMode, redraw) {
  let down = false, last = null;
  const pos = e => {
    const r = canvas.getBoundingClientRect();
    return [Math.floor((e.clientY - r.top) * S.stack.H / r.height), Math.floor((e.clientX - r.left) * S.stack.W / r.width)];
  };
  const paintTo = (p) => {
    const lab = S.labels[S.active];
    if (!lab) return;
    const { H, W } = S.stack;
    if (!lab.masks.has(S.frame)) lab.masks.set(S.frame, new Uint8Array(H * W));
    const m = lab.masks.get(S.frame), value = getMode() === 'erase' ? 0 : 1;
    const [r0, c0] = last || p, n = Math.max(Math.abs(p[0] - r0), Math.abs(p[1] - c0), 1);
    for (let s = 0; s <= n; s++) stamp(m, H, W, Math.round(r0 + (p[0] - r0) * s / n), Math.round(c0 + (p[1] - c0) * s / n), S.brush, value);
    last = p;
    redraw();
  };
  canvas.addEventListener('pointerdown', e => {
    if (getMode() === 'view' || !S.labels[S.active]) return;
    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);
    const lab = S.labels[S.active], prev = lab.masks.get(S.frame);
    S.history.push({ label: lab, frame: S.frame, mask: prev ? prev.slice() : null });
    if (S.history.length > 40) S.history.shift();
    down = true; last = null;
    paintTo(pos(e));
  });
  canvas.addEventListener('pointermove', e => { if (down) paintTo(pos(e)); });
  const up = () => { if (down) { down = false; last = null; emit(); } };
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
}

export function undo() {
  const h = S.history.pop();
  if (!h) return false;
  if (h.mask) h.label.masks.set(h.frame, h.mask); else h.label.masks.delete(h.frame);
  emit();
  return true;
}

export function countStrokes(label, f) {
  const m = label.masks.get(f);
  let n = 0;
  if (m) for (const v of m) n += v;
  return n;
}
