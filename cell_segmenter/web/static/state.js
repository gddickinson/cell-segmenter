// Session state shared by the pages: the loaded stack, labels with painted strokes, and the segmentation result.
// Mirrors the desktop app's model: labels are ordered (index = class), strokes live per frame, training uses the strokes
// of one frame, and the result mask is every label except the first ("segmentation > 0" in main_window.py).
import { api, isStatic } from './core.js';
import { call } from './worker_client.js';
import { readTiff } from './tiff.js';

export const PALETTE = ['#3d8bff', '#ffb02e', '#ff5fa2', '#3de0c8', '#b57bff', '#9be564', '#ff7a45'];

export const S = {
  info: null,
  stack: null,      // { name, frames: [TypedArray], H, W, T, dtype, umPerPx, source: 'demo'|'upload', serverId, truth? }
  labels: [],       // [{ name, color, masks: Map(frame -> Uint8Array) }]
  active: 0, frame: 0, brush: 5, opacity: 0.5,
  result: null,     // { pred: Map(frame -> Uint8Array), engine, note, trainFrame, importances? }
  history: [],      // undo snapshots { label, frame, mask }
  listeners: new Set(),
};

export const emit = () => S.listeners.forEach(f => f());

export async function getInfo() {
  if (!S.info) S.info = await api('/info');
  return S.info;
}

export const b64bytes = s => { const bin = atob(s), u = new Uint8Array(bin.length); for (let i = 0; i < bin.length; i++) u[i] = bin.charCodeAt(i); return u; };
export function unpackMask(s, n) {
  const bits = b64bytes(s), m = new Uint8Array(n);
  for (let i = 0; i < n; i++) m[i] = (bits[i >> 3] >> (7 - (i & 7))) & 1;
  return m;
}
export function packMask(m) {
  const bytes = new Uint8Array(Math.ceil(m.length / 8));
  for (let i = 0; i < m.length; i++) if (m[i]) bytes[i >> 3] |= 1 << (7 - (i & 7));
  let bin = '';
  for (let i = 0; i < bytes.length; i += 8192) bin += String.fromCharCode(...bytes.subarray(i, i + 8192));
  return btoa(bin);
}

function reset(stack) {
  S.stack = stack; S.frame = 0; S.result = null; S.history = []; S.active = 0;
  S.brush = S.info?.defaults.brush_size ?? 5;
  S.opacity = S.info?.defaults.overlay_opacity ?? 0.5;
}

export async function loadDemo() {
  const [info, d] = await Promise.all([getInfo(), api('/demo')]);
  const { frames: T, height: H, width: W } = d.summary, N = H * W;
  const all = new Uint16Array(b64bytes(d.frames).buffer), pred = b64bytes(d.python_pred);
  const frames = Array.from({ length: T }, (_, f) => all.slice(f * N, (f + 1) * N));
  reset({ name: 'demo (synthetic)', frames, H, W, T, dtype: d.summary.dtype, umPerPx: null, source: 'demo',
    serverId: 'demo', truth: d.truth.map(t => unpackMask(t, N)) });
  S.labels = d.labels.map(l => ({ name: l.name, color: l.color, masks: new Map([[l.frame, unpackMask(l.mask, N)]]) }));
  S.active = 1;
  await call('load', { frames, H, W });
  await call('forest', { forest: d.forest });
  S.result = { pred: new Map(frames.map((_, f) => [f, pred.slice(f * N, (f + 1) * N)])), engine: 'python',
    trainFrame: 0, importances: d.importances,
    note: `Python RandomForestModel (${info.defaults.n_estimators} trees, seed ${d.summary.forest_seed}) trained on the example strokes of frame 1 when the site was built` };
  emit();
}

export async function loadFile(file, onStatus = () => {}) {
  onStatus(`reading ${file.name}...`);
  const buf = await file.arrayBuffer();
  const t = await readTiff(buf);
  reset({ name: file.name, frames: t.frames, H: t.height, W: t.width, T: t.frames.length, dtype: t.dtype,
    umPerPx: t.umPerPx, source: 'upload', serverId: null });
  S.labels = [{ name: 'Background', color: PALETTE[0], masks: new Map() }, { name: 'Cell', color: PALETTE[1], masks: new Map() }];
  S.active = 1;
  await call('load', { frames: t.frames, H: t.height, W: t.width });
  if (!isStatic()) {   // the local app also gives the file to Python (tifffile + normalize_stack) for the Python engine
    onStatus('sending the stack to the local Python server...');
    const fd = new FormData();
    fd.append('file', file);
    try { S.stack.serverId = (await api('/stack', { body: fd })).id; } catch (e) { onStatus(`Python server could not read it: ${e.message}`); }
  }
  emit();
}

export function strokesOn(frame) {
  return S.labels.map(l => l.masks.get(frame) || new Uint8Array(S.stack.H * S.stack.W));
}

// Train on the strokes of the current frame and predict `frames`, with the Python server or the JS port.
export async function segment(frames, { engine, n_estimators, max_depth, seed }, onProgress) {
  const st = S.stack, trainFrame = S.frame, masks = strokesOn(trainFrame);
  if (engine === 'python') {
    if (!st.serverId) throw new Error('The Python server has not got this stack (upload failed); use the browser engine');
    onProgress({ stage: 'Python: training RandomForestModel and predicting', done: 0, total: 1 });
    const r = await api('/segment', { body: { stack: st.serverId, train_frame: trainFrame, frames, seed,
      labels: S.labels.map((l, i) => ({ name: l.name, mask: packMask(masks[i]) })) } });
    const pred = new Map();
    for (const f of frames) pred.set(f, b64bytes(r.pred[String(f)]));
    S.result = { pred, engine: 'python', trainFrame, importances: r.importances,
      note: `Python RandomForestModel trained on ${r.n_train.toLocaleString()} stroke pixels of frame ${trainFrame + 1} (${r.train_s.toFixed(1)} s), predicted in ${r.predict_s.toFixed(1)} s` };
  } else {
    const t0 = performance.now();
    const tr = await call('train', { frame: trainFrame, masks, n_estimators, max_depth, seed }, onProgress);
    const t1 = performance.now();
    const out = await call('predict', { frames }, onProgress);
    const pred = new Map();
    for (const f of frames) pred.set(f, out[f]);
    S.result = { pred, engine: 'js', trainFrame, forest: tr,
      note: `Browser (JS port): ${tr.trees} trees, ${tr.nodes.toLocaleString()} nodes, trained on ${tr.n_train.toLocaleString()} stroke pixels of frame ${trainFrame + 1} in ${((t1 - t0) / 1000).toFixed(1)} s, predicted in ${((performance.now() - t1) / 1000).toFixed(1)} s` };
  }
  emit();
}

// Predict more frames with the model already trained (JS port, or the Python forest exported for the demo).
export async function predictMore(frames, onProgress) {
  const out = await call('predict', { frames }, onProgress);
  for (const f of frames) S.result.pred.set(f, out[f]);
  emit();
}
