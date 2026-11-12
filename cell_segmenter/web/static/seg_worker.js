// Web Worker: feature extraction, forest training and prediction off the main thread (the desktop app's rule: never
// compute on the UI thread). Messages { id, type, ... } -> replies { id, ok, result } / { id, error } / { id, progress }.
//   load    { frames: [TypedArray raw], H, W }       normalise each frame as normalize_stack (per-frame min-max)
//   train   { training: [{ frame, masks: [Uint8Array per label] }], indices (feature rows, null = all), n_estimators,
//             max_depth, seed }
//           -> { n_train, classes, trees, nodes }  (several frames = "use training data from all frames")
//   forest  { forest: exported JSON }                 use a forest trained by Python
//   predict { frames: [i] }                           -> { [i]: Uint8Array label index per pixel }
//   feature { frame, k }                              -> Float32Array feature map k
import { extractFeatures, normalize } from './features.js';
import { forestFromJson, predictForest, trainForest } from './forest.js';

let norm = [], H = 0, W = 0, forest = null;
const cache = new Map();  // frame -> [36 Float32Array], keep the 2 most recent (memory)

function features(f, report) {
  if (cache.has(f)) { const v = cache.get(f); cache.delete(f); cache.set(f, v); return v; }
  const F = extractFeatures(norm[f], H, W, (i, name) => report({ stage: `features: ${name}`, done: i, total: 36 }))
    .map(m => Float32Array.from(m));
  cache.set(f, F);
  while (cache.size > 2) cache.delete(cache.keys().next().value);
  return F;
}

const pick = (F, indices) => (indices ? indices.map(k => F[k]) : F);

const handlers = {
  load({ frames, H: h, W: w }) {
    H = h; W = w; norm = frames.map(fr => normalize(fr)); cache.clear(); forest = null;
    return { frames: norm.length };
  },
  train({ training, indices = null, n_estimators, max_depth, seed }, report) {
    const X = [], y = [];
    for (const { frame, masks } of training) {
      const F = pick(features(frame, report), indices);
      if (!X.length) F.forEach(() => X.push([]));
      masks.forEach((m, label) => {
        for (let i = 0; i < H * W; i++) if (m[i]) { y.push(label); F.forEach((f, k) => X[k].push(f[i])); }
      });
    }
    if (new Set(y).size < 2) throw new Error('Paint strokes for at least two labels first');
    forest = trainForest(X.map(a => Float32Array.from(a)), y, { n_estimators, max_depth, seed,
      onTree: (t, n) => report({ stage: 'training trees', done: t, total: n }) });
    forest.feature_indices = indices;
    return { n_train: y.length, classes: forest.classes, trees: forest.trees.length,
      nodes: forest.trees.reduce((s, t) => s + t.left.length, 0) };
  },
  forest({ forest: j }) { forest = forestFromJson(j); return { trees: forest.trees.length }; },
  predict({ frames }, report) {
    if (!forest) throw new Error('Train a model first');
    const out = {};
    frames.forEach((f, i) => {
      report({ stage: `frame ${f + 1}: features`, done: i, total: frames.length });
      out[f] = predictForest(forest, pick(features(f, () => {}), forest.feature_indices), H * W);
    });
    return out;
  },
  feature({ frame, k }, report) { return features(frame, report)[k]; },
};

self.onmessage = e => {
  const { id, type, ...args } = e.data;
  try {
    const result = handlers[type](args, progress => self.postMessage({ id, progress }));
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({ id, error: err.message || String(err) });
  }
};
