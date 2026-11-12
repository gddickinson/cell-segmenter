// Random forest for pixel classification, the model behind RandomForestModel (random_forest.py: scikit-learn
// RandomForestClassifier, n_estimators=100, max_depth=10, defaults otherwise).
//  - predictForest(): exact port of sklearn's predict for an exported forest (features cast to float32, go left when
//    x <= threshold, average the per-tree class fractions, take the first maximum). Matches sklearn label for label.
//  - trainForest(): the same algorithm in JS (bootstrap samples, sqrt(n_features) candidate features per split, Gini,
//    thresholds at midpoints, max_depth); random draws differ from sklearn, so a JS-trained forest is a different
//    forest of the same kind, not a copy of the Python one (agreement is measured in forest.test.mjs).
// A forest is { classes: [label index...], n_features, trees: [{ feature, threshold, left, right, value }] } where
// value holds n_classes fractions per node and left[i] === -1 marks a leaf.
import { rng as makeRng } from './core.js';

export const RF_DEFAULTS = { n_estimators: 100, max_depth: 10 };  // config.RF_N_ESTIMATORS, config.RF_MAX_DEPTH

// Per-pixel class probabilities. X: array of n_features Float64Array/Float32Array maps (feature-major), N pixels.
export function predictProba(forest, X, N) {
  const K = forest.classes.length, F = X.length, proba = new Float64Array(N * K);
  const Xf = X.map(f => (f instanceof Float32Array ? f : Float32Array.from(f)));
  for (const t of forest.trees) {
    const { feature, threshold, left, right, value } = t;
    for (let i = 0; i < N; i++) {
      let n = 0;
      while (left[n] !== -1) n = Xf[feature[n]][i] <= threshold[n] ? left[n] : right[n];
      const o = i * K, v = n * K;
      for (let k = 0; k < K; k++) proba[o + k] += value[v + k];
    }
  }
  const T = forest.trees.length;
  for (let i = 0; i < proba.length; i++) proba[i] /= T;
  if (F !== forest.n_features) throw new Error(`forest expects ${forest.n_features} features, got ${F}`);
  return proba;
}

// Predicted label index per pixel (forest.classes values), as RandomForestModel.predict.
export function predictForest(forest, X, N) {
  const K = forest.classes.length, proba = predictProba(forest, X, N), out = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    let best = 0;
    for (let k = 1; k < K; k++) if (proba[i * K + k] > proba[i * K + best]) best = k;
    out[i] = forest.classes[best];
  }
  return out;
}

// --- training ---------------------------------------------------------------------------------------------------
// X: n_features arrays of length n (sample-major per feature), y: Int array of label indices.
export function trainForest(X, y, { n_estimators = RF_DEFAULTS.n_estimators, max_depth = RF_DEFAULTS.max_depth, seed = 1,
  onTree = () => {} } = {}) {
  const F = X.length, n = y.length;
  const Xf = X.map(f => (f instanceof Float32Array ? f : Float32Array.from(f)));
  const classes = [...new Set(y)].sort((a, b) => a - b), K = classes.length;
  const yi = Int32Array.from(y, v => classes.indexOf(v));
  const mtry = Math.max(1, Math.floor(Math.sqrt(F)));
  const rnd = makeRng(seed);
  const trees = [];
  for (let t = 0; t < n_estimators; t++) {
    const w = new Float64Array(n);  // bootstrap counts as sample weights (as sklearn)
    for (let i = 0; i < n; i++) w[Math.floor(rnd() * n)]++;
    const idx = [];
    for (let i = 0; i < n; i++) if (w[i] > 0) idx.push(i);
    trees.push(growTree(Xf, yi, w, Int32Array.from(idx), K, mtry, max_depth, rnd));
    onTree(t + 1, n_estimators);
  }
  return { classes, n_features: F, trees };
}

function growTree(X, y, w, idx, K, mtry, maxDepth, rnd) {
  const feature = [], threshold = [], left = [], right = [], value = [];
  const F = X.length, order = new Int32Array(F);
  const node = (ids, depth) => {
    const id = feature.length;
    feature.push(-2); threshold.push(-2); left.push(-1); right.push(-1);
    const counts = new Float64Array(K);
    let tot = 0;
    for (const i of ids) { counts[y[i]] += w[i]; tot += w[i]; }
    for (let k = 0; k < K; k++) value.push(counts[k] / tot);
    let gini = 1;
    for (let k = 0; k < K; k++) gini -= (counts[k] / tot) ** 2;
    if (depth >= maxDepth || ids.length < 2 || gini <= 0) return id;
    // candidate features: random permutation, keep drawing until mtry non-constant ones were tried (as sklearn)
    for (let f = 0; f < F; f++) order[f] = f;
    let best = null, tried = 0;
    for (let f = 0; f < F && tried < mtry; f++) {
      const j = f + Math.floor(rnd() * (F - f));
      const tmp = order[f]; order[f] = order[j]; order[j] = tmp;
      const col = X[order[f]];
      const sorted = Int32Array.from(ids).sort((a, b) => col[a] - col[b]);
      if (col[sorted[sorted.length - 1]] <= col[sorted[0]] + 1e-7) continue;
      tried++;
      const lc = new Float64Array(K);
      let lw = 0;
      for (let p = 0; p < sorted.length - 1; p++) {
        const i = sorted[p];
        lc[y[i]] += w[i]; lw += w[i];
        const a = col[i], b = col[sorted[p + 1]];
        if (b <= a + 1e-7) continue;
        const rw = tot - lw;
        let gl = 1, gr = 1;
        for (let k = 0; k < K; k++) { gl -= (lc[k] / lw) ** 2; gr -= ((counts[k] - lc[k]) / rw) ** 2; }
        const imp = (lw * gl + rw * gr) / tot;
        if (!best || imp < best.imp - 1e-12) best = { imp, f: order[f], thr: a / 2 + b / 2 };
      }
    }
    if (!best) return id;
    const col = X[best.f], L = [], R = [];
    for (const i of ids) (col[i] <= best.thr ? L : R).push(i);
    if (!L.length || !R.length) return id;
    feature[id] = best.f; threshold[id] = best.thr;
    left[id] = node(L, depth + 1);
    right[id] = node(R, depth + 1);
    return id;
  };
  node(Array.from(idx), 0);
  return { feature: Int32Array.from(feature), threshold: Float64Array.from(threshold), left: Int32Array.from(left),
    right: Int32Array.from(right), value: Float64Array.from(value) };
}

// Plain-array forest from the JSON exported by the Python side (cell_segmenter/web/segment_service.py).
export function forestFromJson(j) {
  return { classes: j.classes, n_features: j.n_features, trees: j.trees.map(t => ({
    feature: Int32Array.from(t.feature), threshold: Float64Array.from(t.threshold), left: Int32Array.from(t.left),
    right: Int32Array.from(t.right), value: Float64Array.from(t.value) })) };
}
