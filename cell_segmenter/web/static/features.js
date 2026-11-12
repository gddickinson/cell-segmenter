// JS port of cell_segmenter/models/features.py::FeatureExtractor.extract_features: the same 36 feature maps, in the
// same order, with the skimage / scipy boundary conventions. Input: a [0, 1] Float64Array frame (normalised as the GUI
// does on load). Tested against the Python output in features.test.mjs (tests/fixtures/features_case.json).
import {
  convolve2d, edgeAxis, edgeMagnitude, gaussianFilter, ndiSobel,
  PREWITT_SMOOTH, SCHARR_SMOOTH, SOBEL_SMOOTH,
} from './filters.js';

// Parameters of FeatureExtractor.__init__ (features.py)
export const PARAMS = {
  sigmas: [1, 2, 4], hessian_sigmas: [1, 2, 4], gabor_frequencies: [0.1, 0.25, 0.5],
  gabor_orientations: [0, 45, 90, 135], entropy_radius: 5, structure_sigma: 1,
};

// Feature groups of the training panel (models/features.py::FEATURE_GROUPS) -> rows of extractFeatures().
export const FEATURE_GROUPS = {
  'Original Intensity': [0],
  'Gaussian Features': [1, 2, 3, 4, 5, 6, 7, 8, 9],
  'Edge Features': [10, 11, 12, 13],
  'LBP Features': [14],
  'Hessian Features': [15, 16, 17, 18, 19, 20],
  'Gabor Features': [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
  'Entropy Features': [33],
  'Structure Tensor': [34, 35],
};

// Rows for the selected group names; null = all (as feature_indices() in Python).
export function featureIndices(names) {
  if (!names || !names.length) return null;
  const idx = [...new Set(names.flatMap(n => { if (!FEATURE_GROUPS[n]) throw new Error(`Unknown feature group ${n}`); return FEATURE_GROUPS[n]; }))].sort((a, b) => a - b);
  return idx.length === 36 ? null : idx;
}

export const FEATURE_NAMES = (() => {
  const n = ['Intensity (normalised)'];
  for (const s of PARAMS.sigmas) n.push(`Gaussian σ=${s}`, `Gradient magnitude σ=${s}`, `Laplacian σ=${s}`);
  n.push('Sobel edges', 'Scharr edges', 'Roberts edges', 'Prewitt edges', 'LBP (uniform, P=8, R=1)');
  for (const s of PARAMS.hessian_sigmas) n.push(`Hessian determinant σ=${s}`, `Hessian trace σ=${s}`);
  for (const f of PARAMS.gabor_frequencies) for (const t of PARAMS.gabor_orientations) n.push(`Gabor f=${f} θ=${t}°`);
  n.push(`Local entropy (${PARAMS.entropy_radius}×${PARAMS.entropy_radius})`, 'Structure-tensor orientation', 'Structure-tensor coherency');
  return n;
})();

export function normalize(img) {
  let lo = Infinity, hi = -Infinity;
  for (const v of img) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const out = new Float64Array(img.length);
  if (lo === hi) return out;
  for (let i = 0; i < img.length; i++) out[i] = (img[i] - lo) / (hi - lo);
  return out;
}

const map2 = (a, b, f) => { const o = new Float64Array(a.length); for (let i = 0; i < a.length; i++) o[i] = f(a[i], b[i]); return o; };

// skimage.filters.gabor_kernel(frequency, theta, bandwidth=1, n_stds=3): real part, (2*y0+1) x (2*x0+1)
export function gaborKernel(frequency, theta) {
  const b = 1, pref = 1 / Math.PI * Math.sqrt(Math.log(2) / 2) * (2 ** b + 1) / (2 ** b - 1);
  const sx = pref / frequency, sy = sx, ct = Math.cos(theta), st = Math.sin(theta);
  const x0 = Math.ceil(Math.max(Math.abs(3 * sx * ct), Math.abs(3 * sy * st), 1));
  const y0 = Math.ceil(Math.max(Math.abs(3 * sy * ct), Math.abs(3 * sx * st), 1));
  const kh = 2 * y0 + 1, kw = 2 * x0 + 1, k = new Float64Array(kh * kw), norm = 1 / (2 * Math.PI * sx * sy);
  for (let a = 0; a < kh; a++) for (let c = 0; c < kw; c++) {
    const y = a - y0, x = c - x0, rotx = x * ct + y * st, roty = -x * st + y * ct;
    k[a * kw + c] = Math.exp(-0.5 * (rotx ** 2 / sx ** 2 + roty ** 2 / sy ** 2)) * Math.cos(2 * Math.PI * frequency * rotx) * norm;
  }
  return { k, kh, kw };
}

// skimage.feature.local_binary_pattern(image, P=8, R=1, method='uniform') (note: skimage counts 0/1 transitions
// without the wrap-around pair, and samples neighbours by bilinear interpolation with zero outside the image).
export function lbpUniform(img, H, W, P = 8, R = 1) {
  const rp = [], cp = [];
  for (let i = 0; i < P; i++) {
    rp.push(Math.round(-R * Math.sin(2 * Math.PI * i / P) * 1e5) / 1e5);
    cp.push(Math.round(R * Math.cos(2 * Math.PI * i / P) * 1e5) / 1e5);
  }
  const px = (r, c) => (r < 0 || c < 0 || r >= H || c >= W ? 0 : img[r * W + c]);
  const out = new Float64Array(H * W), s = new Int8Array(P);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
    const centre = img[r * W + c];
    for (let i = 0; i < P; i++) {
      const rr = r + rp[i], cc = c + cp[i];
      const minr = Math.floor(rr), minc = Math.floor(cc), maxr = Math.ceil(rr), maxc = Math.ceil(cc);
      const dr = rr - minr, dc = cc - minc;
      const top = (1 - dc) * px(minr, minc) + dc * px(minr, maxc);
      const bottom = (1 - dc) * px(maxr, minc) + dc * px(maxr, maxc);
      const t = (1 - dr) * top + dr * bottom;
      s[i] = t - centre >= 0 ? 1 : 0;
    }
    let changes = 0;
    for (let i = 0; i < P - 1; i++) changes += s[i] !== s[i + 1] ? 1 : 0;
    let lbp = 0;
    if (changes <= 2) for (let i = 0; i < P; i++) lbp += s[i]; else lbp = P + 1;
    out[r * W + c] = lbp;
  }
  return out;
}

// numpy.rint: round half to even
export const rint = x => { const f = Math.floor(x), d = x - f; return d > 0.5 ? f + 1 : d < 0.5 ? f : (f % 2 === 0 ? f : f + 1); };

// skimage.filters.rank.entropy(img_as_ubyte(image), np.ones((n, n))): base-2 entropy of the 8-bit histogram in an
// n x n window centred on each pixel, counting only pixels inside the image.
export function localEntropy(img, H, W, n = 5) {
  const u8 = new Uint8Array(H * W);
  for (let i = 0; i < u8.length; i++) u8[i] = Math.min(255, Math.max(0, rint(img[i] * 255)));
  const out = new Float64Array(H * W), hist = new Int32Array(256), c = n >> 1, LN2 = Math.log(2);
  for (let r = 0; r < H; r++) for (let x = 0; x < W; x++) {
    hist.fill(0);
    let pop = 0;
    const r0 = Math.max(0, r - c), r1 = Math.min(H - 1, r - c + n - 1), x0 = Math.max(0, x - c), x1 = Math.min(W - 1, x - c + n - 1);
    for (let y = r0; y <= r1; y++) for (let q = x0; q <= x1; q++) { hist[u8[y * W + q]]++; pop++; }
    let e = 0;
    for (let b = 0; b < 256; b++) if (hist[b]) { const p = hist[b] / pop; e -= p * Math.log(p) / LN2; }
    out[r * W + x] = e;
  }
  return out;
}

// All 36 feature maps for one [0, 1] frame. onStep(i, name) reports progress.
export function extractFeatures(image, H, W, onStep = () => {}) {
  const F = [], push = m => { onStep(F.length, FEATURE_NAMES[F.length]); F.push(m); };
  push(normalize(image));
  for (const s of PARAMS.sigmas) {
    const g = gaussianFilter(image, H, W, s, { mode: 'nearest' });
    push(g);
    const gy = edgeAxis(g, H, W, SOBEL_SMOOTH, 0), gx = edgeAxis(g, H, W, SOBEL_SMOOTH, 1);
    push(map2(gy, gx, (a, b) => Math.sqrt(a * a + b * b)));
    push(convolve2d(g, H, W, [0, -1, 0, -1, 4, -1, 0, -1, 0], 3, 3, 'reflect'));
  }
  push(edgeMagnitude(image, H, W, SOBEL_SMOOTH));
  push(edgeMagnitude(image, H, W, SCHARR_SMOOTH));
  const pd = convolve2d(image, H, W, [1, 0, 0, -1], 2, 2, 'reflect'), nd = convolve2d(image, H, W, [0, 1, -1, 0], 2, 2, 'reflect');
  push(map2(pd, nd, (a, b) => Math.sqrt(a * a + b * b) / Math.SQRT2));
  push(edgeMagnitude(image, H, W, PREWITT_SMOOTH));
  push(lbpUniform(image, H, W));
  for (const s of PARAMS.hessian_sigmas) {
    const hxx = gaussianFilter(image, H, W, s, { order: [2, 0] });
    const hyy = gaussianFilter(image, H, W, s, { order: [0, 2] });
    const hxy = gaussianFilter(image, H, W, s, { order: [1, 1] });
    const det = new Float64Array(H * W), tr = new Float64Array(H * W);
    for (let i = 0; i < det.length; i++) { det[i] = hxx[i] * hyy[i] - hxy[i] ** 2; tr[i] = hxx[i] + hyy[i]; }
    push(det); push(tr);
  }
  for (const f of PARAMS.gabor_frequencies) for (const t of PARAMS.gabor_orientations) {
    const { k, kh, kw } = gaborKernel(f, t * Math.PI / 180);
    push(convolve2d(image, H, W, k, kh, kw, 'reflect'));
  }
  push(localEntropy(image, H, W, PARAMS.entropy_radius));
  // skimage.feature.structure_tensor(sigma=1, mode='constant', order='rc') -> Arr, Arc, Acc
  const d0 = ndiSobel(image, H, W, 0, 'constant'), d1 = ndiSobel(image, H, W, 1, 'constant');
  const sm = a => gaussianFilter(a, H, W, PARAMS.structure_sigma, { mode: 'constant' });
  const axx = sm(map2(d0, d0, (a, b) => a * b)), axy = sm(map2(d0, d1, (a, b) => a * b)), ayy = sm(map2(d1, d1, (a, b) => a * b));
  const ori = new Float64Array(H * W), coh = new Float64Array(H * W);
  for (let i = 0; i < ori.length; i++) {
    ori[i] = Math.atan2(2 * axy[i], ayy[i] - axx[i]) / 2;
    coh[i] = Math.sqrt((axx[i] - ayy[i]) ** 2 + 4 * axy[i] ** 2) / (axx[i] + ayy[i] + 1e-10);
  }
  push(ori); push(coh);
  return F;
}

