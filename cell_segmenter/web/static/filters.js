// Image filters with scipy.ndimage semantics (pure JS, Float64Array images in row-major order, H rows x W cols).
// Boundary modes follow scipy: 'reflect' (d c b a | a b c d), 'nearest' (a a a | a b c d), 'constant' (cval).
// Tested against Python in features.test.mjs.

// Index into [0, n) for an out-of-range i, or -1 for 'constant'.
export function boundaryIndex(i, n, mode) {
  if (i >= 0 && i < n) return i;
  if (mode === 'constant') return -1;
  if (mode === 'nearest') return i < 0 ? 0 : n - 1;
  // reflect: period 2n, half-sample symmetric
  const p = 2 * n;
  let k = ((i % p) + p) % p;
  return k < n ? k : p - 1 - k;
}

// scipy.ndimage.correlate1d along axis (0 = rows, 1 = cols): out[i] = sum_k w[k] * in[i + k - floor(len/2)].
export function correlate1d(img, H, W, w, axis, mode = 'reflect', cval = 0) {
  const out = new Float64Array(H * W), n = w.length, c = n >> 1;
  const len = axis === 0 ? H : W;
  const idx = new Int32Array(len + n);
  for (let i = 0; i < len + n; i++) idx[i] = boundaryIndex(i - c, len, mode);
  if (axis === 1) {
    for (let r = 0; r < H; r++) {
      const row = r * W;
      for (let x = 0; x < W; x++) {
        let s = 0;
        for (let k = 0; k < n; k++) { const j = idx[x + k]; s += w[k] * (j < 0 ? cval : img[row + j]); }
        out[row + x] = s;
      }
    }
  } else {
    for (let r = 0; r < H; r++) {
      for (let x = 0; x < W; x++) {
        let s = 0;
        for (let k = 0; k < n; k++) { const j = idx[r + k]; s += w[k] * (j < 0 ? cval : img[j * W + x]); }
        out[r * W + x] = s;
      }
    }
  }
  return out;
}

// Pad an image by (py, px) on each side with a boundary mode (constant -> cval).
function pad(img, H, W, py, px, mode, cval) {
  const PH = H + 2 * py, PW = W + 2 * px, out = new Float64Array(PH * PW);
  const ci = new Int32Array(PW);
  for (let x = 0; x < PW; x++) ci[x] = boundaryIndex(x - px, W, mode);
  for (let y = 0; y < PH; y++) {
    const r = boundaryIndex(y - py, H, mode);
    for (let x = 0; x < PW; x++) out[y * PW + x] = r < 0 || ci[x] < 0 ? cval : img[r * W + ci[x]];
  }
  return { p: out, PW };
}

// scipy.ndimage.convolve with a 2-D kernel k (kh x kw, row-major): out[i,j] = sum k[a,b] * in[i + kh//2 - a, j + kw//2 - b].
export function convolve2d(img, H, W, k, kh, kw, mode = 'reflect', cval = 0) {
  const cy = kh >> 1, cx = kw >> 1;
  // pad enough on both sides: indices range from i + cy - (kh-1) to i + cy
  const py = Math.max(cy, kh - 1 - cy), px = Math.max(cx, kw - 1 - cx);
  const { p, PW } = pad(img, H, W, py, px, mode, cval);
  const out = new Float64Array(H * W);
  // collect non-zero taps once: offset into padded image relative to (i+py, j+px)
  const taps = [], offs = [];
  for (let a = 0; a < kh; a++) for (let b = 0; b < kw; b++) {
    const v = k[a * kw + b];
    if (v !== 0) { taps.push(v); offs.push((cy - a) * PW + (cx - b)); }
  }
  const T = taps.length, tv = Float64Array.from(taps), to = Int32Array.from(offs);
  for (let i = 0; i < H; i++) {
    const base = (i + py) * PW + px;
    for (let j = 0; j < W; j++) {
      const b0 = base + j;
      let s = 0;
      for (let t = 0; t < T; t++) s += tv[t] * p[b0 + to[t]];
      out[i * W + j] = s;
    }
  }
  return out;
}

// scipy.ndimage._gaussian_kernel1d(sigma, order, radius): sampled Gaussian (normalised) or its 1st/2nd derivative.
export function gaussianKernel1d(sigma, order, radius) {
  const n = 2 * radius + 1, phi = new Float64Array(n), c2 = -0.5 / (sigma * sigma);
  let sum = 0;
  for (let i = 0; i < n; i++) { const x = i - radius; phi[i] = Math.exp(c2 * x * x); sum += phi[i]; }
  for (let i = 0; i < n; i++) phi[i] /= sum;
  if (order === 0) return phi;
  const s2 = sigma * sigma, out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const x = i - radius;
    const q = order === 1 ? -x / s2 : order === 2 ? (x * x - s2) / (s2 * s2) : NaN;
    out[i] = q * phi[i];
  }
  return out;
}

// scipy.ndimage.gaussian_filter(img, sigma, order=(oy, ox), mode, truncate=4): 1-D passes along rows then cols.
export function gaussianFilter(img, H, W, sigma, { order = [0, 0], mode = 'reflect', cval = 0, truncate = 4 } = {}) {
  const radius = Math.floor(truncate * sigma + 0.5);
  let out = img;
  for (const axis of [0, 1]) {
    const w = gaussianKernel1d(sigma, order[axis], radius).reverse();
    out = correlate1d(out, H, W, w, axis, mode, cval);
  }
  return out;
}

// Generic separable edge filter (skimage.filters._generic_edge_filter): edge weights along `axis`, smoothing along the
// other, applied with ndi.convolve (mode 'reflect'). Returns the signed response.
export function edgeAxis(img, H, W, smooth, axis, edge = [1, 0, -1]) {
  const k = new Float64Array(9);
  for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) k[a * 3 + b] = axis === 0 ? edge[a] * smooth[b] : smooth[a] * edge[b];
  return convolve2d(img, H, W, k, 3, 3, 'reflect');
}

// Edge magnitude sqrt((gy^2 + gx^2) / 2), as skimage.filters.sobel / scharr / prewitt with axis=None.
export function edgeMagnitude(img, H, W, smooth) {
  const g0 = edgeAxis(img, H, W, smooth, 0), g1 = edgeAxis(img, H, W, smooth, 1), out = new Float64Array(H * W);
  const r2 = Math.sqrt(2);
  for (let i = 0; i < out.length; i++) out[i] = Math.sqrt(g0[i] * g0[i] + g1[i] * g1[i]) / r2;
  return out;
}

export const SOBEL_SMOOTH = [0.25, 0.5, 0.25];
export const SCHARR_SMOOTH = [3 / 16, 10 / 16, 3 / 16];
export const PREWITT_SMOOTH = [1 / 3, 1 / 3, 1 / 3];

// scipy.ndimage.sobel(img, axis, mode): correlate1d [-1, 0, 1] along axis, [1, 2, 1] along the other (not normalised).
export function ndiSobel(img, H, W, axis, mode = 'reflect', cval = 0) {
  const d = correlate1d(img, H, W, [-1, 0, 1], axis, mode, cval);
  return correlate1d(d, H, W, [1, 2, 1], 1 - axis, mode, cval);
}
