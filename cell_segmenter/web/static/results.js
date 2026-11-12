// Results panel of the Segment page: label fractions, objects (8-connected components of the result mask) with
// optional physical units, accuracy against the known masks of the demo, and exports (PNG, TIFF stack, CSV).
import { $, download, esc } from './core.js';
import { S } from './state.js';
import { compose } from './viewer.js';
import { objectsCsv, objectStats } from './objects.js';
import { writeTiff } from './tiff.js';

let minArea = 10, umOverride = '';
const fmt = (v, d = 1) => (v === null || v === undefined ? '–' : (+v).toFixed(d));

function stats(f) {
  const st = S.stack, pred = S.result.pred.get(f), N = st.H * st.W, mask = new Uint8Array(N);
  for (let i = 0; i < N; i++) mask[i] = pred[i] > 0 ? 1 : 0;
  const fr = st.frames[f];
  let lo = Infinity, hi = -Infinity;
  for (const v of fr) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const image = Float64Array.from(fr, v => (hi > lo ? (v - lo) / (hi - lo) : 0));  // normalised, as the model sees it
  const um = umOverride !== '' ? +umOverride : st.umPerPx;
  return objectStats(mask, st.H, st.W, { image, umPerPx: um > 0 ? um : null, minAreaPx: minArea });
}

export function renderResults(host) {
  const r = S.result, st = S.stack;
  if (!r) { host.innerHTML = ''; return; }
  const pred = r.pred.get(S.frame);
  if (!pred) {
    host.innerHTML = `<div class="panel"><p class="mute">Frame ${S.frame + 1} has not been segmented yet
      (${r.pred.size} of ${st.T} frames have a result). Use “Segment this frame” or “Segment all frames”.</p></div>`;
    return;
  }
  const N = st.H * st.W, counts = new Array(S.labels.length).fill(0);
  for (let i = 0; i < N; i++) counts[pred[i]]++;
  const s = stats(S.frame), truth = st.truth?.[S.frame];
  let acc = null;
  if (truth) { let ok = 0; for (let i = 0; i < N; i++) ok += (pred[i] > 0) === !!truth[i]; acc = ok / N; }
  const um = s.um_per_px;
  host.innerHTML = `
  <div class="panel">
    <h3>Frame ${S.frame + 1}: result</h3>
    <div class="tiles">
      <div class="tile"><b>${s.summary.n_objects}</b><span>objects ≥ ${minArea} px</span></div>
      <div class="tile"><b>${(100 * s.summary.mask_fraction).toFixed(1)}%</b><span>of the frame in the result mask</span></div>
      <div class="tile"><b>${s.summary.median_area_px === null ? '–' : um ? fmt(s.summary.median_area_px * um * um, 1) : fmt(s.summary.median_area_px, 0)}</b>
        <span>median object area (${um ? 'µm²' : 'px'})</span></div>
      ${acc === null ? '' : `<div class="tile"><b>${(100 * acc).toFixed(1)}%</b><span>pixels agreeing with the known cell masks</span></div>`}
    </div>
    <p class="small">${S.labels.map((l, i) => `<span><i class="sw" style="background:${l.color}"></i>${esc(l.name)}: ${(100 * counts[i] / N).toFixed(1)}%</span>`).join(' &nbsp; ')}</p>
    <div class="row">
      <div><label for="minarea">Minimum object area (px)</label><input type="number" id="minarea" min="0" value="${minArea}"></div>
      <div><label for="umpx">Pixel size (µm/px)${st.umPerPx ? '' : ', not in the file'}</label>
        <input type="number" id="umpx" min="0" step="any" placeholder="${st.umPerPx ? +st.umPerPx.toPrecision(4) : 'pixel units only'}" value="${esc(umOverride)}"></div>
    </div>
    <p class="small mute">Objects are 8-connected regions of the result mask (labels after the first). Units: ${esc(s.units)}.
      Coordinates are pixel centres, origin at the top-left pixel (row, column).</p>
    <div class="scroll"><table class="tbl"><thead><tr><th>#</th><th>area ${um ? 'µm²' : 'px'}</th><th>centroid (row, col)</th>
      <th>eq. diameter ${um ? 'µm' : 'px'}</th><th>eccentricity</th><th>mean intensity</th></tr></thead><tbody>
      ${s.objects.slice(0, 100).map(o => `<tr><td>${o.label}</td><td>${um ? fmt(o.area_um2, 1) : o.area_px}</td>
        <td>${fmt(o.centroid_row)}, ${fmt(o.centroid_col)}</td><td>${um ? fmt(o.equivalent_diameter_um, 2) : fmt(o.equivalent_diameter_px)}</td>
        <td>${fmt(o.eccentricity, 2)}</td><td>${fmt(o.mean_intensity, 3)}</td></tr>`).join('')}
      </tbody></table></div>
    ${s.objects.length > 100 ? `<p class="small mute">First 100 of ${s.objects.length} objects shown; the CSV has all.</p>` : ''}
    <h3>Export</h3>
    <div class="exports">
      <button class="alt" data-x="mask-png">Mask PNG (this frame)</button>
      <button class="alt" data-x="labels-png">Label PNG (value = label index)</button>
      <button class="alt" data-x="overlay-png">Overlay PNG (as shown)</button>
      <button class="alt" data-x="mask-tif">Mask TIFF stack</button>
      <button class="alt" data-x="labels-tif">Label TIFF stack</button>
      <button class="alt" data-x="csv">Objects CSV</button>
    </div>
    <p class="small mute">TIFF stacks and the CSV contain the ${r.pred.size} segmented frame${r.pred.size > 1 ? 's' : ''}
      (${[...r.pred.keys()].sort((a, b) => a - b).map(f => f + 1).join(', ')}); masks are 0/255, label images hold the label index.</p>
  </div>`;
  $('#minarea', host).onchange = e => { minArea = Math.max(0, +e.target.value || 0); renderResults(host); };
  $('#umpx', host).onchange = e => { umOverride = e.target.value.trim(); renderResults(host); };
  host.querySelector('.exports').onclick = e => { const x = e.target.dataset.x; if (x) exportAs(x); };
}

function base() { return (S.stack.name || 'stack').replace(/\.(ome\.)?tiff?$/i, '').replace(/[^\w.-]+/g, '_'); }

function pngOf(imgData) {
  const c = document.createElement('canvas');
  c.width = imgData.width; c.height = imgData.height;
  c.getContext('2d').putImageData(imgData, 0, 0);
  return new Promise(res => c.toBlob(res, 'image/png'));
}

function grayImage(values, scale) {
  const { W, H } = S.stack, img = new ImageData(W, H);
  for (let i = 0; i < W * H; i++) { const v = values[i] * scale; img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = v; img.data[4 * i + 3] = 255; }
  return img;
}

async function exportAs(kind) {
  const st = S.stack, f = S.frame, pred = S.result.pred.get(f), frames = [...S.result.pred.keys()].sort((a, b) => a - b);
  const tag = `${base()}_frame${f + 1}`;
  const um = umOverride !== '' ? +umOverride : st.umPerPx;
  if (kind === 'mask-png') download(`${tag}_mask.png`, await pngOf(grayImage(pred.map(v => (v > 0 ? 1 : 0)), 255)));
  if (kind === 'labels-png') download(`${tag}_labels.png`, await pngOf(grayImage(pred, 1)));
  if (kind === 'overlay-png') download(`${tag}_overlay.png`, await pngOf(compose(f, { strokes: true, result: true })));
  if (kind === 'mask-tif' || kind === 'labels-tif') {
    const data = frames.map(k => { const p = S.result.pred.get(k); return kind === 'mask-tif' ? p.map(v => (v > 0 ? 255 : 0)) : p.slice(); });
    download(`${base()}_${kind === 'mask-tif' ? 'mask' : 'labels'}.tif`, new Blob([writeTiff(data, st.W, st.H, { bits: 8, umPerPx: um > 0 ? um : null })], { type: 'image/tiff' }));
  }
  if (kind === 'csv') {
    const parts = frames.map((k, i) => {
      const csv = objectsCsv(stats(k), k + 1);
      return i ? csv.split('\n').slice(1).join('\n') : csv;
    });
    download(`${base()}_objects.csv`, parts.join(''), 'text/csv');
  }
}
