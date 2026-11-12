// Features page: any of the 36 feature maps of any frame (computed by the JS port in the worker), and the feature
// importances of the last Python-trained forest (the demo forest, or a local-app run).
import { $, app, esc, route } from './core.js';
import { FEATURE_NAMES } from './features.js';
import { getInfo, loadDemo, S } from './state.js';
import { call } from './worker_client.js';

let current = 0;

route('features', async () => {
  await getInfo();
  if (!S.stack) await loadDemo();
  app().innerHTML = `
  <h1>Feature maps</h1>
  <p class="lead">The classifier sees every pixel through 36 numbers: intensity, Gaussian smoothing, gradients and
    Laplacians at three scales, four edge filters, local binary patterns, Hessian determinant and trace, twelve Gabor
    filters, local entropy and the structure tensor (<code>models/features.py</code>). Pick one to see it on the current stack.</p>
  <div class="featview">
    <section>
      <div class="row">
        <div><label for="fsel">Feature</label><select id="fsel">${FEATURE_NAMES.map((n, i) => `<option value="${i}">${i}. ${esc(n)}</option>`).join('')}</select></div>
        <div id="fframectl"><label for="fframe">Frame <b id="fframev"></b></label><input type="range" id="fframe" min="0" max="${S.stack.T - 1}" value="${S.frame}"></div>
      </div>
      <p class="status" id="fstatus" role="status" aria-live="polite"></p>
      <div class="viewwrap"><canvas id="fview" aria-label="Feature map"></canvas></div>
      <p class="small mute" id="frange"></p>
    </section>
    <aside class="panel"><h3>Feature importance</h3><div id="imp"></div></aside>
  </div>`;
  $('#fsel').value = current;
  $('#fsel').onchange = e => { current = +e.target.value; show(); };
  $('#fframe').oninput = e => { S.frame = +e.target.value; show(); };
  $('#fframectl').classList.toggle('hide', S.stack.T < 2);
  importances();
  show();
});

async function show() {
  const { H, W } = S.stack, f = S.frame;
  $('#fframev').textContent = `${f + 1} / ${S.stack.T}`;
  $('#fstatus').textContent = 'computing...';
  const m = await call('feature', { frame: f, k: current }, p => { if ($('#fstatus')) $('#fstatus').textContent = `${p.stage} (${p.done}/${p.total})`; });
  if (!$('#fview')) return;
  let lo = Infinity, hi = -Infinity;
  for (const v of m) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const cv = $('#fview'); cv.width = W; cv.height = H;
  const img = new ImageData(W, H), k = hi > lo ? 255 / (hi - lo) : 0;
  for (let i = 0; i < H * W; i++) { const g = (m[i] - lo) * k; img.data[4 * i] = img.data[4 * i + 1] = img.data[4 * i + 2] = g; img.data[4 * i + 3] = 255; }
  cv.getContext('2d').putImageData(img, 0, 0);
  $('#fstatus').textContent = '';
  $('#frange').textContent = `${FEATURE_NAMES[current]} on frame ${f + 1}: range ${lo.toPrecision(4)} to ${hi.toPrecision(4)} (shown min to max).`;
}

function importances() {
  const imp = S.result?.importances;
  if (!imp) { $('#imp').innerHTML = '<p class="small mute">Shown for forests trained by the Python code (the demo forest, or the Python engine of the local app). The JS port does not compute them.</p>'; return; }
  const rows = S.result.featureIndices;  // columns of a forest trained on a feature subset -> feature rows
  const max = Math.max(...imp), order = imp.map((v, i) => [v, rows ? rows[i] : i]).sort((a, b) => b[0] - a[0]);
  $('#imp').innerHTML = `<p class="small mute">Mean decrease in Gini impurity (scikit-learn), from: ${esc(S.result.note)}.</p>
    <div class="bars">${order.map(([v, i]) => `<button data-k="${i}">${i}. ${esc(FEATURE_NAMES[i])}</button>
      <span class="bar" style="width:${(100 * v / max).toFixed(1)}%"></span><span class="mono">${(100 * v).toFixed(1)}%</span>`).join('')}</div>`;
  $('#imp').onclick = e => { const k = e.target.dataset.k; if (k !== undefined) { current = +k; $('#fsel').value = k; show(); } };
}
