// Segment page: the desktop workflow in the browser. Load a TIFF stack (or the demo), add labels, paint strokes, train
// the random forest on the strokes of the current frame, segment this frame or all frames, inspect and export.
import { $, app, esc, isStatic, onLeave, route } from './core.js';
import { classesPainted, getInfo, loadDemo, loadFile, PALETTE, S, segment, trainingFrames } from './state.js';
import { attachBrush, compose, countStrokes, undo } from './viewer.js';
import { renderResults } from './results.js';

let mode = 'paint', zoom = 0, show = { strokes: true, result: true, truth: false };

route('segment', async () => {
  const info = await getInfo();
  const d = info.defaults;
  app().innerHTML = `
  <h1>Segment a stack</h1>
  <p class="lead">Paint a few strokes for each label on one frame, train the random forest on them, and segment the
    frame or the whole stack. Label order matters: the result mask is every label except the first, as in the desktop app.</p>
  <div class="bench">
    <aside>
      <div class="panel"><h3>Data</h3>
        <button id="demo" class="alt">Load the demo stack</button>
        <label for="file">Or open a TIFF stack (grayscale, 8/16/32-bit)</label>
        <input type="file" id="file" accept=".tif,.tiff,image/tiff">
        <p class="small mute" id="stackinfo"></p>
      </div>
      <div class="panel"><h3>Labels</h3>
        <ul class="labels" id="labels"></ul>
        <button id="addlab" class="alt">Add label</button>
      </div>
      <div class="panel"><h3>Brush</h3>
        <div class="seg" role="group" aria-label="Tool" id="modes">
          <button data-m="view">Navigate</button><button data-m="paint">Paint</button><button data-m="erase">Erase</button></div>
        <div class="ctl"><label for="brush">Brush size <b id="brushv"></b> px</label>
          <input type="range" id="brush" min="1" max="30" step="1"></div>
        <button id="undo" class="alt">Undo</button><button id="clear" class="alt">Clear this frame</button>
      </div>
      <div class="panel"><h3>Model</h3>
        <label for="engine">Engine</label>
        <select id="engine">
          ${isStatic() ? '' : '<option value="python">Python (RandomForestModel, local server)</option>'}
          <option value="js">Browser (JS port)</option>
        </select>
        ${isStatic() ? '<p class="static-note">The static site runs the JS port in your browser. The local app can also train with the Python code.</p>' : ''}
        <div class="row">
          <div><label for="trees">Trees</label><input type="number" id="trees" min="5" max="300" value="${d.n_estimators}"></div>
          <div><label for="depth">Max depth</label><input type="number" id="depth" min="2" max="30" value="${d.max_depth}"></div>
        </div>
        <label for="seed">Random seed (desktop default ${d.random_state ?? 'none'}; blank = a new forest each run)</label>
        <input type="number" id="seed" min="0" step="1" placeholder="random" value="${d.random_state ?? ''}">
        <details class="feats"><summary>Features: <span id="featcount">all 36</span></summary>
          ${Object.entries(info.features.groups).map(([g, rows]) => `<label class="chk"><input type="checkbox" data-group="${esc(g)}" checked> ${esc(g)} (${rows.length})</label>`).join('')}
        </details>
        <label class="chk"><input type="checkbox" id="allframes"> Use training data from all frames</label>
        <button id="segone">Segment this frame</button><button id="segall" class="alt">Segment all frames</button>
        <div class="progress" aria-hidden="true"><i id="bar"></i></div>
        <p class="status" id="status" role="status" aria-live="polite"></p>
      </div>
    </aside>
    <section>
      <div class="toolbar">
        <div class="ctl" style="flex:1;min-width:180px" id="framectl"><label for="frame">Frame <b id="framev"></b></label>
          <input type="range" id="frame" min="0" max="0" step="1" value="0"></div>
        <div class="ctl" style="min-width:150px"><label for="opacity">Result opacity <b id="opv"></b></label>
          <input type="range" id="opacity" min="0" max="1" step="0.05"></div>
      </div>
      <div class="toolbar">
        <label class="chk"><input type="checkbox" id="sh-strokes"> strokes</label>
        <label class="chk"><input type="checkbox" id="sh-result"> result</label>
        <label class="chk" id="truthctl"><input type="checkbox" id="sh-truth"> known cell outlines (demo)</label>
        <div class="seg" role="group" aria-label="Zoom" id="zoom">
          <button data-z="0">Fit</button><button data-z="1">1×</button><button data-z="2">2×</button><button data-z="4">4×</button></div>
      </div>
      <div class="viewwrap" id="wrap"><canvas id="view" aria-label="Current frame with strokes and segmentation overlay"></canvas></div>
      <div class="legend" id="legend"></div>
      <div id="results"></div>
    </section>
  </div>`;
  if (!S.stack) await loadDemo();
  wire();
  refresh();
  const listener = () => refresh();
  S.listeners.add(listener);
  onLeave(() => S.listeners.delete(listener));
});

function setStatus(msg, busy = false, frac = null) {
  $('#status').textContent = msg;
  $('#status').classList.toggle('busy', busy);
  $('#bar').style.width = frac === null ? '0' : `${Math.round(frac * 100)}%`;
}

function draw() {
  const cv = $('#view'), { W, H } = S.stack;
  if (cv.width !== W || cv.height !== H) { cv.width = W; cv.height = H; }
  cv.getContext('2d').putImageData(compose(S.frame, show), 0, 0);
}

function wire() {
  $('#demo').onclick = async () => { setStatus('loading the demo...', true); await loadDemo(); setStatus('demo loaded'); };
  $('#file').onchange = async e => {
    const f = e.target.files[0];
    if (!f) return;
    try { await loadFile(f, m => setStatus(m, true)); setStatus(`loaded ${f.name}: paint strokes for at least two labels, then segment`); }
    catch (err) { setStatus(`Could not open ${f.name}: ${err.message}`); }
  };
  $('#addlab').onclick = () => {
    const i = S.labels.length;
    S.labels.push({ name: `Label ${i + 1}`, color: PALETTE[i % PALETTE.length], masks: new Map() });
    S.active = i; S.result = null; refresh();
  };
  $('#modes').onclick = e => { const m = e.target.dataset.m; if (m) { mode = m; refresh(); } };
  $('#zoom').onclick = e => { const z = e.target.dataset.z; if (z !== undefined) { zoom = +z; refresh(); } };
  const brush = $('#brush');
  brush.value = S.brush;
  brush.oninput = () => { S.brush = +brush.value; $('#brushv').textContent = S.brush; };
  $('#undo').onclick = () => { if (!undo()) setStatus('nothing to undo'); };
  $('#clear').onclick = () => { S.labels.forEach(l => l.masks.delete(S.frame)); S.history = []; refresh(); };
  $('#frame').oninput = e => { S.frame = +e.target.value; refresh(); };
  $('#opacity').oninput = e => { S.opacity = +e.target.value; refresh(); };
  for (const k of Object.keys(show)) { const c = $(`#sh-${k}`); c.checked = show[k]; c.onchange = () => { show[k] = c.checked; draw(); }; }
  attachBrush($('#view'), () => mode, draw);
  $('.feats').onchange = () => {
    const rows = [...document.querySelectorAll('[data-group]')].filter(c => c.checked).reduce((s, c) => s + S.info.features.groups[c.dataset.group].length, 0);
    $('#featcount').textContent = rows === 36 ? 'all 36' : `${rows} of 36`;
  };
  $('#segone').onclick = () => run([S.frame]);
  $('#segall').onclick = () => run(S.stack.frames.map((_, i) => i));
}

async function run(frames) {
  const engine = $('#engine').value, seedTxt = $('#seed').value.trim();
  const allFrames = $('#allframes').checked;
  const groups = [...document.querySelectorAll('[data-group]')], picked = groups.filter(c => c.checked).map(c => c.dataset.group);
  if (!picked.length) { setStatus('Select at least one feature group'); return; }
  const features = picked.length === groups.length ? null : picked;
  const opts = { engine, allFrames, features, n_estimators: Math.max(1, +$('#trees').value || 100), max_depth: Math.max(1, +$('#depth').value || 10),
    seed: seedTxt === '' ? (engine === 'js' ? (Math.random() * 2 ** 31) >>> 0 : null) : +seedTxt };
  if (classesPainted(trainingFrames(allFrames)) < 2) {
    setStatus(`Paint strokes for at least two labels ${allFrames ? 'on any frame' : `on frame ${S.frame + 1}`} first`);
    return;
  }
  for (const b of ['#segone', '#segall']) $(b).disabled = true;
  try {
    await segment(frames, opts, p => setStatus(`${p.stage} (${p.done}/${p.total})`, true, p.total ? p.done / p.total : null));
    setStatus(`segmented ${frames.length === 1 ? `frame ${frames[0] + 1}` : `${frames.length} frames`}`);
  } catch (e) {
    setStatus(`Error: ${e.message}`);
  } finally {
    for (const b of ['#segone', '#segall']) $(b).disabled = false;
  }
}

function refresh() {
  if (!$('#view') || !S.stack) return;
  const st = S.stack;
  $('#stackinfo').innerHTML = `<b>${esc(st.name)}</b><br>${st.T} frame${st.T > 1 ? 's' : ''} · ${st.W}×${st.H} px · ${esc(st.dtype)}`
    + ` · pixel size ${st.umPerPx ? `${+st.umPerPx.toPrecision(4)} µm (from the file)` : 'not in the file'}`;
  $('#labels').innerHTML = S.labels.map((l, i) => `<li class="${i === S.active ? 'on' : ''}">
      <input type="color" value="${l.color}" data-col="${i}" aria-label="Colour of ${esc(l.name)}">
      <button class="pick" data-pick="${i}" aria-pressed="${i === S.active}">${i}. ${esc(l.name)}</button>
      <span class="n" title="stroke pixels on this frame">${countStrokes(l, S.frame)}</span>
      <button class="x" data-ren="${i}" aria-label="Rename ${esc(l.name)}" title="Rename">✎</button>
      ${S.labels.length > 2 ? `<button class="x" data-del="${i}" aria-label="Remove ${esc(l.name)}" title="Remove">×</button>` : ''}</li>`).join('');
  $('#labels').onclick = e => {
    const t = e.target.closest('button');
    if (!t) return;
    if (t.dataset.pick) { S.active = +t.dataset.pick; if (mode === 'view') mode = 'paint'; refresh(); }
    if (t.dataset.ren) { const l = S.labels[+t.dataset.ren], n = prompt('Label name', l.name); if (n) { l.name = n.trim() || l.name; refresh(); } }
    if (t.dataset.del) { S.labels.splice(+t.dataset.del, 1); S.active = Math.min(S.active, S.labels.length - 1); S.result = null; refresh(); }
  };
  $('#labels').oninput = e => { const i = e.target.dataset.col; if (i !== undefined) { S.labels[+i].color = e.target.value; draw(); renderLegend(); } };
  $$mode();
  $('#brushv').textContent = S.brush;
  $('#brush').value = S.brush;
  const fr = $('#frame');
  fr.max = st.T - 1; fr.value = S.frame;
  $('#framectl').classList.toggle('hide', st.T < 2);
  $('#framev').textContent = `${S.frame + 1} / ${st.T}`;
  $('#opacity').value = S.opacity; $('#opv').textContent = S.opacity.toFixed(2);
  $('#truthctl').classList.toggle('hide', !st.truth);
  const wrap = $('#wrap'), cv = $('#view');
  wrap.classList.toggle('paint', mode !== 'view');
  wrap.classList.toggle('zoomed', zoom > 0);
  cv.style.width = zoom > 0 ? `${st.W * zoom}px` : '';
  draw();
  renderLegend();
  renderResults($('#results'));
}

function $$mode() {
  for (const b of document.querySelectorAll('#modes button')) b.classList.toggle('on', b.dataset.m === mode);
  for (const b of document.querySelectorAll('#zoom button')) b.classList.toggle('on', +b.dataset.z === zoom);
}

function renderLegend() {
  const r = S.result;
  $('#legend').innerHTML = S.labels.map((l, i) => `<span><i class="sw" style="background:${l.color}"></i>${i}. ${esc(l.name)}${i === 0 ? ' (not in the result mask)' : ''}</span>`).join('')
    + (r ? `<span class="badge ${r.engine === 'python' ? 'py' : 'js'}">${esc(r.note)}</span>` : '<span class="mute">No result yet: segment a frame.</span>');
}
