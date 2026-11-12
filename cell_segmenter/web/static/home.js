// Home: hero, real numbers from the demo and the defaults, cards to the pages.
import { app, isStatic, route } from './core.js';
import { getInfo } from './state.js';

route('home', async () => {
  const info = await getInfo(), d = info.demo, acc = d.accuracy.reduce((s, v) => s + v, 0) / d.accuracy.length;
  app().innerHTML = `
  <section class="hero">
    <p class="kicker">Interactive pixel classification · TIFF stacks</p>
    <h1 class="glow">Cell Segmenter</h1>
    <p class="lead">Paint a few strokes on a microscopy stack, train a random forest on 36 multi-scale image features, and
      segment one frame or the whole stack. A browser version of the PyQt6 desktop tool, built for DIC images from TIRF
      microscopy and in the spirit of ilastik.</p>
    <a class="btn" href="#/segment">Segment the demo</a><a class="btn alt" href="#/features">Feature maps</a><a class="btn alt" href="#/method">How it works</a>
  </section>
  <div class="tiles">
    <div class="tile"><b>${info.features.count}</b><span>features per pixel</span></div>
    <div class="tile"><b>${info.defaults.n_estimators}</b><span>trees, max depth ${info.defaults.max_depth} (config.py)</span></div>
    <div class="tile"><b>${d.frames}×${d.width}×${d.height}</b><span>synthetic demo stack</span></div>
    <div class="tile"><b>${(100 * acc).toFixed(1)}%</b><span>demo pixels agreeing with the known masks (Python model)</span></div>
  </div>
  ${isStatic() ? `<div class="callout">This is the static site: everything runs in your browser. Feature extraction is a
    JavaScript port of the Python code that matches it to better than 1e-13 (relative); the demo forest was trained by the repo's Python
    <code>RandomForestModel</code> when the site was built, and retraining here uses a JS random forest of the same kind.
    Your TIFF files never leave your computer.</div>` : ''}
  <h2>Pages</h2>
  <div class="grid">
    <a class="card" href="#/segment"><b>Segment</b><span>Open a TIFF stack or the demo, paint labels, train, segment a frame or all frames, see objects and export masks (PNG, TIFF) and a CSV.</span></a>
    <a class="card" href="#/features"><b>Feature maps</b><span>Look at each of the 36 features on your stack, and at which ones the forest relied on.</span></a>
    <a class="card" href="#/method"><b>Method</b><span>The workflow, the feature set and the classifier, and how the web version was checked against Python.</span></a>
    <a class="card" href="#/about"><b>About</b><span>Run the desktop app or the local web app, and credits.</span></a>
  </div>
  <h2>The demo stack</h2>
  <div class="card"><b>${d.title}</b><span>${d.description}</span>
    <small>${d.frames} frames · ${d.width}×${d.height} px · ${d.dtype} · ${d.n_cells} cells · example strokes on frame 1:
      ${Object.entries(d.strokes).map(([k, v]) => `${k} ${v} px`).join(', ')}</small></div>`;
});
