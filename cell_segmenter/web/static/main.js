// Entry point: lazy page modules, the mode label, a subtle background of drifting cell outlines, then the router.
import { $, app, isStatic, lazy, route, start } from './core.js';

const PAGES = { 'home.js': ['home'], 'segment.js': ['segment'], 'features_page.js': ['features'], 'docs.js': ['method', 'about'] };
lazy(Object.fromEntries(Object.entries(PAGES).flatMap(([file, names]) => names.map(n => [n, () => import('./' + file)]))));
route('notfound', () => { app().innerHTML = '<h1>Not found</h1><p><a href="#/">Back to the start</a></p>'; });

$('#mode').textContent = isStatic()
  ? 'static site: segmentation runs in your browser (JS port tested against the Python code)'
  : 'local app: Python engine on this computer, or the in-browser JS port';

// Background: a few faint elliptical outlines drifting slowly. Off under prefers-reduced-motion.
function background(canvas) {
  if (!canvas || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const g = canvas.getContext('2d');
  const fit = () => { canvas.width = innerWidth; canvas.height = innerHeight; };
  fit(); addEventListener('resize', fit);
  const cols = ['#3de0c8', '#ffb02e', '#7aa8ff'];
  const cells = Array.from({ length: 11 }, (_, i) => ({ x: Math.random() * innerWidth, y: Math.random() * innerHeight,
    a: 26 + Math.random() * 40, b: 18 + Math.random() * 22, t: Math.random() * Math.PI, vx: (Math.random() - 0.5) * 0.25,
    vy: (Math.random() - 0.5) * 0.25, vt: (Math.random() - 0.5) * 0.002, c: cols[i % 3] }));
  let last = 0;
  const tick = t => {
    requestAnimationFrame(tick);
    if (document.hidden || t - last < 50) return;
    last = t;
    g.clearRect(0, 0, canvas.width, canvas.height);
    for (const c of cells) {
      c.x = (c.x + c.vx + canvas.width) % canvas.width; c.y = (c.y + c.vy + canvas.height) % canvas.height; c.t += c.vt;
      g.strokeStyle = c.c; g.lineWidth = 1.4; g.beginPath(); g.ellipse(c.x, c.y, c.a, c.b, c.t, 0, 2 * Math.PI); g.stroke();
    }
  };
  requestAnimationFrame(tick);
}
background($('#bg'));
start();
