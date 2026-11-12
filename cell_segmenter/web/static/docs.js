// Method and About pages (content from the README, the code and the web tests).
import { app, esc, isStatic, route } from './core.js';
import { FEATURE_NAMES } from './features.js';
import { getInfo } from './state.js';

route('method', async () => {
  const info = await getInfo(), f = info.features, d = info.defaults;
  app().innerHTML = `
  <h1>How it works</h1>
  <p class="lead">A supervised pixel classifier: you label a few pixels of each class, the model learns what those
    pixels look like across many filters, and it labels every other pixel.</p>
  <h2>Workflow (as in the desktop app)</h2>
  <ol class="steps">
    <li><b>Load</b> a TIFF stack (frames × rows × columns). Each frame is scaled to [0, 1] by its own minimum and maximum
      (<code>utils/image_utils.py::normalize_stack</code>).</li>
    <li><b>Create labels</b>, e.g. background and cell. The order matters: label 0 is the first one.</li>
    <li><b>Paint</b> strokes for each label on one frame (brush of radius size/2, <code>utils/brush.py</code>).</li>
    <li><b>Train</b>: every painted pixel of the current frame becomes one training sample with its 36 feature values
      and its label index (<code>models/base.py::prepare_training_data</code>).</li>
    <li><b>Segment</b> the current frame or all frames: every pixel gets the label the forest votes for. The result mask is
      every label except the first (<code>segmentation &gt; 0</code> in <code>gui/main_window.py</code>).</li>
  </ol>
  <h2>Features (<code>models/features.py</code>)</h2>
  <p>Computed on the normalised frame with scikit-image and SciPy: Gaussian σ = ${f.sigmas.join(', ')} (with Sobel
    gradient magnitude and Laplacian of each), Sobel / Scharr / Roberts / Prewitt edges, uniform LBP (P = 8, R = 1),
    Hessian determinant and trace at σ = ${f.hessian_sigmas.join(', ')}, Gabor filters at frequencies
    ${f.gabor_frequencies.join(', ')} and orientations ${f.gabor_orientations.join('°, ')}°, local entropy in a
    ${f.entropy_radius}×${f.entropy_radius} window, and structure-tensor orientation and coherency (σ = ${f.structure_sigma}).</p>
  <details><summary>All ${FEATURE_NAMES.length} features in order</summary><ol start="0">${FEATURE_NAMES.map(n => `<li>${esc(n)}</li>`).join('')}</ol></details>
  <h2>Classifier (<code>models/random_forest.py</code>)</h2>
  <p>scikit-learn <code>RandomForestClassifier</code> with ${d.n_estimators} trees and maximum depth ${d.max_depth}
    (<code>config.py</code>), otherwise default settings: bootstrap samples, √36 = 6 candidate features per split, Gini
    impurity. The desktop app also offers a small CNN (PyTorch, <code>models/cnn.py</code>); it is not part of the web version.</p>
  <h2>The web version and its checks</h2>
  <ul>
    <li><b>Feature extraction</b>: a JavaScript port (<code>features.js</code>, <code>filters.js</code>) reproducing the
      SciPy/scikit-image boundary handling. Node tests compare all 36 maps with the Python output on two images: largest
      relative difference below 1e-13.</li>
    <li><b>Prediction</b>: a forest trained by Python can be exported and evaluated in JS; the predictions are identical
      to scikit-learn's, label for label, including end to end from the image.</li>
    <li><b>Training in the browser</b>: the same algorithm in JS. Random draws differ from scikit-learn, so the trees are
      not the same. On the whole demo stack a JS-trained forest (100 trees, three seeds) gives the same label as the Python
      one on 99.4–99.5% of pixels, and the same agreement with the known cell masks (89.4–89.6% vs 89.4%).</li>
    <li><b>Object statistics</b> (<code>utils/object_stats.py</code>, <code>objects.js</code>): 8-connected components,
      area, centroid, bounding box, equivalent diameter, eccentricity and mean intensity, identical to
      <code>skimage.measure.regionprops</code>. Physical units only when a pixel size is known (from OME or ImageJ metadata, or typed in).</li>
    <li><b>TIFF</b>: a small reader (8/16/32-bit, float, LZW / Deflate / PackBits, multi-page) checked against files
      written by <code>tifffile</code>; exports are ImageJ-compatible stacks.</li>
  </ul>
  ${isStatic() ? '' : '<p>In the local app the Python engine runs the repo\'s own <code>RandomForestModel</code> on the server.</p>'}`;
});

route('about', async () => {
  const info = await getInfo();
  app().innerHTML = `
  <h1>About</h1>
  <p class="lead">Cell Segmenter is a PyQt6 tool for segmenting microscopy image sequences, designed for DIC images from
    TIRF microscopy, with an interactive interface for training machine-learning models on painted labels.</p>
  <h2>Desktop app</h2>
  <pre>pip install -r requirements.txt
python main.py</pre>
  <p>Load TIFF Stack → Add Label → Paint mode → Train Model (Random Forest or CNN) → Segment Current Frame / Segment All Frames.
    The results appear as a new label.</p>
  <h2>Web app (local)</h2>
  <pre>pip install -r requirements-web.txt
python run_web.py          # http://127.0.0.1:8146</pre>
  <p>Same pages as this site, plus the Python engine: uploaded stacks are read with <code>tifffile</code> and segmented with
    the repo's <code>RandomForestModel</code>. Nothing leaves your computer.</p>
  <h2>Static site</h2>
  <p>Built by <code>scripts/build_static.py</code> and published to GitHub Pages. ${isStatic() ? 'You are on it: ' : ''}it runs
    entirely in the browser; the demo's Python results were computed when the site was built.</p>
  <h2>Demo data</h2>
  <p>${esc(info.demo.title)}. ${esc(info.demo.description)}</p>
  <h2>Credits</h2>
  <p>By George Dickinson (<a href="https://github.com/gddickinson/cell-segmenter">gddickinson/cell-segmenter</a>, MIT licence),
    built with substantial assistance from Claude (Anthropic). Inspired by ilastik and other bioimage analysis software.
    Uses NumPy, SciPy, scikit-image, scikit-learn, tifffile, PyQt6 and pyqtgraph.</p>`;
});
