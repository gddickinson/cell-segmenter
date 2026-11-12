// node features.test.mjs: the JS feature extractor reproduces FeatureExtractor.extract_features (Python) on the
// fixture images (tests/fixtures/features_case.json, written by scripts/make_web_fixtures.py).
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { extractFeatures, FEATURE_GROUPS, FEATURE_NAMES, featureIndices } from './features.js';

const fx = JSON.parse(readFileSync(new URL('../../../tests/fixtures/features_case.json', import.meta.url)));
const f64 = s => { const b = Buffer.from(s, 'base64'); return new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length)); };

assert.equal(FEATURE_NAMES.length, fx.n_features);
assert.deepEqual(FEATURE_GROUPS, fx.groups);
assert.deepEqual(featureIndices(['LBP Features', 'Gabor Features']), [14, ...fx.groups['Gabor Features']]);
assert.equal(featureIndices(Object.keys(fx.groups)), null);
console.log('ok feature groups match models/features.py::FEATURE_GROUPS');
for (const c of fx.cases) {
  const [H, W] = c.shape, img = f64(c.image), want = f64(c.features);
  const got = extractFeatures(img, H, W);
  assert.equal(got.length, fx.n_features);
  let worst = 0;
  got.forEach((m, k) => {
    let err = 0, scale = 1e-12;
    for (let i = 0; i < H * W; i++) {
      const w = want[k * H * W + i];
      err = Math.max(err, Math.abs(m[i] - w)); scale = Math.max(scale, Math.abs(w));
    }
    const rel = err / scale;
    worst = Math.max(worst, rel);
    assert.ok(rel < 1e-9, `${c.name}: feature ${k} (${FEATURE_NAMES[k]}) max error ${err} (scale ${scale})`);
  });
  console.log(`ok ${c.name} ${H}x${W}: 36 features match Python, worst relative error ${worst.toExponential(1)}`);
}
