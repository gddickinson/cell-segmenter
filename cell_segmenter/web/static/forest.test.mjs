// node forest.test.mjs: (1) predictForest on a scikit-learn forest exported by the Python side gives sklearn's labels
// exactly; (2) end to end, JS features + the exported forest reproduce RandomForestModel.predict on an image crop;
// (3) a JS-trained forest (same algorithm, different random draws) agrees closely with sklearn on held-out pixels.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { extractFeatures } from './features.js';
import { forestFromJson, predictForest, trainForest } from './forest.js';

const fx = JSON.parse(readFileSync(new URL('../../../tests/fixtures/forest_case.json', import.meta.url)));
const buf = s => { const b = Buffer.from(s, 'base64'); return b.buffer.slice(b.byteOffset, b.byteOffset + b.length); };
const featMajor = (s, n) => { const a = new Float32Array(buf(s)), F = a.length / n; return Array.from({ length: F }, (_, k) => a.subarray(k * n, (k + 1) * n)); };
const forest = forestFromJson(fx.forest);

// 1. exact sklearn predictions
const X = featMajor(fx.X, fx.y.length);
const pred = predictForest(forest, X, fx.y.length);
assert.deepEqual([...pred], fx.sk_pred);
console.log(`ok predictForest = sklearn predict on ${fx.y.length} training pixels (${forest.trees.length} trees)`);

// 2. end to end on an image crop
const [H, W] = fx.crop_shape, img = new Float64Array(buf(fx.crop_image)), want = new Uint8Array(buf(fx.crop_pred));
const p2 = predictForest(forest, extractFeatures(img, H, W), H * W);
let same = 0;
for (let i = 0; i < H * W; i++) same += p2[i] === want[i];
assert.equal(same, H * W, `JS features + forest differ from RandomForestModel.predict on ${H * W - same} pixels`);
console.log(`ok JS features + exported forest = RandomForestModel.predict on a ${H}x${W} crop (${H * W} pixels)`);

// 3. JS training
const Xt = featMajor(fx.X_test, fx.y_test.length);
const js = trainForest(X, fx.y, { n_estimators: 30, seed: 4 });
const pj = predictForest(js, Xt, fx.y_test.length);
let agree = 0, correct = 0, skCorrect = 0;
pj.forEach((v, i) => { agree += v === fx.sk_pred_test[i]; correct += v === fx.y_test[i]; skCorrect += fx.sk_pred_test[i] === fx.y_test[i]; });
const n = fx.y_test.length;
console.log(`ok JS-trained forest (30 trees, 500 strokes): ${(100 * agree / n).toFixed(1)}% agreement with sklearn, `
  + `accuracy ${(100 * correct / n).toFixed(1)}% (sklearn, which trained on all stroke pixels: ${(100 * skCorrect / n).toFixed(1)}%) on ${n} stroke pixels the JS forest did not see`);
assert.ok(agree / n > 0.95 && correct / n > 0.95);
