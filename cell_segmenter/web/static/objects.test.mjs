// node objects.test.mjs: labelling and per-object statistics match utils/object_stats.py (skimage) on the fixture.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { objectsCsv, objectStats } from './objects.js';

const fx = JSON.parse(readFileSync(new URL('../../../tests/fixtures/objects_case.json', import.meta.url)));
const [H, W] = fx.shape;
const bits = Buffer.from(fx.mask, 'base64'), mask = new Uint8Array(H * W);
for (let i = 0; i < H * W; i++) mask[i] = (bits[i >> 3] >> (7 - (i & 7))) & 1;
const b = Buffer.from(fx.image, 'base64'), image = new Float64Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));

const close = (a, e, msg) => (e === null ? assert.equal(a, null, msg) : assert.ok(Math.abs(a - e) <= 1e-9 * Math.max(1, Math.abs(e)), `${msg}: ${a} vs ${e}`));
for (const [want, opts] of [[fx.stats, { image }], [fx.stats_um, { image, umPerPx: 0.16, minAreaPx: 2 }]]) {
  const got = objectStats(mask, H, W, opts);
  assert.equal(got.units, want.units);
  assert.equal(got.objects.length, want.objects.length);
  want.objects.forEach((w, i) => { for (const k of Object.keys(w)) close(got.objects[i][k], w[k], `object ${i} ${k}`); });
  for (const k of Object.keys(want.summary)) close(got.summary[k], want.summary[k], `summary ${k}`);
  console.log(`ok ${got.objects.length} objects (${got.units}) match skimage regionprops`);
}
assert.ok(objectsCsv(objectStats(mask, H, W, { image }), 0).startsWith('frame,label,area_px'));
