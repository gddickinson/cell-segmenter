// node tiff.test.mjs: readTiff decodes files written by tifffile (several dtypes and compressions) to the same
// values, and writeTiff -> readTiff round-trips. With an output path argument it also writes a small TIFF for the
// Python side to check with tifffile (tests/test_web_engine.py).
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync } from 'node:fs';
import { pixelSize, readTiff, writeTiff } from './tiff.js';

const fx = JSON.parse(readFileSync(new URL('../../../tests/fixtures/tiff_case.json', import.meta.url)));
for (const f of fx.files) {
  const b = Buffer.from(f.data, 'base64');
  const t = await readTiff(b.buffer.slice(b.byteOffset, b.byteOffset + b.length));
  const shape = f.shape.length === 3 ? f.shape : [1, ...f.shape];
  assert.equal(t.frames.length, shape[0], f.label);
  assert.equal(t.height, shape[1]); assert.equal(t.width, shape[2]); assert.equal(t.dtype, f.dtype, f.label);
  const flat = t.frames.flatMap(fr => [...fr]);
  if (f.dtype === 'float32') flat.forEach((v, i) => assert.ok(Math.abs(v - f.expected[i]) < 1e-7, f.label));
  else assert.deepEqual(flat, f.expected, f.label);
  console.log(`ok read ${f.label}`);
}

const W = 7, H = 5, frames = [0, 1, 2].map(k => Uint16Array.from({ length: W * H }, (_, i) => (i * 911 + k * 7) % 65536));
for (const bits of [8, 16]) {
  const fr = bits === 8 ? frames.map(f => Uint8Array.from(f, v => v & 255)) : frames;
  const t = await readTiff(writeTiff(fr, W, H, { bits, umPerPx: 0.25 }).buffer);
  assert.equal(t.frames.length, 3); assert.equal(t.dtype, `uint${bits}`);
  t.frames.forEach((f, k) => assert.deepEqual([...f], [...fr[k]]));
  assert.ok(Math.abs(t.umPerPx - 0.25) < 1e-9);
  console.log(`ok writeTiff ${bits}-bit round trip with pixel size`);
}
assert.equal(pixelSize('<OME><Pixels PhysicalSizeX="0.108" PhysicalSizeXUnit="µm"/></OME>', null), 0.108);
assert.equal(pixelSize('<OME><Pixels PhysicalSizeX="160" PhysicalSizeXUnit="nm"/></OME>', null), 0.16);
assert.equal(pixelSize('', 3), null);
if (process.argv[2]) writeFileSync(process.argv[2], writeTiff(frames, W, H, { bits: 16, umPerPx: 0.25 }));
