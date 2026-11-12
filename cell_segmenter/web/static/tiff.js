// Minimal TIFF reader/writer for grayscale stacks (the desktop app reads them with tifffile.imread).
// readTiff(arrayBuffer) -> { frames: [TypedArray], width, height, dtype, umPerPx, description }
//   Reads every page (IFD) of a single-channel TIFF: 8/16/32-bit unsigned or signed integers and 32/64-bit floats,
//   little- or big-endian, strips (not tiles), compression none / LZW / PackBits / Deflate, horizontal predictor.
//   umPerPx comes from OME-XML PhysicalSizeX or an ImageJ description (unit=micron + XResolution), else null.
// writeTiff(frames, width, height, { bits: 8|16, umPerPx }) -> Uint8Array: uncompressed multi-page little-endian TIFF
//   with an ImageJ description, so Fiji/ImageJ and tifffile open it as a stack.
// Tested against files written by tifffile in tiff.test.mjs.

const TYPE_SIZE = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 16: 8 };

export async function readTiff(buffer) {
  const dv = new DataView(buffer), u8 = new Uint8Array(buffer);
  const bo = String.fromCharCode(u8[0], u8[1]);
  if (bo !== 'II' && bo !== 'MM') throw new Error('Not a TIFF file (no II/MM byte-order mark)');
  const le = bo === 'II';
  const magic = dv.getUint16(2, le);
  if (magic === 43) throw new Error('BigTIFF files are not supported in the browser; use the desktop app');
  if (magic !== 42) throw new Error('Not a TIFF file (bad magic number)');
  const frames = [];
  let off = dv.getUint32(4, le), width = 0, height = 0, dtype = '', description = '', xres = null;
  const seen = new Set();
  while (off && !seen.has(off)) {
    seen.add(off);
    const tags = readIfd(dv, off, le);
    off = dv.getUint32(off + 2 + 12 * dv.getUint16(off, le), le);
    const w = tags[256][0], h = tags[257][0], spp = (tags[277] || [1])[0];
    const planar = (tags[284] || [1])[0];
    // planar-separate samples are read as consecutive frames, as tifffile.imread returns them (S, H, W)
    if (spp !== 1 && planar !== 2) throw new Error(`Only grayscale TIFFs are supported; this one has ${spp} interleaved samples per pixel (RGB?)`);
    if (tags[324]) throw new Error('Tiled TIFFs are not supported in the browser; use the desktop app');
    const bits = tags[258][0], fmt = (tags[339] || [1])[0], comp = (tags[259] || [1])[0], pred = (tags[317] || [1])[0];
    const dt = dtypeOf(bits, fmt);
    if (!frames.length) {
      width = w; height = h; dtype = dt;
      if (tags[270]) description = new TextDecoder().decode(new Uint8Array(tags[270])).replace(/\0+$/, '');
      if (tags[282]) xres = tags[282][0];
    } else if (w !== width || h !== height || dt !== dtype) continue;  // thumbnails / other series
    const rps = (tags[278] || [h])[0], offsets = tags[273], counts = tags[279];
    const bpp = bits / 8, rowBytes = w * bpp, raw = new Uint8Array(spp * h * rowBytes), perPlane = Math.ceil(h / rps);
    let pos = 0;
    for (let s = 0; s < offsets.length; s++) {
      const src = u8.subarray(offsets[s], offsets[s] + counts[s]);
      const want = Math.min(rps, h - (s % perPlane) * rps) * rowBytes;
      let data = await decompress(src, comp, want);
      if (pred === 2) data = undoPredictor(data, w, bpp, le);
      raw.set(data.subarray(0, Math.min(want, raw.length - pos)), pos);
      pos += want;
    }
    const plane = h * rowBytes;
    for (let k = 0; k < spp; k++) frames.push(toTyped(raw.subarray(k * plane, (k + 1) * plane), dt, le));
  }
  if (!frames.length) throw new Error('No image pages found');
  return { frames, width, height, dtype, description, umPerPx: pixelSize(description, xres) };
}

function readIfd(dv, off, le) {
  const n = dv.getUint16(off, le), tags = {};
  for (let i = 0; i < n; i++) {
    const e = off + 2 + 12 * i, tag = dv.getUint16(e, le), type = dv.getUint16(e + 2, le), count = dv.getUint32(e + 4, le);
    const size = (TYPE_SIZE[type] || 1) * count;
    const at = size <= 4 ? e + 8 : dv.getUint32(e + 8, le);
    const vals = [];
    for (let k = 0; k < count; k++) {
      const p = at + k * (TYPE_SIZE[type] || 1);
      if (type === 3) vals.push(dv.getUint16(p, le));
      else if (type === 4 || type === 13) vals.push(dv.getUint32(p, le));
      else if (type === 5) vals.push(dv.getUint32(p, le) / dv.getUint32(p + 4, le));
      else if (type === 16) vals.push(Number(dv.getBigUint64(p, le)));
      else vals.push(dv.getUint8(p));
    }
    tags[tag] = vals;
  }
  return tags;
}

function dtypeOf(bits, fmt) {
  if (fmt === 3) { if (bits === 32) return 'float32'; if (bits === 64) return 'float64'; }
  if (fmt === 1 || fmt === 2) {
    const s = fmt === 2 ? 'int' : 'uint';
    if ([8, 16, 32].includes(bits)) return s + bits;
  }
  throw new Error(`Unsupported sample type (${bits}-bit, format ${fmt})`);
}

function toTyped(raw, dt, le) {
  const dv = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const get = { uint8: [1, 'getUint8', Uint8Array], int8: [1, 'getInt8', Int8Array], uint16: [2, 'getUint16', Uint16Array],
    int16: [2, 'getInt16', Int16Array], uint32: [4, 'getUint32', Uint32Array], int32: [4, 'getInt32', Int32Array],
    float32: [4, 'getFloat32', Float32Array], float64: [8, 'getFloat64', Float64Array] }[dt];
  const [size, fn, Ctor] = get, n = raw.length / size, out = new Ctor(n);
  for (let i = 0; i < n; i++) out[i] = dv[fn](i * size, le);
  return out;
}

async function decompress(src, comp, want) {
  if (comp === 1) return src;
  if (comp === 32773) return packbits(src, want);
  if (comp === 5) return lzw(src, want);
  if (comp === 8 || comp === 32946) {
    const ds = new Blob([src]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(ds).arrayBuffer());
  }
  throw new Error(`Unsupported TIFF compression (${comp}); save as uncompressed, LZW or Deflate`);
}

function packbits(src, want) {
  const out = new Uint8Array(want);
  let i = 0, o = 0;
  while (i < src.length && o < want) {
    const n = (src[i++] << 24) >> 24;
    if (n >= 0) { for (let k = 0; k <= n && o < want; k++) out[o++] = src[i++]; }
    else if (n !== -128) { const v = src[i++]; for (let k = 0; k < 1 - n && o < want; k++) out[o++] = v; }
  }
  return out;
}

function lzw(src, want) {
  const out = new Uint8Array(want);
  let o = 0, bitPos = 0, width = 9, prev = null;
  let dict = [];
  const reset = () => { dict = []; for (let i = 0; i < 256; i++) dict.push([i]); dict.push(null, null); width = 9; };
  reset();
  const read = () => {
    let v = 0;
    for (let k = 0; k < width; k++) {
      const byte = src[(bitPos + k) >> 3];
      if (byte === undefined) return 257;
      v = (v << 1) | ((byte >> (7 - ((bitPos + k) & 7))) & 1);
    }
    bitPos += width;
    return v;
  };
  const emit = seq => { for (const b of seq) if (o < want) out[o++] = b; };
  for (;;) {
    const code = read();
    if (code === 257) break;
    if (code === 256) { reset(); prev = null; continue; }
    let seq;
    if (code < dict.length && dict[code]) seq = dict[code];
    else if (prev) seq = prev.concat(prev[0]);
    else throw new Error('Corrupt LZW data');
    emit(seq);
    if (prev) dict.push(prev.concat(seq[0]));
    prev = seq;
    if (dict.length + 1 >= 1 << width && width < 12) width++;  // TIFF "early change"
  }
  return out;
}

function undoPredictor(data, w, bpp, le) {
  const out = new Uint8Array(data), rowBytes = w * bpp, rows = Math.floor(out.length / rowBytes);
  const dv = new DataView(out.buffer);
  for (let r = 0; r < rows; r++) {
    const o = r * rowBytes;
    if (bpp === 1) for (let x = 1; x < w; x++) out[o + x] = (out[o + x] + out[o + x - 1]) & 255;
    else if (bpp === 2) for (let x = 1; x < w; x++) dv.setUint16(o + 2 * x, (dv.getUint16(o + 2 * x, le) + dv.getUint16(o + 2 * x - 2, le)) & 65535, le);
    else if (bpp === 4) for (let x = 1; x < w; x++) dv.setUint32(o + 4 * x, (dv.getUint32(o + 4 * x, le) + dv.getUint32(o + 4 * x - 4, le)) >>> 0, le);
  }
  return out;
}

// Pixel size in µm from OME-XML (PhysicalSizeX, default unit µm) or an ImageJ description with unit=micron/um.
export function pixelSize(description, xres) {
  const ome = /PhysicalSizeX="([0-9.eE+-]+)"/.exec(description || '');
  if (ome) {
    const unit = /PhysicalSizeXUnit="([^"]+)"/.exec(description);
    const f = !unit || /^(µm|um|micron)$/i.test(unit[1]) ? 1 : unit[1] === 'nm' ? 1e-3 : null;
    if (f) return +ome[1] * f;
  }
  if (/ImageJ=/.test(description || '') && /unit=(micron|um|µm|\\u00B5m)/i.test(description) && xres > 0) return 1 / xres;
  return null;
}

export function writeTiff(frames, width, height, { bits = 8, umPerPx = null } = {}) {
  const n = frames.length, bpp = bits / 8, imgBytes = width * height * bpp;
  let desc = `ImageJ=1.11a\nimages=${n}\n${n > 1 ? `slices=${n}\n` : ''}${umPerPx ? 'unit=micron\n' : ''}\0`;
  const descBytes = new TextEncoder().encode(desc);
  const nTags = umPerPx ? 12 : 10, ifdSize = 2 + 12 * nTags + 4;
  const extra = descBytes.length + (umPerPx ? 16 : 0);
  const perFrame = ifdSize + extra + imgBytes, total = 8 + n * perFrame;
  const buf = new ArrayBuffer(total), dv = new DataView(buf), u8 = new Uint8Array(buf);
  u8[0] = 73; u8[1] = 73; dv.setUint16(2, 42, true); dv.setUint32(4, 8, true);
  const res = umPerPx ? [Math.round(1e6 / umPerPx), 1e6] : null;
  for (let f = 0; f < n; f++) {
    const ifd = 8 + f * perFrame, descAt = ifd + ifdSize, resAt = descAt + descBytes.length, dataAt = descAt + extra;
    const tags = [[256, 4, 1, width], [257, 4, 1, height], [258, 3, 1, bits], [259, 3, 1, 1], [262, 3, 1, 1],
      [270, 2, descBytes.length, descAt], [273, 4, 1, dataAt], [277, 3, 1, 1], [278, 4, 1, height], [279, 4, 1, imgBytes]];
    if (res) tags.push([282, 5, 1, resAt], [283, 5, 1, resAt + 8]);
    tags.sort((a, b) => a[0] - b[0]);
    dv.setUint16(ifd, tags.length, true);
    tags.forEach(([tag, type, count, val], i) => {
      const e = ifd + 2 + 12 * i;
      dv.setUint16(e, tag, true); dv.setUint16(e + 2, type, true); dv.setUint32(e + 4, count, true);
      if (type === 3) dv.setUint16(e + 8, val, true); else dv.setUint32(e + 8, val, true);
    });
    dv.setUint32(ifd + 2 + 12 * tags.length, f + 1 < n ? ifd + perFrame : 0, true);
    u8.set(descBytes, descAt);
    if (res) for (const k of [0, 8]) { dv.setUint32(resAt + k, res[0], true); dv.setUint32(resAt + k + 4, res[1], true); }
    const fr = frames[f];
    if (bits === 8) u8.set(fr, dataAt);
    else for (let i = 0; i < fr.length; i++) dv.setUint16(dataAt + 2 * i, fr[i], true);
  }
  return u8;
}
