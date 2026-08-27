/**
 * In-browser YOLOv8-seg inference.
 *
 * A direct port of tools/onnx_reference.py, which is asserted equal to
 * ultralytics' own output by tools/check_onnx_parity.py (area_ratio delta
 * 0.000000 across the sample set). Keep the two in step.
 *
 * The order of operations is load-bearing:
 *   coef @ protos  -> RAW LOGITS (no sigmoid)
 *     -> zero outside the box, box rounded to whole cells in 160-space
 *     -> bilinear upsample 160 -> 640 (half-pixel centres, align_corners=false)
 *     -> keep where logit > 0
 * Applying sigmoid before upsampling looks equivalent and is not: sigmoid is
 * non-linear, and doing it early shifted mask areas by up to 0.0034 of the
 * frame — enough to move a defect across the 0.060 L4-Critical boundary.
 */

const INPUT = 640, PROTO = 160, NCOEF = 32;

let session = null;
let loadPromise = null;

export function modelBytes() { return 47_375_211; }

export async function loadModel(onProgress) {
  if (session) return session;
  if (loadPromise) return loadPromise;

  loadPromise = (async () => {
    const ort = globalThis.ort;
    if (!ort) throw new Error("onnxruntime-web failed to load");
    ort.env.wasm.numThreads = 1;      // no cross-origin isolation on Pages
    ort.env.wasm.simd = true;

    // Stream the weights so the UI can show real progress on a 45 MB file.
    const res = await fetch("model/pothole-seg.onnx");
    if (!res.ok) throw new Error(`model fetch failed: HTTP ${res.status}`);
    const total = Number(res.headers.get("content-length")) || modelBytes();
    const reader = res.body.getReader();
    const chunks = [];
    let received = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      onProgress?.(Math.min(received / total, 1));
    }
    const buf = new Uint8Array(received);
    let off = 0;
    for (const c of chunks) { buf.set(c, off); off += c.length; }

    session = await ort.InferenceSession.create(buf, {
      executionProviders: ["wasm"], graphOptimizationLevel: "all",
    });
    return session;
  })();

  return loadPromise;
}

/** Letterbox into 640x640 with grey 114 padding, preserving aspect ratio. */
function letterbox(imgEl) {
  const c = document.createElement("canvas");
  c.width = INPUT; c.height = INPUT;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, INPUT, INPUT);

  const w = imgEl.naturalWidth || imgEl.width;
  const h = imgEl.naturalHeight || imgEl.height;
  const scale = Math.min(INPUT / w, INPUT / h);
  const nw = Math.round(w * scale), nh = Math.round(h * scale);
  const dx = Math.floor((INPUT - nw) / 2), dy = Math.floor((INPUT - nh) / 2);
  ctx.drawImage(imgEl, dx, dy, nw, nh);
  return { canvas: c, ctx, scale, dx, dy, srcW: w, srcH: h };
}

function toTensor(ctx) {
  const { data } = ctx.getImageData(0, 0, INPUT, INPUT);
  const out = new Float32Array(3 * INPUT * INPUT);
  const plane = INPUT * INPUT;
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    out[p]             = data[i]     / 255;
    out[p + plane]     = data[i + 1] / 255;
    out[p + 2 * plane] = data[i + 2] / 255;
  }
  return out;
}

function iou(a, b) {
  const x1 = Math.max(a[0], b[0]), y1 = Math.max(a[1], b[1]);
  const x2 = Math.min(a[2], b[2]), y2 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  const aa = (a[2] - a[0]) * (a[3] - a[1]);
  const bb = (b[2] - b[0]) * (b[3] - b[1]);
  return inter / (aa + bb - inter + 1e-9);
}

function nms(boxes, scores, thr) {
  const order = scores.map((s, i) => i).sort((p, q) => scores[q] - scores[p]);
  const keep = [];
  const dead = new Set();
  for (const i of order) {
    if (dead.has(i)) continue;
    keep.push(i);
    for (const j of order) {
      if (j !== i && !dead.has(j) && iou(boxes[i], boxes[j]) > thr) dead.add(j);
    }
  }
  return keep;
}

/** Bilinear 160 -> 640, half-pixel centres — matches cv2 INTER_LINEAR and
 *  torch F.interpolate(align_corners=False). */
function upsample(src, srcN, dstN) {
  const dst = new Float32Array(dstN * dstN);
  const ratio = srcN / dstN;
  const map = new Float32Array(dstN), lo = new Int32Array(dstN), hi = new Int32Array(dstN);
  for (let d = 0; d < dstN; d++) {
    let s = (d + 0.5) * ratio - 0.5;
    if (s < 0) s = 0;
    const f = Math.floor(s);
    lo[d] = Math.min(Math.max(f, 0), srcN - 1);
    hi[d] = Math.min(lo[d] + 1, srcN - 1);
    map[d] = s - lo[d];
  }
  for (let y = 0; y < dstN; y++) {
    const y0 = lo[y] * srcN, y1 = hi[y] * srcN, wy = map[y], row = y * dstN;
    for (let x = 0; x < dstN; x++) {
      const x0 = lo[x], x1 = hi[x], wx = map[x];
      const a = src[y0 + x0], b = src[y0 + x1], c = src[y1 + x0], d2 = src[y1 + x1];
      dst[row + x] = a + (b - a) * wx + ((c + (d2 - c) * wx) - (a + (b - a) * wx)) * wy;
    }
  }
  return dst;
}

/**
 * Pure post-processing: raw ONNX outputs in, detections out. No DOM, so
 * tests/test_js_inference_parity.py can run it under node against tensors
 * dumped from Python and prove the port is exact.
 *
 * @param d0 Float32Array for output0, dims (1, 4+nc+32, anchors)
 * @param d1 Float32Array for output1, dims (1, 32, 160, 160)
 */
export function postprocessRaw(d0, dims0, d1, confThres = 0.35, iouThres = 0.7) {
  const ch = dims0[1], anchors = dims0[2];
  const nc = ch - 4 - NCOEF;

  const boxes = [], scores = [], coefIdx = [];
  for (let a = 0; a < anchors; a++) {
    let best = 0, bestC = 0;
    for (let c = 0; c < nc; c++) {
      const v = d0[(4 + c) * anchors + a];
      if (v > best) { best = v; bestC = c; }
    }
    if (best < confThres) continue;
    const cx = d0[a], cy = d0[anchors + a];
    const w = d0[2 * anchors + a], h = d0[3 * anchors + a];
    boxes.push([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]);
    scores.push(best);
    coefIdx.push({ a, cls: bestC });
  }
  if (!boxes.length) return [];

  const keep = nms(boxes, scores, iouThres);
  const detections = [];

  for (const k of keep) {
    const { a, cls } = coefIdx[k];
    // logits = coef · protos, at 160x160
    const logits = new Float32Array(PROTO * PROTO);
    for (let c = 0; c < NCOEF; c++) {
      const coef = d0[(4 + nc + c) * anchors + a];
      if (coef === 0) continue;
      const base = c * PROTO * PROTO;
      for (let p = 0; p < logits.length; p++) logits[p] += coef * d1[base + p];
    }

    // zero outside the box, box rounded to whole cells (ultralytics crop_mask)
    const r = PROTO / INPUT;
    const bx = boxes[k];
    const x1 = Math.round(bx[0] * r), y1 = Math.round(bx[1] * r);
    const x2 = Math.round(bx[2] * r), y2 = Math.round(bx[3] * r);
    for (let y = 0; y < PROTO; y++) {
      const inRow = y >= y1 && y < y2;
      for (let x = 0; x < PROTO; x++) {
        if (!inRow || x < x1 || x >= x2) logits[y * PROTO + x] = 0;
      }
    }

    const up = upsample(logits, PROTO, INPUT);
    let count = 0;
    const mask = new Uint8Array(INPUT * INPUT);
    for (let i = 0; i < up.length; i++) {
      if (up[i] > 0) { mask[i] = 1; count++; }
    }

    detections.push({
      box: bx,
      confidence: scores[k],
      cls,
      mask,
      area_ratio: count / (INPUT * INPUT),
    });
  }

  detections.sort((p, q) => q.confidence - p.confidence);
  return detections;
}

export async function detect(imgEl, confThres = 0.35, iouThres = 0.7) {
  const sess = await loadModel();
  const lb = letterbox(imgEl);
  const ort = globalThis.ort;

  const input = new ort.Tensor("float32", toTensor(lb.ctx), [1, 3, INPUT, INPUT]);
  const feeds = {}; feeds[sess.inputNames[0]] = input;
  const out = await sess.run(feeds);

  const o0 = out[sess.outputNames[0]], o1 = out[sess.outputNames[1]];
  const detections = postprocessRaw(o0.data, o0.dims, o1.data, confThres, iouThres);
  return { detections, letterbox: lb };
}

/** Draw masks + boxes onto a canvas sized to the letterboxed frame. */
export function renderOverlay(lb, detections) {
  const c = document.createElement("canvas");
  c.width = INPUT; c.height = INPUT;
  const ctx = c.getContext("2d");
  ctx.drawImage(lb.canvas, 0, 0);

  const img = ctx.getImageData(0, 0, INPUT, INPUT);
  const px = img.data;
  for (const d of detections) {
    for (let i = 0; i < d.mask.length; i++) {
      if (!d.mask[i]) continue;
      const o = i * 4;
      px[o]     = Math.round(px[o] * 0.45 + 56 * 0.55);
      px[o + 1] = Math.round(px[o + 1] * 0.45 + 82 * 0.55);
      px[o + 2] = Math.round(px[o + 2] * 0.45 + 255 * 0.55);
    }
  }
  ctx.putImageData(img, 0, 0);

  ctx.lineWidth = 2;
  ctx.strokeStyle = "#3852ff";
  ctx.font = "600 15px ui-monospace, Menlo, monospace";
  for (const d of detections) {
    const [x1, y1, x2, y2] = d.box;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const label = `pothole ${d.confidence.toFixed(2)}`;
    const w = ctx.measureText(label).width + 10;
    ctx.fillStyle = "#3852ff";
    ctx.fillRect(x1, Math.max(0, y1 - 20), w, 20);
    ctx.fillStyle = "#fff";
    ctx.fillText(label, x1 + 5, Math.max(14, y1 - 5));
  }
  return c;
}
