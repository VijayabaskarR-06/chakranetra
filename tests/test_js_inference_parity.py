"""Prove the browser's YOLOv8-seg post-processing is an exact port.

The site runs inference client-side, so dashboard/scan.js re-implements the
mask pipeline in JavaScript. This dumps the raw ONNX tensors from Python, runs
BOTH implementations over them, and asserts identical boxes, confidences and —
most importantly — area_ratio, which is what decides the severity band.

Skipped when node or onnxruntime is unavailable.
"""

import glob
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCAN_JS = os.path.join(ROOT, "dashboard", "scan.js")

ort = pytest.importorskip("onnxruntime")
cv2 = pytest.importorskip("cv2")
np = pytest.importorskip("numpy")

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not os.path.exists(SCAN_JS),
    reason="node or scan.js unavailable",
)


def _model_path():
    p = os.path.join(ROOT, "dashboard", "model", "pothole-seg.onnx")
    return p if os.path.exists(p) else None


@pytest.mark.skipif(_model_path() is None, reason="ONNX model not present")
def test_js_postprocessing_matches_python_reference(tmp_path):
    from tools.onnx_reference import postprocess, preprocess

    sess = ort.InferenceSession(_model_path(), providers=["CPUExecutionProvider"])
    samples = sorted(glob.glob(os.path.join(ROOT, "data", "samples", "*.jpg")))
    assert samples, "no sample images"

    compared = 0
    for img_path in samples:
        rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        x, _, _, _ = preprocess(rgb)
        o0, o1 = sess.run(None, {sess.get_inputs()[0].name: x})

        want = postprocess(o0, o1, conf_thres=0.35, iou_thres=0.7)

        bin_path = tmp_path / "t.bin"
        with open(bin_path, "wb") as fh:
            fh.write(o0.astype(np.float32).tobytes())
            fh.write(o1.astype(np.float32).tobytes())

        mask_path = tmp_path / "masks.bin"
        script = f"""
import {{ postprocessRaw }} from {json.dumps(SCAN_JS)};
import {{ readFileSync, writeFileSync }} from "node:fs";
const buf = readFileSync({json.dumps(str(bin_path))});
const n0 = {o0.size}, n1 = {o1.size};
const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
const d0 = new Float32Array(ab, 0, n0);
const d1 = new Float32Array(ab, n0 * 4, n1);
const dets = postprocessRaw(d0, {json.dumps(list(o0.shape))}, d1, 0.35, 0.7);
writeFileSync({json.dumps(str(mask_path))}, Buffer.concat(dets.map(d => Buffer.from(d.mask))));
process.stdout.write(JSON.stringify(dets.map(d => ({{
  box: d.box, confidence: d.confidence, area_ratio: d.area_ratio,
}}))));
"""
        js_file = tmp_path / "run.mjs"
        js_file.write_text(script)
        res = subprocess.run(["node", str(js_file)], capture_output=True, text=True, timeout=300)
        assert res.returncode == 0, res.stderr
        got = json.loads(res.stdout)

        name = os.path.basename(img_path)[:24]
        assert len(got) == len(want), f"{name}: {len(got)} detections vs {len(want)}"

        want_s = sorted(want, key=lambda d: -d["confidence"])
        got_s = sorted(got, key=lambda d: -d["confidence"])
        for w, g in zip(want_s, got_s):
            assert abs(g["confidence"] - w["confidence"]) < 1e-5, name
            for a, b in zip(g["box"], w["box"]):
                assert abs(a - b) < 1e-3, f"{name}: box {a} vs {b}"
            # area_ratio picks the severity band, so it must match bit-for-bit
            assert abs(g["area_ratio"] - w["area_ratio"]) < 1e-9, (
                f"{name}: area_ratio {g['area_ratio']} vs {w['area_ratio']}")
            compared += 1

        # Equal area does not mean equal mask — two different shapes can cover
        # the same pixel count. Compare the masks themselves, pixel for pixel,
        # so the highlighted region is provably the region the model found.
        js_masks = np.fromfile(mask_path, dtype=np.uint8).reshape(len(want), 640, 640)
        order = np.argsort([-d["confidence"] for d in want])
        for slot, wi in enumerate(order):
            wm = want[wi]["mask"].astype(np.uint8)
            jm = js_masks[slot]
            differing = int(np.count_nonzero(wm ^ jm))
            union = int(np.count_nonzero(wm | jm))
            iou = (int(np.count_nonzero(wm & jm)) / union) if union else 1.0
            assert differing == 0, f"{name}: {differing} mask pixels differ (IoU {iou:.6f})"

    assert compared >= 7, f"only compared {compared} detections"
