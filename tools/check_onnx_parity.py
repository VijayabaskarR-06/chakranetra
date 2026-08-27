"""Assert the ONNX + numpy pipeline reproduces ultralytics' own output."""
import glob, os, sys
import cv2, numpy as np, onnxruntime as ort
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.onnx_reference import preprocess, postprocess

CONF = 0.35
onnx_path = hf_hub_download("keremberke/yolov8s-pothole-segmentation", "best.pt").replace(".pt", ".onnx")
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
ref = YOLO(hf_hub_download("keremberke/yolov8s-pothole-segmentation", "best.pt"))

worst_area, worst_conf, total = 0.0, 0.0, 0
for p in sorted(glob.glob("data/samples/*.jpg")):
    bgr = cv2.imread(p)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    x, _, _, _ = preprocess(rgb)
    o0, o1 = sess.run(None, {sess.get_inputs()[0].name: x})
    got = postprocess(o0, o1, conf_thres=CONF)

    r = ref.predict(p, conf=CONF, verbose=False)[0]
    n_ref = 0 if r.boxes is None else len(r.boxes)
    assert len(got) == n_ref, f"{os.path.basename(p)}: {len(got)} vs ultralytics {n_ref}"
    if n_ref == 0:
        continue

    rc = sorted(r.boxes.conf.cpu().numpy().tolist(), reverse=True)
    gc = sorted([d["confidence"] for d in got], reverse=True)
    ra = sorted((r.masks.data.cpu().numpy().reshape(n_ref, -1).mean(1)).tolist(), reverse=True)
    ga = sorted([d["area_ratio"] for d in got], reverse=True)

    for a, b in zip(rc, gc): worst_conf = max(worst_conf, abs(a - b))
    for a, b in zip(ra, ga): worst_area = max(worst_area, abs(a - b))
    total += n_ref
    print(f"  {os.path.basename(p)[:24]:26s} n={n_ref}  "
          f"conf Δ={max(abs(a-b) for a,b in zip(rc,gc)):.5f}  "
          f"area Δ={max(abs(a-b) for a,b in zip(ra,ga)):.5f}")

print(f"\n{total} detections compared")
print(f"max confidence delta : {worst_conf:.6f}")
print(f"max area_ratio delta : {worst_area:.6f}")
assert worst_conf < 1e-3, "confidence drift"
assert worst_area < 2e-3, "area_ratio drift — severity banding would differ"
print("PARITY OK")
