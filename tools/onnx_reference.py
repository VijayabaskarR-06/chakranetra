"""
Reference YOLOv8-seg post-processing on raw ONNX outputs.

This exists to be ported to JavaScript verbatim. It is written in plain numpy
with no ultralytics helpers, so every step has a direct JS equivalent, and it
is checked against ultralytics' own output by tools/check_onnx_parity.py.

Pipeline, matching ultralytics `process_mask(..., upsample=True)` exactly:
    coef @ protos  (RAW LOGITS, no sigmoid)
      -> zero outside the box, box rounded to int in 160-space
      -> bilinear upsample to 640
      -> keep where logit > 0

Applying sigmoid before upsampling is the obvious-looking version and it is
wrong: sigmoid is non-linear, so interpolating probabilities then thresholding
at 0.5 does not equal interpolating logits then thresholding at 0. It shifted
mask areas by up to 0.0034 of the frame, enough to move a defect across the
0.060 L4-Critical boundary.
"""
from __future__ import annotations

import numpy as np

INPUT_SIZE = 640
PROTO_SIZE = 160
NUM_COEF = 32


def letterbox(img_rgb: np.ndarray, size: int = INPUT_SIZE):
    """Resize preserving aspect ratio, pad to square with grey (114)."""
    h, w = img_rgb.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    import cv2

    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def preprocess(img_rgb: np.ndarray):
    canvas, scale, dx, dy = letterbox(img_rgb)
    x = canvas.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None]  # NCHW
    return np.ascontiguousarray(x), scale, dx, dy


def _iou(a, boxes):
    xx1 = np.maximum(a[0], boxes[:, 0]); yy1 = np.maximum(a[1], boxes[:, 1])
    xx2 = np.minimum(a[2], boxes[:, 2]); yy2 = np.minimum(a[3], boxes[:, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (area_a + area_b - inter + 1e-9)


def nms(boxes, scores, iou_thres=0.7):
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        order = order[1:][_iou(boxes[i], boxes[order[1:]]) <= iou_thres]
    return keep


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def postprocess(out0, out1, conf_thres=0.35, iou_thres=0.7):
    """out0 (1,4+nc+32,8400), out1 (1,32,160,160) -> detections with masks."""
    pred = out0[0].T                       # (8400, 4+nc+32)
    nc = pred.shape[1] - 4 - NUM_COEF
    scores_all = pred[:, 4:4 + nc]
    cls = scores_all.argmax(1)
    conf = scores_all.max(1)

    keep0 = conf >= conf_thres
    if not keep0.any():
        return []
    pred, conf, cls = pred[keep0], conf[keep0], cls[keep0]

    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)

    idx = nms(boxes, conf, iou_thres)
    boxes, conf, cls = boxes[idx], conf[idx], cls[idx]
    coefs = pred[idx, 4 + nc:]

    protos = out1[0].reshape(NUM_COEF, -1)          # (32, 160*160)
    masks = (coefs @ protos).reshape(-1, PROTO_SIZE, PROTO_SIZE)   # logits

    import cv2

    ratio = PROTO_SIZE / INPUT_SIZE
    results = []
    for i in range(masks.shape[0]):
        m = masks[i].copy()
        # Zero outside the box, in 160-space, with the box rounded to whole
        # cells — this is what ultralytics' crop_mask does on CPU.
        x1, y1, x2, y2 = np.round(boxes[i] * ratio).astype(int)
        m[:max(y1, 0)] = 0
        m[max(y2, 0):] = 0
        m[:, :max(x1, 0)] = 0
        m[:, max(x2, 0):] = 0
        m = cv2.resize(m, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
        binary = (m > 0.0).astype(np.uint8)
        results.append({
            "box": boxes[i].tolist(),
            "confidence": float(conf[i]),
            "cls": int(cls[i]),
            "mask": binary,
            "area_ratio": float(binary.sum()) / float(binary.size),
        })
    return results
