"""
RoadLens AI — Detector
======================
This module answers one question: "Where are the road defects in this frame?"

How it works (simple version):
  1. We load a YOLOv8 segmentation model that was trained on pothole images.
  2. We give it an image (or a frame pulled from a dashcam video).
  3. It returns, for each defect it finds:
       - a bounding box (where it is)
       - a pixel mask (its exact shape)
       - a confidence score (how sure the model is)
  4. We convert that into a plain Python dict called a "Detection".

Think of the model as a trained inspector who has looked at thousands of
photos of broken roads. We just hand it new photos and record its findings.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict

import cv2
import numpy as np

# The pretrained pothole segmentation model, downloaded once from
# Hugging Face and cached locally after that.
MODEL_REPO = "keremberke/yolov8s-pothole-segmentation"
MODEL_FILE = "best.pt"

# Only keep detections the model is at least this confident about.
DEFAULT_CONFIDENCE = 0.35


@dataclass
class Detection:
    """One road defect found in one frame."""

    defect_type: str          # e.g. "pothole"
    confidence: float         # 0.0 – 1.0, how sure the model is
    box: list                 # [x1, y1, x2, y2] pixel coordinates
    area_ratio: float         # defect area / whole frame area (0–1)
    frame_index: int = 0      # which video frame it came from (0 for photos)
    source: str = ""          # file the frame came from
    lat: float | None = None  # filled in later by geo.py
    lon: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class RoadDefectDetector:
    """Wraps the YOLOv8 model behind a simple .detect() interface.

    The rest of the system never talks to YOLO directly — it only sees
    Detection objects. That means we can swap in a bigger model trained on
    more defect classes (cracks, broken footpaths, faded zebra crossings)
    without changing any other file. The model is a plug-in.
    """

    def __init__(self, confidence: float = DEFAULT_CONFIDENCE, model_path: str | None = None):
        # Import here so the rest of the package can be used without
        # the heavy ML dependencies installed (e.g. on the dashboard side).
        from ultralytics import YOLO

        if model_path is None:
            from huggingface_hub import hf_hub_download
            model_path = hf_hub_download(MODEL_REPO, MODEL_FILE)

        self.model = YOLO(model_path)
        self.confidence = confidence

    # ------------------------------------------------------------------
    # Images
    # ------------------------------------------------------------------
    def detect_image(self, image_path: str, save_annotated_to: str | None = None) -> list[Detection]:
        """Run the model on one photo. Optionally save a copy with the
        detections drawn on it (for reports and the dashboard)."""
        results = self.model.predict(image_path, conf=self.confidence, verbose=False)
        r = results[0]

        detections = self._to_detections(r, source=os.path.basename(image_path))

        if save_annotated_to:
            annotated = self._draw(r)
            cv2.imwrite(save_annotated_to, annotated)

        return detections

    # ------------------------------------------------------------------
    # Video (dashcam / CCTV)
    # ------------------------------------------------------------------
    def detect_video(
        self,
        video_path: str,
        sample_every_n_frames: int = 15,
        save_annotated_dir: str | None = None,
    ) -> list[Detection]:
        """Run the model on a video.

        We do NOT analyse every frame — at 30 fps that would be wasteful
        because consecutive frames look almost identical. Sampling every
        15th frame (~2 frames per second) is plenty to catch a pothole a
        car drives past, and it makes the system ~15x faster.
        """
        cap = cv2.VideoCapture(video_path)
        all_detections: list[Detection] = []
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % sample_every_n_frames == 0:
                results = self.model.predict(frame, conf=self.confidence, verbose=False)
                r = results[0]
                dets = self._to_detections(
                    r, source=os.path.basename(video_path), frame_index=frame_index
                )
                all_detections.extend(dets)

                if dets and save_annotated_dir:
                    os.makedirs(save_annotated_dir, exist_ok=True)
                    out = os.path.join(save_annotated_dir, f"frame_{frame_index:06d}.jpg")
                    cv2.imwrite(out, self._draw(r))

            frame_index += 1

        cap.release()
        return all_detections

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _to_detections(self, result, source: str, frame_index: int = 0) -> list[Detection]:
        """Convert a raw YOLO result into our clean Detection objects."""
        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            return detections

        frame_h, frame_w = result.orig_shape
        frame_area = float(frame_h * frame_w)

        for i in range(len(result.boxes)):
            box = result.boxes.xyxy[i].tolist()
            conf = float(result.boxes.conf[i])
            cls_id = int(result.boxes.cls[i])
            name = result.names.get(cls_id, "defect")

            # Prefer the segmentation mask area (exact shape) over the box
            # area — a pothole is round-ish, a box always over-counts.
            if result.masks is not None and i < len(result.masks.data):
                mask = result.masks.data[i].cpu().numpy()
                # mask may be model-resolution; scale its area to frame size
                mask_area_ratio = float(mask.sum()) / float(mask.size)
                area_ratio = mask_area_ratio
            else:
                bw, bh = box[2] - box[0], box[3] - box[1]
                area_ratio = (bw * bh) / frame_area

            detections.append(
                Detection(
                    defect_type=name,
                    confidence=round(conf, 3),
                    box=[round(v, 1) for v in box],
                    area_ratio=round(area_ratio, 5),
                    frame_index=frame_index,
                    source=source,
                )
            )
        return detections

    def _draw(self, result) -> np.ndarray:
        """Draw masks + boxes + confidence on the frame for humans."""
        return result.plot(line_width=2)
