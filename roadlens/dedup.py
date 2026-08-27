"""
RoadLens AI — Deduplication Engine
==================================
THE problem nobody's demo handles, and the reason naive systems fail in the
real world:

  A dashcam at 30 fps sees the SAME pothole in ~40 consecutive frames.
  Tomorrow, three other vehicles drive the same road and see it again.
  A naive system raises 160 tickets for ONE hole. The municipal officer
  opens the dashboard once, sees garbage, and never opens it again.

RoadLens collapses all sightings of the same physical defect into ONE ticket
that gets STRONGER with every sighting (more sightings = higher trust and
priority), instead of noisier.

How (simple version):
  1. Every detection carries a GPS point (attached by geo.py).
  2. We measure the distance between a new detection and every open ticket
     of the same defect type, using the haversine formula (distance between
     two lat/lon points on Earth).
  3. Closer than MERGE_RADIUS_METERS?  -> same physical defect -> merge:
       - sighting count += 1
       - keep the LARGEST observed size (worst view is the truest view)
       - keep the HIGHEST confidence
  4. Farther? -> genuinely new defect -> new cluster.

This is intentionally simple, explainable clustering — not a black box.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .detector import Detection

# Two sightings within this many metres of each other are treated as the
# same physical defect. Consumer GPS is accurate to ~5 m, potholes cluster
# tightly, so 12 m absorbs GPS noise without merging separate holes on the
# same stretch too aggressively.
MERGE_RADIUS_METERS = 12.0


def _merge_radius() -> float:
    """Effective merge radius: config.yaml / ROADLENS_MERGE_RADIUS if set,
    otherwise the documented default above."""
    try:
        from .config import get_config

        return float(get_config().dedup.merge_radius_meters)
    except Exception:
        return MERGE_RADIUS_METERS


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in metres between two GPS points (standard formula)."""
    R = 6_371_000  # Earth's radius in metres
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


@dataclass
class DefectCluster:
    """One physical defect on the road, built from many sightings."""

    defect_type: str
    lat: float
    lon: float
    max_area_ratio: float     # worst (largest) observed size
    max_confidence: float     # best observed confidence
    sightings: int = 1
    sources: list = field(default_factory=list)  # which files saw it
    evidence_frames: list = field(default_factory=list)
    member_keys: set = field(default_factory=set)  # {(source, frame_index)}

    def absorb(self, det: Detection) -> None:
        """Merge one more sighting of the same defect into this cluster."""
        self.sightings += 1
        self.member_keys.add((det.source, det.frame_index))
        if det.frame_index not in self.evidence_frames:
            self.evidence_frames.append(det.frame_index)
        # Weighted-average the position toward the new fix (reduces GPS noise)
        w = 1.0 / self.sightings
        self.lat = self.lat * (1 - w) + (det.lat or self.lat) * w
        self.lon = self.lon * (1 - w) + (det.lon or self.lon) * w
        self.max_area_ratio = max(self.max_area_ratio, det.area_ratio)
        self.max_confidence = max(self.max_confidence, det.confidence)
        if det.source not in self.sources:
            self.sources.append(det.source)


def cluster_detections(
    detections: list[Detection], merge_radius_m: float | None = None
) -> list[DefectCluster]:
    """Collapse N raw detections into M unique physical defects (M << N)."""
    radius = _merge_radius() if merge_radius_m is None else float(merge_radius_m)
    clusters: list[DefectCluster] = []

    for det in detections:
        if det.lat is None or det.lon is None:
            # No GPS — treat each as unique (still works, just can't dedup)
            clusters.append(_new_cluster(det))
            continue

        merged = False
        for c in clusters:
            if c.defect_type != det.defect_type:
                continue
            # RULE: two detections in the SAME frame of the SAME file are
            # always different physical defects (the model reports each
            # defect in a frame exactly once). Without this rule, two
            # potholes sitting side by side would share one GPS point and
            # be wrongly merged into a single ticket.
            if (det.source, det.frame_index) in c.member_keys:
                continue
            if haversine_m(c.lat, c.lon, det.lat, det.lon) <= radius:
                c.absorb(det)
                merged = True
                break

        if not merged:
            clusters.append(_new_cluster(det))

    return clusters


def _new_cluster(det: Detection) -> DefectCluster:
    return DefectCluster(
        defect_type=det.defect_type,
        lat=det.lat if det.lat is not None else 0.0,
        lon=det.lon if det.lon is not None else 0.0,
        max_area_ratio=det.area_ratio,
        max_confidence=det.confidence,
        sightings=1,
        sources=[det.source],
        evidence_frames=[det.frame_index],
        member_keys={(det.source, det.frame_index)},
    )
