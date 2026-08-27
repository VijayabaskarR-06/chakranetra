"""
RoadLens AI — Geo Module
========================
A detection without a location is useless to a road crew. This module
attaches a GPS point to every detection.

Real dashcams record a GPS track alongside the video (a .gpx file, or GPS
data embedded in the video). The video and the track share a timeline, so:

  frame 450 of a 30 fps video  ->  15.0 seconds into the trip
  -> look up where the vehicle was at t = 15.0 s in the GPS track
  -> if it was between two recorded points, slide proportionally between
     them (linear interpolation)

That is all this file does: turn "frame number" into "lat/lon".
For single photos, we read the GPS from the photo's EXIF tags if present.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from bisect import bisect_left

from .detector import Detection


def _local(tag: str) -> str:
    """Strip any '{namespace}' prefix from an XML tag."""
    return tag.rsplit("}", 1)[-1]


def _iter_local(element, name: str):
    """Iterate descendants whose local tag name matches, any namespace."""
    for child in element.iter():
        if _local(child.tag) == name:
            yield child


class GpxTrack:
    """A GPS track loaded from a .gpx file: a list of (time_s, lat, lon)."""

    def __init__(self, points: list[tuple[float, float, float]]):
        # points must be sorted by time
        self.points = sorted(points)
        self.times = [p[0] for p in self.points]

    @classmethod
    def load(cls, gpx_path: str) -> "GpxTrack":
        """Read a .gpx file. We only need trkpt lat/lon and elapsed time.

        Tag matching ignores the namespace: hard-coding the GPX 1.1 URI made
        every GPX 1.0 file (and any file using a different namespace prefix)
        parse into an empty track, which only surfaced later as a confusing
        "Empty GPS track" from position_at().
        """
        from datetime import datetime

        root = ET.parse(gpx_path).getroot()
        pts = []
        t0 = None
        for i, trkpt in enumerate(_iter_local(root, "trkpt")):
            lat, lon = trkpt.get("lat"), trkpt.get("lon")
            if lat is None or lon is None:
                continue
            lat, lon = float(lat), float(lon)

            time_el = next(_iter_local(trkpt, "time"), None)
            if time_el is not None and time_el.text:
                # ISO timestamps -> seconds since first point
                t = datetime.fromisoformat(time_el.text.strip().replace("Z", "+00:00")).timestamp()
                if t0 is None:
                    t0 = t
                pts.append((t - t0, lat, lon))
            else:
                # No timestamps: assume one point per second
                pts.append((float(i), lat, lon))

        if not pts:
            raise ValueError(f"No track points found in {gpx_path}")
        return cls(pts)

    def position_at(self, t_seconds: float) -> tuple[float, float]:
        """Where was the vehicle t seconds into the trip? (interpolated)"""
        if not self.points:
            raise ValueError("Empty GPS track")
        if t_seconds <= self.times[0]:
            return self.points[0][1], self.points[0][2]
        if t_seconds >= self.times[-1]:
            return self.points[-1][1], self.points[-1][2]

        i = bisect_left(self.times, t_seconds)
        t1, lat1, lon1 = self.points[i - 1]
        t2, lat2, lon2 = self.points[i]
        # How far between the two recorded points are we (0.0 – 1.0)?
        f = (t_seconds - t1) / (t2 - t1) if t2 > t1 else 0.0
        return lat1 + (lat2 - lat1) * f, lon1 + (lon2 - lon1) * f


def attach_gps_from_track(
    detections: list[Detection], track: GpxTrack, fps: float = 30.0
) -> list[Detection]:
    """Give every video detection a lat/lon using the trip's GPS track."""
    for det in detections:
        t = det.frame_index / fps
        det.lat, det.lon = track.position_at(t)
    return detections


def attach_gps_manual(detections: list[Detection], lat: float, lon: float) -> list[Detection]:
    """For single photos reported from a known spot (e.g. citizen app)."""
    for det in detections:
        det.lat, det.lon = lat, lon
    return detections
