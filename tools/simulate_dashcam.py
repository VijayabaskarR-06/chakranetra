"""
Chakranetra — Dashcam Simulator
================================
Makes the video pipeline demoable instead of merely describable: it plays
the part of a fleet vehicle that quietly uploads whatever its dashcam just
recorded, on its own schedule, with no one watching the terminal.

    python tools/simulate_dashcam.py

Every `--interval` seconds it builds a short clip out of real assets already
in this repo — a few of the pothole photos in `data/samples/` stitched into
an .mp4, tagged with a genuine slice of `data/demo_route.gpx` re-based to the
clip's own timeline — and POSTs it to `POST /api/scan/video`, then prints one
line describing what came back. Run `uvicorn server.app:app` in one terminal
and this in another, then watch the dashboard's work queue fill in while you
do nothing.

The route wraps: once the simulated vehicle reaches the end of the 59-second
demo track, the next clip starts again from the beginning — a second lap
over the same road. That is deliberate, not a loop bug. A second sighting of
a pothole already in the queue is exactly what `find_open_at_location`
(roadlens/tickets.py) is for: the clip grows the existing ticket's history
instead of filing a duplicate, which is the whole reason the degradation
model in `roadlens/ml/` can ever train on more than one sighting per defect.

No real footage exists in this repo to upload — data/samples/ holds still
photos, not video — so this is a labelled simulation of a device, not a
simulation of data: the coordinates are the demo route's real recorded GPS,
the photos are the real dataset images, only the "camera driving past and
holding on each frame for a couple of seconds" part is synthesised.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone

import cv2
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from roadlens.geo import GpxTrack   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT, "data", "samples")
ROUTE_GPX = os.path.join(ROOT, "data", "demo_route.gpx")

DEFAULT_FLEET = ["KA-01-AB-1234", "KA-05-CX-7788", "KA-03-EF-4521"]
FRAME_SIZE = (480, 270)          # small on purpose: fast to encode, fast to upload


def load_route() -> GpxTrack:
    return GpxTrack.load(ROUTE_GPX)


def sample_photos() -> list[str]:
    names = sorted(f for f in os.listdir(SAMPLES_DIR) if f.lower().endswith((".jpg", ".png")))
    if not names:
        raise SystemExit(f"No sample photos found in {SAMPLES_DIR}")
    return [os.path.join(SAMPLES_DIR, n) for n in names]


def next_window(cursor: float, window_seconds: float, duration: float) -> tuple[float, float, bool]:
    """Advance the route cursor by one clip's worth of driving.

    Returns (start, end, wrapped) — `wrapped` is True when this window starts
    a fresh lap over the route, which is the moment a clip is likely to
    re-see a pothole an earlier clip already ticketed.
    """
    wrapped = cursor >= duration
    start = 0.0 if wrapped else cursor
    end = min(start + window_seconds, duration)
    return start, end, wrapped


def photos_for_window(photos: list[str], slot: int, n: int) -> list[str]:
    """Deterministic photo selection for one clip, indexed by its position
    within a lap (not by upload count, which keeps growing across laps).

    The same slot always picks the same photos, so a second lap over the
    route reuses exactly the photos the first lap used at that stretch --
    the dashcam "seeing the same pothole again" reliably, rather than by
    chance. That is what turns a wrapped lap into a guaranteed ticket-growth
    event instead of a coin flip.
    """
    start_i = (slot * n) % len(photos)
    return [photos[(start_i + j) % len(photos)] for j in range(n)]


def build_sub_gpx(track: GpxTrack, start: float, end: float, points_per_second: float = 1.0) -> str:
    """A GPX document covering [start, end] of `track`, re-based to t=0.

    attach_gps_from_track() (roadlens/geo.py) interprets a video's frames as
    elapsed seconds *since the track's own first point*, so a sub-window of
    the real route has to be re-timestamped from zero, not sliced with the
    original elapsed times still attached — otherwise every position lookup
    would be seconds too far down the road.
    """
    n = max(2, int(round((end - start) * points_per_second)) + 1)
    base_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
            '<gpx version="1.1"><trk><trkseg>']
    for i in range(n):
        t = start + (end - start) * i / (n - 1)
        lat, lon = track.position_at(t)
        stamp = (base_time + timedelta(seconds=t - start)).isoformat().replace("+00:00", "Z")
        lines.append(f'<trkpt lat="{lat:.6f}" lon="{lon:.6f}"><time>{stamp}</time></trkpt>')
    lines.append("</trkseg></trk></gpx>")
    return "\n".join(lines)


def build_clip(photo_paths: list[str], out_path: str, fps: float, hold_frames: int) -> int:
    """Write photo_paths back-to-back into a video, each held for `hold_frames`
    frames -- a crude but real stand-in for "the vehicle driving past and the
    dashcam holding roughly on each spot for a couple of seconds". Returns the
    total frame count written."""
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, FRAME_SIZE)
    total = 0
    try:
        for path in photo_paths:
            img = cv2.imread(path)
            if img is None:
                continue
            frame = cv2.resize(img, FRAME_SIZE)
            for _ in range(hold_frames):
                writer.write(frame)
                total += 1
    finally:
        writer.release()
    return total


def upload_clip(api: str, video_path: str, gpx_xml: str, fps: float, vehicle_id: str,
                timeout: float = 60.0) -> dict:
    with open(video_path, "rb") as vfh:
        files = {
            "file": (os.path.basename(video_path), vfh, "video/mp4"),
            "gpx": ("track.gpx", gpx_xml.encode("utf-8"), "application/gpx+xml"),
        }
        data = {"fps": fps, "vehicle_id": vehicle_id}
        resp = requests.post(f"{api.rstrip('/')}/api/scan/video", files=files, data=data,
                             timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def describe_response(vehicle_id: str, filename: str, window_seconds: float,
                      wrapped: bool, body: dict) -> str:
    """One human-readable line for what a single upload did."""
    bits = []
    created = body.get("tickets_created") or []
    updated = body.get("tickets_updated") or []
    recurrences = body.get("recurrences") or []
    if created:
        bits.append(f"{len(created)} new ticket(s) {', '.join(created)}")
    if updated:
        bits.append(f"{len(updated)} grew {', '.join(updated)}")
    if recurrences:
        ids = ', '.join(r['original_ticket_id'] for r in recurrences)
        bits.append(f"{len(recurrences)} recurrence(s) reopened {ids}")
    if not bits:
        bits.append("no defects seen")

    lap_note = "  [second lap: re-scanning known road]" if wrapped else ""
    return (f"{vehicle_id:<14} uploaded {filename} "
            f"({window_seconds:.0f}s, {body.get('frames_analyzed')} frame(s) analyzed) "
            f"-> {'; '.join(bits)}{lap_note}")


def wait_for_server(api: str, attempts: int = 10, delay: float = 2.0) -> None:
    for i in range(attempts):
        try:
            requests.get(f"{api.rstrip('/')}/api/stats", timeout=3.0).raise_for_status()
            return
        except requests.RequestException:
            if i == 0:
                print(f"Waiting for {api} to come up "
                     f"(start it with: uvicorn server.app:app --reload)...")
            time.sleep(delay)
    raise SystemExit(f"Could not reach {api} after {attempts * delay:.0f}s. "
                     f"Is the server running?")


def run(args: argparse.Namespace) -> None:
    wait_for_server(args.api)

    track = load_route()
    duration = track.times[-1] - track.times[0]
    photos = sample_photos()

    rng = random.Random(args.seed)
    window_seconds = args.photos_per_clip * args.hold_seconds
    hold_frames = max(1, int(round(args.hold_seconds * args.fps)))
    cursor = 0.0
    # -1 so the first window's `lap_slot = ... + 1` lands on slot 0, not 1 --
    # off by one here silently shifted every lap's photo sequence by one
    # slot, so a wrapped lap's first clip (slot 0) never matched anything
    # the previous lap had actually visited.
    lap_slot = -1
    tmp_dir = os.path.join(ROOT, "output", "dashcam_sim")
    os.makedirs(tmp_dir, exist_ok=True)

    print(f"Simulating a dashcam fleet against {args.api} "
         f"({'forever' if args.count == 0 else f'{args.count} uploads'}, "
         f"every {args.interval:.0f}s)")
    print(f"Route: {duration:.0f}s of real GPS from {os.path.basename(ROUTE_GPX)}, "
         f"{len(photos)} sample photos, ctrl-C to stop.\n")

    i = 0
    while args.count == 0 or i < args.count:
        start, end, wrapped = next_window(cursor, window_seconds, duration)
        cursor = end
        lap_slot = 0 if wrapped else lap_slot + 1

        clip_photos = photos_for_window(photos, lap_slot, args.photos_per_clip)
        vehicle_id = args.fleet[i % len(args.fleet)] if args.fleet else rng.choice(DEFAULT_FLEET)
        filename = f"clip_{i:04d}.mp4"
        video_path = os.path.join(tmp_dir, filename)

        build_clip(clip_photos, video_path, fps=args.fps, hold_frames=hold_frames)
        gpx_xml = build_sub_gpx(track, start, end)

        stamp = datetime.now().strftime("%H:%M:%S")
        try:
            body = upload_clip(args.api, video_path, gpx_xml, fps=args.fps, vehicle_id=vehicle_id)
            print(f"[{stamp}] " + describe_response(vehicle_id, filename, end - start, wrapped, body))
        except requests.RequestException as e:
            print(f"[{stamp}] {vehicle_id:<14} upload of {filename} FAILED: {e}")
        finally:
            if os.path.exists(video_path) and not args.keep_clips:
                os.remove(video_path)

        i += 1
        if args.count == 0 or i < args.count:
            time.sleep(args.interval)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--api", default="http://127.0.0.1:8000", help="base URL of a running Chakranetra API")
    p.add_argument("--interval", type=float, default=15.0, help="seconds between uploads")
    p.add_argument("--count", type=int, default=0, help="number of uploads, 0 = run until Ctrl-C")
    p.add_argument("--fps", type=float, default=10.0, help="frame rate of the synthesised clips")
    p.add_argument("--photos-per-clip", type=int, default=3, dest="photos_per_clip")
    p.add_argument("--hold-seconds", type=float, default=2.5, dest="hold_seconds",
                   help="how long each photo is held, in simulated seconds")
    p.add_argument("--fleet", default=",".join(DEFAULT_FLEET),
                   help="comma-separated vehicle ids to cycle through")
    p.add_argument("--seed", type=int, default=None, help="for a reproducible run")
    p.add_argument("--keep-clips", action="store_true",
                   help="don't delete the synthesised .mp4 files after uploading")
    args = p.parse_args(argv)
    args.fleet = [v.strip() for v in args.fleet.split(",") if v.strip()]
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
