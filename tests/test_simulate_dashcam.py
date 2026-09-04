"""Unit tests for the pure/deterministic pieces of tools/simulate_dashcam.py.

Not a test of the network loop itself (that needs a running server and is
the thing you're meant to watch happen, not assert on) — this covers the
parts that are easy to get subtly wrong: window arithmetic, re-basing a GPX
slice to its own clip, and the response-to-log-line formatting a demo
audience actually reads.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))

cv2 = pytest.importorskip("cv2")

from roadlens.geo import GpxTrack
import simulate_dashcam as sim


@pytest.fixture(scope="module")
def route():
    return sim.load_route()


def test_route_and_photos_load_from_real_repo_assets(route):
    assert len(route.points) > 1
    photos = sim.sample_photos()
    assert len(photos) >= 1
    assert all(os.path.exists(p) for p in photos)


def test_next_window_advances_without_wrapping():
    start, end, wrapped = sim.next_window(cursor=10.0, window_seconds=8.0, duration=59.0)
    assert (start, end, wrapped) == (10.0, 18.0, False)


def test_next_window_clamps_to_route_end():
    start, end, wrapped = sim.next_window(cursor=55.0, window_seconds=8.0, duration=59.0)
    assert start == 55.0
    assert end == 59.0


def test_next_window_wraps_into_a_second_lap():
    start, end, wrapped = sim.next_window(cursor=59.0, window_seconds=8.0, duration=59.0)
    assert wrapped is True
    assert start == 0.0
    assert end == 8.0


def test_sub_gpx_is_re_based_to_zero_and_matches_the_route(route):
    """A sub-window from 20s-30s of the real route must parse back to the
    *same* positions the master track reports at those times — re-based to
    the clip's own t=0, or attach_gps_from_track would look up the wrong
    point for every frame."""
    xml = sim.build_sub_gpx(route, start=20.0, end=30.0)
    tmp = "/tmp/_sim_dashcam_test_sub.gpx"
    with open(tmp, "w") as fh:
        fh.write(xml)
    try:
        sub_track = GpxTrack.load(tmp)
    finally:
        os.remove(tmp)

    assert sub_track.times[0] == pytest.approx(0.0)
    assert sub_track.times[-1] == pytest.approx(10.0, abs=1e-6)

    want_start = route.position_at(20.0)
    want_end = route.position_at(30.0)
    assert sub_track.position_at(0.0) == pytest.approx(want_start, abs=1e-5)
    assert sub_track.position_at(10.0) == pytest.approx(want_end, abs=1e-5)


def test_photos_for_window_repeats_across_laps():
    """A wrapped lap resets to slot 0, and slot 0 must pick the same photos
    lap 1 picked at slot 0 -- that repetition is what guarantees a second
    lap re-sees (and grows) a ticket instead of maybe re-seeing one."""
    photos = [f"p{i}.jpg" for i in range(8)]
    lap1_slot0 = sim.photos_for_window(photos, slot=0, n=3)
    lap2_slot0 = sim.photos_for_window(photos, slot=0, n=3)
    assert lap1_slot0 == lap2_slot0 == ["p0.jpg", "p1.jpg", "p2.jpg"]

    lap1_slot2 = sim.photos_for_window(photos, slot=2, n=3)
    assert lap1_slot2 != lap1_slot0


def test_photos_for_window_wraps_within_the_photo_pool():
    photos = [f"p{i}.jpg" for i in range(4)]
    assert sim.photos_for_window(photos, slot=3, n=3) == ["p1.jpg", "p2.jpg", "p3.jpg"]


def test_build_clip_writes_the_expected_frame_count(tmp_path):
    photos = sim.sample_photos()[:2]
    out = str(tmp_path / "clip.mp4")
    total = sim.build_clip(photos, out, fps=10.0, hold_frames=5)
    assert total == 10
    cap = cv2.VideoCapture(out)
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    cap.release()


def test_describe_response_reports_new_tickets():
    line = sim.describe_response("KA-01-AB-1234", "clip_0001.mp4", 7.5, False, {
        "tickets_created": ["RL-POT-2026-0009"], "tickets_updated": [],
        "recurrences": [], "frames_analyzed": 2,
    })
    assert "1 new ticket(s) RL-POT-2026-0009" in line
    assert "KA-01-AB-1234" in line
    assert "second lap" not in line


def test_describe_response_reports_growth_and_marks_the_second_lap():
    line = sim.describe_response("KA-05-CX-7788", "clip_0012.mp4", 7.5, True, {
        "tickets_created": [], "tickets_updated": ["RL-POT-2026-0003"],
        "recurrences": [], "frames_analyzed": 2,
    })
    assert "grew RL-POT-2026-0003" in line
    assert "second lap" in line


def test_describe_response_reports_recurrences():
    line = sim.describe_response("KA-03-EF-4521", "clip_0020.mp4", 7.5, False, {
        "tickets_created": ["RL-POT-2026-0044"], "tickets_updated": [],
        "recurrences": [{"original_ticket_id": "RL-POT-2026-0002"}],
        "frames_analyzed": 2,
    })
    assert "recurrence(s) reopened RL-POT-2026-0002" in line


def test_describe_response_handles_a_quiet_clip():
    line = sim.describe_response("KA-01-AB-1234", "clip_0005.mp4", 7.5, False, {
        "tickets_created": [], "tickets_updated": [], "recurrences": [],
        "frames_analyzed": 2,
    })
    assert "no defects seen" in line


def test_lap_slot_sequence_repeats_identically_across_laps():
    """Regression test for an off-by-one that shifted every wrapped lap's
    photo sequence by one slot, so a fresh lap's first clip (slot 0) never
    matched anything the previous lap had actually driven past -- the
    'second lap' demo silently never grew a ticket. Replays run()'s own
    lap_slot bookkeeping against a short fake route and asserts the slot
    sequence is periodic with the window count."""
    duration = 24.0          # 3 windows per lap at window_seconds=8
    window_seconds = 8.0
    cursor = 0.0
    lap_slot = -1
    slots = []
    for _ in range(9):       # three full laps
        start, end, wrapped = sim.next_window(cursor, window_seconds, duration)
        cursor = end
        lap_slot = 0 if wrapped else lap_slot + 1
        slots.append(lap_slot)

    assert slots[:3] == [0, 1, 2]           # lap 1
    assert slots[3:6] == [0, 1, 2]          # lap 2 starts back at slot 0
    assert slots[6:9] == [0, 1, 2]          # lap 3, same again


def test_parse_args_splits_the_fleet_list():
    args = sim.parse_args(["--fleet", "KA-01-AB-1234, KA-02-XY-0001", "--count", "3"])
    assert args.fleet == ["KA-01-AB-1234", "KA-02-XY-0001"]
    assert args.count == 3
    assert args.interval == 15.0
