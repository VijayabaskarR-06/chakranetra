"""
RoadLens AI — One-Command Demo
==============================
    python run_demo.py

What it does, end to end, on your CPU, with no API keys:

  1. Runs the pothole segmentation model on the real road photos in
     data/samples/, treating them as frames of a dashcam trip along
     Bengaluru's Outer Ring Road (GPS track: data/demo_route.gpx).
  2. Simulates a SECOND vehicle passing the same stretch (re-scanning the
     first few frames) to demonstrate the deduplication engine: the same
     physical pothole seen by two vehicles becomes ONE stronger ticket,
     not two duplicate tickets.
  3. Scores every unique defect (severity, priority, cost, SLA) and files
     municipal tickets in SQLite.
  4. Walks a few tickets through the workflow (ASSIGNED -> IN_PROGRESS ->
     FIXED) so the dashboard shows a living system, not an empty shell.
  5. Demonstrates the predictive recurrence prevention engine by simulating
     a defect reappearing after repair.
  6. Saves annotated evidence images to output/ and exports the data for
     the dashboard.

Then open dashboard/index.html in a browser (works standalone), or run
`uvicorn server.app:app` for the full live API + dashboard.
"""

import json
import os
import shutil
import sys
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from roadlens.detector import RoadDefectDetector
from roadlens.geo import GpxTrack, attach_gps_from_track
from roadlens.dedup import cluster_detections
from roadlens.tickets import TicketStore
from roadlens.predictive import PredictiveEngine
from roadlens.logger import get_logger

logger = get_logger("demo")

SAMPLES = os.path.join(ROOT, "data", "samples")
GPX = os.path.join(ROOT, "data", "demo_route.gpx")
OUTPUT = os.path.join(ROOT, "output")
DB = os.path.join(ROOT, "roadlens.db")
DASH_DATA = os.path.join(ROOT, "dashboard", "demo_data.js")

FPS = 30


def main():
    print("=" * 62)
    print("  Chakranetra — demo run (v2.0 with Predictive Engine)")
    print("=" * 62)

    if os.path.exists(DB):
        os.remove(DB)
    os.makedirs(OUTPUT, exist_ok=True)

    print("\n[1/6] Loading model + scanning Trip 1 (vehicle KA-01-AB-1234)...")
    detector = RoadDefectDetector()

    images = sorted(
        f for f in os.listdir(SAMPLES) if f.lower().endswith((".jpg", ".png"))
    )
    detections = []
    for i, name in enumerate(images):
        frame_index = i * 45
        dets = detector.detect_image(
            os.path.join(SAMPLES, name),
            save_annotated_to=os.path.join(OUTPUT, f"trip1_{name}"),
        )
        for d in dets:
            d.frame_index = frame_index
            d.source = f"trip1/{name}"
        detections.extend(dets)
        print(f"   frame {frame_index:4d}  {name[:28]:30s} -> {len(dets)} defect(s)")

    print("\n[2/6] Attaching GPS from data/demo_route.gpx ...")
    track = GpxTrack.load(GPX)
    attach_gps_from_track(detections, track, fps=FPS)

    print("\n[3/6] Simulating Trip 2 over the same stretch (dedup test)...")
    trip2 = []
    for i, name in enumerate(images[:4]):
        frame_index = i * 45 + 6
        dets = detector.detect_image(os.path.join(SAMPLES, name))
        for d in dets:
            d.frame_index = frame_index
            d.source = f"trip2/{name}"
        trip2.extend(dets)
    attach_gps_from_track(trip2, track, fps=FPS)
    detections.extend(trip2)

    print(f"\n[4/6] Deduplicating {len(detections)} raw sightings ...")
    clusters = cluster_detections(detections)
    print(f"   {len(detections)} sightings -> {len(clusters)} unique defects")

    store = TicketStore(DB)
    predictive = PredictiveEngine(DB)
    tickets = [store.create_from_cluster(c) for c in clusters]

    if len(tickets) >= 1:
        store.update_status(tickets[0]["id"], "ASSIGNED",
                            note="Auto-routed to Ward 84 crew", assigned_to="Site Engineer R. Kumar")
    if len(tickets) >= 2:
        store.update_status(tickets[1]["id"], "ASSIGNED", assigned_to="Site Engineer P. Shetty")
        store.update_status(tickets[1]["id"], "IN_PROGRESS", note="Crew on site, cold-mix patching")
    if len(tickets) >= 3:
        store.update_status(tickets[2]["id"], "ASSIGNED", assigned_to="Site Engineer A. Farah")
        store.update_status(tickets[2]["id"], "IN_PROGRESS")
        store.update_status(tickets[2]["id"], "FIXED", note="Patched; awaiting AI re-scan verification")

    print("\n[5/6] Demonstrating Predictive Recurrence Prevention...")
    if len(tickets) >= 3:
        ticket = tickets[2]
        predictive.register_fixed_ticket(
            ticket_id=ticket["id"],
            lat=ticket["lat"],
            lon=ticket["lon"],
            defect_type=ticket["defect_type"],
            assigned_crew="Site Engineer A. Farah",
        )
        # Two later vehicle passes both still see a defect at the repaired
        # spot. That is the escalation the feature exists for: one recurrence
        # is bad luck, two means the patch itself was the problem, so the
        # engine raises a repair-quality alert against the crew.
        recurrence = None
        for pass_no in (1, 2):
            recurrence = predictive.check_recurrence_at_location(
                lat=ticket["lat"] + 0.00001 * pass_no,
                lon=ticket["lon"] + 0.00001 * pass_no,
                defect_type=ticket["defect_type"],
            )
            if recurrence:
                store.record_recurrence(ticket["id"])
                print(f"   Pass {pass_no}: defect still present at {ticket['id']} "
                      f"-> recurrence #{recurrence['recurrence_count']}, "
                      f"quality now {recurrence['repair_quality_score']}")
        if recurrence:
            store.update_status(ticket["id"], "REOPENED",
                                note="Defect detected again by a later vehicle pass")
            print(f"   Severity: {recurrence['severity']}")
            print(f"   Recommended action: {recurrence['recommended_action']}")
        else:
            print("   No recurrence detected")

    risk_segments = predictive.compute_risk_segments(store.list())
    print(f"   Risk segments identified: {len(risk_segments)}")
    if risk_segments:
        top = risk_segments[0]
        print(f"   Highest risk segment: score={top.risk_score}, label={top.risk_label}")

    alerts = predictive.get_predictive_alerts(store.list())
    print(f"   Predictive alerts generated: {len(alerts)}")

    crew_perf_export = predictive.get_crew_performance()

    all_tickets = store.list()
    stats = store.stats()

    # Export everything the console renders, all of it computed by the real
    # engines above — the dashboard never re-implements a rule in JavaScript,
    # it only displays what Python decided.
    from dataclasses import asdict

    bundle = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tickets": all_tickets,
        "stats": stats,
        "alerts": alerts,
        "segments": [asdict(s) for s in risk_segments],
        "crews": crew_perf_export,
        "heatmap": predictive.get_heatmap_data(all_tickets),
    }

    with open(DASH_DATA, "w") as f:
        f.write("// Auto-generated by run_demo.py — lets dashboard/index.html work\n")
        f.write("// without the API server running. Every number here was produced\n")
        f.write("// by the Python engines in roadlens/, not computed in the browser.\n")
        f.write("window.CHAKRANETRA_DEMO = ")
        json.dump(bundle, f, indent=1)
        f.write(";\n")

    print("\n[6/6] Tickets filed:")
    print(f"   {'ID':<18} {'Severity':<9} {'Priority':<9} {'Est. cost':<12} Status")
    for t in all_tickets:
        print(f"   {t['id']:<18} {t['severity_label']:<9} {t['priority_score']:<9} "
              f"\u20b9{t['est_cost_inr']:<11,} {t['status']}")

    print("\nStats:", json.dumps(stats, indent=2))

    if crew_perf_export:
        print("\nCrew Performance:")
        for c in crew_perf_export:
            print(f"   {c['crew']}: quality={c['avg_quality_score']}, label={c['performance_label']}")

    print("\nDone.")
    print("  Evidence images:  output/")
    print("  Dashboard:        open dashboard/index.html")
    print("  Live API:         uvicorn server.app:app  ->  http://127.0.0.1:8000")
    print("  API Docs:         http://127.0.0.1:8000/docs")
    print("  Predictive API:   http://127.0.0.1:8000/api/predictive/alerts")


if __name__ == "__main__":
    main()
