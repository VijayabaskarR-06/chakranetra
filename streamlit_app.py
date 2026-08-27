"""
Chakranetra — public demo app (Streamlit Community Cloud)
=========================================================
Upload a road photo, watch it become an accountable municipal ticket.

This file is a THIN presentation layer. Every decision on screen — what counts
as a defect, whether two sightings are the same pothole, how severe it is, what
it costs, when it is due, who it routes to, whether a repair has failed — is
made by the modules in `roadlens/`. Nothing is re-implemented here, so the demo
cannot drift away from the code the tests cover.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timezone

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from roadlens.dedup import cluster_detections, haversine_m      # noqa: E402
from roadlens.geo import attach_gps_manual                      # noqa: E402
from roadlens.predictive import PredictiveEngine                # noqa: E402
from roadlens.tickets import TicketStore                        # noqa: E402

SAMPLES_DIR = os.path.join(ROOT, "data", "samples")
# Streamlit Cloud's filesystem is ephemeral; a fresh container starts with an
# empty queue, which is honest for a public demo.
DB_PATH = os.path.join(tempfile.gettempdir(), "chakranetra_demo.db")

# Bengaluru Outer Ring Road, the stretch the sample trip was shot on.
DEFAULT_LAT, DEFAULT_LON = 12.9516, 77.6995

SEV_COLOR = {4: "#FF4A3A", 3: "#FF9330", 2: "#F5C84C", 1: "#6FA8DC"}

st.set_page_config(page_title="Chakranetra", page_icon="🛣️", layout="wide")


# ---------------------------------------------------------------------------
# Expensive, process-wide singletons
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading the pothole segmentation model…")
def get_detector():
    from roadlens.detector import RoadDefectDetector
    from roadlens.config import get_config

    return RoadDefectDetector(confidence=get_config().detector.confidence_threshold)


@st.cache_resource
def get_store() -> TicketStore:
    return TicketStore(DB_PATH)


@st.cache_resource
def get_predictive() -> PredictiveEngine:
    return PredictiveEngine(DB_PATH)


# ---------------------------------------------------------------------------
# The pipeline — identical to the one the FastAPI endpoint runs
# ---------------------------------------------------------------------------
def scan(image_bytes: bytes, lat: float, lon: float, source_name: str) -> dict:
    store, predictive = get_store(), get_predictive()

    tmp_path = annotated_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        annotated_path = tmp_path.replace(".jpg", "_annotated.jpg")

        detections = get_detector().detect_image(tmp_path, save_annotated_to=annotated_path)
        for d in detections:
            d.source = source_name
        attach_gps_manual(detections, lat, lon)
        clusters = cluster_detections(detections)
        tickets = [store.create_from_cluster(c) for c in clusters]

        # Close the accountability loop: a defect at a spot that was recently
        # repaired is a failed repair, not merely a new pothole.
        recurrences = []
        claimed: set[str] = set()
        for cluster, ticket in zip(clusters, tickets):
            hit = predictive.check_recurrence_at_location(
                lat=cluster.lat, lon=cluster.lon, defect_type=cluster.defect_type,
                exclude_ticket_ids=claimed,
            )
            if hit:
                claimed.add(hit["original_ticket_id"])
                store.record_recurrence(hit["original_ticket_id"])
                store.update_status(
                    hit["original_ticket_id"], "REOPENED",
                    note=f"Defect detected again during scan; new ticket {ticket['id']}",
                )
                recurrences.append(hit)

        annotated = None
        if detections and os.path.exists(annotated_path):
            with open(annotated_path, "rb") as fh:
                annotated = fh.read()

        return {
            "detections": detections,
            "clusters": clusters,
            "tickets": tickets,
            "recurrences": recurrences,
            "annotated": annotated,
        }
    finally:
        for path in (tmp_path, annotated_path):
            if path and os.path.exists(path):
                try:
                    os.unlink(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def ticket_card(t: dict) -> None:
    colour = SEV_COLOR.get(t["severity_level"], "#8B96A3")
    due = datetime.fromisoformat(t["sla_due_at"])
    left = due - datetime.now(timezone.utc)
    hours = left.total_seconds() / 3600
    due_txt = f"{hours/24:.0f}d" if hours >= 48 else f"{hours:.0f}h"

    st.markdown(
        f"""<div style="border-left:4px solid {colour};background:#1F2630;
                    border-radius:7px;padding:12px 15px;margin-bottom:9px">
          <div style="font-family:monospace;font-size:14px">
            <b>{t['id']}</b>
            <span style="background:{colour};color:#14181d;border-radius:3px;
                         padding:1px 7px;font-size:11px;margin-left:8px">
              {t['severity_label']}</span>
            <span style="float:right;color:#8B96A3">{t['status']}</span>
          </div>
          <div style="color:#8B96A3;font-size:13px;margin-top:7px">
            priority <b style="color:#E9EDF2">{t['priority_score']}/100</b> ·
            est <b style="color:#E9EDF2">₹{t['est_cost_inr']:,}</b> ·
            due in <b style="color:#E9EDF2">{due_txt}</b> ·
            seen <b style="color:#E9EDF2">{t['sightings']}×</b>
          </div>
          <div style="color:#8B96A3;font-size:12px;margin-top:4px">
            routed to <b style="color:#E9EDF2">{t['department']}</b>
          </div>
        </div>""",
        unsafe_allow_html=True,
    )


def render_scan_result(result: dict, lat: float, lon: float) -> None:
    n_det, n_clu = len(result["detections"]), len(result["clusters"])

    if not n_det:
        st.warning(
            "No pothole found above the confidence threshold. The bundled model "
            "detects potholes only — try one of the sample photos in the sidebar."
        )
        return

    a, b, c = st.columns(3)
    a.metric("Raw detections", n_det)
    b.metric("Unique defects", n_clu, delta=f"−{n_det - n_clu} merged" if n_det > n_clu else None)
    c.metric("Tickets filed", len(result["tickets"]))

    if result["annotated"]:
        st.image(result["annotated"], caption="Segmentation masks — area drives the severity band",
                 width="stretch")

    for hit in result["recurrences"]:
        st.error(
            f"**Repair failure detected.** A defect has reappeared within "
            f"{get_predictive().recurrence_radius_m:.0f} m of ticket "
            f"`{hit['original_ticket_id']}`, which was marked FIXED. "
            f"Recurrence #{hit['recurrence_count']} — repair quality now "
            f"{hit['repair_quality_score']}. {hit['recommended_action']}"
        )

    st.markdown("#### Tickets created")
    for t in result["tickets"]:
        ticket_card(t)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.markdown(
    """<div style="border-bottom:1px solid #2A323C;margin-bottom:18px">
         <div style="height:6px;background:repeating-linear-gradient(-45deg,
              #F7B500 0 14px,#14181d 14px 28px);margin-bottom:14px"></div>
         <h1 style="font-size:30px;letter-spacing:.04em;margin:0 0 4px">
           CHAKRA<span style="color:#F7B500">NETRA</span></h1>
         <p style="color:#8B96A3;margin:0 0 14px">
           Every vehicle becomes a road inspector. Every defect becomes an
           accountable municipal ticket.</p>
       </div>""",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Try it")
    st.caption(
        "Upload any road photo, or scan one of the sample images from the "
        "pothole-segmentation dataset."
    )
    samples = sorted(f for f in os.listdir(SAMPLES_DIR)) if os.path.isdir(SAMPLES_DIR) else []
    chosen_sample = st.selectbox(
        "Sample photo", ["—"] + samples,
        format_func=lambda s: s if s == "—" else s.split("_jpg")[0],
    )
    st.divider()
    st.markdown("### Location")
    lat = st.number_input("Latitude", value=DEFAULT_LAT, format="%.6f")
    lon = st.number_input("Longitude", value=DEFAULT_LON, format="%.6f")
    st.caption(
        "Two scans within 12 m are treated as one physical defect. Move the "
        "coordinates a little and scan again to watch deduplication merge them."
    )
    st.divider()
    st.markdown(
        "[Operations console](https://vijayabaskarr-06.github.io/chakranetra/) · "
        "[Source](https://github.com/VijayabaskarR-06/chakranetra)"
    )

tab_scan, tab_queue, tab_predict = st.tabs(["Scan", "Work queue", "Predictive"])

with tab_scan:
    uploaded = st.file_uploader("Road photo", type=["jpg", "jpeg", "png"],
                                label_visibility="collapsed")
    go = st.button("Scan for defects", type="primary", width="stretch")

    if go:
        if uploaded is not None:
            payload, name = uploaded.getvalue(), uploaded.name
        elif chosen_sample != "—":
            with open(os.path.join(SAMPLES_DIR, chosen_sample), "rb") as fh:
                payload, name = fh.read(), chosen_sample
        else:
            payload = name = None
            st.warning("Upload a photo or pick a sample from the sidebar first.")

        if payload:
            with st.spinner("Running segmentation, deduplication and scoring…"):
                st.session_state["last"] = (scan(payload, lat, lon, name), lat, lon)

    if "last" in st.session_state:
        result, r_lat, r_lon = st.session_state["last"]
        render_scan_result(result, r_lat, r_lon)

with tab_queue:
    tickets = get_store().list()
    if not tickets:
        st.info("No tickets yet. Scan a photo on the first tab.")
    else:
        s = get_store().stats()
        cols = st.columns(5)
        cols[0].metric("Open", s["open"])
        cols[1].metric("Critical open", s["critical_open"])
        cols[2].metric("Past SLA", s["overdue_sla"])
        cols[3].metric("Backlog", f"₹{s['est_backlog_cost_inr']:,}")
        cols[4].metric("Recurrences", s["total_recurrences"])
        st.divider()
        for t in tickets:
            ticket_card(t)

with tab_predict:
    tickets = get_store().list()
    if not tickets:
        st.info("Predictive analysis needs tickets. Scan a photo on the first tab.")
    else:
        engine = get_predictive()
        alerts = engine.get_predictive_alerts(tickets)
        segments = engine.compute_risk_segments(tickets)
        crews = engine.get_crew_performance()

        st.markdown(f"#### Alerts ({len(alerts)})")
        if not alerts:
            st.success("No alerts. Every monitored repair is holding.")
        for a in alerts:
            body = f"**{a['type'].replace('_', ' ').title()}** — {a['recommendation']}"
            (st.error if a["severity"] == "critical" else st.warning)(body)

        st.markdown(f"#### Risk segments ({len(segments)})")
        for g in segments:
            st.progress(
                min(max(g.risk_score, 0.0), 1.0),
                text=f"{g.risk_label.upper()} · risk {g.risk_score} · "
                     f"{g.total_defects_ever} defect(s) · {g.recommended_action}",
            )

        if crews:
            st.markdown("#### Repair quality by crew")
            for c in crews:
                st.progress(
                    min(max(c["avg_quality_score"], 0.0), 1.0),
                    text=f"{c['crew']} · quality {c['avg_quality_score']} · "
                         f"{str(c['performance_label']).replace('_', ' ')}",
                )
