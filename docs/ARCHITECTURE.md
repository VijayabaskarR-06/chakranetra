# RoadLens AI — Architecture

## Data flow

```
  dashcam video / CCTV / citizen photo
              │
              ▼
   ┌─────────────────────┐    frame sampling (every Nth frame —
   │  detector.py        │    consecutive frames are near-identical,
   │  YOLOv8-seg model   │    so 2 fps is enough and 15× faster)
   └─────────┬───────────┘
             │  Detection {type, confidence, box, area_ratio, frame}
             ▼
   ┌─────────────────────┐    GPX track shares the video's timeline:
   │  geo.py             │    frame 450 @30fps = t=15s → interpolate
   │  GPS attach         │    vehicle position at t=15s
   └─────────┬───────────┘
             │  Detection + lat/lon
             ▼
   ┌─────────────────────┐    haversine clustering (≤12 m = same hole)
   │  dedup.py           │    + same-frame separation rule
   │  N sightings →      │    sightings make a defect STRONGER,
   │  M unique defects   │    never noisier
   └─────────┬───────────┘
             │  DefectCluster {worst size, best confidence, sightings}
             ▼
   ┌─────────────────────┐    plain, explainable arithmetic:
   │  severity.py        │    severity L1–L4, priority 0–100,
   │  assessment         │    ₹ cost, SLA hours, department route
   └─────────┬───────────┘
             ▼
   ┌─────────────────────┐    OPEN → ASSIGNED → IN_PROGRESS →
   │  tickets.py         │    FIXED → VERIFIED (AI re-scan confirms)
   │  SQLite registry    │              └→ REOPENED if still detected
   └─────────┬───────────┘
             │  a cluster matching an OPEN ticket within the merge
             │  radius appends to that ticket's growth history
             │  rather than filing a duplicate
             ├──────────────────────────────┐
             │                              ▼
             │                 ┌─────────────────────────────┐
             │                 │  roadlens/ml/               │
             │                 │  gradient-boosted trees     │
             │                 │  (NumPy, ~400 lines)        │
             │                 │                             │
             │                 │  cost        ₹ + interval   │
             │                 │  degradation days to L(n+1) │
             │                 │  failure     P(comes back)  │
             │                 │  budget      30/60/90 spend │
             │                 └─────────────┬───────────────┘
             │        each corrects severity.py's rule and falls
             │        back to it when it cannot beat it
             ▼                               ▼
   ┌─────────────────────────────────────────┐
   │  server/app.py (FastAPI)                │
   │  /api/scan/image  /api/scan/video       │
   │  /api/tickets  /api/stats  /api/ml/*    │
   └─────────┬───────────────────────────────┘
             ▼
   dashboard/index.html — map + prioritized work queue + SLA tracking
```

## Design decisions worth defending

**The model is a plug-in.** Nothing outside `detector.py` knows YOLO exists — the
rest of the system consumes plain `Detection` objects. Swapping in a multi-class
model (cracks, manholes, footpaths) or an entirely different architecture changes
one file.

**Dedup is geometric, not learned.** Same-defect merging uses GPS distance with an
explicit radius and an explicit same-frame rule. A learned re-identification model
would be more precise but unexplainable; for a system that directs public money,
we start explainable and add ML only where it earns its complexity.

**Severity is arithmetic on purpose.** A commissioner must be able to answer "why
is this ticket Critical?" in one sentence: "It fills 6% of the camera frame, the
model is 75% confident, and two vehicles reported it." Every threshold sits in one
table in `severity.py`.

**Evidence travels with the ticket.** Annotated frames are saved per scan and served
at `/evidence/…` — a crew never argues with a photo of the hole with the AI's own
outline drawn on it.

**SQLite → Postgres is an interface swap.** `TicketStore` is the only class that
touches the database.

**The learned models correct the rules; they never replace them.** `roadlens/ml/`
predicts the *residual* against `severity.py`, so an untrained model reproduces the
rules engine exactly and cold start needs no special case. Training refuses to
install a model that does not beat its baseline on held-out data, which means the
worst case for the ML layer is today's behaviour, not a silent regression. This is
the "add ML only where it earns its complexity" rule above, made mechanical.

**Image and video scans share one filing path.** `_file_and_track` in
`server/app.py` — cluster, match against an already-OPEN ticket or create
one, check recurrence only for the ones that were actually new — is called
by both `/api/scan/image` and `/api/scan/video`. A clip is just more frames
than a photo; the ticket-filing logic underneath does not need to know
which one produced a given detection.

**The learner is hand-written rather than imported.** scikit-learn cannot run in the
browser, and `dashboard/scan.js` has already promised that the console works with no
server. A tree ensemble serialises to JSON and evaluates in twenty lines of
JavaScript, so `tools/generate_ml_js.py` emits the same cost model client-side and
`tests/test_ml_js_parity.py` pins the two together at `1e-9` — the same standard
`tests/test_js_parity.py` holds the severity rules to.

**Provenance is a field, not a footnote.** No repository ships real repair invoices,
so the models train on a labelled synthetic corpus until a city records its own. That
fact travels as `training_data` through the model file, every API response, the
browser bundle and a badge on the dashboard. There is no setting that hides it.

## Where a second data source plugs in

The cost model's strongest feature is `road_class`, and today it is a ticket field
defaulting to `arterial`. Joining tickets against a municipal road-network layer
fills it from geometry with no model change — `roadlens/ml/features.py` already
carries the vocabulary, and an unseen class lands in an `__other__` column rather
than shifting every one-hot column to its right.

## Scaling path

1 vehicle → edge inference on-device (YOLOv8s runs on a phone/Jetson), tickets sync
over 4G. 1,000 vehicles → detections stream to a queue, dedup runs as a windowed
job per road segment, Postgres + PostGIS replaces SQLite, tiles pre-aggregate for
the dashboard. The module boundaries in this repo are drawn exactly where those
production seams are.
