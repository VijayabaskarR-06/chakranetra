# Chakranetra API Reference

Base URL (local): `http://127.0.0.1:8000`

Interactive Swagger docs are auto-generated at [`/docs`](http://127.0.0.1:8000/docs).
ReDoc is available at [`/redoc`](http://127.0.0.1:8000/redoc).

---

## Scan

### `POST /api/scan/image`

Upload a road photo to detect defects, deduplicate against existing tickets, and create or update a ticket.

**Request** — `multipart/form-data`

| Field      | Type   | Required | Description                         |
|------------|--------|----------|-------------------------------------|
| `file`     | binary | ✅       | Road photo (JPEG / PNG)             |
| `lat`      | float  | ✅       | Latitude of the scan location       |
| `lon`      | float  | ✅       | Longitude of the scan location      |
| `source`   | string | ❌       | Source identifier (e.g. `dashcam`, `citizen`) |

**Response** `200 OK`

```json
{
  "detections": [
    {
      "class": "pothole",
      "confidence": 0.82,
      "area_ratio": 0.073,
      "severity": "L4",
      "box": [120, 340, 290, 480]
    }
  ],
  "tickets_created": ["RL-POT-2026-0007"],
  "tickets_updated": ["RL-POT-2026-0003"],
  "evidence_url": "/evidence/scan_20260904_123045.jpg"
}
```

**Behaviour Notes:**
- If a detection matches an existing `OPEN`/`IN_PROGRESS` ticket within the dedup radius (12 m by default), that ticket's sighting count and priority are updated — **no duplicate ticket is created**.
- If a detection matches a `FIXED` ticket within the recurrence radius (15 m by default), the original ticket is moved to `REOPENED` and a recurrence event is logged against the assigned crew.

---

## Tickets

### `GET /api/tickets`

Retrieve the full work queue, sorted by priority (highest first).

**Query Parameters**

| Param        | Type   | Default  | Description                                |
|--------------|--------|----------|--------------------------------------------|
| `status`     | string | *(all)*  | Filter by status: `OPEN`, `ASSIGNED`, `IN_PROGRESS`, `FIXED`, `VERIFIED`, `REOPENED` |
| `department` | string | *(all)*  | Filter by department slug (e.g. `roads`)   |
| `severity`   | string | *(all)*  | Filter by severity level: `L1`, `L2`, `L3`, `L4` |

**Response** `200 OK` — Array of ticket objects (see schema below).

---

### `GET /api/tickets/{ticket_id}`

Retrieve a single ticket with its full sighting history and lifecycle events.

**Path Parameters**

| Param       | Type   | Description        |
|-------------|--------|--------------------|
| `ticket_id` | string | e.g. `RL-POT-2026-0003` |

**Response** `200 OK`

```json
{
  "id": "RL-POT-2026-0003",
  "status": "VERIFIED",
  "severity": "L4",
  "priority": 87,
  "lat": 12.9354,
  "lon": 77.6101,
  "address": "Outer Ring Road, Bengaluru",
  "sightings": 4,
  "estimated_cost": 18000,
  "sla_deadline": "2026-09-05T09:00:00Z",
  "department": "roads",
  "created_at": "2026-09-04T07:12:33Z",
  "updated_at": "2026-09-04T11:45:00Z",
  "history": [
    { "status": "OPEN",        "at": "2026-09-04T07:12:33Z" },
    { "status": "ASSIGNED",    "at": "2026-09-04T08:00:00Z" },
    { "status": "IN_PROGRESS", "at": "2026-09-04T09:30:00Z" },
    { "status": "FIXED",       "at": "2026-09-04T11:00:00Z" },
    { "status": "VERIFIED",    "at": "2026-09-04T11:45:00Z" }
  ],
  "recurrences": 0,
  "evidence_urls": ["/evidence/RL-POT-2026-0003_1.jpg"]
}
```

---

### `POST /api/tickets/{ticket_id}/status`

Move a ticket through the civic workflow.

**Request Body** `application/json`

```json
{
  "status": "ASSIGNED",
  "crew_id": "CREW-007",
  "note": "Assigned to North Zone crew"
}
```

**Valid Transitions**

```
OPEN → ASSIGNED → IN_PROGRESS → FIXED → VERIFIED
                                      ↘ REOPENED (triggered automatically by re-scan)
```

**Response** `200 OK` — Updated ticket object.

---

## Stats

### `GET /api/stats`

Commissioner-level dashboard numbers.

**Response** `200 OK`

```json
{
  "open": 14,
  "critical": 3,
  "past_sla": 2,
  "in_progress": 5,
  "fixed_today": 7,
  "total_backlog": 21,
  "recurrences_this_month": 4,
  "estimated_backlog_cost": 243000
}
```

---

## Predictive

### `GET /api/predictive/alerts`

Active predictive maintenance alerts — raised before citizens complain.

**Response** `200 OK` — Array of alert objects.

```json
[
  {
    "type": "recurrence",
    "severity": "high",
    "segment": "ORR-KM-14",
    "message": "3 recurrences detected. Full-depth repair recommended.",
    "recommended_action": "full_depth_repair",
    "estimated_preventive_cost": 45000,
    "raised_at": "2026-09-04T10:00:00Z"
  }
]
```

---

### `GET /api/predictive/heatmap`

Risk heatmap data for map visualization. Each cell represents a 0.5 km² grid square.

**Response** `200 OK`

```json
[
  {
    "lat": 12.934,
    "lon": 77.610,
    "risk_score": 0.87,
    "defect_count": 6,
    "recurrence_rate": 0.33,
    "recommendation": "preventive_overlay"
  }
]
```

---

### `GET /api/predictive/crews`

Crew performance scores based on repair quality tracking.

**Response** `200 OK`

```json
[
  {
    "crew_id": "CREW-007",
    "repairs_monitored": 12,
    "quality_score": 0.83,
    "rating": "Good",
    "recurrences_caused": 2
  }
]
```

---

### `GET /api/predictive/segments`

Computed risk segments — road stretches ranked by historical defect density.

**Response** `200 OK` — Array of segment objects with `risk_score`, `defect_count`, `length_m`, and geometry.

---

## Severity Reference

| Level | Label    | Frame Area | SLA    | Base Repair Cost |
|-------|----------|------------|--------|------------------|
| L4    | Critical | ≥ 6%       | 24 h   | ₹18,000          |
| L3    | High     | ≥ 2.5%     | 72 h   | ₹9,000           |
| L2    | Medium   | ≥ 0.8%     | 7 days | ₹4,500           |
| L1    | Low      | < 0.8%     | 14 days | ₹1,800          |

Priority score formula:
```
priority = size_score (0–60) + confidence_score (0–25) + sightings_score (0–15)
```

All thresholds are tunable via [`config.yaml`](../config.yaml).

---

## Error Responses

All endpoints return standard HTTP error codes:

| Code | Meaning                              |
|------|--------------------------------------|
| 400  | Bad request — missing or invalid field |
| 404  | Ticket or resource not found         |
| 422  | Validation error (Pydantic)          |
| 500  | Internal server error                |

Error body:
```json
{
  "detail": "Ticket RL-POT-2026-9999 not found"
}
```
