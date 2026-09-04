"""
Chakranetra — Synthetic Bootstrap Corpus
========================================
**Nothing in this file is real data.** It is a simulated city, generated from
the documented process below, and it exists for exactly one reason: so that a
fresh clone of this repository can demonstrate the learned models before any
municipality has entered a single invoice.

It is kept rigorously separate from the ticket database. `seed` never writes
to `roadlens.db`; it returns an in-memory corpus, and any model trained on it
is stamped `training_data: "synthetic_bootstrap"` — a label that travels into
the model JSON, the API response and the dashboard badge. The moment a city
has `MIN_COST_ROWS` real repairs on file, `roadlens.ml.registry` trains on
those instead and the label changes to `"observed"`.

The honest limitation, stated plainly: a model trained here can only
rediscover the process written below. It demonstrates that the machinery
works — that the trees recover a signal the rules engine misses, that the
conformal intervals cover at their stated rate, that the JavaScript port
agrees with Python. It demonstrates nothing whatsoever about Bengaluru's
actual road repair costs, and no figure produced from it should be quoted as
if it did.

The generative process
----------------------
Cost. The rules engine prices a repair from defect size alone. Real repair
cost depends at least as much on *where* the road is — a highway patch needs
lane closure, night working and a heavier mix — and on *when* it is done, as
monsoon work carries dewatering and tack-coat failures. So actual cost is the
rules estimate times a road-class factor, times a monsoon factor, times a
correction for the rules engine under-pricing deep cavities, times lognormal
noise. That structure is precisely what the cost model should recover, and
the road-class factor is what it can learn that the rule cannot express.

Growth. Defects grow exponentially, `area(t) = area(0)*exp(k*t)`, with `k`
rising on heavier road classes and roughly doubling through the monsoon.

Repair failure. Crews differ in workmanship; patches laid during the monsoon
fail far more often; larger defects patched shallow come back.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import numpy as np

from ..severity import assess
from .features import MONSOON_MONTHS, ROAD_CLASSES

# --- the process's parameters. Invented, and labelled as such. --------------

ROAD_CLASS_WEIGHTS = [0.12, 0.33, 0.25, 0.30]        # highway .. residential
ROAD_CLASS_COST_FACTOR = {"highway": 1.95, "arterial": 1.15,
                          "collector": 0.85, "residential": 0.62}
ROAD_CLASS_GROWTH = {"highway": 0.011, "arterial": 0.008,
                     "collector": 0.006, "residential": 0.004}

DEFECT_TYPES = ["pothole", "crack", "manhole", "footpath"]
DEFECT_WEIGHTS = [0.62, 0.22, 0.08, 0.08]

MONSOON_COST_FACTOR = 1.18
MONSOON_GROWTH_FACTOR = 2.1
DEEP_CAVITY_CORRECTION = 1.25      # the rules engine under-prices L4 work
COST_NOISE_SIGMA = 0.12            # lognormal, ~12% spread

CREWS = ["Crew-A", "Crew-B", "Crew-C", "Crew-D", "Crew-E"]
CREW_FAILURE_RATE = {"Crew-A": 0.06, "Crew-B": 0.11, "Crew-C": 0.18,
                     "Crew-D": 0.34, "Crew-E": 0.22}
MONSOON_FAILURE_ODDS = 2.6
DEPARTMENTS = {"pothole": "Road Maintenance Division",
               "crack": "Road Maintenance Division",
               "manhole": "Storm Water & Drainage",
               "footpath": "Footpath & Streetscape Wing"}

SYNTHETIC_PREFIX = "SYNTH-"


def seed(n_tickets: int = 600, seed: int = 20260904,
         now: datetime | None = None) -> dict:
    """Generate a synthetic corpus.

    Returns `{"tickets", "observations", "recurrence_records", "provenance"}`
    in the same shapes `TicketStore` and `PredictiveEngine` produce, so the
    training code has one input format and cannot accidentally treat
    synthetic rows as real ones — the ids are prefixed `SYNTH-`.
    """
    now = now or datetime.now(timezone.utc)
    rng = np.random.default_rng(seed)

    tickets, observations, recurrences = [], [], []

    for i in range(n_tickets):
        road_class = str(rng.choice(ROAD_CLASSES, p=ROAD_CLASS_WEIGHTS))
        defect_type = str(rng.choice(DEFECT_TYPES, p=DEFECT_WEIGHTS))

        # Sizes are lognormal: many small defects, a thin tail of big ones.
        area_ratio = float(np.clip(rng.lognormal(mean=math.log(0.018), sigma=0.85),
                                   0.001, 0.30))
        confidence = float(np.clip(rng.normal(0.72, 0.12), 0.35, 0.99))
        sightings = int(rng.integers(1, 5))

        created = now - timedelta(days=float(rng.uniform(20, 700)))
        a = assess(defect_type, area_ratio, confidence, sightings)

        monsoon = created.month in MONSOON_MONTHS
        factor = (ROAD_CLASS_COST_FACTOR[road_class]
                  * (MONSOON_COST_FACTOR if monsoon else 1.0)
                  * (DEEP_CAVITY_CORRECTION if a.severity_level == 4 else 1.0)
                  * math.exp(rng.normal(0.0, COST_NOISE_SIGMA)))
        actual_cost = int(round(a.est_cost_inr * factor))

        ticket_id = f"{SYNTHETIC_PREFIX}{i:05d}"
        repaired = created + timedelta(days=float(rng.uniform(2, 45)))
        # A slice stays open so the budget forecast has something to forecast.
        is_open = rng.random() < 0.22
        status = "OPEN" if is_open else "VERIFIED"

        tickets.append({
            "id": ticket_id,
            "defect_type": defect_type,
            "lat": 12.95 + float(rng.normal(0, 0.05)),
            "lon": 77.62 + float(rng.normal(0, 0.05)),
            "severity_level": a.severity_level,
            "severity_label": a.severity_label,
            "priority_score": a.priority_score,
            "est_cost_inr": a.est_cost_inr,
            "department": DEPARTMENTS[defect_type],
            "status": status,
            "sightings": sightings,
            "confidence": confidence,
            "area_ratio": area_ratio,
            "created_at": created.isoformat(),
            "sla_due_at": (now + timedelta(days=float(rng.uniform(-20, 120)))).isoformat()
                          if is_open else (created + timedelta(hours=a.sla_hours)).isoformat(),
            "assigned_to": None if is_open else str(rng.choice(CREWS)),
            "recurrence_count": 0,
            "repair_quality_score": 1.0,
            "road_class": road_class,
            "actual_cost_inr": None if is_open else actual_cost,
            "repaired_at": None if is_open else repaired.isoformat(),
            "synthetic": True,
        })

        # -- growth history ---------------------------------------------------
        k = (ROAD_CLASS_GROWTH[road_class]
             * (MONSOON_GROWTH_FACTOR if monsoon else 1.0)
             * (1.4 if defect_type == "pothole" else 0.7)
             * float(math.exp(rng.normal(0.0, 0.35))))
        n_obs = int(rng.integers(2, 6))
        t_obs, area_obs = created, area_ratio
        for j in range(n_obs):
            observations.append({
                "ticket_id": ticket_id,
                "observed_at": t_obs.isoformat(),
                # Sizing a defect from one frame is noisy; the model has to
                # find the growth signal underneath that noise, as it would
                # from real dashcam passes.
                "area_ratio": float(np.clip(area_obs * math.exp(rng.normal(0, 0.06)),
                                            1e-4, 0.6)),
                "confidence": confidence,
                "severity_level": assess(defect_type, area_obs, confidence, 1).severity_level,
                "source": f"pass-{j}",
            })
            gap = float(rng.uniform(7, 60))
            t_obs = t_obs + timedelta(days=gap)
            area_obs = area_obs * math.exp(k * gap)

        # -- repair outcome ---------------------------------------------------
        if not is_open:
            crew = tickets[-1]["assigned_to"]
            repaired_in_monsoon = repaired.month in MONSOON_MONTHS
            p = CREW_FAILURE_RATE[crew]
            odds = (p / (1 - p)) * (MONSOON_FAILURE_ODDS if repaired_in_monsoon else 1.0)
            odds *= 1.0 + 6.0 * area_ratio        # bigger holes come back
            p_fail = odds / (1 + odds)

            failed = bool(rng.random() < p_fail)
            count = int(rng.integers(1, 4)) if failed else 0
            recurrences.append({
                "original_ticket_id": ticket_id,
                "location_lat": tickets[-1]["lat"],
                "location_lon": tickets[-1]["lon"],
                "defect_type": defect_type,
                "first_fixed_at": repaired.isoformat(),
                "recurrence_count": count,
                "repair_quality_score": max(0.0, 1.0 - count * 0.25),
                "assigned_crew": crew,
            })
            tickets[-1]["recurrence_count"] = count

    return {
        "tickets": tickets,
        "observations": observations,
        "recurrence_records": recurrences,
        "provenance": {
            "training_data": "synthetic_bootstrap",
            "generator": "roadlens.ml.bootstrap.seed",
            "seed": seed,
            "n_tickets": n_tickets,
            "warning": "Simulated data. Demonstrates the modelling machinery, "
                       "not any real city's repair costs.",
        },
    }
