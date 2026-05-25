"""Aggregation utilities: compute per-party profiles from classifications.

Copied from the repo-root `analysis/aggregate.py` into the package so the
implementation lives inside `swedish_parliament_policy_classifier` and
editable installs do not rely on ad-hoc repository layout.

Recency weighting
-----------------
Each motion receives a temporal decay weight:

    w(t) = exp(-\lambda · \Delta years)

where \Delta years is the age of the motion in years relative to the most
recent motion in the dataset, and \lambda (lambda) is a configurable decay
constant (default 0.1 — roughly a 10-year half-life).
"""

import json
import math
from datetime import datetime, date
from typing import Dict, Optional

from swedish_parliament_policy_classifier.exports import load_definitions

_RECENCY_LAMBDA_DEFAULT = 0.1  # decay constant (per year)


def _parse_date(date_str: Optional[str]) -> Optional[date]:
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y"):
        try:
            return datetime.strptime(date_str.strip()[:10], fmt).date()
        except ValueError:
            continue
    return None


def _recency_weight(motion_date: Optional[date], reference_date: date, lam: float) -> float:
    if motion_date is None or lam == 0.0:
        return 1.0
    delta_years = max((reference_date - motion_date).days / 365.25, 0.0)
    return math.exp(-lam * delta_years)


def compute_party_profiles(conn, recency_lambda: float = _RECENCY_LAMBDA_DEFAULT) -> Dict[str, Dict[str, float]]:
    """Compute normalized category distributions per party and persist them.

    Each classification's normalized_weight is multiplied by an exponential
    recency weight derived from the motion's date field.
    Returns a mapping: {party: {category: proportion}}
    """
    cur = conn.cursor()
    defs = load_definitions()
    categories = list(defs.keys())

    cur.execute("SELECT MAX(date) FROM normalized_motions WHERE date IS NOT NULL AND date != ''")
    max_row = cur.fetchone()
    max_date_str = max_row[0] if max_row else None
    reference_date: date = _parse_date(max_date_str) or date.today()

    cur.execute(
        """
    SELECT nm.party, c.category, c.normalized_weight, nm.date
    FROM classifications c
    JOIN normalized_motions nm ON c.motion_id = nm.id
    """
    )

    rows = cur.fetchall()
    party_map: Dict[str, Dict[str, float]] = {}

    def _is_valid_party(party_val) -> bool:
        if party_val is None:
            return False
        s = str(party_val).strip()
        if not s:
            return False
        if s.lower() in ("unknown", "nyd"):
            return False
        return True

    for row in rows:
        party = row[0]
        if not _is_valid_party(party):
            continue
        party = str(party).strip()
        category = row[1]
        weight = row[2] or 0.0
        motion_date = _parse_date(row[3])
        recency_w = _recency_weight(motion_date, reference_date, recency_lambda)

        if party not in party_map:
            party_map[party] = {cat: 0.0 for cat in categories}
        if category in party_map[party]:
            party_map[party][category] += float(weight) * recency_w

    for party, d in party_map.items():
        total = sum(d.values())
        if total > 0:
            for cat in categories:
                d[cat] = float(d.get(cat, 0.0) / total)
        else:
            for cat in categories:
                d[cat] = 0.0

        cur.execute(
            "INSERT OR REPLACE INTO party_profiles (party, profile_json, last_updated) VALUES (?, ?, ?)",
            (party, json.dumps(d, ensure_ascii=False), datetime.utcnow().isoformat()),
        )

    conn.commit()
    return party_map


def load_party_profiles(conn) -> Dict[str, Dict[str, float]]:
    cur = conn.cursor()
    cur.execute("SELECT party, profile_json FROM party_profiles")
    rows = cur.fetchall()
    out: Dict[str, Dict[str, float]] = {}
    for row in rows:
        p = row[0]
        if p is None:
            continue
        ps = str(p).strip()
        if not ps or ps.lower() in ("unknown", "nyd"):
            continue
        out[ps] = json.loads(row[1])
    return out
