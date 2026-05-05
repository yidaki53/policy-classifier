"""Aggregation utilities: compute per-party profiles from classifications."""

import json
from datetime import datetime
from typing import Dict

from swedish_parliament_policy_classifier.classifier.scorer import load_definitions


def compute_party_profiles(conn) -> Dict[str, Dict[str, float]]:
    """Compute normalized category distributions per party and persist them.

    Returns a mapping: {party: {category: proportion}}
    """
    cur = conn.cursor()
    defs = load_definitions()
    categories = list(defs.keys())

    cur.execute(
        """
    SELECT nm.party, c.category, SUM(c.normalized_weight) as s
    FROM classifications c
    JOIN normalized_motions nm ON c.motion_id = nm.id
    GROUP BY nm.party, c.category
    """
    )

    rows = cur.fetchall()
    party_map: Dict[str, Dict[str, float]] = {}

    for row in rows:
        party = row[0]
        category = row[1]
        s = row[2] or 0.0
        if party not in party_map:
            party_map[party] = {cat: 0.0 for cat in categories}
        party_map[party][category] = float(s)

    # Normalize per party and persist
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
        out[row[0]] = json.loads(row[1])
    return out
