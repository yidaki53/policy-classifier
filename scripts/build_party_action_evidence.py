#!/usr/bin/env python3
"""Materialize one auditable party-choice row per roll-call decision.

The command reads canonical Riksdag vote Parquet shards sequentially, delegates
vote semantics to ``analysis.action_evidence``, and writes a compressed Parquet
table suitable for later policy-referent mapping and party-position estimation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from swedish_parliament_policy_classifier.analysis.action_evidence import (
    aggregate_party_choices,
)


VOTE_COLUMNS = [
    "rm",
    "beteckning",
    "punkt",
    "votering_id",
    "datum",
    "parti",
    "rost",
]


def build_party_action_evidence(
    vote_dir: str | Path,
    out_path: str | Path,
    *,
    summary_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build and atomically persist party-decision evidence from vote shards."""

    source_dir = Path(vote_dir)
    files = sorted(source_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No vote Parquet shards found under {source_dir}")

    parts: list[pd.DataFrame] = []
    member_records_n = 0
    member_records_with_party_n = 0
    for path in files:
        shard = pd.read_parquet(path, columns=VOTE_COLUMNS)
        member_records_n += int(len(shard))
        party = shard["parti"].fillna("").astype(str).str.strip()
        member_records_with_party_n += int((party != "").sum())
        if not shard.empty:
            parts.append(aggregate_party_choices(shard))

    if not parts:
        raise ValueError("Vote Parquet shards contain no member vote records")

    out = pd.concat(parts, ignore_index=True)
    duplicate = out.duplicated(["decision_id", "party"], keep=False)
    if duplicate.any():
        examples = out.loc[duplicate, ["decision_id", "party"]].head(5).to_dict("records")
        raise ValueError(f"Party decisions span multiple shards or are duplicated: {examples}")
    out = out.sort_values(["decision_date", "decision_id", "party"]).reset_index(drop=True)

    destination = Path(out_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    out.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(destination)

    valid_dates = pd.to_datetime(out["decision_date"], errors="coerce", utc=True).dropna()
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "input_dir": str(source_dir),
        "input_files": [str(path) for path in files],
        "output": str(destination),
        "member_records_n": member_records_n,
        "member_records_with_party_n": member_records_with_party_n,
        "excluded_missing_party_n": member_records_n - member_records_with_party_n,
        "decision_n": int(out["decision_id"].nunique()),
        "party_decision_n": int(len(out)),
        "parties": sorted(out["party"].dropna().astype(str).unique().tolist()),
        "date_min": valid_dates.min().isoformat() if not valid_dates.empty else None,
        "date_max": valid_dates.max().isoformat() if not valid_dates.empty else None,
        "party_choice_counts": {
            str(key): int(value) for key, value in out["party_choice"].value_counts().sort_index().items()
        },
    }

    if summary_path is not None:
        summary_destination = Path(summary_path)
        summary_destination.parent.mkdir(parents=True, exist_ok=True)
        summary_destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Build party-level roll-call action evidence")
    parser.add_argument("--vote-dir", default="data/votering/parquet")
    parser.add_argument("--out", default="output/analysis/party_decision_choices.parquet")
    parser.add_argument(
        "--summary-out",
        default="output/analysis/party_decision_choices_summary.json",
    )
    args = parser.parse_args()

    summary = build_party_action_evidence(
        args.vote_dir,
        args.out,
        summary_path=args.summary_out,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())