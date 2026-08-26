#!/usr/bin/env python3
"""Build a validated publication result bundle from canonical analysis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from swedish_parliament_policy_classifier.analysis.publication_workflow import (
    build_publication_result_bundle,
    load_publication_contract_bundle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a validated action-first publication result bundle")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-dir", default="output/publication_result_bundle")
    parser.add_argument("--title", default="Action-first publication result bundle")
    parser.add_argument("--action-evidence", default="output/analysis/party_decision_choices.parquet")
    parser.add_argument("--party-position", default="output/analysis/party_supported_action_positions.parquet")
    parser.add_argument("--say-do", default="output/analysis/say_do_transitions.parquet")
    parser.add_argument("--evaluation", required=True, help="Prediction parquet with truth, pred, and optional prob_* columns")
    parser.add_argument("--data-cutoff", help="Optional ISO-8601 cutoff; otherwise infer it from action evidence")
    parser.add_argument("--dashboard")
    parser.add_argument("--public-table", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.root).resolve()
    contract_bundle = load_publication_contract_bundle(
        action_evidence_path=root / args.action_evidence,
        party_position_path=root / args.party_position,
        say_do_path=root / args.say_do,
        evaluation_path=root / args.evaluation,
        data_cutoff=args.data_cutoff,
    )
    manifest = build_publication_result_bundle(
        root=root,
        output_dir=root / args.output_dir,
        contract_bundle=contract_bundle,
        title=args.title,
        dashboard_path=(root / args.dashboard) if args.dashboard else None,
        public_table_paths=[root / table for table in args.public_table],
    )
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
