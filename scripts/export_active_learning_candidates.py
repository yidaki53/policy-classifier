#!/usr/bin/env python3
"""Export high-entropy / high-disagreement speech candidates for annotation.

Usage:
    uv run python3 scripts/export_active_learning_candidates.py --help

This script picks the latest `logs/speech_eval_preds_*.csv` (unless
`--preds` is provided), computes entropy over `prob_*` columns, and writes
`logs/active_learning_candidates_{ts}.csv` with a short text preview.
"""

from pathlib import Path
import argparse
from swedish_parliament_policy_classifier.orchestration.speech_pipeline import export_active_learning_candidates


def main():
    p = argparse.ArgumentParser(description="Export active-learning candidates from speech predictions")
    p.add_argument("--preds", help="Path to predictions CSV (optional)")
    p.add_argument("--top", type=int, default=500, help="Number of candidates to export")
    p.add_argument("--out", help="Optional output path")
    args = p.parse_args()

    out = export_active_learning_candidates(preds_csv=args.preds, top_n=args.top, out_path=args.out)
    print("Wrote:", out)


if __name__ == "__main__":
    main()
