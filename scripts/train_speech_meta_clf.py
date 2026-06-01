#!/usr/bin/env python3
"""Train a speech-only LightGBM meta-classifier from `speech_gold_labels`.

Usage examples:

    # Train without tuning (fast):
    uv run python3 scripts/train_speech_meta_clf.py --out models/speech_meta_clf.pkl

    # Train with randomized tuning (longer):
    uv run python3 scripts/train_speech_meta_clf.py --tune --n-iter 20 --out models/speech_meta_clf_tuned.pkl

The script uses the project's `orchestration.speech_pipeline` helper so
behaviour is reproducible programmatically.
"""

import argparse
from pathlib import Path
import logging

from swedish_parliament_policy_classifier.orchestration.runner import SpeechRunner

LOG = logging.getLogger("train_speech_meta_clf")


def main():
    p = argparse.ArgumentParser(description="Train speech-only meta-classifier")
    p.add_argument("--db", default="data/swedish_parliament.db", help="Path to SQLite DB")
    p.add_argument("--out", default="models/speech_meta_clf.pkl", help="Output model path")
    p.add_argument("--tune", action="store_true", help="Run hyperparameter tuning (requires scikit-learn + lightgbm)")
    p.add_argument("--n-iter", type=int, default=12, dest="n_iter", help="Number of RandomizedSearch iterations when tuning")
    p.add_argument("--quiet", action="store_true", help="Show less logging")
    p.add_argument("--no-auto-label", action="store_true", help="Do not auto-label missing speeches before training (opt-out). By default auto-label runs.")
    args = p.parse_args()

    if args.quiet:
        logging.basicConfig(level=logging.WARNING)
    else:
        logging.basicConfig(level=logging.INFO)

    runner = SpeechRunner(db_path=args.db)
    out = runner.train(out_path=args.out, tune=args.tune, n_iter=args.n_iter, auto_label_missing=not args.no_auto_label)
    print("Saved model to:", out)


if __name__ == "__main__":
    main()
