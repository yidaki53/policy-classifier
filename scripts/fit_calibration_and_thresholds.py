#!/usr/bin/env python3
"""Fit probability calibrators and adaptive thresholds on validation data.

Uses existing gold-label predictions to calibrate ensemble probabilities and
learn per-category fallback thresholds. Saves artifacts for production use.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from swedish_parliament_policy_classifier.exports import load_definitions
from swedish_parliament_policy_classifier.classifier.calibration import (
    AdaptiveThresholdManager,
    ProbabilityCalibrator,
)
from swedish_parliament_policy_classifier.runtime.resources import apply_cpu_throttle
from swedish_parliament_policy_classifier.runtime.experiment import ExperimentRun
from datetime import datetime, timezone


def main(
    preds_parquet: str = "logs/speech_eval_preds_*.parquet",
    output_dir: str = "models",
    target_fallback_rate: float = 0.15,
):
    """Fit calibrators and thresholds from prediction logs."""
    import glob
    throttle = apply_cpu_throttle(cpu_fraction=0.5)
    run = ExperimentRun.start(
        enabled=False,
        experiment_name="calibration-fitting",
        run_name=f"calib-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
    )

    # Load latest predictions
    pred_files = sorted(glob.glob(preds_parquet))
    if not pred_files:
        raise FileNotFoundError(f"No prediction files found matching {preds_parquet}")
    
    latest = pred_files[-1]
    print(f"Loading predictions from {latest}")
    df = pd.read_parquet(latest)
    
    required_cols = {"truth", "pred"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"Predictions file missing columns: {required_cols - set(df.columns)}")
    
    # Get category names
    defs = load_definitions()
    category_names = sorted(defs.keys())
    
    # Build probability matrix
    prob_cols = [f"prob_{c}" for c in category_names]
    missing_cols = [c for c in prob_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing probability columns: {missing_cols}")
    
    probs = df[prob_cols].values.astype(np.float32)
    
    # Encode true labels
    le = LabelEncoder()
    le.fit(category_names)
    y_true = le.transform(df["truth"].values)
    
    print(f"Fitting calibrators on {len(y_true)} samples, {len(category_names)} categories...")
    
    # Fit calibrator
    calibrator = ProbabilityCalibrator(category_names)
    calibrated_probs = calibrator.fit_transform(y_true, probs)
    
    # Fit adaptive thresholds
    print(f"Learning adaptive thresholds (target fallback={target_fallback_rate:.2f})...")
    threshold_mgr = AdaptiveThresholdManager(
        category_names,
        default_threshold=0.30,
        min_threshold=0.15,
        max_threshold=0.50,
    )
    threshold_mgr.fit(y_true, probs, target_fallback_rate=target_fallback_rate)
    
    # Estimate fallback rates
    expected_fallback_raw = threshold_mgr.get_expected_fallback_rate(probs)
    expected_fallback_cal = threshold_mgr.get_expected_fallback_rate(calibrated_probs)
    
    print(f"\nExpected fallback rates:")
    print(f"  Raw probabilities: {expected_fallback_raw:.3f}")
    print(f"  Calibrated probabilities: {expected_fallback_cal:.3f}")
    
    # Save artifacts
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    calibrator_path = output_dir / "probability_calibrator.pkl"
    calibrator.save(calibrator_path)
    print(f"Saved calibrator to {calibrator_path}")
    
    threshold_path = output_dir / "adaptive_thresholds.json"
    threshold_mgr.save(threshold_path)
    print(f"Saved thresholds to {threshold_path}")
    
    # Save report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(y_true),
        "n_categories": len(category_names),
        "categories": category_names,
        "expected_fallback_raw": float(expected_fallback_raw),
        "expected_fallback_calibrated": float(expected_fallback_cal),
        "thresholds": threshold_mgr.thresholds,
        "calibrator_categories_fitted": len(calibrator.calibrators),
    }
    
    import json
    report_path = output_dir / "calibration_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved report to {report_path}")
    
    run.log_metrics({
        "expected_fallback_raw": float(expected_fallback_raw),
        "expected_fallback_calibrated": float(expected_fallback_cal),
        "n_categories_fitted": len(calibrator.calibrators),
    })
    run.end(status="FINISHED")
    
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit calibrators and thresholds")
    parser.add_argument("--preds", default="logs/speech_eval_preds_*.parquet", help="Glob for prediction files")
    parser.add_argument("--output-dir", default="models", help="Output directory for artifacts")
    parser.add_argument("--target-fallback-rate", type=float, default=0.15, help="Target fallback rate")
    args = parser.parse_args()
    
    exit(main(
        preds_parquet=args.preds,
        output_dir=args.output_dir,
        target_fallback_rate=args.target_fallback_rate,
    ))