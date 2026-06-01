#!/usr/bin/env python3
"""Run all new speech-side analysis modules as a bundle."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from swedish_parliament_policy_classifier.analysis.ideological_gap import run_ideological_gap_analysis
from swedish_parliament_policy_classifier.analysis.promise_fulfillment import run_promise_fulfillment_analysis
from swedish_parliament_policy_classifier.analysis.speech_visualizations import plot_speech_profiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Run speech analysis suite")
    parser.add_argument("--db", default="data/swedish_parliament.db", help="Path to SQLite database")
    parser.add_argument(
        "--speech-classifications",
        default="data/parquet/speech_classifications_with_rhetoric_full.parquet",
        help="Path to speech classification parquet",
    )
    parser.add_argument(
        "--speech-parquet-dir",
        default="data/speeches/parquet",
        help="Directory with speech parquet files",
    )
    parser.add_argument("--fig-out", default="figures/speeches", help="Speech figure output directory")
    parser.add_argument("--analysis-out", default="output/analysis", help="Analysis output directory")
    parser.add_argument("--cpu-fraction", type=float, default=0.25)
    parser.add_argument("--run-consistency", action="store_true")
    parser.add_argument("--run-recency", action="store_true")
    parser.add_argument("--run-sarimax", action="store_true")
    parser.add_argument("--run-all-linkage", action="store_true", help="Link all speeches to motion or vote-context targets")
    parser.add_argument("--run-axis", action="store_true", help="Compute canonical ideology-axis alignment scores")
    parser.add_argument("--run-contradiction", action="store_true", help="Compute contradiction edge and expected contradiction artifacts")
    parser.add_argument("--run-link-confidence", action="store_true", help="Compute linkage confidence strata artifacts")
    parser.add_argument("--run-uncertainty", action="store_true", help="Bootstrap say-vs-do uncertainty intervals")
    parser.add_argument("--run-link-stability", action="store_true", help="Analyze robustness across link-confidence strata")
    parser.add_argument("--run-latent", action="store_true", help="Fit latent party-ideology factor and intervals")
    parser.add_argument("--run-benchmark-validation", action="store_true", help="Validate latent/internal ideology scores against external benchmarks")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="speech-suite")
    parser.add_argument("--mlflow-tracking-uri", default=None)
    args = parser.parse_args()

    speech = plot_speech_profiles(
        speech_classifications_path=args.speech_classifications,
        speech_parquet_dir=args.speech_parquet_dir,
        out_dir=args.fig_out,
    )
    gap = run_ideological_gap_analysis(
        db_path=args.db,
        speech_classifications_path=args.speech_classifications,
        speech_parquet_dir=args.speech_parquet_dir,
        out_dir=args.analysis_out,
    )
    promise = run_promise_fulfillment_analysis(
        db_path=args.db,
        speech_classifications_path=args.speech_classifications,
        speech_parquet_dir=args.speech_parquet_dir,
        out_dir=args.analysis_out,
    )

    script_dir = Path(__file__).resolve().parent
    consistency_out = None
    recency_out = None
    all_linkage_out = None
    axis_out = None
    contradiction_out = None
    link_confidence_out = None
    uncertainty_out = None
    link_stability_out = None
    latent_out = None
    benchmark_validation_out = None
    env = dict(**os.environ)
    env["CLASSIFIER_CPU_FRACTION"] = str(args.cpu_fraction)

    if args.run_all_linkage:
        cmd = [
            sys.executable,
            str(script_dir / "link_all_speeches_to_action.py"),
            "--speech-classifications",
            args.speech_classifications,
            "--speech-parquet-dir",
            args.speech_parquet_dir,
            "--force",
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        all_linkage_out = proc.stdout

    if args.run_axis:
        cmd = [
            sys.executable,
            str(script_dir / "compute_ideology_axis_alignment.py"),
            "--speech-classifications",
            args.speech_classifications,
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        axis_out = proc.stdout

    if args.run_contradiction:
        cmd = [
            sys.executable,
            str(script_dir / "score_say_vs_do_contradiction.py"),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        contradiction_out = proc.stdout

    if args.run_link_confidence:
        cmd = [
            sys.executable,
            str(script_dir / "compute_link_confidence_strata.py"),
            "--links",
            "data/parquet/speech_action_links.parquet",
            "--out",
            str(Path(args.analysis_out) / "speech_action_link_confidence_strata.parquet"),
            "--summary-out",
            str(Path(args.analysis_out) / "speech_action_link_confidence_summary.json"),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        link_confidence_out = proc.stdout

    if args.run_uncertainty:
        cmd = [
            sys.executable,
            str(script_dir / "bootstrap_say_do_uncertainty.py"),
            "--axis-scores",
            str(Path(args.analysis_out) / "speech_action_axis_scores.parquet"),
            "--links",
            "data/parquet/speech_action_links.parquet",
            "--out",
            str(Path(args.analysis_out) / "say_do_uncertainty_intervals_party.parquet"),
            "--summary-out",
            str(Path(args.analysis_out) / "say_do_uncertainty_summary.json"),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        uncertainty_out = proc.stdout

    if args.run_link_stability:
        cmd = [
            sys.executable,
            str(script_dir / "analyze_link_strata_stability.py"),
            "--axis-scores",
            str(Path(args.analysis_out) / "speech_action_axis_scores.parquet"),
            "--link-strata",
            str(Path(args.analysis_out) / "speech_action_link_confidence_strata.parquet"),
            "--out",
            str(Path(args.analysis_out) / "link_strata_stability_party.parquet"),
            "--summary-out",
            str(Path(args.analysis_out) / "link_strata_stability_summary.json"),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        link_stability_out = proc.stdout

    if args.run_latent:
        cmd = [
            sys.executable,
            str(script_dir / "fit_latent_party_ideology.py"),
            "--axis-scores",
            str(Path(args.analysis_out) / "speech_action_axis_scores.parquet"),
            "--links",
            "data/parquet/speech_action_links.parquet",
            "--consistency",
            str(Path(args.analysis_out) / "consistency_score_party.parquet"),
            "--out",
            str(Path(args.analysis_out) / "party_latent_ideology_estimates.parquet"),
            "--summary-out",
            str(Path(args.analysis_out) / "party_latent_ideology_summary.json"),
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        latent_out = proc.stdout

    if args.run_benchmark_validation:
        cmd = [
            sys.executable,
            str(script_dir / "validate_party_ideology_benchmarks.py"),
            "--internal",
            str(Path(args.analysis_out) / "party_latent_ideology_estimates.parquet"),
            "--internal-party-col",
            "party",
            "--internal-score-col",
            "latent_ideology_score",
            "--external",
            str(Path(args.analysis_out) / "external_party_benchmarks.csv"),
            "--out-merged",
            str(Path(args.analysis_out) / "party_ideology_benchmark_merged.parquet"),
            "--summary-out",
            str(Path(args.analysis_out) / "party_ideology_benchmark_validation.json"),
            "--write-template",
            "--template-out",
            "docs/external_party_benchmarks_template.csv",
        ]
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        benchmark_validation_out = proc.stdout

    if args.run_consistency:
        cmd = [
            sys.executable,
            str(script_dir / "analyze_consistency_trends.py"),
            "--analysis-dir",
            args.analysis_out,
            "--figures-dir",
            "output/manuscript/figures",
            "--cpu-fraction",
            str(args.cpu_fraction),
        ]
        if args.mlflow:
            cmd.extend(["--mlflow", "--mlflow-experiment", f"{args.mlflow_experiment}-consistency"])
            if args.mlflow_tracking_uri:
                cmd.extend(["--mlflow-tracking-uri", args.mlflow_tracking_uri])
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        consistency_out = proc.stdout

    if args.run_recency:
        cmd = [
            sys.executable,
            str(script_dir / "analyze_recency_weighted_trends.py"),
            "--topic-year",
            str(Path(args.analysis_out) / "promise_fulfillment_party_topic_year.parquet"),
            "--out-dir",
            args.analysis_out,
            "--cpu-fraction",
            str(args.cpu_fraction),
            "--election-cadence-years",
            "4",
            "--runup-years",
            "1",
        ]
        if args.run_sarimax:
            cmd.append("--sarimax")
        if args.mlflow:
            cmd.extend(["--mlflow", "--mlflow-experiment", f"{args.mlflow_experiment}-recency"])
            if args.mlflow_tracking_uri:
                cmd.extend(["--mlflow-tracking-uri", args.mlflow_tracking_uri])
        proc = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        recency_out = proc.stdout

    print(
        json.dumps(
            {
                "speech": speech,
                "ideological_gap": gap,
                "promise_fulfillment": promise,
                "all_linkage_stdout": all_linkage_out,
                "axis_stdout": axis_out,
                "contradiction_stdout": contradiction_out,
                "link_confidence_stdout": link_confidence_out,
                "uncertainty_stdout": uncertainty_out,
                "link_stability_stdout": link_stability_out,
                "latent_stdout": latent_out,
                "benchmark_validation_stdout": benchmark_validation_out,
                "consistency_stdout": consistency_out,
                "recency_stdout": recency_out,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
