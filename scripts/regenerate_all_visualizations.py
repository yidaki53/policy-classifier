#!/usr/bin/env python3
"""Regenerate visualization artifacts from latest data outputs.

This script orchestrates all actively used visualization producers and writes
outputs into existing figure destinations in this repository.

Usage:
    uv run python scripts/regenerate_all_visualizations.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path, allow_fail: bool = False) -> dict:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    ok = proc.returncode == 0
    if not ok and not allow_fail:
        raise RuntimeError(
            "Command failed\n"
            f"cmd: {' '.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return {
        "ok": ok,
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def _copy_if_exists(src: Path, dst: Path) -> bool:
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _load_consistency_args(repo_root: Path) -> dict:
    summary = repo_root / "output/analysis/consistency_wrangling_fair_ga_best.json"
    if not summary.exists():
        return {}
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
        best = payload.get("best", {})
        return {
            "speech_motion_weight": best.get("speech_motion_weight"),
            "vote_alignment_weight": best.get("vote_alignment_weight"),
            "fulfillment_fill": best.get("fulfillment_fill"),
            "expected_contradiction_fill": best.get("expected_contradiction_fill"),
            "contradiction_penalty_power": best.get("contradiction_penalty_power"),
        }
    except Exception:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate all visualization artifacts")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--profiles", default="data/parquet/party_profiles_recency.parquet")
    parser.add_argument("--speech-classifications", default="data/parquet/speech_classifications_with_rhetoric_full.parquet")
    parser.add_argument("--rhetoric-parquet", default="data/parquet/speech_rhetoric_labels.parquet")
    parser.add_argument("--speech-parquet-dir", default="data/speeches/parquet")
    parser.add_argument("--votering-parquet", default="data/votering/parquet")
    parser.add_argument("--motion-votes", default="data/parquet/motion_votes.parquet")
    parser.add_argument("--speech-motions", default="data/parquet/speech_motions.parquet")
    parser.add_argument("--analysis-dir", default="output/analysis")
    parser.add_argument("--run-eval-plots", action="store_true")
    parser.add_argument("--run-calibration", action="store_true")
    parser.add_argument("--eval-max-samples", type=int, default=2000)
    parser.add_argument("--skip-legacy-aliases", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.repo_root).resolve()

    commands: list[tuple[str, list[str], bool]] = [
        (
            "manuscript-motion-figures",
            [
                sys.executable,
                "scripts/generate_figures.py",
                "--db",
                args.db,
                "--out-dir",
                "figures/manuscript",
            ],
            False,
        ),
        (
            "party-profiles",
            [
                sys.executable,
                "scripts/visualize.py",
                "--profiles",
                args.profiles,
                "--out",
                "figures",
            ],
            False,
        ),
        (
            "party-profiles-advanced",
            [
                sys.executable,
                "scripts/visualize_advanced.py",
                "--profiles",
                args.profiles,
                "--out",
                "figures",
            ],
            False,
        ),
        (
            "party-profiles-interactive",
            [
                sys.executable,
                "scripts/visualize_interactive.py",
                "--profiles",
                args.profiles,
                "--modality",
                "motion",
                "--out",
                "figures",
                "--file",
                "party_profiles_interactive.html",
            ],
            False,
        ),
        (
            "voting-figures",
            [
                sys.executable,
                "scripts/visualize_voting.py",
                "--votering-parquet",
                args.votering_parquet,
                "--out",
                "figures/voting",
                "--db",
                args.db,
            ],
            False,
        ),
        (
            "speech-profile-figures",
            [
                sys.executable,
                "scripts/analyze_speech_profiles.py",
                "--speech-classifications",
                args.speech_classifications,
                "--speech-parquet-dir",
                args.speech_parquet_dir,
                "--out",
                "figures/speeches",
            ],
            False,
        ),
        (
            "rhetoric-ideology-figures",
            [
                sys.executable,
                "scripts/rhetoric_ideology_crosstab.py",
                "--speech-classifications",
                args.speech_classifications,
                "--rhetoric-parquet",
                args.rhetoric_parquet,
                "--speech-parquet-dir",
                args.speech_parquet_dir,
                "--out-dir",
                "figures/rhetoric",
            ],
            True,
        ),
        (
            "three-way-figures",
            [
                sys.executable,
                "scripts/speeches_analysis.py",
                "--profiles",
                args.profiles,
                "--out-dir",
                "figures/three_way",
                "--speech-motions",
                args.speech_motions,
                "--motion-votes",
                args.motion_votes,
                "--votering-dir",
                args.votering_parquet,
            ],
            False,
        ),
        (
            "manuscript-overlay-figure",
            [
                sys.executable,
                "scripts/generate_manuscript_overlay.py",
                "--profiles",
                args.profiles,
                "--out",
                "output/manuscript/figures/figure_modality_overlay_by_party.png",
            ],
            False,
        ),
    ]

    consistency = _load_consistency_args(root)
    consistency_cmd = [
        sys.executable,
        "scripts/analyze_consistency_trends.py",
        "--analysis-dir",
        args.analysis_dir,
        "--figures-dir",
        "output/manuscript/figures",
    ]
    for flag, key in [
        ("--speech-motion-weight", "speech_motion_weight"),
        ("--vote-alignment-weight", "vote_alignment_weight"),
        ("--fulfillment-fill", "fulfillment_fill"),
        ("--expected-contradiction-fill", "expected_contradiction_fill"),
        ("--contradiction-penalty-power", "contradiction_penalty_power"),
    ]:
        val = consistency.get(key)
        if val is not None:
            consistency_cmd.extend([flag, str(val)])
    commands.append(("consistency-trend-figures", consistency_cmd, False))

    if args.run_eval_plots:
        commands.append(
            (
                "speech-gold-eval-plots",
                [
                    sys.executable,
                    "scripts/evaluate_speech_gold_labels.py",
                    "--gold-parquet",
                    "data/parquet/speech_gold_labels.parquet",
                    "--speech-parquet-dir",
                    args.speech_parquet_dir,
                    "--max-samples",
                    str(args.eval_max_samples),
                ],
                True,
            )
        )

    if args.run_calibration:
        commands.append(
            (
                "speech-calibration-plots",
                [
                    sys.executable,
                    "scripts/run_calibration_checks.py",
                ],
                True,
            )
        )

    results = []
    optional_inputs_by_step = {
        "rhetoric-ideology-figures": [args.rhetoric_parquet],
    }

    if args.dry_run:
        for name, cmd, allow_fail in commands:
            missing_inputs = [p for p in optional_inputs_by_step.get(name, []) if not (root / p).exists()]
            results.append(
                {
                    "step": name,
                    "ok": True,
                    "dry_run": True,
                    "allow_fail": allow_fail,
                    "missing_optional_inputs": missing_inputs,
                    "cmd": cmd,
                }
            )
    else:
        for name, cmd, allow_fail in commands:
            missing_inputs = [p for p in optional_inputs_by_step.get(name, []) if not (root / p).exists()]
            if missing_inputs:
                results.append(
                    {
                        "step": name,
                        "ok": True,
                        "skipped": True,
                        "reason": "missing optional input",
                        "missing_optional_inputs": missing_inputs,
                        "cmd": cmd,
                    }
                )
                continue
            results.append({"step": name, **_run(cmd, cwd=root, allow_fail=allow_fail)})

    aliased = []
    if not args.skip_legacy_aliases and not args.dry_run:
        alias_pairs = [
            ("figures/manuscript/pie_chart_categories.png", "figures/combined/combined_pie_chart_categories.png"),
            ("figures/manuscript/party_motions_stacked.png", "figures/combined/combined_party_motions_stacked.png"),
            (
                "figures/manuscript/party_motions_stacked_normalized.png",
                "figures/combined/combined_party_motions_stacked_normalized.png",
            ),
            ("figures/manuscript/ideology_timeline.png", "figures/combined/combined_ideology_timeline.png"),
            ("figures/manuscript/party_ideology_heatmap.png", "figures/combined/combined_party_ideology_heatmap.png"),
            ("figures/party_profiles_final.png", "figures/combined/combined_heatmap.png"),
            ("figures/party_profiles_final.pdf", "figures/combined/combined_heatmap.pdf"),
            (
                "figures/three_way/divergence_speech_vs_combined_significance.png",
                "figures/combined/three_way_comparison.png",
            ),
        ]
        for src_rel, dst_rel in alias_pairs:
            src = root / src_rel
            dst = root / dst_rel
            if _copy_if_exists(src, dst):
                aliased.append({"src": src_rel, "dst": dst_rel})

    payload = {
        "repo_root": str(root),
        "steps": results,
        "legacy_aliases": aliased,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()