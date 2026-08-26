from pathlib import Path

import pandas as pd

from swedish_parliament_policy_classifier.analysis.contracts import (
    PublicationContractBundle,
    StudySpecification,
)
from swedish_parliament_policy_classifier.analysis.evaluation import (
    bootstrap_confidence_interval,
    cohen_kappa,
    run_sensitivity_analysis,
)
from swedish_parliament_policy_classifier.analysis.publication_workflow import (
    build_blinded_annotation_package,
    build_external_handoff_package,
    build_publication_release_package,
    build_publication_result_bundle,
    load_publication_contract_bundle,
)


def test_blinded_annotation_package_redacts_identifiers(tmp_path: Path) -> None:
    records = [
        {"item_id": "motion-001", "text": "A statement", "gold_label": "left", "notes": "review me"},
        {"item_id": "motion-002", "text": "Another statement", "gold_label": "right", "notes": "review me too"},
    ]

    manifest = build_blinded_annotation_package(records, tmp_path / "review")

    assert manifest["review_count"] == 2
    assert manifest["manifest_path"].exists()
    rows = pd.read_csv(manifest["csv_path"])
    assert "review_id" in rows.columns
    assert rows.iloc[0]["review_id"] == "review-001"
    assert "motion-001" not in rows["text"].iloc[0]


def test_evaluation_helpers_return_stable_summary_values() -> None:
    assert cohen_kappa([0, 1, 0, 1], [0, 1, 0, 1]) == 1.0
    ci = bootstrap_confidence_interval([0.8, 0.9, 0.7, 0.85], n_boot=20, seed=7)
    assert ci[0] <= ci[1]

    frame = pd.DataFrame({"scenario": ["base", "sensitivity"], "weight": [1.0, 0.5]})
    sensitivity = run_sensitivity_analysis(frame, scenario_col="scenario", weight_col="weight")
    assert sensitivity["n_scenarios"] == 2
    assert sensitivity["results"][0]["scenario"] == "base"


def test_publication_contract_bundle_validates_required_columns() -> None:
    spec = StudySpecification(data_cutoff="2024-01-01")
    bundle = PublicationContractBundle(
        study_specification=spec,
        action_evidence=pd.DataFrame({"party": ["S"], "decision": ["yes"]}),
        party_position=pd.DataFrame({"party": ["S"], "position": [0.2]}),
        say_do=pd.DataFrame({"party": ["S"], "transition": ["support_yes"]}),
        evaluation=pd.DataFrame({"metric": ["accuracy"], "value": [0.8]}),
    )

    bundle.validate()


def test_build_publication_release_package_creates_manifest_and_files(tmp_path: Path) -> None:
    dashboard_path = tmp_path / "dashboard.html"
    dashboard_path.write_text("<html></html>", encoding="utf-8")
    manuscript_path = tmp_path / "manuscript.md"
    manuscript_path.write_text("# Manuscript", encoding="utf-8")

    manifest = build_publication_release_package(
        root=tmp_path,
        output_dir=tmp_path / "release",
        title="Release",
        dashboard_path=dashboard_path,
        manuscript_path=manuscript_path,
    )

    assert manifest["release_dir"].exists()
    assert (manifest["release_dir"] / "release_manifest.json").exists()
    assert (manifest["release_dir"] / "dashboard.html").exists()
    assert (manifest["release_dir"] / "manuscript.md").exists()


def test_build_external_handoff_package_creates_regular_and_anonymized_dirs(tmp_path: Path) -> None:
    regular_path = tmp_path / "regular.md"
    regular_path.write_text("# Regular", encoding="utf-8")
    anonymized_path = tmp_path / "anonymized.md"
    anonymized_path.write_text("# Anonymized", encoding="utf-8")

    manifest = build_external_handoff_package(
        root=tmp_path,
        output_dir=tmp_path / "handoff",
        regular_manuscript_path=regular_path,
        anonymized_manuscript_path=anonymized_path,
    )

    assert manifest["handoff_dir"].exists()
    assert (manifest["handoff_dir"] / "regular" / "manuscript.md").exists()
    assert (manifest["handoff_dir"] / "peer_review" / "manuscript.md").exists()


def test_build_publication_result_bundle_validates_contract_and_writes_manifest(tmp_path: Path) -> None:
    contract_bundle = PublicationContractBundle(
        study_specification=StudySpecification(data_cutoff="2024-01-01"),
        action_evidence=pd.DataFrame({"party": ["S"], "decision": ["yes"]}),
        party_position=pd.DataFrame({"party": ["S"], "position": [0.2]}),
        say_do=pd.DataFrame({"party": ["S"], "transition": ["support_yes"]}),
        evaluation=pd.DataFrame({"metric": ["accuracy"], "value": [0.8]}),
    )
    dashboard_path = tmp_path / "dashboard.html"
    dashboard_path.write_text("<html></html>", encoding="utf-8")
    table_path = tmp_path / "public" / "summary.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text("party,value\nS,0.2\n", encoding="utf-8")

    manifest = build_publication_result_bundle(
        root=tmp_path,
        output_dir=tmp_path / "bundle",
        contract_bundle=contract_bundle,
        title="Result bundle",
        dashboard_path=dashboard_path,
        public_table_paths=[table_path],
    )

    assert manifest["bundle_dir"].exists()
    assert (manifest["bundle_dir"] / "publication_manifest.json").exists()
    assert (manifest["bundle_dir"] / "dashboard.html").exists()
    assert (manifest["bundle_dir"] / "public" / "summary.csv").exists()
    assert manifest["contract_summary"]["study_specification"]["schema_version"] == "1.0.0"


def test_load_publication_contract_bundle_normalizes_action_first_artifacts(tmp_path: Path) -> None:
    action_path = tmp_path / "action.parquet"
    positions_path = tmp_path / "positions.parquet"
    transitions_path = tmp_path / "transitions.parquet"
    evaluation_path = tmp_path / "evaluation.parquet"

    pd.DataFrame(
        {
            "party": ["S"],
            "decision_id": ["d1"],
            "decision_date": [pd.Timestamp("2024-01-31", tz="UTC")],
            "party_choice": ["yes"],
        }
    ).to_parquet(action_path, index=False)
    pd.DataFrame({"party": ["S"], "score_0_100": [50.0]}).to_parquet(positions_path, index=False)
    pd.DataFrame(
        {"speech_id": ["s1"], "speech_party": ["S"], "say_do_transition": ["support_yes"]}
    ).to_parquet(transitions_path, index=False)
    pd.DataFrame(
        {"truth": ["left", "right"], "pred": ["left", "left"], "prob_left": [0.8, 0.6], "prob_right": [0.2, 0.4]}
    ).to_parquet(evaluation_path, index=False)

    bundle = load_publication_contract_bundle(
        action_evidence_path=action_path,
        party_position_path=positions_path,
        say_do_path=transitions_path,
        evaluation_path=evaluation_path,
    )

    assert bundle.action_evidence.loc[0, "decision"] == "yes"
    assert bundle.party_position.loc[0, "position"] == 50.0
    assert bundle.say_do.loc[0, "transition"] == "support_yes"
    assert set(bundle.evaluation["metric"]) >= {"accuracy", "macro_f1", "n_samples"}
