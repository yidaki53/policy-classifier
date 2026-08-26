from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import PublicationContractBundle, StudySpecification
from .evaluation import summarize_classification_results


def load_publication_contract_bundle(
    *,
    action_evidence_path: str | Path,
    party_position_path: str | Path,
    say_do_path: str | Path,
    evaluation_path: str | Path,
    data_cutoff: str | None = None,
) -> PublicationContractBundle:
    """Load canonical analysis artifacts into the validated publication contract."""
    action_evidence = pd.read_parquet(action_evidence_path).rename(columns={"party_choice": "decision"})
    party_position = pd.read_parquet(party_position_path).rename(columns={"score_0_100": "position"})
    say_do = pd.read_parquet(say_do_path).rename(
        columns={"speech_party": "party", "say_do_transition": "transition"}
    )
    evaluation_predictions = pd.read_parquet(evaluation_path)

    if data_cutoff is None:
        if "decision_date" not in action_evidence.columns:
            raise ValueError("action evidence must include decision_date when data_cutoff is not provided")
        cutoff = pd.to_datetime(action_evidence["decision_date"], errors="coerce", utc=True).max()
        if pd.isna(cutoff):
            raise ValueError("action evidence must contain at least one valid decision_date")
        data_cutoff = cutoff.isoformat()

    probability_columns = [column for column in evaluation_predictions.columns if column.startswith("prob_")]
    metrics = summarize_classification_results(
        evaluation_predictions,
        label_col="truth",
        prediction_col="pred",
        probability_columns=probability_columns or None,
    )
    evaluation = pd.DataFrame(
        [
            {"metric": metric, "value": value}
            for metric, value in metrics.items()
            if metric not in {"confusion_matrix"} and isinstance(value, (int, float))
        ]
    )
    bundle = PublicationContractBundle(
        study_specification=StudySpecification(data_cutoff=data_cutoff),
        action_evidence=action_evidence,
        party_position=party_position,
        say_do=say_do,
        evaluation=evaluation,
    )
    bundle.validate()
    return bundle


def build_publication_release_package(
    *,
    root: str | Path,
    output_dir: str | Path,
    title: str,
    dashboard_path: str | Path | None = None,
    manuscript_path: str | Path | None = None,
) -> dict[str, Any]:
    """Package the dashboard, manuscript, and a release manifest for external handoff."""
    root_path = Path(root).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    dashboard_source = Path(dashboard_path).resolve() if dashboard_path is not None else None
    manuscript_source = Path(manuscript_path).resolve() if manuscript_path is not None else None

    for source in [dashboard_source, manuscript_source]:
        if source is None or not source.exists():
            continue
        dest = output_path / source.name
        dest.write_bytes(source.read_bytes())
        sources.append({"source": str(source.relative_to(root_path) if source.is_relative_to(root_path) else source), "target": source.name})

    manifest = {
        "title": title,
        "release_dir": str(output_path),
        "files": sources,
    }
    manifest_path = output_path / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["release_dir"] = output_path
    return manifest


def build_external_handoff_package(
    *,
    root: str | Path,
    output_dir: str | Path,
    regular_manuscript_path: str | Path,
    anonymized_manuscript_path: str | Path,
) -> dict[str, Any]:
    """Package regular and anonymized manuscripts for external handoff."""
    root_path = Path(root).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    regular_source = Path(regular_manuscript_path).resolve()
    anonymized_source = Path(anonymized_manuscript_path).resolve()

    regular_target_dir = output_path / "regular"
    peer_review_target_dir = output_path / "peer_review"
    regular_target_dir.mkdir(parents=True, exist_ok=True)
    peer_review_target_dir.mkdir(parents=True, exist_ok=True)

    regular_target = regular_target_dir / "manuscript.md"
    peer_review_target = peer_review_target_dir / "manuscript.md"

    if regular_source.exists():
        shutil.copy2(regular_source, regular_target)
    if anonymized_source.exists():
        shutil.copy2(anonymized_source, peer_review_target)

    manifest_payload = {
        "handoff_dir": str(output_path),
        "regular": {"source": str(regular_source.relative_to(root_path) if regular_source.is_relative_to(root_path) else regular_source), "target": str(regular_target.relative_to(output_path))},
        "peer_review": {"source": str(anonymized_source.relative_to(root_path) if anonymized_source.is_relative_to(root_path) else anonymized_source), "target": str(peer_review_target.relative_to(output_path))},
    }

    manifest_path = output_path / "handoff_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return {**manifest_payload, "handoff_dir": output_path}


def build_publication_result_bundle(
    *,
    root: str | Path,
    output_dir: str | Path,
    contract_bundle: PublicationContractBundle,
    title: str,
    dashboard_path: str | Path | None = None,
    public_table_paths: list[str | Path] | None = None,
) -> dict[str, Any]:
    """Package a validated publication result bundle with contract metadata and public assets."""
    contract_bundle.validate()

    root_path = Path(root).resolve()
    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, Any]] = []
    dashboard_source = Path(dashboard_path).resolve() if dashboard_path is not None else None
    if dashboard_source is not None and dashboard_source.exists():
        dest = output_path / dashboard_source.name
        dest.write_bytes(dashboard_source.read_bytes())
        sources.append({"kind": "dashboard", "source": str(dashboard_source.relative_to(root_path) if dashboard_source.is_relative_to(root_path) else dashboard_source), "target": dashboard_source.name})

    for table_path in public_table_paths or []:
        source = Path(table_path).resolve()
        if not source.exists():
            continue
        dest = output_path / source.relative_to(root_path) if source.is_relative_to(root_path) else output_path / source.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
        sources.append({"kind": "table", "source": str(source.relative_to(root_path) if source.is_relative_to(root_path) else source), "target": str(dest.relative_to(output_path))})

    manifest_payload = {
        "title": title,
        "bundle_dir": str(output_path),
        "contract_summary": {
            "study_specification": contract_bundle.study_specification.to_dict(),
            "row_counts": {
                "action_evidence": len(contract_bundle.action_evidence),
                "party_position": len(contract_bundle.party_position),
                "say_do": len(contract_bundle.say_do),
                "evaluation": len(contract_bundle.evaluation),
            },
        },
        "files": sources,
    }
    manifest_path = output_path / "publication_manifest.json"
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return {**manifest_payload, "bundle_dir": output_path}


def build_blinded_annotation_package(records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    """Create a blinded annotation CSV package from raw review records."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        text = str(record.get("text", ""))
        redacted_text = text.replace("motion-", "item-").replace("motion", "item")
        rows.append(
            {
                "review_id": f"review-{index:03d}",
                "item_id": f"anonymous-{index:03d}",
                "text": redacted_text,
                "gold_label": str(record.get("gold_label", "")),
                "notes": str(record.get("notes", "")),
            }
        )

    frame = pd.DataFrame(rows)
    csv_path = output_path / "blinded_annotations.csv"
    manifest_path = output_path / "blinded_annotations_manifest.json"
    frame.to_csv(csv_path, index=False)

    manifest_payload = {
        "review_count": int(len(frame)),
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
    }
    manifest_path.write_text(json.dumps(manifest_payload, indent=2), encoding="utf-8")
    return {
        "review_count": int(len(frame)),
        "csv_path": csv_path,
        "manifest_path": manifest_path,
    }
