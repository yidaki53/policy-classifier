import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

from scripts.create_train_test_split import _unique_motion_labels, validate_split_sets
from swedish_parliament_policy_classifier.classifier.persist_parquet import persist_classifications_batch
from swedish_parliament_policy_classifier.db.readers import fetch_normalized_motion
from swedish_parliament_policy_classifier.models import ClassificationResult


def test_split_helpers_reject_conflicting_motion_labels() -> None:
    with pytest.raises(ValueError, match="conflicting labels"):
        _unique_motion_labels([("m1", "left"), ("m1", "right")])


def test_split_helpers_require_disjoint_complete_sets() -> None:
    with pytest.raises(ValueError, match="overlap"):
        validate_split_sets({"m1", "m2"}, {"m1"}, {"m1"}, {"m2"})


def test_reader_can_force_sqlite_even_when_parquet_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    parquet_dir = tmp_path / "data" / "parquet"
    parquet_dir.mkdir(parents=True)
    pd.DataFrame(
        {"id": ["m1"], "title": ["Parquet"], "text": ["P"], "date": [None], "party": [None], "metadata": ["{}"]}
    ).to_parquet(parquet_dir / "normalized_motions.parquet", index=False)

    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE normalized_motions (id TEXT, title TEXT, text TEXT, date TEXT, party TEXT, metadata TEXT, doc_type TEXT, full_text_url TEXT)"
    )
    conn.execute("INSERT INTO normalized_motions VALUES (?, ?, ?, ?, ?, ?, ?, ?)", ("m1", "SQLite", "S", None, None, "{}", None, None))
    conn.commit()

    result = fetch_normalized_motion(conn, "m1", source="sqlite")

    assert result is not None
    assert result.title == "SQLite"


def test_classification_persistence_preserves_model_versions(tmp_path: Path) -> None:
    def result(version: str) -> ClassificationResult:
        return ClassificationResult(
            motion_id="m1",
            category="left",
            raw_score=1.0,
            normalized_weight=1.0,
            matched_rules=[],
            classifier_version=version,
            created_at=datetime.now(timezone.utc),
        )

    output = tmp_path / "classifications.parquet"
    persist_classifications_batch(None, [result("v1")], out_parquet=output, lineage_parquet=tmp_path / "lineage.parquet")
    persist_classifications_batch(None, [result("v2")], out_parquet=output, lineage_parquet=tmp_path / "lineage.parquet")

    frame = pd.read_parquet(output)
    assert set(frame["classifier_version"]) == {"v1", "v2"}