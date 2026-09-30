import pandas as pd
import pytest
from pathlib import Path

from swedish_parliament_policy_classifier.io.artifacts import (
    ArtifactSpec,
    LocalParquetArtifactRepository,
)


def test_artifact_repository_validates_schema_and_primary_keys(tmp_path) -> None:
    repository = LocalParquetArtifactRepository(tmp_path)
    spec = ArtifactSpec(
        name="speech_results",
        relative_path=Path("results.parquet"),
        required_columns=("speech_id", "category", "probability"),
        primary_keys=("speech_id", "category"),
    )
    frame = pd.DataFrame(
        {"speech_id": ["s1"], "category": ["left"], "probability": [0.8]}
    )

    path = repository.write(spec, frame)

    assert path.exists()
    pd.testing.assert_frame_equal(repository.read(spec), frame)

    with pytest.raises(ValueError, match="missing required columns"):
        repository.validate(spec, frame.drop(columns=["probability"]))


def test_artifact_repository_rejects_duplicate_primary_keys(tmp_path) -> None:
    repository = LocalParquetArtifactRepository(tmp_path)
    spec = ArtifactSpec("results", "results.parquet", ("id",), ("id",))
    duplicate = pd.DataFrame({"id": ["x", "x"]})

    with pytest.raises(ValueError, match="duplicate primary keys"):
        repository.write(spec, duplicate)