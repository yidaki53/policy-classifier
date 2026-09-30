import pickle

from swedish_parliament_policy_classifier.classifier.model_provider import ModelArtifactProvider


def test_model_provider_loads_ordered_pickle_candidate(tmp_path) -> None:
    artifact_path = tmp_path / "model.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump({"model": "sentinel"}, handle)

    provider = ModelArtifactProvider()

    assert provider.first_mapping([tmp_path / "missing.pkl", artifact_path]) == {"model": "sentinel"}


def test_model_provider_rejects_non_model_mappings(tmp_path) -> None:
    artifact_path = tmp_path / "metadata.pkl"
    with artifact_path.open("wb") as handle:
        pickle.dump({"version": "test"}, handle)

    assert ModelArtifactProvider().first_mapping([artifact_path]) is None