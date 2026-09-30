import pytest

from swedish_parliament_policy_classifier.classifier.features import FeatureSpec


def test_feature_spec_rejects_reordered_features() -> None:
    spec = FeatureSpec(["left", "right"], max_topics=1)

    spec.validate_feature_names(spec.feature_names())

    with pytest.raises(ValueError, match="Feature schema mismatch"):
        spec.validate_feature_names(list(reversed(spec.feature_names())))