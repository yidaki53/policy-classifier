"""Training facade for ensemble models.

The implementation remains compatible with the historical classifier module while
callers migrate to an explicit training namespace.
"""

from swedish_parliament_policy_classifier.classifier.ensemble import (
    prepare_training_data_from_gold_labels,
    train_meta_classifier,
)

__all__ = ["prepare_training_data_from_gold_labels", "train_meta_classifier"]
