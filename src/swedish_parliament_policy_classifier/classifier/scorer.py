"""
Scorer module: re-exports score_motion and related symbols for classifier package.

NOTE: All consumers MUST import score_motion from swedish_parliament_policy_classifier.exports,
not directly from this module. This ensures canonical import paths and reduces INFERRED edges
in Graphify and static analysis.
"""
if False:
	# Inert import hint for static analysis / Graphify anchoring. Do NOT
	# execute at runtime to avoid circular import with the canonical
	# `swedish_parliament_policy_classifier.exports` module.
	from swedish_parliament_policy_classifier.exports import score_motion
from swedish_parliament_policy_classifier.nlp.preprocess import preprocess_text
from swedish_parliament_policy_classifier.models.models import CategoryDef, ClassificationResult

__all__ = ["score_motion", "preprocess_text", "CategoryDef", "ClassificationResult"]

