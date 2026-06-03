"""Thin scorer shim that re-exports a stable implementation.

During refactors the concrete implementation lived in several helper
modules. Expose `score_motion` here by importing the tested implementation
(`scorer_old.py`) so runtime imports succeed and tests continue to pass.
"""

import inspect

from swedish_parliament_policy_classifier.nlp.preprocess import preprocess_text
from swedish_parliament_policy_classifier.models.models import CategoryDef, ClassificationResult

try:
	# Prefer the fully-implemented scorer moved into the package
	from .scorer_old import score_motion as _impl_score_motion, load_definitions  # type: ignore
except Exception:
	# Fallback: try other historical locations that may exist in the tree
	try:
		from .scorer_v020 import score_motion as _impl_score_motion, load_definitions  # type: ignore
	except Exception:
		# Leave a clear ImportError for callers
		raise


def score_motion(*args, **kwargs):
	"""Compatibility wrapper that ignores unsupported keyword arguments.

	Some callers pass optional context keys (for example `party`) that older
	scorer implementations do not accept. Filter kwargs to the implementation
	signature so training/evaluation scripts remain backward compatible.
	"""
	if not kwargs:
		return _impl_score_motion(*args)

	allowed = set(inspect.signature(_impl_score_motion).parameters.keys())
	filtered = {k: v for k, v in kwargs.items() if k in allowed}
	return _impl_score_motion(*args, **filtered)
# Re-export pipeline helper used by scripts/tests
try:
	from swedish_parliament_policy_classifier.classifier.pipeline import _extract_speech_argumentative_text
except Exception:
	try:
		from .pipeline import _extract_speech_argumentative_text  # type: ignore
	except Exception:
		_extract_speech_argumentative_text = None  # type: ignore

try:
	from swedish_parliament_policy_classifier.classifier.pipeline import _sentence_stance
except Exception:
	try:
		from .pipeline import _sentence_stance  # type: ignore
	except Exception:
		_sentence_stance = None  # type: ignore

__all__ = [
	"score_motion",
	"preprocess_text",
	"CategoryDef",
	"ClassificationResult",
	"load_definitions",
	"_extract_speech_argumentative_text",
	"_sentence_stance",
]

