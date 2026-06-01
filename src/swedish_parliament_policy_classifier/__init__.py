"""Package initializer for ``swedish_parliament_policy_classifier``.

This file makes the package behave as a namespace during development while
also allowing top-level repository packages (e.g. ``classifier``, ``db``)
to be importable as subpackages of
``swedish_parliament_policy_classifier``. This keeps the ``src/`` layout for
packaging while making tests and local imports work.
"""
from __future__ import annotations

from pkgutil import extend_path

# Keep namespace capability for packaging tools. Do NOT append the repo
# root to `__path__` — that created ambiguous imports where top-level
# packages in the repo could be resolved as
# `swedish_parliament_policy_classifier.<submodule>` which breaks tests.
__path__ = extend_path(__path__, __name__)

__all__ = []

# Ensure Hugging Face token is available to downstream libs that expect
# `HUGGING_FACE_HUB_TOKEN`. Many CI/scripts set `HF_TOKEN`; propagate it
# automatically so transformers/sentence-transformers find the token.
try:
	import os
	# If the hub token isn't already present, try several programmatic
	# fallbacks so importing the package makes CLI/script execution
	# behave the same as running `env HUGGING_FACE_HUB_TOKEN=...`.
	if not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
		# 1) Try to load a repo `.env` file if python-dotenv is available.
		try:
			from dotenv import find_dotenv, load_dotenv

			envpath = find_dotenv()
			if envpath:
				load_dotenv(envpath, override=False)
		except Exception:
			# dotenv is optional; don't fail imports if it's not installed.
			pass

		# 2) Environment variants commonly used in CI and local shells.
		hf = (
			os.environ.get("HF_TOKEN")
			or os.environ.get("HF_HUB_TOKEN")
			or os.environ.get("HUGGING_FACE_HUB_TOKEN")
		)

		# 3) If still missing, try the conventional Hugging Face token file.
		if not hf:
			try:
				token_path = os.path.expanduser("~/.huggingface/token")
				if os.path.exists(token_path):
					with open(token_path, "r", encoding="utf8") as fh:
						t = fh.read().strip()
						if t:
							hf = t
			except Exception:
				hf = None

		if hf:
			os.environ["HUGGING_FACE_HUB_TOKEN"] = hf
except Exception:
	# Best-effort only; importing the package should never raise because of
	# token-loading logic.
	pass
