"""Model artifact provider used by classifier components."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Iterable, Optional

from swedish_parliament_policy_classifier.io import loader


class ModelArtifactProvider:
    """Load the first valid model artifact from an ordered candidate list."""

    def load(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        try:
            if path.suffix == ".zst":
                import zstandard

                with path.open("rb") as handle:
                    decompressor = zstandard.ZstdDecompressor()
                    with decompressor.stream_reader(handle) as reader:
                        return pickle.load(reader)
            return loader.load_pickle(path)
        except Exception:
            return None

    def first_mapping(self, candidates: Iterable[Path]) -> Optional[dict[str, Any]]:
        for candidate in candidates:
            artifact = self.load(candidate)
            if isinstance(artifact, dict) and ("model" in artifact or "clf" in artifact):
                return artifact
        return None


__all__ = ["ModelArtifactProvider"]