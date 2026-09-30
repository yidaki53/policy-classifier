"""Cached access to speech text artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Collection, Mapping, Protocol

import pandas as pd


class SpeechTextRepository(Protocol):
    def get(self, speech_id: str) -> str:
        """Return speech text or an empty string when the ID is unavailable."""
        ...

    def get_many(self, speech_ids: Collection[str]) -> Mapping[str, str]:
        """Return text for requested IDs without changing missing-ID semantics."""
        ...


@dataclass
class InMemorySpeechTextRepository:
    texts: Mapping[str, str] = field(default_factory=dict)

    def get(self, speech_id: str) -> str:
        return str(self.texts.get(str(speech_id), "") or "")

    def get_many(self, speech_ids: Collection[str]) -> Mapping[str, str]:
        return {str(speech_id): self.get(str(speech_id)) for speech_id in speech_ids}


@dataclass
class ParquetSpeechTextRepository:
    directory: Path
    _texts: dict[str, str] | None = field(default=None, init=False, repr=False)

    def _load(self) -> dict[str, str]:
        if self._texts is not None:
            return self._texts

        texts: dict[str, str] = {}
        for path in sorted(self.directory.glob("*.parquet")):
            frame = pd.read_parquet(path, columns=["anforande_id", "anforandetext"])
            for speech_id, text in frame.itertuples(index=False, name=None):
                if pd.isna(speech_id):
                    continue
                texts.setdefault(str(speech_id), str(text or ""))
        self._texts = texts
        return texts

    def get(self, speech_id: str) -> str:
        return self._load().get(str(speech_id), "")

    def get_many(self, speech_ids: Collection[str]) -> Mapping[str, str]:
        texts = self._load()
        return {str(speech_id): texts.get(str(speech_id), "") for speech_id in speech_ids}


__all__ = ["InMemorySpeechTextRepository", "ParquetSpeechTextRepository", "SpeechTextRepository"]