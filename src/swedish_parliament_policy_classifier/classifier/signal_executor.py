"""Executable signal-provider boundary for classifier migration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Protocol, Sequence

from swedish_parliament_policy_classifier.models import CategoryDef


@dataclass(frozen=True)
class SignalContext:
    item_id: str
    text: str
    categories: Mapping[str, CategoryDef]


class SignalProvider(Protocol):
    name: str

    def compute(self, context: SignalContext) -> Mapping[str, float]:
        ...


@dataclass(frozen=True)
class CallableSignalProvider:
    name: str
    function: Callable[[SignalContext], Mapping[str, float]]

    def compute(self, context: SignalContext) -> Mapping[str, float]:
        return self.function(context)


@dataclass(frozen=True)
class SignalRun:
    scores: Mapping[str, Mapping[str, float]]
    failed: Mapping[str, str]


class RequiredSignalError(RuntimeError):
    pass


class SignalExecutor:
    def run(
        self,
        providers: Sequence[SignalProvider],
        context: SignalContext,
        *,
        required: frozenset[str] = frozenset(),
    ) -> SignalRun:
        scores: dict[str, Mapping[str, float]] = {}
        failed: dict[str, str] = {}
        for provider in providers:
            try:
                scores[provider.name] = {
                    str(category): float(value)
                    for category, value in provider.compute(context).items()
                }
            except Exception as exc:
                if provider.name in required:
                    raise RequiredSignalError(
                        f"Required signal {provider.name!r} failed for {context.item_id}: {exc}"
                    ) from exc
                failed[provider.name] = str(exc)
                scores[provider.name] = {}
        return SignalRun(scores=scores, failed=failed)


__all__ = [
    "CallableSignalProvider",
    "RequiredSignalError",
    "SignalContext",
    "SignalExecutor",
    "SignalProvider",
    "SignalRun",
]