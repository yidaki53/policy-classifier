import pytest

from swedish_parliament_policy_classifier.classifier.signal_executor import (
    RequiredSignalError,
    SignalContext,
    SignalExecutor,
)


class WorkingSignal:
    name = "keyword"

    def compute(self, context):
        return {"left": len(context.text)}


class BrokenSignal:
    name = "embedding"

    def compute(self, context):
        raise RuntimeError("model unavailable")


def test_signal_executor_records_optional_failure() -> None:
    run = SignalExecutor().run(
        [WorkingSignal(), BrokenSignal()],
        SignalContext("m1", "abc", {}),
    )

    assert run.scores["keyword"] == {"left": 3.0}
    assert run.scores["embedding"] == {}
    assert "model unavailable" in run.failed["embedding"]


def test_signal_executor_raises_for_required_failure() -> None:
    with pytest.raises(RequiredSignalError, match="Required signal 'embedding'"):
        SignalExecutor().run(
            [BrokenSignal()],
            SignalContext("m1", "abc", {}),
            required=frozenset({"embedding"}),
        )