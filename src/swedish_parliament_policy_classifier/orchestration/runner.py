"""High-level runner for speech workflows.

Provides a small `SpeechRunner` class that composes lower-level helpers so
CLI scripts can use a single, testable entrypoint for the common flows:
train -> classify -> export.
"""
from pathlib import Path
from typing import Optional
import logging

from swedish_parliament_policy_classifier.orchestration import speech_pipeline

LOG = logging.getLogger(__name__)


class SpeechRunner:
    def __init__(self, db_path: str = "data/swedish_parliament.db"):
        self.db_path = db_path

    def train(self, out_path: str = "models/speech_meta_clf.pkl", tune: bool = False, n_iter: int = 12, **kwargs) -> Path:
        LOG.info("Starting speech meta-classifier training: tune=%s n_iter=%d", tune, n_iter)
        # Only forward kwargs that the underlying function accepts to avoid
        # TypeError caused by unexpected parameters from CLI flags.
        try:
            import inspect

            sig = inspect.signature(speech_pipeline.train_and_save_speech_meta_classifier)
            accepted = {k for k in sig.parameters.keys()}
            fwd = {k: v for k, v in kwargs.items() if k in accepted}
        except Exception:
            fwd = {}

        return speech_pipeline.train_and_save_speech_meta_classifier(db_path=self.db_path, out_path=out_path, tune=tune, n_iter=n_iter, **fwd)

    def export_active_learning(self, top_n: int = 500, preds_csv: Optional[str] = None, out_path: Optional[str] = None) -> Path:
        LOG.info("Exporting top %d active-learning candidates", top_n)
        return speech_pipeline.export_active_learning_candidates(db_path=self.db_path, preds_csv=preds_csv, top_n=top_n, out_path=out_path)

    def classify_speeches(self, **kwargs) -> int:
        """Run the existing scripts/classify_speeches.py flow via dynamic import.

        This keeps the runner lightweight while reusing the canonical script
        implementation.
        """
        import importlib.util
        from pathlib import Path

        script_path = Path("scripts") / "classify_speeches.py"
        if not script_path.exists():
            raise FileNotFoundError(script_path)

        spec = importlib.util.spec_from_file_location("classify_speeches_script", str(script_path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
        if hasattr(module, "classify_speeches"):
            return module.classify_speeches(db_path=self.db_path, **kwargs)
        raise RuntimeError("classify_speeches function not found in script")
