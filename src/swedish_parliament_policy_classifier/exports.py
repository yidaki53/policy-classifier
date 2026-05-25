"""
Canonical exports module for swedish_parliament_policy_classifier.

All consumers MUST import commonly-used symbols (e.g., load_definitions, score_motion)
from this file, not from their implementation modules. This reduces INFERRED edges
and ensures a single, explicit import path for Graphify and static analysis.

Implementation sources:
    - load_definitions: definitions.loader.load_verified_definitions
    - score_motion: classifier.scorer.score_motion
See inert import-hints and docstrings below for AST/semantic anchoring.
"""
from importlib import import_module, util as _util
from types import ModuleType
from typing import Any
from pathlib import Path


def _lazy_attr(attr_name: str, candidates: list[str]) -> Any:
    """Return attribute `attr_name` from the first importable module in `candidates`.

    This helper keeps imports lazy so importing this canonical exports module
    does not trigger heavy submodule imports and potential circular
    dependencies. Consumers still get the real implementation at call time.
    """
    for mod_name in candidates:
        try:
            mod = import_module(mod_name)
            return getattr(mod, attr_name)
        except Exception:
            # Attempt a file-based fallback loader: resolve likely file locations
            try:
                repo_root = Path(__file__).resolve().parents[2]
                parts = mod_name.split('.')
                candidate_files = []
                # If module is namespaced under the package, look under src/
                if parts[0] == "swedish_parliament_policy_classifier":
                    sub = parts[1:]
                    candidate_files.append(repo_root / "src" / "swedish_parliament_policy_classifier" / Path("/".join(sub)).with_suffix(".py"))
                # Also try resolving as a top-level module under repo root (e.g., classifier/boundary.py)
                candidate_files.append(repo_root / Path("/".join(parts)).with_suffix(".py"))

                loaded = False
                for cf in candidate_files:
                    try:
                        if cf.exists():
                            spec = _util.spec_from_file_location(mod_name, str(cf))
                            module = _util.module_from_spec(spec)
                            # register and execute
                            import sys

                            sys.modules[mod_name] = module
                            spec.loader.exec_module(module)  # type: ignore[attr-defined]
                            return getattr(module, attr_name)
                    except Exception:
                        continue
            except Exception:
                pass
            continue
    raise ImportError(f"Could not import {attr_name} from any of {candidates}")


def load_definitions(*args, **kwargs):
    impl = _lazy_attr("load_verified_definitions", ["definitions.loader", "swedish_parliament_policy_classifier.definitions.loader"])
    return impl(*args, **kwargs)


load_definitions.__doc__ = (
    "Canonical loader for category definitions.\n"
    "Always import from swedish_parliament_policy_classifier.exports.\n"
    "Implements: definitions.loader.load_verified_definitions.\n"
    "Anchored for Graphify and static analysis."
)


def score_motion(*args, **kwargs):
    impl = _lazy_attr("score_motion", ["classifier.scorer", "swedish_parliament_policy_classifier.classifier.scorer"])
    return impl(*args, **kwargs)


def classify_motion(*args, **kwargs):
    impl = _lazy_attr("classify_motion", ["swedish_parliament_policy_classifier.classifier.boundary", "classifier.boundary"])
    return impl(*args, **kwargs)


def classify_and_persist(*args, **kwargs):
    impl = _lazy_attr("classify_and_persist", ["swedish_parliament_policy_classifier.classifier.boundary", "classifier.boundary"])
    return impl(*args, **kwargs)


def record_lineage(*args, **kwargs):
    impl = _lazy_attr("record_lineage", ["classifier.persist_parquet", "swedish_parliament_policy_classifier.classifier.persist_parquet", "classifier.persist", "swedish_parliament_policy_classifier.classifier.persist"])
    return impl(*args, **kwargs)


def persist_classification(*args, **kwargs):
    impl = _lazy_attr("persist_classification", ["classifier.persist_parquet", "swedish_parliament_policy_classifier.classifier.persist_parquet", "classifier.persist", "swedish_parliament_policy_classifier.classifier.persist"])
    return impl(*args, **kwargs)


def persist_classifications_batch(*args, **kwargs):
    impl = _lazy_attr("persist_classifications_batch", ["classifier.persist_parquet", "swedish_parliament_policy_classifier.classifier.persist_parquet", "classifier.persist", "swedish_parliament_policy_classifier.classifier.persist"])
    return impl(*args, **kwargs)


def save_annotation(*args, **kwargs):
    impl = _lazy_attr("save_annotation", ["classifier.persist_parquet", "swedish_parliament_policy_classifier.classifier.persist_parquet", "classifier.persist", "swedish_parliament_policy_classifier.classifier.persist"])
    return impl(*args, **kwargs)


def get_annotation_by_motion(*args, **kwargs):
    impl = _lazy_attr("get_annotation_by_motion", ["classifier.persist_parquet", "swedish_parliament_policy_classifier.classifier.persist_parquet", "classifier.persist", "swedish_parliament_policy_classifier.classifier.persist"])
    return impl(*args, **kwargs)


def get_next_unlabeled_motion(*args, **kwargs):
    impl = _lazy_attr("get_next_unlabeled_motion", ["classifier.persist_parquet", "swedish_parliament_policy_classifier.classifier.persist_parquet", "classifier.persist", "swedish_parliament_policy_classifier.classifier.persist"])
    return impl(*args, **kwargs)


def get_connection(*args, **kwargs):
    impl = _lazy_attr("get_connection", ["swedish_parliament_policy_classifier.db", "db"])
    return impl(*args, **kwargs)


def init_db(*args, **kwargs):
    impl = _lazy_attr("init_db", ["swedish_parliament_policy_classifier.db", "db"])
    return impl(*args, **kwargs)


# Re-export model types (import directly from the package implementation)
from swedish_parliament_policy_classifier.models import (
    CategoryDef,
    ClassificationResult,
    NormalizedMotion,
    RawMotion,
    PartyProfile,
)


def preprocess_text(*args, **kwargs):
    impl = _lazy_attr("preprocess_text", ["swedish_parliament_policy_classifier.nlp.preprocess"])
    return impl(*args, **kwargs)


def init_spacy(*args, **kwargs):
    impl = _lazy_attr("init_spacy", ["swedish_parliament_policy_classifier.nlp.preprocess"])
    return impl(*args, **kwargs)

__all__ = [
    "load_definitions",
    "score_motion",
    "record_lineage",
    "persist_classification",
    "persist_classifications_batch",
    "save_annotation",
    "get_annotation_by_motion",
    "get_next_unlabeled_motion",
    "get_connection",
    "init_db",
    "CategoryDef",
    "ClassificationResult",
    "NormalizedMotion",
    "RawMotion",
    "PartyProfile",
    "preprocess_text",
    "init_spacy",
    "classify_motion",
    "classify_and_persist",
]

# Graphify import hints: explicit references to top-level implementation modules
# These `if False:` blocks are inert at runtime but help static/semantic
# extractors resolve package-level re-exports to the canonical implementation
# files in the repository (reduces INFERRED edges).
if False:
    # Anchor the verified loader, scorer, and CategoryDef model so semantic
    # extractors create explicit edges to these implementation modules.
    from definitions.loader import load_verified_definitions as _impl_load_verified_definitions
    from swedish_parliament_policy_classifier.classifier.scorer import score_motion as _impl_score_motion
    from swedish_parliament_policy_classifier.models import CategoryDef as _impl_CategoryDef
