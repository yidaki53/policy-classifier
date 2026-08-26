"""Analysis helpers exposed from the package-local `aggregate` implementation.

The implementation now lives inside the package so import resolution is
deterministic for editable installs and CI.
"""
from . import aggregate as aggregate
from .contracts import PublicationContractBundle, StudySpecification
from .evaluation import (
    bootstrap_confidence_interval,
    cohen_kappa,
    run_sensitivity_analysis,
    summarize_classification_results,
)
from .publication_workflow import (
    build_blinded_annotation_package,
    build_external_handoff_package,
    build_publication_release_package,
    build_publication_result_bundle,
    load_publication_contract_bundle,
)

compute_party_profiles = aggregate.compute_party_profiles
load_party_profiles = aggregate.load_party_profiles

__all__ = [
    "aggregate",
    "bootstrap_confidence_interval",
    "build_blinded_annotation_package",
    "build_external_handoff_package",
    "build_publication_release_package",
    "build_publication_result_bundle",
    "cohen_kappa",
    "compute_party_profiles",
    "load_party_profiles",
    "load_publication_contract_bundle",
    "PublicationContractBundle",
    "run_sensitivity_analysis",
    "StudySpecification",
    "summarize_classification_results",
]
