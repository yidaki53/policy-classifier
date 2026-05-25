"""Analysis helpers exposed from the package-local `aggregate` implementation.

The implementation now lives inside the package so import resolution is
deterministic for editable installs and CI.
"""
from . import aggregate as aggregate

compute_party_profiles = aggregate.compute_party_profiles
load_party_profiles = aggregate.load_party_profiles

__all__ = ["compute_party_profiles", "load_party_profiles", "aggregate"]
