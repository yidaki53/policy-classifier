"""Training workflows kept separate from runtime classifier entrypoints."""

from .ensemble import train_meta_classifier

__all__ = ["train_meta_classifier"]
