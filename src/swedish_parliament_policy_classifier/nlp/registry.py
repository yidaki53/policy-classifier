"""Simple model registry to hold adapters for injection and testing.

This is intentionally tiny: callers may register mock adapters in tests and
production code can register real adapters during startup. The registry is
global to keep wiring straightforward for CLI scripts.
"""
from typing import Dict, Optional
from threading import Lock

_REG: Dict[str, object] = {}
_LOCK = Lock()


def register(name: str, adapter: object) -> None:
    with _LOCK:
        _REG[name] = adapter


def get(name: str) -> Optional[object]:
    with _LOCK:
        return _REG.get(name)


def unregister(name: str) -> None:
    with _LOCK:
        if name in _REG:
            del _REG[name]
