from __future__ import annotations

from typing import Iterable


class ProgressHook:
    """Lightweight, optional progress wrapper with tqdm fallback.

    Usage:
        with ProgressHook("label", total=100) as hook:
            for item in items:
                ...do work...
                hook.advance()
    """

    def __init__(self, desc: str = "", total: int | None = None, unit: str = "it"):
        self.desc = desc
        self.total = total
        self.unit = unit
        self._pbar = None

    def __enter__(self):
        try:
            from tqdm.auto import tqdm

            self._pbar = tqdm(total=self.total, desc=self.desc, unit=self.unit)
        except Exception:
            self._pbar = None
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def advance(self, n: int = 1):
        if self._pbar is not None:
            try:
                self._pbar.update(n)
            except Exception:
                pass

    def close(self):
        if self._pbar is not None:
            try:
                self._pbar.close()
            except Exception:
                pass


class TqdmIter:
    """Wrap any iterable with an optional tqdm progress bar."""

    def __init__(self, iterable: Iterable, desc: str = "", total: int | None = None, unit: str = "it"):
        self.iterable = iterable
        self.desc = desc
        self.total = total
        self.unit = unit
        self._pbar = None
        self._it = None

    def __iter__(self):
        try:
            from tqdm.auto import tqdm

            self._it = iter(self.iterable)
            total = self.total
            if total is None:
                try:
                    total = len(self.iterable)  # type: ignore[arg-type]
                except Exception:
                    total = None
            self._pbar = tqdm(total=total, desc=self.desc, unit=self.unit)
        except Exception:
            self._it = iter(self.iterable)
            self._pbar = None
        return self

    def __next__(self):
        try:
            val = next(self._it)
        except StopIteration:
            if self._pbar is not None:
                try:
                    self._pbar.close()
                except Exception:
                    pass
            raise
        if self._pbar is not None:
            try:
                self._pbar.update(1)
            except Exception:
                pass
        return val


def tqdm_iter(iterable: Iterable, desc: str = "", total: int | None = None, unit: str = "it") -> Iterable:
    return TqdmIter(iterable, desc=desc, total=total, unit=unit)


__all__ = ["ProgressHook", "TqdmIter", "tqdm_iter"]