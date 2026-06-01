"""IO helpers for compressed pickle/JSON and Parquet export.

Provides transparent zstd-backed read/write helpers for pickles and JSON,
and parquet read/write helpers using pandas/pyarrow. The pickle/json
helpers auto-detect a `.zst` compressed sibling file if present so
existing code paths that refer to `.pkl` still work once compressed
artifacts are written alongside.
"""
from pathlib import Path
import pickle
import json
import io
from typing import Any, Optional

try:
    import zstandard as zstd
except Exception:  # pragma: no cover - optional dependency
    zstd = None


def _ensure_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def save_pickle(path: Path, obj: Any, compress: bool = True, level: int = 3) -> Path:
    path = Path(path)
    _ensure_parent(path)
    if compress:
        if zstd is None:
            raise RuntimeError("zstandard not installed; install zstandard to write compressed pickles")
        # write to <path>.zst to avoid clobbering uncompressed files
        target = path if path.suffix in (".zst", ".zstd") else Path(str(path) + ".zst")
        with open(target, "wb") as fh:
            cctx = zstd.ZstdCompressor(level=level)
            with cctx.stream_writer(fh) as compressor:
                pickle.dump(obj, compressor)
        return target
    else:
        with open(path, "wb") as fh:
            pickle.dump(obj, fh)
        return path


def load_pickle(path: Path) -> Any:
    path = Path(path)
    # prefer the given path, but allow a .zst sibling for compressed artifacts
    if not path.exists():
        cand = Path(str(path) + ".zst")
        if cand.exists():
            path = cand
    if path.suffix in (".zst", ".zstd"):
        if zstd is None:
            raise RuntimeError("zstandard not installed; install zstandard to read compressed pickles")
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                return pickle.load(reader)
    else:
        with open(path, "rb") as fh:
            return pickle.load(fh)


def save_json(path: Path, obj: Any, compress: bool = True, level: int = 3) -> Path:
    path = Path(path)
    _ensure_parent(path)
    if compress:
        if zstd is None:
            raise RuntimeError("zstandard not installed; install zstandard to write compressed JSON")
        target = path if path.suffix in (".zst", ".zstd") else Path(str(path) + ".zst")
        with open(target, "wb") as fh:
            cctx = zstd.ZstdCompressor(level=level)
            with cctx.stream_writer(fh) as compressor:
                # use a text wrapper so json.dump writes unicode properly
                with io.TextIOWrapper(compressor, encoding="utf-8") as t:
                    json.dump(obj, t, ensure_ascii=False)
        return target
    else:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False)
        return path


def load_json(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        cand = Path(str(path) + ".zst")
        if cand.exists():
            path = cand
    if path.suffix in (".zst", ".zstd"):
        if zstd is None:
            raise RuntimeError("zstandard not installed; install zstandard to read compressed JSON")
        with open(path, "rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as t:
                    return json.load(t)
    else:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)


def save_parquet(df, path: Path, compression: str = "snappy") -> Path:
    """Save a pandas DataFrame to Parquet using pyarrow."""
    path = Path(path)
    _ensure_parent(path)
    try:
        df.to_parquet(path, index=False, compression=compression)
    except Exception as e:
        # surface a clear error for missing dependencies
        raise RuntimeError("Failed to write parquet; ensure pandas and pyarrow are installed: %s" % e)
    return path


def load_parquet(path: Path):
    try:
        import pandas as pd
    except Exception as e:
        raise RuntimeError("pandas required to read parquet: %s" % e)
    return pd.read_parquet(path)
