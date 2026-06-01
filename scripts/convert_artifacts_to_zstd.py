"""Compress existing .pkl and .json artifacts to zstd siblings.

This script streams files into zstd-compressed siblings (e.g. `file.pkl` -> `file.pkl.zst`)
without loading them into memory or unpickling.

Usage:
    python scripts/convert_artifacts_to_zstd.py --root data --ext .pkl .json --yes

By default it recursively scans `data/` for `.pkl` and `.json` files and creates
`.zst` siblings for any that don't already have them.
"""
from pathlib import Path
import argparse
import logging
import sys

LOG = logging.getLogger(__name__)


def compress_file(src: Path, level: int = 3) -> Path:
    try:
        import zstandard as zstd
    except Exception:
        raise RuntimeError("zstandard is required; install with `pip install zstandard`")

    target = Path(str(src) + ".zst")
    if target.exists():
        LOG.info("Skipping %s (target exists: %s)", src, target)
        return target

    LOG.info("Compressing %s -> %s", src, target)
    cctx = zstd.ZstdCompressor(level=level)
    with open(src, "rb") as fh_in, open(target, "wb") as fh_out:
        with cctx.stream_writer(fh_out) as compressor:
            while True:
                chunk = fh_in.read(65536)
                if not chunk:
                    break
                compressor.write(chunk)
    return target


def find_files(root: Path, exts):
    for ext in exts:
        pattern = f"**/*{ext}"
        for p in root.glob(pattern):
            if p.is_file():
                # skip already-compressed files
                if p.suffix in (".zst", ".zstd"):
                    continue
                # skip files that already have .zst sibling
                if Path(str(p) + ".zst").exists():
                    LOG.debug("Skipping %s (zst sibling exists)", p)
                    continue
                yield p


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data"))
    parser.add_argument("--ext", nargs="+", default=[".pkl", ".json"],
                        help="File extensions to compress (e.g. .pkl .json)")
    parser.add_argument("--level", type=int, default=3)
    parser.add_argument("--yes", action="store_true", help="Proceed without prompt")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO)

    if not args.root.exists():
        LOG.error("Root directory %s does not exist", args.root)
        sys.exit(2)

    files = list(find_files(args.root, args.ext))
    if not files:
        LOG.info("No files to compress in %s", args.root)
        return 0

    LOG.info("Found %d files to compress", len(files))
    if not args.yes:
        print(f"About to compress {len(files)} files under {args.root}. Proceed? [y/N] ", end="")
        r = input().strip().lower()
        if r not in ("y", "yes"):
            LOG.info("Cancelled by user")
            return 0

    for p in files:
        try:
            compress_file(p, level=args.level)
        except Exception as e:
            LOG.exception("Failed to compress %s: %s", p, e)

    LOG.info("Compression complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
