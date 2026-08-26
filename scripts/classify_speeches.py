#!/usr/bin/env python3
"""Classify speech texts using the full motion classifier pipeline.

Applies score_motion() with skip_policy_extraction=True so the full text
(including encoder/BERT/zero-shot/meta-classifier) is used, but without the
motion-specific sentence stripping.  Results go into speech_classifications.

Supports resuming (skips speeches already in DB) and periodic GPU cleanup.

Usage:
    uv run python scripts/classify_speeches.py --db data/swedish_parliament.db
"""

import argparse
import gc
import inspect
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from swedish_parliament_policy_classifier.db import init_db

from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion

# Canonical usage: load_definitions and classify_motion are always imported from swedish_parliament_policy_classifier.exports
# This explicit import path anchors the code graph and reduces INFERRED edges for Graphify/static analysis.
# See also: classifier/scorer.py, definitions/loader.py, and src/swedish_parliament_policy_classifier/exports.py
if False:
    from swedish_parliament_policy_classifier.exports import load_definitions as _ld, classify_motion as _cm
    _ = _ld, _cm
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher

try:
    from swedish_parliament_policy_classifier.classifier.ensemble import load_meta_classifier
except Exception:
    load_meta_classifier = None

try:
    from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
except Exception:
    load_topic_distributions = None


_SCORE_MOTION_PARAMS = set(inspect.signature(classify_motion).parameters.keys())


def _score_motion_compat(**kwargs):
    # Keep compatibility across scorer versions by only passing supported args.
    filtered = {k: v for k, v in kwargs.items() if k in _SCORE_MOTION_PARAMS}
    return classify_motion(**filtered)

DEFAULT_MAX_TEXT = 30_000  # Use full speech text
BATCH_SIZE = 500  # rows per DB commit
MEMORY_FLUSH_INTERVAL = 50


def _list_speech_parquets() -> list[Path]:
    d = Path("data/speeches/parquet")
    return sorted(d.glob("*.parquet"))


def _already_classified(conn: sqlite3.Connection) -> set[str]:
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT speech_id FROM speech_classifications")
    return {r[0] for r in cur.fetchall()}


def _insert_batch(conn: sqlite3.Connection, batch: list[tuple]):
    try:
        from swedish_parliament_policy_classifier.db.repo import SpeechClassificationRepo

        repo = SpeechClassificationRepo(conn=conn)
        repo.bulk_insert(batch)
    except Exception:
        cur = conn.cursor()
        cur.executemany(
            """INSERT INTO speech_classifications (
                speech_id, category, raw_score, normalized_weight, matched_rules,
                classifier_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
            batch,
        )
        conn.commit()


def _strip_html(text: str) -> str:
    if not text:
        return ""
    # If it looks like HTML, strip tags with a lightweight fallback.
    if "<" in text and ">" in text:
        return re.sub(r"<[^>]+>", " ", text)
    return text


def classify_speeches(
    db_path: str = "data/swedish_parliament.db",
    limit: Optional[int] = None,
    use_embeddings: bool = True,
    use_zero_shot: bool = True,
    max_text_len: int = DEFAULT_MAX_TEXT,
    use_ollama: bool = False,
    rhetoric_parquet: Optional[str] = None,
):
    conn = init_db(db_path)
    defs = load_definitions()

    matcher = None
    if use_embeddings:
        try:
            matcher = EmbeddingMatcher()
            if matcher.model is None:
                matcher = None
        except Exception as e:
            print(f"Embedding matcher unavailable: {e}", file=sys.stderr)
            matcher = None

    already = _already_classified(conn)
    print(f"Skipping {len(already):,} already-classified speeches", file=sys.stderr)

    topic_dists = load_topic_distributions() if load_topic_distributions else None

    # Optional: load precomputed rhetoric scores (speech-level)
    rhet_map = {}
    if rhetoric_parquet:
        rp = Path(rhetoric_parquet)
        if rp.exists():
            try:
                rhet_df = pd.read_parquet(rp)
                if "speech_id" in rhet_df.columns:
                    for _, rr in rhet_df.iterrows():
                        sid = str(rr.get("speech_id")) if rr.get("speech_id") is not None else None
                        if not sid:
                            continue
                        rhet_map[sid] = {
                            "irony": float(rr.get("irony", 0.0)) if rr.get("irony") is not None else 0.0,
                            "sarcasm": float(rr.get("sarcasm", 0.0)) if rr.get("sarcasm") is not None else 0.0,
                            "posturing": float(rr.get("posturing", 0.0)) if rr.get("posturing") is not None else 0.0,
                            "none": float(rr.get("none", 0.0)) if rr.get("none") is not None else 0.0,
                            "top_label": rr.get("top_label") if "top_label" in rr.index else None,
                        }
                print(f"Loaded rhetoric scores for {len(rhet_map):,} speech IDs from {rp}", file=sys.stderr)
            except Exception as e:
                print(f"Failed to read rhetoric parquet {rp}: {e}", file=sys.stderr)
        else:
            print(f"Rhetoric parquet not found: {rp}", file=sys.stderr)
    meta_clf = None
    if load_meta_classifier is not None:
        try:
            # Prefer a speech-specific meta-classifier if present to avoid
            # overwriting or reusing the motions ensemble artifact.
            from pathlib import Path

            speech_model = Path("models/speech_meta_clf.pkl")
            if speech_model.exists():
                meta_clf = load_meta_classifier(model_path=speech_model)
            else:
                meta_clf = load_meta_classifier()
        except Exception as e:
            print(f"Meta-classifier not loaded: {e}", file=sys.stderr)

    # Check torch for GPU cleanup
    try:
        import torch as _torch
        _has_torch = True
    except ImportError:
        _has_torch = False

    files = _list_speech_parquets()
    if not files:
        print("No speech parquet files found in data/speeches/parquet/", file=sys.stderr)
        return 0

    processed = 0
    skipped = 0
    batch: list[tuple] = []
    start_time = time.time()

    for f in files:
        print(f"Processing {f.name}...", file=sys.stderr)
        df = pd.read_parquet(f)
        # Ensure required columns exist
        if "anforande_id" not in df.columns or "anforandetext" not in df.columns:
            print(f"  Skipping {f.name}: missing required columns", file=sys.stderr)
            continue

        for _, row in df.iterrows():
            speech_id = str(row["anforande_id"])
            if speech_id in already:
                skipped += 1
                continue

            raw_text = row.get("anforandetext") or ""
            text = _strip_html(raw_text)  # Pass full text, let scorer handle truncation
            party = row.get("parti") or None

            results = _score_motion_compat(
                motion_id=speech_id,
                text=text,
                categories=defs,
                party=None,  # Never use party for speech classification
                embedding_matcher=matcher,
                use_zero_shot=use_zero_shot,
                topic_distributions=topic_dists,
                meta_clf=meta_clf,
                skip_policy_extraction=True,
                use_speech_preprocessing=True,
                use_ollama=use_ollama,
                ollama_weight=0.55,
                rhetoric_scores=rhet_map.get(speech_id),
            )

            for r in results:
                batch.append((
                    r.motion_id,
                    r.category,
                    float(r.raw_score),
                    float(r.normalized_weight),
                    json.dumps(r.matched_rules, ensure_ascii=False),
                    r.classifier_version,
                    r.created_at.isoformat(),
                ))

            processed += 1
            if len(batch) >= BATCH_SIZE * len(defs):
                _insert_batch(conn, batch)
                batch.clear()
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                print(f"  {processed:,} processed  ({rate:.1f} speeches/s)", file=sys.stderr)
                sys.stderr.flush()

            if processed % MEMORY_FLUSH_INTERVAL == 0:
                if _has_torch and _torch.cuda.is_available():
                    _torch.cuda.empty_cache()
                gc.collect()

            if limit and processed >= limit:
                break

        if limit and processed >= limit:
            break

    if batch:
        _insert_batch(conn, batch)
        batch.clear()

    elapsed = time.time() - start_time
    print(
        f"\nDone: classified {processed} speeches in {elapsed:.1f}s "
        f"({processed/elapsed:.1f} speeches/s); skipped {skipped:,} already classified",
        file=sys.stderr,
    )
    conn.close()
    return processed


def main():
    parser = argparse.ArgumentParser(description="Classify speech texts")
    parser.add_argument("--db", default="data/swedish_parliament.db")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-text-len", type=int, default=DEFAULT_MAX_TEXT)
    parser.add_argument("--no-embeddings", dest="use_embeddings", action="store_false")
    parser.add_argument("--no-zero-shot", dest="use_zero_shot", action="store_false")
    parser.add_argument("--ollama", dest="use_ollama", action="store_true", help="Enable Ollama LLM fallback for speech classification")
    parser.add_argument("--rhetoric-parquet", default=None, help="Path to speech_rhetoric_labels.parquet to include rhetoric scores")
    args = parser.parse_args()
    # Prefer the Parquet-first implementation if available
    try:
        from importlib import util as _util
        import sys

        script_path = Path(__file__).resolve().parent / "classify_speeches_parquet.py"
        if script_path.exists():
            argv = [str(script_path)]
            if args.limit:
                argv += ["--limit", str(args.limit)]
            if not args.use_embeddings:
                argv += ["--no-embeddings"]
            if not args.use_zero_shot:
                argv += ["--no-zero-shot"]
            if args.use_ollama:
                argv += ["--ollama"]
            if args.rhetoric_parquet:
                argv += ["--rhetoric-parquet", args.rhetoric_parquet]

            old_argv = sys.argv
            sys.argv = argv
            try:
                spec = _util.spec_from_file_location("classify_speeches_parquet", str(script_path))
                mod = _util.module_from_spec(spec)
                sys.modules["classify_speeches_parquet"] = mod
                spec.loader.exec_module(mod)
            finally:
                sys.argv = old_argv
            return
    except Exception:
        pass

    # Fallback: run DB-backed implementation
    classify_speeches(
        args.db,
        limit=args.limit,
        use_embeddings=args.use_embeddings,
        use_zero_shot=args.use_zero_shot,
        max_text_len=args.max_text_len,
        use_ollama=args.use_ollama,
        rhetoric_parquet=args.rhetoric_parquet,
    )


if __name__ == "__main__":
    main()
