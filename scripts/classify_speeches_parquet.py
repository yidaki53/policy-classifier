#!/usr/bin/env python3
"""Parquet-first speech classification.

Reads `data/speeches/parquet/*.parquet`, runs `score_motion()` per speech,
and writes per-category rows to a Parquet output file.

Usage:
    uv run python3 scripts/classify_speeches_parquet.py --input-dir data/speeches/parquet --out data/parquet/speech_classifications.parquet

Low-heat mode example:
    CLASSIFIER_CPU_FRACTION=0.25 uv run python3 scripts/classify_speeches_parquet.py --sleep-every 50 --sleep-seconds 0.2
"""

from __future__ import annotations

import os
import signal
import sys
import time
import json
import traceback
from pathlib import Path

# ── Module-level crash handler (installed BEFORE any imports that may segfault) ──
_current_speech_id = None
_current_speech_text = None
_crash_log_path = Path("logs/classify_crash.log")
_hang_log_path = Path("logs/classify_hang.log")


def _crash_handler(signum, frame):
    """Handle SIGSEGV and other fatal signals with diagnostic logging."""
    signal_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
    crash_info = {
        "signal": signal_name,
        "speech_id": _current_speech_id,
        "text_length": len(_current_speech_text) if _current_speech_text else None,
        "text_preview": _current_speech_text[:500] if _current_speech_text else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "phase": "module_import_or_main",
    }
    _crash_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_crash_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(crash_info, ensure_ascii=False) + "\n")
    print(f"\n[FATAL] {signal_name} caught at speech_id={_current_speech_id}", file=sys.stderr)
    print(f"[FATAL] Crash log written to {_crash_log_path}", file=sys.stderr)
    sys.exit(1)


# Install crash handlers immediately, before any import that may trigger a segfault
# in a C extension (torch, transformers, spacy, numpy, etc.).
signal.signal(signal.SIGSEGV, _crash_handler)
signal.signal(signal.SIGABRT, _crash_handler)

# Load environment from `.env` when present
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# Inject Hugging Face token into commonly used env vars early so downstream
# libraries (transformers, sentence-transformers, huggingface_hub) see it
# even if they are imported later in this script's execution.
try:
    token = None
    for k in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            token = v
            break
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HF_HUB_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
except Exception:
    pass
else:
    try:
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            with token_file.open("r") as fh:
                tok = fh.read().strip()
                if tok:
                    os.environ.setdefault("HF_TOKEN", tok)
                    os.environ.setdefault("HF_HUB_TOKEN", tok)
                    os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)
    except Exception:
        pass

import argparse
import gc
import inspect
import threading
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm

from swedish_parliament_policy_classifier.exports import load_definitions
from swedish_parliament_policy_classifier.classifier.scorer import (
    score_motion as pipeline_score_motion,
    score_speech as pipeline_score_speech,
)
from swedish_parliament_policy_classifier.classifier.scorer import _load_speech_meta_classifier
from swedish_parliament_policy_classifier.definitions.registry import snapshot_definitions, write_snapshot_manifest

if False:
    from swedish_parliament_policy_classifier.exports import load_definitions as _ld
    _ = _ld
from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher
from swedish_parliament_policy_classifier.runtime.resources import apply_cpu_throttle, thermal_safe_defaults
from swedish_parliament_policy_classifier.runtime.experiment import ExperimentRun
from swedish_parliament_policy_classifier.classifier.persistence_port import ParquetClassificationWriter


try:
    from swedish_parliament_policy_classifier.classifier.ensemble import load_meta_classifier
except Exception:
    load_meta_classifier = None

try:
    from swedish_parliament_policy_classifier.nlp.topic_modeler import load_topic_distributions
except Exception:
    load_topic_distributions = None


_SCORE_MOTION_PARAMS = set(inspect.signature(pipeline_score_motion).parameters.keys())


def _score_motion_compat(**kwargs):
    filtered = {k: v for k, v in kwargs.items() if k in _SCORE_MOTION_PARAMS}
    try:
        if filtered:
            return pipeline_score_motion(**filtered)
    except TypeError:
        pass

    motion_id = kwargs.get("motion_id") or kwargs.get("speech_id") or kwargs.get("id")
    text = kwargs.get("text") or kwargs.get("raw_text") or kwargs.get("anforandetext")
    if motion_id is not None and text is not None:
        rest = {k: v for k, v in kwargs.items() if k not in {"motion_id", "text"}}
        return pipeline_score_motion(motion_id, text, **rest)

    return pipeline_score_motion(**filtered)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    if "<" in text and ">" in text:
        import re
        return re.sub(r"<[^>]+>", " ", text)
    return text


def flush_rows(out_path: Path, rows: list[dict]) -> int:
    """Persist buffered rows to parquet and return total row count in output file."""
    if not rows:
        if out_path.exists():
            try:
                return len(pd.read_parquet(out_path))
            except Exception:
                return 0
        return 0

    chunk_df = pd.DataFrame(rows)
    writer = ParquetClassificationWriter(output_path=out_path)
    return writer.write(chunk_df)


def _flush_rhetoric_rows(rhet_path: Path, rows: list[dict]) -> int:
    """Persist generated rhetoric rows to parquet and return output row count."""
    if not rows:
        if rhet_path.exists():
            try:
                return len(pd.read_parquet(rhet_path))
            except Exception:
                return 0
        return 0

    chunk_df = pd.DataFrame(rows)
    if rhet_path.exists():
        try:
            prev = pd.read_parquet(rhet_path)
            out_df = pd.concat([prev, chunk_df], ignore_index=True)
            if "speech_id" in out_df.columns:
                out_df = out_df.drop_duplicates(subset=["speech_id"], keep="last")
        except Exception:
            out_df = chunk_df
    else:
        out_df = chunk_df

    rhet_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(rhet_path, index=False, compression="zstd")
    return len(out_df)


def _build_rhetoric_predictor(model_name: str, device: int, hypothesis_template: str):
    """Create a lazy zero-shot rhetoric predictor for missing speech rhetoric rows."""
    from transformers import pipeline

    zsc = pipeline("zero-shot-classification", model=model_name, device=device)
    labels = ["irony", "sarcasm", "posturing", "none"]

    def _predict(text: str) -> dict:
        if not text:
            return {"irony": 0.0, "sarcasm": 0.0, "posturing": 0.0, "none": 1.0, "top_label": "none"}

        out = zsc(text, labels, multi_label=True, hypothesis_template=hypothesis_template)
        label_scores = {str(lbl): float(scr) for lbl, scr in zip(out.get("labels", []), out.get("scores", []))}
        for lbl in labels:
            label_scores.setdefault(lbl, 0.0)
        top_label = max(labels, key=lambda x: label_scores.get(x, 0.0))
        return {
            "irony": float(label_scores.get("irony", 0.0)),
            "sarcasm": float(label_scores.get("sarcasm", 0.0)),
            "posturing": float(label_scores.get("posturing", 0.0)),
            "none": float(label_scores.get("none", 0.0)),
            "top_label": top_label,
        }

    return _predict


def _read_existing_speech_ids(out_path: Path) -> tuple[set[str], int]:
    """(legacy) Read existing speech_ids using pandas iteration."""
    existing_speech_ids: set[str] = set()
    total_rows_written = 0

    if not out_path.exists():
        return existing_speech_ids, total_rows_written

    try:
        for chunk in pd.read_parquet(out_path, columns=["speech_id"]).itertuples(index=False):
            if hasattr(chunk, 'speech_id') and chunk.speech_id is not None:
                existing_speech_ids.add(str(chunk.speech_id))
        total_rows_written = len(existing_speech_ids) * 7
        print(f"[RESUME] Loaded {len(existing_speech_ids)} existing speech IDs", flush=True)
    except Exception as e:
        print(f"[WARNING] Failed to load existing classifications: {e}", flush=True)
        print("[WARNING] Proceeding without resume; speeches will be reprocessed.", flush=True)
        existing_speech_ids = set()
        total_rows_written = 0

    return existing_speech_ids, total_rows_written


# ── Timeout-guarded speech classification ──

def _classify_speech_with_timeout(
    speech_id: str,
    text: str,
    defs: dict,
    matcher,
    use_zero_shot: bool,
    topic_dists,
    speech_meta_clf,
    use_ollama: bool,
    rhetoric_scores,
    timeout_seconds: int,
) -> list | None:
    """Run pipeline_score_speech in a sub-thread with a hard timeout."""
    results_holder = []
    exception_holder = []

    def _worker():
        try:
            results = pipeline_score_speech(
                speech_id=speech_id,
                text=text,
                categories=defs,
                party=None,
                embedding_matcher=matcher,
                use_zero_shot=use_zero_shot,
                topic_distributions=topic_dists,
                speech_meta_clf=speech_meta_clf,
                use_ollama=use_ollama,
                rhetoric_scores=rhetoric_scores,
            )
            results_holder.append(results)
        except Exception as e:
            exception_holder.append(e)

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)

    if thread.is_alive():
        raise TimeoutError(
            f"Speech {speech_id} timed out after {timeout_seconds}s "
            f"(text length={len(text)}, preview={text[:200]})"
        )

    if exception_holder:
        raise exception_holder[0]

    if results_holder:
        return results_holder[0]

    return []


def _log_hanged_speech(speech_id: str, text: str) -> None:
    """Append a hang record to the hang log for skip-on-resume."""
    hang_info = {
        "speech_id": speech_id,
        "text_length": len(text) if text else 0,
        "text_preview": text[:500] if text else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _hang_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(_hang_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(hang_info, ensure_ascii=False) + "\n")


def _read_hanged_speech_ids() -> set[str]:
    """Read previously-hanged speech IDs from the hang log."""
    hanged: set[str] = set()
    if _hang_log_path.exists():
        try:
            with open(_hang_log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        sid = rec.get("speech_id")
                        if sid:
                            hanged.add(sid)
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
    return hanged


# ── Graceful-shutdown flush (Ctrl+C / SIGTERM) ──
_sigterm_ctx = {
    "rows": None,
    "out_path": None,
    "rhet_out_path": None,
    "generated_rhet_rows": None,
    "existing_speech_ids": None,
    "newly_classified_ids": None,
    "total_rows_written": 0,
}


def _sigterm_handler(signum, frame):
    """Flush buffered rows on SIGTERM/SIGINT before exiting."""
    rows = _sigterm_ctx.get("rows")
    out_path = _sigterm_ctx.get("out_path")
    rhet_out_path = _sigterm_ctx.get("rhet_out_path")
    generated_rhet_rows = _sigterm_ctx.get("generated_rhet_rows")
    existing_speech_ids = _sigterm_ctx.get("existing_speech_ids")
    newly_classified_ids = _sigterm_ctx.get("newly_classified_ids")

    if rows is not None and len(rows) > 0 and out_path is not None:
        print(f"\n[SIGNAL] Received signal {signum}. Flushing {len(rows)} buffered rows...", flush=True)
        try:
            flushed_ids = {str(x.get("speech_id")) for x in rows if x.get("speech_id") is not None}
            new_count = flush_rows(out_path, rows)
            _sigterm_ctx["total_rows_written"] = new_count
            if newly_classified_ids is not None:
                newly_classified_ids.update(flushed_ids)
            if rhet_out_path is not None and generated_rhet_rows:
                rhet_df = pd.DataFrame(generated_rhet_rows)
                if rhet_out_path.exists():
                    try:
                        prev = pd.read_parquet(rhet_out_path)
                        rhet_df = pd.concat([prev, rhet_df], ignore_index=True)
                        if "speech_id" in rhet_df.columns:
                            rhet_df = rhet_df.drop_duplicates(subset=["speech_id"], keep="last")
                    except:
                        pass
                rhet_df.to_parquet(rhet_out_path, index=False, compression="zstd")
            print(f"[SIGNAL] Flushed {len(rows)} rows. Exiting.", flush=True)
        except Exception as e:
            print(f"[SIGNAL] Flush failed: {e}", flush=True)
    sys.exit(1)


signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)


def _read_existing_speech_ids_fast(out_path: Path) -> tuple[set[str], int]:
    """Read existing speech_ids using pyarrow batch iteration (fast, memory-efficient)."""
    existing_speech_ids: set[str] = set()
    total_rows = 0
    out_str = str(out_path)

    if not Path(out_str).exists():
        print(f"[RESUME] Output file not found: {out_str}. Starting fresh.", flush=True)
        return existing_speech_ids, total_rows

    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(out_str)
        total_rows = pf.metadata.num_rows
        print(f"[RESUME] Found existing output: {total_rows} rows in {out_str}", flush=True)

        if total_rows == 0:
            return existing_speech_ids, total_rows

        schema = pf.schema_arrow
        col_names = schema.names
        print(f"[RESUME] Schema columns: {col_names}", flush=True)

        speech_col = "speech_id"
        if speech_col not in col_names:
            alt = [c for c in col_names if "speech" in c.lower() or "anforande" in c.lower() or "id" in c.lower()]
            if alt:
                speech_col = alt[0]
                print(f"[RESUME] 'speech_id' not found, using '{speech_col}' instead", flush=True)
            else:
                print(f"[RESUME] No ID column found. Columns: {col_names}", flush=True)
                return existing_speech_ids, total_rows

        for batch in pf.iter_batches(batch_size=50000, columns=[speech_col]):
            arr = batch.column(speech_col)
            for val in arr.to_pylist():
                if val is not None:
                    existing_speech_ids.add(str(val))

        print(f"[RESUME] Loaded {len(existing_speech_ids)} existing speech IDs from {total_rows} rows", flush=True)
    except Exception as e:
        print(f"[RESUME] FAILED to read existing classifications: {e}", flush=True)
        print(f"[RESUME] Falling back to pandas-based reader...", flush=True)
        try:
            df = pd.read_parquet(out_str, columns=["speech_id"])
            for sid in df["speech_id"].dropna().unique():
                existing_speech_ids.add(str(sid))
            total_rows = len(df)
            print(f"[RESUME] Pandas fallback loaded {len(existing_speech_ids)} existing speech IDs from {total_rows} rows", flush=True)
        except Exception as e2:
            print(f"[RESUME] Pandas fallback ALSO failed: {e2}", flush=True)
            print(f"[RESUME] Proceeding WITHOUT resume (will reprocess ALL speeches)", flush=True)
            existing_speech_ids = set()
            total_rows = 0

    return existing_speech_ids, total_rows


def main():
    global _current_speech_id, _current_speech_text

    from swedish_parliament_policy_classifier.cli_base import (
        apply_resource_controls,
        build_common_parser,
        start_experiment,
    )

    p = build_common_parser("Parquet-first speech classification")
    p.add_argument("--input-dir", default="data/speeches/parquet")
    p.add_argument("--out", default="data/parquet/speech_classifications.parquet")
    p.add_argument("--rhetoric-parquet", default=None, help="Path to speech_rhetoric_labels.parquet to include rhetoric scores")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-embeddings", dest="use_embeddings", action="store_false", help="Disable embedding matcher")
    p.add_argument("--no-zero-shot", dest="use_zero_shot", action="store_false", help="Disable zero-shot signal")
    p.add_argument("--ollama", dest="use_ollama", action="store_true", help="Enable Ollama LLM fallback")
    p.add_argument("--quiet", dest="quiet", action="store_true")
    p.add_argument("--flush-every", type=int, default=1000, help="Flush buffered classifications every N speeches")
    p.add_argument("--cuda-cache-every", type=int, default=200, help="Clear CUDA cache every N speeches (0 disables)")
    p.add_argument("--auto-generate-rhetoric", action="store_true", help="Generate missing rhetoric scores on the fly")
    p.add_argument("--rhetoric-model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
    p.add_argument("--rhetoric-device", type=int, default=None)
    p.add_argument("--rhetoric-hypothesis-template", default="Det här uttalandet är {}.")
    p.add_argument("--persist-generated-rhetoric", action="store_true")
    p.add_argument("--generated-rhetoric-out", default=None)
    p.add_argument("--min-total-input-rows", type=int, default=100000, help="Fail if speech parquet has fewer rows")
    p.add_argument("--speech-timeout", type=int, default=300, help="Max seconds per speech classification (0=disabled)")
    args = p.parse_args()

    ctx = apply_resource_controls(args)
    throttle = ctx["throttle"]
    run = start_experiment(args, "classify-speeches-parquet", experiment_name="speech-classification")

    input_dir = Path(args.input_dir)
    out_path = Path(args.out)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        print("No speech parquet files found in", input_dir)
        return 1

    input_rows = 0
    for pf in files:
        try:
            import pyarrow.parquet as pq
            input_rows += int(pq.ParquetFile(pf).metadata.num_rows)
        except Exception:
            input_rows += len(pd.read_parquet(pf))
    if input_rows < args.min_total_input_rows:
        raise ValueError(
            f"Input corpus is too small ({input_rows} rows). "
            f"Expected at least {args.min_total_input_rows}; use full parquet shards."
        )

    defs = load_definitions()
    defs_snapshot = snapshot_definitions(version_prefix="speech-classifier-defs")
    defs_manifest = Path("logs") / "definitions_snapshot_speech_classifier.json"
    write_snapshot_manifest(defs_manifest, defs_snapshot)
    topic_dists = load_topic_distributions() if load_topic_distributions else None

    # Load precomputed rhetoric scores
    rhet_map = {}
    rhet_out_path = None
    generated_rhet_rows: list[dict] = []
    if args.rhetoric_parquet:
        try:
            rhet_df = pd.read_parquet(args.rhetoric_parquet)
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
        except Exception as e:
            print("Failed to read rhetoric parquet:", e)

    if args.persist_generated_rhetoric:
        if args.generated_rhetoric_out:
            rhet_out_path = Path(args.generated_rhetoric_out)
        elif args.rhetoric_parquet:
            rhet_out_path = Path(args.rhetoric_parquet)
        else:
            rhet_out_path = Path("data/parquet/speech_rhetoric_labels_autogen.parquet")

    rhetoric_predictor = None
    if args.auto_generate_rhetoric:
        try:
            rhetoric_device = args.rhetoric_device
            if rhetoric_device is None:
                try:
                    import torch
                    rhetoric_device = 0 if torch.cuda.is_available() else -1
                except Exception:
                    rhetoric_device = -1
            rhetoric_predictor = _build_rhetoric_predictor(
                model_name=args.rhetoric_model,
                device=rhetoric_device,
                hypothesis_template=args.rhetoric_hypothesis_template,
            )
            print(f"Auto rhetoric generation enabled (device={rhetoric_device}, model={args.rhetoric_model})")
        except Exception as e:
            print("Failed to initialize auto rhetoric predictor:", e)
            rhetoric_predictor = None

    # Try embedding matcher (optional)
    matcher = None
    if args.use_embeddings:
        try:
            matcher = EmbeddingMatcher()
            if matcher.model is None:
                matcher = None
            else:
                # Warm up the SentenceTransformer to avoid JIT compilation delay on first speech
                print("Warming up embedding model...", flush=True)
                matcher.encode(["test"])
                print("Embedding model warmup complete.", flush=True)
        except Exception as e:
            print("Embedding matcher unavailable:", e)
            matcher = None

    speech_meta_clf = None
    try:
        speech_meta_clf = _load_speech_meta_classifier()
        if speech_meta_clf is not None:
            print("Speech meta-classifier loaded.", flush=True)
    except Exception as e:
        print(f"Speech meta-classifier not loaded: {e}", file=sys.stderr)

    meta_clf = None
    if load_meta_classifier is not None and speech_meta_clf is None:
        try:
            meta_clf = load_meta_classifier()
        except Exception as e:
            print(f"Motion meta-classifier not loaded: {e}", file=sys.stderr)
            meta_clf = None

    # Load previously-hanged speech IDs to skip them on resume
    hanged_speech_ids = _read_hanged_speech_ids()
    if hanged_speech_ids:
        print(f"[RESUME] Loaded {len(hanged_speech_ids)} previously-hanged speech IDs to skip", flush=True)

    # Read existing speech IDs for resume (using fast pyarrow-based reader)
    existing_speech_ids, total_rows_written = _read_existing_speech_ids_fast(out_path)
    newly_classified_ids: set[str] = set()  # IDs classified in this run, not yet in original file
    _sigterm_ctx["newly_classified_ids"] = newly_classified_ids

    rows = []
    processed = 0
    start = time.time()
    _last_sp_time = time.time()

    # EWMA-based speed tracker (0.0025 alpha ≈ 400-speech half-life)
    _ewma_sps = None  # exponentially-weighted seconds-per-speech

    # Compute how many new speeches we expect
    new_to_classify = input_rows - len(existing_speech_ids) - len(hanged_speech_ids)
    print(
        f"[STATUS] {len(existing_speech_ids)} already classified | "
        f"{len(hanged_speech_ids)} hung/skipped | "
        f"{new_to_classify} new to process | "
        f"{input_rows} total in input files",
        flush=True,
    )

    pbar = None
    if not args.quiet:
        pbar = tqdm(
            total=new_to_classify if args.limit is None else min(args.limit, new_to_classify),
            desc="speeches",
            unit="rows",
            mininterval=3.0,
            initial=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]",
            file=open(os.devnull, "w"),
        )

    torch_mod = None
    cuda_available = False
    if args.cuda_cache_every and args.cuda_cache_every > 0:
        try:
            import torch as _torch
            torch_mod = _torch
            cuda_available = bool(torch_mod.cuda.is_available())
        except Exception:
            torch_mod = None
            cuda_available = False

    for f in files:
        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f"SKIP unreadable parquet {f}: {e}")
            continue
        if "anforande_id" not in df.columns or "anforandetext" not in df.columns:
            continue

        # Build a dedup set for this file: existing + already classified in this run + hung
        file_seen = set()
        file_seen.update(existing_speech_ids)
        file_seen.update(newly_classified_ids)
        file_seen.update(hanged_speech_ids)

        for idx, r in df.iterrows():
            raw_speech_id = r.get("anforande_id")
            speech_id = str(raw_speech_id) if raw_speech_id is not None else None
            if not speech_id or speech_id in ("nan", "None", ""):
                continue
            if speech_id in file_seen:
                continue

            raw_text = r.get("anforandetext") or ""
            if not isinstance(raw_text, str):
                raw_text = str(raw_text) if raw_text is not None else ""
            text = _strip_html(raw_text)

            _current_speech_id = speech_id
            _current_speech_text = text

            rhetoric_scores = rhet_map.get(speech_id)
            if rhetoric_scores is None and rhetoric_predictor is not None:
                try:
                    rhetoric_scores = rhetoric_predictor(text[:2500])
                    rhet_map[speech_id] = rhetoric_scores
                    if args.persist_generated_rhetoric:
                        generated_rhet_rows.append(
                            {
                                "speech_id": speech_id,
                                "irony": float(rhetoric_scores.get("irony", 0.0)),
                                "sarcasm": float(rhetoric_scores.get("sarcasm", 0.0)),
                                "posturing": float(rhetoric_scores.get("posturing", 0.0)),
                                "none": float(rhetoric_scores.get("none", 0.0)),
                                "top_label": rhetoric_scores.get("top_label", None),
                                "generated_at": pd.Timestamp.utcnow().isoformat(),
                            }
                        )
                except Exception:
                    rhetoric_scores = None

            # Classify with optional timeout
            try:
                if args.speech_timeout and args.speech_timeout > 0:
                    results = _classify_speech_with_timeout(
                        speech_id=speech_id,
                        text=text,
                        defs=defs,
                        matcher=matcher,
                        use_zero_shot=args.use_zero_shot,
                        topic_dists=topic_dists,
                        speech_meta_clf=speech_meta_clf,
                        use_ollama=args.use_ollama,
                        rhetoric_scores=rhetoric_scores,
                        timeout_seconds=args.speech_timeout,
                    )
                else:
                    results = pipeline_score_speech(
                        speech_id=speech_id,
                        text=text,
                        categories=defs,
                        party=None,
                        embedding_matcher=matcher,
                        use_zero_shot=args.use_zero_shot,
                        topic_distributions=topic_dists,
                        speech_meta_clf=speech_meta_clf,
                        use_ollama=args.use_ollama,
                        rhetoric_scores=rhetoric_scores,
                    )
            except TimeoutError as e:
                print(f"\n[TIMEOUT] {e}", file=sys.stderr)
                _log_hanged_speech(speech_id, text)
                if rows:
                    flushed_ids = {str(x.get("speech_id")) for x in rows if x.get("speech_id") is not None}
                    total_rows_written = flush_rows(out_path, rows)
                    rows = []
                    newly_classified_ids.update(flushed_ids)
                continue
            except Exception as e:
                print(f"\n[ERROR] Failed to classify speech_id={speech_id}: {e}", file=sys.stderr)
                print(f"[ERROR] Text length: {len(text)}, preview: {text[:200]}", file=sys.stderr)
                traceback.print_exc()
                results = []

            # ── Track per-speech time with EWMA ──
            now = time.time()
            speech_sec = now - _last_sp_time
            _last_sp_time = now
            # Only count actual classification time, not flush/GC overhead
            if speech_sec < 10.0:  # ignore outlier flushes (>10s)
                if processed >= 5:  # skip warmup speeches from ETA
                    if _ewma_sps is None:
                        _ewma_sps = speech_sec
                    else:
                        _ewma_sps = 0.999 * _ewma_sps + 0.001 * speech_sec

            confidences = [float(rr.normalized_weight) for rr in results] if results else [0.0]
            conf = max(confidences) if confidences else 0.0

            probs_by_category = {str(rr.category): float(rr.normalized_weight) for rr in results}
            if isinstance(defs, dict):
                for cat in defs.keys():
                    probs_by_category.setdefault(str(cat), 0.0)
            all_category_probs_json = json.dumps(probs_by_category, ensure_ascii=False, sort_keys=True)

            for rr in results:
                rows.append(
                    {
                        "speech_id": rr.motion_id,
                        "category": rr.category,
                        "raw_score": float(rr.raw_score),
                        "normalized_weight": float(rr.normalized_weight),
                        "category_probability": float(rr.normalized_weight),
                        "all_category_probs_json": all_category_probs_json,
                        "matched_rules": json.dumps(rr.matched_rules, ensure_ascii=False),
                        "classifier_version": rr.classifier_version,
                        "created_at": rr.created_at.isoformat(),
                        "confidence": float(conf),
                        "label_source": "auto_parquet",
                    }
                )

            processed += 1
            if pbar is not None:
                pbar.update(1)

            if processed % 20 == 0:
                elapsed = time.time() - start
                remaining = new_to_classify - processed if new_to_classify > 0 else 0
                if remaining < 0:
                    remaining = 0
                # total_done = persisted existing + new classified in this run (flushed + buffer)
                total_done = len(existing_speech_ids) + len(newly_classified_ids) + len(rows)
                total_done_display = min(total_done, input_rows) if input_rows > 0 else total_done
                # Clamp remaining to non-negative for ETA calc
                remaining = max(0, new_to_classify - processed)
                title_pct = 100 * total_done_display / input_rows if input_rows > 0 else 0
                new_pct = 100 * processed / new_to_classify if new_to_classify > 0 else 0

                # Compute ETA from recency-weighted EWMA
                if _ewma_sps is not None and remaining > 0:
                    eta_sec = remaining * _ewma_sps
                else:
                    eta_sec = 0

                if eta_sec < 60:
                    eta_str = f"{eta_sec:.0f}s"
                elif eta_sec < 3600:
                    eta_str = f"{eta_sec / 60:.0f}m"
                else:
                    hours = int(eta_sec // 3600)
                    mins = int((eta_sec % 3600) // 60)
                    eta_str = f"{hours}h{mins}m"

                sps_str = f"{_ewma_sps:.2f}s" if _ewma_sps else "warming..."

                msg = (
                    f"[PROGRESS] {processed}/{new_to_classify} ({new_pct:.0f}% new) | "
                    f"done={total_done_display}/{input_rows} ({title_pct:.0f}% total) | "
                    f"⏱ {sps_str}/sp | "
                    f"⌛ {eta_str}"
                )
                # Write directly to stderr to bypass tqdm's stdout capture
                sys.stderr.write(msg + "\n")
                sys.stderr.flush()

            if args.flush_every and args.flush_every > 0 and processed % args.flush_every == 0:
                flushed_speech_ids = {str(x.get("speech_id")) for x in rows if x.get("speech_id") is not None}
                total_rows_written = flush_rows(out_path, rows)
                rows = []
                newly_classified_ids.update(flushed_speech_ids)
                file_seen.update(flushed_speech_ids)
                if args.persist_generated_rhetoric and rhet_out_path is not None:
                    _flush_rhetoric_rows(rhet_out_path, generated_rhet_rows)
                    generated_rhet_rows = []

            if processed % 50 == 0:
                gc.collect()
                if hasattr(pd, '_cache'):
                    pd._cache.clear()

            if processed % 500 == 0:
                gc.collect(2)
                if matcher is not None and hasattr(matcher, '_cached_cat_embs'):
                    del matcher._cached_cat_embs
                    matcher._cached_cat_embs = None

            if cuda_available and torch_mod is not None and processed % args.cuda_cache_every == 0:
                try:
                    torch_mod.cuda.empty_cache()
                except Exception:
                    pass

            if args.sleep_every and args.sleep_every > 0 and args.sleep_seconds > 0 and processed % args.sleep_every == 0:
                time.sleep(args.sleep_seconds)

            if args.limit and processed >= args.limit:
                break

        if args.limit and processed >= args.limit:
            break

        del df
        gc.collect()

    total_rows_written = flush_rows(out_path, rows)
    if args.persist_generated_rhetoric and rhet_out_path is not None:
        _flush_rhetoric_rows(rhet_out_path, generated_rhet_rows)
    if pbar is not None:
        pbar.close()
    elapsed = time.time() - start
    run.log_params(
        {
            "input_dir": str(input_dir),
            "out": str(out_path),
            "cpu_fraction": args.cpu_fraction,
            "max_threads": throttle.get("max_threads"),
            "input_rows": input_rows,
            "min_total_input_rows": args.min_total_input_rows,
            "use_zero_shot": args.use_zero_shot,
            "use_embeddings": args.use_embeddings,
            "use_ollama": args.use_ollama,
            "speech_timeout": args.speech_timeout,
            "definitions_version": defs_snapshot.version,
        }
    )
    run.log_metrics(
        {
            "processed_speeches": processed,
            "rows_written": total_rows_written,
            "elapsed_seconds": elapsed,
            "speeches_per_second": (processed / elapsed) if elapsed > 0 else 0.0,
        }
    )
    run.log_artifact(str(out_path))
    run.log_artifact(str(defs_manifest))
    if args.persist_generated_rhetoric and rhet_out_path is not None and rhet_out_path.exists():
        run.log_artifact(str(rhet_out_path))
    run.end(status="FINISHED")
    print(f"Wrote {total_rows_written} rows to {out_path} (processed {processed} speeches in {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())