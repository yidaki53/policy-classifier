#!/usr/bin/env python3
"""Parquet-first speech classification.

Reads `data/speeches/parquet/*.parquet`, runs `score_motion()` per speech,
and writes per-category rows to a Parquet output file.

Usage:
    uv run python3 scripts/classify_speeches_parquet.py --input-dir data/speeches/parquet --out data/parquet/speech_classifications.parquet
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm.auto import tqdm


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


def _strip_html(text: str) -> str:
    if not text:
        return ""
    if "<" in text and ">" in text:
        # basic strip for HTML-like fragments
        import re

        return re.sub(r"<[^>]+>", " ", text)
    return text


def _flush_rows(out_path: Path, rows: list[dict]) -> int:
    """Persist buffered rows to parquet and return total row count in output file."""
    if not rows:
        if out_path.exists():
            try:
                return len(pd.read_parquet(out_path))
            except Exception:
                return 0
        return 0

    chunk_df = pd.DataFrame(rows)
    if out_path.exists():
        try:
            prev = pd.read_parquet(out_path)
            out_df = pd.concat([prev, chunk_df], ignore_index=True)
        except Exception:
            out_df = chunk_df
    else:
        out_df = chunk_df

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    return len(out_df)


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
    out_df.to_parquet(rhet_path, index=False)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default="data/speeches/parquet")
    p.add_argument("--out", default="data/parquet/speech_classifications.parquet")
    p.add_argument("--rhetoric-parquet", default=None, help="Path to speech_rhetoric_labels.parquet to include rhetoric scores")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-embeddings", dest="use_embeddings", action="store_false", help="Disable embedding matcher")
    p.add_argument("--no-zero-shot", dest="use_zero_shot", action="store_false", help="Disable zero-shot signal")
    p.add_argument("--ollama", dest="use_ollama", action="store_true", help="Enable Ollama LLM fallback for speech classification")
    p.add_argument("--quiet", dest="quiet", action="store_true")
    p.add_argument("--flush-every", type=int, default=1000, help="Flush buffered classifications to parquet every N speeches")
    p.add_argument("--cuda-cache-every", type=int, default=200, help="Clear CUDA cache every N speeches (0 disables)")
    p.add_argument("--auto-generate-rhetoric", action="store_true", help="Generate missing rhetoric scores on the fly when not present in rhetoric parquet")
    p.add_argument("--rhetoric-model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", help="Model to use for on-the-fly rhetoric generation")
    p.add_argument("--rhetoric-device", type=int, default=None, help="Transformers device for rhetoric generation (e.g. 0 for GPU, -1 for CPU)")
    p.add_argument("--rhetoric-hypothesis-template", default="Det här uttalandet är {}.", help="Hypothesis template for rhetoric zero-shot generation")
    p.add_argument("--persist-generated-rhetoric", action="store_true", help="Persist on-the-fly generated rhetoric rows to parquet")
    p.add_argument("--generated-rhetoric-out", default=None, help="Optional output parquet path for generated rhetoric rows")
    args = p.parse_args()

    input_dir = Path(args.input_dir)
    out_path = Path(args.out)

    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        print("No speech parquet files found in", input_dir)
        return 1

    defs = load_definitions()
    topic_dists = load_topic_distributions() if load_topic_distributions else None

    # Optional: load precomputed rhetoric scores (speech-level)
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
        except Exception as e:
            print("Embedding matcher unavailable:", e)
            matcher = None

    # Load meta-classifier if present (speech-specific preferred)
    meta_clf = None
    if load_meta_classifier is not None:
        try:
            speech_model = Path("models/speech_meta_clf.pkl")
            speech_model_alt = Path("models/speech_meta_clf_parquet.pkl")
            if speech_model.exists():
                meta_clf = load_meta_classifier(model_path=speech_model)
            elif speech_model_alt.exists():
                try:
                    meta_clf = load_meta_classifier(model_path=speech_model_alt)
                except Exception:
                    meta_clf = None
            else:
                meta_clf = load_meta_classifier()
        except Exception:
            meta_clf = None

    rows = []
    processed = 0
    total_rows_written = 0
    start = time.time()
    pbar = None
    if not args.quiet:
        pbar = tqdm(total=args.limit if args.limit else None, desc="speeches", unit="rows")

    # If output exists and resume desired, read processed speech_ids
    existing_speech_ids = set()
    if out_path.exists():
        try:
            existing = pd.read_parquet(out_path)
            if "speech_id" in existing.columns:
                existing_speech_ids = set(existing["speech_id"].astype(str).unique())
            total_rows_written = len(existing)
        except Exception:
            existing_speech_ids = set()

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
        df = pd.read_parquet(f)
        if "anforande_id" not in df.columns or "anforandetext" not in df.columns:
            # skip incompatible files
            continue

        for _, r in df.iterrows():
            speech_id = str(r["anforande_id"]) if r.get("anforande_id") is not None else None
            if not speech_id:
                continue
            if speech_id in existing_speech_ids:
                continue

            raw_text = r.get("anforandetext") or ""
            text = _strip_html(raw_text)

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

            results = _score_motion_compat(
                motion_id=speech_id,
                text=text,
                categories=defs,
                party=None,
                embedding_matcher=matcher,
                use_zero_shot=args.use_zero_shot,
                topic_distributions=topic_dists,
                meta_clf=meta_clf,
                skip_policy_extraction=True,
                use_speech_preprocessing=True,
                use_ollama=args.use_ollama,
                rhetoric_scores=rhetoric_scores,
            )

            # compute confidence = max normalized weight across categories
            confidences = [float(rr.normalized_weight) for rr in results] if results else [0.0]
            conf = max(confidences) if confidences else 0.0

            # Keep a full per-speech probability vector for downstream analysis.
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

            if args.flush_every and args.flush_every > 0 and processed % args.flush_every == 0:
                flushed_speech_ids = {str(x.get("speech_id")) for x in rows if x.get("speech_id") is not None}
                total_rows_written = _flush_rows(out_path, rows)
                rows = []
                existing_speech_ids.update(flushed_speech_ids)
                if args.persist_generated_rhetoric and rhet_out_path is not None:
                    _flush_rhetoric_rows(rhet_out_path, generated_rhet_rows)
                    generated_rhet_rows = []

            # memory flush
            if processed % 50 == 0:
                gc.collect()

            if cuda_available and torch_mod is not None and processed % args.cuda_cache_every == 0:
                try:
                    torch_mod.cuda.empty_cache()
                except Exception:
                    pass

            if args.limit and processed >= args.limit:
                break

        if args.limit and processed >= args.limit:
            break

    total_rows_written = _flush_rows(out_path, rows)
    if args.persist_generated_rhetoric and rhet_out_path is not None:
        _flush_rhetoric_rows(rhet_out_path, generated_rhet_rows)
    if pbar is not None:
        pbar.close()
    elapsed = time.time() - start
    print(f"Wrote {total_rows_written} rows to {out_path} (processed {processed} speeches in {elapsed:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
