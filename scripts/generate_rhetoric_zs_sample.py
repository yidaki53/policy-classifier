#!/usr/bin/env python3
"""Generate rhetoric labels for a sample using zero-shot classification.

This uses `transformers.pipeline('zero-shot-classification')` with a
Swedish-compatible NLI model to get per-label scores for rhetoric classes.

Usage:
  uv run python3 scripts/generate_rhetoric_zs_sample.py --parquet-dir data/parquet --out data/parquet/speech_rhetoric_labels_zs_sample.parquet --n 100
"""

from pathlib import Path
import argparse
import json
import math
import os
import re

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

try:
    token = None
    for k in ("HF_TOKEN", "HF_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGING_FACE_TOKEN"):
        v = os.environ.get(k)
        if v:
            token = v
            break
    if not token:
        token_file = Path.home() / ".huggingface" / "token"
        if token_file.exists():
            token = token_file.read_text(encoding="utf-8").strip()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HF_HUB_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
except Exception:
    pass

try:
    from transformers import pipeline
except Exception:
    pipeline = None


LABELS = ["irony", "sarcasm", "posturing", "none"]
HYPOTHESIS = "Den här texten är {}."


def find_text_column(df):
    for c in ("anforandetext", "text", "speech_text", "body", "content"):
        if c in df.columns:
            return c
    for c in df.columns:
        if "text" in c.lower():
            return c
    return None


def split_text_chunks(text: str, granularity: str = "paragraph"):
    if not text:
        return []
    if granularity == "paragraph":
        parts = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        if not parts:
            parts = [p.strip() for p in text.split('\n') if p.strip()]
        return parts
    if granularity == "sentence":
        parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        return parts
    if granularity == "line":
        return [l.strip() for l in text.splitlines() if l.strip()]
    return [text.strip()]


def get_best_snippet(zs_pipeline, text: str, label: str, granularity: str, hypothesis_template: str, context_size: int = 1):
    best_idx = -1
    best_score = -1.0
    best_chunk = ""
    chunks = split_text_chunks(text, granularity)
    if not chunks:
        return best_chunk, 0.0, ""
    for i, chunk in enumerate(chunks):
        if not chunk:
            continue
        try:
            outp = zs_pipeline(chunk, [label], hypothesis_template=hypothesis_template, multi_label=False)
            score = 0.0
            if outp and "scores" in outp and len(outp["scores"]) > 0:
                score = float(outp["scores"][0])
            else:
                for lab, sc in zip(outp.get("labels", []), outp.get("scores", [])):
                    if lab == label:
                        score = float(sc)
                        break
            if score > best_score:
                best_score = score
                best_chunk = chunk
                best_idx = i
        except Exception:
            continue
    if best_score < 0:
        best_score = 0.0
    # build context window around best_idx
    if best_idx < 0:
        return best_chunk, best_score, ""
    start = max(0, best_idx - int(context_size))
    end = min(len(chunks), best_idx + int(context_size) + 1)
    context_chunks = chunks[start:end]
    context_combined = "\n\n".join(context_chunks).strip()
    return best_chunk, best_score, context_combined


def main(parquet_dir: str, out: str, n: int, multi_label: bool, threshold: float, hypothesis_template: str, model: str, device: int, extract_snippet: bool = False, snip_granularity: str = "paragraph", snippet_for_all: bool = False, snip_context_size: int = 1, debug: bool = False, flush_every: int = 0, read_batch_size: int = 512):
    if pipeline is None:
        raise SystemExit("transformers not available in this environment")

    p = Path(parquet_dir)
    speech_path = p / "speech_gold_labels.parquet"
    if not speech_path.exists():
        raise SystemExit(f"speech parquet not found: {speech_path}")

    # Prepare output
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    # If streaming/flush enabled and output exists, remove it to start fresh
    if flush_every and out_p.exists():
        out_p.unlink()

    # Use pyarrow metadata to determine number of rows without loading entire file
    parquet_file = pq.ParquetFile(str(speech_path))
    total_rows = parquet_file.metadata.num_rows

    zs = pipeline("zero-shot-classification", model=model, device=device)

    # Helper to flush buffered rows to Parquet using pyarrow ParquetWriter
    writer = None
    def flush_buffer(buffer):
        nonlocal writer
        if not buffer:
            return 0
        chunk_df = pd.DataFrame(buffer)
        table = pa.Table.from_pandas(chunk_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(out_p), table.schema)
        writer.write_table(table)
        written = len(chunk_df)
        return written

    processed = 0
    buffer = []

    def process_row(r):
        # Build a single output row dict from a record/Series
        sid = r.get("speech_id") or r.get("anforande_id") or r.get("id") or None
        text = (r.get(text_col) or "")
        try:
            outp = zs(text, LABELS, hypothesis_template=hypothesis_template, multi_label=bool(multi_label))
            scores_map = {lab: 0.0 for lab in LABELS}
            for lab, score in zip(outp.get("labels", []), outp.get("scores", [])):
                scores_map[lab] = float(score)

            if not multi_label:
                ssum = sum(scores_map.values())
                if ssum <= 0 or math.isclose(ssum, 0.0):
                    norm = {k: 0.0 for k in scores_map}
                else:
                    norm = {k: float(v / ssum) for k, v in scores_map.items()}
                scores_out = norm
            else:
                scores_out = scores_map

            if multi_label:
                predicted = [k for k, v in scores_out.items() if v >= float(threshold)]
            else:
                top = max(scores_out.items(), key=lambda kv: kv[1])[0]
                predicted = [top]

            top_label = max(scores_out.items(), key=lambda kv: kv[1])[0]
            top_score = float(scores_out.get(top_label, 0.0))

            row = {
                "speech_id": sid,
                "irony": float(scores_out.get("irony", 0.0)),
                "sarcasm": float(scores_out.get("sarcasm", 0.0)),
                "posturing": float(scores_out.get("posturing", 0.0)),
                "none": float(scores_out.get("none", 0.0)),
                "top_label": str(top_label or ""),
                "top_score": top_score,
                "raw_labels": json.dumps(outp.get("labels", []), ensure_ascii=False),
                "raw_scores": json.dumps(outp.get("scores", []), ensure_ascii=False),
                "raw_scores_map": json.dumps(scores_map, ensure_ascii=False),
                "predicted_labels": json.dumps(predicted, ensure_ascii=False),
            }
            for lab in LABELS:
                row[f"pred_{lab}"] = int(lab in predicted)

            if extract_snippet:
                for lab in LABELS:
                    if snippet_for_all or (lab in predicted):
                        best_chunk, best_score, best_context = get_best_snippet(zs, text, lab, snip_granularity, hypothesis_template, snip_context_size)
                        row[f"snippet_{lab}"] = best_chunk
                        row[f"snippet_score_{lab}"] = float(best_score)
                        row[f"snippet_context_{lab}"] = best_context
                    else:
                        row[f"snippet_{lab}"] = ""
                        row[f"snippet_score_{lab}"] = 0.0
                        row[f"snippet_context_{lab}"] = ""
            return row
        except Exception:
            if debug:
                import traceback
                print(f"Exception processing speech_id={sid}", flush=True)
                traceback.print_exc()
            return {
                "speech_id": sid,
                "irony": 0.0,
                "sarcasm": 0.0,
                "posturing": 0.0,
                "none": 1.0,
                "top_label": "",
                "top_score": 0.0,
                "raw_labels": "[]",
                "raw_scores": "[]",
                "raw_scores_map": json.dumps({k: 0.0 for k in LABELS}, ensure_ascii=False),
                "predicted_labels": json.dumps([], ensure_ascii=False),
                **{f"pred_{lab}": 0 for lab in LABELS},
            }

    # Decide reading strategy: if flush requested or n >= total_rows, stream from disk
    use_streaming = bool(flush_every and flush_every > 0) or (n >= total_rows)

    text_col = None
    # progress bar: expected total rows to process
    expected_total = int(total_rows if use_streaming else min(n, total_rows))
    pbar = tqdm(total=expected_total, desc="zero-shot", unit="rows")
    if use_streaming and n >= total_rows:
        # Stream entire Parquet file in batches
        for batch in parquet_file.iter_batches(batch_size=read_batch_size):
            bdf = batch.to_pandas()
            if text_col is None:
                text_col = find_text_column(bdf)
                if text_col is None:
                    if "raw_response" in bdf.columns:
                        bdf["__text"] = bdf["raw_response"].fillna("")
                        text_col = "__text"
                    else:
                        raise SystemExit("Could not find text column in speech parquet")
            for _, r in bdf.iterrows():
                row = process_row(r)
                buffer.append(row)
                processed += 1
                pbar.update(1)
                if flush_every and len(buffer) >= flush_every:
                    written = flush_buffer(buffer)
                    buffer.clear()
        # flush remaining
        if buffer:
            written = flush_buffer(buffer)
            buffer.clear()
    else:
        # Read (and possibly sample) into pandas for smaller jobs
        df = pd.read_parquet(speech_path)
        text_col = find_text_column(df)
        if text_col is None:
            if "raw_response" in df.columns:
                df["__text"] = df["raw_response"].fillna("")
                text_col = "__text"
            else:
                raise SystemExit("Could not find text column in speech parquet")
        sample = df.sample(n=min(n, len(df)), random_state=42)
        for _, r in sample.iterrows():
            row = process_row(r)
            buffer.append(row)
            processed += 1
            pbar.update(1)
            if flush_every and len(buffer) >= flush_every:
                written = flush_buffer(buffer)
                buffer.clear()
        if buffer:
            # If streaming to disk was requested, flush via writer; otherwise build final DF
            if flush_every:
                written = flush_buffer(buffer)
                buffer.clear()
            else:
                out_df = pd.DataFrame(buffer)
                out_df.to_parquet(out_p, index=False)
                pbar.close()
                print(f"Wrote {len(out_df)} rows to {out_p}")
                return

    # close writer if used and print summary
    if writer is not None:
        writer.close()
        # attempt to read metadata for row count
        try:
            meta = pq.read_metadata(str(out_p))
            total_written = meta.num_rows
        except Exception:
            total_written = processed
        pbar.close()
        print(f"Wrote {total_written} rows to {out_p}")
    else:
        # If we reached here without writer, but didn't already write out_df, write now
        if buffer:
            out_df = pd.DataFrame(buffer)
            out_df.to_parquet(out_p, index=False)
            pbar.close()
            print(f"Wrote {len(out_df)} rows to {out_p}")
        else:
            pbar.close()
            print(f"Processed {processed} rows; no output written (check --out and --flush-every)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", default="data/parquet")
    parser.add_argument("--out", default="data/parquet/speech_rhetoric_labels_zs_sample.parquet")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--multi-label", dest="multi_label", action="store_true", help="Run zero-shot in multi-label mode")
    parser.add_argument("--threshold", type=float, default=0.5, help="Score threshold for multi-label positive assignment")
    parser.add_argument("--hypothesis-template", dest="hypothesis_template", default=HYPOTHESIS, help="Hypothesis template for zero-shot (use {} for label)")
    parser.add_argument("--model", default="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli", help="Zero-shot model to use")
    parser.add_argument("--device", type=int, default=-1, help="Device id for transformers pipeline (-1 for CPU)")
    parser.add_argument("--extract-snippet", dest="extract_snippet", action="store_true", help="Extract best paragraph/sentence for predicted labels")
    parser.add_argument("--snip-granularity", dest="snip_granularity", choices=["paragraph","sentence","line"], default="paragraph", help="Granularity for snippet extraction")
    parser.add_argument("--snippet-for-all", dest="snippet_for_all", action="store_true", help="Extract snippets for all labels (not only predicted ones)")
    parser.add_argument("--snip-context-size", dest="snip_context_size", type=int, default=1, help="Number of neighboring chunks (before/after) to include as context for snippets")
    parser.add_argument("--debug", dest="debug", action="store_true", help="Print exception tracebacks for per-row failures")
    parser.add_argument("--flush-every", dest="flush_every", type=int, default=0, help="Flush results to Parquet every N rows (0 disables streaming write)")
    parser.add_argument("--read-batch-size", dest="read_batch_size", type=int, default=512, help="Batch size for reading Parquet when streaming")
    args = parser.parse_args()
    main(args.parquet_dir, args.out, args.n, args.multi_label, args.threshold, args.hypothesis_template, args.model, args.device, args.extract_snippet, args.snip_granularity, args.snippet_for_all, args.snip_context_size, args.debug, args.flush_every, args.read_batch_size)
