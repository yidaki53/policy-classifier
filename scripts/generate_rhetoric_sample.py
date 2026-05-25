#!/usr/bin/env python3
"""Generate a small sample of speech rhetoric labels using simple heuristics.

This produces a Parquet file with columns: speech_id, irony, sarcasm,
posturing, none, top_label. It's a lightweight smoke-test to validate the
end-to-end merge/training workflow.

Usage:
  uv run python3 scripts/generate_rhetoric_sample.py --parquet-dir data/parquet --out data/parquet/speech_rhetoric_labels_sample.parquet --n 10
"""

from pathlib import Path
import argparse
import re
import json

import pandas as pd
from tqdm.auto import tqdm


def find_text_column(df):
    candidates = [
        "text",
        "speech_text",
        "body",
        "content",
        "full_text",
        "speech",
        "motion_text",
        "title",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    for c in df.columns:
        if "text" in c.lower():
            return c
    return None


def score_rhetoric(text: str):
    t = (text or "").lower()
    # Simple keyword-based heuristics for a quick smoke-test
    irony_terms = ["ironi", "ironiskt", "ironisk"]
    sarcasm_terms = ["sarkasm", "sarkastisk", "sarkastiskt"]
    posturing_terms = [
        "vi kräver",
        "vi måste",
        "vi vill",
        "vi föreslår",
        "skall",
        "ska",
        "bör",
        "måste",
        "krävs",
        "kräver",
        "hårdare",
        "tuffare",
        "hårdare tag",
        "krav",
    ]

    irony = sum(1 for s in irony_terms if s in t)
    sarcasm = sum(1 for s in sarcasm_terms if s in t)
    posturing = sum(1 for s in posturing_terms if s in t)

    # Rhetorical question detection increases posturing signal
    if re.search(r"\?\s*$", t.strip()):
        posturing += 1

    counts = [float(irony), float(sarcasm), float(posturing)]
    total = sum(counts)
    if total == 0:
        return {
            "irony": 0.0,
            "sarcasm": 0.0,
            "posturing": 0.0,
            "none": 1.0,
            "top_label": None,
        }

    irony_score = counts[0] / total
    sarcasm_score = counts[1] / total
    posturing_score = counts[2] / total
    none_score = 0.0

    scores = {
        "irony": round(float(irony_score), 3),
        "sarcasm": round(float(sarcasm_score), 3),
        "posturing": round(float(posturing_score), 3),
        "none": round(float(none_score), 3),
    }

    top_label = max(scores.items(), key=lambda kv: kv[1])[0]
    if scores[top_label] == 0:
        top_label = None
    scores["top_label"] = top_label
    return scores


def main(parquet_dir: str, out: str, n: int, seed: int):
    p = Path(parquet_dir)
    speech_path = p / "speech_gold_labels.parquet"
    if not speech_path.exists():
        raise SystemExit(f"speech parquet not found: {speech_path}")

    df = pd.read_parquet(speech_path)
    text_col = find_text_column(df)
    if text_col is None:
        # Try to extract from raw_response JSON if present
        if "raw_response" in df.columns:
            def extract_text(raw):
                try:
                    j = json.loads(raw) if isinstance(raw, str) else raw
                    if isinstance(j, dict):
                        for k in ("text", "speech_text", "content"):
                            if k in j and isinstance(j[k], str):
                                return j[k]
                except Exception:
                    return ""
                return ""

            df["__text"] = df["raw_response"].fillna("").map(extract_text)
            text_col = "__text"
        else:
            raise SystemExit("Could not locate a text column in speech_gold_labels.parquet")

    sample = df.sample(n=min(n, len(df)), random_state=seed) if n > 0 else df
    rows = []
    pbar = tqdm(total=len(sample) if hasattr(sample, '__len__') else None, desc="sample", unit="rows")
    for _, r in sample.iterrows():
        sid = r.get("speech_id") or r.get("id") or None
        txt = r.get(text_col, "") if text_col in r.index else r[text_col]
        sc = score_rhetoric(str(txt))
        rows.append({
            "speech_id": sid,
            "irony": sc["irony"],
            "sarcasm": sc["sarcasm"],
            "posturing": sc["posturing"],
            "none": sc["none"],
            "top_label": sc["top_label"],
        })
        pbar.update(1)
    pbar.close()

    out_df = pd.DataFrame(rows)
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_p, index=False)
    print(f"Wrote {len(out_df)} rows to {out_p}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", default="data/parquet")
    parser.add_argument("--out", default="data/parquet/speech_rhetoric_labels_sample.parquet")
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    main(args.parquet_dir, args.out, args.n, args.seed)
