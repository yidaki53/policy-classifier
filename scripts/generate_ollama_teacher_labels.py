#!/usr/bin/env python3
"""Generate 7-dimension ideological probability vectors from a local ollama LLM.

Usage:
    uv run python scripts/generate_ollama_teacher_labels.py --sample-file stratified_sample_ids.txt --model qwen2.5-coder-14b-32k:latest --out logs/ollama_teacher_labels.json

The model is prompted to output a JSON object with 7 keys (far_left, left, centre_left,
centre, centre_right, right, far_right) as independent probability scores summing to 1.0.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Optional
from fractions import Fraction

import pandas as pd

# System prompt for 7-dimension ideological classification
SYSTEM_PROMPT = """Du är en expert på svensk partipolitik och retorisk analys.

Uppgift: Analysera nedanstående tal från Sveriges riksdag och bedöm den ideologiska inriktningen på ett kontunierligt 7-dimensionellt spektrum.

Kategorier (motsvarande Britannicas definitioner av politiskt spektrum):
1. far_left: Radikalt vänster, anti-kapitalistiskt, systemförändrande, revolutionärt, kollektivistiskt.
2. left: Vänster, omfördelning, arbetstagarrättigheter, välfärdsutbyggnad, progressiv beskattning.
3. centre_left: Center-vänster, socialdemokratisk, reformistisk, välfärdsstat, miljö, jämlikhet inom marknadsekonomin.
4. centre: Mitten, pragmatisk, icke-ideologisk, balanserad, evidensbaserad, kompromissvillig.
5. centre_right: Center-höger, marknadsorienterad, fiskalt konservativ, moderat reform, näringslivsvänlig.
6. right: Höger, konservativ, avreglering, suveränitet, lag och ordning, tradition, lägre skatter.
7. far_right: Extremhöger, nationalistisk, anti-immigration, kulturkonservativ, kulturbevarande, stark suveränitet.

Instruktioner:
- Bedöm talretoriken, inte partiets officiella ideologi.
- Ett tal kan innehålla drag från FLERA ideologier samtidigt. Ge en NYANSERAD fördelning.
- Varje tal får en sannolikhetsfördelning över alla 7 kategorier. Summan ska vara 1.0.
- Var inte rädd för extrema värden om retoriken är tydlig. Men undvik 0.0 eller 1.0 där det inte är absolut.
- Output endast ett JSON-objekt med exakt dessa nycklar: far_left, left, centre_left, centre, centre_right, right, far_right.
"""


JSON_PATTERN = re.compile(r'\{[^}]*\}')


def call_ollama(text: str, model: str = "qwen2.5-coder-14b-32k:latest", temp: float = 0.2) -> Optional[dict]:
    """Call local ollama and extract JSON with 7 probability scores."""
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Klassificera följande tal:\n\n{text[:3000]}"},
            ],
            options={"temperature": temp, "num_predict": 512},
        )
        content = response["message"]["content"]
        # Extract JSON object
        match = JSON_PATTERN.search(content)
        if not match:
            return None
        obj = json.loads(match.group())
        # Normalize to 7 keys using exact Fraction arithmetic
        keys = ["far_left", "left", "centre_left", "centre", "centre_right", "right", "far_right"]
        out = {k: Fraction(str(obj.get(k, 0.0))).limit_denominator(1000) for k in keys}
        total = sum(out.values())
        if total > 0:
            out = {k: v / total for k, v in out.items()}
        else:
            out = {k: Fraction(1, 7) for k in keys}
        return out
    except Exception as e:
        print(f"Ollama call failed: {e}", file=sys.stderr)
        return None


def load_speeches_from_ids(sample_ids: list[str], parquet_dir: Path) -> pd.DataFrame:
    files = sorted(parquet_dir.glob("*.parquet"))
    dfs = [pd.read_parquet(f) for f in files]
    all_df = pd.concat(dfs, ignore_index=True)
    df = all_df[all_df["anforande_id"].isin(sample_ids)]
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-file", default="stratified_sample_ids.txt")
    p.add_argument("--parquet-dir", default="data/speeches/parquet")
    p.add_argument("--model", default="qwen2.5-coder-14b-32k:latest")
    p.add_argument("--temp", type=float, default=0.2)
    p.add_argument("--sleep", type=float, default=0.05)
    p.add_argument("--out", default="logs/ollama_teacher_labels.json")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    with open(args.sample_file) as f:
        sample_ids = [line.strip() for line in f if line.strip()]

    if args.limit:
        sample_ids = sample_ids[:args.limit]

    parquet_dir = Path(args.parquet_dir)
    df = load_speeches_from_ids(sample_ids, parquet_dir)

    results = []
    for _, row in df.iterrows():
        speech_id = str(row["anforande_id"])
        text = row.get("anforandetext") or ""
        speaker = row.get("talare", "")
        party = row.get("parti", "")
        title = row.get("avsnittsrubrik", "")

        print(f"Processing {speaker} ({party}) - {speech_id}...", file=sys.stderr)
        scores = call_ollama(text, model=args.model, temp=args.temp)
        if scores is not None:
            results.append({
                "speech_id": speech_id,
                "speaker": speaker,
                "party": party,
                "title": title,
                "scores": scores,
            })
            print(f"  -> {scores}", file=sys.stderr)
        else:
            print(f"  FAILED", file=sys.stderr)

        if args.sleep > 0:
            time.sleep(args.sleep)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    def _convert_fractions(obj):
        if isinstance(obj, Fraction):
            return float(obj)
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=_convert_fractions)

    print(f"Wrote {len(results)} teacher labels to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
