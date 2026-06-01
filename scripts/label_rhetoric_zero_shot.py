#!/usr/bin/env python3
"""
Zero-shot rhetoric labelling using a local Ollama model.

Prototype: reads a small CSV with speech snippets (defaults to
`logs/selected_10_speeches_full.csv`) and writes per-speech scores for
`irony`, `sarcasm`, `posturing`, and `none`.

Usage:
  uv run python3 scripts/label_rhetoric_zero_shot.py --in logs/selected_10_speeches_full.csv --out logs/rhetoric_llm_labels.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import ollama

LOG = logging.getLogger(__name__)

DEFAULT_MODEL = "llama3.1:8b"
DEFAULT_TEMP = 0.2


def clean_html(s: str) -> str:
    if not s:
        return ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_prompt(text: str, max_chars: int = 2000) -> str:
    truncated = text[:max_chars]
    prompt = f"""
Du är en språkvetenskaplig annotatör.

Uppgift: Bedöm i vilken mån följande tal innehåller IRONI, SARKASM eller RETORISK POSTURING (uppvisning/positionstagande utan substantiellt argument).
Ge för varje kategori en siffra 0-100 som din skattning av sannolikheten att inslaget förekommer. Ge också en kort motivering (1-2 meningar).

SVARA ENDAST med ett JSON-objekt i exakt detta format:
{{"irony": <0-100>, "sarcasm": <0-100>, "posturing": <0-100>, "none": <0-100>, "reasoning": "<kort motivering>"}}

Text:
---
{truncated}
---
"""
    return prompt


def query_ollama(prompt: str, model: str, temp: float = DEFAULT_TEMP):
    try:
        resp = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}], options={"temperature": temp})
        content = resp.get("message", {}).get("content", "").strip()
        # strip fences
        if content.startswith("```"):
            content = re.sub(r"^```[a-zA-Z]*\n?", "", content)
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
        return content
    except Exception as e:
        LOG.warning("Ollama chat failed: %s", e)
        return None


def parse_json_from_text(text: str):
    if not text:
        return None
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    s = m.group(0)
    s = s.replace("'", '"')
    s = re.sub(r",\s*\}", "}", s)
    try:
        obj = json.loads(s)
        return obj
    except json.JSONDecodeError:
        # best-effort extraction
        obj = {}
        for k in ["irony", "sarcasm", "posturing", "none"]:
            pat = rf'"{k}"\s*:\s*(\d+(?:\.\d+)?)'
            mm = re.search(pat, s)
            if mm:
                obj[k] = float(mm.group(1))
        m2 = re.search(r'"reasoning"\s*:\s*"([^"]+)"', s)
        if m2:
            obj["reasoning"] = m2.group(1)
        return obj if obj else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_csv", default="logs/selected_10_speeches_full.csv")
    p.add_argument("--out", dest="out_csv", default=None)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--temp", type=float, default=DEFAULT_TEMP)
    p.add_argument("--sleep", type=float, default=0.05)
    args = p.parse_args()

    in_path = Path(args.in_csv)
    if in_path.suffix.lower() == '.parquet':
        df = pd.read_parquet(in_path)
    else:
        df = pd.read_csv(args.in_csv)
    if args.limit:
        df = df.head(args.limit)

    rows = []
    for _, r in df.iterrows():
        sid = r.get("speech_id")
        text = ""
        if "text_preview" in r and pd.notna(r["text_preview"]):
            text = r["text_preview"]
        elif "snippet" in r and pd.notna(r["snippet"]):
            text = r["snippet"]
        else:
            text = ""

        text_clean = clean_html(str(text))
        prompt = build_prompt(text_clean)
        raw = query_ollama(prompt, args.model, temp=args.temp)
        parsed = parse_json_from_text(raw) if raw else None
        irony = parsed.get("irony") if parsed and "irony" in parsed else 0.0
        sarcasm = parsed.get("sarcasm") if parsed and "sarcasm" in parsed else 0.0
        posturing = parsed.get("posturing") if parsed and "posturing" in parsed else 0.0
        none = parsed.get("none") if parsed and "none" in parsed else 0.0
        reasoning = parsed.get("reasoning") if parsed and "reasoning" in parsed else (raw[:240] if raw else "")

        try:
            irony = float(irony)
        except Exception:
            irony = 0.0
        try:
            sarcasm = float(sarcasm)
        except Exception:
            sarcasm = 0.0
        try:
            posturing = float(posturing)
        except Exception:
            posturing = 0.0
        try:
            none = float(none)
        except Exception:
            none = 0.0

        scores = {"irony": irony, "sarcasm": sarcasm, "posturing": posturing, "none": none}
        top_label = max(scores, key=lambda k: scores[k])
        top_score = scores[top_label]

        rows.append({
            "speech_id": sid,
            "irony": irony,
            "sarcasm": sarcasm,
            "posturing": posturing,
            "none": none,
            "top_label": top_label,
            "top_score": top_score,
            "reasoning": reasoning,
            "text_preview": text_clean,
        })

        time.sleep(args.sleep)

    out = args.out_csv or f"logs/rhetoric_llm_labels_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.parquet"
    out_path = Path(out)
    if out_path.suffix.lower() == '.csv':
        out_path = out_path.with_suffix('.parquet')
    pd.DataFrame(rows).to_parquet(out_path, index=False, compression='zstd')
    print("Wrote", out_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
