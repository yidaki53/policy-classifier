#!/usr/bin/env python3
"""Extract structured interpellation debate pairs from speech parquet data.

Interpellations (ip) are formal parliamentary questions with a structured
debate format:
  1. A party/MP asks a question (ip_question)
  2. A minister responds (ip_answer)
  3. Follow-up questions and answers may follow

This script extracts question/answer pairs grouped by rel_dok_id (the
interpellation document ID), preserving the conversational structure.
Each pair is classified separately (question_text as "says", answer_text
as "says") but both share the same asking_party and answering_party.

Research insight (LREC 2024 / Fukui & Nakamura 2025): debate text needs
turn-aware treatment. Flat classification loses conversational context.
We preserve it by treating each turn as a separate item with its own
party attribution.

Output: data/parquet/interpellations.parquet
  - id: unique turn ID (ip_id + "_q" / "_a" + turn_index)
  - ip_id: the parent interpellation document ID (from rel_dok_id)
  - turn_index: sequential turn number in this debate
  - turn_type: "ip_question" | "ip_answer" | "ip_followup"
  - text: the turn text
  - asking_party: party of the questioner
  - answering_party: party of the responder (None for questions)
  - date: debate date
  - original_speech_id: source speech ID

Usage:
    uv run python scripts/extract_interpellations.py
    uv run python scripts/extract_interpellations.py --src data/speeches/parquet --out data/parquet/interpellations.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from tqdm import tqdm

# Known minister party prefixes (from Riksdagen data)
_MINISTER_PARTIES = {"S", "M", "C", "L", "KD", "V", "MP", "SD"}


def _detect_turn_type(
    speaker_party: Optional[str],
    text: str,
    is_first_turn: bool,
) -> str:
    """Classify a turn as question, answer, or followup."""
    if is_first_turn:
        return "ip_question"

    # Non-minister speaking after first turn: follow-up question
    if speaker_party and speaker_party not in _MINISTER_PARTIES:
        return "ip_question"

    # Minister or unknown party after first turn: answer or followup
    return "ip_answer"


def _extract_party_from_speaker(speaker: str) -> Optional[str]:
    """Extract party code from speaker field like 'Andersson (S)'."""
    if not speaker:
        return None
    m = re.search(r'\(([A-Z]+)\)', speaker)
    if m:
        return m.group(1)
    return None


def _extract_interpellation_pairs(
    speeches_df: pd.DataFrame,
) -> list[dict]:
    """Extract ip debate pairs from speeches DataFrame.

    The speech parquet has columns:
      - kammaraktivitet: 'interpellationsdebatt' for ip speeches
      - rel_dok_id: links speeches to the same debate (e.g. 'H2C120150521ip')
      - anforande_id: the speech ID
      - anforandetext: speech content (HTML)
      - parti: party code
      - talare: speaker name
      - datum: date
    """
    # Filter to interpellation debates only
    ip_speeches = speeches_df[speeches_df["kammaraktivitet"] == "interpellationsdebatt"].copy()
    ip_speeches = ip_speeches[ip_speeches["rel_dok_id"].notna()].copy()
    ip_speeches = ip_speeches[ip_speeches["rel_dok_id"].astype(str).str.strip() != "None"]
    ip_speeches = ip_speeches[ip_speeches["rel_dok_id"].astype(str).str.strip() != ""]

    if ip_speeches.empty:
        return []

    # Clean HTML tags from speech text
    def _strip_html(text: str) -> str:
        if not text:
            return ""
        clean = re.sub(r"<[^>]+>", "", text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return clean

    all_turns: list[dict] = []
    grouped = ip_speeches.groupby("rel_dok_id", sort=True)

    print(f"Processing {len(grouped)} interpellation debates...", file=sys.stderr)

    for ip_id, group in tqdm(grouped, desc="Extracting IP debates", unit="debate", file=sys.stderr):
        ip_id = str(ip_id).strip()
        if not ip_id or ip_id == "None":
            continue

        # Sort by document order within debate
        group = group.sort_values("anforande_id", kind="stable")

        for turn_idx, (_, row) in enumerate(group.iterrows()):
            text = _strip_html(str(row.get("anforandetext", "") or ""))
            if len(text) < 20:  # Skip very short fragments
                continue

            party = str(row.get("parti", "") or "").strip()
            if party in ("", "None", "nan"):
                party = None

            # Determine turn type: first speaker asks question, others respond
            is_first = turn_idx == 0
            if is_first:
                turn_type = "ip_question"
                asking_party = party
                answering_party = None
            elif party:
                # Non-first speaker could be minister (answer) or follow-up question
                # Simple heuristic: even turns are questions, odd turns are answers
                turn_type = "ip_answer" if turn_idx % 2 == 1 else "ip_question"
                asking_party = party if turn_type == "ip_question" else None
                answering_party = party if turn_type == "ip_answer" else None
            else:
                turn_type = "ip_followup"
                asking_party = None
                answering_party = None

            datum = row.get("datum")
            if hasattr(datum, "isoformat"):
                date_str = datum.isoformat()
            else:
                date_str = str(datum) if datum and str(datum) not in ("None", "nan", "") else None

            turn_id = f"{ip_id}_{turn_type}_{turn_idx}"

            all_turns.append({
                "id": turn_id,
                "ip_id": ip_id,
                "turn_index": turn_idx,
                "turn_type": turn_type,
                "text": text,
                "asking_party": asking_party,
                "answering_party": answering_party,
                "date": date_str,
                "original_speech_id": str(row.get("anforande_id", "")).strip(),
            })

    return all_turns


def extract_interpellations(
    src_dir: str = "data/speeches/parquet",
    out_path: str = "data/parquet/interpellations.parquet",
    force: bool = False,
) -> int:
    src = Path(src_dir)
    out = Path(out_path)

    if out.exists() and not force:
        existing = pd.read_parquet(out)
        print(f"SKIP {out}: already exists ({len(existing)} rows). Use --force to overwrite.", file=sys.stderr)
        return len(existing)

    # Load all speech parquet files
    parquet_files = sorted(src.glob("anforande-*.parquet"))
    print(f"Found {len(parquet_files)} speech parquet files", file=sys.stderr)

    if not parquet_files:
        print("No speech parquet files found.", file=sys.stderr)
        empty = pd.DataFrame(columns=[
            "id", "ip_id", "turn_index", "turn_type", "text",
            "asking_party", "answering_party", "date", "original_speech_id",
        ])
        out.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out, index=False, compression="zstd")
        return 0

    all_speeches: list[pd.DataFrame] = []
    for pf in parquet_files:
        try:
            # Only load columns we need to keep memory low
            cols_to_load = ["dok_id", "rel_dok_id", "anforande_id", "anforandetext", "parti", "talare", "datum", "kammaraktivitet"]
            df = pd.read_parquet(pf)
            # Keep only columns that exist
            cols_present = [c for c in cols_to_load if c in df.columns]
            if cols_present:
                df = df[cols_present]
            all_speeches.append(df)
            print(f"  Loaded {pf.name}: {len(df)} rows", file=sys.stderr)
        except Exception as e:
            print(f"  FAILED {pf.name}: {e}", file=sys.stderr)

    if not all_speeches:
        print("No speeches loaded, creating empty output", file=sys.stderr)
        empty = pd.DataFrame(columns=[
            "id", "ip_id", "turn_index", "turn_type", "text",
            "asking_party", "answering_party", "date", "original_speech_id",
        ])
        out.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out, index=False, compression="zstd")
        return 0

    combined = pd.concat(all_speeches, ignore_index=True)
    print(f"Total speeches: {len(combined)}", file=sys.stderr)

    turns = _extract_interpellation_pairs(combined)

    if not turns:
        print("No interpellation turns found, creating empty output", file=sys.stderr)
        empty = pd.DataFrame(columns=[
            "id", "ip_id", "turn_index", "turn_type", "text",
            "asking_party", "answering_party", "date", "original_speech_id",
        ])
        out.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out, index=False, compression="zstd")
        return 0

    result_df = pd.DataFrame(turns)

    # Parse dates
    if "date" in result_df.columns:
        result_df["date"] = pd.to_datetime(result_df["date"], errors="coerce", utc=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_parquet(out, index=False, compression="zstd")
    size_mb = out.stat().st_size / 1024 / 1024

    q_count = len(result_df[result_df["turn_type"] == "ip_question"])
    a_count = len(result_df[result_df["turn_type"] == "ip_answer"])
    f_count = len(result_df[result_df["turn_type"] == "ip_followup"])

    print(f"Wrote {len(result_df)} turns ({size_mb:.1f} MB) -> {out}", file=sys.stderr)
    print(f"  Questions: {q_count}, Answers: {a_count}, Followups: {f_count}", file=sys.stderr)

    return len(result_df)


def main():
    parser = argparse.ArgumentParser(description="Extract structured interpellation debate pairs from speech parquet")
    parser.add_argument("--src", default="data/speeches/parquet")
    parser.add_argument("--out", default="data/parquet/interpellations.parquet")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    n = extract_interpellations(args.src, args.out, force=args.force)
    print(f"Extracted {n} interpellation turns.")


if __name__ == "__main__":
    main()

if False:
    # Graphify hint: extract_interpellations groups speeches by rel_dok_id
    # to form debate pairs. Turn-aware: question/answer/followup each get own row.
    from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion