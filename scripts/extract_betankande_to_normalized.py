#!/usr/bin/env python3
"""Convert existing betankande Parquet files to normalized schema for classification.

The existing betankande parquet (data/betankande/parquet/) contains:
  dok_id, rm, beteckning, organ, datum, titel, ref_dok_ids, ref_dok_count

This script normalizes into:
  id, title, text (titel), date (datum), party (organ), doc_type ('betankande')

The 'organ' (committee) is used as a proxy party since betankanden reflect
committee positions rather than individual party authorship.
Outputs to data/parquet/betankande_normalized.parquet.

Usage:
    uv run python scripts/extract_betankande_to_normalized.py
    uv run python scripts/extract_betankande_to_normalized.py --src data/betankande/parquet --out data/parquet/betankande_normalized.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


# Committee codes -> descriptive names (for better classification context)
_COMMITTEE_NAMES = {
    "AU": "Arbetsmarknadsutskottet",
    "CU": "Civilutskottet",
    "FiU": "Finansutskottet",
    "FöU": "Försvarsutskottet",
    "JuU": "Justitieutskottet",
    "KrU": "Kulturutskottet",
    "KU": "Konstitutionsutskottet",
    "LU": "Lagutskottet",
    "MU": "Miljö- och jordbruksutskottet",
    "NU": "Näringsutskottet",
    "SfU": "Socialförsäkringsutskottet",
    "SkU": "Skatteutskottet",
    "UbU": "Utbildningsutskottet",
    "UU": "Utrikesutskottet",
    "SoU": "Socialutskottet",
    "TU": "Trafikutskottet",
    "UoF": "Utrikesförsvarsutskottet",
    "HUS": "Hem- och omvårdnadsutskottet",
    "EV": "Europautskottet",
}


def normalize_betankande(
    src_dir: str = "data/betankande/parquet",
    out_path: str = "data/parquet/betankande_normalized.parquet",
    force: bool = False,
) -> int:
    src = Path(src_dir)
    out = Path(out_path)

    if out.exists() and not force:
        existing = pd.read_parquet(out)
        print(f"SKIP {out}: already exists ({len(existing)} rows). Use --force to overwrite.", file=sys.stderr)
        return len(existing)

    parquet_files = sorted(src.glob("bet-*.parquet"))
    print(f"Found {len(parquet_files)} betankande parquet files", file=sys.stderr)

    if not parquet_files:
        print("No betankande parquet files found.", file=sys.stderr)
        empty = pd.DataFrame(columns=["id", "title", "text", "date", "party", "doc_type", "organ", "rm"])
        out.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out, index=False, compression="zstd")
        return 0

    all_rows = []
    for pf in parquet_files:
        print(f"READ {pf.name} ...", file=sys.stderr)
        try:
            df = pd.read_parquet(pf)
            if df.empty:
                continue

            rows = []
            for _, r in df.iterrows():
                dok_id = str(r.get("dok_id") or "").strip()
                if not dok_id:
                    continue

                titel = str(r.get("titel") or "").strip()
                datum = r.get("datum")
                organ = str(r.get("organ") or "").strip()

                # Build enriched text: title + committee context
                organ_label = _COMMITTEE_NAMES.get(organ, organ)
                text = f"{titel}\n\nUtskott: {organ_label}"

                # Format date
                date_str = None
                if datum is not None:
                    if hasattr(datum, "isoformat"):
                        date_str = datum.isoformat()
                    else:
                        date_str = str(datum)

                rm = str(r.get("rm") or "").strip()
                rows.append({
                    "id": dok_id,
                    "title": titel,
                    "text": text,
                    "date": date_str,
                    "party": organ,  # organ is the "party" proxy for betankande
                    "doc_type": "betankande",
                    "organ": organ,
                    "rm": rm if rm else None,
                })
            if rows:
                all_rows.append(pd.DataFrame(rows))
                print(f"  {len(rows)} rows from {pf.name}", file=sys.stderr)
        except Exception as e:
            print(f"  FAILED {pf.name}: {e}", file=sys.stderr)

    if not all_rows:
        print("No rows extracted, creating empty output", file=sys.stderr)
        empty = pd.DataFrame(columns=["id", "title", "text", "date", "party", "doc_type", "organ", "rm"])
        out.parent.mkdir(parents=True, exist_ok=True)
        empty.to_parquet(out, index=False, compression="zstd")
        return 0

    combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["id"], keep="first")

    # Parse dates properly
    if "date" in combined.columns:
        combined["date"] = pd.to_datetime(combined["date"], errors="coerce", utc=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False, compression="zstd")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"Wrote {len(combined)} total rows ({size_mb:.1f} MB) -> {out}", file=sys.stderr)
    return len(combined)


def main():
    parser = argparse.ArgumentParser(description="Normalize betankande Parquet files to classification schema")
    parser.add_argument("--src", default="data/betankande/parquet")
    parser.add_argument("--out", default="data/parquet/betankande_normalized.parquet")
    parser.add_argument("--force", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    n = normalize_betankande(args.src, args.out, force=args.force)
    print(f"Normalized {n} betankande rows.")


if __name__ == "__main__":
    main()

if False:
    # Graphify hint: betankande uses the same classification pipeline as motions
    # Key difference: organ (committee code) is used as party proxy instead of actual party
    from swedish_parliament_policy_classifier.exports import load_definitions, classify_motion