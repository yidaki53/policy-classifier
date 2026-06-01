#!/usr/bin/env python3
"""
Simple lookup utility for speeches.

Features:
- lookup by `--speech-id`(s)
- lookup by `--speaker` substring
- lookup by `--party`
- lookup by predicted `--category` (uses latest `logs/speech_eval_preds_*.csv` when available)
- outputs CSV or prints table with text snippets

Usage examples:
  python3 scripts/lookup_speeches.py --speech-id <id> --out logs/sample.csv
  python3 scripts/lookup_speeches.py --speaker "Åkesson" --limit 50

This script prefers a `logs/speech_eval_preds_*.csv` file for prediction columns but will fall back
to the sqlite DB `speech_classifications` table when the logs file is not present.
"""

from __future__ import annotations
import argparse
import sqlite3
import os
import glob
import pandas as pd
from typing import List, Optional


def find_latest_preds_csv(logs_dir: str = "logs") -> Optional[str]:
    # Prefer parquet preds, fall back to CSV
    p_par = os.path.join(logs_dir, "speech_eval_preds_*.parquet")
    files = sorted(glob.glob(p_par))
    if files:
        return files[-1]
    pattern = os.path.join(logs_dir, "speech_eval_preds_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def load_preds_df(preds_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if preds_csv and os.path.exists(preds_csv):
        if preds_csv.lower().endswith('.parquet'):
            return pd.read_parquet(preds_csv)
        return pd.read_csv(preds_csv)
    p = find_latest_preds_csv()
    if p:
        if p.lower().endswith('.parquet'):
            return pd.read_parquet(p)
        return pd.read_csv(p)
    return None


def find_speeches_in_parquets(parquet_dir: str, speech_ids: List[str]) -> dict:
    """Return a map speech_id -> record dict with keys talare, parti, anforandetext (snippet)"""
    found = {sid: None for sid in speech_ids}
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    for pf in files:
        try:
            # cheap check: read only ids
            cols = pd.read_parquet(pf, nrows=0).columns.tolist()
        except Exception:
            try:
                dfp = pd.read_parquet(pf)
                cols = dfp.columns.tolist()
            except Exception:
                continue
        id_cols = [c for c in ("anforande_id", "speech_id") if c in cols]
        if not id_cols:
            continue
        use_cols = [c for c in ["anforande_id", "speech_id", "anforandetext", "talare", "parti", "talare_namn"] if c in cols]
        try:
            dfp = pd.read_parquet(pf, columns=use_cols)
        except Exception:
            try:
                dfp = pd.read_parquet(pf)
            except Exception:
                continue
        for sid in speech_ids:
            if found[sid] is not None:
                continue
            col = id_cols[0]
            m = dfp[dfp[col] == sid]
            if not m.empty:
                r = m.iloc[0]
                rec = {
                    "speech_id": sid,
                    "talare": r.get("talare") or r.get("talare_namn"),
                    "parti": r.get("parti"),
                    "anforandetext": r.get("anforandetext"),
                    "source_file": pf,
                }
                found[sid] = rec
        # early exit if all found
        if all(v is not None for v in found.values()):
            break
    return found


def find_by_speaker(parquet_dir: str, speaker_substr: str, limit: int = 100) -> List[str]:
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    results = []
    for pf in files:
        try:
            dfp = pd.read_parquet(pf, columns=["anforande_id", "talare", "anforandetext", "parti"])
        except Exception:
            try:
                dfp = pd.read_parquet(pf)
            except Exception:
                continue
        if "talare" not in dfp.columns:
            continue
        mask = dfp["talare"].astype(str).str.contains(speaker_substr, case=False, na=False)
        m = dfp[mask]
        for sid in m["anforande_id"].tolist():
            results.append(sid)
            if len(results) >= limit:
                return results
    return results


def query_db_latest_classifications(db_path: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    q = """
    SELECT sc.speech_id, sc.category, sc.raw_score, sc.normalized_weight, sc.created_at
    FROM speech_classifications sc
    JOIN (
        SELECT speech_id, MAX(created_at) as max_created FROM speech_classifications GROUP BY speech_id
    ) latest ON sc.speech_id = latest.speech_id AND sc.created_at = latest.max_created
    """
    df = pd.read_sql_query(q, conn)
    conn.close()
    return df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="data/swedish_parliament.db")
    p.add_argument("--parquet-dir", default="data/speeches/parquet")
    p.add_argument("--preds-csv", default=None)
    p.add_argument("--speech-id", action="append", help="Speech id to look up (can repeat)")
    p.add_argument("--speaker", help="Search speaker name substring")
    p.add_argument("--party", help="Filter by party")
    p.add_argument("--category", help="Filter by predicted category label (e.g. far_right)")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--out", help="Write CSV to path")
    p.add_argument("--show-text", action="store_true", help="Include full speech text (may be large)")
    p.add_argument("--snippet-length", type=int, default=240)
    args = p.parse_args()

    preds_df = load_preds_df(args.preds_csv)

    speech_ids = []
    if args.speech_id:
        speech_ids = args.speech_id
    elif args.speaker:
        speech_ids = find_by_speaker(args.parquet_dir, args.speaker, limit=args.limit)
    elif args.party and preds_df is not None:
        # filter preds by party requires joining with parquet metadata
        # fallback: scan parquets for party, collect ids
        ids = []
        files = sorted(glob.glob(os.path.join(args.parquet_dir, "*.parquet")))
        for pf in files:
            try:
                dfp = pd.read_parquet(pf, columns=["anforande_id", "parti"])
            except Exception:
                try:
                    dfp = pd.read_parquet(pf)
                except Exception:
                    continue
            if "parti" not in dfp.columns:
                continue
            m = dfp[dfp["parti"].astype(str).str.lower() == args.party.lower()]
            ids.extend(m["anforande_id"].tolist())
            if len(ids) >= args.limit:
                break
        speech_ids = ids[: args.limit]
    elif args.category and preds_df is not None:
        speech_ids = preds_df[preds_df["pred"] == args.category]["speech_id"].tolist()[: args.limit]
    elif preds_df is not None:
        # no filter: return top N preds
        speech_ids = preds_df["speech_id"].tolist()[: args.limit]
    else:
        # fallback: try DB latest classifications
        try:
            dfc = query_db_latest_classifications(args.db)
            if args.category:
                dfc = dfc[dfc["category"] == args.category]
            speech_ids = dfc["speech_id"].tolist()[: args.limit]
        except Exception:
            print("No preds CSV and DB query failed. Nothing to do.")
            return

    if not speech_ids:
        print("No speeches found for given query")
        return

    # Grab parquet metadata for speech ids
    meta_map = find_speeches_in_parquets(args.parquet_dir, speech_ids)

    # Build output rows
    rows = []
    # prepare preds df lookup
    if preds_df is not None:
        prob_cols = [c for c in preds_df.columns if c.startswith("prob_")]
        preds_df["top_prob"] = preds_df[prob_cols].max(axis=1)
        preds_map = preds_df.set_index("speech_id")
    else:
        preds_map = None

    for sid in speech_ids:
        meta = meta_map.get(sid) if meta_map else None
        talare = meta.get("talare") if meta else None
        parti = meta.get("parti") if meta else None
        text = meta.get("anforandetext") if meta else None
        snippet = None
        if text is not None:
            t = str(text)
            snippet = t[: args.snippet_length].replace("\n", " ")
        truth = None
        pred = None
        top_prob = None
        if preds_map is not None and sid in preds_map.index:
            r = preds_map.loc[sid]
            truth = r.get("truth")
            pred = r.get("pred")
            top_prob = float(r.get("top_prob")) if "top_prob" in r.index else None
        rows.append({
            "speech_id": sid,
            "talare": talare,
            "parti": parti,
            "truth": truth,
            "pred": pred,
            "top_prob": top_prob,
            "snippet": snippet,
        })

    outdf = pd.DataFrame(rows)
    if args.out:
        outp = args.out
        if outp.lower().endswith('.csv'):
            outp = outp[:-4] + '.parquet'
        outdf.to_parquet(outp, index=False, compression='zstd')
        print(f"Wrote {len(outdf)} rows to {outp}")
    print(outdf.to_string(index=False))


if __name__ == "__main__":
    main()
