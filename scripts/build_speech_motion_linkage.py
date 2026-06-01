#!/usr/bin/env python3
"""Build a simple speech->motion linkage by explicit document ID, then by
party/date proximity and top-category match.

Outputs `data/parquet/speech_motions.parquet` with columns:
    speech_id, motion_id, party, speech_date, motion_date, category, speech_weight, motion_weight, days_diff,
    link_source, speech_dok_id, speech_rel_dok_id

This is a conservative, many-to-one linking: each speech first tries to resolve a
motion via the raw speech document reference (`rel_dok_id`/`dok_id`), then falls
back to the closest-in-time motion by the same party with the same top category
within `--window-days`.
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import timedelta

import glob
import json
import re

import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.analysis.speech_visualizations import (
    load_speech_classifications,
    load_speech_metadata,
)


ELECTION_YEARS = {2010, 2014, 2018, 2022, 2026}


def _clean_doc_id(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "none":
        return None
    return text


def _is_election_runup_year(year: object, window_years: int = 1) -> bool:
    try:
        y = int(year)
    except Exception:
        return False
    return any((election_year - window_years) <= y <= election_year for election_year in ELECTION_YEARS)


def _extract_motion_candidates(
    entry: dict,
    token_re: re.Pattern[str],
    motion_lower_map: dict[str, str],
    motion_aliases: dict[str, str],
) -> tuple[list[str], dict[str, str]]:
    found: list[str] = []
    source_by_motion: dict[str, str] = {}

    # 0) explicit document IDs from the raw speech row
    for key in ("rel_dok_id", "dok_id", "relaterat_id", "relaterat"):
        value = _clean_doc_id(entry.get(key))
        if not value:
            continue
        vstr = value.lower()
        mid = motion_lower_map.get(vstr) or motion_aliases.get(vstr)
        if mid:
            found.append(mid)
            source_by_motion.setdefault(mid, key)

    # 1) try metadata JSON keys
    if "metadata" in entry and entry["metadata"]:
        try:
            md = json.loads(entry["metadata"]) if isinstance(entry["metadata"], str) else entry["metadata"]
            if isinstance(md, dict):
                for k in ("dok_id", "hangar_id", "relaterat_id", "dok_id_utf8"):
                    v = md.get(k)
                    if v:
                        vstr = str(v).lower()
                        if vstr in motion_lower_map:
                            mid = motion_lower_map[vstr]
                            found.append(mid)
                            source_by_motion.setdefault(mid, f"metadata.{k}")
                        elif vstr in motion_aliases:
                            mid = motion_aliases[vstr]
                            found.append(mid)
                            source_by_motion.setdefault(mid, f"metadata.{k}")
        except Exception:
            pass

    # 2) try title
    if "title" in entry and entry["title"]:
        toks = token_re.findall(str(entry["title"]))
        for t in toks:
            tl = t.lower()
            if tl in motion_lower_map:
                mid = motion_lower_map[tl]
                found.append(mid)
                source_by_motion.setdefault(mid, "title")
            elif tl in motion_aliases:
                mid = motion_aliases[tl]
                found.append(mid)
                source_by_motion.setdefault(mid, "title")
            else:
                tt = tl.strip(':#,.')
                if tt in motion_aliases:
                    mid = motion_aliases[tt]
                    found.append(mid)
                    source_by_motion.setdefault(mid, "title")

    # 3) try text body
    if "text" in entry and entry["text"]:
        toks = token_re.findall(str(entry["text"]))
        for t in toks:
            tl = t.lower()
            if tl in motion_lower_map:
                mid = motion_lower_map[tl]
                found.append(mid)
                source_by_motion.setdefault(mid, "text")
            elif tl in motion_aliases:
                mid = motion_aliases[tl]
                found.append(mid)
                source_by_motion.setdefault(mid, "text")
            else:
                tt = tl.strip(':#,.')
                if tt in motion_aliases:
                    mid = motion_aliases[tt]
                    found.append(mid)
                    source_by_motion.setdefault(mid, "text")

    seen = set()
    out = []
    for x in found:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out, source_by_motion


def _load_betankande_motion_refs(betankande_parquet_dir: str) -> dict[str, list[str]]:
    refs: dict[str, list[str]] = {}
    for p in sorted(glob.glob(os.path.join(betankande_parquet_dir, "*.parquet"))):
        try:
            df = pd.read_parquet(p, columns=["dok_id", "ref_dok_ids"])
        except Exception:
            continue
        if df.empty or "dok_id" not in df.columns:
            continue
        for dok_id, ref_dok_ids in df[["dok_id", "ref_dok_ids"]].itertuples(index=False):
            key = _clean_doc_id(dok_id)
            if not key or not ref_dok_ids:
                continue
            try:
                ref_list = json.loads(ref_dok_ids) if isinstance(ref_dok_ids, str) else ref_dok_ids
            except Exception:
                continue
            if not isinstance(ref_list, list):
                continue
            cleaned = [_clean_doc_id(v) for v in ref_list]
            cleaned = [v for v in cleaned if v]
            if cleaned:
                refs[key] = cleaned
    return refs


def _select_candidate_motion(
    candidate_ids: list[str],
    party: str,
    cats: list[str],
    speech_date: pd.Timestamp,
    motions: pd.DataFrame,
) -> pd.Series | None:
    if not candidate_ids:
        return None
    sub = motions[motions["motion_id"].isin(candidate_ids)].copy()
    if sub.empty:
        return None

    party = str(party).strip()
    cats_set = {str(c) for c in cats if c is not None}
    if party:
        party_sub = sub[sub["party"].astype(str).str.strip() == party]
        if not party_sub.empty:
            sub = party_sub
    if cats_set:
        cat_sub = sub[sub["category"].astype(str).isin(cats_set)]
        if not cat_sub.empty:
            sub = cat_sub

    if "motion_date_parsed" in sub.columns and pd.notna(speech_date):
        sub = sub.copy()
        sub["_days_diff"] = (sub["motion_date_parsed"] - speech_date).abs().dt.days
        sub = sub.sort_values(["_days_diff", "motion_weight"], ascending=[True, False])
    else:
        sub = sub.sort_values(["motion_weight"], ascending=[False])

    if sub.empty:
        return None
    return sub.iloc[0]


def top_category_per_speech(speech_cls: pd.DataFrame) -> pd.DataFrame:
    # speech_cls: speech_id, category, normalized_weight
    df = speech_cls.copy()
    df = df.sort_values(["speech_id", "normalized_weight"], ascending=[True, False])
    top = df.groupby("speech_id", sort=False).first().reset_index()
    top = top.rename(columns={"normalized_weight": "speech_weight", "category": "category"})
    return top[["speech_id", "category", "speech_weight"]]


def top_categories_map(speech_cls: pd.DataFrame, top_n: int = 3) -> dict:
    """Return mapping speech_id -> list of top N categories (ordered)."""
    df = speech_cls.copy()
    df = df.sort_values(["speech_id", "normalized_weight"], ascending=[True, False])
    head = df.groupby("speech_id", sort=False).head(top_n)
    return head.groupby("speech_id")["category"].apply(list).to_dict()


def top_category_per_motion(classifications_path: str) -> pd.DataFrame:
    df = pd.read_parquet(classifications_path, columns=["motion_id", "category", "normalized_weight"]).copy()
    df = df.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
    top = df.groupby("motion_id", sort=False).first().reset_index()
    top = top.rename(columns={"normalized_weight": "motion_weight", "motion_id": "motion_id"})
    return top[["motion_id", "category", "motion_weight"]]


def main():
    parser = argparse.ArgumentParser(description="Link speeches to motions by party/date/category")
    parser.add_argument("--speech-classifications", default="data/parquet/speech_classifications_with_rhetoric_full.parquet")
    parser.add_argument("--speech-parquet-dir", default="data/speeches/parquet")
    parser.add_argument("--classifications", default="data/parquet/classifications.parquet")
    parser.add_argument("--normalized-motions", default="data/parquet/normalized_motions.parquet")
    parser.add_argument("--betankande-parquet-dir", default="data/betankande/parquet")
    parser.add_argument("--out", default="data/parquet/speech_motions.parquet")
    parser.add_argument("--window-days", type=int, default=90)
    parser.add_argument("--top-n", type=int, default=1, help="Consider top-N categories per speech when linking (>=1)")
    parser.add_argument("--id-only", action="store_true", help="Only link when explicit motion IDs found in speech metadata/text; skip time-based fallback")
    parser.add_argument("--semantic", action="store_true", help="Use semantic embedding similarity as a fallback for unlinked speeches")
    parser.add_argument("--semantic-top-k", type=int, default=1, help="Number of top semantic candidates to keep per speech")
    parser.add_argument("--semantic-threshold", type=float, default=0.6, help="Min similarity for semantic match (cosine or overlap score)")
    parser.add_argument("--model-name", type=str, default="paraphrase-multilingual-MiniLM-L12-v2", help="SentenceTransformer model name for embeddings")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of speeches to process (0=unlimited)")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.out) and not args.force:
        print(f"Output {args.out} exists. Use --force to overwrite.")
        return

    print("Loading speech classifications and metadata...")
    speech_cls = load_speech_classifications(args.speech_classifications)
    speech_meta = load_speech_metadata(args.speech_parquet_dir)
    if speech_meta.empty or speech_cls.empty:
        print("No speech metadata or classifications found; exiting.")
        return

    # primary/top-1 category (kept for backwards compatibility and main category field)
    speech_top = top_category_per_speech(speech_cls)
    # mapping for top-N categories (used to relax category matching)
    topn_map = top_categories_map(speech_cls, top_n=args.top_n)
    speech_df = speech_top.merge(speech_meta, left_on="speech_id", right_on="speech_id", how="left")
    speech_df = speech_df.rename(columns={"date": "speech_date", "party": "party"})
    speech_df["speech_date_parsed"] = pd.to_datetime(speech_df.get("speech_date", None), errors="coerce")
    speech_df = speech_df[speech_df["party"].notna() & speech_df["speech_date_parsed"].notna()].copy()

    print("Loading motion top categories and metadata...")
    motion_top = top_category_per_motion(args.classifications)
    # Read normalized motions. Pandas can sometimes drop or mishandle timezone-aware
    # timestamp columns when a subset of columns is requested, so prefer pyarrow
    # when available and fall back to pandas. Also attempt to extract dates from
    # the `metadata` JSON if the explicit `date` column is missing.
    try:
        nm = pd.read_parquet(args.normalized_motions, columns=["id", "party", "date"]).copy()
    except Exception:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(args.normalized_motions, columns=["id", "party", "date"])  # type: ignore
            nm = table.to_pandas()
        except Exception:
            nm = pd.read_parquet(args.normalized_motions)

    # If `date` was not present when reading with a column projection, try reading
    # the full table via pyarrow and converting to pandas to preserve timestamp types.
    if "date" not in nm.columns:
        try:
            import pyarrow.parquet as pq

            table = pq.read_table(args.normalized_motions)  # type: ignore
            nm = table.to_pandas()
        except Exception:
            nm = pd.read_parquet(args.normalized_motions)

    nm = nm.rename(columns={"id": "motion_id", "date": "motion_date", "party": "party"})

    # If motion_date is still missing or entirely NaT, try to extract from metadata JSON
    if ("motion_date" not in nm.columns) or nm["motion_date"].isna().all():
        if "metadata" in nm.columns:
            import json

            def _extract_date(m):
                try:
                    md = json.loads(m) if isinstance(m, str) else m
                    if isinstance(md, dict):
                        for k in ("systemdatum", "datum", "date", "system_date"):
                            if k in md and md[k]:
                                return md[k]
                except Exception:
                    return None

            nm["motion_date"] = nm["metadata"].apply(_extract_date)

    nm["motion_date_parsed"] = pd.to_datetime(nm.get("motion_date", None), errors="coerce")
    motions = motion_top.merge(nm, on="motion_id", how="left")
    motions = motions[motions["party"].notna() & motions["motion_date_parsed"].notna()].copy()

    # Normalize datetimes to timezone-aware UTC to avoid tz-naive vs tz-aware comparisons
    def _ensure_utc(series: pd.Series) -> pd.Series:
        try:
            if getattr(series.dt, "tz", None) is None:
                return series.dt.tz_localize("UTC")
            return series.dt.tz_convert("UTC")
        except Exception:
            return pd.to_datetime(series).dt.tz_localize("UTC", ambiguous="NaT", nonexistent="NaT")

    speech_df["speech_date_parsed"] = _ensure_utc(speech_df["speech_date_parsed"])
    motions["motion_date_parsed"] = _ensure_utc(motions["motion_date_parsed"])
    motion_lookup = motions.set_index("motion_id", drop=False)
    betankande_motion_refs = _load_betankande_motion_refs(args.betankande_parquet_dir)

    print("Linking speeches to motions (ID-first; fallback to nearest-in-time)")
    rows = []
    window = pd.Timedelta(days=args.window_days)
    # index motions by party+category for quick filter (needed for time-based or semantic fallback)
    motions_grouped = motions.groupby(["party", "category"]) if (not getattr(args, "id_only", False) or getattr(args, "semantic", False)) else None

    # also ensure nm timestamps are timezone-aware for id-based lookups
    if "motion_date_parsed" in nm.columns:
        try:
            nm["motion_date_parsed"] = _ensure_utc(nm["motion_date_parsed"])
        except Exception:
            nm["motion_date_parsed"] = pd.to_datetime(nm.get("motion_date", None), errors="coerce")

    # Build quick lookup maps for explicit id-based linking
    # nm may contain the full normalized motions table (with metadata)
    nm_index = nm.set_index("motion_id") if "motion_id" in nm.columns else None
    motion_ids = set()
    if nm_index is not None:
        motion_ids = set([str(x) for x in nm_index.index.unique() if pd.notna(x)])
    # also try to harvest dok_id/hangar aliases from nm.metadata if available
    motion_aliases = {}
    if "metadata" in nm.columns:
        key_id_col = "motion_id" if "motion_id" in nm.columns else ("id" if "id" in nm.columns else None)
        if key_id_col is not None:
            for mid, meta in nm[[key_id_col, "metadata"]].itertuples(index=False):
                try:
                    if isinstance(meta, str):
                        md = json.loads(meta)
                    else:
                        md = meta
                    if isinstance(md, dict):
                        # Primary identifier fields
                        for key in ("dok_id", "hangar_id", "dok_id_utf8", "dokid"):
                            if key in md and md[key]:
                                motion_aliases[str(md[key]).lower()] = str(mid)
                        # Beteckning (human-friendly short id like 'So291') and number variants
                        if "beteckning" in md and md["beteckning"]:
                            b = str(md["beteckning"]).strip()
                            if b:
                                bl = b.lower()
                                motion_aliases[bl] = str(mid)
                                # also add variant with space between letters and digits: 'so 291'
                                m = re.match(r"^([A-Za-zÅÄÖåäö]+)\s*([0-9]+)$", bl)
                                if m:
                                    motion_aliases[f"{m.group(1)} {m.group(2)}"] = str(mid)
                        # rm (parliamentary term) combined with beteckning: '1994/95:So291'
                        if "rm" in md and md.get("rm") and "beteckning" in md and md.get("beteckning"):
                            rm = str(md.get("rm")).strip()
                            b = str(md.get("beteckning")).strip()
                            if rm and b:
                                motion_aliases[f"{rm}:{b}".lower()] = str(mid)
                                motion_aliases[f"{rm}: {b}".lower()] = str(mid)
                                motion_aliases[f"{rm} {b}".lower()] = str(mid)
                except Exception:
                    continue
    motion_lower_map = {m.lower(): m for m in motion_ids}

    # pre-scan speech parquet files for text/metadata fields to enable id-extraction
    def _gather_speech_refs(speech_parquet_dir: str, speech_ids: set) -> dict:
        refs = {}
        for p in sorted(glob.glob(os.path.join(speech_parquet_dir, "*.parquet"))):
            try:
                df = pd.read_parquet(p)
            except Exception:
                continue
            sid_col = None
            for c in ("anforande_id", "speech_id"):
                if c in df.columns:
                    sid_col = c
                    break
            if not sid_col:
                continue
            candidate_cols = [c for c in ("dok_id", "rel_dok_id", "text", "metadata", "title", "relaterat_id", "relaterat") if c in df.columns]
            if not candidate_cols:
                continue
            subset = df[[sid_col] + candidate_cols].copy()
            subset[sid_col] = subset[sid_col].astype(str)
            for r in subset.itertuples(index=False):
                sid = getattr(r, sid_col)
                if sid not in speech_ids:
                    continue
                entry = refs.setdefault(sid, {})
                for col in candidate_cols:
                    val = getattr(r, col, None)
                    if pd.isna(val):
                        val = None
                    entry[col] = val
        return refs

    speech_ids_set = set(speech_df["speech_id"].astype(str).tolist())
    speech_refs = _gather_speech_refs(args.speech_parquet_dir, speech_ids_set)

    token_re = re.compile(r"\\b[A-Za-z0-9:/-]{3,30}\\b")

    def _find_motion_ids_in_speech(sid: str) -> tuple[list[str], dict[str, str]]:
        if sid not in speech_refs:
            return [], {}
        return _extract_motion_candidates(speech_refs[sid], token_re, motion_lower_map, motion_aliases)

    # Prepare semantic embeddings for motions if requested
    matcher = None
    motion_embs = None
    motion_norms = None
    motion_ids_list = []
    motions_for_embed = None
    motions_party_index = {}
    motion_token_sets = None
    if getattr(args, "semantic", False):
        print("Preparing semantic candidate index for motions...")
        try:
            try:
                from swedish_parliament_policy_classifier.nlp.embedding_matcher import EmbeddingMatcher as _EmbeddingMatcher
            except Exception:
                _EmbeddingMatcher = None
            matcher = _EmbeddingMatcher(model_name=args.model_name) if _EmbeddingMatcher is not None else None
            # Build representative text for each motion
            def _motion_text(row):
                parts = []
                if "title" in row and pd.notna(row["title"]):
                    parts.append(str(row["title"]))
                if "text" in row and pd.notna(row["text"]):
                    parts.append(str(row["text"]))
                # try to enrich with metadata fields
                try:
                    md = json.loads(row["metadata"]) if isinstance(row.get("metadata"), str) else row.get("metadata")
                    if isinstance(md, dict):
                        for k in ("beteckning", "rm"):
                            if k in md and md[k]:
                                parts.append(str(md[k]))
                except Exception:
                    pass
                return " ".join(parts)

            motions_for_embed = motions.copy()
            motions_for_embed["__embed_text__"] = motions_for_embed.apply(_motion_text, axis=1)
            motion_ids_list = motions_for_embed["motion_id"].astype(str).tolist()
            motion_texts = motions_for_embed["__embed_text__"].fillna("").astype(str).tolist()
            M = len(motion_texts)
            if getattr(matcher, "model", None) is not None:
                batch = 256
                embs = []
                for i in range(0, M, batch):
                    chunk = motion_texts[i : i + batch]
                    embs.append(matcher.encode(chunk))
                if embs:
                    motion_embs = np.vstack(embs)
                    motion_norms = np.linalg.norm(motion_embs, axis=1)
                else:
                    motion_embs = np.zeros((0, 1))
                    motion_norms = np.array([])
            else:
                # prepare token-set fallback
                motion_token_sets = [set(re.findall(r"\w{3,}", t.lower())) for t in motion_texts]

            # build party index
            for idx, p in enumerate(motions_for_embed["party"].astype(str).tolist()):
                motions_party_index.setdefault(p, []).append(idx)
        except Exception as e:
            print("Semantic matcher setup failed:", e)

    limit = int(getattr(args, "limit", 0) or 0)
    count = 0
    for _, s in speech_df.iterrows():
        if limit and count >= limit:
            break
        count += 1
        # precompute common fields
        party = s["party"]
        cat = s["category"]
        cats = topn_map.get(s["speech_id"], [cat])
        sdate = s["speech_date_parsed"]
        speech_ref = speech_refs.get(s["speech_id"], {})

        # First, prefer a two-hop committee bridge if the speech references a
        # betänkande document that explicitly points at motion ids.
        rel_doc_id = _clean_doc_id(speech_ref.get("rel_dok_id"))
        if rel_doc_id and rel_doc_id in betankande_motion_refs:
            chosen = _select_candidate_motion(betankande_motion_refs[rel_doc_id], party, cats, sdate, motions)
            if chosen is not None:
                motion_date_val = chosen.get("motion_date_parsed")
                days_diff_val = None
                try:
                    if pd.notna(motion_date_val) and pd.notna(sdate):
                        days_diff_val = int(abs((motion_date_val - sdate).days))
                except Exception:
                    days_diff_val = None

                rows.append(
                    {
                        "speech_id": s["speech_id"],
                        "motion_id": chosen["motion_id"],
                        "party": s.get("party"),
                        "speech_date": sdate,
                        "motion_date": motion_date_val,
                        "category": cat,
                        "speech_weight": float(s["speech_weight"]),
                        "motion_weight": float(chosen.get("motion_weight", 0.0)),
                        "days_diff": int(days_diff_val) if days_diff_val is not None else None,
                        "link_source": "betankande_ref_dok_id",
                        "speech_dok_id": _clean_doc_id(speech_ref.get("dok_id")),
                        "speech_rel_dok_id": rel_doc_id,
                        "similarity": None,
                    }
                )
                continue

        # First, prefer explicit ID-based linking if speech references a known motion id
        id_candidates, id_sources = _find_motion_ids_in_speech(s["speech_id"])
        if id_candidates:
            chosen_mid = None
            # prefer candidate whose party matches the speech party
            for mid in id_candidates:
                try:
                    # check in nm_index or motions
                    if mid in motions["motion_id"].values:
                        mrow = motions[motions["motion_id"] == mid].iloc[0]
                        if str(mrow.get("party", "")).strip() == str(s.get("party", "")).strip():
                            chosen_mid = mid
                            break
                    elif nm_index is not None and mid in nm_index.index:
                        mrow = nm_index.loc[mid]
                        if str(mrow.get("party", "")).strip() == str(s.get("party", "")).strip():
                            chosen_mid = mid
                            break
                except Exception:
                    continue
            if not chosen_mid:
                chosen_mid = id_candidates[0]

            # resolve motion row and write link (no time-window filtering for explicit ids)
            motion_row = None
            motion_weight = 0.0
            motion_date_val = None
            if chosen_mid in motion_lookup.index:
                motion_row = motion_lookup.loc[chosen_mid]
                motion_weight = float(motion_row.get("motion_weight", 0.0))
                motion_date_val = motion_row.get("motion_date_parsed")
            elif nm_index is not None and chosen_mid in nm_index.index:
                mrow = nm_index.loc[chosen_mid]
                # mrow may be a Series or DataFrame if duplicates
                if hasattr(mrow, "to_dict"):
                    mrowd = dict(mrow)
                else:
                    mrowd = mrow
                motion_weight = float(0.0)
                motion_date_val = mrowd.get("motion_date_parsed") if isinstance(mrowd, dict) else None

            # ensure utc for motion_date_val
            try:
                if pd.notna(motion_date_val):
                    if getattr(motion_date_val, "tzinfo", None) is None:
                        motion_date_val = pd.to_datetime(motion_date_val).tz_localize("UTC")
                    else:
                        motion_date_val = pd.to_datetime(motion_date_val).tz_convert("UTC")
            except Exception:
                pass

            days_diff_val = None
            try:
                if pd.notna(motion_date_val) and pd.notna(s["speech_date_parsed"]):
                    days_diff_val = int(abs((motion_date_val - s["speech_date_parsed"]).days))
            except Exception:
                days_diff_val = None

            rows.append(
                {
                    "speech_id": s["speech_id"],
                    "motion_id": chosen_mid,
                    "party": s.get("party"),
                    "speech_date": s["speech_date_parsed"],
                    "motion_date": motion_date_val,
                    "category": cat,
                    "speech_weight": float(s["speech_weight"]),
                    "motion_weight": float(motion_weight),
                    "days_diff": int(days_diff_val) if days_diff_val is not None else None,
                    "link_source": id_sources.get(chosen_mid, "explicit_id"),
                    "speech_dok_id": _clean_doc_id(speech_ref.get("dok_id")),
                    "speech_rel_dok_id": _clean_doc_id(speech_ref.get("rel_dok_id")),
                    "similarity": None,
                }
            )
            # skip fallback nearest-in-time logic when ID matched
            continue

        # If user requested ID-only linking, skip the time-based fallback
        if getattr(args, "id_only", False):
            continue

        # collect candidate motions for any of the top categories
        cand_frames = []
        for c in cats:
            key = (party, c)
            if motions_grouped is not None and key in motions_grouped.groups:
                cand_frames.append(motions_grouped.get_group(key))
        if not cand_frames:
            # No candidate motions by party+category; try semantic fallback if enabled
            if getattr(args, "semantic", False) and ( (matcher is not None and getattr(matcher, "model", None) is not None and motion_embs is not None) or motion_token_sets is not None ):
                sid = s["speech_id"]
                speech_text = None
                if sid in speech_refs:
                    speech_text = speech_refs[sid].get("text") or speech_refs[sid].get("title")
                if not speech_text:
                    speech_text = str(s.get("title") or "")
                if not speech_text:
                    continue
                chosen_mid = None
                chosen_score = None
                chosen_days = None
                # semantic model path
                if matcher is not None and getattr(matcher, "model", None) is not None and motion_embs is not None:
                    try:
                        q = matcher.encode([speech_text])[0]
                        qn = float(np.linalg.norm(q))
                        cand_idx = motions_party_index.get(str(party), list(range(len(motion_ids_list))))
                        if not cand_idx:
                            cand_idx = list(range(len(motion_ids_list)))
                        emb_sub = motion_embs[cand_idx]
                        norms_sub = motion_norms[cand_idx]
                        denom = norms_sub * (qn if qn > 0 else 1e-12)
                        scores = (emb_sub @ q) / denom
                        k = max(1, int(getattr(args, "semantic_top_k", 1)))
                        if len(scores) == 0:
                            pass
                        else:
                            if k >= len(scores):
                                order = np.argsort(-scores)
                            else:
                                order = np.argpartition(-scores, k - 1)[:k]
                                order = order[np.argsort(-scores[order])]
                            best_pos = order[0]
                            best_score = float(scores[best_pos])
                            if best_score >= float(getattr(args, "semantic_threshold", 0.6)):
                                global_idx = cand_idx[best_pos]
                                chosen_mid = motion_ids_list[global_idx]
                                chosen_score = best_score
                                mrow = motions_for_embed.iloc[global_idx]
                                motion_date_val = mrow.get("motion_date_parsed")
                                try:
                                    if pd.notna(motion_date_val):
                                        if getattr(motion_date_val, "tzinfo", None) is None:
                                            motion_date_val = pd.to_datetime(motion_date_val).tz_localize("UTC")
                                        else:
                                            motion_date_val = pd.to_datetime(motion_date_val).tz_convert("UTC")
                                        chosen_days = int(abs((motion_date_val - s["speech_date_parsed"]).days))
                                except Exception:
                                    chosen_days = None
                    except Exception:
                        chosen_mid = None
                # token-overlap fallback
                if chosen_mid is None and motion_token_sets is not None:
                    toks = set(re.findall(r"\w{3,}", str(speech_text).lower()))
                    cand_idx = motions_party_index.get(str(party), list(range(len(motion_token_sets))))
                    best_score = 0.0
                    best_idx = None
                    for idx in cand_idx:
                        mts = motion_token_sets[idx]
                        if not mts:
                            continue
                        score = len(toks & mts) / max(1, len(mts))
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                    if best_idx is not None and best_score >= float(getattr(args, "semantic_threshold", 0.6)):
                        chosen_mid = motion_ids_list[best_idx]
                        chosen_score = float(best_score)
                        try:
                            mrow = motions_for_embed.iloc[best_idx]
                            motion_date_val = mrow.get("motion_date_parsed")
                            if pd.notna(motion_date_val):
                                if getattr(motion_date_val, "tzinfo", None) is None:
                                    motion_date_val = pd.to_datetime(motion_date_val).tz_localize("UTC")
                                else:
                                    motion_date_val = pd.to_datetime(motion_date_val).tz_convert("UTC")
                                chosen_days = int(abs((motion_date_val - s["speech_date_parsed"]).days))
                        except Exception:
                            chosen_days = None
                if chosen_mid is not None:
                    rows.append(
                        {
                            "speech_id": s["speech_id"],
                            "motion_id": chosen_mid,
                            "party": party,
                            "speech_date": sdate,
                            "motion_date": motion_date_val if 'motion_date_val' in locals() else None,
                            "category": cat,
                            "speech_weight": float(s["speech_weight"]),
                            "motion_weight": float(0.0),
                            "days_diff": int(chosen_days) if chosen_days is not None else None,
                            "similarity": float(chosen_score) if chosen_score is not None else None,
                        }
                    )
                    continue
            continue
        cand = pd.concat(cand_frames, ignore_index=True)
        # filter by window
        cand = cand[(cand["motion_date_parsed"] >= (sdate - window)) & (cand["motion_date_parsed"] <= (sdate + window))]
        if cand.empty:
            # Attempt semantic fallback when requested and we have a matcher or token fallback
            if getattr(args, "semantic", False) and ( (matcher is not None and getattr(matcher, "model", None) is not None and motion_embs is not None) or motion_token_sets is not None ):
                # choose speech text for embedding/overlap
                sid = s["speech_id"]
                speech_text = None
                if sid in speech_refs:
                    speech_text = speech_refs[sid].get("text") or speech_refs[sid].get("title")
                if not speech_text:
                    speech_text = str(s.get("title") or "")
                if not speech_text:
                    continue

                chosen_mid = None
                chosen_score = None
                chosen_days = None

                # semantic model path
                if matcher is not None and getattr(matcher, "model", None) is not None and motion_embs is not None:
                    try:
                        q = matcher.encode([speech_text])[0]
                        qn = float(np.linalg.norm(q))
                        cand_idx = motions_party_index.get(str(party), list(range(len(motion_ids_list))))
                        if not cand_idx:
                            cand_idx = list(range(len(motion_ids_list)))
                        emb_sub = motion_embs[cand_idx]
                        norms_sub = motion_norms[cand_idx]
                        denom = norms_sub * (qn if qn > 0 else 1e-12)
                        scores = (emb_sub @ q) / denom
                        # select top-k
                        k = max(1, int(getattr(args, "semantic_top_k", 1)))
                        if len(scores) == 0:
                            pass
                        else:
                            if k >= len(scores):
                                order = np.argsort(-scores)
                            else:
                                order = np.argpartition(-scores, k - 1)[:k]
                                order = order[np.argsort(-scores[order])]
                            best_pos = order[0]
                            best_score = float(scores[best_pos])
                            if best_score >= float(getattr(args, "semantic_threshold", 0.6)):
                                global_idx = cand_idx[best_pos]
                                chosen_mid = motion_ids_list[global_idx]
                                chosen_score = best_score
                                # lookup motion_date
                                mrow = motions_for_embed.iloc[global_idx]
                                motion_date_val = mrow.get("motion_date_parsed")
                                try:
                                    if pd.notna(motion_date_val):
                                        if getattr(motion_date_val, "tzinfo", None) is None:
                                            motion_date_val = pd.to_datetime(motion_date_val).tz_localize("UTC")
                                        else:
                                            motion_date_val = pd.to_datetime(motion_date_val).tz_convert("UTC")
                                        chosen_days = int(abs((motion_date_val - s["speech_date_parsed"]).days))
                                except Exception:
                                    chosen_days = None
                    except Exception:
                        chosen_mid = None

                # token-overlap fallback
                if chosen_mid is None and motion_token_sets is not None:
                    toks = set(re.findall(r"\w{3,}", str(speech_text).lower()))
                    cand_idx = motions_party_index.get(str(party), list(range(len(motion_token_sets))))
                    best_score = 0.0
                    best_idx = None
                    for idx in cand_idx:
                        mts = motion_token_sets[idx]
                        if not mts:
                            continue
                        score = len(toks & mts) / max(1, len(mts))
                        if score > best_score:
                            best_score = score
                            best_idx = idx
                    if best_idx is not None and best_score >= float(getattr(args, "semantic_threshold", 0.6)):
                        chosen_mid = motion_ids_list[best_idx]
                        chosen_score = float(best_score)
                        try:
                            mrow = motions_for_embed.iloc[best_idx]
                            motion_date_val = mrow.get("motion_date_parsed")
                            if pd.notna(motion_date_val):
                                if getattr(motion_date_val, "tzinfo", None) is None:
                                    motion_date_val = pd.to_datetime(motion_date_val).tz_localize("UTC")
                                else:
                                    motion_date_val = pd.to_datetime(motion_date_val).tz_convert("UTC")
                                chosen_days = int(abs((motion_date_val - s["speech_date_parsed"]).days))
                        except Exception:
                            chosen_days = None

                if chosen_mid is not None:
                    rows.append(
                        {
                            "speech_id": s["speech_id"],
                            "motion_id": chosen_mid,
                            "party": party,
                            "speech_date": sdate,
                            "motion_date": motion_date_val if 'motion_date_val' in locals() else None,
                            "category": cat,
                            "speech_weight": float(s["speech_weight"]),
                            "motion_weight": float(0.0),
                            "days_diff": int(chosen_days) if chosen_days is not None else None,
                            "similarity": float(chosen_score) if chosen_score is not None else None,
                        }
                    )
                    continue
            continue
        # select nearest by absolute days diff
        cand = cand.copy()
        cand["days_diff"] = (cand["motion_date_parsed"] - sdate).abs().dt.days
        # drop duplicate motion_ids (may appear if multiple categories matched)
        cand = cand.drop_duplicates(subset=["motion_id"], keep="first")
        best = cand.nsmallest(1, "days_diff").iloc[0]
        rows.append(
            {
                "speech_id": s["speech_id"],
                "motion_id": best["motion_id"],
                "party": party,
                "speech_date": sdate,
                "motion_date": best["motion_date_parsed"],
                "category": cat,
                "speech_weight": float(s["speech_weight"]),
                "motion_weight": float(best.get("motion_weight", 0.0)),
                "days_diff": int(best["days_diff"]),
                "similarity": None,
            }
        )

    if not rows:
        print("No links found within the configured window.")
        # write empty parquet for downstream checks
        outdf = pd.DataFrame(columns=["speech_id", "motion_id", "party", "speech_date", "motion_date", "category", "speech_weight", "motion_weight", "days_diff", "similarity"])
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        outdf.to_parquet(args.out, index=False, compression="zstd")
        return

    outdf = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    outdf.to_parquet(args.out, index=False, compression="zstd")
    print(f"Wrote {len(outdf)} speech->motion links to {args.out}")


if __name__ == "__main__":
    main()
