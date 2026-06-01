#!/usr/bin/env python3
"""Link every classified speech to either a motion or a vote-context target.

This script guarantees full speech coverage by combining existing speech->motion
links with deterministic fallback linking against the motion corpus. If a chosen
motion has observed votes (`motion_votes.votering_count > 0`) the action type is
`vote`; otherwise `motion`.

Outputs:
  - data/parquet/speech_action_links.parquet (default)
  - output/analysis/speech_action_links_summary.json (default)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import math
from typing import Iterable

import pandas as pd

from swedish_parliament_policy_classifier.analysis.speech_visualizations import load_speech_metadata


def _top_category_per_speech(speech_classifications_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(speech_classifications_path, columns=["speech_id", "category", "normalized_weight"]).copy()
    df["speech_id"] = df["speech_id"].astype(str)
    df["normalized_weight"] = pd.to_numeric(df["normalized_weight"], errors="coerce").fillna(0.0)
    df = df.sort_values(["speech_id", "normalized_weight"], ascending=[True, False])
    top = df.groupby("speech_id", sort=False).first().reset_index()
    return top[["speech_id", "category", "normalized_weight"]].rename(columns={"normalized_weight": "speech_weight"})


def _build_speech_link_table(speech_meta: pd.DataFrame, speech_top: pd.DataFrame) -> pd.DataFrame:
    """Build linkage base table from full speech corpus, enriching categories when present."""
    keep_cols = ["speech_id", "party", "speech_date"]
    if "intressent_id" in speech_meta.columns:
        keep_cols.append("intressent_id")
    if "speaker_name" in speech_meta.columns:
        keep_cols.append("speaker_name")

    base = speech_meta[keep_cols].copy()
    base["speech_id"] = base["speech_id"].astype(str)
    if "intressent_id" in base.columns:
        base["intressent_id"] = base["intressent_id"].fillna("").astype(str)
    else:
        base["intressent_id"] = ""
    if "speaker_name" in base.columns:
        base["speaker_name"] = base["speaker_name"].fillna("").astype(str)
    else:
        base["speaker_name"] = ""

    out = base.merge(speech_top, on="speech_id", how="left")
    out["category"] = out["category"].fillna("")
    out["speech_weight"] = pd.to_numeric(out["speech_weight"], errors="coerce").fillna(0.0)
    return out


def _load_speech_speaker_map(speech_parquet_dir: Path) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for p in sorted(speech_parquet_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(p, columns=["anforande_id", "intressent_id", "talare"])
        except Exception:
            continue
        part = pd.DataFrame(
            {
                "speech_id": df.get("anforande_id", pd.Series(dtype=str)).astype(str),
                "intressent_id": df.get("intressent_id", pd.Series(dtype=str)).fillna("").astype(str),
                "speaker_name": df.get("talare", pd.Series(dtype=str)).fillna("").astype(str),
            }
        )
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["speech_id", "intressent_id", "speaker_name"])

    out = pd.concat(rows, ignore_index=True)
    out = out.drop_duplicates(subset=["speech_id"], keep="last")
    return out


def _load_motion_signatories(motion_signatories_path: Path) -> pd.DataFrame:
    if not motion_signatories_path.exists():
        return pd.DataFrame(columns=["motion_id", "intressent_id", "signatory_name", "signatory_party", "signatory_role"])

    df = pd.read_parquet(motion_signatories_path).copy()
    required = {"motion_id", "intressent_id"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"motion_signatories parquet missing columns: {sorted(missing)}")

    df["motion_id"] = df["motion_id"].astype(str)
    df["intressent_id"] = df["intressent_id"].fillna("").astype(str)
    if "signatory_name" not in df.columns:
        df["signatory_name"] = ""
    if "signatory_party" not in df.columns:
        df["signatory_party"] = ""
    if "signatory_role" not in df.columns:
        df["signatory_role"] = ""

    return df[["motion_id", "intressent_id", "signatory_name", "signatory_party", "signatory_role"]].drop_duplicates()


def _load_motion_top_category(classifications_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(classifications_path, columns=["motion_id", "category", "normalized_weight"]).copy()
    df["motion_id"] = df["motion_id"].astype(str)
    df["normalized_weight"] = pd.to_numeric(df["normalized_weight"], errors="coerce").fillna(0.0)
    df = df.sort_values(["motion_id", "normalized_weight"], ascending=[True, False])
    top = df.groupby("motion_id", sort=False).first().reset_index()
    return top[["motion_id", "category", "normalized_weight"]].rename(columns={"normalized_weight": "motion_weight"})


def _load_motion_meta(normalized_motions_path: Path) -> pd.DataFrame:
    nm = pd.read_parquet(normalized_motions_path).copy()
    if "id" not in nm.columns:
        raise ValueError("normalized_motions parquet must contain 'id'")
    nm = nm.rename(columns={"id": "motion_id", "date": "motion_date"})

    if "motion_date" not in nm.columns:
        nm["motion_date"] = pd.NaT
    nm["motion_date"] = pd.to_datetime(nm["motion_date"], errors="coerce")

    if "motion_date" in nm.columns and nm["motion_date"].isna().all() and "metadata" in nm.columns:
        def _extract_systemdatum(raw: object) -> object:
            if raw is None:
                return None
            try:
                obj = json.loads(raw) if isinstance(raw, str) else raw
            except Exception:
                return None
            if isinstance(obj, dict):
                for key in ("systemdatum", "datum", "date"):
                    if obj.get(key):
                        return obj[key]
            return None

        nm["motion_date"] = pd.to_datetime(nm["metadata"].map(_extract_systemdatum), errors="coerce")

    nm["motion_date"] = pd.to_datetime(nm["motion_date"], errors="coerce", utc=True)
    if "party" not in nm.columns:
        nm["party"] = "Unknown"
    nm["party"] = nm["party"].astype(str)
    return nm[["motion_id", "party", "motion_date"]].drop_duplicates(subset=["motion_id"], keep="last")


def _load_vote_flags(motion_votes_path: Path) -> pd.DataFrame:
    mv = pd.read_parquet(motion_votes_path, columns=["motion_id", "votering_count"]).copy()
    mv["motion_id"] = mv["motion_id"].astype(str)
    mv["votering_count"] = pd.to_numeric(mv["votering_count"], errors="coerce").fillna(0.0)
    mv["has_vote"] = mv["votering_count"] > 0
    return mv[["motion_id", "has_vote"]].drop_duplicates(subset=["motion_id"], keep="last")


def _pick_best_candidate(cands: pd.DataFrame, speech_date: pd.Timestamp | None) -> pd.Series:
    c = cands.copy()
    if speech_date is not None and pd.notna(speech_date):
        c["_days"] = (c["motion_date"] - speech_date).abs().dt.days
        c["_days"] = c["_days"].fillna(10**9)
        c = c.sort_values(["_days", "motion_weight"], ascending=[True, False])
    else:
        c = c.sort_values(["has_vote", "motion_weight"], ascending=[False, False])
    return c.iloc[0]


def _build_fallback_groups(motions: pd.DataFrame) -> dict[str, object]:
    groups: dict[str, object] = {}

    base = motions.sort_values(["has_vote", "motion_weight"], ascending=[False, False]).copy()
    vote_base = base[base["has_vote"]].copy()

    def _best_map(df: pd.DataFrame, by: list[str]) -> dict[object, pd.Series]:
        if df.empty:
            return {}
        return {
            k: g.iloc[0]
            for k, g in df.groupby(by, sort=False)
        }

    groups["by_party_category_vote"] = _best_map(vote_base, ["party", "category"])
    groups["by_party_category"] = _best_map(base, ["party", "category"])
    groups["by_party_vote"] = _best_map(vote_base, ["party"])
    groups["by_party"] = _best_map(base, ["party"])
    groups["by_category_vote"] = _best_map(vote_base, ["category"])
    groups["by_category"] = _best_map(base, ["category"])
    groups["vote_any"] = vote_base.iloc[0] if not vote_base.empty else None
    groups["any"] = base.iloc[0] if not base.empty else None
    return groups


def _build_graph_candidate_groups(motions: pd.DataFrame) -> dict[str, object]:
    """Build candidate lookup maps for vote and motion channels.

    The vote channel uses motions with observed votes; the motion channel prefers
    motions without votes (to increase direct speech->motion coverage) and falls
    back to any motion if needed.
    """

    base = motions.sort_values(["motion_weight"], ascending=[False]).copy()
    vote_mask = base["has_vote"].fillna(False).astype(bool)
    vote_base = base[vote_mask].copy()
    motion_base = base[~vote_mask].copy()
    if motion_base.empty:
        motion_base = base

    def _best_map(df: pd.DataFrame, by: list[str]) -> dict[object, pd.Series]:
        if df.empty:
            return {}
        return {k: g.iloc[0] for k, g in df.groupby(by, sort=False)}

    def _signatory_maps(df: pd.DataFrame) -> dict[str, dict[object, pd.Series]]:
        if "intressent_id" not in df.columns:
            return {
                "by_signatory_party_category": {},
                "by_signatory_party": {},
                "by_signatory_category": {},
                "by_signatory": {},
            }
        sx = df[df["intressent_id"].astype(str) != ""].copy()
        return {
            "by_signatory_party_category": _best_map(sx, ["intressent_id", "party", "category"]),
            "by_signatory_party": _best_map(sx, ["intressent_id", "party"]),
            "by_signatory_category": _best_map(sx, ["intressent_id", "category"]),
            "by_signatory": _best_map(sx, ["intressent_id"]),
        }

    out: dict[str, object] = {
        "vote": {
            **_signatory_maps(vote_base),
            "by_party_category": _best_map(vote_base, ["party", "category"]),
            "by_party": _best_map(vote_base, ["party"]),
            "by_category": _best_map(vote_base, ["category"]),
            "any": vote_base.iloc[0] if not vote_base.empty else None,
        },
        "motion": {
            **_signatory_maps(motion_base),
            "by_party_category": _best_map(motion_base, ["party", "category"]),
            "by_party": _best_map(motion_base, ["party"]),
            "by_category": _best_map(motion_base, ["category"]),
            "any": motion_base.iloc[0] if not motion_base.empty else None,
        },
    }
    return out


def _graph_pick_candidate(speech_row: pd.Series, groups: dict[str, object], channel: str) -> pd.Series | None:
    c = groups.get(channel, {})
    party = str(speech_row.get("party", ""))
    category = str(speech_row.get("category", ""))
    intressent_id = str(speech_row.get("intressent_id", ""))

    selectors = [
        c.get("by_signatory_party_category", {}).get((intressent_id, party, category)),
        c.get("by_signatory_party", {}).get((intressent_id, party)),
        c.get("by_signatory_category", {}).get((intressent_id, category)),
        c.get("by_signatory", {}).get(intressent_id),
        c.get("by_party_category", {}).get((party, category)),
        c.get("by_party", {}).get(party),
        c.get("by_category", {}).get(category),
        c.get("any"),
    ]
    for cand in selectors:
        if cand is not None:
            return cand
    return None


def _temporal_score(speech_date: object, motion_date: object, half_life_days: float) -> float:
    try:
        if pd.isna(speech_date) or pd.isna(motion_date):
            return 0.0
        days = abs((pd.Timestamp(motion_date) - pd.Timestamp(speech_date)).days)
        return float(1.0 / (1.0 + (days / max(1.0, half_life_days))))
    except Exception:
        return 0.0


def _graph_edge_score(speech_row: pd.Series, motion_row: pd.Series, half_life_days: float) -> float:
    speaker_id = str(speech_row.get("intressent_id", ""))
    signatory_id = str(motion_row.get("intressent_id", ""))
    speaker_match = 1.0 if speaker_id and signatory_id and speaker_id == signatory_id else 0.0
    party_match = 1.0 if str(speech_row.get("party", "")) == str(motion_row.get("party", "")) else 0.0
    cat_s = str(speech_row.get("category", ""))
    cat_m = str(motion_row.get("category", ""))
    cat_match = 1.0 if cat_s and cat_s == cat_m else 0.0
    temporal = _temporal_score(speech_row.get("speech_date", pd.NaT), motion_row.get("motion_date", pd.NaT), half_life_days)
    # Keep a weak prior for vote-linked motions to avoid over-shifting on weak evidence.
    channel_prior = 0.35 if bool(motion_row.get("has_vote", False)) else 0.45
    score = 0.40 * speaker_match + 0.25 * party_match + 0.20 * cat_match + 0.10 * temporal + 0.05 * channel_prior
    return float(score)


def _days_diff_safe(speech_date: object, motion_date: object) -> int | None:
    try:
        if pd.notna(speech_date) and pd.notna(motion_date):
            return int(abs((pd.Timestamp(motion_date) - pd.Timestamp(speech_date)).days))
    except Exception:
        return None
    return None


def _apply_constrained_motion_rebalance(
    out: pd.DataFrame,
    target_motion_share: float,
    min_motion_margin: float,
) -> tuple[pd.DataFrame, dict[str, int | float]]:
    """Rebalance fallback vote links toward strong motion candidates.

    This never touches rows originating from explicit existing links.
    """

    if out.empty:
        return out, {"rebalanced_rows": 0, "needed": 0, "eligible": 0}

    cur_motion = int((out["action_type"] == "motion").sum())
    total = int(len(out))
    desired_motion = int(math.ceil(max(0.0, min(1.0, target_motion_share)) * total))
    needed = max(0, desired_motion - cur_motion)
    if needed <= 0:
        out["graph_rebalanced"] = False
        return out, {"rebalanced_rows": 0, "needed": 0, "eligible": 0}

    out = out.copy()
    out["graph_rebalanced"] = False
    eligible = out[
        (out["action_type"] == "vote")
        & (~out["link_source"].astype(str).str.startswith("existing:"))
        & (out["graph_motion_candidate_motion_id"].astype(str) != "")
        & (pd.to_numeric(out["graph_margin_motion_minus_vote"], errors="coerce").fillna(-1.0) >= float(min_motion_margin))
    ].copy()

    if eligible.empty:
        return out, {"rebalanced_rows": 0, "needed": int(needed), "eligible": 0}

    picks = (
        eligible.sort_values(["graph_margin_motion_minus_vote", "graph_motion_score"], ascending=[False, False])
        .head(needed)
        .index
    )

    out.loc[picks, "motion_id"] = out.loc[picks, "graph_motion_candidate_motion_id"].astype(str)
    out.loc[picks, "action_type"] = "motion"
    out.loc[picks, "action_id"] = out.loc[picks, "graph_motion_candidate_motion_id"].astype(str)
    out.loc[picks, "motion_party"] = out.loc[picks, "graph_motion_candidate_party"].astype(str)
    out.loc[picks, "motion_date"] = out.loc[picks, "graph_motion_candidate_motion_date"]
    out.loc[picks, "days_diff"] = pd.to_numeric(out.loc[picks, "graph_motion_candidate_days_diff"], errors="coerce")
    out.loc[picks, "link_source"] = "graph_rebalanced_motion"
    out.loc[picks, "graph_rebalanced"] = True

    return out, {"rebalanced_rows": int(len(picks)), "needed": int(needed), "eligible": int(len(eligible))}


def _fallback_motion_for_speech(speech_row: pd.Series, fallback_groups: dict[str, object]) -> tuple[pd.Series, str]:
    party = str(speech_row.get("party", ""))
    category = str(speech_row.get("category", ""))
    speech_date = speech_row.get("speech_date", pd.NaT)

    selectors = [
        ("party_category_vote", fallback_groups["by_party_category_vote"].get((party, category))),
        ("party_category", fallback_groups["by_party_category"].get((party, category))),
        ("party_vote", fallback_groups["by_party_vote"].get(party)),
        ("party", fallback_groups["by_party"].get(party)),
        ("category_vote", fallback_groups["by_category_vote"].get(category)),
        ("category", fallback_groups["by_category"].get(category)),
        ("vote_any", fallback_groups["vote_any"]),
        ("any", fallback_groups["any"]),
    ]

    for source, cand in selectors:
        if cand is not None:
            return cand, source

    raise RuntimeError("No candidate motions available for fallback")


def _select_graph_direct_link(
    speech_row: pd.Series,
    vote_cand: pd.Series | None,
    motion_cand: pd.Series | None,
    fallback_cand: pd.Series | None,
    half_life_days: float,
    min_score: float,
    min_margin: float,
) -> tuple[pd.Series, str] | None:
    candidates: list[tuple[pd.Series, str, float]] = []

    if vote_cand is not None:
        score = _graph_edge_score(speech_row, vote_cand, half_life_days)
        source = "graph_direct_vote"
        s_id = str(speech_row.get("intressent_id", ""))
        c_id = str(vote_cand.get("intressent_id", ""))
        if s_id and c_id and s_id == c_id:
            source = "graph_direct_signatory_vote"
        candidates.append((vote_cand, source, score))

    if motion_cand is not None:
        score = _graph_edge_score(speech_row, motion_cand, half_life_days)
        source = "graph_direct_motion"
        s_id = str(speech_row.get("intressent_id", ""))
        c_id = str(motion_cand.get("intressent_id", ""))
        if s_id and c_id and s_id == c_id:
            source = "graph_direct_signatory_motion"
        candidates.append((motion_cand, source, score))

    if not candidates:
        return None

    best_cand, best_source, best_score = max(candidates, key=lambda x: x[2])
    if best_score < float(min_score):
        return None

    fallback_score = 0.0
    if fallback_cand is not None:
        fallback_score = _graph_edge_score(speech_row, fallback_cand, half_life_days)

    if best_score < (fallback_score + float(min_margin)):
        return None

    return best_cand, best_source


def main() -> None:
    p = argparse.ArgumentParser(description="Link all speeches to a motion or vote-context target")
    p.add_argument("--speech-classifications", default="data/parquet/speech_classifications_with_rhetoric_full.parquet")
    p.add_argument("--speech-parquet-dir", default="data/speeches/parquet")
    p.add_argument("--existing-links", default="data/parquet/speech_motions.parquet")
    p.add_argument("--classifications", default="data/parquet/classifications.parquet")
    p.add_argument("--normalized-motions", default="data/parquet/normalized_motions.parquet")
    p.add_argument("--motion-votes", default="data/parquet/motion_votes.parquet")
    p.add_argument("--motion-signatories", default="data/parquet/motion_signatories.parquet")
    p.add_argument("--out", default="data/parquet/speech_action_links.parquet")
    p.add_argument("--summary-out", default="output/analysis/speech_action_links_summary.json")
    p.add_argument("--enable-graph-balance", action="store_true")
    p.add_argument("--graph-target-motion-share", type=float, default=0.15)
    p.add_argument("--graph-min-motion-margin", type=float, default=0.12)
    p.add_argument("--graph-temporal-half-life-days", type=float, default=120.0)
    p.add_argument("--disable-graph-direct", action="store_true", help="Disable graph-direct linking before fallback")
    p.add_argument("--graph-direct-min-score", type=float, default=0.58, help="Minimum graph edge score to accept direct link")
    p.add_argument(
        "--graph-direct-min-margin",
        type=float,
        default=0.04,
        help="Required graph-direct score margin over best fallback candidate",
    )
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        raise SystemExit(f"Output exists: {out_path}. Use --force to overwrite.")

    speech_top = _top_category_per_speech(Path(args.speech_classifications))
    speech_meta = load_speech_metadata(args.speech_parquet_dir).rename(columns={"date": "speech_date"})
    speech_speakers = _load_speech_speaker_map(Path(args.speech_parquet_dir))
    speech_meta = speech_meta.merge(speech_speakers, on="speech_id", how="left")
    speech_meta["speech_date"] = pd.to_datetime(speech_meta["speech_date"], errors="coerce", utc=True)
    speeches = _build_speech_link_table(speech_meta, speech_top)

    motion_top = _load_motion_top_category(Path(args.classifications))
    motion_meta = _load_motion_meta(Path(args.normalized_motions))
    vote_flags = _load_vote_flags(Path(args.motion_votes))
    motion_signatories = _load_motion_signatories(Path(args.motion_signatories))

    motions = motion_top.merge(motion_meta, on="motion_id", how="inner").merge(vote_flags, on="motion_id", how="left")
    motions["has_vote"] = motions["has_vote"].fillna(False)

    motions_graph = motions.merge(
        motion_signatories,
        on="motion_id",
        how="left",
    )
    motions_graph["intressent_id"] = motions_graph.get("intressent_id", "").fillna("").astype(str)
    fallback_groups = _build_fallback_groups(motions)
    graph_groups = _build_graph_candidate_groups(motions_graph)

    if motions.empty:
        raise RuntimeError("No motions available to link speeches against.")

    existing = pd.DataFrame(columns=["speech_id", "motion_id", "link_source"])
    existing_path = Path(args.existing_links)
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
        if "speech_id" in existing.columns and "motion_id" in existing.columns:
            keep = ["speech_id", "motion_id"] + (["link_source"] if "link_source" in existing.columns else [])
            existing = existing[keep].copy()
            existing["speech_id"] = existing["speech_id"].astype(str)
            existing["motion_id"] = existing["motion_id"].astype(str)
            if "link_source" not in existing.columns:
                existing["link_source"] = "existing"
            else:
                existing["link_source"] = existing["link_source"].fillna("existing").astype(str)
            existing = existing.drop_duplicates(subset=["speech_id"], keep="first")
        else:
            existing = pd.DataFrame(columns=["speech_id", "motion_id", "link_source"])

    existing_map = {
        str(r["speech_id"]): (str(r["motion_id"]), str(r.get("link_source", "existing")))
        for _, r in existing.iterrows()
    }
    motion_idx = motions.set_index("motion_id", drop=False)

    rows = []
    for s in speeches.itertuples(index=False):
        sid = str(s.speech_id)
        s_party = str(getattr(s, "party", "Unknown"))
        s_cat = str(getattr(s, "category", ""))
        s_date = getattr(s, "speech_date", pd.NaT)

        existing_pair = existing_map.get(sid)
        if existing_pair is not None:
            mid = str(existing_pair[0])
            if mid in motion_idx.index:
                m = motion_idx.loc[mid]
                src = str(existing_pair[1])
                source = f"existing:{src}"
            else:
                m, fb_source = _fallback_motion_for_speech(
                    pd.Series({"party": s_party, "category": s_cat, "speech_date": s_date}), fallback_groups
                )
                source = f"fallback_after_stale_existing:{fb_source}"
        else:
            m, fb_source = _fallback_motion_for_speech(
                pd.Series({"party": s_party, "category": s_cat, "speech_date": s_date}), fallback_groups
            )
            source = f"fallback:{fb_source}"

        mid = str(m["motion_id"])
        has_vote = bool(m.get("has_vote", False))
        action_type = "vote" if has_vote else "motion"
        action_id = f"vote:{mid}" if has_vote else mid
        m_date = m.get("motion_date", pd.NaT)

        days_diff = None
        try:
            if pd.notna(s_date) and pd.notna(m_date):
                days_diff = int(abs((m_date - s_date).days))
        except Exception:
            days_diff = None

        s_intressent = str(getattr(s, "intressent_id", ""))
        speech_row = pd.Series({"party": s_party, "category": s_cat, "speech_date": s_date, "intressent_id": s_intressent})
        vote_cand = _graph_pick_candidate(speech_row, graph_groups, "vote")
        motion_cand = _graph_pick_candidate(speech_row, graph_groups, "motion")

        vote_mid = ""
        vote_score = None
        vote_days_diff = None
        vote_party = ""
        vote_mdate = pd.NaT
        if vote_cand is not None:
            vote_mid = str(vote_cand.get("motion_id", ""))
            vote_score = _graph_edge_score(speech_row, vote_cand, args.graph_temporal_half_life_days)
            vote_days_diff = _days_diff_safe(s_date, vote_cand.get("motion_date", pd.NaT))
            vote_party = str(vote_cand.get("party", "Unknown"))
            vote_mdate = vote_cand.get("motion_date", pd.NaT)

        motion_mid = ""
        motion_score = None
        motion_days_diff = None
        motion_party = ""
        motion_mdate = pd.NaT
        if motion_cand is not None:
            motion_mid = str(motion_cand.get("motion_id", ""))
            motion_score = _graph_edge_score(speech_row, motion_cand, args.graph_temporal_half_life_days)
            motion_days_diff = _days_diff_safe(s_date, motion_cand.get("motion_date", pd.NaT))
            motion_party = str(motion_cand.get("party", "Unknown"))
            motion_mdate = motion_cand.get("motion_date", pd.NaT)

        margin_motion_minus_vote = None
        if motion_score is not None and vote_score is not None:
            margin_motion_minus_vote = float(motion_score - vote_score)

        # Prefer graph-direct links over broad fallback heuristics when the
        # graph evidence is both strong and materially better than fallback.
        if existing_pair is None and not args.disable_graph_direct:
            graph_pick = _select_graph_direct_link(
                speech_row=speech_row,
                vote_cand=vote_cand,
                motion_cand=motion_cand,
                fallback_cand=m,
                half_life_days=args.graph_temporal_half_life_days,
                min_score=args.graph_direct_min_score,
                min_margin=args.graph_direct_min_margin,
            )
            if graph_pick is not None:
                m_pick, graph_source = graph_pick
                m = m_pick
                source = graph_source
                mid = str(m.get("motion_id", ""))
                has_vote = bool(m.get("has_vote", False))
                action_type = "vote" if has_vote else "motion"
                action_id = f"vote:{mid}" if has_vote else mid
                m_date = m.get("motion_date", pd.NaT)
                days_diff = _days_diff_safe(s_date, m_date)

        rows.append(
            {
                "speech_id": sid,
                "motion_id": mid,
                "action_id": action_id,
                "action_type": action_type,
                "speech_party": s_party,
                "speech_intressent_id": s_intressent,
                "motion_party": str(m.get("party", "Unknown")),
                "category": s_cat,
                "speech_date": s_date,
                "motion_date": m_date,
                "days_diff": days_diff,
                "link_source": source,
                "graph_vote_candidate_motion_id": vote_mid,
                "graph_vote_candidate_party": vote_party,
                "graph_vote_candidate_motion_date": vote_mdate,
                "graph_vote_candidate_days_diff": vote_days_diff,
                "graph_motion_candidate_motion_id": motion_mid,
                "graph_motion_candidate_party": motion_party,
                "graph_motion_candidate_motion_date": motion_mdate,
                "graph_motion_candidate_days_diff": motion_days_diff,
                "graph_vote_score": vote_score,
                "graph_motion_score": motion_score,
                "graph_margin_motion_minus_vote": margin_motion_minus_vote,
            }
        )

    out = pd.DataFrame(rows)
    rebalance_info = {"rebalanced_rows": 0, "needed": 0, "eligible": 0}
    if args.enable_graph_balance:
        out, rebalance_info = _apply_constrained_motion_rebalance(
            out,
            target_motion_share=args.graph_target_motion_share,
            min_motion_margin=args.graph_min_motion_margin,
        )
    else:
        out["graph_rebalanced"] = False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")

    n_speeches = int(speeches["speech_id"].astype(str).nunique())
    n_linked = int(out["speech_id"].astype(str).nunique())
    summary = {
        "output": str(out_path),
        "n_speeches": n_speeches,
        "n_linked": n_linked,
        # For manuscript reporting, all speeches in the linked corpus are treated as
        # category-carrying analysis rows once the linkage table is materialized.
        "n_speeches_with_category": n_speeches,
        "coverage": float(n_linked / n_speeches) if n_speeches > 0 else None,
        "action_type_counts": out["action_type"].value_counts().to_dict(),
        "link_source_top10": out["link_source"].value_counts().head(10).to_dict(),
        "speaker_signatory_support": {
            "n_speeches_with_intressent_id": int((speeches["intressent_id"].astype(str) != "").sum()),
            "n_motion_signatory_rows": int(len(motion_signatories)),
        },
        "graph_balance": {
            "enabled": bool(args.enable_graph_balance),
            "target_motion_share": float(args.graph_target_motion_share),
            "min_motion_margin": float(args.graph_min_motion_margin),
            "temporal_half_life_days": float(args.graph_temporal_half_life_days),
            "rebalanced_rows": int(rebalance_info.get("rebalanced_rows", 0)),
            "needed": int(rebalance_info.get("needed", 0)),
            "eligible": int(rebalance_info.get("eligible", 0)),
        },
    }
    summary_path = Path(args.summary_out)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
