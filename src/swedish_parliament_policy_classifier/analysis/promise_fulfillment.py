from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from swedish_parliament_policy_classifier.analysis.speech_visualizations import IDEOLOGY_ORDER, load_speech_metadata

ELECTION_YEARS = {2010, 2014, 2018, 2022, 2026}


def _top_category_from_long(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    work = df[[id_col, "category", "normalized_weight"]].copy()
    work["normalized_weight"] = pd.to_numeric(work["normalized_weight"], errors="coerce").fillna(0.0)
    idx = work.groupby(id_col)["normalized_weight"].idxmax()
    out = work.loc[idx, [id_col, "category"]].rename(columns={"category": "topic"})
    return out


def _read_parquet_table(parquet_dir: Path, stem: str, columns: list[str]) -> pd.DataFrame:
    candidates = [
        parquet_dir / f"{stem}.parquet",
        parquet_dir / f"{stem}.parquet.zst",
        parquet_dir / f"{stem}.pq",
    ]
    for p in candidates:
        if p.exists():
            return pd.read_parquet(p, columns=columns)
    raise FileNotFoundError(f"Missing parquet table for '{stem}' under {parquet_dir}")


def _load_motions_votes(parquet_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(parquet_dir)
    cls = _read_parquet_table(root, "classifications", ["motion_id", "category", "normalized_weight"]).copy()
    nm = _read_parquet_table(root, "normalized_motions", ["id", "party", "doc_type", "metadata"]).copy()

    cls["motion_id"] = cls["motion_id"].astype(str)
    cls["normalized_weight"] = pd.to_numeric(cls["normalized_weight"], errors="coerce").fillna(0.0)
    nm["id"] = nm["id"].astype(str)
    nm["date"] = pd.NaT
    if "metadata" in nm.columns:
        md = nm["metadata"].fillna("").astype(str)

        def _extract_systemdatum(s: str) -> str | None:
            if not s:
                return None
            try:
                obj = json.loads(s)
            except Exception:
                return None
            val = obj.get("systemdatum")
            return str(val) if val else None

        nm["date"] = pd.to_datetime(md.map(_extract_systemdatum), errors="coerce")

    df = cls.merge(nm, left_on="motion_id", right_on="id", how="inner")
    df = df[df["party"].notna()].copy()

    top = _top_category_from_long(df.rename(columns={"motion_id": "item_id"}), "item_id")
    meta = (
        df[["motion_id", "party", "date", "doc_type"]]
        .drop_duplicates(subset=["motion_id"], keep="last")
        .rename(columns={"motion_id": "item_id"})
    )
    out = meta.merge(top, on="item_id", how="left")
    motions = out[out["doc_type"].isin(["mot", "prop"])].copy()

    mv = _read_parquet_table(root, "motion_votes", ["motion_id", "votering_count"]).copy()
    mv["motion_id"] = mv["motion_id"].astype(str)
    mv["votering_count"] = pd.to_numeric(mv["votering_count"], errors="coerce").fillna(0.0)
    mv = mv[mv["votering_count"] > 0].copy()

    votes = motions.merge(mv, left_on="item_id", right_on="motion_id", how="inner")
    votes["vote_weight"] = votes["votering_count"]
    return motions, votes


def _load_speeches(speech_classifications_path: str | Path, speech_parquet_dir: str | Path) -> pd.DataFrame:
    cls = pd.read_parquet(speech_classifications_path)[["speech_id", "category", "normalized_weight"]].copy()
    top = _top_category_from_long(cls, "speech_id")
    meta = load_speech_metadata(speech_parquet_dir)
    out = meta.merge(top, on="speech_id", how="left")
    return out


def _year_bucket(d: pd.Series) -> pd.Series:
    return pd.to_datetime(d, errors="coerce").dt.year


def _ideology_index(topic: pd.Series) -> pd.Series:
    idx = {c: i for i, c in enumerate(IDEOLOGY_ORDER)}
    return topic.map(idx).astype(float)


def run_promise_fulfillment_analysis(
    db_path: str | Path,
    speech_classifications_path: str | Path,
    speech_parquet_dir: str | Path,
    out_dir: str | Path = "output/analysis",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    speeches = _load_speeches(speech_classifications_path, speech_parquet_dir)
    parquet_dir = Path(speech_classifications_path).resolve().parent
    motions, votes = _load_motions_votes(parquet_dir)

    speeches["year"] = _year_bucket(speeches["date"])
    motions["year"] = _year_bucket(motions["date"])
    votes["year"] = _year_bucket(votes["date"])

    # Count by party/topic/year
    s_cnt = speeches.groupby(["party", "topic", "year"], dropna=False).size().rename("S").reset_index()
    m_cnt = motions.groupby(["party", "topic", "year"], dropna=False).size().rename("M").reset_index()
    v_cnt = votes.groupby(["party", "topic", "year"], dropna=False)["vote_weight"].sum().rename("V").reset_index()

    frame = s_cnt.merge(m_cnt, on=["party", "topic", "year"], how="left").merge(
        v_cnt, on=["party", "topic", "year"], how="left"
    )
    frame[["M", "V"]] = frame[["M", "V"]].fillna(0.0)

    # Promise pathways per party/topic/year
    matched_sm = np.minimum(frame["S"], frame["M"])
    matched_smv = np.minimum(matched_sm, frame["V"])
    speech_motion_no_vote = matched_sm - matched_smv
    speech_no_motion = frame["S"] - matched_sm

    denom = frame["S"].replace(0, np.nan)
    frame["pct_speech_motion_vote"] = matched_smv / denom
    frame["pct_speech_no_motion"] = speech_no_motion / denom
    frame["pct_speech_motion_no_vote"] = speech_motion_no_vote / denom
    frame[["pct_speech_motion_vote", "pct_speech_no_motion", "pct_speech_motion_no_vote"]] = frame[
        ["pct_speech_motion_vote", "pct_speech_no_motion", "pct_speech_motion_no_vote"]
    ].fillna(0.0)

    frame["election_year"] = frame["year"].isin(ELECTION_YEARS)

    expected_path = out_dir / "speech_action_expected_contradiction_party_topic_year.parquet"
    if expected_path.exists():
        exp = pd.read_parquet(expected_path)
        exp = exp[["party", "topic", "year", "expected_contradiction", "expected_uphold"]].copy()
        exp["year"] = pd.to_numeric(exp["year"], errors="coerce")
        frame = frame.merge(exp, on=["party", "topic", "year"], how="left")
        frame["expected_contradiction"] = pd.to_numeric(frame["expected_contradiction"], errors="coerce").fillna(0.0)
        frame["expected_uphold"] = pd.to_numeric(frame["expected_uphold"], errors="coerce").fillna(1.0)
    else:
        frame["expected_contradiction"] = 0.0
        frame["expected_uphold"] = 1.0

    # Time-aware controls: simple ideology drift proxy by party-year
    drift_src = pd.concat(
        [
            speeches.assign(modality="speech", item_topic=speeches["topic"]),
            motions.assign(modality="motion", item_topic=motions["topic"]),
            votes.assign(modality="vote", item_topic=votes["topic"]),
        ],
        ignore_index=True,
    )
    drift_src["year"] = _year_bucket(drift_src["date"])
    drift_src["ideology_idx"] = _ideology_index(drift_src["item_topic"])

    drift = (
        drift_src.groupby(["party", "modality", "year"], as_index=False)["ideology_idx"].mean()
        .rename(columns={"ideology_idx": "mean_ideology_idx"})
    )

    # Aggregate to party-level percentages
    party_summary = (
        frame.groupby("party", as_index=False)[
            [
                "pct_speech_motion_vote",
                "pct_speech_no_motion",
                "pct_speech_motion_no_vote",
                "expected_contradiction",
                "expected_uphold",
            ]
        ]
        .mean()
    )

    frame_parquet = out_dir / "promise_fulfillment_party_topic_year.parquet"
    party_parquet = out_dir / "promise_fulfillment_party_summary.parquet"
    drift_parquet = out_dir / "party_ideology_drift_by_modality_year.parquet"
    frame.to_parquet(frame_parquet, index=False, compression="zstd")
    party_summary.to_parquet(party_parquet, index=False, compression="zstd")
    drift.to_parquet(drift_parquet, index=False, compression="zstd")

    payload = {
        "outputs": {
            "party_topic_year": str(frame_parquet),
            "party_summary": str(party_parquet),
            "ideology_drift": str(drift_parquet),
        },
        "party_count": int(frame["party"].nunique()),
        "rows_party_topic_year": int(len(frame)),
    }
    (out_dir / "promise_fulfillment_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
