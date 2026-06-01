from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine, jensenshannon

from swedish_parliament_policy_classifier.analysis.speech_visualizations import IDEOLOGY_ORDER, load_speech_metadata


def _norm(v: np.ndarray) -> np.ndarray:
    s = float(v.sum())
    if s <= 0:
        return np.ones_like(v) / len(v)
    return v / s


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


def _party_modality_profiles_from_parquet(parquet_dir: str | Path) -> pd.DataFrame:
    root = Path(parquet_dir)
    cls = _read_parquet_table(root, "classifications", ["motion_id", "category", "normalized_weight"]).copy()
    nm = _read_parquet_table(root, "normalized_motions", ["id", "party", "doc_type"]).copy()

    cls["motion_id"] = cls["motion_id"].astype(str)
    cls["normalized_weight"] = pd.to_numeric(cls["normalized_weight"], errors="coerce").fillna(0.0)
    nm["id"] = nm["id"].astype(str)

    joined = cls.merge(nm, left_on="motion_id", right_on="id", how="inner")
    modality_map = {"mot": "motion", "prop": "motion", "votering": "vote"}
    joined["modality"] = joined["doc_type"].map(modality_map)

    motions = joined[joined["modality"].eq("motion") & joined["party"].notna()].copy()
    motion_grp = (
        motions.groupby(["party", "category"], as_index=False)["normalized_weight"]
        .mean()
        .rename(columns={"normalized_weight": "weight"})
    )
    motion_grp["modality"] = "motion"

    mv = _read_parquet_table(root, "motion_votes", ["motion_id", "votering_count"]).copy()
    mv["motion_id"] = mv["motion_id"].astype(str)
    mv["votering_count"] = pd.to_numeric(mv["votering_count"], errors="coerce").fillna(0.0)
    mv = mv[mv["votering_count"] > 0]

    vote_src = motions.merge(mv, on="motion_id", how="inner")
    vote_src["weight"] = vote_src["normalized_weight"] * vote_src["votering_count"]
    vote_grp = vote_src.groupby(["party", "category"], as_index=False)["weight"].sum()
    vote_grp["modality"] = "vote"

    out = pd.concat([motion_grp, vote_grp], ignore_index=True)
    return out[["party", "modality", "category", "weight"]]


def _speech_profiles(speech_classifications_path: str | Path, speech_parquet_dir: str | Path) -> pd.DataFrame:
    cls = pd.read_parquet(speech_classifications_path)[["speech_id", "category", "normalized_weight"]].copy()
    cls["speech_id"] = cls["speech_id"].astype(str)
    meta = load_speech_metadata(speech_parquet_dir)[["speech_id", "party"]].copy()
    df = cls.merge(meta, on="speech_id", how="left")
    df["party"] = df["party"].fillna("Unknown")
    out = df.groupby(["party", "category"], as_index=False)["normalized_weight"].mean()
    out["modality"] = "speech"
    return out.rename(columns={"normalized_weight": "weight"})[["party", "modality", "category", "weight"]]


def build_modality_profiles(
    db_path: str | Path,
    speech_classifications_path: str | Path,
    speech_parquet_dir: str | Path,
) -> pd.DataFrame:
    parquet_dir = Path(speech_classifications_path).resolve().parent
    a = _party_modality_profiles_from_parquet(parquet_dir)
    b = _speech_profiles(speech_classifications_path, speech_parquet_dir)
    allp = pd.concat([a, b], ignore_index=True)

    totals = allp.groupby(["party", "modality"], as_index=False)["weight"].sum().rename(columns={"weight": "sum_w"})
    out = allp.merge(totals, on=["party", "modality"], how="left")
    out["proportion"] = np.where(out["sum_w"] > 0, out["weight"] / out["sum_w"], 0.0)
    return out[["party", "modality", "category", "proportion"]]


def _vector(df: pd.DataFrame, party: str, modality: str) -> np.ndarray:
    s = (
        df[(df["party"] == party) & (df["modality"] == modality)]
        .set_index("category")["proportion"]
        .reindex(IDEOLOGY_ORDER, fill_value=0.0)
        .to_numpy(dtype=float)
    )
    return _norm(s)


def _sign_flip_pvalue(x: np.ndarray, n_perm: int = 10000, seed: int = 42) -> float:
    rng = np.random.default_rng(seed)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    obs = abs(x.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(x)))
    sims = np.abs((signs * x).mean(axis=1))
    return float((np.sum(sims >= obs) + 1) / (n_perm + 1))


def run_ideological_gap_analysis(
    db_path: str | Path,
    speech_classifications_path: str | Path,
    speech_parquet_dir: str | Path,
    out_dir: str | Path = "output/analysis",
) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prof = build_modality_profiles(db_path, speech_classifications_path, speech_parquet_dir)
    parties = sorted(prof["party"].unique().tolist())

    rows = []
    for p in parties:
        vs = _vector(prof, p, "speech")
        vm = _vector(prof, p, "motion")
        vv = _vector(prof, p, "vote")

        pairs = [
            ("speech_vs_motion", vs, vm),
            ("speech_vs_vote", vs, vv),
            ("motion_vs_vote", vm, vv),
        ]
        for label, a, b in pairs:
            rows.append(
                {
                    "party": p,
                    "comparison": label,
                    "js_distance": float(jensenshannon(a, b, base=2.0)),
                    "cosine_distance": float(cosine(a, b)) if np.any(a) and np.any(b) else 0.0,
                }
            )

    gap_df = pd.DataFrame(rows)
    gap_parquet = out_dir / "ideological_gap_party.parquet"
    gap_df.to_parquet(gap_parquet, index=False, compression="zstd")

    global_tests = {}
    for comp in sorted(gap_df["comparison"].unique()):
        arr = gap_df.loc[gap_df["comparison"] == comp, "js_distance"].to_numpy(dtype=float)
        global_tests[comp] = {
            "mean_js_distance": float(np.nanmean(arr)) if len(arr) else float("nan"),
            "p_value_signflip": _sign_flip_pvalue(arr),
            "n_parties": int(len(arr)),
        }

    payload = {
        "global_tests": global_tests,
        "party_count": len(parties),
        "output_party_parquet": str(gap_parquet),
    }
    (out_dir / "ideological_gap_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
