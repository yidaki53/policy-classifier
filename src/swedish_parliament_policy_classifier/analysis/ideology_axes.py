from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from swedish_parliament_policy_classifier.definitions.loader import load_verified_definitions


def canonical_axis_order() -> list[str]:
    defs = load_verified_definitions()
    return list(defs.keys())


def _build_axis_matrix(long_df: pd.DataFrame, id_col: str, axis_order: list[str]) -> pd.DataFrame:
    w = long_df[[id_col, "category", "normalized_weight"]].copy()
    w[id_col] = w[id_col].astype(str)
    w["category"] = w["category"].astype(str)
    w["normalized_weight"] = pd.to_numeric(w["normalized_weight"], errors="coerce").fillna(0.0)
    w["normalized_weight"] = w["normalized_weight"].clip(lower=0.0)

    pivot = (
        w.pivot_table(index=id_col, columns="category", values="normalized_weight", aggfunc="mean", fill_value=0.0)
        .reindex(columns=axis_order, fill_value=0.0)
        .reset_index()
    )

    vec = pivot[axis_order].to_numpy(dtype=float)
    vec = np.clip(vec, 0.0, None)
    den = vec.sum(axis=1, keepdims=True)
    den[den <= 0] = 1.0
    norm = vec / den
    pivot[axis_order] = norm
    return pivot


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def compute_axis_alignment(
    speech_classifications_path: str | Path,
    motion_classifications_path: str | Path,
    speech_action_links_path: str | Path,
    out_path: str | Path,
) -> dict:
    axis = canonical_axis_order()

    speech = pd.read_parquet(speech_classifications_path, columns=["speech_id", "category", "normalized_weight"])
    motion = pd.read_parquet(motion_classifications_path, columns=["motion_id", "category", "normalized_weight"])
    links = pd.read_parquet(speech_action_links_path, columns=["speech_id", "motion_id", "action_type", "category"])

    speech_axis = _build_axis_matrix(speech, "speech_id", axis)
    motion_axis = _build_axis_matrix(motion, "motion_id", axis)

    merged = (
        links.merge(speech_axis, on="speech_id", how="left", suffixes=("", "_speech"))
        .merge(motion_axis, on="motion_id", how="left", suffixes=("_speech", "_motion"))
    )

    s_cols = [f"{c}_speech" for c in axis]
    m_cols = [f"{c}_motion" for c in axis]
    for c in s_cols + m_cols:
        if c not in merged.columns:
            merged[c] = 0.0

    rows = []
    for r in merged.itertuples(index=False):
        svec = np.array([float(getattr(r, c)) for c in s_cols], dtype=float)
        mvec = np.array([float(getattr(r, c)) for c in m_cols], dtype=float)

        s_sum = float(svec.sum())
        m_sum = float(mvec.sum())
        if s_sum > 0:
            svec = svec / s_sum
        else:
            svec = np.full(len(axis), 1.0 / len(axis), dtype=float)
        if m_sum > 0:
            mvec = mvec / m_sum
        else:
            mvec = np.full(len(axis), 1.0 / len(axis), dtype=float)

        s_idx = int(np.argmax(svec)) if svec.size else 0
        m_idx = int(np.argmax(mvec)) if mvec.size else 0
        rows.append(
            {
                "speech_id": str(r.speech_id),
                "motion_id": str(r.motion_id),
                "action_type": str(r.action_type),
                "topic": str(r.category),
                "axis_order": json.dumps(axis),
                "speech_axis": json.dumps(svec.tolist()),
                "action_axis": json.dumps(mvec.tolist()),
                "axis_js_distance": float(jensenshannon(svec, mvec, base=2.0)),
                "axis_cosine_distance": _cosine_dist(svec, mvec),
                "axis_peak_speech": axis[s_idx],
                "axis_peak_action": axis[m_idx],
                "axis_peak_flip": bool(s_idx != m_idx),
            }
        )

    out = pd.DataFrame(rows)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False, compression="zstd")

    return {
        "output": str(out_path),
        "rows": int(len(out)),
        "mean_axis_js_distance": float(out["axis_js_distance"].mean()) if len(out) else None,
        "peak_flip_rate": float(out["axis_peak_flip"].mean()) if len(out) else None,
    }
