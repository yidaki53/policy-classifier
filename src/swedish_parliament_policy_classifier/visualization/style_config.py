"""Academic figure styling configuration.

Provides a consistent scientific/political-science visual identity for all
manuscript figures.  Every exported figure carries an author credit, sample
size, data range, source citation and generation date.
"""

import matplotlib.pyplot as plt
from datetime import datetime, timezone
from pathlib import Path
import json

# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
AUTHOR_NAME = "Robin Öberg"
DATA_SOURCE = "Riksdagen open data (data.riksdagen.se)"

# ---------------------------------------------------------------------------
# Typography & palette
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Computer Modern"],
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.grid": False,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
})

CATEGORY_ORDER = [
    "far_left",
    "left",
    "centre_left",
    "centre",
    "centre_right",
    "right",
    "far_right",
]

CATEGORY_LABELS = {
    "far_left": "Far Left",
    "left": "Left",
    "centre_left": "Centre-Left",
    "centre": "Centre",
    "centre_right": "Centre-Right",
    "right": "Right",
    "far_right": "Far Right",
}

CATEGORY_COLORS = {
    "far_left":      "#8B0000",
    "left":          "#CC3333",
    "centre_left":   "#FF7F7F",
    "centre":        "#BBBBBB",
    "centre_right":  "#7FB3D5",
    "right":         "#3366AA",
    "far_right":     "#00008B",
}


def compute_ideology_score_from_proportions(proportions: dict[str, float | int]) -> float:
    """Compute a net left-right score in [-1, 1] with the centre category as neutral."""
    left_mass = sum(float(proportions.get(cat, 0.0)) for cat in ["far_left", "left", "centre_left"])
    right_mass = sum(float(proportions.get(cat, 0.0)) for cat in ["centre_right", "right", "far_right"])
    total = left_mass + right_mass
    if total <= 0:
        return 0.0
    return float((right_mass - left_mass) / total)

# ---------------------------------------------------------------------------
# Party labels and colors (fetched from Riksdagen API or parquet data)
# ---------------------------------------------------------------------------
BAD_PARTY_VALUES = {"", "-", "NYD", "Unknown", "None", "nan", "null", "N/A"}


def _is_substantive_party(value: object) -> bool:
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    if text.lower() in {item.lower() for item in BAD_PARTY_VALUES}:
        return False
    return True


def _infer_current_parties() -> set[str]:
    """Infer parties currently represented in the Riksdag.

    Source priority:
      1. Existing cache (data/.parliamentary_party_cache.json)
      2. Riksdag API personlista (data.riksdagen.se)
      3. Party metadata from definitions/political_spectrum.yaml
      4. Conservative fallback (last resort)
    """
    repo_root = Path(__file__).resolve().parents[3]

    # 1) Try cache first
    cache_path = repo_root / "data" / ".parliamentary_party_cache.json"
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            parties = cache.get("parties", [])
            if parties:
                return {str(p).strip() for p in parties if _is_substantive_party(p)}
        except Exception:
            pass

    # 2) Try Riksdag API
    try:
        import requests
        resp = requests.get(
            "https://data.riksdagen.se/personlista/?utformat=json",
            timeout=10,
        )
        resp.raise_for_status()
        payload = resp.json()
        persons = (((payload.get("personlista") or {}).get("person")) or [])
        api_parties = {str(p.get("parti", "")).strip() for p in persons if p.get("parti")}
        if api_parties:
            return {p for p in api_parties if _is_substantive_party(p)}
    except Exception:
        pass

    # 3) Try YAML definitions
    defs_path = repo_root / "src" / "swedish_parliament_policy_classifier" / "definitions" / "political_spectrum.yaml"
    if defs_path.exists():
        try:
            import yaml
            with open(defs_path) as f:
                content = yaml.safe_load(f)
            if content:
                return {str(k).strip() for k in content.keys() if _is_substantive_party(k)}
        except Exception:
            pass

    # 4) Conservative fallback
    return {"S", "M", "SD", "V", "C", "KD", "L", "MP"}


CURRENT_PARTIES = _infer_current_parties()

PARTY_LABELS = {
    "S":  "Socialdemokraterna",
    "M":  "Moderaterna",
    "SD": "Sverigedemokraterna",
    "V":  "Vänsterpartiet",
    "C":  "Centerpartiet",
    "KD": "Kristdemokraterna",
    "L":  "Liberaler",
    "MP": "Miljöpartiet",
    "-":  "Ej specificerad",
}

PARTY_COLORS_PLOT = {
    "S":  "#CC3333",   # red
    "M":  "#3366AA",   # blue
    "SD": "#00008B",   # dark blue
    "V":  "#8B0000",   # dark red
    "C":  "#7FB3D5",   # light blue
    "KD": "#FF7F7F",   # light red
    "L":  "#33CCFF",   # cyan
    "MP": "#33CC33",   # green
    "-":  "#888888",  # gray
}


def add_figure_credits(
    fig,
    *,
    n_total: int | None = None,
    n_parties: int | None = None,
    date_range: str | None = None,
    extra_text: str | None = None,
    recency_weighted: bool = False,
    author: str = AUTHOR_NAME,
    source: str = DATA_SOURCE,
) -> None:
    """Add an academic footer with author, n, date range, source and generation date.

    The footer is placed at the bottom-right of the figure canvas, outside the
    plotting area, using a small sans-serif font so it is legible but unobtrusive.
    """
    parts: list[str] = []
    current_year = datetime.now(timezone.utc).year
    if author:
        parts.append(f"Author: {author} ({current_year})")
    if n_total is not None:
        parts.append(f"n = {n_total:,}")
    if n_parties is not None:
        parts.append(f"parties = {n_parties}")
    if date_range:
        parts.append(f"Period: {date_range}")
    if source:
        parts.append(f"Source: {source}")
    if extra_text:
        parts.append(extra_text)

    gen_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    parts.append(f"Generated: {gen_date}")

    if recency_weighted:
        parts.append("Recency-weighted (λ=0.3 yr⁻¹, half-life ≈2.3 yr)")

    footer_text = "  |  ".join(parts)

    fig.text(
        0.99, 0.005, footer_text,
        ha="right", va="bottom",
        fontsize=7,
        fontfamily="sans-serif",
        color="#555555",
        transform=fig.transFigure,
    )


def set_publication_defaults() -> None:
    """Call once at module import to apply the rcParams above."""
    # Already applied at import time via the dict update above.
    pass


def query_summary_stats(conn) -> dict:
    """Return global summary stats from the database for figure captions."""
    import sqlite3
    cur = conn.cursor()
    cur.execute("""
        SELECT MIN(nm.date), MAX(nm.date),
               COUNT(DISTINCT nm.party),
               COUNT(DISTINCT c.motion_id)
        FROM classifications c
        JOIN normalized_motions nm ON c.motion_id = nm.id
        WHERE nm.party IS NOT NULL AND nm.party != '' AND nm.party != 'NYD'
    """)
    row = cur.fetchone()
    min_date = (row[0] or "")[:4] if row[0] else "?"
    max_date = (row[1] or "")[:4] if row[1] else "?"
    return {
        "date_range": f"{min_date}-{max_date}",
        "n_parties": row[2] or 0,
        "n_motions": row[3] or 0,
    }