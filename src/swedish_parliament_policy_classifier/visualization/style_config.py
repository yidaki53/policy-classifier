"""Academic figure styling configuration.

Provides a consistent scientific/political-science visual identity for all
manuscript figures.  Every exported figure carries an author credit, sample
size, data range, source citation and generation date.
"""

import matplotlib.pyplot as plt
from datetime import datetime, timezone

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
