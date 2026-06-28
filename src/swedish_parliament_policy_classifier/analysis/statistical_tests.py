"""Paired statistical tests and FDR correction utilities.

Provides:
- choose_test(a, b): decide between paired t-test and Wilcoxon
- run_paired_test(a, b): run selected test and return stats
- apply_bh_correction(pvals): Benjamini-Hochberg FDR
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
from scipy import stats


def _normal_enough(diff: np.ndarray) -> bool:
    """Check if paired differences are plausibly normal for small n.
    
    For n >= 30 the central limit theorem makes normality diagnostics
    unnecessary (paired t-test is robust).  For 8 <= n < 30 we use
    the Shapiro-Wilk test with a conservative alpha (0.01) to avoid
    false negatives.  For n < 8 we lack power to detect non-normality,
    so we conservatively assume non-normal and use Wilcoxon.
    
    Reference: Lumley et al. (2002). "The importance of the normality
    assumption in large public health data sets." Annu Rev Public Health.
    """
    n = len(diff)
    if n < 8:
        return False
    if n >= 30:
        return True  # CLT guarantees t-test robustness
    # 8 <= n < 30: run Shapiro with a conservative alpha
    try:
        p = float(stats.shapiro(diff).pvalue)
        return p > 0.01
    except Exception:
        return False


def cohen_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    md = float(np.mean(diff))
    sd = float(np.std(diff, ddof=1))
    if sd == 0:
        return 0.0
    return md / sd


def wilcoxon_effect_r(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    n = len(diff)
    if n == 0:
        return 0.0
    # compute W and approximate Z
    try:
        res = stats.wilcoxon(a, b, zero_method="wilcox", correction=False)
        W = float(res.statistic)
        # expected W mean and SD under H0
        mu = n * (n + 1) / 4.0
        sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 24.0)
        z = (W - mu) / sigma if sigma > 0 else 0.0
        r = z / math.sqrt(n) if n > 0 else 0.0
        return float(abs(r))
    except Exception:
        return 0.0


def paired_t_confidence_interval(a: np.ndarray, b: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    diff = a - b
    n = len(diff)
    if n == 0:
        return (float('nan'), float('nan'))
    md = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / math.sqrt(n))
    df = n - 1
    tcrit = stats.t.ppf(1 - alpha / 2.0, df) if df > 0 else 0.0
    return (md - tcrit * se, md + tcrit * se)


def run_paired_test(a: Iterable[float], b: Iterable[float]) -> Dict:
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    if len(a) != len(b):
        # try to align by truncation
        n = min(len(a), len(b))
        a = a[:n]
        b = b[:n]

    diff = a - b
    n = len(diff)
    result = {
        "n": int(n),
        "test": None,
        "stat": None,
        "pvalue": None,
        "effect_size": None,
        "ci": (None, None),
    }

    if n == 0:
        return result

    use_t = _normal_enough(diff)

    if use_t:
        tstat, p = stats.ttest_rel(a, b, nan_policy="omit")
        d = cohen_d_paired(a, b)
        ci = paired_t_confidence_interval(a, b)
        result.update({"test": "paired_t", "stat": float(tstat), "pvalue": float(p), "effect_size": float(d), "ci": ci})
    else:
        try:
            w = stats.wilcoxon(a, b, zero_method="wilcox", correction=False)
            p = float(w.pvalue)
            r = wilcoxon_effect_r(a, b)
            result.update({"test": "wilcoxon", "stat": float(w.statistic), "pvalue": float(p), "effect_size": float(r), "ci": (None, None)})
        except Exception:
            # fallback to t-test if wilcoxon fails
            tstat, p = stats.ttest_rel(a, b, nan_policy="omit")
            d = cohen_d_paired(a, b)
            ci = paired_t_confidence_interval(a, b)
            result.update({"test": "paired_t_fallback", "stat": float(tstat), "pvalue": float(p), "effect_size": float(d), "ci": ci})

    return result


def apply_bh_correction(pvals: List[float]) -> List[float]:
    """Apply Benjamini-Hochberg FDR correction; returns adjusted p-values in same order."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.empty(n, dtype=float)
    cummin = 1.0
    for i in range(n - 1, -1, -1):
        rank = i + 1
        val = min(cummin, ranked[i] * n / rank)
        cummin = val
        adjusted[i] = val
    # reorder to original
    adj_orig = np.empty(n, dtype=float)
    adj_orig[order] = adjusted
    # ensure monotonicity
    adj_orig = np.minimum.accumulate(adj_orig[::-1])[::-1]
    return adj_orig.tolist()
