"""Uncertainty: paired bootstrap confidence intervals and exact McNemar tests."""

from math import comb

import numpy as np


def bootstrap_diff_ci(a, b, n_boot=10000, seed=633, alpha=0.05):
    """95% CI for mean(b) - mean(a) on paired per-item scores (resampling items)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    lo, hi = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(b.mean() - a.mean()), float(lo), float(hi)


def bootstrap_mean_ci(x, n_boot=10000, seed=633, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(x.mean()), float(lo), float(hi)


def mcnemar_exact(a_pass, b_pass):
    """Exact two-sided McNemar test on paired pass/fail. Returns (b01, b10, p_value).

    b01 = items A failed and B passed; b10 = items A passed and B failed.
    """
    a, b = np.asarray(a_pass, dtype=bool), np.asarray(b_pass, dtype=bool)
    b01 = int(np.sum(~a & b))
    b10 = int(np.sum(a & ~b))
    n = b01 + b10
    if n == 0:
        return b01, b10, 1.0
    k = min(b01, b10)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return b01, b10, min(1.0, p)
