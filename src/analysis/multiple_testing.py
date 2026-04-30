"""Multiple-testing adjustments for primary outcome families.

Implements Holm-Bonferroni and Benjamini-Hochberg (FDR) corrections
without external dependencies. Both functions accept a list of raw
p-values, may include NaN for under-powered comparisons, and return
adjusted p-values in the original order.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


def _prep(p_values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(p_values), dtype=float)
    valid_mask = ~np.isnan(arr)
    return arr, valid_mask


def holm_bonferroni(p_values: Iterable[float]) -> list[float]:
    """Holm step-down adjusted p-values, NaN-preserving."""
    arr, valid_mask = _prep(p_values)
    out = arr.copy()
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return [float(x) for x in out]
    valid_p = arr[valid_idx]
    order = np.argsort(valid_p)
    m = len(valid_p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx_in_sorted in enumerate(order):
        raw = valid_p[idx_in_sorted]
        candidate = min(1.0, raw * (m - rank))
        running_max = max(running_max, candidate)
        adjusted[idx_in_sorted] = running_max
    out[valid_idx] = adjusted
    return [float(x) for x in out]


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    """BH step-up FDR-adjusted p-values, NaN-preserving."""
    arr, valid_mask = _prep(p_values)
    out = arr.copy()
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return [float(x) for x in out]
    valid_p = arr[valid_idx]
    order = np.argsort(valid_p)
    m = len(valid_p)
    adjusted_sorted = np.empty(m, dtype=float)
    running_min = 1.0
    for rank in range(m - 1, -1, -1):
        idx_in_sorted = order[rank]
        raw = valid_p[idx_in_sorted]
        candidate = min(1.0, raw * m / (rank + 1))
        running_min = min(running_min, candidate)
        adjusted_sorted[idx_in_sorted] = running_min
    out[valid_idx] = adjusted_sorted
    return [float(x) for x in out]
