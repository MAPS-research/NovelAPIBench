"""Statistical procedures (Appendix D.5).

Tasks are not independent draws: Stage 3 writes up to three tasks per target API from one
knowledge bundle. Every interval therefore resamples *target APIs*, not tasks:

* ``mean_ci``          95% BCa cluster-bootstrap interval of a task-weighted mean
                       (10,000 resamples of target APIs, seed 20260912);
* ``paired_diff_ci``   the same on per-task differences between two arms scored on the same
                       tasks (one resampling draw applied to both arms);
* ``mcnemar_exact``    exact McNemar test on paired binary outcomes;
* ``holm``             Holm step-down correction within a family of comparisons;
* ``interaction_test`` difference of a paired contrast between two disjoint sets of APIs
                       (novelty type, Section 4.2 and Appendix E.2): stratified API-cluster bootstrap
                       percentile interval (10,000) and a two-sided API-level label-permutation
                       p-value (5,000), seed 2026.

``clusters`` is a sequence parallel to the values naming each task's target API. Cluster codes
are assigned in order of first appearance, so the bootstrap draws (and hence the interval
endpoints, not the point estimates) depend on the row order of the input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats as _sps

DEFAULT_N_BOOT = 10000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 20260912


@dataclass(frozen=True)
class Interval:
    """A point estimate with a confidence interval (fractions, not percentages)."""

    point: float
    lo: float
    hi: float
    n: int
    n_clusters: int

    def fmt(self, digits: int = 1, scale: float = 100.0) -> str:
        return (f"{scale * self.point:.{digits}f} "
                f"[{scale * self.lo:.{digits}f}, {scale * self.hi:.{digits}f}]")


# ---------------------------------------------------------------------------------------------
# Cluster bootstrap
# ---------------------------------------------------------------------------------------------

def _cluster_index(clusters: Sequence) -> tuple[np.ndarray, int]:
    """Dense integer codes for the cluster labels, in order of first appearance."""
    uniq: dict = {}
    codes = np.empty(len(clusters), dtype=np.int64)
    for i, c in enumerate(clusters):
        codes[i] = uniq.setdefault(c, len(uniq))
    return codes, len(uniq)


def _cluster_sums(values: np.ndarray, codes: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    sums = np.bincount(codes, weights=values, minlength=k)
    counts = np.bincount(codes, minlength=k).astype(np.float64)
    return sums, counts


def _boot_means(sums: np.ndarray, counts: np.ndarray, n_boot: int,
                rng: np.random.Generator) -> np.ndarray:
    """Bootstrap distribution of the task-weighted mean when whole clusters are resampled."""
    k = len(sums)
    idx = rng.integers(0, k, size=(n_boot, k))
    num = sums[idx].sum(axis=1)
    den = counts[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _jackknife_means(sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Leave-one-cluster-out means (BCa acceleration)."""
    den = counts.sum() - counts
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0, (sums.sum() - sums) / den, np.nan)
    return out[np.isfinite(out)]


def _bca_interval(boot: np.ndarray, theta_hat: float, jack: np.ndarray,
                  alpha: float) -> tuple[float, float]:
    """Bias-corrected and accelerated percentile interval.

    Falls back to the percentile interval when the bias correction is undefined (every
    replicate on one side of the estimate, e.g. a pass@1 of 0) or the acceleration degenerates.
    """
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return float("nan"), float("nan")
    lo_pct, hi_pct = 100 * alpha / 2, 100 * (1 - alpha / 2)

    prop = float(np.mean(boot < theta_hat))
    if not 0.0 < prop < 1.0:
        return float(np.percentile(boot, lo_pct)), float(np.percentile(boot, hi_pct))
    z0 = _sps.norm.ppf(prop)

    jack_mean = jack.mean()
    num = float(np.sum((jack_mean - jack) ** 3))
    den = 6.0 * float(np.sum((jack_mean - jack) ** 2)) ** 1.5
    a = num / den if den > 0 else 0.0

    def adjusted(z: float) -> float | None:
        d = 1 - a * (z0 + z)
        if abs(d) < 1e-12:
            return None
        return float(_sps.norm.cdf(z0 + (z0 + z) / d))

    p_lo, p_hi = adjusted(_sps.norm.ppf(alpha / 2)), adjusted(_sps.norm.ppf(1 - alpha / 2))
    if p_lo is None or p_hi is None:
        return float(np.percentile(boot, lo_pct)), float(np.percentile(boot, hi_pct))
    p_lo = min(max(p_lo, 1e-6), 1 - 1e-6)
    p_hi = min(max(p_hi, 1e-6), 1 - 1e-6)
    return float(np.percentile(boot, 100 * p_lo)), float(np.percentile(boot, 100 * p_hi))


def _cluster_ci(v: np.ndarray, clusters: Sequence | None, alpha: float, n_boot: int,
                seed: int) -> Interval:
    n = v.size
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"), 0, 0)
    codes, k = _cluster_index(list(clusters) if clusters is not None else list(range(n)))
    sums, counts = _cluster_sums(v, codes, k)
    theta = float(v.mean())
    boot = _boot_means(sums, counts, n_boot, np.random.default_rng(seed))
    lo, hi = _bca_interval(boot, theta, _jackknife_means(sums, counts), alpha)
    return Interval(theta, lo, hi, n, k)


def mean_ci(values: Iterable[float], clusters: Sequence | None = None, *,
            alpha: float = DEFAULT_ALPHA, n_boot: int = DEFAULT_N_BOOT,
            seed: int = DEFAULT_SEED) -> Interval:
    """Cluster-bootstrap BCa interval of the mean of per-task values (pass@1 or a 0/1 share)."""
    return _cluster_ci(np.asarray(list(values), dtype=np.float64), clusters, alpha, n_boot, seed)


def paired_diff_ci(a: Iterable[float], b: Iterable[float], clusters: Sequence | None = None, *,
                   alpha: float = DEFAULT_ALPHA, n_boot: int = DEFAULT_N_BOOT,
                   seed: int = DEFAULT_SEED) -> Interval:
    """Interval of ``mean(a) - mean(b)`` for two arms scored on the same tasks (``a[i]`` and
    ``b[i]`` are the same task); the resampled clusters are shared by both arms."""
    va = np.asarray(list(a), dtype=np.float64)
    vb = np.asarray(list(b), dtype=np.float64)
    if va.shape != vb.shape:
        raise ValueError(f"paired arms must align: {va.shape} vs {vb.shape}")
    return _cluster_ci(va - vb, clusters, alpha, n_boot, seed)


# ---------------------------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------------------------

def mcnemar_exact(a: Iterable[float], b: Iterable[float]) -> tuple[int, int, float]:
    """Exact McNemar test. Returns ``(n01, n10, p)``, where ``n10`` counts tasks that ``a``
    solves and ``b`` does not."""
    va = np.asarray(list(a), dtype=np.float64) > 0.5
    vb = np.asarray(list(b), dtype=np.float64) > 0.5
    n10 = int(np.sum(va & ~vb))
    n01 = int(np.sum(~va & vb))
    m = n01 + n10
    if m == 0:
        return n01, n10, 1.0
    return n01, n10, float(_sps.binomtest(n10, m, 0.5).pvalue)


def holm(pvalues: dict, alpha: float = DEFAULT_ALPHA) -> dict:
    """Holm-Bonferroni step-down correction: ``{key: (adjusted_p, adjusted_p <= alpha)}``."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out: dict = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = min(1.0, max(running, (m - i) * p))
        out[key] = (running, running <= alpha)
    return out


def interaction_test(diff_a: Sequence[float], api_a: Sequence[str],
                     diff_b: Sequence[float], api_b: Sequence[str], *,
                     n_boot: int = 10000, n_perm: int = 5000, seed: int = 2026) -> dict:
    """``mean(diff | A) - mean(diff | B)`` for per-task paired differences on two disjoint API sets.

    Interval: stratified cluster bootstrap (APIs resampled with replacement within A and within
    B, every task of a drawn API kept, task-weighted means), 2.5/97.5 percentiles.
    p-value: two-sided permutation of the A/B label across APIs with group sizes fixed,
    ``(k + 1) / (n_perm + 1)``. APIs are taken in sorted order; the means are computed from
    per-API sums and task counts.
    """
    def per_api(diff: Sequence[float], api: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        names = np.asarray(api, dtype=object)
        d = np.asarray(diff, dtype=np.float64)
        keys = sorted(set(names))
        pos = {k: i for i, k in enumerate(keys)}
        codes = np.fromiter((pos[x] for x in names), dtype=np.int64, count=len(names))
        return _cluster_sums(d, codes, len(keys))

    sa, ca = per_api(diff_a, api_a)
    sb, cb = per_api(diff_b, api_b)
    na, nb = len(sa), len(sb)
    obs = sa.sum() / ca.sum() - sb.sum() / cb.sum()
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, na, na)
        ib = rng.integers(0, nb, nb)
        boots[i] = sa[ia].sum() / ca[ia].sum() - sb[ib].sum() / cb[ib].sum()
    s, c = np.concatenate([sa, sb]), np.concatenate([ca, cb])
    perm = np.empty(n_perm)
    for i in range(n_perm):
        idx = rng.permutation(na + nb)
        x, y = idx[:na], idx[na:]
        perm[i] = s[x].sum() / c[x].sum() - s[y].sum() / c[y].sum()
    p = (np.sum(np.abs(perm) >= abs(obs)) + 1) / (n_perm + 1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"delta": float(obs), "ci_lo": float(lo), "ci_hi": float(hi), "perm_p": float(p)}
