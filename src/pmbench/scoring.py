"""Proper scoring rules, reliability curves, and calibration fits.

Pure functions over (predictions, outcomes). No database, no network — this is
the part that has to be obviously correct, so it is trivially testable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

EPS = 1e-6  # log loss clip; an unclipped 0.0 on a YES outcome is infinite


def brier(p: float, outcome: int) -> float:
    """Squared error. Lower is better; 0.25 is the always-0.5 baseline."""
    return (p - outcome) ** 2


def log_loss(p: float, outcome: int) -> float:
    """Negative log likelihood in nats. Punishes confident errors far harder
    than Brier does, which is exactly why both are reported."""
    q = min(max(p, EPS), 1 - EPS)
    return -(math.log(q) if outcome == 1 else math.log(1 - q))


def brier_decomposition(
    probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> tuple[float, float, float]:
    """Murphy decomposition: Brier = reliability - resolution + uncertainty.

    reliability  how far bucket means sit from bucket frequencies (lower better)
    resolution   how much the forecasts vary from the base rate (higher better)
    uncertainty  base_rate * (1 - base_rate); a property of the questions, not
                 of the forecaster — this is why comparing raw Brier scores
                 across different question sets is meaningless.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return float("nan"), float("nan"), float("nan")
    base = y.mean()
    uncertainty = base * (1 - base)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    reliability = resolution = 0.0
    for b in range(n_bins):
        m = idx == b
        n_b = int(m.sum())
        if not n_b:
            continue
        reliability += n_b * (p[m].mean() - y[m].mean()) ** 2
        resolution += n_b * (y[m].mean() - base) ** 2
    n = p.size
    return reliability / n, resolution / n, uncertainty


@dataclass
class Bin:
    lo: float
    hi: float
    n: int
    mean_pred: float
    freq_actual: float


def reliability_bins(
    probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> list[Bin]:
    """Buckets for the reliability diagram. A calibrated forecaster has
    mean_pred == freq_actual in every bucket (the diagonal)."""
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    out: list[Bin] = []
    for b in range(n_bins):
        m = idx == b
        n_b = int(m.sum())
        if not n_b:
            continue
        out.append(Bin(edges[b], edges[b + 1], n_b, float(p[m].mean()), float(y[m].mean())))
    return out


def expected_calibration_error(
    probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10
) -> float:
    """Sample-weighted mean |predicted - actual| across buckets."""
    bins = reliability_bins(probs, outcomes, n_bins)
    n = sum(b.n for b in bins)
    if not n:
        return float("nan")
    return sum(b.n * abs(b.mean_pred - b.freq_actual) for b in bins) / n


def overconfidence(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Signed miscalibration: mean(prediction) - mean(outcome) on the
    "predicted direction" scale.

    Folds each forecast onto [0.5, 1] by taking the side it favours, then
    compares stated confidence to hit rate. Positive means the forecaster
    claims more certainty than it earns — the usual LLM failure mode.
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(outcomes, dtype=float)
    if p.size == 0:
        return float("nan")
    side = p >= 0.5
    stated = np.where(side, p, 1 - p)
    correct = np.where(side, y, 1 - y)
    return float(stated.mean() - correct.mean())


# --------------------------------------------------------------------------
# calibration fits
# --------------------------------------------------------------------------


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, EPS, 1 - EPS)
    return np.log(q / (1 - q))


def _sigmoid(z: np.ndarray | float):
    """Overflow-safe logistic. `1/(1+exp(-z))` raises for |z| ~ 750."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def _nll(x: np.ndarray, y: np.ndarray, a: float, b: float, ridge: float) -> float:
    z = np.clip(a * x + b, -60, 60)
    # log(1+exp(z)) computed stably
    ll = y * z - np.logaddexp(0.0, z)
    return float(-ll.sum() + 0.5 * ridge * (a * a + b * b))


def fit_platt(probs: Sequence[float], outcomes: Sequence[int], iters: int = 200,
              ridge: float = 1e-3) -> tuple[float, float]:
    """Platt scaling: fit sigmoid(a * logit(p) + b) by damped Newton (IRLS).

    `a < 1` means the raw forecasts were too extreme, i.e. overconfident —
    reading `a` off a fit is a cleaner statement of overconfidence than eyeballing
    a reliability diagram.

    Damping and a small ridge penalty are not optional here. Forecast sets are
    routinely near-separable (a model that says 0.99 and is usually right), and
    undamped Newton on separable logistic data diverges to infinite weights —
    which is how you get a "calibration fit" that returns a=2.4e7 and then
    overflows when applied.
    """
    x = _logit(np.asarray(probs, dtype=float))
    y = np.asarray(outcomes, dtype=float)
    if x.size == 0:
        return 1.0, 0.0
    a, b = 1.0, 0.0
    cur = _nll(x, y, a, b, ridge)
    for _ in range(iters):
        z = np.clip(a * x + b, -60, 60)
        p = _sigmoid(z)
        w = np.clip(p * (1 - p), 1e-8, None)
        r = p - y
        g = np.array([float((r * x).sum()) + ridge * a, float(r.sum()) + ridge * b])
        h = np.array([
            [float((w * x * x).sum()) + ridge, float((w * x).sum())],
            [float((w * x).sum()), float(w.sum()) + ridge],
        ])
        try:
            step = np.linalg.solve(h, g)
        except np.linalg.LinAlgError:
            break
        # Backtracking line search: Newton points downhill, but the full step
        # can overshoot badly on near-separable data.
        t = 1.0
        for _ in range(40):
            na, nb = a - t * float(step[0]), b - t * float(step[1])
            nxt = _nll(x, y, na, nb, ridge)
            if nxt <= cur:
                break
            t *= 0.5
        else:
            break
        if abs(cur - nxt) < 1e-10 and t * abs(step).max() < 1e-9:
            a, b, cur = na, nb, nxt
            break
        a, b, cur = na, nb, nxt
    return a, b


def apply_platt(p: float, a: float, b: float) -> float:
    q = min(max(p, EPS), 1 - EPS)
    z = a * math.log(q / (1 - q)) + b
    return float(_sigmoid(np.array([z]))[0])


def fit_temperature(probs: Sequence[float], outcomes: Sequence[int]) -> float:
    """Temperature scaling — Platt with the intercept pinned at 0.

    T > 1 softens the forecasts (the overconfident case); T < 1 sharpens them.
    """
    a, _ = fit_platt(probs, outcomes)
    return 1.0 / a if a else float("inf")


@dataclass
class Summary:
    n: int
    brier: float
    log_loss: float
    ece: float
    overconfidence: float
    base_rate: float
    reliability: float
    resolution: float
    uncertainty: float


def summarise(probs: Sequence[float], outcomes: Sequence[int], n_bins: int = 10) -> Summary:
    p = list(probs)
    y = list(outcomes)
    if not p:
        nan = float("nan")
        return Summary(0, nan, nan, nan, nan, nan, nan, nan, nan)
    rel, res, unc = brier_decomposition(p, y, n_bins)
    return Summary(
        n=len(p),
        brier=sum(brier(a, b) for a, b in zip(p, y)) / len(p),
        log_loss=sum(log_loss(a, b) for a, b in zip(p, y)) / len(p),
        ece=expected_calibration_error(p, y, n_bins),
        overconfidence=overconfidence(p, y),
        base_rate=sum(y) / len(y),
        reliability=rel,
        resolution=res,
        uncertainty=unc,
    )
