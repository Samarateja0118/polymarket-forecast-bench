"""The three baselines, in increasing order of difficulty.

    1. always 0.5                 — Brier 0.25 by construction
    2. category base rate         — computed leave-one-out (see below)
    3. market price at forecast   — aggregated money; the real bar

Baseline 2 has a trap. Computing a category's base rate from the same resolved
markets you are scoring lets the baseline peek at the answers, which inflates it
and makes your model look worse than it is. It is computed leave-one-out here:
each market's baseline excludes that market's own outcome.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .scoring import Summary, brier, log_loss, summarise


@dataclass
class Row:
    """One scored forecast, with everything needed to compare against baselines."""

    market_id: str
    category: str
    probability: float
    market_price: float | None
    outcome: int
    horizon_days: float | None = None
    ensemble_sd: float | None = None
    config_name: str = ""


def always_half(rows: Sequence[Row]) -> list[float]:
    return [0.5] * len(rows)


def category_base_rate(rows: Sequence[Row], prior_strength: float = 4.0) -> list[float]:
    """Leave-one-out base rate for each row's category.

    Smoothed toward the global rate with a Beta prior of `prior_strength`
    pseudo-observations, so a category with two resolved markets doesn't emit
    0.0 or 1.0.
    """
    by_cat: dict[str, list[int]] = defaultdict(list)
    for r in rows:
        by_cat[r.category].append(r.outcome)
    total = sum(r.outcome for r in rows)
    n_all = len(rows)
    global_rate = total / n_all if n_all else 0.5

    out: list[float] = []
    for r in rows:
        ys = by_cat[r.category]
        n_cat = len(ys) - 1                       # leave this market out
        s_cat = sum(ys) - r.outcome
        p = (s_cat + prior_strength * global_rate) / (n_cat + prior_strength)
        out.append(float(min(max(p, 1e-6), 1 - 1e-6)))
    return out


def market_prices(rows: Sequence[Row]) -> list[float | None]:
    """Price recorded at forecast time. Never read a later snapshot here — that
    is the lookahead bug that makes a backtest look profitable."""
    return [r.market_price for r in rows]


@dataclass
class Comparison:
    name: str
    n: int
    model: Summary
    baseline: Summary
    brier_delta: float          # model - baseline; negative means the model wins
    brier_skill_score: float    # 1 - model/baseline; positive means the model wins
    ci_low: float               # 95% paired-bootstrap CI on brier_delta
    ci_high: float
    p_two_sided: float


def paired_bootstrap(
    model_scores: Sequence[float],
    baseline_scores: Sequence[float],
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """CI and two-sided p-value on the paired Brier difference.

    Paired because both arms score the *same* markets: the market-to-market
    variance is enormous and unpaired tests will not see a real difference
    through it.
    """
    d = np.asarray(model_scores, dtype=float) - np.asarray(baseline_scores, dtype=float)
    n = d.size
    if n < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # Fraction of resamples on the far side of zero, doubled.
    frac = float(min((means <= 0).mean(), (means >= 0).mean()))
    return float(lo), float(hi), min(1.0, 2 * frac)


def compare(rows: Sequence[Row], baseline_probs: Sequence[float | None], name: str
            ) -> Comparison:
    """Score model vs one baseline on the markets where the baseline exists."""
    pairs = [
        (r, b) for r, b in zip(rows, baseline_probs)
        if b is not None and r.probability == r.probability  # drop NaN forecasts
    ]
    if not pairs:
        nan = float("nan")
        empty = summarise([], [])
        return Comparison(name, 0, empty, empty, nan, nan, nan, nan, nan)

    rs = [r for r, _ in pairs]
    bs = [float(b) for _, b in pairs]
    ys = [r.outcome for r in rs]
    ps = [r.probability for r in rs]

    m_scores = [brier(p, y) for p, y in zip(ps, ys)]
    b_scores = [brier(b, y) for b, y in zip(bs, ys)]
    lo, hi, pval = paired_bootstrap(m_scores, b_scores)
    m_mean = sum(m_scores) / len(m_scores)
    b_mean = sum(b_scores) / len(b_scores)

    return Comparison(
        name=name,
        n=len(pairs),
        model=summarise(ps, ys),
        baseline=summarise(bs, ys),
        brier_delta=m_mean - b_mean,
        brier_skill_score=(1 - m_mean / b_mean) if b_mean else float("nan"),
        ci_low=lo,
        ci_high=hi,
        p_two_sided=pval,
    )


def all_comparisons(rows: Sequence[Row]) -> list[Comparison]:
    return [
        compare(rows, always_half(rows), "always_0.5"),
        compare(rows, category_base_rate(rows), "category_base_rate"),
        compare(rows, market_prices(rows), "market_price"),
    ]


def format_comparisons(comps: Iterable[Comparison]) -> str:
    lines = [
        f"{'baseline':<20} {'n':>5} {'model':>8} {'base':>8} {'delta':>9} "
        f"{'95% CI':>19} {'p':>7} {'skill':>8}",
        "-" * 92,
    ]
    for c in comps:
        lines.append(
            f"{c.name:<20} {c.n:>5} {c.model.brier:>8.4f} {c.baseline.brier:>8.4f} "
            f"{c.brier_delta:>+9.4f} [{c.ci_low:>+7.4f},{c.ci_high:>+7.4f}] "
            f"{c.p_two_sided:>7.3f} {c.brier_skill_score:>+8.3f}"
        )
    lines.append("")
    lines.append("delta = model Brier - baseline Brier; negative means the model is better.")
    return "\n".join(lines)
