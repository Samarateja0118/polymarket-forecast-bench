"""Phase 4: the questions the infrastructure exists to answer."""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .baselines import Comparison, Row, all_comparisons
from .scoring import Summary, apply_platt, brier, fit_platt, fit_temperature, summarise


def load_rows(conn: sqlite3.Connection, config_name: str | None = None) -> list[Row]:
    """Every scored forecast, joined to the fields analysis needs."""
    sql = (
        "SELECT f.market_id, f.config_name, f.probability, f.market_price, "
        "       f.horizon_days, f.ensemble_sd, m.category, s.outcome "
        "FROM scores s JOIN forecasts f ON f.id = s.forecast_id "
        "JOIN markets m ON m.id = f.market_id"
    )
    params: tuple = ()
    if config_name:
        sql += " WHERE f.config_name = ?"
        params = (config_name,)
    return [
        Row(
            market_id=r["market_id"], category=r["category"] or "Other",
            probability=float(r["probability"]), 
            market_price=None if r["market_price"] is None else float(r["market_price"]),
            outcome=int(r["outcome"]),
            horizon_days=r["horizon_days"], ensemble_sd=r["ensemble_sd"],
            config_name=r["config_name"],
        )
        for r in conn.execute(sql, params)
    ]


def by_config(rows: Sequence[Row]) -> dict[str, list[Row]]:
    out: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        out[r.config_name].append(r)
    return dict(out)


def leaderboard(rows: Sequence[Row]) -> dict[str, list[Comparison]]:
    return {name: all_comparisons(rs) for name, rs in sorted(by_config(rows).items())}


# --------------------------------------------------------------------------
# Q: is the model overconfident, and can a fit correct it?
# --------------------------------------------------------------------------


@dataclass
class CalibrationResult:
    n_train: int
    n_test: int
    platt_a: float
    platt_b: float
    temperature: float
    brier_before: float
    brier_after: float

    @property
    def improved(self) -> bool:
        return self.brier_after < self.brier_before

    def __str__(self) -> str:
        verdict = "overconfident" if self.platt_a < 1 else "underconfident"
        return (
            f"Platt a={self.platt_a:.3f} b={self.platt_b:+.3f} (T={self.temperature:.2f}) "
            f"-> forecasts are {verdict}\n"
            f"held-out Brier {self.brier_before:.4f} -> {self.brier_after:.4f} "
            f"({'improved' if self.improved else 'WORSE'}; n_train={self.n_train}, "
            f"n_test={self.n_test})"
        )


def calibration_fit(rows: Sequence[Row], *, test_frac: float = 0.3, seed: int = 0
                    ) -> CalibrationResult:
    """Fit Platt scaling on a train split, report the gain on held-out test.

    Fitting and evaluating on the same forecasts always "improves" calibration.
    The split is the whole point; a correction that does not transfer is not a
    correction.
    """
    idx = np.arange(len(rows))
    np.random.default_rng(seed).shuffle(idx)
    cut = int(len(rows) * (1 - test_frac))
    train = [rows[i] for i in idx[:cut]]
    test = [rows[i] for i in idx[cut:]]
    if len(train) < 10 or len(test) < 5:
        nan = float("nan")
        return CalibrationResult(len(train), len(test), nan, nan, nan, nan, nan)

    a, b = fit_platt([r.probability for r in train], [r.outcome for r in train])
    t = fit_temperature([r.probability for r in train], [r.outcome for r in train])
    before = sum(brier(r.probability, r.outcome) for r in test) / len(test)
    after = sum(brier(apply_platt(r.probability, a, b), r.outcome) for r in test) / len(test)
    return CalibrationResult(len(train), len(test), a, b, t, before, after)


# --------------------------------------------------------------------------
# Q: does ensemble disagreement predict error? (-> an abstention rule)
# --------------------------------------------------------------------------


@dataclass
class AbstentionCurve:
    keep_fracs: list[float]
    briers: list[float]
    ns: list[int]
    correlation: float   # Spearman(|error|, ensemble_sd)

    def __str__(self) -> str:
        head = f"Spearman(|error|, ensemble SD) = {self.correlation:+.3f}\n"
        head += f"{'keep':>6} {'n':>5} {'Brier':>8}\n"
        rows = "\n".join(
            f"{k:>6.0%} {n:>5} {b:>8.4f}"
            for k, n, b in zip(self.keep_fracs, self.ns, self.briers)
        )
        return head + rows


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    return float((ra * rb).sum() / denom) if denom else float("nan")


def abstention_curve(rows: Sequence[Row]) -> AbstentionCurve:
    """Drop the most-disagreeing forecasts and watch the Brier score.

    If disagreement is a real uncertainty signal, keeping only the confident
    forecasts should lower the Brier score monotonically. If the curve is flat,
    the ensemble spread is noise and the abstention rule is worthless — which
    is itself a publishable result about the aggregation method.
    """
    usable = [r for r in rows if r.ensemble_sd is not None]
    if not usable:
        return AbstentionCurve([], [], [], float("nan"))
    err = np.array([abs(r.probability - r.outcome) for r in usable])
    sd = np.array([float(r.ensemble_sd) for r in usable])
    order = np.argsort(sd)  # most agreement first
    keep_fracs, briers, ns = [], [], []
    for frac in (1.0, 0.9, 0.75, 0.5, 0.25):
        k = max(1, int(len(usable) * frac))
        sub = [usable[i] for i in order[:k]]
        keep_fracs.append(frac)
        ns.append(k)
        briers.append(sum(brier(r.probability, r.outcome) for r in sub) / k)
    return AbstentionCurve(keep_fracs, briers, ns, _spearman(err, sd))


# --------------------------------------------------------------------------
# Q: which categories? Q: does horizon hurt?
# --------------------------------------------------------------------------


def by_category(rows: Sequence[Row], min_n: int = 5) -> dict[str, tuple[Summary, float | None]]:
    """Per-category (model summary, mean market-price Brier)."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r.category].append(r)
    out = {}
    for cat, rs in groups.items():
        if len(rs) < min_n:
            continue
        s = summarise([r.probability for r in rs], [r.outcome for r in rs])
        priced = [r for r in rs if r.market_price is not None]
        mkt = (sum(brier(r.market_price, r.outcome) for r in priced) / len(priced)
               if priced else None)
        out[cat] = (s, mkt)
    return dict(sorted(out.items(), key=lambda kv: kv[1][0].brier))


HORIZON_BUCKETS = [(0, 7), (7, 30), (30, 90), (90, 365), (365, 10_000)]


def by_horizon(rows: Sequence[Row], min_n: int = 5) -> dict[str, tuple[Summary, float | None]]:
    out = {}
    for lo, hi in HORIZON_BUCKETS:
        rs = [r for r in rows
              if r.horizon_days is not None and lo <= r.horizon_days < hi]
        if len(rs) < min_n:
            continue
        s = summarise([r.probability for r in rs], [r.outcome for r in rs])
        priced = [r for r in rs if r.market_price is not None]
        mkt = (sum(brier(r.market_price, r.outcome) for r in priced) / len(priced)
               if priced else None)
        out[f"{lo}-{hi}d"] = (s, mkt)
    return out


# --------------------------------------------------------------------------
# Q: does search help, or does it just anchor on the price it finds?
# --------------------------------------------------------------------------


@dataclass
class AnchoringResult:
    n: int
    mean_abs_gap_a: float   # |forecast - market price|, config A
    mean_abs_gap_b: float   # ... config B
    corr_a: float           # correlation(forecast, market price)
    corr_b: float

    def __str__(self) -> str:
        return (
            f"paired markets n={self.n}\n"
            f"  mean |forecast - price|: A {self.mean_abs_gap_a:.4f}  "
            f"B {self.mean_abs_gap_b:.4f}\n"
            f"  corr(forecast, price):   A {self.corr_a:+.3f}  B {self.corr_b:+.3f}\n"
            "  B hugging the price more tightly than A is the anchoring signature: "
            "search found the market, it did not find the world."
        )


def anchoring(rows: Sequence[Row], config_a: str, config_b: str) -> AnchoringResult:
    """Compare how tightly two configs track the market price on shared markets.

    The interesting failure mode for a search-enabled config is not "it is
    wrong" but "it read the price off Polymarket and repeated it", which looks
    like skill against every baseline except the market itself.
    """
    ga = {r.market_id: r for r in rows if r.config_name == config_a}
    gb = {r.market_id: r for r in rows if r.config_name == config_b}
    shared = [m for m in ga.keys() & gb.keys() if ga[m].market_price is not None]
    if len(shared) < 3:
        nan = float("nan")
        return AnchoringResult(len(shared), nan, nan, nan, nan)
    pa = np.array([ga[m].probability for m in shared])
    pb = np.array([gb[m].probability for m in shared])
    mk = np.array([ga[m].market_price for m in shared], dtype=float)
    return AnchoringResult(
        n=len(shared),
        mean_abs_gap_a=float(np.abs(pa - mk).mean()),
        mean_abs_gap_b=float(np.abs(pb - mk).mean()),
        corr_a=float(np.corrcoef(pa, mk)[0, 1]),
        corr_b=float(np.corrcoef(pb, mk)[0, 1]),
    )
