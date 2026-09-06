"""The contamination filter, and the rest of the market-selection policy.

This module is the difference between a forecasting benchmark and a memory
test. Everything here exists to make one claim defensible:

    the model could not have known the answer when it answered.

Every exclusion is recorded by name so the funnel can be reported rather than
assumed. `pmbench eligible --explain` prints the counts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone

from .config import ModelConfig
from .gamma import effective_end


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    v = value.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(v)
    except ValueError:
        try:
            dt = datetime.fromisoformat(v[:19] + "+00:00")
        except ValueError:
            return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class EligibilityPolicy:
    """Selection policy. Report these numbers alongside any result."""

    mode: str = "live"
    # "live":          forecast open markets and wait for the world to answer.
    #                  Slow (weeks to a first label) and completely clean.
    # "retrospective": forecast markets that have ALREADY resolved, standing at
    #                  a past `as_of`. Gives you a labelled dataset today
    #                  instead of in six weeks, and is safe for exactly one
    #                  reason: the cutoff rule still applies, so the model is
    #                  only ever asked about outcomes that post-date its
    #                  training. Requires a real cutoff — a "first_observed"
    #                  cutoff is today's date, so nothing resolved can pass it,
    #                  and the selection will correctly come back empty.

    # --- contamination control ---
    strict_contamination: bool = False
    # False: require end_date > cutoff. The market resolves after the model's
    #        knowledge ends, so the *answer* cannot be memorised. This is the
    #        standard bar and what the headline numbers use.
    # True:  additionally require start_date > cutoff, so the market did not
    #        exist during training at all. Removes the subtler leak where the
    #        model memorised early trading on a long-running market. Costs a lot
    #        of sample size; use it as a robustness check on the headline.

    # --- baseline-3 validity ---
    min_volume_usd: float = 5_000.0
    # "Beat the market price" is only a meaningful baseline if the price is
    # actually aggregated money. A $200 market is one person's opinion.
    max_spread: float = 0.10
    # A wide spread means the "price" is a midpoint between two stale quotes;
    # scoring against it measures the spread, not the forecast.

    # --- question quality ---
    min_price: float = 0.02
    max_price: float = 0.98
    # Near-degenerate markets ("will the sun rise") are free Brier points for
    # everyone and swamp the mean. Excluding them makes the comparison
    # discriminative — but it IS a choice that flatters nothing and must be
    # reported, because it removes the easiest questions from every arm.

    # --- horizon ---
    min_horizon_days: float = 2.0    # under this, resolution is mechanical
    max_horizon_days: float = 365.0  # over this, you wait a year for one label

    require_resolution_criteria: bool = True
    min_criteria_chars: int = 80
    # A market with no written resolution criteria cannot be graded fairly and
    # is disproportionately likely to resolve on a technicality.


@dataclass
class Verdict:
    eligible: bool
    reasons: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.eligible


def evaluate(
    market: dict,
    snapshot: dict | None,
    cfg: ModelConfig,
    cutoff: date,
    *,
    now: datetime,
    policy: EligibilityPolicy | None = None,
) -> Verdict:
    """Decide whether `market` may be forecast by `cfg` at time `now`.

    `market` is a `markets` row; `snapshot` the latest `price_snapshots` row.
    `cutoff` comes from ModelConfig.cutoff_date() and is passed explicitly so
    that callers cannot accidentally forecast with an unresolved cutoff.
    """
    p = policy or EligibilityPolicy()
    bad: list[str] = []

    # effective_end, not end_date: a resolved neg-risk market's end_date can sit
    # years past the day it actually settled. See gamma.effective_end().
    end = parse_ts(effective_end(market))
    start = parse_ts(market.get("start_date"))

    # ---- 1. contamination ----------------------------------------------
    if end is None:
        bad.append("no_end_date")
    elif end.date() <= cutoff:
        # The single most important line in this file.
        bad.append("resolves_before_cutoff")
    if p.strict_contamination:
        if start is None:
            bad.append("no_start_date")
        elif start.date() <= cutoff:
            bad.append("existed_before_cutoff")

    # ---- 2. decided-ness, which flips meaning between the two modes ------
    if p.mode == "retrospective":
        # `now` is the as-of instant we are pretending to stand at. The market
        # must resolve after it (otherwise there is nothing to forecast) and
        # must have a label by today (otherwise there is nothing to score).
        if not market.get("resolved"):
            bad.append("not_yet_resolved")
        if market.get("outcome") is None:
            bad.append("no_outcome")
        if end is not None and end <= now:
            bad.append("as_of_after_resolution")
    else:
        if market.get("resolved"):
            bad.append("already_resolved")
        if market.get("closed"):
            bad.append("already_closed")
        if not market.get("active"):
            bad.append("inactive")
        if market.get("archived"):
            bad.append("archived")
        if end is not None and end <= now:
            bad.append("end_date_passed")

    # ---- 3. horizon ------------------------------------------------------
    if end is not None:
        horizon = (end - now).total_seconds() / 86400.0
        if horizon < p.min_horizon_days:
            bad.append("horizon_too_short")
        elif horizon > p.max_horizon_days:
            bad.append("horizon_too_long")

    # ---- 4. tradeable enough for baseline 3 -----------------------------
    if snapshot is None:
        bad.append("no_price_snapshot")
    else:
        vol = snapshot.get("volume")
        if vol is None or vol < p.min_volume_usd:
            bad.append("volume_too_low")
        spread = snapshot.get("spread")
        if spread is not None and spread > p.max_spread:
            bad.append("spread_too_wide")
        price = snapshot.get("yes_price")
        if price is None:
            bad.append("no_price")
        elif not (p.min_price <= price <= p.max_price):
            bad.append("price_degenerate")

    # ---- 5. gradeable ----------------------------------------------------
    if p.require_resolution_criteria:
        crit = market.get("resolution_criteria") or ""
        if len(crit) < p.min_criteria_chars:
            bad.append("no_resolution_criteria")

    return Verdict(not bad, bad)


def cutoff_note(cfg: ModelConfig, cutoff: date, policy: EligibilityPolicy) -> str:
    """Human-readable provenance line, stored on every forecast row."""
    rule = "start_date & end_date > cutoff" if policy.strict_contamination else "end_date > cutoff"
    note = f"{cutoff.isoformat()} ({cfg.cutoff_basis}); mode: {policy.mode}; rule: {rule}"
    if cfg.web_search:
        note += "; WEB SEARCH ON — cutoff guarantee does not bound what the model saw"
    return note
