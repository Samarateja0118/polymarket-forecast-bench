from datetime import date, datetime, timezone

import pytest

from pmbench.config import ModelConfig
from pmbench.eligibility import EligibilityPolicy, evaluate, parse_ts

NOW = datetime(2026, 9, 6, tzinfo=timezone.utc)
CUTOFF = date(2026, 1, 1)
CFG = ModelConfig(name="t", model_id="m", effective_cutoff="2026-01-01",
                  cutoff_basis="documented")


def market(**over):
    base = dict(
        id="1", question="Q?", resolution_criteria="x" * 200,
        start_date="2026-02-01T00:00:00Z", end_date="2026-12-01T00:00:00Z",
        resolved=0, closed=0, active=1, archived=0,
    )
    base.update(over)
    return base


def snap(**over):
    base = dict(yes_price=0.4, volume=50_000.0, liquidity=10_000.0, spread=0.01)
    base.update(over)
    return base


def test_clean_market_is_eligible():
    v = evaluate(market(), snap(), CFG, CUTOFF, now=NOW)
    assert v.eligible, v.reasons


def test_market_resolving_before_cutoff_is_rejected():
    """The single rule that separates a forecasting benchmark from a memory test."""
    v = evaluate(market(end_date="2025-06-01T00:00:00Z"), snap(), CFG, CUTOFF, now=NOW)
    assert "resolves_before_cutoff" in v.reasons


def test_end_date_exactly_at_cutoff_is_rejected():
    v = evaluate(market(end_date="2026-01-01T00:00:00Z"), snap(), CFG, CUTOFF, now=NOW)
    assert "resolves_before_cutoff" in v.reasons


def test_strict_mode_also_rejects_markets_that_existed_during_training():
    m = market(start_date="2025-03-01T00:00:00Z")
    assert evaluate(m, snap(), CFG, CUTOFF, now=NOW).eligible
    strict = EligibilityPolicy(strict_contamination=True)
    v = evaluate(m, snap(), CFG, CUTOFF, now=NOW, policy=strict)
    assert "existed_before_cutoff" in v.reasons


def test_thin_markets_are_rejected_so_the_price_baseline_means_something():
    v = evaluate(market(), snap(volume=200.0), CFG, CUTOFF, now=NOW)
    assert "volume_too_low" in v.reasons


def test_wide_spread_rejected():
    v = evaluate(market(), snap(spread=0.4), CFG, CUTOFF, now=NOW)
    assert "spread_too_wide" in v.reasons


def test_degenerate_prices_rejected():
    assert "price_degenerate" in evaluate(market(), snap(yes_price=0.995), CFG, CUTOFF, now=NOW).reasons
    assert "price_degenerate" in evaluate(market(), snap(yes_price=0.004), CFG, CUTOFF, now=NOW).reasons


def test_already_resolved_or_closed_rejected():
    assert "already_resolved" in evaluate(market(resolved=1), snap(), CFG, CUTOFF, now=NOW).reasons
    assert "already_closed" in evaluate(market(closed=1), snap(), CFG, CUTOFF, now=NOW).reasons


def test_horizon_bounds():
    soon = market(end_date="2026-09-07T00:00:00Z")
    assert "horizon_too_short" in evaluate(soon, snap(), CFG, CUTOFF, now=NOW).reasons
    far = market(end_date="2030-01-01T00:00:00Z")
    assert "horizon_too_long" in evaluate(far, snap(), CFG, CUTOFF, now=NOW).reasons


def test_missing_resolution_criteria_rejected():
    v = evaluate(market(resolution_criteria=""), snap(), CFG, CUTOFF, now=NOW)
    assert "no_resolution_criteria" in v.reasons


def test_no_snapshot_rejected():
    v = evaluate(market(), None, CFG, CUTOFF, now=NOW)
    assert "no_price_snapshot" in v.reasons


def test_cutoff_auto_refuses_to_guess():
    auto = ModelConfig(name="a", model_id="unseen")
    with pytest.raises(ValueError, match="never been called"):
        auto.cutoff_date({})
    assert auto.cutoff_date({"unseen": "2026-05-05"}) == date(2026, 5, 5)


def test_parse_ts_handles_gamma_formats():
    assert parse_ts("2026-01-01T00:00:00Z").tzinfo is not None
    assert parse_ts("2026-01-01T00:00:00.228Z").year == 2026
    assert parse_ts(None) is None
    assert parse_ts("garbage") is None
