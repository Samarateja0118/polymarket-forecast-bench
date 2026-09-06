"""Storage, dedup, resolution, baselines — the parts that touch the database."""
import json

import pytest

from pmbench.baselines import (
    Row, all_comparisons, category_base_rate, compare, market_prices, paired_bootstrap,
)
from pmbench.db import connect, observed_cutoffs, record_observation
from pmbench.gamma import normalise, pick_category, snapshot_of
from pmbench.ingest import IngestStats, append_snapshot, upsert_market
from pmbench.prompting import register
from pmbench.resolve import decide, score_pending


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    yield c
    c.close()


# ---------------- gamma normalisation ----------------

EVENT = {"id": "9", "title": "E", "tags": [{"label": "Sports"}, {"label": "Politics"}]}


def raw_market(**over):
    base = {
        "id": "1", "conditionId": "0xc", "slug": "s", "question": "Q?",
        "description": "d" * 200, "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.4", "0.6"]', "clobTokenIds": '["tok1", "tok2"]',
        "endDate": "2026-12-01T00:00:00Z", "startDate": "2026-02-01T00:00:00Z",
        "active": True, "closed": False, "volumeNum": 50000.0, "liquidityNum": 9000.0,
        "bestBid": 0.39, "bestAsk": 0.41, "spread": 0.02,
    }
    base.update(over)
    return base


def test_non_binary_markets_are_dropped_not_coerced():
    """Scoring assumes one probability against a binary outcome; silently
    flattening a 4-way market into that shape is how benchmarks go wrong."""
    assert normalise(EVENT, raw_market(outcomes='["A","B","C"]')) is None
    assert normalise(EVENT, raw_market()) is not None


def test_category_uses_priority_not_tag_order():
    assert pick_category(EVENT["tags"]) == "Politics"
    assert pick_category([]) == "Other"


def test_snapshot_parses_string_encoded_prices():
    s = snapshot_of(raw_market())
    assert s["yes_price"] == pytest.approx(0.4)
    assert s["volume"] == 50000.0


# ---------------- ingest ----------------


def test_snapshots_dedupe_until_something_moves(conn):
    st = IngestStats()
    row = normalise(EVENT, raw_market())
    upsert_market(conn, row, "2026-09-06T00:00:00Z", st)
    snap = snapshot_of(raw_market())

    append_snapshot(conn, "1", snap, "2026-09-06T00:00:00Z", st)
    append_snapshot(conn, "1", snap, "2026-09-06T01:00:00Z", st)
    append_snapshot(conn, "1", snap, "2026-09-06T02:00:00Z", st)
    assert st.snapshots_written == 1 and st.snapshots_deduped == 2

    moved = snapshot_of(raw_market(outcomePrices='["0.55", "0.45"]'))
    append_snapshot(conn, "1", moved, "2026-09-06T03:00:00Z", st)
    assert st.snapshots_written == 2


def test_sub_cent_volume_noise_is_not_a_price_change(conn):
    st = IngestStats()
    upsert_market(conn, normalise(EVENT, raw_market()), "t", st)
    append_snapshot(conn, "1", snapshot_of(raw_market()), "t1", st)
    append_snapshot(conn, "1", snapshot_of(raw_market(volumeNum=50000.000001)), "t2", st)
    assert st.snapshots_written == 1


def test_upsert_is_idempotent_and_flags_real_changes(conn):
    st = IngestStats()
    upsert_market(conn, normalise(EVENT, raw_market()), "t", st)
    upsert_market(conn, normalise(EVENT, raw_market()), "t", st)
    assert st.markets_new == 1 and st.markets_changed == 0
    upsert_market(conn, normalise(EVENT, raw_market(question="Q2?")), "t", st)
    assert st.markets_changed == 1
    assert conn.execute("SELECT COUNT(*) FROM markets").fetchone()[0] == 1


def test_ingest_never_clobbers_a_resolution(conn):
    """resolve.py owns `resolved`/`outcome`; a later poll must not reset them."""
    st = IngestStats()
    upsert_market(conn, normalise(EVENT, raw_market()), "t", st)
    conn.execute("UPDATE markets SET resolved=1, outcome=1 WHERE id='1'")
    upsert_market(conn, normalise(EVENT, raw_market(question="Q3?")), "t", st)
    r = conn.execute("SELECT resolved, outcome FROM markets WHERE id='1'").fetchone()
    assert r["resolved"] == 1 and r["outcome"] == 1


# ---------------- cutoff observation ----------------


def test_first_observed_never_moves_forward(conn):
    record_observation(conn, "m", "2026-05-01")
    record_observation(conn, "m", "2026-09-01")
    assert observed_cutoffs(conn)["m"] == "2026-05-01"


# ---------------- prompt versioning ----------------


def test_editing_a_prompt_version_in_place_raises(conn, monkeypatch):
    import pmbench.prompting as P
    monkeypatch.setattr(P, "load", lambda v: "body one")
    register(conn, "vX")
    monkeypatch.setattr(P, "load", lambda v: "body two")
    with pytest.raises(ValueError, match="has changed"):
        register(conn, "vX")


# ---------------- resolution ----------------


def test_resolution_requires_closed_and_settled_prices():
    assert decide(raw_market(closed=True, outcomePrices='["1","0"]')).outcome == 1
    assert decide(raw_market(closed=True, outcomePrices='["0","1"]')).outcome == 0
    assert decide(raw_market(closed=False, outcomePrices='["1","0"]')).outcome is None


def test_closed_but_still_trading_is_not_a_resolution():
    r = decide(raw_market(closed=True, outcomePrices='["0.6","0.4"]'))
    assert r.outcome is None and "unsettled" in r.source


def test_disputed_resolutions_are_held():
    r = decide(raw_market(closed=True, outcomePrices='["1","0"]',
                          umaResolutionStatuses='["disputed"]'))
    assert r.outcome is None and "disputed" in r.source


def test_resolution_records_its_evidence():
    assert "prices=" in decide(raw_market(closed=True, outcomePrices='["1","0"]')).source


def test_score_pending_writes_one_row_per_forecast(conn):
    st = IngestStats()
    upsert_market(conn, normalise(EVENT, raw_market()), "t", st)
    conn.execute("UPDATE markets SET resolved=1, outcome=1 WHERE id='1'")
    conn.execute("INSERT INTO runs(id, ts) VALUES ('r','t')")
    conn.execute(
        "INSERT INTO forecasts(id, run_id, market_id, ts, config_name, model_id, "
        "prompt_version, probability, market_price, horizon_days) "
        "VALUES ('f','r','1','t','c','m','v1',0.8,0.6,30)"
    )
    conn.commit()
    assert score_pending(conn) == 1
    assert score_pending(conn) == 0          # idempotent
    s = conn.execute("SELECT * FROM scores WHERE forecast_id='f'").fetchone()
    assert s["brier"] == pytest.approx(0.04)
    assert s["brier_market"] == pytest.approx(0.16)


# ---------------- baselines ----------------


def rows(n=40):
    out = []
    for i in range(n):
        cat = "A" if i % 2 else "B"
        out.append(Row(f"m{i}", cat, 0.6 if i % 3 else 0.3, 0.55, int(i % 4 == 0), 30.0, 0.05, "cfg"))
    return out


def test_category_base_rate_leaves_the_market_itself_out():
    """Using a market's own outcome in its baseline lets the baseline peek."""
    rs = [Row("a", "C", 0.5, 0.5, 1), Row("b", "C", 0.5, 0.5, 1), Row("c", "C", 0.5, 0.5, 0)]
    base = category_base_rate(rs, prior_strength=0.0)
    assert base[0] == pytest.approx(0.5)   # own YES excluded -> 1 of 2 remaining
    assert base[2] == pytest.approx(1.0, abs=1e-5)  # own NO excluded; clamped off 1.0


def test_all_three_baselines_are_scored():
    comps = all_comparisons(rows())
    assert [c.name for c in comps] == ["always_0.5", "category_base_rate", "market_price"]
    assert all(c.n > 0 for c in comps)


def test_always_half_baseline_is_exactly_quarter():
    c = all_comparisons(rows())[0]
    assert c.baseline.brier == pytest.approx(0.25)


def test_comparison_skips_markets_with_no_recorded_price():
    rs = rows()
    rs[0].market_price = None
    assert compare(rs, market_prices(rs), "market_price").n == len(rs) - 1


def test_paired_bootstrap_finds_a_real_difference_and_ignores_a_fake_one():
    same = [0.2] * 50
    lo, hi, p = paired_bootstrap(same, same)
    assert p == pytest.approx(1.0)
    better = [0.10] * 50
    worse = [0.20] * 50
    lo, hi, p = paired_bootstrap(better, worse)
    assert hi < 0 and p < 0.05


# ---------------- closedTime vs endDate ----------------


def test_closed_time_is_parsed_from_gammas_non_iso_format():
    from pmbench.gamma import _closed_time
    assert _closed_time({"closedTime": "2026-08-08 20:27:18+00"}) == "2026-08-08T20:27:18+00:00"
    assert _closed_time({}) is None


def test_effective_end_prefers_settlement_over_the_group_end_date():
    """A neg-risk market that settled in Aug 2026 carries endDate 2029-01-19.
    Using endDate would invent a two-and-a-half-year horizon that never was."""
    from pmbench.gamma import effective_end
    m = {"resolved": 1, "closed_time": "2026-08-08T20:27:18+00:00",
         "end_date": "2029-01-19T23:59:00Z"}
    assert effective_end(m) == "2026-08-08T20:27:18+00:00"
    assert effective_end({**m, "resolved": 0}) == "2029-01-19T23:59:00Z"
    assert effective_end({"resolved": 1, "end_date": "2029-01-19T23:59:00Z"}) \
        == "2029-01-19T23:59:00Z"


def test_singular_uma_status_is_trusted_over_the_proposal_array():
    r = decide(raw_market(closed=True, outcomePrices='["0","1"]',
                          umaResolutionStatus="resolved",
                          umaResolutionStatuses='["proposed"]'))
    assert r.outcome == 0 and "resolved" in r.source


def test_a_dispute_on_either_uma_field_holds_the_resolution():
    assert decide(raw_market(closed=True, outcomePrices='["1","0"]',
                             umaResolutionStatus="disputed")).outcome is None
    assert decide(raw_market(closed=True, outcomePrices='["1","0"]',
                             umaResolutionStatuses='["disputed"]')).outcome is None
