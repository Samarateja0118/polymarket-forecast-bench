"""The as-of price lookup is the anti-lookahead guarantee for retrospective runs."""
from datetime import datetime, timedelta, timezone

import pytest

from pmbench.db import connect
from pmbench.pricing import price_as_of

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)


@pytest.fixture
def conn(tmp_path):
    c = connect(tmp_path / "t.db")
    c.execute("INSERT INTO markets(id, question) VALUES ('1','Q')")
    for hours, price in [(0, 0.30), (24, 0.45), (48, 1.00)]:
        c.execute(
            "INSERT INTO price_snapshots(market_id, ts, yes_price, content_hash) "
            "VALUES ('1', ?, ?, ?)",
            ((T0 + timedelta(hours=hours)).isoformat().replace("+00:00", "Z"),
             price, f"h{hours}"),
        )
    c.commit()
    yield c
    c.close()


def test_returns_the_last_price_at_or_before_as_of(conn):
    assert price_as_of(conn, "1", T0 + timedelta(hours=30)) == pytest.approx(0.45)
    assert price_as_of(conn, "1", T0) == pytest.approx(0.30)


def test_never_returns_a_future_price(conn):
    """For a resolved market the latest price IS the answer. Reading it would
    make any backtest look superhuman and be entirely fake."""
    got = price_as_of(conn, "1", T0 + timedelta(hours=47),
                      max_staleness=timedelta(days=2))
    assert got == pytest.approx(0.45)     # the hour-24 price
    assert got != 1.00                    # NOT the hour-48 settlement


def test_no_price_before_as_of_returns_none(conn):
    assert price_as_of(conn, "1", T0 - timedelta(days=1)) is None


def test_stale_prices_are_refused(conn):
    assert price_as_of(conn, "1", T0 + timedelta(days=30)) is None
    assert price_as_of(conn, "1", T0 + timedelta(days=30),
                       max_staleness=timedelta(days=60)) == pytest.approx(1.00)
