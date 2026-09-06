"""As-of pricing from CLOB history — the market-price baseline for
retrospective runs.

In a live run, "the market price at forecast time" is just the snapshot we took
that minute. In a retrospective run we are pretending to stand at some past
`as_of`, and reading today's price would be lookahead of the most damaging kind:
for a resolved market today's price IS the answer (0 or 1), so a backtest built
on it would show superhuman skill and be entirely fake.

So retrospective prices come from CLOB `prices-history`, and this module refuses
to return a price stamped after `as_of`.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta, timezone

from .config import iso
from .gamma import GammaClient

log = logging.getLogger(__name__)

MAX_STALENESS = timedelta(hours=12)


def _parse(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


async def backfill_history(
    conn: sqlite3.Connection, market_ids: list[str], *, fidelity: int = 60
) -> int:
    """Pull full price history for these markets into `price_snapshots`.

    The snapshots table is the cache: history rows carry their real timestamps,
    so as_of lookups are a plain SQL query afterwards.
    """
    rows = conn.execute(
        f"SELECT id, clob_token_yes, start_date, closed_time, end_date FROM markets "
        f"WHERE id IN ({','.join('?' * len(market_ids))}) AND clob_token_yes IS NOT NULL",
        market_ids,
    ).fetchall()
    written = 0
    now = datetime.now(timezone.utc)
    async with GammaClient() as client:
        for r in rows:
            start = _parse(r["start_date"]) or (now - timedelta(days=365))
            end = _parse(r["closed_time"]) or min(_parse(r["end_date"]) or now, now)
            if end <= start:
                continue
            try:
                hist = await client.price_history_window(
                    r["clob_token_yes"], start, end, fidelity=fidelity
                )
            except Exception as exc:  # noqa: BLE001
                log.warning("history %s failed: %s", r["id"], exc)
                continue
            have = {
                row["ts"] for row in conn.execute(
                    "SELECT ts FROM price_snapshots WHERE market_id = ?", (r["id"],)
                )
            }
            for point in hist:
                ts = iso(datetime.fromtimestamp(point["t"], tz=timezone.utc))
                if ts in have:
                    continue
                conn.execute(
                    "INSERT INTO price_snapshots(market_id, ts, yes_price, content_hash) "
                    "VALUES (?,?,?,?)",
                    (r["id"], ts, float(point["p"]), f"clob:{point['t']}"),
                )
                written += 1
            conn.commit()
    return written


def price_as_of(
    conn: sqlite3.Connection, market_id: str, as_of: datetime,
    *, max_staleness: timedelta = MAX_STALENESS,
) -> float | None:
    """Last observed YES price at or before `as_of`, or None if too stale.

    Strictly `<= as_of`. This one comparison is the anti-lookahead guarantee for
    every retrospective number in the repo.
    """
    row = conn.execute(
        "SELECT ts, yes_price FROM price_snapshots "
        "WHERE market_id = ? AND ts <= ? AND yes_price IS NOT NULL "
        "ORDER BY ts DESC LIMIT 1",
        (market_id, iso(as_of)),
    ).fetchone()
    if row is None:
        return None
    seen = datetime.fromisoformat(row["ts"].replace("Z", "+00:00"))
    if as_of - seen > max_staleness:
        return None
    return float(row["yes_price"])
