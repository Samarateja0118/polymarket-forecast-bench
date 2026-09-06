"""Poll for resolutions and write scores.

Deciding that a market resolved is the one place where being sloppy corrupts
the labels themselves, so the rules are explicit and every decision carries its
evidence in `markets.outcome_source`.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field

from .config import iso, utcnow
from .gamma import GammaClient, _jsonish, _f
from .scoring import brier, log_loss

log = logging.getLogger(__name__)

SETTLED_PRICE = 0.99  # a resolved binary market prices at 0 or 1, not 0.97


@dataclass
class Resolution:
    outcome: int | None
    source: str


def decide(raw: dict) -> Resolution:
    """Read a resolution off a raw Gamma market, or decline to.

    Rules, in order:
      - a disputed UMA resolution is not a resolution; wait for it to settle
      - the market must be closed
      - YES/NO prices must have collapsed to the extremes (>=0.99 / <=0.01);
        a closed market still trading at 0.6 has not actually paid out
    Anything else returns None and gets retried on the next poll.
    """
    # Two different fields, and they disagree. `umaResolutionStatus` (singular)
    # is the market's settlement state — "resolved" once UMA has finalised it.
    # `umaResolutionStatuses` (plural) is a per-proposal array that can still
    # read ["proposed"] on a fully settled market. Trust the singular; use the
    # plural only to spot disputes.
    status = str(raw.get("umaResolutionStatus") or "").lower()
    statuses = [str(s).lower() for s in (_jsonish(raw.get("umaResolutionStatuses")) or [])]
    if "disput" in status or any("disput" in s for s in statuses):
        return Resolution(None, f"held:disputed:{status or statuses}")
    if not raw.get("closed"):
        return Resolution(None, "open")

    prices = _jsonish(raw.get("outcomePrices")) or []
    if len(prices) < 2:
        return Resolution(None, "no_outcome_prices")
    yes, no = _f(prices[0]), _f(prices[1])
    if yes is None or no is None:
        return Resolution(None, "unparseable_outcome_prices")

    evidence = f"closed,prices={yes:.4f}/{no:.4f},uma={status or statuses or 'none'}"
    if yes >= SETTLED_PRICE and no <= 1 - SETTLED_PRICE:
        return Resolution(1, f"gamma:{evidence}")
    if no >= SETTLED_PRICE and yes <= 1 - SETTLED_PRICE:
        return Resolution(0, f"gamma:{evidence}")
    return Resolution(None, f"closed_but_unsettled:{evidence}")


@dataclass
class ResolveStats:
    checked: int = 0
    newly_resolved: int = 0
    still_open: int = 0
    held: int = 0
    flipped: list[str] = field(default_factory=list)
    scored: int = 0

    def __str__(self) -> str:
        s = (f"checked={self.checked} resolved={self.newly_resolved} "
             f"open={self.still_open} held={self.held} scored={self.scored}")
        if self.flipped:
            s += f"  !! OUTCOME FLIPS: {self.flipped}"
        return s


def pending_market_ids(conn: sqlite3.Connection, *, recheck: bool = False) -> list[str]:
    """Markets we have forecast but not yet resolved (or all, with --recheck)."""
    sql = (
        "SELECT DISTINCT m.id FROM markets m JOIN forecasts f ON f.market_id = m.id "
        "WHERE f.error IS NULL"
    )
    if not recheck:
        sql += " AND m.resolved = 0"
    return [r["id"] for r in conn.execute(sql)]


async def poll(conn: sqlite3.Connection, *, recheck: bool = False) -> ResolveStats:
    st = ResolveStats()
    ids = pending_market_ids(conn, recheck=recheck)
    if not ids:
        return st
    async with GammaClient() as client:
        raws = await client.markets(ids)

    now_iso = iso(utcnow())
    for raw in raws:
        st.checked += 1
        mid = str(raw.get("id"))
        res = decide(raw)
        prev = conn.execute(
            "SELECT resolved, outcome FROM markets WHERE id = ?", (mid,)
        ).fetchone()

        if res.outcome is None:
            if res.source.startswith("held"):
                st.held += 1
            else:
                st.still_open += 1
            conn.execute(
                "UPDATE markets SET closed = ?, active = ?, last_seen_ts = ? WHERE id = ?",
                (int(bool(raw.get("closed"))), int(bool(raw.get("active"))), now_iso, mid),
            )
            continue

        if prev and prev["resolved"] and prev["outcome"] != res.outcome:
            # Rare, but it happens (UMA disputes overturn an early call). Loud,
            # because it means previously written scores are wrong.
            st.flipped.append(mid)
            log.error("market %s outcome flipped %s -> %s", mid, prev["outcome"], res.outcome)
        elif not (prev and prev["resolved"]):
            st.newly_resolved += 1

        from .gamma import _closed_time
        settled = _closed_time(raw)
        conn.execute(
            "UPDATE markets SET resolved = 1, outcome = ?, outcome_source = ?, "
            "closed_time = COALESCE(?, closed_time), "
            "resolved_ts = COALESCE(resolved_ts, ?, ?), closed = 1, last_seen_ts = ? "
            "WHERE id = ?",
            (res.outcome, res.source, settled, settled, now_iso, now_iso, mid),
        )
    conn.commit()
    st.scored = score_pending(conn)
    return st


def score_pending(conn: sqlite3.Connection) -> int:
    """Score every forecast whose market has resolved and that isn't scored yet."""
    rows = conn.execute(
        "SELECT f.id, f.probability, f.market_price, f.horizon_days, m.outcome "
        "FROM forecasts f JOIN markets m ON m.id = f.market_id "
        "LEFT JOIN scores s ON s.forecast_id = f.id "
        "WHERE m.resolved = 1 AND m.outcome IS NOT NULL "
        "  AND f.error IS NULL AND f.probability IS NOT NULL AND s.forecast_id IS NULL"
    ).fetchall()
    now_iso = iso(utcnow())
    n = 0
    for r in rows:
        y = int(r["outcome"])
        p = float(r["probability"])
        mp = r["market_price"]
        conn.execute(
            "INSERT INTO scores(forecast_id, outcome, brier, log_loss, brier_market, "
            "log_loss_market, horizon_days, scored_ts) VALUES (?,?,?,?,?,?,?,?)",
            (r["id"], y, brier(p, y), log_loss(p, y),
             brier(float(mp), y) if mp is not None else None,
             log_loss(float(mp), y) if mp is not None else None,
             r["horizon_days"], now_iso),
        )
        n += 1
    conn.commit()
    return n


def rescore_all(conn: sqlite3.Connection) -> int:
    """Drop and rebuild the scores table (use after an outcome flip)."""
    conn.execute("DELETE FROM scores")
    conn.commit()
    return score_pending(conn)
