"""Collector: poll Gamma, upsert markets, append price snapshots.

Snapshots are content-hashed and only written when something actually moved, so
polling hourly for a year doesn't produce 8,760 identical rows per market.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass

from .config import iso, utcnow
from .gamma import GammaClient, normalise, snapshot_of

log = logging.getLogger(__name__)

DESCRIPTIVE_FIELDS = (
    "question", "resolution_criteria", "resolution_source",
    "end_date", "start_date", "category", "clob_token_yes", "closed_time",
)
SNAPSHOT_FIELDS = ("yes_price", "best_bid", "best_ask", "spread", "volume", "liquidity")


def _hash(payload: dict, fields: tuple[str, ...], *, round_to: int | None = None) -> str:
    parts = []
    for f in fields:
        v = payload.get(f)
        if round_to is not None and isinstance(v, float):
            v = round(v, round_to)
        parts.append(f"{f}={v!r}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


@dataclass
class IngestStats:
    events: int = 0
    markets_seen: int = 0
    markets_new: int = 0
    markets_changed: int = 0
    snapshots_written: int = 0
    snapshots_deduped: int = 0
    skipped_non_binary: int = 0

    def __str__(self) -> str:
        return (
            f"events={self.events} markets={self.markets_seen} "
            f"(new={self.markets_new} changed={self.markets_changed} "
            f"non_binary_skipped={self.skipped_non_binary}) "
            f"snapshots={self.snapshots_written} deduped={self.snapshots_deduped}"
        )


def upsert_market(conn: sqlite3.Connection, row: dict, now_iso: str, st: IngestStats) -> None:
    row = dict(row)
    row["content_hash"] = _hash(row, DESCRIPTIVE_FIELDS)
    prev = conn.execute(
        "SELECT content_hash FROM markets WHERE id = ?", (row["id"],)
    ).fetchone()

    if prev is None:
        st.markets_new += 1
        cols = list(row) + ["first_seen_ts", "last_seen_ts"]
        vals = [row[c] for c in row] + [now_iso, now_iso]
        conn.execute(
            f"INSERT INTO markets ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            vals,
        )
        return

    if prev["content_hash"] != row["content_hash"]:
        st.markets_changed += 1
    # `resolved`, `outcome` and friends are owned by resolve.py; never clobbered here.
    sets = ",".join(f"{c} = ?" for c in row if c != "id")
    conn.execute(
        f"UPDATE markets SET {sets}, last_seen_ts = ? WHERE id = ?",
        [row[c] for c in row if c != "id"] + [now_iso, row["id"]],
    )


def append_snapshot(
    conn: sqlite3.Connection, market_id: str, snap: dict, now_iso: str, st: IngestStats
) -> None:
    if snap.get("yes_price") is None:
        return
    # Round before hashing: sub-cent noise in `volume` is not a price change.
    h = _hash(snap, SNAPSHOT_FIELDS, round_to=4)
    last = conn.execute(
        "SELECT content_hash FROM price_snapshots WHERE market_id = ? "
        "ORDER BY ts DESC, id DESC LIMIT 1",
        (market_id,),
    ).fetchone()
    if last is not None and last["content_hash"] == h:
        st.snapshots_deduped += 1
        return
    conn.execute(
        "INSERT INTO price_snapshots "
        "(market_id, ts, yes_price, best_bid, best_ask, spread, volume, liquidity, content_hash) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (market_id, now_iso, snap["yes_price"], snap["best_bid"], snap["best_ask"],
         snap["spread"], snap["volume"], snap["liquidity"], h),
    )
    st.snapshots_written += 1


async def collect_open(
    conn: sqlite3.Connection,
    *,
    max_pages: int = 20,
    page_size: int = 100,
    end_date_min: str | None = None,
) -> IngestStats:
    """Snapshot every open binary market Gamma will show us, highest volume first."""
    st = IngestStats()
    now_iso = iso(utcnow())
    async with GammaClient() as client:
        async for event in client.iter_events(
            closed=False, max_pages=max_pages, page_size=page_size, end_date_min=end_date_min
        ):
            st.events += 1
            for raw in event.get("markets") or []:
                row = normalise(event, raw)
                if row is None:
                    st.skipped_non_binary += 1
                    continue
                st.markets_seen += 1
                upsert_market(conn, row, now_iso, st)
                append_snapshot(conn, row["id"], snapshot_of(raw), now_iso, st)
            conn.commit()
    conn.commit()
    log.info("ingest: %s", st)
    return st


async def refresh(conn: sqlite3.Connection, market_ids: list[str]) -> IngestStats:
    """Re-poll specific markets (used before forecasting, and by resolve)."""
    st = IngestStats()
    now_iso = iso(utcnow())
    async with GammaClient() as client:
        raws = await client.markets(market_ids)
    for raw in raws:
        event = (raw.get("events") or [{}])[0]
        # /markets/{id} nests a thin event without tags; keep the stored category.
        row = normalise(event, raw)
        if row is None:
            st.skipped_non_binary += 1
            continue
        existing = conn.execute(
            "SELECT category, tags_json, event_title FROM markets WHERE id = ?", (row["id"],)
        ).fetchone()
        if existing and row["category"] == "Other":
            row["category"] = existing["category"]
            row["tags_json"] = existing["tags_json"]
            row["event_title"] = existing["event_title"] or row["event_title"]
        st.markets_seen += 1
        upsert_market(conn, row, now_iso, st)
        append_snapshot(conn, row["id"], snapshot_of(raw), now_iso, st)
    conn.commit()
    return st


def latest_snapshot(conn: sqlite3.Connection, market_id: str) -> dict | None:
    r = conn.execute(
        "SELECT * FROM price_snapshots WHERE market_id = ? ORDER BY ts DESC, id DESC LIMIT 1",
        (market_id,),
    ).fetchone()
    return dict(r) if r else None


async def collect_closed(
    conn: sqlite3.Connection,
    *,
    end_date_min: str,
    end_date_max: str | None = None,
    max_pages: int = 20,
    page_size: int = 100,
) -> IngestStats:
    """Backfill markets that have already resolved, for retrospective runs.

    `end_date_min` is not optional and should be set to the model's cutoff: a
    market that resolved before the cutoff is a memory test, and there is no
    reason to spend bandwidth storing it.
    """
    from .resolve import decide  # local import: resolve imports scoring, not ingest

    st = IngestStats()
    now_iso = iso(utcnow())
    async with GammaClient() as client:
        async for event in client.iter_events(
            closed=True, max_pages=max_pages, page_size=page_size,
            end_date_min=end_date_min, end_date_max=end_date_max,
            order="endDate", ascending=False,
        ):
            st.events += 1
            for raw in event.get("markets") or []:
                row = normalise(event, raw)
                if row is None:
                    st.skipped_non_binary += 1
                    continue
                st.markets_seen += 1
                upsert_market(conn, row, now_iso, st)
                res = decide(raw)
                if res.outcome is not None:
                    # closed_time, not end_date — see gamma.effective_end().
                    conn.execute(
                        "UPDATE markets SET resolved = 1, outcome = ?, outcome_source = ?, "
                        "resolved_ts = COALESCE(resolved_ts, ?, ?) WHERE id = ?",
                        (res.outcome, res.source, row.get("closed_time"),
                         row.get("end_date") or now_iso, row["id"]),
                    )
            conn.commit()
    conn.commit()
    log.info("backfill: %s", st)
    return st
