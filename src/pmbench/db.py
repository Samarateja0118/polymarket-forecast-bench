"""SQLite storage. One file, WAL mode, idempotent schema."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import db_path

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS markets (
    id                TEXT PRIMARY KEY,      -- Gamma market id
    condition_id      TEXT,
    slug              TEXT,
    question          TEXT NOT NULL,
    resolution_criteria TEXT,                -- Gamma `description`
    resolution_source TEXT,
    event_id          TEXT,
    event_title       TEXT,
    category          TEXT,                  -- primary tag label
    tags_json         TEXT,
    start_date        TEXT,                  -- ISO8601 UTC
    end_date          TEXT,                  -- ISO8601 UTC
    created_at        TEXT,
    clob_token_yes    TEXT,
    neg_risk          INTEGER DEFAULT 0,
    active            INTEGER DEFAULT 0,
    closed            INTEGER DEFAULT 0,
    archived          INTEGER DEFAULT 0,
    resolved          INTEGER DEFAULT 0,     -- 1 once we have a confident outcome
    outcome           INTEGER,               -- 1 = YES, 0 = NO, NULL = unknown
    outcome_source    TEXT,                  -- how we decided (audit trail)
    resolved_ts       TEXT,
    closed_time       TEXT,                  -- Gamma `closedTime`: when the
                                             -- market ACTUALLY settled. See
                                             -- gamma.effective_end().
    volume_latest     REAL,                  -- latest known totals; for
    liquidity_latest  REAL,                  -- selection only, never scoring
    first_seen_ts     TEXT,
    last_seen_ts      TEXT,
    content_hash      TEXT                   -- hash of the descriptive fields
);
CREATE INDEX IF NOT EXISTS ix_markets_end   ON markets(end_date);
CREATE INDEX IF NOT EXISTS ix_markets_res   ON markets(resolved, end_date);
CREATE INDEX IF NOT EXISTS ix_markets_cat   ON markets(category);

CREATE TABLE IF NOT EXISTS price_snapshots (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    market_id     TEXT NOT NULL REFERENCES markets(id),
    ts            TEXT NOT NULL,
    yes_price     REAL,
    best_bid      REAL,
    best_ask      REAL,
    spread        REAL,
    volume        REAL,
    liquidity     REAL,
    content_hash  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_snap_market ON price_snapshots(market_id, ts);

-- One `runs` row per invocation of `pmbench forecast`, so a shifted result can
-- always be traced to the code + config that produced it.
CREATE TABLE IF NOT EXISTS runs (
    id           TEXT PRIMARY KEY,
    ts           TEXT NOT NULL,
    git_sha      TEXT,
    config_json  TEXT,
    notes        TEXT
);

CREATE TABLE IF NOT EXISTS forecasts (
    id             TEXT PRIMARY KEY,
    run_id         TEXT REFERENCES runs(id),
    market_id      TEXT NOT NULL REFERENCES markets(id),
    ts             TEXT NOT NULL,
    config_name    TEXT NOT NULL,          -- ModelConfig.name
    model_id       TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    probability    REAL,                   -- trimmed-mean ensemble estimate
    reasoning      TEXT,                   -- reasoning from the median sample
    n_samples      INTEGER,
    samples_json   TEXT,                   -- every raw sample, for re-analysis
    ensemble_sd    REAL,                   -- uncertainty signal (NOT self-reported)
    -- Anti-lookahead: the market price is captured at forecast time and frozen
    -- here. Baseline 3 reads this column, never a later snapshot.
    market_price   REAL,
    market_volume  REAL,
    horizon_days   REAL,
    cutoff_used    TEXT,                   -- the contamination bound in force
    cutoff_basis   TEXT,
    search_enabled INTEGER DEFAULT 0,
    input_tokens   INTEGER DEFAULT 0,
    output_tokens  INTEGER DEFAULT 0,
    cost_usd       REAL DEFAULT 0,
    error          TEXT
);
CREATE INDEX IF NOT EXISTS ix_fc_market ON forecasts(market_id);
CREATE INDEX IF NOT EXISTS ix_fc_cfg    ON forecasts(config_name, ts);
-- At most one forecast per (market, config, prompt version, run).
CREATE UNIQUE INDEX IF NOT EXISTS ux_fc_dedup
    ON forecasts(market_id, config_name, prompt_version, run_id);

CREATE TABLE IF NOT EXISTS scores (
    forecast_id   TEXT PRIMARY KEY REFERENCES forecasts(id),
    outcome       INTEGER NOT NULL,
    brier         REAL NOT NULL,
    log_loss      REAL NOT NULL,
    brier_market  REAL,                    -- baseline 3, same outcome
    log_loss_market REAL,
    horizon_days  REAL,
    scored_ts     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompt_versions (
    version    TEXT PRIMARY KEY,
    body_hash  TEXT NOT NULL,
    body       TEXT NOT NULL,
    added_ts   TEXT NOT NULL
);

-- Underpins cutoff_basis="first_observed": the earliest date on which this
-- harness successfully called a model is a safe upper bound on its cutoff.
CREATE TABLE IF NOT EXISTS model_observations (
    model_id        TEXT PRIMARY KEY,
    first_called_on TEXT NOT NULL          -- ISO date, UTC
);
"""


# Additive migrations for databases created by an earlier version.
MIGRATIONS = [
    ("markets", "closed_time", "TEXT"),
    ("markets", "volume_latest", "REAL"),
    ("markets", "liquidity_latest", "REAL"),
]


def connect(path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(path or db_path(), timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    for table, column, decl in MIGRATIONS:
        cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
    conn.commit()
    return conn


def observed_cutoffs(conn: sqlite3.Connection) -> dict[str, str]:
    return {
        r["model_id"]: r["first_called_on"]
        for r in conn.execute("SELECT model_id, first_called_on FROM model_observations")
    }


def record_observation(conn: sqlite3.Connection, model_id: str, on_date: str) -> None:
    """Record first successful call. Never moves the date forward."""
    conn.execute(
        "INSERT INTO model_observations(model_id, first_called_on) VALUES (?, ?) "
        "ON CONFLICT(model_id) DO UPDATE SET first_called_on = "
        "  MIN(model_observations.first_called_on, excluded.first_called_on)",
        (model_id, on_date),
    )
    conn.commit()
