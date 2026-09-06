"""Prompts are versioned data, not string literals.

When a result shifts you need to know whether the model changed or you did, so
every prompt body is hashed and written to the `prompt_versions` table the first
time it is used. Editing a template without renaming it raises.
"""
from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from pathlib import Path

from .config import iso, utcnow
from .eligibility import parse_ts

PROMPT_DIR = Path(__file__).parent / "prompts"

SYSTEM = (
    "You are a careful superforecaster. You are calibrated: when you say 70%, "
    "the thing happens about 70% of the time. You are willing to give extreme "
    "probabilities when the evidence supports them and unwilling to give them "
    "when it does not."
)


def available() -> list[str]:
    return sorted(p.stem for p in PROMPT_DIR.glob("*.txt"))


def load(version: str) -> str:
    path = PROMPT_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"unknown prompt version {version!r}; have {available()}")
    return path.read_text()


def body_hash(body: str) -> str:
    return hashlib.sha256(body.encode()).hexdigest()[:32]


def register(conn: sqlite3.Connection, version: str) -> str:
    """Pin the prompt body. Raises if a version's text changed under us."""
    body = load(version)
    h = body_hash(body)
    row = conn.execute(
        "SELECT body_hash FROM prompt_versions WHERE version = ?", (version,)
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO prompt_versions(version, body_hash, body, added_ts) VALUES (?,?,?,?)",
            (version, h, body, iso(utcnow())),
        )
        conn.commit()
    elif row["body_hash"] != h:
        raise ValueError(
            f"prompt {version!r} has changed since it was first used "
            f"({row['body_hash']} -> {h}). Forecasts already recorded under this "
            f"version were made with different text. Add a new version file "
            f"instead of editing this one."
        )
    return h


def render(version: str, market: dict, *, now: datetime | None = None) -> str:
    now = now or utcnow()
    end = parse_ts(market.get("end_date"))
    horizon = (end - now).total_seconds() / 86400.0 if end else float("nan")
    criteria = (market.get("resolution_criteria") or "").strip() or "(none published)"
    return load(version).format(
        today=now.strftime("%Y-%m-%d"),
        question=market.get("question", ""),
        resolution_criteria=criteria,
        end_date=(end.strftime("%Y-%m-%d") if end else "unknown"),
        horizon_days=horizon,
    )
