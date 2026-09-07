"""Orchestration: pick eligible markets, run the ensemble, persist rows."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime

from . import ingest, prompting
from .config import ModelConfig, iso, utcnow
from .db import observed_cutoffs, record_observation
from .eligibility import EligibilityPolicy, cutoff_note, evaluate, parse_ts
from .gamma import effective_end
from .forecast import (
    SampleFn, anthropic_sampler, forecast_market, new_forecast_id, new_run_id,
)
from .pricing import price_as_of

log = logging.getLogger(__name__)


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:  # noqa: BLE001
        return None


@dataclass
class Selection:
    markets: list[dict] = field(default_factory=list)
    rejected: Counter = field(default_factory=Counter)
    considered: int = 0

    def funnel(self) -> str:
        lines = [f"considered {self.considered} -> eligible {len(self.markets)}"]
        for reason, n in self.rejected.most_common():
            lines.append(f"  {reason:<28} {n}")
        return "\n".join(lines)


def select(
    conn,
    cfg: ModelConfig,
    *,
    policy: EligibilityPolicy | None = None,
    limit: int | None = None,
    now: datetime | None = None,
    exclude_forecast_by: str | None = None,
) -> Selection:
    """Apply the eligibility policy and report the full rejection funnel.

    The funnel is the point: a benchmark that says "500 forecasts" without
    saying what it threw away is not reproducible.
    """
    now = now or utcnow()
    policy = policy or EligibilityPolicy()
    cutoff = cfg.cutoff_date(observed_cutoffs(conn))
    sel = Selection()

    already: set[str] = set()
    if exclude_forecast_by:
        already = {
            r["market_id"] for r in conn.execute(
                "SELECT DISTINCT market_id FROM forecasts "
                "WHERE config_name = ? AND prompt_version = ? AND error IS NULL",
                (cfg.name, exclude_forecast_by),
            )
        }

    # Highest-volume first: the market-price baseline is strongest there, which
    # is the comparison we actually care about.
    # The latest snapshot per market, via a window function. The obvious
    # spelling — LEFT JOIN price_snapshots s ON s.id = (correlated subquery) —
    # is quadratic: SQLite re-runs the subquery per candidate row, so it is
    # instant on a fresh database and takes minutes once price history is
    # backfilled.
    rows = conn.execute(
        "WITH latest AS ("
        "  SELECT market_id, yes_price, volume, liquidity, spread, ts, "
        "         ROW_NUMBER() OVER (PARTITION BY market_id ORDER BY ts DESC, id DESC) rn "
        "  FROM price_snapshots) "
        "SELECT m.*, s.yes_price, s.volume, s.liquidity, s.spread, s.ts AS snap_ts "
        "FROM markets m LEFT JOIN latest s ON s.market_id = m.id AND s.rn = 1 "
        "ORDER BY COALESCE(s.volume, m.volume_latest, 0) DESC"
    ).fetchall()

    for row in rows:
        sel.considered += 1
        m = dict(row)
        if m["id"] in already:
            sel.rejected["already_forecast_this_version"] += 1
            continue
        if policy.mode == "retrospective":
            # Never the latest snapshot: for a resolved market that is the
            # settled 0/1, i.e. the answer. Price must come from history at or
            # before `now` (the as-of instant).
            price = price_as_of(conn, m["id"], now)
            # Volume/liquidity are the market's final figures, which include
            # trading after `as_of`. They gate SELECTION only and never enter a
            # score, but it is a mild lookahead and is stated here rather than
            # hidden: retrospective runs are biased toward markets that ended
            # up busy.
            snap = None if price is None else {
                "yes_price": price, "volume": m["volume_latest"],
                "liquidity": m["liquidity_latest"], "spread": None,
            }
            m["yes_price"] = price
        else:
            snap = None if m.get("yes_price") is None else {
                "yes_price": m["yes_price"], "volume": m["volume"],
                "liquidity": m["liquidity"], "spread": m["spread"],
            }
        verdict = evaluate(m, snap, cfg, cutoff, now=now, policy=policy)
        if verdict.eligible:
            sel.markets.append(m)
            if limit and len(sel.markets) >= limit:
                break
        else:
            for reason in verdict.reasons:
                sel.rejected[reason] += 1
    return sel


async def probe(conn, cfgs: list[ModelConfig], sample_fn: SampleFn | None = None) -> dict[str, str]:
    """Cheapest possible call per distinct model, to stamp `first_observed`.

    This is what makes cutoff_basis="first_observed" work without anyone having
    to look up a training-cutoff date: you cannot call a model before it exists,
    so today's date is a safe upper bound on its cutoff.
    """
    sample_fn = sample_fn or _default_sampler()
    today = utcnow().date().isoformat()
    seen: dict[str, str] = {}
    for model_id in sorted({c.model_id for c in cfgs}):
        cfg = next(c for c in cfgs if c.model_id == model_id)
        s = await sample_fn(cfg, "Return probability 0.5 with reasoning 'probe'.")
        if s.error:
            log.warning("probe %s failed: %s", model_id, s.error)
            continue
        record_observation(conn, model_id, today)
        seen[model_id] = today
    return {**observed_cutoffs(conn), **seen}


def _default_sampler() -> SampleFn:
    """Routes per config.provider; each provider's client is built on first use."""
    from .providers import build_router

    return build_router()


@dataclass
class RunStats:
    run_id: str
    written: int = 0
    failed: int = 0
    cost_usd: float = 0.0
    funnels: dict[str, str] = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"run {self.run_id}: forecasts={self.written} failed={self.failed} "
                f"cost=${self.cost_usd:.2f}")


async def run(
    conn,
    cfgs: list[ModelConfig],
    *,
    prompt_version: str,
    n_samples: int = 5,
    limit: int = 25,
    policy: EligibilityPolicy | None = None,
    paraphrase_versions: list[str] | None = None,
    concurrency: int = 4,
    sample_fn: SampleFn | None = None,
    notes: str = "",
    as_of: datetime | None = None,
) -> RunStats:
    policy = policy or EligibilityPolicy()
    sample_fn = sample_fn or _default_sampler()
    prompting.register(conn, prompt_version)
    for v in paraphrase_versions or []:
        prompting.register(conn, v)

    run_id = new_run_id()
    # `now` is the instant the model is told it is standing at, and every
    # horizon, price lookup and prompt date derives from it. In a retrospective
    # run it is the as-of date, not the wall clock.
    now = as_of or utcnow()
    conn.execute(
        "INSERT INTO runs(id, ts, git_sha, config_json, notes) VALUES (?,?,?,?,?)",
        (run_id, iso(now), git_sha(), json.dumps({
            "configs": [c.name for c in cfgs],
            "prompt_version": prompt_version,
            "paraphrase_versions": paraphrase_versions,
            "n_samples": n_samples,
            "limit": limit,
            "policy": policy.__dict__,
            "as_of": iso(now),
        }, default=str), notes),
    )
    conn.commit()

    st = RunStats(run_id)
    sem = asyncio.Semaphore(concurrency)

    for cfg in cfgs:
        cutoff = cfg.cutoff_date(observed_cutoffs(conn))
        sel = select(conn, cfg, policy=policy, limit=limit, now=now,
                     exclude_forecast_by=prompt_version)
        st.funnels[cfg.name] = sel.funnel()
        log.info("%s: %d eligible markets (cutoff %s)", cfg.name, len(sel.markets), cutoff)
        note = cutoff_note(cfg, cutoff, policy)

        async def one(market: dict, cfg=cfg, cutoff=cutoff, note=note) -> None:
            async with sem:
                fc = await forecast_market(
                    sample_fn, cfg, market,
                    prompt_version=prompt_version,
                    n_samples=n_samples,
                    paraphrase_versions=paraphrase_versions,
                    now=now,
                )
            end = parse_ts(effective_end(market))
            horizon = (end - now).total_seconds() / 86400.0 if end else None
            cost = cfg.cost_usd(fc.input_tokens, fc.output_tokens)
            conn.execute(
                "INSERT OR REPLACE INTO forecasts (id, run_id, market_id, ts, config_name, "
                "model_id, prompt_version, probability, reasoning, n_samples, samples_json, "
                "ensemble_sd, market_price, market_volume, horizon_days, cutoff_used, "
                "cutoff_basis, search_enabled, input_tokens, output_tokens, cost_usd, error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    new_forecast_id(), run_id, market["id"], iso(now), cfg.name,
                    cfg.model_id, prompt_version,
                    None if fc.error else fc.probability,
                    fc.reasoning, fc.n_samples,
                    json.dumps([s.__dict__ for s in fc.samples]),
                    None if fc.error else fc.ensemble_sd,
                    market.get("yes_price"), market.get("volume"), horizon,
                    note, cfg.cutoff_basis, int(cfg.web_search),
                    fc.input_tokens, fc.output_tokens, cost, fc.error,
                ),
            )
            st.cost_usd += cost
            if fc.error:
                st.failed += 1
            else:
                st.written += 1
                record_observation(conn, cfg.model_id, now.date().isoformat())

        await asyncio.gather(*(one(m) for m in sel.markets))
        conn.commit()

    conn.commit()
    return st
