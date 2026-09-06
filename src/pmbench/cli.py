"""Command line entry point.

    pmbench ingest              poll Gamma, snapshot open markets
    pmbench probe               stamp first-observed dates for cutoff control
    pmbench eligible            show the selection funnel (no API calls)
    pmbench forecast            run the ensemble and store forecasts
    pmbench resolve             poll for resolutions and score
    pmbench report              tables + figures
    pmbench status              what is in the database
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from datetime import datetime, timezone

from . import analysis, ingest, pricing, prompting, report, resolve, runner
from .baselines import all_comparisons, format_comparisons
from .config import REPORT_DIR, load_models, utcnow
from .db import connect, observed_cutoffs
from .eligibility import EligibilityPolicy
from .gamma import effective_end


def _as_of(args) -> datetime | None:
    raw = getattr(args, "as_of", None)
    if not raw:
        return None
    dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _policy(args) -> EligibilityPolicy:
    return EligibilityPolicy(
        mode=getattr(args, "mode", "live"),
        strict_contamination=args.strict_contamination,
        min_volume_usd=args.min_volume,
        max_spread=args.max_spread,
        min_horizon_days=args.min_horizon,
        max_horizon_days=args.max_horizon,
    )


def _configs(args, models):
    if not args.configs:
        return list(models.values())
    missing = [c for c in args.configs if c not in models]
    if missing:
        sys.exit(f"unknown config(s): {missing}; have {sorted(models)}")
    return [models[c] for c in args.configs]


# --------------------------------------------------------------------------


def cmd_ingest(args) -> None:
    conn = connect()
    if args.closed_since:
        st = asyncio.run(ingest.collect_closed(
            conn, end_date_min=args.closed_since, end_date_max=args.closed_until,
            max_pages=args.pages, page_size=args.page_size))
    else:
        st = asyncio.run(ingest.collect_open(
            conn, max_pages=args.pages, page_size=args.page_size))
    print(st)


def cmd_history(args) -> None:
    """Backfill CLOB price history so retrospective runs have as-of prices."""
    conn = connect()
    # Highest-volume resolved markets first: those are the ones where the
    # market-price baseline is worth beating, and history calls are one request
    # per market so the budget goes where it counts.
    ids = [r["id"] for r in conn.execute(
        "SELECT m.id FROM markets m WHERE m.resolved = 1 "
        "  AND m.clob_token_yes IS NOT NULL AND m.closed_time IS NOT NULL "
        "  AND LENGTH(COALESCE(m.resolution_criteria,'')) >= 80 "
        "ORDER BY COALESCE(m.volume_latest, 0) DESC, m.closed_time DESC LIMIT ?",
        (args.limit,))]
    if not ids:
        print("no resolved markets with a CLOB token — run `pmbench ingest --closed-since` first")
        return
    n = asyncio.run(pricing.backfill_history(conn, ids, fidelity=args.fidelity))
    print(f"{len(ids)} markets, {n} new history points")


def cmd_probe(args) -> None:
    conn = connect()
    models = load_models()
    seen = asyncio.run(runner.probe(conn, _configs(args, models)))
    for model_id, day in sorted(seen.items()):
        print(f"{model_id:<24} first observed {day}")
    if not seen:
        print("no models probed — check credentials (ANTHROPIC_API_KEY or `ant auth login`)")


def cmd_eligible(args) -> None:
    conn = connect()
    models = load_models()
    for cfg in _configs(args, models):
        try:
            cutoff = cfg.cutoff_date(observed_cutoffs(conn))
        except ValueError as exc:
            print(f"\n=== {cfg.name} ===\n  {exc}")
            continue
        sel = runner.select(conn, cfg, policy=_policy(args), limit=args.limit,
                            now=_as_of(args))
        print(f"\n=== {cfg.name}  (cutoff {cutoff}, basis {cfg.cutoff_basis}) ===")
        print(sel.funnel())
        if args.show:
            for m in sel.markets[: args.show]:
                vol = m["volume"] if m["volume"] is not None else m["volume_latest"]
                print(f"  [{m['category']:<12}] {m['question'][:76]}")
                print(f"      price={m['yes_price']:.3f} "
                      f"vol=${vol or 0:,.0f} ends {effective_end(m)}")


def cmd_forecast(args) -> None:
    conn = connect()
    models = load_models()
    cfgs = _configs(args, models)
    st = asyncio.run(runner.run(
        conn, cfgs,
        prompt_version=args.prompt_version,
        n_samples=args.n_samples,
        limit=args.limit,
        policy=_policy(args),
        paraphrase_versions=args.paraphrase or None,
        concurrency=args.concurrency,
        notes=args.notes,
        as_of=_as_of(args),
    ))
    for name, funnel in st.funnels.items():
        print(f"\n=== {name} ===\n{funnel}")
    print(f"\n{st}")


def cmd_resolve(args) -> None:
    conn = connect()
    if args.rescore:
        print(f"rescored {resolve.rescore_all(conn)} forecasts")
        return
    st = asyncio.run(resolve.poll(conn, recheck=args.recheck))
    print(st)


def cmd_report(args) -> None:
    conn = connect()
    rows = analysis.load_rows(conn)
    if not rows:
        print("no scored forecasts yet — run `pmbench resolve` once markets close")
        return

    print(f"scored forecasts: {len(rows)}  markets: {len({r.market_id for r in rows})}\n")
    for name, comps in analysis.leaderboard(rows).items():
        print(f"=== {name} ===")
        print(format_comparisons(comps))
        cfg_rows = [r for r in rows if r.config_name == name]
        print("\n" + str(analysis.calibration_fit(cfg_rows)))
        print("\n" + str(analysis.abstention_curve(cfg_rows)) + "\n")

    if args.anchoring:
        a, b = args.anchoring
        print(f"=== anchoring: {a} vs {b} ===")
        print(analysis.anchoring(rows, a, b))

    if not args.no_figures:
        out = Path(args.out or REPORT_DIR)
        # Belt and braces: a figure must never leave here unmarked if any of the
        # outcomes behind it were synthesised. Detected from the database rather
        # than trusted from a flag, so it cannot be forgotten.
        synthetic = conn.execute(
            "SELECT COUNT(*) FROM markets m JOIN forecasts f ON f.market_id = m.id "
            "JOIN scores s ON s.forecast_id = f.id WHERE m.outcome_source = 'SYNTHETIC'"
        ).fetchone()[0]
        mark = "SYNTHETIC" if synthetic else None
        if mark:
            print(f"\n!! {synthetic} scored forecasts have SYNTHETIC outcomes — "
                  f"figures will be watermarked and are not a result.\n")
        for p in report.build_all(rows, out, watermark=mark):
            print(f"wrote {p}")


def cmd_status(args) -> None:
    conn = connect()
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    print(f"markets            {q('SELECT COUNT(*) FROM markets')}")
    print(f"  resolved         {q('SELECT COUNT(*) FROM markets WHERE resolved=1')}")
    print(f"price snapshots    {q('SELECT COUNT(*) FROM price_snapshots')}")
    print(f"runs               {q('SELECT COUNT(*) FROM runs')}")
    print(f"forecasts          {q('SELECT COUNT(*) FROM forecasts')}")
    print(f"  errored          {q('SELECT COUNT(*) FROM forecasts WHERE error IS NOT NULL')}")
    print(f"scored             {q('SELECT COUNT(*) FROM scores')}")
    cost = conn.execute("SELECT COALESCE(SUM(cost_usd),0) FROM forecasts").fetchone()[0]
    print(f"spend              ${cost:,.2f}")
    print(f"prompt versions    {sorted(prompting.available())}")
    obs = observed_cutoffs(conn)
    print(f"model observations {obs or '(none — run `pmbench probe`)'}")


# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pmbench", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_policy(sp):
        sp.add_argument("--mode", choices=["live", "retrospective"], default="live",
                        help="live: forecast open markets and wait. retrospective: "
                             "forecast already-resolved markets from a past --as-of, "
                             "which needs a documented cutoff to be safe.")
        sp.add_argument("--as-of", default=None, metavar="ISO8601",
                        help="pretend it is this instant (retrospective mode)")
        sp.add_argument("--strict-contamination", action="store_true",
                        help="also require the market to have STARTED after the cutoff")
        sp.add_argument("--min-volume", type=float, default=5000.0)
        sp.add_argument("--max-spread", type=float, default=0.10)
        sp.add_argument("--min-horizon", type=float, default=2.0)
        sp.add_argument("--max-horizon", type=float, default=365.0)

    s = sub.add_parser("ingest", help="poll Gamma and snapshot markets")
    s.add_argument("--pages", type=int, default=20)
    s.add_argument("--page-size", type=int, default=100)
    s.add_argument("--closed-since", default=None, metavar="ISO8601",
                   help="backfill RESOLVED markets ending after this date "
                        "(set it to your model's cutoff)")
    s.add_argument("--closed-until", default=None, metavar="ISO8601")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("history", help="backfill CLOB price history for as-of pricing")
    s.add_argument("--limit", type=int, default=200)
    s.add_argument("--fidelity", type=int, default=60, help="minutes per point")
    s.set_defaults(func=cmd_history)

    s = sub.add_parser("probe", help="record first-observed dates (cutoff control)")
    s.add_argument("--configs", nargs="*", default=[])
    s.set_defaults(func=cmd_probe)

    s = sub.add_parser("eligible", help="show the selection funnel; makes no API calls")
    s.add_argument("--configs", nargs="*", default=[])
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--show", type=int, default=0, help="print the first N eligible markets")
    add_policy(s)
    s.set_defaults(func=cmd_eligible)

    s = sub.add_parser("forecast", help="run the ensemble (spends money)")
    s.add_argument("--configs", nargs="*", default=[])
    s.add_argument("--prompt-version", default="v2_reference_class")
    s.add_argument("--paraphrase", nargs="*", default=[],
                   help="cycle these prompt versions across samples")
    s.add_argument("--n-samples", type=int, default=5)
    s.add_argument("--limit", type=int, default=25, help="markets per config")
    s.add_argument("--concurrency", type=int, default=4)
    s.add_argument("--notes", default="")
    add_policy(s)
    s.set_defaults(func=cmd_forecast)

    s = sub.add_parser("resolve", help="poll for resolutions and score")
    s.add_argument("--recheck", action="store_true",
                   help="re-poll already-resolved markets and warn on flips")
    s.add_argument("--rescore", action="store_true", help="rebuild the scores table")
    s.set_defaults(func=cmd_resolve)

    s = sub.add_parser("report", help="tables and figures")
    s.add_argument("--out", default=None)
    s.add_argument("--no-figures", action="store_true")
    s.add_argument("--anchoring", nargs=2, metavar=("CONFIG_A", "CONFIG_B"))
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("status", help="what is in the database")
    s.set_defaults(func=cmd_status)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    args.func(args)


if __name__ == "__main__":
    main()
