"""End-to-end smoke test with a synthetic forecaster — no API key, no spend.

Runs the real ingest -> select -> forecast -> resolve -> score -> report chain
against a copy of the database, substituting a stub sampler for the model and
synthetic outcomes for the world.

Outcomes are drawn as y ~ Bernoulli(market_price), which makes the market price
*perfectly calibrated by construction*. That gives the test a real assertion
rather than a "did it crash" check:

  - the market baseline must come out near-perfectly calibrated,
  - the deliberately overconfident synthetic model must score worse than it,
  - and the reliability diagram must show it sitting below the diagonal.

If the arithmetic in scoring.py or baselines.py breaks, this fails loudly.

    python scripts/smoke.py --limit 150
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pmbench import analysis, report, resolve, runner  # noqa: E402
from pmbench.baselines import format_comparisons  # noqa: E402
from pmbench.config import load_models  # noqa: E402
from pmbench.db import connect, record_observation  # noqa: E402
from pmbench.eligibility import EligibilityPolicy  # noqa: E402
from pmbench.forecast import Sample  # noqa: E402


def overconfident_sampler(sharpening: float = 2.2, noise: float = 0.06, seed: int = 7):
    """A stand-in for the model: takes a noisy read on the truth, then states it
    far more confidently than it earns. The usual LLM failure mode, simulated."""
    rng = random.Random(seed)

    async def sample(cfg, prompt: str) -> Sample:
        # A weak signal correlated with the question text, so the same market
        # gets a stable-ish belief across samples.
        base = (hash(prompt[:400]) % 1000) / 1000.0
        belief = min(max(base + rng.gauss(0, noise), 0.01), 0.99)
        sharp = 0.5 + (belief - 0.5) * sharpening
        return Sample(min(max(sharp, 0.001), 0.999), "synthetic", 900, 300)

    return sample


def synthesise_outcomes(conn, seed: int = 11) -> int:
    """Draw y ~ Bernoulli(price recorded at forecast time)."""
    rng = random.Random(seed)
    rows = conn.execute(
        "SELECT DISTINCT f.market_id, f.market_price FROM forecasts f "
        "WHERE f.market_price IS NOT NULL AND f.error IS NULL"
    ).fetchall()
    for r in rows:
        y = 1 if rng.random() < float(r["market_price"]) else 0
        conn.execute(
            "UPDATE markets SET resolved = 1, outcome = ?, outcome_source = 'SYNTHETIC' "
            "WHERE id = ?",
            (y, r["market_id"]),
        )
    conn.commit()
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150, help="markets per config")
    ap.add_argument("--n-samples", type=int, default=5)
    ap.add_argument("--source-db", default=str(ROOT / "data" / "pmbench.db"))
    ap.add_argument("--db", default=str(ROOT / "data" / "smoke.db"))
    ap.add_argument("--out", default=str(ROOT / "reports" / "smoke"))
    args = ap.parse_args()

    src, dst = Path(args.source_db), Path(args.db)
    if not src.exists():
        print(f"no market data at {src} — run `pmbench ingest` first")
        return 2
    # SQLite's backup API, not shutil.copyfile: the source runs in WAL mode, so
    # a plain file copy omits the -wal and yields a torn "database disk image is
    # malformed" clone.
    for suffix in ("", "-wal", "-shm"):
        Path(str(dst) + suffix).unlink(missing_ok=True)
    source = sqlite3.connect(src)
    target = sqlite3.connect(dst)
    with target:
        source.backup(target)
    source.close()
    target.close()
    conn = connect(dst)
    conn.execute("DELETE FROM scores")
    conn.execute("DELETE FROM forecasts")
    conn.execute("UPDATE markets SET resolved = 0, outcome = NULL "
                 "WHERE outcome_source = 'SYNTHETIC'")
    conn.commit()

    models = load_models()
    cfgs = [models["haiku45-temp"], models["sonnet5"]]
    # sonnet5 ships with cutoff_basis="first_observed"; stamp an observation so
    # the smoke test doesn't need to call the API just to learn a date.
    record_observation(conn, models["sonnet5"].model_id, "2025-06-01")

    st = asyncio.run(runner.run(
        conn, cfgs,
        prompt_version="v2_reference_class",
        n_samples=args.n_samples,
        limit=args.limit,
        policy=EligibilityPolicy(min_volume_usd=20_000),
        sample_fn=overconfident_sampler(),
        notes="synthetic smoke test",
    ))
    print(st)

    print(f"synthesised outcomes for {synthesise_outcomes(conn)} markets")
    print(f"scored {resolve.score_pending(conn)} forecasts\n")

    rows = analysis.load_rows(conn)
    if not rows:
        print("FAIL: no scored rows")
        return 1

    ok = True
    for name, comps in analysis.leaderboard(rows).items():
        print(f"=== {name} ===")
        print(format_comparisons(comps))
        cfg_rows = [r for r in rows if r.config_name == name]
        cal = analysis.calibration_fit(cfg_rows)
        print("\n" + str(cal))
        print("\n" + str(analysis.abstention_curve(cfg_rows)) + "\n")

        market_cmp = next(c for c in comps if c.name == "market_price")
        if market_cmp.baseline.ece > 0.10:
            print(f"FAIL: market baseline should be calibrated by construction, "
                  f"ECE={market_cmp.baseline.ece:.3f}")
            ok = False
        if market_cmp.brier_delta <= 0:
            print("FAIL: the overconfident synthetic model should lose to the market")
            ok = False
        if not (cal.platt_a < 1.0):
            print(f"FAIL: Platt slope should be < 1 for overconfident input, got {cal.platt_a:.3f}")
            ok = False

    for p in report.build_all(rows, Path(args.out)):
        print(f"wrote {p}")

    print("\nSMOKE TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
