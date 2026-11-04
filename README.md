# pmbench — a self-labeling forecasting benchmark for LLMs

[![ci](https://github.com/Samarateja0118/polymarket-forecast-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/Samarateja0118/polymarket-forecast-bench/actions/workflows/ci.yml)

Prediction markets are a rare thing: a dataset that labels itself. You record a
forecast, and weeks later the world tells you whether you were right, for free,
with a timestamp. Most LLM evaluations have to construct ground truth by hand.
This one generates it automatically, forever.

The artifact is not "the model made money." It is **"N forecasts across M model
configurations, scored against resolved real-world events, with the calibration
curve."** A negative result — a frontier model is measurably less calibrated
than a prediction market, here is the reliability diagram — is a real result,
and a checkable one.

No trading code lives in this repository. Every Polymarket endpoint it touches
is read-only and unauthenticated.

---

## The one detail that makes it a benchmark instead of a memory test

> **Only forecast markets that resolve after the model's knowledge cutoff.**

Otherwise you are measuring recall, and every number downstream is worthless.
[`eligibility.py`](src/pmbench/eligibility.py) enforces this and reports the
full rejection funnel, so a run is reproducible rather than merely quoted:

```
=== haiku45-temp  (cutoff 2025-02-01, basis documented) ===
considered 8990 -> eligible 1447
  price_degenerate             3831
  inactive                     2807
  no_price_snapshot            2662
  volume_too_low               2268
  horizon_too_short            1872
  ...
```

Knowing a model's cutoff is itself a problem — models misreport their own, and
Anthropic does not publish one for every model. `configs/models.toml` therefore
records *how* each cutoff is known, via `cutoff_basis`:

| basis | meaning | safe? |
|---|---|---|
| `documented` | published by the vendor; put the URL in `cutoff_source` | yes, and tightest |
| `release_date` | the model's public release date | yes — a cutoff always precedes release |
| `first_observed` | the date this harness first successfully called the model | yes, and needs no outside knowledge — but over-estimates by however long the model has shipped, which costs sample size |

`first_observed` is the default and is auto-probed by `pmbench probe`. The code
**refuses to run** rather than guess a cutoff it does not have.

Over-estimating a cutoff costs you markets. Under-estimating it invalidates the
whole study. When in doubt, over-estimate.

---

## Two ways to get labels

**Live** (`--mode live`, the default) — forecast open markets and wait for the
world. Completely clean, and slow: four to six weeks to a meaningful sample.

**Retrospective** (`--mode retrospective --as-of <date>`) — forecast markets
that have *already resolved*, standing at a past instant. You get a labelled
dataset today instead of in six weeks, and it is safe for exactly one reason:
the cutoff rule still applies, so the model is only ever asked about outcomes
that post-date its training.

Retrospective mode has one hard requirement and one honest caveat:

- It needs a **real** cutoff. A `first_observed` cutoff is today's date, so
  nothing already resolved can clear it and the selection correctly comes back
  empty. You cannot use the fast path without stating what you believe.
- The market price must come from CLOB history at or before the as-of instant,
  never from the latest snapshot — for a resolved market the latest price *is*
  the answer, and a backtest built on it would look superhuman and be entirely
  fake. [`pricing.price_as_of`](src/pmbench/pricing.py) enforces the `<= as_of`
  comparison; [`test_pricing.py`](tests/test_pricing.py) pins it.

---

## Quick start

```bash
python3 -m venv .venv && ./.venv/bin/pip install -e ".[dev]"
```

```bash
./.venv/bin/pmbench ingest --pages 20
```

```bash
./.venv/bin/pmbench eligible --configs opus5-high --show 10
```

```bash
./.venv/bin/pmbench forecast --configs opus5-high sonnet5 --limit 50 --n-samples 5
```

```bash
./.venv/bin/pmbench resolve && ./.venv/bin/pmbench report
```

Forecasting needs credentials — either `ANTHROPIC_API_KEY` or an `ant auth
login` profile. Everything else (ingest, eligibility, scoring, figures) runs
without them.

To exercise the whole chain with no API key and no spend:

```bash
./.venv/bin/python scripts/smoke.py --limit 150
```

---

## Commands

| command | what it does | spends money |
|---|---|---|
| `ingest` | poll Gamma, upsert markets, append content-hashed price snapshots | no |
| `ingest --closed-since <date>` | backfill already-resolved markets for retrospective runs | no |
| `history` | backfill CLOB price history so as-of pricing works | no |
| `probe` | one cheap call per model to stamp `first_observed` | trivially |
| `eligible` | print the selection funnel | no |
| `forecast` | run the ensemble and store forecasts | **yes** |
| `resolve` | poll for resolutions, score against Brier and log loss | no |
| `report` | leaderboard, calibration fit, abstention curve, figures | no |
| `status` | what is in the database, including cumulative spend | no |

---

## How a forecast is made

```python
@dataclass
class Forecast:
    probability: float      # 0-1, validated; trimmed mean over N samples
    reasoning: str          # from the sample nearest the estimate
    confidence: float       # derived from ensemble spread, NOT self-reported
    ensemble_sd: float
```

N independent samples per market, aggregated by trimmed mean so one bad parse
cannot drag the estimate. The spread between samples is the uncertainty signal.
The model is never asked how confident it is — self-reported confidence from
LLMs is close to worthless, whereas disagreement between independent samples is
a real signal that [`analysis.abstention_curve`](src/pmbench/analysis.py)
validates against realised error.

### On sampling: `temperature` is gone

The obvious design is "sample N times at temperature > 0." That is no longer
available. **`temperature` was removed from Claude Opus 5, Sonnet 5, and the
4.6/4.7/4.8 family — sending it returns a 400.** Among current models only
Haiku 4.5 still accepts it.

So ensemble diversity comes from three axes instead, and
`ModelConfig.supports_temperature` gates the fourth so a config cannot silently
400:

1. **Independent calls.** These models are not deterministic; adaptive thinking
   in particular takes different paths across identical requests. This is the
   default axis, and what `--n-samples` varies.
2. **Prompt paraphrase** (`--paraphrase v1_base v2_reference_class`). Coarser
   and more honest: it measures sensitivity to framing, not just decode noise.
3. **Effort level**, as separate configs (`opus5-high` vs `opus5-low`).

Axis 1 is weaker than temperature sampling would be. **If sample SDs come back
near zero across the board, that is a finding about the aggregation method, not
evidence that the model is confident** — report it that way.

### Prompts are versioned data

Every prompt body is hashed into the `prompt_versions` table on first use.
Editing a version in place raises, because forecasts already recorded under that
name were made with different text. When a result shifts you need to know
whether the model changed or you did.

---

## Scoring

Brier and log loss on every resolved forecast, plus the Murphy decomposition
(`reliability - resolution + uncertainty`). Both proper scoring rules are
reported because log loss punishes confident errors far harder than Brier does,
and the gap between them is informative.

`uncertainty` is `base_rate * (1 - base_rate)` — a property of the *questions*,
not of the forecaster. This is why raw Brier scores are not comparable across
different question sets, and why every comparison here is paired on the same
markets.

### Three baselines, in increasing difficulty

1. **Always 0.5** — Brier 0.25 by construction.
2. **Category base rate** — computed **leave-one-out**. Using a market's own
   outcome in its own baseline lets the baseline peek at the answer, which
   inflates it and makes the model look worse than it is.
3. **The market price at forecast time.** The real one, and brutal. Aggregated
   money is hard to beat.

Significance is a **paired bootstrap** on the per-market Brier difference.
Paired, because both arms score the same markets and the market-to-market
variance is enormous — an unpaired test will not see a real difference through
it.

> **The numbers below are not a result.** They come from `scripts/smoke.py`,
> whose "model" is a deliberately overconfident stub and whose outcomes are
> synthesised. They show what the output *looks like*. No forecast in this
> repository has yet been made by an actual model — see [Status](#status).
> Figures generated from synthetic outcomes are stamped `SYNTHETIC` and their
> CSVs carry a matching comment line, so a stray PNG cannot be mistaken for a
> finding later.

```
baseline                 n    model     base     delta              95% CI       p    skill
--------------------------------------------------------------------------------------------
always_0.5             150   0.3744   0.2500   +0.1244 [+0.0591,+0.1940]   0.000   -0.497
category_base_rate     150   0.3744   0.1577   +0.2166 [+0.1371,+0.2968]   0.000   -1.374
market_price           150   0.3744   0.1028   +0.2715 [+0.1897,+0.3552]   0.000   -2.641
```

---

## The research questions

Each produces a chart, and each is a function you can call:

| question | entry point |
|---|---|
| Is it overconfident, and does a correction transfer? | `analysis.calibration_fit` — Platt scaling fit on a train split, gain reported on held-out test. Fitting and evaluating on the same forecasts always "improves" calibration; the split is the whole point. |
| Does ensemble disagreement predict error? | `analysis.abstention_curve` — Brier as the most-disagreeing forecasts are dropped, plus Spearman(&#124;error&#124;, SD). A flat curve means the spread is noise, which is itself a result. |
| Does web search help, or just anchor on the price it finds? | `analysis.anchoring` — compares how tightly two configs track the market on shared markets. The interesting failure is not "wrong" but "read the price off Polymarket and repeated it", which looks like skill against every baseline except the market. |
| Which categories? | `analysis.by_category` |
| Does quality decay with horizon? | `analysis.by_horizon` |

---

## Things that will bite you, which are handled here

**`endDate` is not the resolution date.** Markets inside a neg-risk group (the
"who will be the next X" events, a large fraction of Polymarket) inherit the
group's far-future `endDate`. One that settled on 2026-08-08 carries
`endDate` 2029-01-19. Computing a horizon from `endDate` invents a
two-and-a-half-year forecast that never happened. `gamma.effective_end()` uses
`closedTime` for anything resolved.

**`umaResolutionStatus` and `umaResolutionStatuses` disagree.** The singular
field is the settlement state; the plural is a per-proposal array that can still
read `["proposed"]` on a fully settled market. `resolve.decide()` trusts the
singular and uses the plural only to spot disputes.

**A closed market is not necessarily a resolved one.** `resolve.decide()`
additionally requires prices collapsed to the extremes (>= 0.99 / <= 0.01), holds
anything disputed, and records its evidence in `markets.outcome_source`.
`resolve --recheck` re-polls settled markets and shouts if an outcome flips.

**`interval=max` on CLOB price history is not "all of it."** The endpoint caps
at roughly 740 points, so at hourly fidelity you get about a month and older
data is silently missing. `history` walks the market's life in ~14-day windows
and stitches.

**Multi-outcome markets are dropped, not flattened.** Scoring assumes one
probability against a binary outcome; coercing a 4-way market into that shape is
how a benchmark goes quietly wrong.

**Snapshots are content-hashed.** Polling hourly for a year produces one row per
actual move, not 8,760 identical rows per market. Sub-cent volume noise is
rounded out of the hash so it does not count as a price change.

---

## Schema

```
markets(id, question, resolution_criteria, category, start_date, end_date,
        closed_time, resolved, outcome, outcome_source, volume_latest, ...)
price_snapshots(market_id, ts, yes_price, best_bid, best_ask, spread,
                volume, liquidity, content_hash)
runs(id, ts, git_sha, config_json, notes)
forecasts(id, run_id, market_id, ts, config_name, model_id, prompt_version,
          probability, reasoning, samples_json, ensemble_sd,
          market_price, horizon_days, cutoff_used, cutoff_basis,
          search_enabled, input_tokens, output_tokens, cost_usd, error)
scores(forecast_id, outcome, brier, log_loss, brier_market, log_loss_market, ...)
prompt_versions(version, body_hash, body, added_ts)
model_observations(model_id, first_called_on)
```

`forecasts.market_price` is frozen at forecast time and is the only price the
market baseline ever reads. Every run records its git SHA, its full config, and
its as-of instant.

---

## Status

Working and exercised against live Polymarket data: ingestion (~15k markets,
~5.8k with resolutions), the eligibility funnel, retrospective selection with
as-of pricing, scoring, baselines, calibration, and all five figures. 63 unit
tests.

**Not yet verified against the live API:** the model-calling path in
`forecast.anthropic_sampler`. It was written against the current Messages API
contract and is unit-tested through an injectable `SampleFn`, but no request has
been made from this machine — there are no Anthropic credentials on it. Expect
to shake out request-shape details (in particular whether `output_config.format`
is accepted alongside the web-search tool; there is a documented fallback path
that retries without structured outputs and leaves the prompt unchanged) on the
first real run. Start with `--limit 3 --n-samples 2`.

## Phase 5, and money

Simulating trades against recorded forecasts and recorded historical prices is
deliberately not implemented. Do it only if the calibration numbers justify it,
model the real fee curve, and if the result says you have edge, assume you have
a lookahead bug until you have proved otherwise. Phase 6 is "probably don't."

## License

MIT.
