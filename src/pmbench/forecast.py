"""The forecasting harness.

One estimate = N independent samples, aggregated by trimmed mean. The spread
between samples is the uncertainty signal. We never ask the model how confident
it is: self-reported confidence from LLMs is close to worthless, whereas
disagreement between independent samples is a real signal that can be validated
against realised error (see analysis.disagreement_vs_error).

A note on sampling, because the plan this was built from assumed otherwise:
`temperature` was REMOVED from Claude Opus 5, Sonnet 5, and the 4.6/4.7/4.8
family — sending it returns a 400. So "sample at temperature > 0" is not
available on the models worth benchmarking. Diversity instead comes from:

  1. Independent calls. These models are not deterministic; adaptive thinking
     in particular takes different paths across identical requests. This is the
     default axis and is what `n_samples` varies.
  2. Prompt paraphrase. Different `prompt_version` per sample — a coarser,
     more honest ensemble (it measures sensitivity to framing, not just decode
     noise). Enable with `paraphrase_versions`.
  3. `temperature`, only where the model still accepts it (Haiku 4.5). Gated on
     ModelConfig.supports_temperature so a config can't silently 400.

Axis 1 is weaker than temperature sampling would be. If sample SDs come back
near zero across the board, that is a finding about the aggregation method, not
about the model's uncertainty — report it as such rather than concluding the
model is confident.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol, Sequence

from .config import ModelConfig, iso, utcnow
from .prompting import render

log = logging.getLogger(__name__)

SYSTEM = (
    "You are a careful superforecaster. You are calibrated: when you say 70%, "
    "the thing happens about 70% of the time. You give extreme probabilities "
    "when the evidence supports them and refuse to when it does not.\n\n"
    "Respond with a single JSON object and nothing else:\n"
    '{"probability": <number between 0 and 1>, "reasoning": "<your reasoning>"}'
)

RESPONSE_SCHEMA = {
    "type": "json_schema",
    "schema": {
        "type": "object",
        "properties": {
            "probability": {"type": "number", "minimum": 0, "maximum": 1},
            "reasoning": {"type": "string"},
        },
        "required": ["probability", "reasoning"],
        "additionalProperties": False,
    },
}

WEB_SEARCH_TOOL = {"type": "web_search_20260209", "name": "web_search", "max_uses": 5}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


@dataclass
class Sample:
    probability: float
    reasoning: str
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None


@dataclass
class Forecast:
    """One aggregated estimate for one market under one config."""

    probability: float          # 0-1, validated; trimmed mean of the samples
    reasoning: str              # reasoning from the sample nearest the estimate
    confidence: float           # derived from ensemble spread, NOT self-reported
    ensemble_sd: float
    samples: list[Sample] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None

    @property
    def n_samples(self) -> int:
        return sum(1 for s in self.samples if s.error is None)


# --------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------


def trimmed_mean(values: Sequence[float], proportion: float = 0.2) -> float:
    """Mean after dropping `proportion` from each tail.

    Falls back to the median for samples too small to trim, so a single wild
    parse never drags the estimate.
    """
    xs = sorted(values)
    if not xs:
        raise ValueError("no values")
    k = int(len(xs) * proportion)
    if len(xs) - 2 * k < 1:
        return statistics.median(xs)
    core = xs[k: len(xs) - k] if k else xs
    return sum(core) / len(core)


def spread_confidence(sd: float) -> float:
    """Map ensemble SD to a 0-1 confidence.

    SD of probabilities is bounded by 0.5 (maximal disagreement is half the
    samples at 0 and half at 1), so 1 - 2*sd spans the range. This is a
    monotone relabelling of the SD for convenience — the SD is the real
    quantity and is what gets stored and analysed.
    """
    return max(0.0, min(1.0, 1.0 - 2.0 * sd))


def aggregate(samples: list[Sample]) -> Forecast:
    ok = [s for s in samples if s.error is None]
    tok_in = sum(s.input_tokens for s in samples)
    tok_out = sum(s.output_tokens for s in samples)
    if not ok:
        first = next((s.error for s in samples if s.error), "no samples")
        return Forecast(float("nan"), "", 0.0, float("nan"), samples,
                        tok_in, tok_out, error=first)
    probs = [s.probability for s in ok]
    est = trimmed_mean(probs)
    sd = statistics.stdev(probs) if len(probs) > 1 else 0.0
    nearest = min(ok, key=lambda s: abs(s.probability - est))
    return Forecast(est, nearest.reasoning, spread_confidence(sd), sd,
                    samples, tok_in, tok_out)


# --------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def parse_response(text: str) -> tuple[float, str]:
    """Tolerant parse. Structured outputs make this trivial; the fallback path
    (web-search configs, where output_config.format may be rejected) does not."""
    blob = text.strip()
    if not blob.startswith("{"):
        m = _JSON_RE.search(blob)
        if not m:
            raise ValueError(f"no JSON object in response: {blob[:200]!r}")
        blob = m.group(0)
    data = json.loads(blob)
    p = float(data["probability"])
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"probability out of range: {p}")
    return p, str(data.get("reasoning", ""))


class SampleFn(Protocol):
    async def __call__(self, cfg: ModelConfig, prompt: str) -> Sample: ...


def build_request(cfg: ModelConfig, prompt: str, *, structured: bool = True) -> dict[str, Any]:
    """Assemble the Messages API request for one sample."""
    req: dict[str, Any] = {
        "model": cfg.model_id,
        # Adaptive thinking spends tokens before the JSON starts, and a
        # truncated response fails the parse and burns the sample. This is a
        # ceiling, not a spend.
        "max_tokens": 16000,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": prompt}],
    }
    output_config: dict[str, Any] = {}
    if structured:
        output_config["format"] = RESPONSE_SCHEMA
    if cfg.effort:
        output_config["effort"] = cfg.effort
    if output_config:
        req["output_config"] = output_config
    if cfg.thinking == "adaptive":
        req["thinking"] = {"type": "adaptive"}
    if cfg.supports_temperature and cfg.temperature is not None:
        # Only Haiku 4.5 among current models; 400 on Opus 5 / Sonnet 5 / 4.6+.
        req["temperature"] = cfg.temperature
    if cfg.web_search:
        req["tools"] = [WEB_SEARCH_TOOL]
    return req


def anthropic_sampler(client: Any) -> SampleFn:
    """Adapt an `anthropic.AsyncAnthropic` into a SampleFn."""

    async def sample(cfg: ModelConfig, prompt: str) -> Sample:
        for structured in (True, False):
            req = build_request(cfg, prompt, structured=structured)
            try:
                resp = await client.messages.create(**req)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                msg = f"{type(exc).__name__}: {exc}"
                # A 400 naming the output format means this config can't use
                # structured outputs (search tools + format). Retry once plain;
                # the prompt is unchanged, so the arms stay comparable.
                if structured and "400" in msg and (
                    "output_config" in msg or "format" in msg
                ):
                    log.warning("%s: structured output rejected, retrying plain", cfg.name)
                    continue
                return Sample(float("nan"), "", error=msg)

            usage = getattr(resp, "usage", None)
            tin = getattr(usage, "input_tokens", 0) or 0
            tout = getattr(usage, "output_tokens", 0) or 0

            if getattr(resp, "stop_reason", None) == "refusal":
                details = getattr(resp, "stop_details", None)
                cat = getattr(details, "category", None)
                return Sample(float("nan"), "", tin, tout, error=f"refusal:{cat}")

            texts = [b.text for b in resp.content if getattr(b, "type", None) == "text"]
            if not texts:
                return Sample(float("nan"), "", tin, tout, error="no text block in response")
            try:
                p, reasoning = parse_response(texts[-1])
            except (ValueError, KeyError, TypeError) as exc:
                return Sample(float("nan"), "", tin, tout, error=f"parse: {exc}")
            return Sample(p, reasoning, tin, tout)

        return Sample(float("nan"), "", error="unreachable")

    return sample


# --------------------------------------------------------------------------
# ensemble
# --------------------------------------------------------------------------


async def forecast_market(
    sample_fn: SampleFn,
    cfg: ModelConfig,
    market: dict,
    *,
    prompt_version: str,
    n_samples: int = 5,
    paraphrase_versions: Sequence[str] | None = None,
    now: datetime | None = None,
) -> Forecast:
    """Run the ensemble for one market.

    `paraphrase_versions`, if given, cycles prompt versions across samples so
    the spread reflects framing sensitivity as well as decode noise.
    """
    now = now or utcnow()
    versions = list(paraphrase_versions) if paraphrase_versions else [prompt_version]
    prompts = [render(versions[i % len(versions)], market, now=now) for i in range(n_samples)]
    samples = await asyncio.gather(*(sample_fn(cfg, p) for p in prompts))
    return aggregate(list(samples))


def new_run_id() -> str:
    return f"run_{utcnow().strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:6]}"


def new_forecast_id() -> str:
    return f"fc_{uuid.uuid4().hex[:16]}"
