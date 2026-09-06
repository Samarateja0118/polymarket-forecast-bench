import asyncio
import math

import pytest

from pmbench.config import ModelConfig
from pmbench.forecast import (
    Sample, aggregate, build_request, forecast_market, parse_response,
    spread_confidence, trimmed_mean,
)

OPUS = ModelConfig(name="o", model_id="claude-opus-5", effort="high",
                   thinking="adaptive", effective_cutoff="2026-01-01")
HAIKU = ModelConfig(name="h", model_id="claude-haiku-4-5", effort="",
                    thinking="off", supports_temperature=True, temperature=1.0,
                    effective_cutoff="2025-02-01")


def test_temperature_is_never_sent_to_models_that_reject_it():
    """Opus 5 / Sonnet 5 / 4.6+ return 400 on `temperature`. This is the guard."""
    assert "temperature" not in build_request(OPUS, "p")
    assert build_request(HAIKU, "p")["temperature"] == 1.0


def test_effort_and_thinking_only_where_configured():
    r = build_request(OPUS, "p")
    assert r["output_config"]["effort"] == "high"
    assert r["thinking"] == {"type": "adaptive"}
    h = build_request(HAIKU, "p")
    assert "thinking" not in h
    assert "effort" not in h.get("output_config", {})


def test_structured_output_can_be_disabled_without_changing_the_prompt():
    with_fmt = build_request(OPUS, "p", structured=True)
    without = build_request(OPUS, "p", structured=False)
    assert "format" in with_fmt["output_config"]
    assert "format" not in without.get("output_config", {})
    assert with_fmt["messages"] == without["messages"]
    assert with_fmt["system"] == without["system"]


def test_web_search_tool_only_when_enabled():
    assert "tools" not in build_request(OPUS, "p")
    searchy = ModelConfig(name="s", model_id="claude-opus-5", web_search=True)
    assert build_request(searchy, "p")["tools"][0]["name"] == "web_search"


def test_parse_response_handles_bare_and_wrapped_json():
    assert parse_response('{"probability": 0.3, "reasoning": "r"}') == (0.3, "r")
    p, _ = parse_response('Sure!\n```json\n{"probability":0.7,"reasoning":"r"}\n```')
    assert p == 0.7


def test_parse_response_rejects_out_of_range_and_garbage():
    with pytest.raises(ValueError):
        parse_response('{"probability": 1.4, "reasoning": "r"}')
    with pytest.raises(ValueError):
        parse_response("no json here")


def test_trimmed_mean_ignores_a_wild_sample():
    assert trimmed_mean([0.4, 0.42, 0.44, 0.46, 0.99]) == pytest.approx(0.44, abs=0.02)


def test_trimmed_mean_falls_back_to_median_when_too_small_to_trim():
    assert trimmed_mean([0.1, 0.9]) == pytest.approx(0.5)


def test_aggregate_derives_confidence_from_spread_not_self_report():
    agree = aggregate([Sample(0.30, "a"), Sample(0.31, "b"), Sample(0.29, "c")])
    disagree = aggregate([Sample(0.05, "a"), Sample(0.50, "b"), Sample(0.95, "c")])
    assert agree.ensemble_sd < disagree.ensemble_sd
    assert agree.confidence > disagree.confidence
    assert spread_confidence(0.5) == 0.0


def test_aggregate_survives_partial_failures():
    fc = aggregate([Sample(0.4, "a"), Sample(float("nan"), "", error="boom")])
    assert fc.error is None and fc.n_samples == 1 and fc.probability == 0.4


def test_aggregate_reports_error_when_every_sample_failed():
    fc = aggregate([Sample(float("nan"), "", error="boom")])
    assert fc.error == "boom" and math.isnan(fc.probability)


def test_reasoning_comes_from_the_sample_nearest_the_estimate():
    fc = aggregate([Sample(0.10, "low"), Sample(0.50, "mid"), Sample(0.90, "high")])
    assert fc.reasoning == "mid"


def test_forecast_market_runs_n_samples_and_paraphrases():
    seen = []

    async def fake(cfg, prompt):
        seen.append(prompt)
        return Sample(0.42, "r", 10, 5)

    market = dict(question="Q?", resolution_criteria="c" * 100,
                  end_date="2026-12-01T00:00:00Z")
    fc = asyncio.run(forecast_market(
        fake, OPUS, market, prompt_version="v1_base", n_samples=4,
        paraphrase_versions=["v1_base", "v2_reference_class"],
    ))
    assert fc.n_samples == 4 and fc.input_tokens == 40
    assert len(set(seen)) == 2   # two distinct framings cycled
