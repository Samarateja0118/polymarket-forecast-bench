"""Provider backends behind the single `SampleFn` protocol.

The benchmark is deliberately provider-agnostic: eligibility, scoring,
baselines, calibration and the report layer never learn which vendor produced a
probability. Only this module and `ModelConfig.provider` know.

One asymmetry has to be reported rather than hidden. Gemini accepts
`temperature`; current Anthropic models reject it (400). So a Gemini arm can
draw a *true* temperature ensemble while an Anthropic arm can only draw
independent calls. That is a confound in any cross-provider comparison of
ensemble spread — it is not evidence that one vendor is more self-consistent
than the other. Compare spreads only within a provider, or pin the Gemini arm
to `supports_temperature = false` so both sides use the same weaker axis. The
`gemini38-flash-fixed` config exists for exactly that controlled comparison.
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .config import ModelConfig
from .forecast import (
    RESPONSE_SCHEMA, SYSTEM, Sample, SampleFn, anthropic_sampler, parse_response,
)

log = logging.getLogger(__name__)

MAX_OUTPUT_TOKENS = 16000

# The JSON contract, in the shape google-genai wants. Same fields as the
# Anthropic RESPONSE_SCHEMA so both arms are answering the identical question.
GEMINI_SCHEMA: dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "probability": {"type": "NUMBER"},
        "reasoning": {"type": "STRING"},
    },
    "required": ["probability", "reasoning"],
}


def build_gemini_config(cfg: ModelConfig) -> Any:
    """Assemble a GenerateContentConfig for one sample."""
    from google.genai import types

    kwargs: dict[str, Any] = {
        "system_instruction": SYSTEM,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_mime_type": "application/json",
        "response_schema": GEMINI_SCHEMA,
        # We pass no tools, so the SDK's automatic function calling has nothing
        # to do; leaving it on emits a warning on every call and wraps the
        # request in machinery we do not want in the loop.
        "automatic_function_calling": types.AutomaticFunctionCallingConfig(disable=True),
    }
    if cfg.supports_temperature and cfg.temperature is not None:
        kwargs["temperature"] = cfg.temperature

    # "off" is the only case where we send an explicit budget: several 3.x
    # models reject a zero budget, so we let the model choose unless told not
    # to think at all, and surface the rejection rather than swallowing it.
    if cfg.thinking == "off":
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    elif cfg.thinking in ("low", "medium", "high"):
        kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=cfg.thinking)

    return types.GenerateContentConfig(**kwargs)


def gemini_sampler(client: Any) -> SampleFn:
    """Adapt a `google.genai.Client` into a SampleFn."""

    async def sample(cfg: ModelConfig, prompt: str) -> Sample:
        try:
            resp = await client.aio.models.generate_content(
                model=cfg.model_id,
                contents=prompt,
                config=build_gemini_config(cfg),
            )
        except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
            return Sample(float("nan"), "", error=f"{type(exc).__name__}: {exc}")

        usage = getattr(resp, "usage_metadata", None)
        tin = getattr(usage, "prompt_token_count", 0) or 0
        # Thinking tokens bill as output; folding them in keeps the cost column
        # comparable to the Anthropic arm's.
        tout = (getattr(usage, "candidates_token_count", 0) or 0) + (
            getattr(usage, "thoughts_token_count", 0) or 0
        )

        blocked = getattr(getattr(resp, "prompt_feedback", None), "block_reason", None)
        if blocked:
            return Sample(float("nan"), "", tin, tout, error=f"blocked:{blocked}")
        if not getattr(resp, "candidates", None):
            return Sample(float("nan"), "", tin, tout, error="no candidates returned")

        finish = getattr(resp.candidates[0], "finish_reason", None)
        text = getattr(resp, "text", None)
        if not text:
            return Sample(float("nan"), "", tin, tout,
                          error=f"empty response (finish_reason={finish})")
        try:
            p, reasoning = parse_response(text)
        except (ValueError, KeyError, TypeError) as exc:
            return Sample(float("nan"), "", tin, tout, error=f"parse: {exc}")
        return Sample(p, reasoning, tin, tout)

    return sample


def build_router(clients: dict[str, Any] | None = None) -> SampleFn:
    """One SampleFn that dispatches on `cfg.provider`.

    Clients are built lazily so a run using only one provider never needs the
    other's credentials — which is the whole reason a Gemini-only run works on
    a machine with no Anthropic key.
    """
    made: dict[str, SampleFn] = dict(clients or {})

    def backend(provider: str) -> SampleFn:
        if provider in made:
            return made[provider]
        if provider == "anthropic":
            import anthropic

            made[provider] = anthropic_sampler(anthropic.AsyncAnthropic())
        elif provider == "gemini":
            from google import genai

            from google.genai import types

            key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError(
                    "GEMINI_API_KEY is not set. Put it in .env (gitignored) — "
                    "see .env.example."
                )
            # 503 UNAVAILABLE ("high demand") is common and transient; without
            # retries it silently halves an ensemble, and a 1-sample ensemble
            # carries no spread information at all. 429 is the free-tier rate
            # limit, which backoff also absorbs.
            made[provider] = gemini_sampler(genai.Client(
                api_key=key,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(
                        attempts=5, initial_delay=1.0, max_delay=30.0,
                        http_status_codes=[429, 500, 502, 503, 504],
                    ),
                ),
            ))
        else:
            raise ValueError(f"unknown provider {provider!r}")
        return made[provider]

    async def route(cfg: ModelConfig, prompt: str) -> Sample:
        try:
            fn = backend(cfg.provider)
        except Exception as exc:  # noqa: BLE001
            return Sample(float("nan"), "", error=f"{type(exc).__name__}: {exc}")
        return await fn(cfg, prompt)

    return route
