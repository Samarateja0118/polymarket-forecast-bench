"""Paths, environment, and the model registry."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]

# Load .env on import so a credential never has to be typed into a shell (where
# it lands in history) or pasted anywhere it would be recorded. .env is
# gitignored; .env.example is the tracked template.
load_dotenv(REPO_ROOT / ".env")
DATA_DIR = REPO_ROOT / "data"
REPORT_DIR = REPO_ROOT / "reports"
CONFIG_DIR = REPO_ROOT / "configs"

CUTOFF_AUTO = "first_observed"


def db_path() -> Path:
    raw = os.environ.get("PMBENCH_DB", str(DATA_DIR / "pmbench.db"))
    p = Path(raw)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(ts: datetime) -> str:
    return ts.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass(frozen=True)
class ModelConfig:
    """One row of the benchmark grid: a model plus the knobs we vary."""

    name: str
    model_id: str
    effort: str = "high"
    thinking: str = "adaptive"          # "adaptive" | "off"
    supports_temperature: bool = False
    temperature: float | None = None
    web_search: bool = False
    effective_cutoff: str = CUTOFF_AUTO  # ISO date, or the CUTOFF_AUTO sentinel
    cutoff_basis: str = CUTOFF_AUTO
    cutoff_source: str = ""
    price_in_per_mtok: float = 0.0
    price_out_per_mtok: float = 0.0
    extra: dict = field(default_factory=dict)

    def cutoff_date(self, observed: dict[str, str] | None = None) -> date:
        """Resolve the cutoff to a concrete date.

        `observed` maps model_id -> ISO date of first successful call, from the
        `model_observations` table. Raises rather than guessing, because a wrong
        cutoff silently invalidates every downstream number.
        """
        if self.effective_cutoff != CUTOFF_AUTO:
            return date.fromisoformat(self.effective_cutoff)
        seen = (observed or {}).get(self.model_id)
        if seen is None:
            raise ValueError(
                f"model config {self.name!r} uses cutoff_basis={CUTOFF_AUTO!r} but "
                f"{self.model_id!r} has never been called. Run `pmbench probe` first, "
                f"or set an explicit effective_cutoff in configs/models.toml."
            )
        return date.fromisoformat(seen)

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens / 1e6 * self.price_in_per_mtok
            + output_tokens / 1e6 * self.price_out_per_mtok
        )


def load_models(path: Path | None = None) -> dict[str, ModelConfig]:
    path = path or (CONFIG_DIR / "models.toml")
    with path.open("rb") as fh:
        raw = tomllib.load(fh)
    out: dict[str, ModelConfig] = {}
    for row in raw.get("model", []):
        row = dict(row)
        known = {f for f in ModelConfig.__dataclass_fields__ if f != "extra"}
        extra = {k: v for k, v in row.items() if k not in known}
        cfg = ModelConfig(**{k: v for k, v in row.items() if k in known}, extra=extra)
        if cfg.name in out:
            raise ValueError(f"duplicate model config name {cfg.name!r} in {path}")
        out[cfg.name] = cfg
    if not out:
        raise ValueError(f"no [[model]] entries in {path}")
    return out
