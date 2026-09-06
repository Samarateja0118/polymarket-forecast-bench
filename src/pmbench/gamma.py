"""Async clients for Polymarket's Gamma API and the CLOB price-history endpoint.

Read-only, unauthenticated. No trading code lives anywhere in this repo.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, AsyncIterator, Iterable, Sequence

import httpx

log = logging.getLogger(__name__)

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class GammaClient:
    """Bounded-concurrency, retrying HTTP client.

    Polymarket has no published rate limit for these endpoints; `concurrency`
    and `min_interval` keep us politely under whatever it is.
    """

    concurrency: int = 6
    timeout: float = 30.0
    max_retries: int = 4
    min_interval: float = 0.05

    def __post_init__(self) -> None:
        self._sem = asyncio.Semaphore(self.concurrency)
        self._client: httpx.AsyncClient | None = None
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> "GammaClient":
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            headers={"User-Agent": "pmbench/0.1 (research; read-only)"},
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._client:
            await self._client.aclose()
        self._client = None

    async def _throttle(self) -> None:
        async with self._lock:
            now = asyncio.get_running_loop().time()
            wait = self._last + self.min_interval - now
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_running_loop().time()

    async def get(self, url: str, params: dict[str, Any] | None = None) -> Any:
        assert self._client is not None, "use `async with GammaClient() as c`"
        delay = 1.0
        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            async with self._sem:
                await self._throttle()
                try:
                    r = await self._client.get(url, params=params)
                except (httpx.TransportError, httpx.TimeoutException) as exc:
                    last_exc = exc
                else:
                    if r.status_code == 200:
                        return r.json()
                    if r.status_code == 404:
                        return None
                    if r.status_code not in RETRY_STATUS:
                        r.raise_for_status()
                    last_exc = httpx.HTTPStatusError(
                        f"{r.status_code} for {url}", request=r.request, response=r
                    )
            if attempt < self.max_retries:
                await asyncio.sleep(delay)
                delay *= 2
        raise RuntimeError(f"GET {url} failed after {self.max_retries + 1} attempts") from last_exc

    # ---------- Gamma ----------

    async def iter_events(
        self,
        *,
        closed: bool = False,
        end_date_min: str | None = None,
        end_date_max: str | None = None,
        order: str = "volume",
        ascending: bool = False,
        page_size: int = 100,
        max_pages: int = 50,
    ) -> AsyncIterator[dict]:
        """Page through /events.

        We ingest via events rather than markets because the event payload
        carries `tags` (our category label) *and* nests the full market objects,
        so one request gives us both.
        """
        for page in range(max_pages):
            params: dict[str, Any] = {
                "limit": page_size,
                "offset": page * page_size,
                "closed": str(closed).lower(),
                "order": order,
                "ascending": str(ascending).lower(),
            }
            if end_date_min:
                params["end_date_min"] = end_date_min
            if end_date_max:
                params["end_date_max"] = end_date_max
            batch = await self.get(f"{GAMMA}/events", params)
            if not batch:
                return
            for ev in batch:
                yield ev
            if len(batch) < page_size:
                return

    async def market(self, market_id: str) -> dict | None:
        return await self.get(f"{GAMMA}/markets/{market_id}")

    async def markets(self, market_ids: Sequence[str]) -> list[dict]:
        """Fetch many markets. Gamma ignores repeated ?id=, so this fans out."""
        results = await asyncio.gather(
            *(self.market(mid) for mid in market_ids), return_exceptions=True
        )
        out: list[dict] = []
        for mid, res in zip(market_ids, results):
            if isinstance(res, BaseException):
                log.warning("market %s fetch failed: %s", mid, res)
            elif res:
                out.append(res)
        return out

    # ---------- CLOB ----------

    async def price_history(
        self, clob_token_id: str, *, interval: str = "max", fidelity: int = 60
    ) -> list[dict]:
        """Historical mid-prices for one outcome token. [{t: epoch_s, p: float}]

        NB: `interval=max` is not "all of it" — the endpoint caps the response
        at roughly 740 points, so at fidelity=60 you get about a month and
        anything older is silently missing. Use `price_history_window` for
        retrospective work where the gaps would matter.
        """
        data = await self.get(
            f"{CLOB}/prices-history",
            {"market": clob_token_id, "interval": interval, "fidelity": fidelity},
        )
        return (data or {}).get("history", []) or []

    async def price_history_window(
        self,
        clob_token_id: str,
        start: datetime,
        end: datetime,
        *,
        fidelity: int = 60,
        chunk_days: int = 14,
    ) -> list[dict]:
        """Full history over [start, end], fetched in windows and stitched.

        The endpoint rejects a startTs/endTs span it considers too long (and
        also rejects windows that reach back before the token existed), so we
        walk forward in `chunk_days` slices and tolerate per-window failures
        rather than losing the whole market to one bad slice.
        """
        out: list[dict] = []
        cursor = start
        step = timedelta(days=chunk_days)
        while cursor < end:
            stop = min(cursor + step, end)
            try:
                data = await self.get(
                    f"{CLOB}/prices-history",
                    {
                        "market": clob_token_id,
                        "startTs": int(cursor.timestamp()),
                        "endTs": int(stop.timestamp()),
                        "fidelity": fidelity,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                log.debug("history window %s..%s failed: %s", cursor, stop, exc)
                data = None
            out.extend((data or {}).get("history", []) or [])
            cursor = stop
        seen: set[int] = set()
        deduped = []
        for point in sorted(out, key=lambda d: d["t"]):
            if point["t"] not in seen:
                seen.add(point["t"])
                deduped.append(point)
        return deduped


# ---------- normalisation ----------


def _jsonish(value: Any) -> Any:
    """Gamma returns several array fields as JSON-encoded strings."""
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return value


def _f(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# Gamma tags are a flat, noisy list ("Politics", "Earn 4%", "Recurring"...).
# We map to a small, stable category set; the raw list is stored alongside.
CATEGORY_PRIORITY = [
    "Crypto", "Elections", "Politics", "Geopolitics", "Economy", "Business",
    "Tech", "Science", "Sports", "Culture", "Weather", "Health",
]


def pick_category(tags: Iterable[dict]) -> str:
    labels = {t.get("label", "") for t in tags or []}
    for cat in CATEGORY_PRIORITY:
        if cat in labels:
            return cat
    for t in tags or []:
        label = t.get("label")
        if label and t.get("forceShow") is not None:
            return label
    return "Other"


def _closed_time(market: dict) -> str | None:
    """Gamma returns `closedTime` as "2026-08-08 20:27:18+00" — space-separated,
    two-digit offset. Not ISO, and datetime.fromisoformat rejects it."""
    raw = market.get("closedTime")
    if not raw or not isinstance(raw, str):
        return None
    v = raw.strip().replace(" ", "T")
    if v.endswith("+00"):
        v = v[:-3] + "+00:00"
    return v


def effective_end(market: dict) -> str | None:
    """When the question was actually decided.

    `endDate` is NOT the resolution date. Markets inside a neg-risk group (the
    "who will be the next X" events, which are a large fraction of Polymarket)
    inherit the group's far-future endDate — one settled in Aug 2026 carries
    endDate 2029-01-19. Computing a horizon from endDate would invent a
    two-and-a-half-year forecast that never happened, so anything resolved uses
    `closed_time` instead.
    """
    if market.get("resolved") and market.get("closed_time"):
        return market["closed_time"]
    return market.get("end_date")


def snapshot_of(market: dict) -> dict:
    """Extract the volatile price/liquidity fields from a raw Gamma market."""
    prices = _jsonish(market.get("outcomePrices")) or []
    yes = _f(prices[0]) if prices else None
    bid, ask = _f(market.get("bestBid")), _f(market.get("bestAsk"))
    spread = _f(market.get("spread"))
    if spread is None and bid is not None and ask is not None:
        spread = ask - bid
    return {
        "yes_price": yes,
        "best_bid": bid,
        "best_ask": ask,
        "spread": spread,
        "volume": _f(market.get("volumeNum")) or _f(market.get("volume")),
        "liquidity": _f(market.get("liquidityNum")) or _f(market.get("liquidity")),
    }


def normalise(event: dict, market: dict) -> dict | None:
    """Flatten one (event, market) pair into a `markets` row.

    Returns None for anything that isn't a plain binary Yes/No market — the
    scoring code assumes a single probability against a binary outcome, and
    silently coercing multi-outcome markets into that shape is how benchmarks
    get quietly wrong.
    """
    outcomes = _jsonish(market.get("outcomes")) or []
    if [str(o).strip().lower() for o in outcomes] != ["yes", "no"]:
        return None

    tokens = _jsonish(market.get("clobTokenIds")) or []
    tags = event.get("tags") or []
    snap = snapshot_of(market)

    return {
        "id": str(market.get("id")),
        "condition_id": market.get("conditionId"),
        "slug": market.get("slug"),
        "question": (market.get("question") or "").strip(),
        "resolution_criteria": (market.get("description") or "").strip(),
        "resolution_source": market.get("resolutionSource") or "",
        "event_id": str(event.get("id")) if event.get("id") is not None else None,
        "event_title": event.get("title"),
        "category": pick_category(tags),
        "tags_json": json.dumps([t.get("label") for t in tags if t.get("label")]),
        "start_date": market.get("startDate") or event.get("startDate"),
        "end_date": market.get("endDate") or event.get("endDate"),
        "created_at": market.get("createdAt"),
        "closed_time": _closed_time(market),
        "clob_token_yes": str(tokens[0]) if tokens else None,
        "neg_risk": int(bool(market.get("negRisk"))),
        # Latest known totals, kept on the market row so a resolved market
        # (whose only snapshots are CLOB history, which carries no volume) can
        # still be size-filtered. Selection only — never scored.
        "volume_latest": snap["volume"],
        "liquidity_latest": snap["liquidity"],
        "active": int(bool(market.get("active"))),
        "closed": int(bool(market.get("closed"))),
        "archived": int(bool(market.get("archived"))),
    }
