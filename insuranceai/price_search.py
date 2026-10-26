"""Web price search for repair parts, labor and paint (India, INR).

Prices come from search snippets, so they are indicative only. Every result is
returned as a range with its sources; nothing here produces a single price.
"""

from __future__ import annotations

import logging
import os
import re
import statistics
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Literal

import httpx

logger = logging.getLogger(__name__)

Component = Literal["parts", "labor", "paint"]

# Plausibility bounds (INR) used to discard unrelated numbers found in snippets.
PRICE_BOUNDS: dict[str, tuple[float, float]] = {
    "parts": (150.0, 500_000.0),
    "labor": (100.0, 100_000.0),
    "paint": (300.0, 150_000.0),
}

_PRICE_PATTERN = re.compile(
    r"(?:₹|rs\.?|inr)\s*([\d,]+(?:\.\d+)?)"
    r"(?:\s*(?:-|–|—|to)\s*(?:₹|rs\.?|inr)?\s*([\d,]+(?:\.\d+)?))?",
    re.IGNORECASE,
)
MIN_USABLE_PRICES = 3
SEARCH_TIMEOUT_SECONDS = 15.0


@dataclass(frozen=True)
class PriceSource:
    url: str
    title: str
    retrieved_at: str


@dataclass
class PriceResult:
    query_id: str
    component: str
    part: str
    status: Literal["ok", "not_estimable", "error", "limit_reached", "not_configured"]
    low: float | None = None
    high: float | None = None
    price_count: int = 0
    confidence: Literal["low", "medium", "none"] = "none"
    sources: list[PriceSource] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "query_id": self.query_id,
            "component": self.component,
            "part": self.part,
            "status": self.status,
            "low": self.low,
            "high": self.high,
            "price_count": self.price_count,
            "confidence": self.confidence,
            "sources": [s.__dict__ for s in self.sources[:3]],
            "note": self.note,
        }


def extract_prices(text: str, component: str) -> list[float]:
    """Pull INR amounts out of text, expanding 'a - b' ranges and dropping implausible values."""
    low_bound, high_bound = PRICE_BOUNDS[component]
    prices: list[float] = []
    for match in _PRICE_PATTERN.finditer(text or ""):
        for group in match.groups():
            if not group:
                continue
            try:
                value = float(group.replace(",", ""))
            except ValueError:
                continue
            if low_bound <= value <= high_bound:
                prices.append(value)
    return prices


def summarize_prices(prices: list[float]) -> tuple[float, float, str] | None:
    """Return (low, high, confidence) from price points, or None when there is no usable data."""
    if not prices:
        return None
    median = statistics.median(prices)
    kept = [p for p in prices if median / 4 <= p <= median * 4]
    if not kept:
        return None
    low, high = min(kept), max(kept)
    if len(kept) < MIN_USABLE_PRICES:
        # Too few data points: widen the range rather than imply precision.
        return low * 0.75, high * 1.25, "low"
    return low, high, ("medium" if len(kept) >= 5 else "low")


def build_query(component: str, part: str, operation: str, vehicle: str) -> str:
    vehicle = vehicle.strip() or "car"
    if component == "parts":
        return f"{vehicle} {part} price India ₹"
    if component == "labor":
        return f"{vehicle} {part} {operation} labour charges India ₹"
    return f"{part} car painting cost India ₹ per panel"


class _SearchProvider:
    name = "base"

    def search(self, query: str) -> list[tuple[str, str, str]]:
        """Return (title, url, text) tuples."""
        raise NotImplementedError


class TavilyProvider(_SearchProvider):
    name = "tavily"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def search(self, query: str) -> list[tuple[str, str, str]]:
        response = httpx.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"query": query, "max_results": 8, "search_depth": "basic"},
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return [
            (r.get("title", ""), r.get("url", ""), r.get("content", ""))
            for r in response.json().get("results", [])
        ]


class SerperProvider(_SearchProvider):
    name = "serper"

    def __init__(self, api_key: str):
        self._api_key = api_key

    def search(self, query: str) -> list[tuple[str, str, str]]:
        response = httpx.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": self._api_key},
            json={"q": query, "gl": "in", "num": 10},
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        rows = [
            (r.get("title", ""), r.get("link", ""), r.get("snippet", ""))
            for r in data.get("organic", [])
        ]
        rows += [
            (r.get("title", ""), r.get("link", ""), f"{r.get('title', '')} {r.get('price', '')}")
            for r in data.get("shopping", [])
        ]
        return rows


def get_provider() -> _SearchProvider | None:
    """Pick a provider from PRICE_SEARCH_PROVIDER, falling back to whichever key is set."""
    tavily_key = os.getenv("TAVILY_SEARCH_API", "").strip()
    serper_key = os.getenv("SERPER_SEARCH_API", "").strip()
    preferred = os.getenv("PRICE_SEARCH_PROVIDER", "").strip().lower()
    if preferred == "serper" and serper_key:
        return SerperProvider(serper_key)
    if preferred == "tavily" and tavily_key:
        return TavilyProvider(tavily_key)
    if tavily_key:
        return TavilyProvider(tavily_key)
    if serper_key:
        return SerperProvider(serper_key)
    return None


# Process-local cache of (provider, query, component) -> (expires_at, prices, sources).
_CACHE: dict[tuple[str, str, str], tuple[float, list[float], list[PriceSource]]] = {}


def _cache_ttl_seconds() -> float:
    try:
        return float(os.getenv("PRICE_SEARCH_CACHE_TTL_HOURS", "24")) * 3600
    except ValueError:
        return 24 * 3600


def _max_calls() -> int:
    try:
        return max(int(os.getenv("PRICE_SEARCH_MAX_CALLS", "16")), 0)
    except ValueError:
        return 16


class PriceSearchSession:
    """Per-claim search state: enforces the call cap and records every result by query_id."""

    def __init__(
        self,
        provider: _SearchProvider | None = None,
        max_calls: int | None = None,
        on_result: Callable[["PriceSearchSession", PriceResult], None] | None = None,
    ):
        self.provider = provider if provider is not None else get_provider()
        self.max_calls = _max_calls() if max_calls is None else max_calls
        self.on_result = on_result
        self.calls = 0
        self.results: dict[str, PriceResult] = {}

    @property
    def configured(self) -> bool:
        return self.provider is not None

    def search(self, part: str, component: str, operation: str, vehicle: str) -> PriceResult:
        query_id = f"q{len(self.results) + 1}"
        part = (part or "").strip()[:80]
        if component not in PRICE_BOUNDS or not part:
            result = PriceResult(query_id, component, part, "error", note="Invalid part or component.")
        elif self.provider is None:
            result = PriceResult(query_id, component, part, "not_configured", note="Price search is not configured.")
        elif self.calls >= self.max_calls:
            result = PriceResult(query_id, component, part, "limit_reached", note="Search limit reached for this claim.")
        else:
            result = self._lookup(query_id, part, component, operation, vehicle)
        self.results[query_id] = result
        if self.on_result is not None:
            try:
                self.on_result(self, result)
            except Exception:  # progress reporting must never break a search
                logger.debug("price.search.on_result_failed", exc_info=True)
        logger.info(
            "price.search.result query_id=%s component=%s part=%r status=%s prices=%d confidence=%s",
            query_id, component, part, result.status, result.price_count, result.confidence,
        )
        return result

    def _lookup(self, query_id: str, part: str, component: str, operation: str, vehicle: str) -> PriceResult:
        assert self.provider is not None
        query = build_query(component, part, operation, vehicle)
        key = (self.provider.name, query, component)
        cached = _CACHE.get(key)
        if cached and cached[0] > time.time():
            prices, sources = cached[1], cached[2]
            logger.debug("price.search.cache_hit query=%r", query)
        else:
            self.calls += 1
            try:
                rows = self.provider.search(query)
            except Exception as exc:  # network/provider failure must not fail the claim
                logger.warning("price.search.failed provider=%s error=%s", self.provider.name, type(exc).__name__)
                return PriceResult(query_id, component, part, "error", note="Price search failed.")
            retrieved = datetime.now(timezone.utc).date().isoformat()
            prices, sources = [], []
            for title, url, text in rows:
                found = extract_prices(f"{title} {text}", component)
                if found:
                    prices.extend(found)
                    sources.append(PriceSource(url=url, title=title[:120], retrieved_at=retrieved))
            _CACHE[key] = (time.time() + _cache_ttl_seconds(), prices, sources)
            logger.debug("price.search.fetched query=%r rows=%d prices=%d", query, len(rows), len(prices))

        summary = summarize_prices(prices)
        if summary is None:
            return PriceResult(query_id, component, part, "not_estimable", sources=sources,
                               note="No usable prices found in search results.")
        low, high, confidence = summary
        return PriceResult(query_id, component, part, "ok", round(low), round(high), len(prices),
                           confidence, sources)


_current_session: ContextVar[PriceSearchSession | None] = ContextVar("price_search_session", default=None)


def set_price_session(session: PriceSearchSession | None):
    return _current_session.set(session)


def reset_price_session(token) -> None:
    _current_session.reset(token)


def get_price_session() -> PriceSearchSession | None:
    return _current_session.get()
