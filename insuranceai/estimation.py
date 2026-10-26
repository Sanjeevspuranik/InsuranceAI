"""Turn the estimation agent's plan into deterministic, range-only cost estimates.

The LLM only names parts and cites search query ids. Every number comes from the
recorded PriceSearchSession results and is combined by claims_calc.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, ValidationError, field_validator

from insuranceai.claims_calc import (
    DISCLAIMER,
    ZERO,
    ClaimBreakdown,
    LineItem,
    PolicyInput,
    Range,
    compute_claimable,
)
from insuranceai.price_search import PriceResult, PriceSearchSession

logger = logging.getLogger(__name__)

MAX_ITEMS = 12
INTERNAL_NOTE = (
    "Possible internal damage is not confirmed and is NOT included in the claimable "
    "estimate. These amounts apply only if an inspection confirms the damage."
)
_CONFIDENCE_ORDER = {"none": 0, "low": 1, "medium": 2}


class PlannedItem(BaseModel):
    part: str
    material: Literal["metal", "rubber_plastic", "fibreglass", "glass"] = "metal"
    operation: Literal["repair", "replace", "repaint"] = "replace"
    parts_query_id: Optional[str] = None
    labor_query_id: Optional[str] = None
    paint_query_id: Optional[str] = None
    likelihood: Optional[Literal["low", "medium", "high"]] = None
    reason: Optional[str] = None

    @field_validator("part")
    @classmethod
    def _clean_part(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("part must not be empty")
        return value[:80]


class EstimationPlan(BaseModel):
    vehicle: str = "unknown"
    visible_damage: list[PlannedItem] = []
    internal_damage: list[PlannedItem] = []


def parse_plan(text: str) -> EstimationPlan:
    """Extract the first JSON object from the agent's final message, ignoring surrounding text."""
    decoder = json.JSONDecoder()
    text = text or ""
    last_error: Exception | None = None
    for match in re.finditer(r"\{", text):
        try:
            data, _ = decoder.raw_decode(text, match.start())
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(data, dict) and ("visible_damage" in data or "internal_damage" in data):
            plan = EstimationPlan.model_validate(data)
            plan.visible_damage = plan.visible_damage[:MAX_ITEMS]
            plan.internal_damage = plan.internal_damage[:MAX_ITEMS]
            return plan
    raise ValueError("No valid JSON plan found in the estimation response.") from last_error


def _range_dict(r: Range | None) -> dict | None:
    return None if r is None else {"low": round(r.low), "high": round(r.high)}


def _lookup(session: PriceSearchSession, query_id: Optional[str], component: str) -> PriceResult | None:
    result = session.results.get(query_id) if query_id else None
    if result is None or result.component != component or result.status != "ok":
        return None
    return result


def _build_item(session: PriceSearchSession, planned: PlannedItem) -> dict[str, Any]:
    results = {
        "parts": _lookup(session, planned.parts_query_id, "parts"),
        "labor": _lookup(session, planned.labor_query_id, "labor"),
        "paint": _lookup(session, planned.paint_query_id, "paint"),
    }
    expected = {"labor"} | ({"parts"} if planned.operation == "replace" else set()) | (
        {"paint"} if planned.operation == "repaint" else set()
    )
    missing = sorted(k for k in expected if results[k] is None)
    found = [v for v in results.values() if v is not None]

    def rng(key: str) -> Range | None:
        v = results[key]
        return Range(v.low, v.high) if v is not None else None

    ranges = {k: rng(k) for k in results}
    subtotal = None
    if found:
        subtotal = sum((r for r in ranges.values() if r is not None), ZERO)
    confidence = "none"
    if found:
        confidence = min((v.confidence for v in found), key=lambda c: _CONFIDENCE_ORDER[c])
    sources = []
    for v in found:
        sources.extend(s.__dict__ for s in v.sources[:2])
    item = {
        "part": planned.part,
        "material": planned.material,
        "operation": planned.operation,
        "parts": _range_dict(ranges["parts"]),
        "labor": _range_dict(ranges["labor"]),
        "paint": _range_dict(ranges["paint"]),
        "subtotal": _range_dict(subtotal),
        "estimable": subtotal is not None,
        "confidence": confidence,
        "missing_components": missing,
        "sources": sources,
    }
    if planned.likelihood:
        item["likelihood"] = planned.likelihood
        item["reason"] = (planned.reason or "")[:200]
    return item


def _total(items: list[dict[str, Any]]) -> dict | None:
    ranges = [Range(i["subtotal"]["low"], i["subtotal"]["high"]) for i in items if i["subtotal"]]
    return _range_dict(sum(ranges, ZERO)) if ranges else None


def build_estimates(
    plan: EstimationPlan, session: PriceSearchSession
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (cost_estimate, internal_damage_estimate) dicts, ranges only."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    visible = [_build_item(session, p) for p in plan.visible_damage]
    internal = [_build_item(session, p) for p in plan.internal_damage]
    provider = session.provider.name if session.provider else None
    common = {
        "currency": "INR",
        "region": "India",
        "provider": provider,
        "generated_at": now,
    }
    assumptions = [
        "Prices are indicative ranges gathered from web search results, not quotes.",
        "Only damage visible in the photo is priced; hidden damage is listed separately.",
        "Actual cost depends on the garage, parts availability (OEM vs aftermarket) and inspection.",
    ]
    if any(not i["estimable"] for i in visible):
        assumptions.append("Some items could not be priced from search results and are excluded from the total.")
    cost = {
        **common,
        "status": "estimated",
        "vehicle": plan.vehicle,
        "items": visible,
        "total": _total(visible),
        "assumptions": assumptions,
        "disclaimer": DISCLAIMER,
    }
    internal_estimate = {
        **common,
        "status": "estimated",
        "items": internal,
        "total": _total(internal),
        "included_in_claimable": False,
        "note": INTERNAL_NOTE,
    }
    return cost, internal_estimate


def unavailable_estimate(reason: str) -> dict[str, Any]:
    return {"status": "unavailable", "reason": reason, "disclaimer": DISCLAIMER}


def line_items_from_estimate(cost_estimate: dict[str, Any]) -> list[LineItem]:
    """Rebuild LineItems from a stored estimate so policy changes can be recomputed without the model."""
    items = []
    for i in cost_estimate.get("items", []):
        if not i.get("estimable"):
            continue

        def rng(key: str) -> Range:
            v = i.get(key)
            return Range(v["low"], v["high"]) if v else ZERO

        items.append(LineItem(i["part"], i["material"], i["operation"], rng("parts"), rng("labor"), rng("paint")))
    return items


def compute_breakdown(cost_estimate: dict[str, Any], policy: PolicyInput) -> ClaimBreakdown | None:
    if cost_estimate.get("status") != "estimated":
        return None
    items = line_items_from_estimate(cost_estimate)
    if not items:
        return None
    return compute_claimable(items, policy)


def default_breakdown_dict(cost_estimate: dict[str, Any]) -> dict[str, Any] | None:
    breakdown = compute_breakdown(cost_estimate, PolicyInput())
    if breakdown is None:
        return None
    data = breakdown.to_dict()
    data["assumptions"].append(
        "Vehicle age, engine size and IDV were not provided, so defaults were used "
        "(vehicle treated as new, up to 1500cc, no IDV cap). Provide policy details for a closer range."
    )
    return data


def _fmt(r: dict | None) -> str:
    return "not estimable" if not r else f"₹{r['low']:,} - ₹{r['high']:,}"


def render_estimate_markdown(
    cost: dict[str, Any], internal: dict[str, Any] | None, breakdown: dict[str, Any] | None
) -> str:
    """Markdown block appended to the PDF (the UI renders the structured fields instead)."""
    if cost.get("status") != "estimated":
        return (
            "## Estimated Repair Cost (Indicative Range)\n"
            f"A cost estimate is not available for this claim: {cost.get('reason', 'unknown reason')}.\n"
            f"{DISCLAIMER}\n"
        )
    lines = ["## Estimated Repair Cost (Indicative Range)", f"Vehicle (estimate): {cost.get('vehicle', 'unknown')}"]
    for i in cost["items"]:
        lines.append(f"- {i['part']} ({i['operation']}): {_fmt(i['subtotal'])} [confidence: {i['confidence']}]")
    lines.append(f"Total visible damage: {_fmt(cost['total'])}")
    if breakdown:
        lines += [
            "",
            "## Estimated Claimable Amount (Indicative Range)",
            f"Under the '{breakdown['template']}' standard template: {_fmt(breakdown['claimable'])}",
            f"Deductible applied: ₹{breakdown['deductible_inr']:,.0f}",
        ]
        lines += [f"- {a}" for a in breakdown["assumptions"]]
    if internal and internal.get("items"):
        lines += ["", "## Possible Internal Damage (Not Confirmed)", internal["note"]]
        for i in internal["items"]:
            lines.append(
                f"- {i['part']} (likelihood: {i.get('likelihood', 'unknown')}): {_fmt(i['subtotal'])} if confirmed"
            )
        lines.append(f"Additional if all confirmed: {_fmt(internal['total'])}")
    lines += ["", "## Estimate Assumptions", *[f"- {a}" for a in cost["assumptions"]]]
    sources = {s["url"] for i in cost["items"] for s in i["sources"] if s.get("url")}
    if sources:
        lines.append(f"Price data retrieved {cost['generated_at'][:10]} from {len(sources)} web sources via {cost['provider']}.")
    lines += ["", DISCLAIMER]
    return "\n".join(lines)


def parse_failure_message(exc: Exception) -> str:
    if isinstance(exc, (ValidationError, ValueError)):
        return "The pricing plan could not be interpreted."
    return "Pricing research failed."
