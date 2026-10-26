"""Deterministic repair-cost and claimable-amount calculator (India motor).

All outputs are (low, high) ranges. No LLM arithmetic is involved: the agent
supplies price ranges and this module applies depreciation, deductibles,
co-pay and IDV rules from rules/india_motor.json.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

Material = Literal["metal", "rubber_plastic", "fibreglass", "glass"]
Operation = Literal["repair", "replace", "repaint"]

RULES_PATH = Path(__file__).parent / "rules" / "india_motor.json"
DISCLAIMER = (
    "Indicative estimate only, not a settlement or approval. Final amounts depend "
    "on physical inspection, your actual policy terms and insurer review."
)


@dataclass(frozen=True)
class Range:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low < 0 or self.high < self.low:
            raise ValueError(f"Invalid range: {self.low}..{self.high}")

    def scale(self, factor: float) -> "Range":
        return Range(self.low * factor, self.high * factor)

    def __add__(self, other: "Range") -> "Range":
        return Range(self.low + other.low, self.high + other.high)


ZERO = Range(0.0, 0.0)


@dataclass(frozen=True)
class LineItem:
    part: str
    material: Material
    operation: Operation
    parts_cost: Range = ZERO
    labor_cost: Range = ZERO
    paint_cost: Range = ZERO


@dataclass(frozen=True)
class PolicyInput:
    template: str = "standard_comprehensive"
    idv_inr: float | None = None
    vehicle_age_months: int = 0
    engine_cc: int = 1200
    voluntary_deductible_inr: float = 0.0
    copay_pct: float | None = None
    consumables_addon: bool = False

    def __post_init__(self) -> None:
        if self.idv_inr is not None and self.idv_inr <= 0:
            raise ValueError("idv_inr must be positive")
        if self.vehicle_age_months < 0:
            raise ValueError("vehicle_age_months must be >= 0")
        if self.engine_cc <= 0:
            raise ValueError("engine_cc must be positive")
        if self.voluntary_deductible_inr < 0:
            raise ValueError("voluntary_deductible_inr must be >= 0")
        if self.copay_pct is not None and not 0 <= self.copay_pct <= 100:
            raise ValueError("copay_pct must be between 0 and 100")


@dataclass
class ClaimBreakdown:
    template: str
    covered: bool
    gross_repair: Range
    depreciation: Range
    deductible_inr: float
    copay_pct: float
    claimable: Range
    possible_total_loss: bool = False
    assumptions: list[str] = field(default_factory=list)
    disclaimer: str = DISCLAIMER

    def to_dict(self) -> dict:
        def r(x: Range) -> dict[str, float]:
            return {"low": round(x.low, 2), "high": round(x.high, 2)}

        return {
            "template": self.template,
            "covered": self.covered,
            "gross_repair": r(self.gross_repair),
            "depreciation": r(self.depreciation),
            "deductible_inr": self.deductible_inr,
            "copay_pct": self.copay_pct,
            "claimable": r(self.claimable),
            "possible_total_loss": self.possible_total_loss,
            "assumptions": self.assumptions,
            "disclaimer": self.disclaimer,
        }


@lru_cache(maxsize=1)
def load_rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def list_policy_templates() -> dict[str, dict]:
    return load_rules()["policy_templates"]


def metal_depreciation_rate(age_months: int) -> float:
    for band in load_rules()["metal_depreciation_by_age_months"]:
        if band["max_months"] is None or age_months <= band["max_months"]:
            return band["rate"]
    raise AssertionError("depreciation bands must end with an open-ended band")


def depreciation_rate(material: Material, age_months: int) -> float:
    if material == "metal":
        return metal_depreciation_rate(age_months)
    return load_rules()["flat_depreciation"][material]


def compulsory_deductible(engine_cc: int) -> float:
    table = load_rules()["compulsory_deductible_inr"]
    return float(table["up_to_1500cc"] if engine_cc <= 1500 else table["above_1500cc"])


def compute_claimable(items: list[LineItem], policy: PolicyInput) -> ClaimBreakdown:
    rules = load_rules()
    templates = rules["policy_templates"]
    if policy.template not in templates:
        raise ValueError(f"Unknown policy template: {policy.template}")
    template = templates[policy.template]

    assumptions = [
        f"Policy template '{template['label']}' is a standard template, not your actual policy.",
        "Prices are indicative ranges from web search and are not quotes.",
    ]

    gross = ZERO
    depreciation = ZERO
    for item in items:
        gross = gross + item.parts_cost + item.labor_cost + item.paint_cost
        if template["zero_depreciation"]:
            continue
        # Depreciation applies to replaced parts and paint material, never to labor.
        if item.operation == "replace":
            depreciation = depreciation + item.parts_cost.scale(
                depreciation_rate(item.material, policy.vehicle_age_months)
            )
        if item.paint_cost.high > 0:
            depreciation = depreciation + item.paint_cost.scale(
                rules["flat_depreciation"]["paint"]
            )

    if template["zero_depreciation"]:
        assumptions.append("Zero-depreciation add-on assumed: no parts depreciation applied.")
    if not policy.consumables_addon:
        assumptions.append("Consumables (oil, nuts, bolts etc.) are not included.")

    deductible = compulsory_deductible(policy.engine_cc) + policy.voluntary_deductible_inr
    copay = policy.copay_pct if policy.copay_pct is not None else template["default_copay_pct"]

    if not template["own_damage"]:
        assumptions.append("This template has no own-damage cover, so repair costs are not claimable.")
        return ClaimBreakdown(policy.template, False, gross, depreciation, deductible, copay, ZERO, False, assumptions)

    def claim(gross_v: float, dep_v: float) -> float:
        net = max(gross_v - dep_v - deductible, 0.0) * (1 - copay / 100)
        return min(net, policy.idv_inr) if policy.idv_inr else net

    claimable = Range(claim(gross.low, depreciation.low), claim(gross.high, depreciation.high))
    if policy.idv_inr is None:
        assumptions.append("IDV not provided, so no sum-insured cap or total-loss check was applied.")

    total_loss = bool(
        policy.idv_inr and gross.high >= policy.idv_inr * rules["total_loss_threshold_of_idv"]
    )
    if total_loss:
        assumptions.append("Repair cost may approach the total-loss threshold; the insurer may assess a total loss.")
    if copay:
        assumptions.append(f"{copay:g}% co-pay applied.")

    return ClaimBreakdown(
        policy.template, True, gross, depreciation, deductible, copay, claimable, total_loss, assumptions
    )
