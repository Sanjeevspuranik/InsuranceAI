"""Shared image-to-report workflow for the Gradio UI and command-line entry point."""

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional, Tuple
from uuid import uuid4

from langchain_core.messages import HumanMessage

from insuranceai.agent import InsuranceAgent
from insuranceai.estimation import (
    build_estimates,
    default_breakdown_dict,
    parse_failure_message,
    parse_plan,
    render_estimate_markdown,
    unavailable_estimate,
)
from insuranceai.price_search import PriceSearchSession, reset_price_session, set_price_session
from insuranceai.prompts import ESTIMATION_PROMPT, SYSTEM_PROMPT
from insuranceai.tools import get_insurance_tools, get_pricing_tools
from insuranceai.utils import (
    build_damage_report_message,
    encode_image_for_llm,
    extract_assessment_rationale,
    extract_reasoning_token_count,
    get_chat_model,
    _response_text,
)

logger = logging.getLogger(__name__)
ESTIMATION_RECURSION_LIMIT = 40


@dataclass(frozen=True)
class ClaimPipelineResult:
    report_text: str
    rationale: str
    pdf_path: str
    reasoning_tokens: Optional[int]
    cost_estimate: Optional[dict[str, Any]] = None
    internal_damage_estimate: Optional[dict[str, Any]] = None
    claim_breakdown: Optional[dict[str, Any]] = None


@lru_cache(maxsize=1)
def _get_report_agent() -> InsuranceAgent:
    return InsuranceAgent(
        model=get_chat_model(),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        log_tools=False,
    )


@lru_cache(maxsize=1)
def _get_estimation_agent() -> InsuranceAgent:
    return InsuranceAgent(
        model=get_chat_model(),
        tools=get_pricing_tools(),
        system_prompt="You are a careful repair-cost research assistant. Never invent prices.",
        log_tools=False,
    )


def _run_estimation(
    detections: dict,
    report_text: str,
    img_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
):
    """Run the tool-using price research agent; never raises, so claims still complete."""

    def on_result(current: PriceSearchSession, result) -> None:
        if progress_callback is None:
            return
        # Searches advance progress from 60% to 72% of the whole pipeline.
        share = min(len(current.results) / max(current.max_calls, 1), 1.0)
        progress_callback(
            0.6 + 0.12 * share,
            f"Researching repair prices: {result.part} ({result.component}), "
            f"search {len(current.results)} of up to {current.max_calls}...",
        )

    session = PriceSearchSession(on_result=on_result)
    if not session.configured:
        logger.info("estimation.skipped reason=price_search_not_configured")
        return unavailable_estimate("Price search is not configured."), None, None, None

    compact = [
        {k: p[k] for k in ("label", "confidence", "bbox") if k in p}
        for p in detections.get("predictions", [])
    ]
    prompt = (
        ESTIMATION_PROMPT.replace("{json_obj}", json.dumps(compact))
        .replace("{report_text}", report_text)
        .replace("{max_searches}", str(session.max_calls))
    )
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": encode_image_for_llm(img_path)}},
        ]
    )
    token = set_price_session(session)
    try:
        response = _get_estimation_agent().invoke([message], recursion_limit=ESTIMATION_RECURSION_LIMIT)
        content = response.content
        text = _response_text(content)
        plan = parse_plan(text)
        cost, internal = build_estimates(plan, session)
        breakdown = default_breakdown_dict(cost)
        logger.info(
            "estimation.completed searches=%d visible_items=%d internal_items=%d",
            session.calls, len(cost["items"]), len(internal["items"]),
        )
        return cost, internal, breakdown, extract_reasoning_token_count(response)
    except Exception as exc:
        logger.exception("estimation.failed")
        return unavailable_estimate(parse_failure_message(exc)), None, None, None
    finally:
        reset_price_session(token)


def run_claim_pipeline(
    img_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Tuple[str, str, str]:
    """Run the claim pipeline, preserving its original three-value return contract."""
    result = run_claim_pipeline_with_usage(img_path, progress_callback)
    return result.report_text, result.rationale, result.pdf_path


def run_claim_pipeline_with_usage(
    img_path: str,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> ClaimPipelineResult:
    """Run the claim pipeline and capture provider-reported reasoning token usage."""
    if not img_path or not Path(img_path).is_file():
        raise ValueError("Please upload a valid vehicle image.")

    tools = {tool.name: tool for tool in get_insurance_tools()}
    run_dir = Path("segmented_outputs") / uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=True)

    def report_progress(progress: float, description: str) -> None:
        if progress_callback is not None:
            progress_callback(progress, description)

    report_progress(0.1, "Detecting vehicle damage...")
    detections = tools["predict_image"].invoke({"img_path": img_path})

    report_progress(0.45, "Generating the image-based assessment...")
    response = _get_report_agent().invoke(
        [build_damage_report_message(detections, img_path)]
    )
    report_text = response.content
    if not isinstance(report_text, str):
        raise TypeError("The model returned a report in an unsupported format.")
    rationale = extract_assessment_rationale(report_text)
    if not rationale:
        rationale = "The model did not return a separate assessment rationale."

    report_progress(0.6, "Researching indicative repair prices...")
    cost_estimate, internal_estimate, breakdown, estimation_tokens = _run_estimation(
        detections, report_text, img_path, report_progress
    )
    pdf_report_text = f"{report_text}\n\n{render_estimate_markdown(cost_estimate, internal_estimate, breakdown)}"

    report_progress(0.75, "Creating the annotated image...")
    visualization_path = str(run_dir / "segmented_damage.jpg")
    tools["visualize_segments"].invoke(
        {"img_path": img_path, "output_path": visualization_path}
    )

    report_progress(0.9, "Preparing the PDF report...")
    pdf_path = str(run_dir / "damage_report.pdf")
    tools["create_damage_pdf"].invoke(
        {
            "report_text": pdf_report_text,
            "images_dir": str(run_dir),
            "output_path": pdf_path,
        }
    )
    report_progress(1.0, "Analysis complete.")
    report_tokens = extract_reasoning_token_count(response)
    token_parts = [t for t in (report_tokens, estimation_tokens) if t]
    return ClaimPipelineResult(
        report_text=report_text,
        rationale=rationale,
        pdf_path=pdf_path,
        reasoning_tokens=sum(token_parts) if token_parts else None,
        cost_estimate=cost_estimate,
        internal_damage_estimate=internal_estimate,
        claim_breakdown=breakdown,
    )
