"""Shared image-to-report workflow for the Gradio UI and command-line entry point."""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional, Tuple
from uuid import uuid4

from insuranceai.agent import InsuranceAgent
from insuranceai.prompts import SYSTEM_PROMPT
from insuranceai.tools import get_insurance_tools
from insuranceai.utils import (
    build_damage_report_message,
    extract_assessment_rationale,
    extract_reasoning_token_count,
    get_chat_model,
)


@dataclass(frozen=True)
class ClaimPipelineResult:
    report_text: str
    rationale: str
    pdf_path: str
    reasoning_tokens: Optional[int]


@lru_cache(maxsize=1)
def _get_report_agent() -> InsuranceAgent:
    return InsuranceAgent(
        model=get_chat_model(),
        tools=[],
        system_prompt=SYSTEM_PROMPT,
        log_tools=False,
    )


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

    report_progress(0.75, "Creating the annotated image...")
    visualization_path = str(run_dir / "segmented_damage.jpg")
    tools["visualize_segments"].invoke(
        {"img_path": img_path, "output_path": visualization_path}
    )

    report_progress(0.9, "Preparing the PDF report...")
    pdf_path = str(run_dir / "damage_report.pdf")
    tools["create_damage_pdf"].invoke(
        {
            "report_text": report_text,
            "images_dir": str(run_dir),
            "output_path": pdf_path,
        }
    )
    report_progress(1.0, "Analysis complete.")
    return ClaimPipelineResult(
        report_text=report_text,
        rationale=rationale,
        pdf_path=pdf_path,
        reasoning_tokens=extract_reasoning_token_count(response),
    )
