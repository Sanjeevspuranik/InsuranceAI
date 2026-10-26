"""
tools.py
Wraps InsuranceAI functions into LangChain tools using @tool decorator.
"""

import json
from typing import Literal

from langchain_core.tools import tool
from insuranceai.price_search import get_price_session
from insuranceai.utils import yolo_model
from insuranceai.utils import (
    predict_image,
    generate_damage_report,
    create_damage_pdf,
    visualize_segments,
)

# Tool: YOLO Prediction


@tool("predict_image")
def predict_image_tool(img_path: str):
    """Run YOLOv8-seg model to detect car damage from an image path."""
    return predict_image(model=yolo_model, img_path=img_path)

# Tool: GPT Damage Report


@tool("generate_damage_report", return_direct=False)
def generate_damage_report_tool(json_obj: dict = None, img_path: str = None):
    """Generate a GPT-6-Luna damage report from detections and an optional image."""
    if json_obj is None and img_path:
        json_obj = predict_image(model=yolo_model, img_path=img_path)
    elif json_obj is None:
        raise ValueError("Either json_obj or img_path must be provided.")
    return generate_damage_report(json_obj, img_path=img_path)

# Tool: PDF Generator


@tool("create_damage_pdf", return_direct=False)
def create_damage_pdf_tool(report_text: str, images_dir: str = "segmented_classes", output_path: str = "damage_report.pdf"):
    """Create a styled PDF from the report, including annotated evidence images."""
    return create_damage_pdf(report_text, images_dir, output_path)

# Tool: Visualization


@tool("visualize_segments", return_direct=False)
def visualize_segments_tool(img_path: str, output_path: str = "segmented_damage.jpg"):
    """Overlay segmentation masks and labels on the original image."""
    return visualize_segments(img_path=img_path, output_path=output_path)


# Tool: Repair price search (used by the estimation agent only)


@tool("search_repair_prices")
def search_repair_prices_tool(
    part: str,
    component: Literal["parts", "labor", "paint"],
    operation: Literal["repair", "replace", "repaint"] = "replace",
    vehicle: str = "",
) -> str:
    """Search the web for an indicative Indian price range (INR) for one vehicle part.

    Use component='parts' for the part price, 'labor' for labour charges and
    'paint' for painting cost. Returns a JSON range with a query_id; cite that
    query_id in your final answer. Never invent prices.
    """
    session = get_price_session()
    if session is None:
        return json.dumps({"status": "not_configured", "note": "Price search is not available."})
    return json.dumps(session.search(part, component, operation, vehicle).to_dict())


def get_pricing_tools():
    return [search_repair_prices_tool]


# Collect all tools in a list


def get_insurance_tools():
    return [
        predict_image_tool,
        generate_damage_report_tool,
        create_damage_pdf_tool,
        visualize_segments_tool,
    ]
