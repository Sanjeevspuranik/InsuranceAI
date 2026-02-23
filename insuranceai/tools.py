"""
tools.py
Wraps InsuranceAI functions into LangChain tools using @tool decorator.
"""

from langchain_core.tools import tool
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
    """Generate a professional insurance damage report using GPT-4o-mini."""
    if json_obj is None and img_path:
        # Run YOLO again if no JSON provided
        json_obj = predict_image(model=yolo_model, img_path=img_path)
    elif json_obj is None:
        raise ValueError("Either json_obj or img_path must be provided.")
    return generate_damage_report(json_obj)

# Tool: PDF Generator


@tool("create_damage_pdf", return_direct=False)
def create_damage_pdf_tool(report_text: str, images_dir: str = "segmented_classes", output_path: str = "damage_report.pdf"):
    """Create a PDF including the damage report and segmentation images."""
    return create_damage_pdf(report_text, images_dir, output_path)

# Tool: Visualization


@tool("visualize_segments", return_direct=False)
def visualize_segments_tool(img_path: str):
    """Overlay segmentation masks and labels on the original image."""
    return visualize_segments(img_path=img_path)

# Collect all tools in a list


def get_insurance_tools():
    return [
        predict_image_tool,
        generate_damage_report_tool,
        create_damage_pdf_tool,
        visualize_segments_tool,
    ]
