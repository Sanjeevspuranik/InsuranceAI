"""
your_module.py

Core utility functions powering the InsuranceAI workflow.

This module provides:
1. YOLO-based vehicle damage detection
2. Structured JSON extraction from detections
3. LLM-powered insurance damage report generation
4. Automated PDF report creation
5. Segmentation visualization (no bounding boxes)
"""

import cv2
import numpy as np
import os
import json
import base64
import mimetypes
import re
from functools import lru_cache
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from fpdf import FPDF
from ultralytics import YOLO
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from insuranceai.prompts import (
    DAMAGE_REPORT_PROMPT,
)

load_dotenv()

# ==========================================================
# 1. YOLO Model Loading
# ==========================================================

yolo_model = YOLO("model/CarDD_FN_best.pt")


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    """Create the configured OpenAI-compatible chat model once per process."""
    return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-6-luna"))


# ==========================================================
# 2. YOLO Prediction → Compact JSON dict
# ==========================================================


def predict_image(model=yolo_model, img_path=None):
    if img_path is None or not os.path.exists(img_path):
        raise ValueError("Valid image path must be provided.")

    results = model.predict(source=img_path)

    predictions = []
    for r in results:
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            predictions.append({
                "label": r.names[cls_id],
                "confidence": float(box.conf[0]),
                "bbox": box.xyxy[0].tolist(),
                # Instead of full mask arrays, just store a flag
                "has_mask": r.masks is not None
            })

    return {"predictions": predictions}

# ==========================================================
# 3. Multimodal damage report
# ==========================================================


def encode_image_for_llm(img_path: str) -> str:
    """Encode a supported image as a data URL for a multimodal chat message."""
    image_path = Path(img_path)
    if not image_path.is_file():
        raise ValueError("A valid image file must be provided.")

    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise ValueError("Image must be JPEG, PNG, or WEBP.")

    image_bytes = image_path.read_bytes()
    if len(image_bytes) > 20 * 1024 * 1024:
        raise ValueError("Image must be 20 MB or smaller.")

    encoded_image = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded_image}"


def build_damage_report_message(
    json_obj: dict, img_path: Optional[str] = None
) -> HumanMessage:
    """Build the model request, attaching the uploaded image when available."""
    compact_preds = [
        {
            key: p[key]
            for key in ("label", "confidence", "bbox", "has_mask")
            if key in p
        }
        for p in json_obj.get("predictions", [])
    ]

    prompt = DAMAGE_REPORT_PROMPT.format(
        json_obj=json.dumps(compact_preds, indent=2)
    )

    if img_path is None:
        return HumanMessage(content=prompt)

    return HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": encode_image_for_llm(img_path)},
            },
        ]
    )


def _response_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = [
            block["text"]
            for block in content
            if isinstance(block, dict) and isinstance(block.get("text"), str)
        ]
        return "\n".join(text_parts)
    raise TypeError("The chat model returned an unsupported response format.")


def extract_reasoning_token_count(response) -> Optional[int]:
    """Return a provider-reported reasoning token count without exposing reasoning text."""
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        output_details = usage.get("output_token_details")
        if isinstance(output_details, dict):
            count = output_details.get(
                "reasoning", output_details.get("reasoning_tokens")
            )
            if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                return count

    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, dict):
            completion_details = token_usage.get("completion_tokens_details")
            if isinstance(completion_details, dict):
                count = completion_details.get("reasoning_tokens")
                if isinstance(count, int) and not isinstance(count, bool) and count > 0:
                    return count
    return None


def generate_damage_report(
    json_obj: dict,
    model: Optional[BaseChatModel] = None,
    img_path: Optional[str] = None,
) -> str:
    """Generate a report using detection data and, optionally, the uploaded image."""
    response = (model or get_chat_model()).invoke(
        [build_damage_report_message(json_obj, img_path)]
    )
    return _response_text(response.content)


def extract_assessment_rationale(report_text: str) -> str:
    """Extract the concise, user-facing rationale section from a generated report."""
    match = re.search(
        r"(?ims)^\s{0,3}#{0,6}\s*Assessment Rationale(?:\s*\(Brief\))?\s*:?\s*$"
        r"\s*(.*?)"
        r"(?=^\s{0,3}#{1,6}\s+|\Z)",
        report_text,
    )
    return match.group(1).strip() if match else ""

# ==========================================================
# 4. PDF Report Generator
# ==========================================================


class _DamageReportPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.set_margins(16, 24, 16)
        self.set_auto_page_break(auto=True, margin=18)
        self.alias_nb_pages()

    def header(self):
        self.set_y(10)
        self.set_font("Arial", "B", 9)
        self.set_text_color(46, 69, 92)
        self.cell(0, 7, "INSURANCEAI  |  DAMAGE ASSESSMENT", ln=True)
        self.set_draw_color(196, 207, 218)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_y(28)

    def footer(self):
        self.set_y(-14)
        self.set_font("Arial", "I", 8)
        self.set_text_color(100, 110, 120)
        self.cell(120, 5, "AI-assisted preliminary report - human review required")
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="R")


def _pdf_text(text: str) -> str:
    replacements = {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2022": "-",
        "\u00a0": " ",
    }
    text = text.translate(str.maketrans(replacements))
    return text.encode("latin-1", "replace").decode("latin-1")


def _plain_markdown(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return _pdf_text(text.strip())


def _write_pdf_heading(pdf: _DamageReportPDF, text: str, level: int) -> None:
    heading = _plain_markdown(text)
    if pdf.get_y() + 28 > pdf.page_break_trigger:
        pdf.add_page()

    if level == 1:
        pdf.ln(2)
        pdf.set_font("Arial", "B", 19)
        pdf.set_text_color(28, 54, 80)
        pdf.multi_cell(0, 10, heading)
        pdf.ln(2)
    elif level == 2:
        pdf.ln(3)
        pdf.set_font("Arial", "B", 12)
        pdf.set_text_color(35, 79, 120)
        pdf.multi_cell(0, 8, heading)
        pdf.set_draw_color(210, 220, 230)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(2)
    else:
        pdf.ln(2)
        pdf.set_font("Arial", "B", 10)
        pdf.set_text_color(55, 70, 85)
        pdf.multi_cell(0, 6, heading)


def _write_pdf_paragraph(pdf: _DamageReportPDF, text: str, bullet: str = "") -> None:
    clean_text = _plain_markdown(text)
    if not clean_text:
        return

    pdf.set_font("Arial", size=10)
    pdf.set_text_color(35, 40, 45)
    if bullet:
        indent = min(len(bullet) * 2, 8)
        pdf.set_x(pdf.l_margin + indent)
        pdf.multi_cell(pdf.w - pdf.r_margin - pdf.get_x(), 5.5, f"{bullet} {clean_text}")
    else:
        pdf.multi_cell(0, 5.5, clean_text)
    pdf.ln(1)


def _write_report_markdown(pdf: _DamageReportPDF, report_text: str) -> None:
    paragraph = []

    def flush_paragraph() -> None:
        if paragraph:
            _write_pdf_paragraph(pdf, " ".join(line.strip() for line in paragraph))
            paragraph.clear()

    for raw_line in report_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue

        heading_match = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading_match:
            flush_paragraph()
            _write_pdf_heading(pdf, heading_match.group(2), len(heading_match.group(1)))
            continue

        if re.fullmatch(r"[-*_]{3,}", line):
            flush_paragraph()
            continue

        list_match = re.match(r"^(\s*)([-*+]|\d+[.)])\s+(.+)$", raw_line)
        if list_match:
            flush_paragraph()
            nesting = min(len(list_match.group(1)) // 2, 2)
            marker = list_match.group(2) if list_match.group(2)[0].isdigit() else "-"
            _write_pdf_paragraph(
                pdf,
                list_match.group(3),
                bullet=f"{marker}" if nesting == 0 else f"  {marker}",
            )
            continue

        if line.startswith(">"):
            line = line[1:].strip()
        if line.startswith("|") and line.endswith("|"):
            line = line.strip("|").replace("|", "  |  ")
            if re.fullmatch(r"[\s|:-]+", line):
                continue
        paragraph.append(line)

    flush_paragraph()


def create_damage_pdf(report_text, images_dir="segmented_classes", output_path="damage_report.pdf"):
    if not isinstance(report_text, str) or not report_text.strip():
        raise ValueError("Report text must not be empty.")

    pdf = _DamageReportPDF()
    pdf.set_title("InsuranceAI Vehicle Damage Assessment")
    pdf.set_author("InsuranceAI")
    pdf.set_subject("AI-assisted vehicle damage assessment")
    pdf.add_page()
    pdf.set_font("Arial", "B", 9)
    pdf.set_text_color(100, 110, 120)
    pdf.cell(0, 6, f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True)
    _write_report_markdown(pdf, report_text)

    image_paths = []
    if images_dir and Path(images_dir).is_dir():
        image_paths = sorted(
            path for path in Path(images_dir).iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    if image_paths:
        pdf.add_page()
        _write_pdf_heading(pdf, "Image Evidence", 2)
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not read report image: {image_path}")

            if pdf.get_y() + 18 > pdf.page_break_trigger:
                pdf.add_page()
            pdf.set_font("Arial", "B", 9)
            pdf.set_text_color(55, 70, 85)
            pdf.multi_cell(0, 6, _pdf_text(image_path.name))

            image_height_px, image_width_px = image.shape[:2]
            available_width = pdf.w - pdf.l_margin - pdf.r_margin
            available_height = pdf.page_break_trigger - pdf.get_y() - 2
            scale = min(
                available_width / image_width_px,
                available_height / image_height_px,
            )
            display_width = image_width_px * scale
            display_height = image_height_px * scale
            if pdf.get_y() + display_height > pdf.page_break_trigger:
                pdf.add_page()
            pdf.image(
                str(image_path),
                x=pdf.l_margin + (available_width - display_width) / 2,
                y=pdf.get_y(),
                w=display_width,
                h=display_height,
            )
            pdf.set_y(pdf.get_y() + display_height + 5)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_file))
    return str(output_file)

# ==========================================================
# 5. Visualization (YOLO masks → overlay image)
# ==========================================================


def visualize_segments(img_path, output_path="segmented_damage.jpg"):
    """
    Overlay segmentation masks from YOLO Results on the original image
    with distinct colors and labels (no bounding boxes).

    Args:
        img_path (str): Path to the original image
        output_path (str): Path to save the visualization image

    Returns:
        str: Path to the saved visualization image
    """
    results = yolo_model.predict(source=img_path)

    # Define distinct colors for classes (BGR format)
    class_colors = {
        0: (0, 0, 255),    # dent - red
        1: (255, 0, 0),    # scratch - blue
        2: (0, 255, 255),  # crack - yellow
        3: (0, 255, 0),    # glass shatter - green
        4: (255, 0, 255),  # lamp broken - magenta
        5: (255, 255, 0)   # tire flat - cyan
    }

    for r in results:
        h, w, _ = r.orig_img.shape
        canvas = r.orig_img.copy()

        if r.masks is not None:
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls[0])
                label = r.names[cls_id]
                color = class_colors.get(cls_id, (255, 255, 255))

                mask = r.masks.data[i].cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                mask_resized = cv2.resize(
                    mask, (w, h), interpolation=cv2.INTER_NEAREST)

                colored_mask = np.zeros_like(canvas)
                for c in range(3):
                    colored_mask[:, :, c] = (mask_resized > 0) * color[c]

                canvas = cv2.addWeighted(canvas, 1, colored_mask, 0.5, 0)

                ys, xs = np.where(mask_resized > 0)
                if len(xs) > 0 and len(ys) > 0:
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    cv2.putText(canvas, label, (cx, cy),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_file), canvas):
            raise OSError(f"Could not write segmentation image to {output_file}.")

    return output_path
