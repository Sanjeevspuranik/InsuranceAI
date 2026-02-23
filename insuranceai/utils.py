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
from fpdf import FPDF
from openai import OpenAI
from ultralytics import YOLO
from insuranceai.prompts import (
    DAMAGE_REPORT_PROMPT,
    EXECUTIVE_SUMMARY_PROMPT,
    VISUALIZATION_PROMPT,
    WORKFLOW_EXPLANATION_PROMPT,
    COST_ESTIMATION_PROMPT,
)

# ==========================================================
# 1. YOLO Model Loading
# ==========================================================

yolo_model = YOLO("model/CarDD_FN_best.pt")

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
# 3. GPT-Based Damage Report (compact input only)
# ==========================================================


def generate_damage_report(json_obj, model="gpt-4o-mini"):
    client = OpenAI()

    # Summarize predictions for GPT (avoid huge arrays)
    compact_preds = [
        {"label": p["label"], "confidence": p["confidence"]}
        for p in json_obj.get("predictions", [])
    ]

    prompt = DAMAGE_REPORT_PROMPT.format(
        json_obj=json.dumps(compact_preds, indent=2)
    )

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4
    )
    return response.choices[0].message.content

# ==========================================================
# 4. PDF Report Generator
# ==========================================================


def create_damage_pdf(report_text, images_dir="segmented_classes", output_path="damage_report.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, report_text)

    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Segmentation Images", ln=True)

    if os.path.exists(images_dir):
        for img_file in os.listdir(images_dir):
            if img_file.lower().endswith((".jpg", ".png")):
                pdf.add_page()
                pdf.cell(0, 10, img_file, ln=True)
                pdf.image(os.path.join(images_dir, img_file),
                          x=10, y=30, w=180)

    pdf.output(output_path)
    return output_path

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

        if r.masks is None:
            continue

        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            color = class_colors.get(cls_id, (255, 255, 255))

            # Get mask and resize to original image size
            mask = r.masks.data[i].cpu().numpy()
            mask = (mask * 255).astype(np.uint8)
            mask_resized = cv2.resize(
                mask, (w, h), interpolation=cv2.INTER_NEAREST)

            # Create colored overlay
            colored_mask = np.zeros_like(canvas)
            for c in range(3):
                colored_mask[:, :, c] = (mask_resized > 0) * color[c]

            # Blend mask into canvas
            canvas = cv2.addWeighted(canvas, 1, colored_mask, 0.5, 0)

            # Compute centroid for label placement
            ys, xs = np.where(mask_resized > 0)
            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                cv2.putText(canvas, label, (cx, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        cv2.imwrite(output_path, canvas)
        print(f"Saved segmentation visualization: {output_path}")

    return output_path
