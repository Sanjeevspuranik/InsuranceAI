"""
main.py
CLI entry point for InsuranceAI project.
"""

import argparse
import os
from insuranceai.tools import get_insurance_tools
from insuranceai.agent import InsuranceAgent
from langchain_openai import ChatOpenAI
from insuranceai.utils import (
    predict_image,
    generate_damage_report,
    create_damage_pdf,
    visualize_segments,
    yolo_model,
)


def run_pipeline(img_path: str):
    """
    Run InsuranceAI pipeline on uploaded image.
    """
    # Step 1: Run YOLO prediction (compact dict for GPT)
    results_dict = predict_image(model=yolo_model, img_path=img_path)

    # Step 2: Generate GPT damage report
    report_text = generate_damage_report(results_dict)

    # Step 3: Create segmentation visualization in dedicated folder
    output_dir = "segmented_outputs"
    os.makedirs(output_dir, exist_ok=True)
    vis_path = os.path.join(output_dir, "segmented_damage.jpg")
    visualize_segments(img_path=img_path, output_path=vis_path)

    # Step 4: Generate PDF including report + visualization
    pdf_path = create_damage_pdf(report_text,
                                 images_dir=output_dir,
                                 output_path="damage_report.pdf")

    return report_text, pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run InsuranceAI pipeline")
    parser.add_argument("img_path", type=str, help="Path to car image")
    args = parser.parse_args()

    report_text, pdf_path = run_pipeline(args.img_path)

    print("=== InsuranceAI Report ===")
    print(report_text)
    print(f"\nPDF saved at: {pdf_path}")
