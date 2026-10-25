"""
main.py
CLI entry point for InsuranceAI project.
"""

import argparse
from insuranceai.workflow import run_claim_pipeline


def run_pipeline(img_path: str):
    report_text, _, pdf_path = run_claim_pipeline(img_path)
    return report_text, pdf_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run InsuranceAI pipeline")
    parser.add_argument("img_path", type=str, help="Path to car image")
    args = parser.parse_args()

    report_text, pdf_path = run_pipeline(args.img_path)

    print("=== InsuranceAI Report ===")
    print(report_text)
    print(f"\nPDF saved at: {pdf_path}")
