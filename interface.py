"""
interface.py
Gradio interface for InsuranceAI project.
"""

import gradio as gr
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

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Load tools
tools = get_insurance_tools()

# Create agent (LangGraph orchestration)
agent = InsuranceAgent(model=llm, tools=tools,
                       system_prompt="You are InsuranceAI.")


def run_pipeline(img_path: str):
    """
    Run InsuranceAI pipeline on uploaded image.
    """
    # Step 1: Run YOLO prediction (dict summary for GPT)
    results_dict = predict_image(model=yolo_model, img_path=img_path)

    # Step 2: Generate GPT damage report (compact input only)
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


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🚗 InsuranceAI — Car Damage Assessment")

    with gr.Row():
        img_input = gr.Image(type="filepath", label="Upload Car Image")
        report_output = gr.Markdown(label="Generated Report")
        pdf_output = gr.File(label="Download PDF Report")

    run_btn = gr.Button("Run Analysis")

    run_btn.click(fn=run_pipeline,
                  inputs=img_input,
                  outputs=[report_output, pdf_output])

if __name__ == "__main__":
    demo.launch()
