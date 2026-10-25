"""
interface.py
Gradio interface for InsuranceAI project.
"""

import gradio as gr
from insuranceai.workflow import run_claim_pipeline


def run_pipeline(img_path: str, progress=gr.Progress()):
    return run_claim_pipeline(
        img_path,
        progress_callback=lambda value, description: progress(
            value, desc=description
        ),
    )


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🚗 InsuranceAI — Car Damage Assessment")

    img_input = gr.Image(type="filepath", label="Upload Car Image")
    run_btn = gr.Button("Run Analysis")

    rationale_output = gr.Textbox(
        label="Assessment rationale (brief)",
        interactive=False,
        lines=4,
    )
    report_output = gr.Markdown()
    pdf_output = gr.File(label="Download PDF Report")

    run_btn.click(fn=run_pipeline,
                  inputs=img_input,
                  outputs=[report_output, rationale_output, pdf_output])

if __name__ == "__main__":
    demo.launch()
