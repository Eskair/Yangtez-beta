# app.py
# -*- coding: utf-8 -*-

from pathlib import Path
import gradio as gr

from run_review import run_review


def extract_report_and_json(result: dict):
    report_path = None
    review_json_path = None

    for _, v in result.items():
        if isinstance(v, Path):
            v = str(v)
        if not isinstance(v, str):
            continue

        if v.endswith(".md") and report_path is None:
            report_path = v

        if v.endswith(".json") and "review" in Path(v).name.lower() and review_json_path is None:
            review_json_path = v

    return report_path, review_json_path


def auto_run_review(file_obj):
    if file_obj is None:
        return "Upload a PDF to generate the formal review report."

    file_path = Path(file_obj)
    if file_path.suffix.lower() != ".pdf":
        raise gr.Error("Only PDF files are supported.")

    yield "⏳ **Yangtze AI is analyzing the proposal...**"

    result = run_review(file_path=file_path, use_ocr=True)
    report_path, _ = extract_report_and_json(result)

    report_text = "Report not found."
    if report_path and Path(report_path).exists():
        report_text = Path(report_path).read_text(encoding="utf-8", errors="ignore")

    yield report_text


CSS = """
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding-top: 12px !important;
    padding-left: 10px !important;
    padding-right: 10px !important;
}

body {
    background: #09111f;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* ===== Hero ===== */
.hero {
    border: 1px solid #23314e;
    border-radius: 20px;
    padding: 26px 28px 22px 28px;
    background: linear-gradient(180deg, #0d1730 0%, #09111f 100%);
    margin-bottom: 22px;
}

.hero-title {
    font-size: 36px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
    line-height: 1.08;
}

.hero-sub {
    font-size: 15px;
    color: #dbe4f0;
    line-height: 1.7;
}

/* ===== Main symmetric layout ===== */
#main-panels {
    gap: 18px !important;
}

.panel-outer {
    border: 1px solid rgba(20, 31, 49, 0.65) !important;
    border-radius: 22px !important;
    background: #556279 !important;
    padding: 18px !important;
    min-height: 560px !important;
    box-shadow: none !important;
}

.panel-inner {
    border: 1px solid rgba(30, 41, 59, 0.8) !important;
    border-radius: 20px !important;
    background: rgba(85, 98, 121, 0.95) !important;
    padding: 18px !important;
    min-height: 520px !important;
    box-shadow: none !important;
}

.panel-heading {
    height: 64px;
    display: flex;
    align-items: flex-start;
    font-size: 20px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 12px;
}

/* This row is the actual one-box content area */
.content-slot {
    min-height: 390px !important;
}

/* Upload side */
.upload-zone {
    min-height: 390px !important;
}

.upload-zone > .wrap {
    min-height: 390px !important;
}

.upload-zone .file-upload,
.upload-zone .file-upload-secondary {
    min-height: 390px !important;
}

.upload-zone {
    border: 2px dashed rgba(255,255,255,0.88) !important;
    border-radius: 0 !important;
    background: #23324a !important;
}

/* Report side */
.report-zone {
    min-height: 390px !important;
    border-radius: 22px !important;
    background: #061225 !important;
    padding: 18px 20px !important;
    overflow-y: auto !important;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
}

.report-zone,
.report-zone > div {
    border: none !important;
}

.report-zone .prose,
.report-zone .markdown {
    color: #f8fafc !important;
    line-height: 1.8 !important;
    font-size: 15px !important;
    max-width: 100% !important;
}

.report-zone h1,
.report-zone h2,
.report-zone h3,
.report-zone h4,
.report-zone p,
.report-zone li,
.report-zone strong {
    color: #f8fafc !important;
}

/* Mobile */
@media (max-width: 980px) {
    .hero-title {
        font-size: 30px;
    }

    .panel-outer,
    .panel-inner {
        min-height: auto !important;
    }

    .panel-heading {
        height: auto;
        min-height: 48px;
    }

    .content-slot,
    .upload-zone,
    .upload-zone > .wrap,
    .upload-zone .file-upload,
    .upload-zone .file-upload-secondary,
    .report-zone {
        min-height: 300px !important;
    }
}
"""


with gr.Blocks(
    title="Yangtze AI Reviewer",
    css=CSS,
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ),
) as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-title">Yangtze AI Reviewer</div>
        <div class="hero-sub">
            Upload a proposal PDF and receive a structured expert-level scientific review report.
        </div>
    </div>
    """)

    with gr.Row(elem_id="main-panels", equal_height=True):
        with gr.Column(scale=1, min_width=0):
            with gr.Group(elem_classes=["panel-outer"]):
                with gr.Group(elem_classes=["panel-inner"]):
                    gr.Markdown("### Upload Area", elem_classes=["panel-heading"])
                    with gr.Group(elem_classes=["content-slot"]):
                        file_input = gr.File(
                            label="",
                            file_types=[".pdf"],
                            type="filepath",
                            show_label=False,
                            elem_classes=["upload-zone"],
                        )

        with gr.Column(scale=1, min_width=0):
            with gr.Group(elem_classes=["panel-outer"]):
                with gr.Group(elem_classes=["panel-inner"]):
                    gr.Markdown("### Formal Review Report", elem_classes=["panel-heading"])
                    with gr.Group(elem_classes=["content-slot"]):
                        report_md = gr.Markdown(
                            value="Upload a PDF to generate the formal review report.",
                            show_label=False,
                            elem_classes=["report-zone"],
                        )

    file_input.change(
        fn=auto_run_review,
        inputs=[file_input],
        outputs=report_md,
    )

if __name__ == "__main__":
    demo.launch()