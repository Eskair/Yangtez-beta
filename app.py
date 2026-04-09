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
        return (
            "Upload a PDF to generate the formal review report.",
            gr.update(value=None, visible=False)
        )

    file_path = Path(file_obj)
    if file_path.suffix.lower() != ".pdf":
        raise gr.Error("Only PDF files are supported.")

    yield (
        "⏳ **Yangtze AI is analyzing the proposal...**\n\nPlease wait while the system generates the formal review report.",
        gr.update(value=None, visible=False)
    )

    result = run_review(file_path=file_path, use_ocr=True)
    report_path, _ = extract_report_and_json(result)

    report_text = "Report not found."
    download_update = gr.update(value=None, visible=False)

    if report_path and Path(report_path).exists():
        report_text = Path(report_path).read_text(encoding="utf-8", errors="ignore")
        download_update = gr.update(value=report_path, visible=True)

    yield (
        report_text,
        download_update
    )


CSS = """
.gradio-container {
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding-top: 10px !important;
}

body {
    background: #0b0f19;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* ===== Hero ===== */
.hero {
    border: 1px solid #1f2a44;
    border-radius: 18px;
    padding: 28px 28px 24px 28px;
    background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
    margin-bottom: 22px;
}

.hero-title {
    font-size: 38px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 10px;
    letter-spacing: -0.5px;
}

.hero-sub {
    font-size: 15px;
    color: #cbd5e1;
    line-height: 1.8;
}

/* ===== Two symmetric panels ===== */
.panel-box {
    border: 1px solid #1e293b !important;
    border-radius: 18px !important;
    background: #455165 !important;
    padding: 28px 24px 24px 24px !important;
    min-height: 560px !important;
    box-shadow: none !important;
}

.panel-title {
    font-size: 24px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 24px;
}

.content-frame {
    height: 390px;
    display: flex;
    align-items: stretch;
}

#upload_box,
#report_box {
    width: 100%;
    height: 100%;
    border: 3px dashed rgba(255,255,255,0.85);
    border-radius: 0;
    background: #1e293b;
    overflow: hidden;
}

/* Upload component fills left frame */
#upload_box > .wrap,
#upload_box .or,
#upload_box .file-preview,
#upload_box .file-upload,
#upload_box .center {
    min-height: 100% !important;
}

/* Report box styling */
#report_box {
    border-style: solid;
    border-width: 0;
    background: #0b1220;
    border-radius: 18px;
    padding: 22px 24px;
    overflow-y: auto;
}

#report_box .prose,
#report_box .markdown {
    color: #f8fafc !important;
    line-height: 1.8 !important;
    font-size: 15px !important;
}

#report_box h1,
#report_box h2,
#report_box h3,
#report_box h4,
#report_box p,
#report_box li,
#report_box strong {
    color: #f8fafc !important;
}

/* Download file */
.download-wrap {
    margin-top: 18px;
}

/* Make the download component clean */
.download-wrap .file-wrap,
.download-wrap .file-preview,
.download-wrap .file {
    border-radius: 12px !important;
}

/* Remove extra dark block feeling on markdown outer wrapper */
.report-clean > div {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
}

/* Responsive */
@media (max-width: 900px) {
    .panel-box {
        min-height: auto !important;
    }

    .content-frame {
        height: 320px;
    }
}
"""


with gr.Blocks(
    title="Yangtze AI Reviewer",
    css=CSS,
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate", neutral_hue="slate"),
) as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-title">Yangtze AI Reviewer</div>
        <div class="hero-sub">
            Upload a proposal PDF and receive a structured expert-level scientific review report.
        </div>
    </div>
    """)

    with gr.Row(equal_height=True):
        with gr.Column(scale=1, min_width=420):
            with gr.Group(elem_classes=["panel-box"]):
                gr.HTML('<div class="panel-title">Upload Area</div>')
                gr.HTML('<div class="content-frame">')

                file_input = gr.File(
                    label="",
                    file_types=[".pdf"],
                    type="filepath",
                    elem_id="upload_box",
                    show_label=False
                )

                gr.HTML('</div>')

        with gr.Column(scale=1, min_width=420):
            with gr.Group(elem_classes=["panel-box"]):
                gr.HTML('<div class="panel-title">Formal Review Report</div>')
                gr.HTML('<div class="content-frame">')

                report_md = gr.Markdown(
                    value="Upload a PDF to generate the formal review report.",
                    elem_id="report_box",
                    elem_classes=["report-clean"],
                    show_label=False
                )

                gr.HTML('</div>')

                with gr.Group(elem_classes=["download-wrap"]):
                    report_file = gr.File(
                        label="Download Report (.md)",
                        visible=False
                    )

    file_input.change(
        fn=auto_run_review,
        inputs=[file_input],
        outputs=[report_md, report_file],
    )

if __name__ == "__main__":
    demo.launch()