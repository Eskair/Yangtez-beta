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
    max-width: 1280px !important;
    margin: 0 auto !important;
    padding-top: 10px !important;
    padding-left: 8px !important;
    padding-right: 8px !important;
}

body {
    background: #09111f;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* ===== Hero ===== */
.hero {
    border: 1px solid #23314e;
    border-radius: 18px;
    padding: 24px 26px 20px 26px;
    background: linear-gradient(180deg, #0d1730 0%, #09111f 100%);
    margin-bottom: 18px;
}

.hero-title {
    font-size: 34px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 8px;
    letter-spacing: -0.5px;
    line-height: 1.08;
}

.hero-sub {
    font-size: 14px;
    color: #dbe4f0;
    line-height: 1.7;
}

/* ===== Main layout ===== */
#main-panels {
    display: flex !important;
    flex-wrap: nowrap !important;
    gap: 14px !important;
    align-items: stretch !important;
}

#main-panels > .gr-column {
    flex: 1 1 0 !important;
    min-width: 0 !important;
}

/* ===== Outer cards ===== */
.panel-box {
    border: 1px solid rgba(20, 31, 49, 0.65) !important;
    border-radius: 20px !important;
    background: #556279 !important;
    padding: 16px !important;
    min-height: 500px !important;
    box-shadow: none !important;
}

/* ===== Inner cards ===== */
.inner-shell {
    border: 1px solid rgba(30, 41, 59, 0.8);
    border-radius: 18px;
    background: rgba(85, 98, 121, 0.92);
    padding: 16px;
    min-height: 460px;
}

.panel-title {
    font-size: 20px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 14px;
    line-height: 1.2;
}

.stage-wrap {
    height: 360px;
    display: flex;
    align-items: center;
    justify-content: center;
}

/* ===== Upload box ===== */
#upload_box {
    width: 100%;
    height: 330px !important;
    min-height: 330px !important;
    border: 2px dashed rgba(255,255,255,0.88) !important;
    border-radius: 0 !important;
    background: #23324a !important;
}

#upload_box > .wrap {
    min-height: 330px !important;
}

/* ===== Report display box ===== */
#report_shell {
    width: 100%;
    height: 330px;
    border-radius: 20px;
    background: #061225;
    padding: 0;
    overflow: hidden;
    box-shadow: 0 8px 22px rgba(0,0,0,0.18);
}

#report_box {
    width: 100%;
    height: 100%;
    border: none !important;
    border-radius: 20px !important;
    background: #061225 !important;
    padding: 18px 20px !important;
    overflow-y: auto !important;
}

/* Markdown text inside report */
#report_box .prose,
#report_box .markdown {
    color: #f8fafc !important;
    line-height: 1.8 !important;
    font-size: 15px !important;
    max-width: 100% !important;
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

.report-clean {
    border: none !important;
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* ===== Mobile ===== */
@media (max-width: 980px) {
    .hero-title {
        font-size: 28px;
    }

    #main-panels {
        flex-wrap: wrap !important;
    }

    #main-panels > .gr-column {
        min-width: 100% !important;
    }

    .panel-box {
        min-height: auto !important;
    }

    .inner-shell {
        min-height: auto !important;
    }

    .stage-wrap {
        height: auto;
    }

    #upload_box,
    #upload_box > .wrap,
    #report_shell {
        height: 280px !important;
        min-height: 280px !important;
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
            with gr.Group(elem_classes=["panel-box"]):
                gr.HTML("""
                <div class="inner-shell">
                    <div class="panel-title">Upload Area</div>
                    <div class="stage-wrap">
                """)

                file_input = gr.File(
                    label="",
                    file_types=[".pdf"],
                    type="filepath",
                    show_label=False,
                    elem_id="upload_box",
                )

                gr.HTML("""
                    </div>
                </div>
                """)

        with gr.Column(scale=1, min_width=0):
            with gr.Group(elem_classes=["panel-box"]):
                gr.HTML("""
                <div class="inner-shell">
                    <div class="panel-title">Formal Review Report</div>
                    <div class="stage-wrap">
                        <div id="report_shell">
                """)

                report_md = gr.Markdown(
                    value="Upload a PDF to generate the formal review report.",
                    show_label=False,
                    elem_id="report_box",
                    elem_classes=["report-clean"],
                )

                gr.HTML("""
                        </div>
                    </div>
                </div>
                """)

    file_input.change(
        fn=auto_run_review,
        inputs=[file_input],
        outputs=report_md,
    )

if __name__ == "__main__":
    demo.launch()