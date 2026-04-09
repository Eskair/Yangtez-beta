# app.py
# -*- coding: utf-8 -*-

import json
import re
from pathlib import Path

import gradio as gr
from run_review import run_review


# =========================
# Helpers
# =========================
def safe_text(x, default="-"):
    if x is None:
        return default
    s = str(x).strip()
    return s if s else default


def fmt_score(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"


def fmt_conf(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return "-"


def normalize_verdict(verdict: str) -> str:
    v = safe_text(verdict, "UNKNOWN").upper()
    if v not in {"SUPPORT", "HOLD", "REJECT"}:
        return v
    return v


def verdict_class(verdict: str) -> str:
    v = normalize_verdict(verdict)
    if v == "SUPPORT":
        return "verdict-support"
    if v == "REJECT":
        return "verdict-reject"
    return "verdict-hold"


def render_summary_card(review: dict) -> str:
    proposal_id = safe_text(review.get("proposal_id"))
    profiler_mode = safe_text(review.get("profiler_mode"))
    overall_score = fmt_score(review.get("overall_score_10"))
    confidence = fmt_conf(review.get("confidence"))
    verdict = normalize_verdict(review.get("verdict"))

    return f"""
    <div class="summary-shell">
        <div class="summary-grid">
            <div class="summary-card verdict-card {verdict_class(verdict)}">
                <div class="summary-label">Final Verdict</div>
                <div class="summary-value verdict-value">{verdict}</div>
                <div class="summary-sub">Structured review conclusion</div>
            </div>

            <div class="summary-card">
                <div class="summary-label">Overall Score</div>
                <div class="summary-value">{overall_score} / 10</div>
                <div class="summary-sub">Overall expert evaluation score</div>
            </div>

            <div class="summary-card">
                <div class="summary-label">Confidence</div>
                <div class="summary-value">{confidence}</div>
                <div class="summary-sub">Review confidence estimate</div>
            </div>

            <div class="summary-card">
                <div class="summary-label">Proposal ID</div>
                <div class="summary-value small-value">{proposal_id}</div>
                <div class="summary-sub">Current review run identifier</div>
            </div>
        </div>

        <div class="meta-row">
            <div><span class="meta-key">Profiler mode:</span> {profiler_mode}</div>
        </div>
    </div>
    """


def render_dimension_cards(review: dict) -> str:
    dims = review.get("dimension_scores", {}) or {}

    order = [
        ("team", "Team"),
        ("objectives", "Objectives"),
        ("strategy", "Strategy"),
        ("innovation", "Innovation"),
        ("feasibility", "Feasibility"),
    ]

    items = []
    for key, label in order:
        value = fmt_score(dims.get(key))
        items.append(
            f"""
            <div class="dim-card">
                <div class="dim-name">{label}</div>
                <div class="dim-score">{value}</div>
            </div>
            """
        )

    return f"""
    <div class="dim-shell">
        <div class="panel-title">Dimension Scores</div>
        <div class="dim-grid">
            {''.join(items)}
        </div>
    </div>
    """


def render_empty_summary() -> str:
    return """
    <div class="placeholder-box">
        <div class="placeholder-title">Review summary will appear here</div>
        <div class="placeholder-text">
            Upload a proposal PDF, then click <b>Start Review</b>.
        </div>
    </div>
    """


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


def run_ui_review(file_obj, proposal_id):

    if file_obj is None:
        raise gr.Error("Please upload a PDF file first.")

    # 🔥 STEP 1：立即显示运行状态
    yield (
        "⏳ **Yangtze AI is analyzing your proposal...**",
        """
        <div class="placeholder-box">
            🔍 Parsing document structure...<br>
            📊 Extracting key information...<br>
            🤖 Running expert evaluation...
        </div>
        """,
        """
        <div class="placeholder-box">
            Evaluating dimensions:<br>
            • Team<br>
            • Objectives<br>
            • Strategy<br>
            • Innovation<br>
            • Feasibility
        </div>
        """,
        "",
        None,
        None
    )

    file_path = Path(file_obj)
    if file_path.suffix.lower() != ".pdf":
        raise gr.Error("Only PDF files are supported.")

    pid = proposal_id.strip() if proposal_id else None
    if pid == "":
        pid = None

    result = run_review(file_path=file_path, proposal_id=pid, use_ocr=True)

    report_path, review_json_path = extract_report_and_json(result)

    review = json.loads(Path(review_json_path).read_text(encoding="utf-8"))

    report_md = ""
    if report_path and Path(report_path).exists():
        report_md = Path(report_path).read_text(encoding="utf-8", errors="ignore")

    status_md = f"""
✅ **Review completed**

- **Proposal:** `{review.get("proposal_id")}`
- **Final Verdict:** **{review.get("verdict")}**
- **Score:** **{review.get("overall_score_10")} / 10**
- **Confidence:** **{review.get("confidence")}**
"""

    summary_html = render_summary_card(review)
    dimension_html = render_dimension_cards(review)

    return (
        status_md,
        summary_html,
        dimension_html,
        report_md,
        report_path,
        review_json_path
    )


def clear_ui():
    return (
        None,   # file
        "",     # proposal id
        "",     # status
        render_empty_summary(),
        render_empty_summary(),
        "",     # report markdown
        None,   # report download
        None,   # json download
    )


def auto_run_review(file_obj):
    if file_obj is None:
        return (
            "Upload a PDF file to start review automatically.",
            "The formal review report will appear here after upload.",
            None
        )

    file_path = Path(file_obj)
    if file_path.suffix.lower() != ".pdf":
        raise gr.Error("Only PDF files are supported.")

    yield (
        "⏳ **Yangtze AI is analyzing the proposal...**",
        "Generating formal review report...",
        None
    )

    result = run_review(file_path=file_path, use_ocr=True)
    report_path, _ = extract_report_and_json(result)

    report_md_text = "Report not found."
    if report_path and Path(report_path).exists():
        report_md_text = Path(report_path).read_text(encoding="utf-8", errors="ignore")

    yield (
        "✅ **Review completed successfully**",
        report_md_text,
        report_path if report_path and Path(report_path).exists() else None
    )


# =========================
# CSS
# =========================
CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}

/* ===== 背景 ===== */
body {
    background: #0b0f19;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.app-wrap {
    margin-top: 10px;
}

/* ===== 顶部 Hero ===== */
.hero {
    border: 1px solid #1f2a44;
    border-radius: 18px;
    padding: 28px 28px 22px 28px;
    background: linear-gradient(180deg, #111827 0%, #0b1220 100%);
    margin-bottom: 22px;
}

.hero-title {
    font-size: 40px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 6px;
    letter-spacing: -0.5px;
}

.hero-sub {
    font-size: 15px;
    color: #9ca3af;
    line-height: 1.8;
    margin-bottom: 20px;
}

/* ===== Steps ===== */
.steps {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
}

.step-card {
    border: 1px solid #1f2a44;
    border-radius: 14px;
    padding: 16px;
    background: #0f172a;
    transition: 0.2s;
}

.step-card:hover {
    border-color: #3b82f6;
}

.step-no {
    width: 28px;
    height: 28px;
    line-height: 28px;
    border-radius: 999px;
    background: #2563eb;
    color: white;
    text-align: center;
    font-weight: 700;
    margin-bottom: 10px;
}

.step-title {
    color: #f1f5f9;
    font-size: 15px;
    font-weight: 700;
    margin-bottom: 4px;
}

.step-desc {
    color: #9ca3af;
    font-size: 13px;
    line-height: 1.6;
}

/* ===== Panel ===== */
.panel-box {
    border: 1px solid #1e293b;
    border-radius: 16px;
    background: #0f172a;
    padding: 20px;
    margin-bottom: 16px;
}

.panel-title {
    font-size: 22px;
    font-weight: 800;
    color: #f8fafc;
    margin-bottom: 6px;
}

.panel-desc {
    color: #9ca3af;
    font-size: 14px;
    margin-bottom: 14px;
}

.help-note {
    border: 1px solid #1f2a44;
    background: #0b1220;
    color: #94a3b8;
    border-radius: 12px;
    padding: 12px 14px;
    font-size: 13px;
    line-height: 1.7;
    margin-top: 8px;
}

/* ===== 按钮 ===== */
button.primary {
    background: #2563eb !important;
    border: none !important;
}

button.primary:hover {
    background: #1d4ed8 !important;
}

/* ===== Summary ===== */
.summary-shell {
    margin-top: 10px;
}

.summary-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 14px;
}

.summary-card {
    border: 1px solid #1e293b;
    border-radius: 16px;
    background: #0f172a;
    padding: 18px;
    transition: 0.2s;
}

.summary-card:hover {
    transform: translateY(-2px);
}

.summary-label {
    font-size: 12px;
    color: #64748b;
    margin-bottom: 8px;
    text-transform: uppercase;
}

.summary-value {
    font-size: 30px;
    font-weight: 900;
    color: #f8fafc;
}

.small-value {
    font-size: 20px;
}

.summary-sub {
    margin-top: 8px;
    color: #94a3b8;
    font-size: 12px;
}

/* ===== Verdict 强化 ===== */
.verdict-card {
    border-width: 2px;
}

.verdict-support {
    border-color: #22c55e;
    background: rgba(34,197,94,0.08);
}

.verdict-hold {
    border-color: #f59e0b;
    background: rgba(245,158,11,0.08);
}

.verdict-reject {
    border-color: #ef4444;
    background: rgba(239,68,68,0.08);
}

/* ===== Dimension ===== */
.dim-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 14px;
    margin-top: 14px;
}

.dim-card {
    border: 1px solid #1e293b;
    border-radius: 14px;
    background: #0b1220;
    padding: 18px;
    text-align: center;
}

.dim-name {
    color: #94a3b8;
    font-size: 13px;
    margin-bottom: 6px;
}

.dim-score {
    color: #f8fafc;
    font-size: 26px;
    font-weight: 800;
}

/* ===== Report ===== */
.report-box {
    border: 1px solid #1e293b;
    border-radius: 16px;
    background: #0b1220;
    padding: 20px 24px;
    line-height: 1.9;
    font-size: 15px;
    max-width: 900px;
    margin: 0 auto;
}

/* ===== Placeholder ===== */
.placeholder-box {
    border: 1px dashed #334155;
    border-radius: 16px;
    padding: 24px;
    background: #0b1220;
}

.placeholder-title {
    color: #f8fafc;
    font-size: 18px;
    font-weight: 700;
}

.placeholder-text {
    color: #94a3b8;
    margin-top: 6px;
}

/* ===== 响应式 ===== */
@media (max-width: 1000px) {
    .summary-grid { grid-template-columns: 1fr 1fr; }
    .dim-grid { grid-template-columns: 1fr 1fr; }
}

@media (max-width: 600px) {
    .summary-grid { grid-template-columns: 1fr; }
    .dim-grid { grid-template-columns: 1fr; }
}
"""


# =========================
# UI
# =========================
with gr.Blocks(
    title="Yangtze AI Reviewer",
    css=CSS,
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:

    gr.HTML("""
    <div class="hero">
        <div class="hero-title">Yangtze AI Reviewer</div>
        <div class="hero-sub">
            Upload a proposal PDF and receive a structured expert-level scientific review report.
        </div>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            with gr.Group(elem_classes=["panel-box"]):
                gr.HTML('<div class="panel-title">Upload Area</div>')

                file_input = gr.File(
                    label="Proposal PDF",
                    file_types=[".pdf"],
                    type="filepath"
                )

                upload_status = gr.Markdown(
                    "Upload a PDF file to start review automatically."
                )

        with gr.Column(scale=8):
            with gr.Group(elem_classes=["panel-box"]):
                gr.HTML('<div class="panel-title">Formal Review Report</div>')

                report_md = gr.Markdown(
                    value="The formal review report will appear here after upload.",
                    elem_classes=["report-box"]
                )

                report_file = gr.File(label="Download Report (.md)")

    file_input.change(
        fn=auto_run_review,
        inputs=[file_input],
        outputs=[upload_status, report_md, report_file],
    )

if __name__ == "__main__":
    demo.launch()