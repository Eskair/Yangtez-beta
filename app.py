# app.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import html
import traceback
import warnings
from pathlib import Path

import gradio as gr

# Spaces may call launch() without css/theme; Gradio 6 falls back to Blocks' deprecated fields.
warnings.filterwarnings(
    "ignore",
    message=r"The parameters have been moved from the Blocks constructor to the launch\(\) method in Gradio 6\.0: theme, css\..*",
    category=UserWarning,
)

from run_review import run_review

try:
    from openai import APIConnectionError, APIError, AuthenticationError, RateLimitError
except ImportError:
    AuthenticationError = APIConnectionError = RateLimitError = APIError = None  # type: ignore[misc, assignment]


PLACEHOLDER = """Upload a **PDF** proposal on the left. Analysis usually takes **about 1–5 minutes** for longer files or when OCR runs.

When the report is ready, **Download** and **Copy** will appear below this panel."""


def _is_ready_report(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    if s.startswith("## Review in progress"):
        return False
    if s.startswith("## Couldn't complete"):
        return False
    if s == PLACEHOLDER.strip():
        return False
    if s == "Report not found." or s.startswith("Report not found"):
        return False
    return True


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


def _progress_md(phase: str, *, step: int) -> str:
    """step: 1–4 marks which pipeline stage is active (for table + UX)."""
    stages = [
        (1, "Prepare", "Validate file and start the review run"),
        (2, "Extract", "PDF text, OCR, and layout reconstruction"),
        (3, "Analyze", "Domain profile, evidence selection, scoring, model calls"),
        (4, "Report", "Assemble the written review document"),
    ]
    body_rows: list[str] = []
    for n, title, detail in stages:
        if n < step:
            status_cell = "Complete"
        elif n == step:
            status_cell = "<strong>Active</strong>"
        else:
            status_cell = "Waiting"
        body_rows.append(
            "<tr>"
            f"<td>{n}</td>"
            f"<td>{html.escape(title)}</td>"
            f"<td>{html.escape(detail)}</td>"
            f"<td>{status_cell}</td>"
            "</tr>"
        )
    # HTML table + scoped CSS class so short columns use nowrap (markdown tables break mid-word in narrow layouts).
    table = (
        '<table class="yangtze-pipeline-status">'
        "<thead><tr>"
        "<th>Step</th><th>Stage</th><th>What runs</th><th>Status</th>"
        "</tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>"
    )
    return f"""## Review in progress

### Pipeline

{table}

**Current activity:** {html.escape(phase)}

Typical runtime is **about 1–5 minutes**; large PDFs, OCR, or slow APIs can take longer. Please keep this page open."""


def _error_markdown(title: str, detail: str, hints: list[str]) -> str:
    lines = [
        "## Couldn't complete the review",
        "",
        f"**What happened:** {title}",
        "",
    ]
    d = (detail or "").strip()
    if d:
        lines.extend(["```", d[:1200], "```", ""])
    if hints:
        lines.append("**You can try:**")
        for h in hints:
            lines.append(f"- {h}")
    return "\n".join(lines)


def _friendly_exception(exc: BaseException) -> tuple[str, str, list[str]]:
    msg = str(exc).strip() or exc.__class__.__name__
    low = msg.lower()

    if AuthenticationError and isinstance(exc, AuthenticationError):
        return (
            "API authentication failed",
            msg,
            [
                "Confirm `OPENAI_API_KEY` (or your provider key) is set in `.env` and reload the app.",
                "Check that the key is valid and not expired.",
            ],
        )

    if RateLimitError and isinstance(exc, RateLimitError):
        return (
            "API rate limit reached",
            msg,
            [
                "Wait a minute and try again.",
                "If you use a shared key, retry when traffic is lower.",
            ],
        )

    if APIConnectionError and isinstance(exc, APIConnectionError):
        return (
            "Could not reach the API",
            msg,
            [
                "Check your internet connection and any firewall or proxy settings.",
                "Confirm the API base URL is correct if you use a custom endpoint.",
            ],
        )

    if APIError and isinstance(exc, APIError):
        return (
            "The API returned an error",
            msg,
            [
                "Retry once; if it persists, check provider status and your account limits.",
            ],
        )

    if isinstance(exc, TimeoutError):
        return (
            "The operation timed out",
            msg,
            [
                "Try a smaller PDF or run again when the network is stable.",
            ],
        )

    if isinstance(exc, FileNotFoundError):
        return (
            "A required file was not found",
            msg,
            [
                "Ensure the PDF uploaded completely and try again.",
            ],
        )

    if "tesseract" in low or "pytesseract" in low:
        return (
            "OCR (Tesseract) is not available or misconfigured",
            msg,
            [
                "Install Tesseract OCR and ensure it is on your `PATH`, or use a PDF with embedded text.",
            ],
        )

    if "api key" in low or "invalid_api_key" in low or "401" in low:
        return (
            "Missing or invalid API key",
            msg,
            [
                "Set the correct key in `.env` and restart the app.",
            ],
        )

    return (
        f"{exc.__class__.__name__}",
        msg,
        [
            "Retry with the same PDF; if it keeps failing, try a smaller file.",
            "Check the terminal log for the full traceback.",
        ],
    )


def _idle_actions():
    return (
        gr.update(value=None, visible=False),
        gr.update(value=""),
        gr.update(visible=False, interactive=False),
        gr.update(visible=False),
    )


def auto_run_review(file_obj, progress=gr.Progress()):
    if file_obj is None:
        yield (PLACEHOLDER, *_idle_actions())
        return

    file_path = Path(file_obj)
    if file_path.suffix.lower() != ".pdf":
        yield (
            _error_markdown(
                "Unsupported file type",
                f"Got `{file_path.suffix or '(no extension)'}`.",
                ["Upload a file whose name ends with `.pdf`."],
            ),
            *_idle_actions(),
        )
        return

    progress(0.02, desc="Starting review…")
    yield (
        _progress_md("Queued — preparing pipeline…", step=2),
        *_idle_actions(),
    )

    progress(0.14, desc="Extracting text & layout…")
    yield (
        _progress_md(
            "Running full review (PDF → profiling → scoring → report)…",
            step=3,
        ),
        *_idle_actions(),
    )

    try:
        result = run_review(file_path=file_path, use_ocr=True)
    except BaseException as exc:
        title, detail, hints = _friendly_exception(exc)
        tb_lines = traceback.format_exception(exc.__class__, exc, exc.__traceback__, limit=12)
        tb_short = "".join(tb_lines).strip()
        yield (
            _error_markdown(
                title,
                f"{detail}\n\n---\nTechnical summary (truncated):\n{tb_short[:2500]}",
                hints,
            ),
            *_idle_actions(),
        )
        progress(1.0, desc="Stopped with an error")
        return

    report_path, _ = extract_report_and_json(result)
    report_text = "Report not found."
    download_path = None
    if report_path and Path(report_path).exists():
        report_text = Path(report_path).read_text(encoding="utf-8", errors="ignore")
        download_path = str(Path(report_path).resolve())

    ready = _is_ready_report(report_text)
    progress(0.92, desc="Formatting report…")
    progress(1.0, desc="Done")
    yield (
        report_text,
        gr.update(value=download_path, visible=ready and download_path is not None),
        gr.update(value=report_text if ready else ""),
        gr.update(visible=ready, interactive=ready and bool(report_text.strip())),
        gr.update(visible=ready),
    )


CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 20px 16px 32px !important;
}

body {
    background: radial-gradient(1200px 600px at 20% -10%, #1e3a5f 0%, transparent 55%),
                radial-gradient(900px 500px at 95% 0%, #0c4a6e 0%, transparent 50%),
                #0b1120;
    font-family: "Segoe UI", system-ui, -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
}

/* ===== Hero (id + !important: Gradio/HF theme can override plain .class colors) ===== */
.hero,
#yangtze-hero.hero {
    border: 1px solid rgba(148, 163, 184, 0.12);
    border-radius: 16px;
    padding: 28px 32px 26px;
    background: rgba(15, 23, 42, 0.55);
    backdrop-filter: blur(8px);
    margin-bottom: 24px;
    box-shadow: 0 1px 0 rgba(255, 255, 255, 0.04) inset;
    text-align: left !important;
}

.hero-kicker,
#yangtze-hero.hero .hero-kicker {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #38bdf8 !important;
    margin-bottom: 10px;
}

.hero-title,
#yangtze-hero.hero .hero-title {
    font-size: clamp(28px, 4vw, 34px);
    font-weight: 700;
    color: #f1f5f9 !important;
    margin: 0 0 12px 0;
    letter-spacing: -0.02em;
    line-height: 1.15;
}

.hero-sub,
#yangtze-hero.hero .hero-sub {
    font-size: 15px;
    color: #94a3b8 !important;
    line-height: 1.65;
    max-width: 52ch;
    margin: 0;
}

/* ===== Two columns: one flat card each (no nested inset panels) ===== */
#main-panels {
    align-items: stretch !important;
    gap: 20px !important;
}

.review-card {
    border: 1px solid rgba(148, 163, 184, 0.16) !important;
    border-radius: 16px !important;
    background: rgba(15, 23, 42, 0.72) !important;
    padding: 18px 18px 16px !important;
    min-height: 520px !important;
    box-shadow: 0 12px 32px rgba(0, 0, 0, 0.22) !important;
    gap: 12px !important;
    display: flex !important;
    flex-direction: column !important;
}

.review-card,
.review-card > .form,
.review-card > .gr-form {
    border-style: solid !important;
}

.review-card .form,
.review-card .gr-form {
    background: transparent !important;
    box-shadow: none !important;
}

.card-label {
    margin: 0 !important;
    padding: 0 !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.card-label p,
.card-label h3 {
    margin: 0 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    text-transform: uppercase !important;
    color: #cbd5e1 !important;
}

/* Upload: single dashed target inside the card (not a second “card”) */
.upload-zone {
    flex: 1 1 auto !important;
    min-height: 420px !important;
    border: 1px dashed rgba(148, 163, 184, 0.35) !important;
    border-radius: 12px !important;
    background: rgba(30, 41, 59, 0.25) !important;
    overflow: hidden !important;
    transition: border-color 0.2s ease, background 0.2s ease;
}

.upload-zone:hover {
    border-color: rgba(56, 189, 248, 0.45) !important;
    background: rgba(30, 41, 59, 0.38) !important;
}

.upload-zone > .wrap,
.upload-zone .file-upload,
.upload-zone .file-upload-secondary {
    min-height: 420px !important;
}

/* Report text: scroll inside the same card, no inner frame */
.report-zone {
    flex: 1 1 auto !important;
    min-height: 420px !important;
    max-height: 560px !important;
    padding: 8px 4px 12px !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.report-zone,
.report-zone > div {
    border: none !important;
    box-shadow: none !important;
}

.report-zone .prose,
.report-zone .markdown {
    color: #e2e8f0 !important;
    line-height: 1.75 !important;
    font-size: 15px !important;
    max-width: 100% !important;
}

.report-zone h1,
.report-zone h2 {
    color: #f8fafc !important;
    font-weight: 600 !important;
    margin-top: 1.1em !important;
}

.report-zone h3,
.report-zone h4 {
    color: #e2e8f0 !important;
}

.report-zone p,
.report-zone li {
    color: #cbd5e1 !important;
}

.report-zone strong {
    color: #f1f5f9 !important;
}

.report-zone .gradio-markdown,
.report-zone .gr-markdown,
.report-zone .markdown {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

.report-zone::-webkit-scrollbar {
    width: 8px;
}
.report-zone::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.35);
    border-radius: 999px;
}
.report-zone::-webkit-scrollbar-track {
    background: transparent;
}

/* Report actions: one row, equal-width controls */
#report-actions {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    justify-content: center !important;
    gap: 12px !important;
    margin-top: 4px !important;
    width: 100% !important;
}

#report-actions > div {
    flex: 1 1 0 !important;
    min-width: 0 !important;
    max-width: 320px !important;
    display: flex !important;
}

#report-actions button,
#report-actions a {
    width: 100% !important;
    justify-content: center !important;
    border-radius: 10px !important;
    box-sizing: border-box !important;
}

/* ===== Gradio built-in block progress (class names are hashed; match substrings) ===== */
.gradio-container [class*="progress-bar-wrap"] {
    width: min(100%, 400px) !important;
    height: 10px !important;
    border-radius: 9999px !important;
    border: 1px solid rgba(56, 189, 248, 0.22) !important;
    background: rgba(2, 6, 23, 0.88) !important;
    box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.45) !important;
    overflow: hidden !important;
}

.gradio-container [class*="progress-bar-wrap"] [class*="progress-bar"] {
    border-radius: 9999px !important;
    background: linear-gradient(90deg, #0369a1 0%, #0ea5e9 38%, #38bdf8 72%, #7dd3fc 100%) !important;
    box-shadow: 0 0 16px rgba(14, 165, 233, 0.4) !important;
}

.gradio-container [class*="eta-bar"] {
    border-radius: 9999px !important;
    background: linear-gradient(90deg, rgba(14, 165, 233, 0.18), rgba(56, 189, 248, 0.55)) !important;
    opacity: 1 !important;
}

.gradio-container [class*="progress-level-inner"] {
    font-family: "Segoe UI", system-ui, sans-serif !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    color: #e2e8f0 !important;
    max-width: min(100%, 480px) !important;
    line-height: 1.45 !important;
    text-align: center !important;
}

.gradio-container [class*="meta-text"] {
    font-family: "Segoe UI", system-ui, sans-serif !important;
    font-size: 12px !important;
    color: #94a3b8 !important;
    font-variant-numeric: tabular-nums !important;
}

.gradio-container [class*="wrap"][class*="generating"] {
    border-radius: 14px !important;
    border-color: rgba(56, 189, 248, 0.42) !important;
    box-shadow: 0 0 0 1px rgba(15, 23, 42, 0.55), 0 12px 40px rgba(0, 0, 0, 0.32) !important;
}

/* Pipeline status table (shown while review runs) */
.report-zone table {
    width: 100% !important;
    border-collapse: collapse !important;
    font-size: 13px !important;
    margin: 10px 0 14px !important;
    border: 1px solid rgba(148, 163, 184, 0.18) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

.report-zone thead th {
    background: rgba(30, 41, 59, 0.65) !important;
    color: #f1f5f9 !important;
    font-weight: 600 !important;
    padding: 8px 10px !important;
    text-align: left !important;
}

.report-zone tbody td {
    padding: 8px 10px !important;
    border-top: 1px solid rgba(148, 163, 184, 0.12) !important;
    color: #cbd5e1 !important;
    vertical-align: top !important;
}

.report-zone tbody tr:hover td {
    background: rgba(30, 41, 59, 0.35) !important;
}

/* Progress pipeline: keep Step / Stage / Status on one line (Gradio otherwise wraps inside words). */
.report-zone table.yangtze-pipeline-status {
    table-layout: auto !important;
}
.report-zone table.yangtze-pipeline-status th:nth-child(1),
.report-zone table.yangtze-pipeline-status td:nth-child(1) {
    white-space: nowrap !important;
    text-align: center !important;
    width: 1% !important;
}
.report-zone table.yangtze-pipeline-status th:nth-child(2),
.report-zone table.yangtze-pipeline-status td:nth-child(2),
.report-zone table.yangtze-pipeline-status th:nth-child(4),
.report-zone table.yangtze-pipeline-status td:nth-child(4) {
    white-space: nowrap !important;
    width: 1% !important;
}
.report-zone table.yangtze-pipeline-status th:nth-child(3),
.report-zone table.yangtze-pipeline-status td:nth-child(3) {
    white-space: normal !important;
    word-break: break-word !important;
}

@media (max-width: 980px) {
    .hero-title {
        font-size: 26px;
    }

    .review-card {
        min-height: auto !important;
    }

    .upload-zone,
    .upload-zone > .wrap,
    .upload-zone .file-upload,
    .upload-zone .file-upload-secondary {
        min-height: 280px !important;
    }

    .report-zone {
        max-height: 70vh !important;
    }
}
"""


# Theme + css on Blocks so Hugging Face Spaces still pick them up if launch() is called
# without css/theme (Gradio 6 falls back to Blocks' deprecated theme/css in that case).
_APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
)

with gr.Blocks(title="Yangtze AI Reviewer", theme=_APP_THEME, css=CSS) as demo:

    gr.HTML("""
    <div id="yangtze-hero" class="hero">
        <div class="hero-kicker">Proposal intelligence</div>
        <div class="hero-title">Yangtze AI Reviewer</div>
        <p class="hero-sub">
            Upload a proposal PDF and receive a structured expert-level scientific review report.
        </p>
    </div>
    """)

    with gr.Row(elem_id="main-panels", equal_height=True):
        with gr.Column(scale=1, min_width=0, elem_classes=["review-card"]):
            gr.Markdown("Proposal PDF", elem_classes=["card-label"])
            file_input = gr.File(
                label="",
                file_types=[".pdf"],
                type="filepath",
                show_label=False,
                elem_classes=["upload-zone"],
            )

        with gr.Column(scale=1, min_width=0, elem_classes=["review-card"]):
            gr.Markdown("Review report", elem_classes=["card-label"])
            report_md = gr.Markdown(
                value=PLACEHOLDER,
                show_label=False,
                elem_classes=["report-zone"],
                # Pipeline progress uses a small HTML <table> with nowrap columns; sanitizer can strip tables.
                sanitize_html=False,
            )
            with gr.Row(elem_id="report-actions", visible=False) as actions_row:
                download_btn = gr.DownloadButton(
                    "Download (.md)",
                    value=None,
                    visible=False,
                    size="sm",
                    variant="secondary",
                )
                copy_btn = gr.Button(
                    "Copy report",
                    size="sm",
                    variant="secondary",
                    visible=False,
                )
            raw_report_store = gr.Textbox(
                value="",
                label="",
                show_label=False,
                visible=False,
                lines=1,
                max_lines=1,
            )

    file_input.change(
        fn=auto_run_review,
        inputs=[file_input],
        outputs=[report_md, download_btn, raw_report_store, copy_btn, actions_row],
        show_progress="full",
    )

    copy_js = """
    (text) => {
        const t = text || "";
        if (!t) return;
        navigator.clipboard.writeText(t).then(() => {
            console.log("Report copied to clipboard");
        }).catch((e) => console.error(e));
    }
    """
    copy_btn.click(fn=None, inputs=[raw_report_store], outputs=None, js=copy_js)

if __name__ == "__main__":
    demo.launch(
        css=CSS,
        theme=_APP_THEME,
        footer_links=[],
    )
