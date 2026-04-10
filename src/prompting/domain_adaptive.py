# -*- coding: utf-8 -*-
from __future__ import annotations

GLOBAL_PROMPT_SUFFIX = """
If quantitative metrics are not explicitly provided, do not penalize the proposal by default.
Instead, infer quality from contextual evidence, domain knowledge, and implicit signals.
""".strip()

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

DIM_ORDER = ["team", "objectives", "strategy", "innovation", "feasibility"]

UNIVERSAL_REVIEW_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "T1_problem": {
        "title": "Problem / Need / Motivation",
        "dimension": "objectives",
        "question_template": (
            "Assess whether the proposal addresses an important and well-motivated problem. "
            "Pay special attention to: {evaluation_focus.problem}. "
            "Respect domain terminology: {terminology}. "
            + GLOBAL_PROMPT_SUFFIX
        ),
    },
    "T2_objectives": {
        "title": "Objectives / Scope",
        "dimension": "objectives",
        "question_template": (
            "Assess whether the proposal objectives are clear, specific, and appropriately scoped. "
            "Pay special attention to: {evaluation_focus.objectives}. "
            "Respect domain terminology: {terminology}. "
            + GLOBAL_PROMPT_SUFFIX
        ),
    },
    "T3_methods": {
        "title": "Methods / Technical Approach",
        "dimension": "strategy",
        "question_template": (
            "Assess whether the proposed methods or technical approach are coherent and fit the objectives. "
            "Method signals: {methods}. "
            "Respect domain terminology: {terminology}. "
            + GLOBAL_PROMPT_SUFFIX
        ),
    },
    "T4_evidence": {
        "title": "Evidence / Data / Resources",
        "dimension": "feasibility",
        "question_template": (
            "Assess whether the proposal provides adequate evidence, data, resources, or enabling conditions to support the work. "
            "Method signals: {methods}. "
            "Respect domain terminology: {terminology}. "
            + GLOBAL_PROMPT_SUFFIX
        ),
    },
    "T5_feasibility": {
        "title": "Feasibility / Execution",
        "dimension": "feasibility",
        "question_template": (
            "Assess whether the work appears executable in practice. "
            "Pay special attention to: {evaluation_focus.feasibility}. "
            "Risk signals: {risks}. "
            + GLOBAL_PROMPT_SUFFIX
        ),
    },
    "T6_innovation": {
        "title": "Innovation / Differentiation",
        "dimension": "innovation",
        "question_template": (
            "Assess the novelty or differentiating contribution of the proposal. "
            "Pay special attention to: {evaluation_focus.innovation}. "
            "Respect domain terminology: {terminology}. "
            + GLOBAL_PROMPT_SUFFIX
        ),
    },
    "T7_risks": {
        "title": "Risks / Failure Modes / Mitigation",
        "dimension": "feasibility",
        "question_template": (
            "Identify the major risks and evaluate whether mitigation thinking is credible. "
            "Risk signals: {risks}. "
            "Method signals: {methods}. "
            + GLOBAL_PROMPT_SUFFIX
        ),
    },
    "T8_team": {
        "title": "Team / Capability / Governance",
        "dimension": "team",
        "question_template": (
            "Assess whether the team and governance setup are adequate for the proposed work. "
            "Pay special attention to: {evaluation_focus.team}. "
            "Respect domain terminology: {terminology}. "
            + GLOBAL_PROMPT_SUFFIX
        ),
    },
    "T9_outcomes": {
        "title": "Outcomes / Impact / Evaluation",
        "dimension": "objectives",
        "question_template": (
            "Assess whether the expected outcomes, impact claims, and evaluation logic are credible. "
            "Pay special attention to: {evaluation_focus.outcomes}. "
            "Respect domain terminology: {terminology}. "
            + GLOBAL_PROMPT_SUFFIX
        ),
    },
}

ALLOWED_SLOTS = {
    "evaluation_focus.problem",
    "evaluation_focus.objectives",
    "evaluation_focus.feasibility",
    "evaluation_focus.innovation",
    "evaluation_focus.team",
    "evaluation_focus.outcomes",
    "methods",
    "risks",
    "terminology",
}

TEMPLATE_SLOT_MAP = {
    "T1_problem": ["evaluation_focus.problem", "terminology"],
    "T2_objectives": ["evaluation_focus.objectives", "terminology"],
    "T3_methods": ["methods", "terminology"],
    "T4_evidence": ["methods", "terminology"],
    "T5_feasibility": ["evaluation_focus.feasibility", "risks"],
    "T6_innovation": ["evaluation_focus.innovation", "terminology"],
    "T7_risks": ["risks", "methods"],
    "T8_team": ["evaluation_focus.team", "terminology"],
    "T9_outcomes": ["evaluation_focus.outcomes", "terminology"],
}


@dataclass(frozen=True)
class ReviewTask:
    task_id: str
    template_id: str
    dimension: str
    title: str


REVIEW_TASKS: List[ReviewTask] = [
    ReviewTask("problem", "T1_problem", "objectives", "Problem significance"),
    ReviewTask("objectives", "T2_objectives", "objectives", "Objectives and scope"),
    ReviewTask("methods", "T3_methods", "strategy", "Methods and approach"),
    ReviewTask("evidence", "T4_evidence", "feasibility", "Evidence and resources"),
    ReviewTask("feasibility", "T5_feasibility", "feasibility", "Execution feasibility"),
    ReviewTask("innovation", "T6_innovation", "innovation", "Innovation"),
    ReviewTask("risks", "T7_risks", "feasibility", "Risks and mitigation"),
    ReviewTask("team", "T8_team", "team", "Team and governance"),
    ReviewTask("outcomes", "T9_outcomes", "objectives", "Outcomes and evaluation"),
]

QUESTION_SEARCH_HINTS = {
    "team": [
        "team expertise",
        "roles and responsibilities",
        "governance",
        "核心团队",
        "组织架构",
        "咨询委员会",
        "成员与职责",
    ],
    "objectives": ["problem significance", "scope", "outcomes", "evaluation"],
    "strategy": ["methods", "technical approach", "workflow"],
    "innovation": ["novelty", "differentiation", "prior work"],
    "feasibility": ["resources", "risks", "execution plan", "dependencies"],
}

# Single source of truth for LLM domain profiling (run_review + CLI profilers).
PROFILER_SYSTEM_PROMPT = """
You are a strict domain profiler for proposal review.

Your task is ONLY to derive a clean domain profile from the CURRENT document text.

Hard rules:
- Use ONLY the provided document text.
- Ignore any prior tasks, prior files, templates, examples, cached memory, or default domains.
- Do not rewrite the proposal and do not generate review questions.
- Do not invent domain details, methods, risks, or terminology not supported by the text.
- Do NOT infer a domain from generic proposal boilerplate alone.
- Do NOT output a narrow field label (e.g., aerospace, biomedical, a specific disease) unless the document clearly and repeatedly supports it with explicit terms.
- If evidence is weak or ambiguous, return "unknown" for domain.primary and keep lists short or empty.
- Prefer labels grounded in repeated concrete terms from the text; when evidence is thin, prefer broader, still text-grounded labels over over-specific guesses.
- Keep all lists short, concrete, and normalized.
- Return valid JSON only.

Return JSON with this structure:
{
  "domain": {
    "primary": "string",
    "secondary": ["string"]
  },
  "evaluation_focus": {
    "problem": ["string"],
    "objectives": ["string"],
    "feasibility": ["string"],
    "innovation": ["string"],
    "team": ["string"],
    "outcomes": ["string"]
  },
  "methods": ["string"],
  "risks": ["string"],
  "terminology": ["string"],
  "document_form": {
    "primary": "grant_proposal | feasibility_study | business_plan | technical_report | unknown",
    "confidence": 0.0,
    "rationale": "short text-grounded reason"
  }
}
""".strip()


def sanitize_document_form(raw: Any) -> Dict[str, Any]:
    allowed = {"grant_proposal", "feasibility_study", "business_plan", "technical_report", "unknown"}
    if not isinstance(raw, dict):
        return {"primary": "unknown", "confidence": 0.0, "rationale": ""}
    primary = str(raw.get("primary", "unknown")).strip().lower().replace(" ", "_")
    if primary not in allowed:
        primary = "unknown"
    try:
        conf = float(raw.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    conf = max(0.0, min(1.0, conf))
    rationale = " ".join(str(raw.get("rationale", "")).split())[:240]
    return {"primary": primary, "confidence": conf, "rationale": rationale}


def get_document_form_prompt_suffix(profile: Dict[str, Any]) -> str:
    """Reviewer-facing instruction block; keep English for current LLM prompts in run_review."""
    form = sanitize_document_form(profile.get("document_form"))
    primary = form["primary"]
    if primary == "business_plan":
        return (
            "\n\n[Document form: business / financing plan] "
            "Weight milestones, commercial traction, unit economics, customers and partnerships, "
            "and execution governance. Do not apply pure academic-grant criteria alone."
        )
    if primary == "grant_proposal":
        return (
            "\n\n[Document form: research grant style] "
            "Weight scientific problem, novelty vs prior work, methodology rigor, and research feasibility."
        )
    if primary == "feasibility_study":
        return (
            "\n\n[Document form: feasibility / engineering study] "
            "Weight investment logic, engineering deliverables, schedule, and risk controls."
        )
    if primary == "technical_report":
        return (
            "\n\n[Document form: technical report / whitepaper] "
            "Weight technical depth, reproducibility, and validation evidence."
        )
    return ""


def normalize_list(items: Any, max_items: int = 6) -> List[str]:
    if items is None:
        return []

    if isinstance(items, str):
        items = [items]
    elif not isinstance(items, (list, tuple, set)):
        items = [items]

    # Step 1: 基础清洗
    cleaned = []
    for item in items:
        text = " ".join(str(item).split()).strip()
        if not text:
            continue

        # 去掉纯标点（关键修复）
        if text in {"、", ";", "；", ",", ".", "。"}:
            continue

        cleaned.append(text)

    # Step 2: 合并“单字碎片”（核心修复）
    merged = []
    buffer = ""

    for token in cleaned:
        # 如果是单个中文字符 → 拼接
        if len(token) == 1 and '\u4e00' <= token <= '\u9fff':
            buffer += token
        else:
            if buffer:
                merged.append(buffer)
                buffer = ""
            merged.append(token)

    if buffer:
        merged.append(buffer)

    # Step 3: 去重 + 限制长度
    seen, out = set(), []
    for text in merged:
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)

        if len(out) >= max_items:
            break
    return out


def _resolve(profile: Dict[str, Any], slot: str) -> str:
    node: Any = profile
    for part in slot.split("."):
        if not isinstance(node, dict):
            return "not specified in domain profile"
        node = node.get(part)
        if node is None:
            return "not specified in domain profile"

    if isinstance(node, list):
        vals = normalize_list(node)
        return "; ".join(vals) if vals else "not specified in domain profile"

    if isinstance(node, str):
        text = " ".join(node.split()).strip()
        return text or "not specified in domain profile"

    text = " ".join(str(node).split()).strip()
    return text or "not specified in domain profile"


def sanitize_domain_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(profile, dict):
        profile = {}

    domain = profile.get("domain") if isinstance(profile.get("domain"), dict) else {}
    eval_focus = profile.get("evaluation_focus") if isinstance(profile.get("evaluation_focus"), dict) else {}

    primary = " ".join(str(domain.get("primary", "unknown")).split()).strip() or "unknown"
    bad_domains = {
        "general proposal",
        "general",
        "unknown",
        "misc",
        "other",
        "n/a",
        "na",
    }
    if primary.lower() in bad_domains:
        primary = "unknown"

    methods = normalize_list(profile.get("methods", []), max_items=6)
    risks = normalize_list(profile.get("risks", []), max_items=6)
    terminology = normalize_list(profile.get("terminology", []), max_items=12)

    # 如果 methods / risks 太碎，就尽量用 terminology 补救
    def _too_fragmented(values: List[str]) -> bool:
        if not values:
            return True
        short_count = len([x for x in values if len(x) <= 1])
        return short_count >= max(2, len(values) // 2 + 1)

    if _too_fragmented(methods):
        fallback_methods = [t for t in terminology if len(t) >= 2][:6]
        if fallback_methods:
            methods = fallback_methods

    if _too_fragmented(risks):
        fallback_risks = [t for t in terminology if len(t) >= 2][:6]
        if fallback_risks:
            risks = fallback_risks

    return {
        "domain": {
            "primary": primary,
            "secondary": normalize_list(domain.get("secondary", []), max_items=4),
        },
        "evaluation_focus": {
            "problem": normalize_list(eval_focus.get("problem", []), max_items=5),
            "objectives": normalize_list(eval_focus.get("objectives", []), max_items=5),
            "feasibility": normalize_list(eval_focus.get("feasibility", []), max_items=5),
            "innovation": normalize_list(eval_focus.get("innovation", []), max_items=5),
            "team": normalize_list(eval_focus.get("team", []), max_items=5),
            "outcomes": normalize_list(eval_focus.get("outcomes", []), max_items=5),
        },
        "methods": methods,
        "risks": risks,
        "terminology": terminology,
        "document_form": sanitize_document_form(profile.get("document_form")),
    }


def inject_template(template_id: str, profile: Dict[str, Any]) -> str:
    template = UNIVERSAL_REVIEW_TEMPLATES[template_id]["question_template"]
    safe_profile = sanitize_domain_profile(profile)
    rendered = template
    for slot in TEMPLATE_SLOT_MAP[template_id]:
        rendered = rendered.replace("{" + slot + "}", _resolve(safe_profile, slot))
    return rendered


def build_specialized_question(task: ReviewTask, profile: Dict[str, Any]) -> str:
    injected = inject_template(task.template_id, profile)
    return f"[{task.title}] {injected}"


def load_domain_profile(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return sanitize_domain_profile({})
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return sanitize_domain_profile({})
    return sanitize_domain_profile(data)