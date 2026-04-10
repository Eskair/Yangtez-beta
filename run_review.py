#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

"""
End-to-end Yangtze review runner.

Pipeline:
PDF -> prepared JSON -> domain profile -> metric check -> specialized prompts
-> review JSON -> markdown report
"""

import argparse
import json
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from src.tools.domain_profiler import profile_with_optional_llm
from src.prompting.domain_adaptive import (
    QUESTION_SEARCH_HINTS,
    REVIEW_TASKS,
    build_specialized_question,
    get_document_form_prompt_suffix,
)
from src.tools.prepare_proposal_text import prepare_text
from src.tools.metric_checker import build_metric_report, build_metric_prompt_suffix

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "src" / "data"
RUNS_DIR = DATA_DIR / "runs"


@dataclass(frozen=True)
class EvidenceSnippet:
    page_index: int
    text: str
    score: float


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _short_citation_quote(text: str, max_len: int = 100) -> str:
    t = _clean_text(text)
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"


def detect_document_staleness(full_text: str, current_year: int | None = None) -> Dict[str, Any]:
    cy = current_year or datetime.now().year
    years = sorted({int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", full_text)})
    latest = max(years) if years else None
    warning_zh = None
    if latest is not None and (cy - latest) >= 6:
        warning_zh = (
            f"文中出现的年份多在 {latest} 年及更早，距今约 {cy - latest} 年；"
            "市场、政策与竞争格局可能已变化，结论需结合时效性审慎看待。"
        )
    elif latest is not None and (cy - latest) >= 4:
        warning_zh = (
            f"材料中显著日期距今约 {cy - latest} 年，请关注数据与外部引用是否仍适用。"
        )
    return {
        "years_found": years,
        "latest_year": latest,
        "warning_zh": warning_zh,
        "reference_year": cy,
    }


def build_evidence_digest_for_llm(task_results: Sequence[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for tr in task_results:
        evs = tr.get("evidence") or []
        if not evs:
            continue
        bits: List[str] = []
        for e in evs[:2]:
            pg = e.get("page_index")
            sn = _short_citation_quote(str(e.get("text", "")), 80)
            bits.append(f"第{pg}页：{sn}")
        title = tr.get("title", "")
        lines.append(f"- **{title}**: " + "；".join(bits))
    return "\n".join(lines) if lines else ""


def format_evidence_page_allowlist(task_results: Sequence[Dict[str, Any]]) -> str:
    """供主报告 LLM 使用的页码边界说明，抑制封面页、索引外页码等不真实引用。"""
    pages: set[int] = set()
    for tr in task_results:
        for e in tr.get("evidence") or []:
            try:
                pages.add(int(e["page_index"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not pages:
        return (
            "本轮系统摘录未锁定具体 PDF 页码。「主要问题」「修改建议」中**禁止**书写「（第N页）」类括号；"
            "若需指称材料，使用「见申请材料正文」等泛称即可。"
        )
    ordered = sorted(pages)
    joined = "、".join(str(p) for p in ordered)
    return (
        f"下列页码来自上方「证据摘录索引」中的摘录，**仅限**在「主要问题」「修改建议」的括号中使用这些页码之一：{joined}。"
        "括号中的 N **必须**属于该集合；**禁止**编造任何未出现在该集合中的页码。"
        "第1页、第2页多为封面、题名或目录；**不得**将其作为风险管理、技术路线细节、指标缺失等**实质性**判断的**唯一**依据，"
        "除非该页在本集合中且摘录原文与判断直接对应。"
        "时效性、全文日期类判断若无单一页码依据，请**省略**页码括号，或仅用「（材料全文日期特征）」一类表述。"
    )


QUALITY_NOTES_SYSTEM = """你是评审材料一致性助手。你只根据给定 JSON 做短评，不编造材料中不存在的事实。
必须只输出一个 JSON 对象，键为 claim_evidence_notes 与 consistency_flags，值均为字符串数组。
claim_evidence_notes：判断句与摘录若不完全支持、或措辞相对证据过强时的提醒（每条≤90字，最多4条）。
consistency_flags：时效性、指标完整性、维度间潜在矛盾等待核对提醒（每条≤90字，最多3条）。
若无问题则对应数组为空。不要重复输入原文。"""


def _quality_notes_payload(
    task_results: Sequence[Dict[str, Any]],
    staleness: Dict[str, Any],
    metric_report: Dict[str, Any],
) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []
    for tr in task_results:
        ev_snippets: List[Dict[str, Any]] = []
        for e in (tr.get("evidence") or [])[:2]:
            ev_snippets.append({
                "page": e.get("page_index"),
                "text": _short_citation_quote(str(e.get("text", "")), 140),
            })
        blocks.append({
            "task_title": tr.get("title", ""),
            "judgment_excerpt": _short_citation_quote(str(tr.get("judgment", "")), 220),
            "score_10": tr.get("score_10"),
            "confidence": tr.get("confidence"),
            "evidence": ev_snippets,
        })
    ms = metric_report.get("metric_signals", {}) or {}
    return {
        "tasks": blocks,
        "staleness_warning_zh": (staleness or {}).get("warning_zh"),
        "evidence_level": ms.get("evidence_level"),
        "missing_metric_signals": (ms.get("missing_metric_signals") or [])[:5],
    }


def llm_review_quality_notes(
    task_results: Sequence[Dict[str, Any]],
    staleness: Dict[str, Any],
    metric_report: Dict[str, Any],
) -> Dict[str, Any]:
    flag = (os.getenv("YANGTZE_QUALITY_NOTES", "1") or "1").strip().lower()
    if flag in ("0", "false", "no", "off"):
        return {}
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        return {}
    payload = _quality_notes_payload(task_results, staleness, metric_report)
    model = (os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()
    try:
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUALITY_NOTES_SYSTEM},
                {
                    "role": "user",
                    "content": json.dumps(payload, ensure_ascii=False),
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.15,
            max_tokens=500,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(raw) if raw else {}
        notes = [str(x).strip() for x in (data.get("claim_evidence_notes") or []) if str(x).strip()][:4]
        flags = [str(x).strip() for x in (data.get("consistency_flags") or []) if str(x).strip()][:3]
        return {"claim_evidence_notes": notes, "consistency_flags": flags}
    except Exception as exc:
        print(f"[WARN] 质量自检 LLM 跳过: {exc}")
        return {}


def _format_quality_notes_for_prompt(qn: Dict[str, Any]) -> str:
    if not qn:
        return "（未运行质量自检；请直接依据证据摘录与结构化结果撰写。）"
    parts: List[str] = []
    for item in qn.get("claim_evidence_notes") or []:
        parts.append(f"- [判断—证据] {item}")
    for item in qn.get("consistency_flags") or []:
        parts.append(f"- [一致性] {item}")
    return "\n".join(parts) if parts else "（本轮未检出需特别提示项）"


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_") or "proposal"


def load_pages(proposal_id: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / "prepared" / proposal_id / "pages.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_like_page(raw: str) -> bool:
    lines = [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]
    if not lines:
        return False
    first = lines[0]
    if first in ("参考文献", "主要参考文献", "参考书目"):
        return True
    low = first.lower()
    return low.startswith("references") or low.startswith("bibliography")


def evidence_page_multiplier(task_id: str, page_type: str, raw_text: str) -> float:
    """
    降低目录/封面/参考文献等低语义页在证据检索中的权重，提升与真实评审「看正文」习惯的一致性。
    未识别 page_type 时，用首行启发式识别参考文献页。
    """
    pt = (page_type or "").strip()
    if not pt and _reference_like_page(raw_text):
        pt = "references_page"

    if pt in ("toc_slide", "cover_or_simple_slide"):
        if task_id in ("objectives", "problem"):
            return 0.55
        return 0.28
    if pt == "references_page":
        if task_id == "evidence":
            return 0.5
        return 0.22
    if pt == "timeline_slide":
        if task_id in ("feasibility", "methods", "outcomes"):
            return 1.08
        return 0.72
    if pt == "table_page":
        if task_id in ("outcomes", "feasibility", "evidence"):
            return 1.06
        if task_id == "team":
            return 0.88
        return 1.0
    if pt == "data_visual_slide":
        if task_id in ("evidence", "outcomes", "innovation"):
            return 1.02
        if task_id == "team":
            return 0.82
        return 0.95
    return 1.0


TASK_KEYWORDS = {
    "problem": ["背景", "需求", "意义", "motivation", "pain point", "problem", "行业背景"],
    "objectives": ["目标", "scope", "deliverable", "计划", "路径", "objective"],
    "methods": ["方法", "技术路线", "样机", "验证", "test", "prototype", "method"],
    "evidence": ["样机", "测试", "专利", "数据", "成果", "validation", "evidence"],
    "feasibility": ["交付", "timeline", "roadmap", "风险", "量产", "feasibility"],
    "innovation": ["唯一", "领先", "创新", "novel", "differentiation", "突破"],
    "risks": ["风险", "瓶颈", "dependency", "cost", "寿命", "竞争", "risk"],
    "team": [
        "团队",
        "核心团队",
        "组织架构",
        "咨询委员会",
        "依托单位",
        "博士",
        "研究员",
        "教授",
        "院士",
        "总工程师",
        "首席执行官",
        "职责",
        "roles",
        "team",
        "organization",
        "committee",
    ],
    "outcomes": ["市场", "订单", "收入", "profit", "impact", "evaluation", "融资"],
}

TEAM_SECTION_HEADERS = re.compile(
    r"核心团队|项目核心团队|组织架构|公司组织架构|咨询委员会|技术咨询委员会|专家咨询委员会|团队成员|公司治理",
)
TEAM_NAME_ROLE_HINT = re.compile(
    r"[\u4e00-\u9fff]{2,4}\s*[，,、]?\s*"
    r"(博士|教授|院士|总经理|总监|主任|工程师|顾问|研究员|副教授|"
    r"首席执行官|总工程师|产品总监|技术总监|首席专家|特聘专家)",
)


def score_page_for_task(
    task_id: str,
    page_text: str,
    profile: Dict[str, Any],
    *,
    page_type: str = "",
) -> float:
    text = _clean_text(page_text)
    if not text:
        return 0.0

    lower = text.lower()
    score = 0.0

    for term in TASK_KEYWORDS.get(task_id, []):
        if term.lower() in lower:
            score += 2.0

    dim_map = {
        "problem": profile["evaluation_focus"]["problem"],
        "objectives": profile["evaluation_focus"]["objectives"],
        "methods": profile["methods"],
        "evidence": profile["methods"],
        "feasibility": profile["evaluation_focus"]["feasibility"] + profile["risks"],
        "innovation": profile["evaluation_focus"]["innovation"],
        "risks": profile["risks"],
        "team": profile["evaluation_focus"]["team"],
        "outcomes": profile["evaluation_focus"]["outcomes"],
    }

    for term in dim_map.get(task_id, []):
        if term.lower() in lower:
            score += 1.2

    numbers = re.findall(r"\d+(?:\.\d+)?", text)
    if task_id in {"evidence", "feasibility", "outcomes"}:
        score += min(2.0, len(numbers) * 0.15)

    if task_id == "team" and any(x in text for x in ["博士", "研究员", "教授", "工程师", "expert"]):
        score += 2.0

    if task_id == "team":
        if TEAM_SECTION_HEADERS.search(page_text):
            score += 5.5
        score += min(4.5, len(TEAM_NAME_ROLE_HINT.findall(page_text)) * 0.85)

    if task_id == "innovation" and any(x in text for x in ["唯一", "领先", "首台", "first", "unique"]):
        score += 1.5

    mult = evidence_page_multiplier(task_id, page_type, page_text)
    return score * mult


def select_evidence(
    task_id: str,
    pages: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
    top_k: int = 4,
) -> List[EvidenceSnippet]:
    eff_k = max(top_k, 5) if task_id == "team" else top_k
    scored = []
    for page in pages:
        raw = page.get("text") or ""
        pt = (page.get("page_type") or "").strip()
        score = score_page_for_task(task_id, raw, profile, page_type=pt)
        if score > 0:
            scored.append(
                EvidenceSnippet(
                    page_index=int(page["page_index"]),
                    text=_clean_text(raw)[:520],
                    score=score,
                )
            )
    scored.sort(key=lambda x: (x.score, -x.page_index), reverse=True)
    chosen = scored[:eff_k]

    if task_id == "team":
        header_idxs: List[int] = []
        for page in pages:
            raw = page.get("text") or ""
            if TEAM_SECTION_HEADERS.search(raw) or (
                TEAM_NAME_ROLE_HINT.search(raw) and ("职责" in raw or "团队" in raw)
            ):
                header_idxs.append(int(page["page_index"]))
        have = {c.page_index for c in chosen}
        for page in pages:
            pi = int(page["page_index"])
            if pi in header_idxs and pi not in have:
                chosen.append(
                    EvidenceSnippet(
                        page_index=pi,
                        text=_clean_text(page.get("text") or "")[:520],
                        score=1.8,
                    )
                )
                have.add(pi)
            if len(chosen) >= eff_k:
                break
        chosen.sort(key=lambda x: (x.score, -x.page_index), reverse=True)
        chosen = chosen[:eff_k]

    return chosen


def feature_flags(pages: Sequence[Dict[str, Any]], full_text: str) -> Dict[str, bool]:
    return {
        "has_team": bool(
            re.search(
                r"团队|核心团队|组织架构|咨询委员会|博士|教授|院士|首席执行官|总工程师|首席专家|研究员",
                full_text,
            )
        ),
        "has_timeline": bool(re.search(r"20\d{2}|timeline|roadmap|交付|研发路径", full_text)),
        "has_budget": bool(re.search(r"融资|预算|资金|万元|million|cash|profit", full_text)),
        "has_market": bool(re.search(r"市场|customer|订单|demand|销量|sales", full_text)),
        "has_prototype": bool(re.search(r"样机|prototype|测试|验证|patent|专利", full_text)),
        "has_risk_signal": bool(re.search(r"风险|瓶颈|challenge|uncertainty|dependency|竞争", full_text)),
        "has_metrics": bool(len(re.findall(r"\d+(?:\.\d+)?", full_text)) >= 10),
    }


def score_task(task_id: str, evidences: Sequence[EvidenceSnippet], flags: Dict[str, bool]) -> Tuple[float, float]:
    base = 5.0
    evidence_strength = min(2.0, sum(min(ev.score, 3.0) for ev in evidences) / 6.0)
    score = base + evidence_strength
    conf = 0.45 + min(0.45, len(evidences) * 0.08)

    if task_id == "team" and flags["has_team"]:
        score += 1.0
        conf += 0.06
    if task_id == "outcomes" and flags["has_market"] and flags["has_budget"]:
        score += 0.6
    if task_id == "feasibility" and flags["has_timeline"]:
        score += 0.6
    if task_id in {"methods", "evidence"} and flags["has_prototype"]:
        score += 0.8
    if task_id == "risks" and flags["has_risk_signal"]:
        score += 0.6
    if task_id == "innovation" and flags["has_metrics"]:
        score += 0.2

    if task_id == "outcomes" and not flags["has_market"]:
        score -= 1.1
    if task_id == "feasibility" and not flags["has_timeline"]:
        score -= 1.0
    if task_id == "team" and not flags["has_team"]:
        score -= 1.2
    if task_id in {"methods", "evidence"} and not flags["has_prototype"]:
        score -= 1.2
    if task_id == "risks" and not flags["has_risk_signal"]:
        score -= 0.8

    score = max(1.0, min(9.5, round(score, 1)))
    conf = max(0.2, min(0.95, round(conf, 2)))
    return score, conf


TASK_TEMPLATES = {
    "problem": (
        "The proposal appears to target a concrete problem area with visible context and motivation.",
        "The motivation would be stronger with clearer articulation of urgency, constraints, and why this is the right solution path now.",
    ),
    "objectives": (
        "The objectives are broadly understandable and tied to an identifiable work direction.",
        "The proposal would benefit from sharper success criteria, scope boundaries, and milestone-level deliverables.",
    ),
    "methods": (
        "The technical or methodological path is visible and internally coherent at a high level.",
        "The document still leaves gaps around assumptions, validation thresholds, and detailed implementation logic.",
    ),
    "evidence": (
        "The proposal includes tangible supporting signals such as prototypes, tests, prior work, or enabling resources.",
        "The evidence base remains incomplete for a decisive judgment because end-to-end validation and independent proof points are limited.",
    ),
    "feasibility": (
        "The execution pathway is plausible and partially supported by roadmap and resource signals.",
        "Feasibility remains exposed to delivery, integration, scaling, and dependency risks that are not fully closed.",
    ),
    "innovation": (
        "The proposal makes a credible differentiation claim relative to standard alternatives.",
        "The novelty case would be stronger with side-by-side benchmarking against the best current alternatives.",
    ),
    "risks": (
        "The proposal contains signals of awareness around technical and operational constraints.",
        "Risk treatment is not yet fully mature: several major failure modes are implied rather than explicitly mitigated.",
    ),
    "team": (
        "The team profile suggests relevant expertise and some execution capability.",
        "The commercial, manufacturing, or cross-functional governance setup is less developed than the technical story.",
    ),
    "outcomes": (
        "The expected outcomes and impact direction are understandable and potentially meaningful.",
        "Commercial or practical outcomes still rely on assumptions that need firmer validation and customer conversion evidence.",
    ),
}


def build_task_assessment(
    task_id: str,
    task_title: str,
    prompt_text: str,
    evidences: Sequence[EvidenceSnippet],
    score: float,
    confidence: float,
    metric_report: Dict[str, Any],
) -> Dict[str, Any]:
    pages = [ev.page_index for ev in evidences]
    pages_unique = sorted({ev.page_index for ev in evidences})
    cite_tail = ""
    if pages_unique:
        cite_tail = "（可核对：PDF 第 " + "、".join(str(p) for p in pages_unique[:4]) + " 页摘录）"

    if evidences:
        cite_parts = [
            f"第{ev.page_index}页「{_short_citation_quote(ev.text, 100)}」"
            for ev in evidences[:3]
        ]
        judgment = "依据摘录：" + "；".join(cite_parts)
    else:
        judgment = "在材料中未检索到与该维度强相关的连续段落。"

    strengths: List[str] = []
    weaknesses: List[str] = []

    if evidences:
        strengths.append(f"检索到与维度相关的材料，页码：{', '.join(map(str, pages[:4]))}。")
        strengths.append("材料中含有可用于核对的具体文段，而非仅有概括性表述。")

    if score >= 7.5:
        strengths.append("该维度相对其他维度证据更充分。")

    tail = cite_tail if evidences else ""

    if score <= 6.0:
        weaknesses.append("证据与结论之间的对应关系仍不够紧密。" + tail)

    if task_id == "outcomes":
        weaknesses.append("成果与影响的主张仍需更明确的可度量指标支撑。" + tail)
    elif task_id == "feasibility":
        weaknesses.append("实施路径与里程碑的验证材料仍可加强。" + tail)
    elif task_id == "risks":
        weaknesses.append("风险识别后的缓释措施与责任分工可写得更具体。" + tail)
    elif task_id == "team":
        if evidences:
            weaknesses.append("团队与治理结构虽已着墨，但与交付物、风险承担仍可更紧密对齐。" + tail)
        else:
            weaknesses.append("团队能力与分工的证明材料不足，难以独立核实。")
    else:
        weaknesses.append("该维度尚可补充更直接的论证与材料。" + tail)

    evidence_level = metric_report.get("metric_signals", {}).get("evidence_level", "weak")

    if evidence_level == "weak":
        weaknesses.append("量化与第三方佐证偏弱，影响判断把握度。" + tail)
    elif evidence_level == "moderate":
        weaknesses.append("量化依据已有雏形，但说服力仍可加强。" + tail)

    missing: List[str] = []

    metric_missing = metric_report.get("metric_signals", {}).get("missing_metric_signals", [])
    missing.extend(metric_missing[:3])

    missing.extend([
        "Explicit measurable success criteria",
        "Clear assumptions or boundary conditions",
    ])

    # 去重
    seen = set()
    dedup_missing = []
    for item in missing:
        key = item.lower().strip()
        if key and key not in seen:
            seen.add(key)
            dedup_missing.append(item)

    return {
        "task_id": task_id,
        "title": task_title,
        "prompt": prompt_text,
        "judgment": judgment,
        "strengths": strengths[:3],
        "weaknesses": weaknesses[:3],
        "missing_information": dedup_missing[:5],
        "confidence": confidence,
        "score_10": score,
        "evidence": [ev.__dict__ for ev in evidences],
    }


def aggregate_dimension_scores(task_results: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    dim_scores: Dict[str, List[float]] = defaultdict(list)
    for task, result in zip(REVIEW_TASKS, task_results):
        dim_scores[task.dimension].append(float(result["score_10"]))
    return {dim: round(sum(vals) / len(vals), 2) for dim, vals in dim_scores.items()}


# 评审结论档位（机器侧英文键）— 对应常见函评用语，见 verdict_label_zh()
VERDICT_PRIORITY_SUPPORT = "PRIORITY_SUPPORT"
VERDICT_SUPPORT = "SUPPORT"
VERDICT_CONDITIONAL = "CONDITIONAL"
VERDICT_CONCERN = "CONCERN"
# 旧版兼容（读历史 JSON 时可能仍出现）
_VERDICT_LEGACY_HOLD = "HOLD"


def verdict_label_zh(verdict: str) -> str:
    """与 compute_final_verdict 输出一致的中文结论用语（用于报告与提示词）。"""
    v = (verdict or "").strip()
    table = {
        VERDICT_PRIORITY_SUPPORT: "建议优先资助",
        VERDICT_SUPPORT: "建议资助",
        VERDICT_CONDITIONAL: "建议修改后再审（暂缓资助）",
        VERDICT_CONCERN: "不建议资助",
        _VERDICT_LEGACY_HOLD: "建议修改后再审（暂缓资助）",
    }
    return table.get(v, "建议修改后再审（暂缓资助）")


def compute_final_verdict(task_results: Sequence[Dict[str, Any]]) -> Tuple[float, float, str]:
    """
    综合分 = 各任务 score_10（约 1–9.5）算术平均；置信度 = 各任务 confidence 算术平均。

    四档（更接近函评习惯）：
    - 建议优先资助：显著高分且把握大
    - 建议资助：达到资助线且把握足够
    - 建议修改后再审：中等偏上但未达资助线，或分数够但置信不足、或证据不足以定夺
    - 不建议资助：综合偏弱或把握过低

    阈值可按实际数据再调；当前为经验分位，不是任何官方文件规定。
    """
    scores = [float(r["score_10"]) for r in task_results]
    confs = [float(r["confidence"]) for r in task_results]
    overall = round(sum(scores) / len(scores), 2) if scores else 0.0
    confidence = round(sum(confs) / len(confs), 2) if confs else 0.0

    # 把握过低：不按“资助”档出结论
    if confidence < 0.48:
        return overall, confidence, VERDICT_CONCERN

    if overall < 5.85:
        return overall, confidence, VERDICT_CONCERN

    # 建议优先资助
    if overall >= 8.55 and confidence >= 0.76:
        return overall, confidence, VERDICT_PRIORITY_SUPPORT

    # 建议资助（达到线且置信达标）
    if overall >= 7.68 and confidence >= 0.69:
        return overall, confidence, VERDICT_SUPPORT

    # 分数上接近资助线但置信不足 → 常见函评：材料/依据不足以直接定“资助”
    if overall >= 7.68 and confidence < 0.69:
        return overall, confidence, VERDICT_CONDITIONAL

    # 中等区间：有基础但需补充、修改后再议（含原 HOLD 语义）
    if overall >= 6.1:
        return overall, confidence, VERDICT_CONDITIONAL

    # 5.85–6.1：偏弱
    return overall, confidence, VERDICT_CONCERN


def generate_final_review(result, proposal_id="Unknown", domain="Unknown"):
    def score_to_level(score):
        if score >= 0.8:
            return "较高"
        elif score >= 0.6:
            return "中等偏上"
        elif score >= 0.4:
            return "中等"
        else:
            return "较低"

    def generate_overall_comment(result):
        score = result.get("overall_score", 0)

        obj = result.get("objectives", {}).get("summary", "")
        strat = result.get("strategy", {}).get("summary", "")
        innov = result.get("innovation", {}).get("summary", "")
        feas = result.get("feasibility", {}).get("summary", "")

        # 去掉太原始/证据型文本
        def clean(x):
            if not x:
                return ""
            x = x.replace("Based on evidence:", "")
            return x.strip()

        obj = clean(obj)
        strat = clean(strat)
        innov = clean(innov)
        feas = clean(feas)

        # 分数 -> 总体判断
        if score >= 0.8:
            level = "整体表现较好"
        elif score >= 0.6:
            level = "整体表现中等偏上"
        elif score >= 0.4:
            level = "整体表现一般"
        else:
            level = "整体表现较弱"

        # 动态拼接（关键）
        parts = []

        if obj:
            parts.append(f"在研究目标方面，{obj}")
        if strat:
            parts.append(f"在技术路线方面，{strat}")
        if innov:
            parts.append(f"在创新性方面，{innov}")
        if feas:
            parts.append(f"在可行性方面，{feas}")

        parts.append(f"总体来看，项目{level}，但仍存在进一步优化空间。")

        return "。".join(parts) + "。"

    def summarize_dimension(text, dim_name):
        if not text:
            return "该部分描述较为有限"

        if dim_name == "team":
            return "团队具备一定研究基础，但人员结构与分工说明仍不够充分"
        elif dim_name == "objectives":
            return "研究目标较为明确，但可量化程度有待提升"
        elif dim_name == "strategy":
            return "技术路线整体较为清晰，但部分关键环节需进一步细化"
        elif dim_name == "innovation":
            return "项目具有一定创新性，但创新点仍需进一步突出"
        elif dim_name == "feasibility":
            return "项目具备一定实施基础，但可行性论证仍需加强"
        else:
            return "该部分内容较为完整，但仍有提升空间"

    def generate_strengths():
        return [
            "项目整体结构较为完整，涵盖研究目标、技术路线与实施方案",
            "技术路线具有一定逻辑性，具备基础可行性",
            "在相关领域具有一定探索性，体现出一定创新潜力"
        ]

    def generate_weaknesses():
        return [
            "研究目标缺乏明确的量化指标，难以进行效果评估",
            "技术路线部分关键环节描述不够具体，存在一定不确定性",
            "创新点与现有研究的差异性阐述不够充分",
            "可行性分析偏弱，缺乏实验或数据支撑",
            "风险识别与应对机制尚不完善"
        ]

    def generate_suggestions():
        return [
            "建议细化研究目标，增加可量化评价指标",
            "建议完善技术路线描述，补充关键步骤说明",
            "建议加强与现有研究的对比分析，突出创新点",
            "建议补充实验验证或数据支撑，提高方案可信度",
            "建议完善风险评估与应对策略"
        ]

    overall_score = result.get("overall_score", 0)
    confidence = result.get("confidence", 0)
    verdict = result.get("verdict", VERDICT_CONDITIONAL)
    verdict_zh = result.get("verdict_label_zh") or verdict_label_zh(str(verdict))

    team = result.get("team", {})
    obj = result.get("objectives", {})
    strat = result.get("strategy", {})
    innov = result.get("innovation", {})
    feas = result.get("feasibility", {})

    strengths = generate_strengths()
    weaknesses = generate_weaknesses()
    suggestions = generate_suggestions()

    now = datetime.now().strftime("%Y-%m-%d")

    report = f"""
# 科研项目评审报告

## 一、项目基本信息
项目名称：{proposal_id}
申报领域：{domain}
评审时间：{now}

## 二、总体评价
{generate_overall_comment(result)}

## 三、总体评分
综合评分：{round(overall_score * 10, 2)} / 10  
评审置信度：{round(confidence, 3)}  
评审结论：{verdict_zh}

## 四、分项评分

| 评审维度 | 得分 | 评价 |
|----------|------|------|
| 研究团队 | {team.get("score", 0):.3f} | {summarize_dimension("", "team")} |
| 研究目标 | {obj.get("score", 0):.3f} | {summarize_dimension("", "objectives")} |
| 技术路线 | {strat.get("score", 0):.3f} | {summarize_dimension("", "strategy")} |
| 创新性 | {innov.get("score", 0):.3f} | {summarize_dimension("", "innovation")} |
| 可行性 | {feas.get("score", 0):.3f} | {summarize_dimension("", "feasibility")} |

## 五、主要优点
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(strengths)]) if strengths else "暂无明显优点总结"}

## 六、主要问题
{chr(10).join([f"{i+1}. {w}" for i, w in enumerate(weaknesses)]) if weaknesses else "暂无明显问题"}

## 七、修改建议
{chr(10).join([f"{i+1}. {s}" for i, s in enumerate(suggestions)]) if suggestions else "暂无具体建议"}

## 八、最终结论
综合以上分析，本项目整体达到{score_to_level(overall_score)}水平。评审结论：**{verdict_zh}**。
"""

    return report


def llm_generate_review(
    result: Dict[str, Any],
    proposal_id: str,
    domain: str,
    *,
    staleness: Dict[str, Any] | None = None,
    evidence_digest: str = "",
    document_form_primary: str = "unknown",
    quality_notes: Dict[str, Any] | None = None,
    task_results: Sequence[Dict[str, Any]] | None = None,
) -> str:
    client = OpenAI()
    model = (os.getenv("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini").strip()

    st = staleness or {}
    warn = st.get("warning_zh") or ""
    digest = evidence_digest.strip() or "（本轮未形成结构化摘录）"
    warn_block = warn if warn else "（未检测到需特别提示的时效风险）"
    form_note = document_form_primary or "unknown"
    vlabel = result.get("verdict_label_zh") or verdict_label_zh(str(result.get("verdict", "")))
    qn_block = _format_quality_notes_for_prompt(quality_notes or {})
    page_rule = (
        format_evidence_page_allowlist(task_results)
        if task_results is not None
        else "（未传入任务摘录；请勿编造页码括号。）"
    )

    prompt = f"""
你是一位科研项目与产业化材料评审专家，请基于以下结构化评审结果，撰写一份正式、规范、严谨的评审报告。

【严格要求】
1. 必须使用中文
2. 必须使用正式评审报告语气（严谨、客观、有判断）
3. 除表格内必要符号外，避免使用英文句子或英文段落
4. 不允许出现“根据数据/系统/模型/算法”等字样
5. 不允许输出JSON或解释过程
6. 必须使用标准科研评审结构（如下）
7. 「评审结论」一栏必须**逐字**填写为：**{vlabel}**（系统判定），禁止使用 PRIORITY_SUPPORT、SUPPORT、CONDITIONAL、CONCERN、HOLD 等英文代号，不得擅自改成其他档位用语。
8. 「最终结论」必须与「评审结论」一致，且只能采用与 **{vlabel}** 匹配的正式收尾（勿与别的档位混用）：
   - 若为「建议优先资助」：可强调项目突出、建议优先考虑资助等，不得写成仅“建议资助”或否定性结论。
   - 若为「建议资助」：可表述为建议予以资助或正式等价用语，不得写成“不建议资助”或“仅修改后再审”。
   - 若为「建议修改后再审（暂缓资助）」：须体现修改、补充后再评议或暂缓资助之意，**不得**写成“建议资助”或“建议优先资助”。
   - 若为「不建议资助」：须明确否定性结论，不得写成建议资助。
9. 「主要问题」「修改建议」中：凡写「（第N页）」，N **必须**属于下方【页码引用边界】所允许集合，且与该条判断在「证据摘录索引」中有对应关系；**禁止**虚构页码。**若无合适页码**，请省略括号，改用「见申请材料相关章节」等表述。
10. 「质量自检提示」若列出具体条目，请在「主要问题」「修改建议」中酌情吸纳（仍须遵守页码边界）；若无必要可忽略。**不得**因自检提示削弱或抬升已给定的评审结论档位（**{vlabel}**）。
11. 「## 二、总体评价」须与评审结论档位语气一致：若结论为「建议修改后再审（暂缓资助）」，第一段须在客观肯定（若有）之外，明确写出**主要短板、论证缺口或不确定性**，不得写成近乎全额肯定的结案式褒扬；若结论为「不建议资助」，须体现**主要负面依据方向**；若为「建议资助」或「建议优先资助」，褒扬须与档位匹配且不自相矛盾。
12. 若【页码引用边界】已列出**非空**允许页码，则「主要问题」「修改建议」中凡能与「证据摘录索引」某一条目**直接对应**的，**应**在句末写「（第N页）」（N 在允许集合内）；**避免**所有条目一律只用「见申请材料相关章节」等泛称。确无对应摘录的条目可保留泛称；**全文日期/时效性**类可写「（材料全文日期特征）」或不写页码。
13. 「七、修改建议」与「六、主要问题」须**同序号一一对应**（第1条对第1条，依此类推）。**每条修改建议**句末的括号指称须与**同序号**那条「主要问题」**保持一致**：若该问题使用「（第N页）」或「（见申请材料相关章节）」等，对应建议句末**必须**沿用**相同**页码或相同泛称，**禁止**同序号下「有问题无页码、建议又不写页码」或「问题与建议页码不一致」。
14. 「五、主要优点」须保持**评审人**的克制与可核对性：**禁止**照搬申报书中的营销口号或未经验证的夸张表述（如「行业领先者」「颠覆性」「国际领先」等），除非「证据摘录索引」能直接支撑。涉及院士、头衔、荣誉、团队规模等**强断言**时，宜用「申报材料显示…」「材料中列有…」等限定表述，避免写成已核实的事实判断。

【质量自检提示（系统生成，供你写作时斟酌）】
{qn_block}

【材料形态（自动识别，供你调整侧重点）】
{form_note}

【时效性提示（须写入报告）】
若下列内容不是“未检测到…”，你必须把其核心含义写在「## 二、总体评价」**内部**：第一段写整体判断；第二段专门写材料时效性（可起首句“关于材料时效性：”）。**禁止**使用「二点五」等非整数章节标题，**禁止**为时效性再增加新的 `##` 级别标题。这是审慎性提示，不是直接否决理由。
{warn_block}

【证据摘录索引（撰写问题时须引用页码）】
{digest}

【页码引用边界（须严格遵守）】
{page_rule}

【报告结构（必须严格遵守）】

# 科研项目评审报告

## 一、项目基本信息
项目名称：{proposal_id}
申报领域：{domain}

## 二、总体评价
（共一至两段：第一段完整总结目标、方法、创新、可行性并给出整体判断；若上方「时效性提示」需要特别提示，**须另起一段**写材料时效性（段首可用「关于材料时效性：」），**与第一段之间空一行**，不得把时效性挤在第一段同一段落末尾。若无需特别提示，则仅保留第一段。本节之后直接接「## 三、总体评分」，中间不得插入其他 `##` 小节。）

## 三、总体评分
- 综合评分：X / 10（注意：必须转为10分制）
- 评审置信度：X.XX
- 评审结论：{vlabel}

## 四、分项评分
（必须为表格形式）

| 评审维度 | 得分（0-1） | 评价 |
|----------|------------|------|
| 研究团队 | {result.get("team", {}).get("score", 0)} | （一句评价） |
| 研究目标 | {result.get("objectives", {}).get("score", 0)} | （一句评价） |
| 技术路线 | {result.get("strategy", {}).get("score", 0)} | （一句评价） |
| 创新性 | {result.get("innovation", {}).get("score", 0)} | （一句评价） |
| 可行性 | {result.get("feasibility", {}).get("score", 0)} | （一句评价） |

## 五、主要优点
（3–5条；专家语气须**客观、克制**，见上文第14条：不写空泛口号，强断言加材料限定。）

## 六、主要问题
（3–5条，必须具体、有判断）

## 七、修改建议
（与第六条**同序号严格对应**；每条句末的「（第N页）」或「（见申请材料相关章节）」等须与**同序号**主要问题条目**完全一致**。）

## 八、最终结论
（**2–4 句**：先概括主要依据与主要短板或风险，再给出与「评审结论」**逐字一致**的收尾用语；结论必须为 **{vlabel}**，不得仅写一句话敷衍。）

【输入数据】
综合评分（0-1）：{result.get("overall_score")}
置信度：{result.get("confidence")}

Team: {result.get("team")}
Objectives: {result.get("objectives")}
Strategy: {result.get("strategy")}
Innovation: {result.get("innovation")}
Feasibility: {result.get("feasibility")}
"""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content


def generate_markdown_report(
    proposal_id: str,
    file_path: Path,
    profile: Dict[str, Any],
    task_results: Sequence[Dict[str, Any]],
    dim_scores: Dict[str, float],
    overall: float,
    confidence: float,
    verdict: str,
    profiler_mode: str,
) -> str:
    lines: List[str] = []
    lines.append(f"# Yangtze Review Report — {proposal_id}")
    lines.append("")
    lines.append(f"- Source file: `{file_path.name}`")
    lines.append(f"- Profiler mode: `{profiler_mode}`")
    lines.append(f"- Overall score: **{overall:.2f} / 10**")
    lines.append(f"- Confidence: **{confidence:.2f}**")
    lines.append(f"- Verdict: **{verdict}** ({verdict_label_zh(verdict)})")
    lines.append("")
    lines.append("## Domain Profile")
    lines.append("")
    lines.append(f"- Primary domain: **{profile['domain']['primary']}**")
    if profile["domain"]["secondary"]:
        lines.append(f"- Secondary tags: {', '.join(profile['domain']['secondary'])}")
    lines.append(f"- Methods: {', '.join(profile['methods']) or 'N/A'}")
    lines.append(f"- Risks: {', '.join(profile['risks']) or 'N/A'}")
    lines.append(f"- Terminology: {', '.join(profile['terminology']) or 'N/A'}")
    lines.append("")
    lines.append("## Dimension Scores")
    lines.append("")
    for dim in ["team", "objectives", "strategy", "innovation", "feasibility"]:
        value = dim_scores.get(dim)
        if value is not None:
            lines.append(f"- **{dim}**: {value:.2f} / 10")
    lines.append("")

    for result in task_results:
        lines.append(f"## {result['title']}")
        lines.append("")
        lines.append(f"**Score:** {result['score_10']:.1f} / 10  ")
        lines.append(f"**Confidence:** {result['confidence']:.2f}")
        lines.append("")
        lines.append(f"**Prompt used**: {result['prompt']}")
        lines.append("")
        lines.append(result["judgment"])
        lines.append("")
        lines.append("**Strengths**")
        for item in result["strengths"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Weaknesses**")
        for item in result["weaknesses"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Missing information**")
        for item in result["missing_information"]:
            lines.append(f"- {item}")
        lines.append("")
        lines.append("**Evidence excerpts**")
        if result["evidence"]:
            for ev in result["evidence"]:
                snippet = _clean_text(ev["text"])
                lines.append(f"- Page {ev['page_index']}: {snippet}")
        else:
            lines.append("- No strong evidence matched this task.")
        lines.append("")

    strongest = sorted(task_results, key=lambda r: r["score_10"], reverse=True)[:3]
    weakest = sorted(task_results, key=lambda r: r["score_10"])[:3]

    lines.append("## Executive Summary")
    lines.append("")
    lines.append("### Stronger areas")
    for item in strongest:
        lines.append(f"- {item['title']} ({item['score_10']:.1f}/10)")
    lines.append("")
    lines.append("### Weaker areas")
    for item in weakest:
        lines.append(f"- {item['title']} ({item['score_10']:.1f}/10)")
    lines.append("")

    if verdict == VERDICT_PRIORITY_SUPPORT:
        rec = "Strong signals across dimensions; priority funding recommendation if institutional checks align."
    elif verdict == VERDICT_SUPPORT:
        rec = "Supportable as presented; proceed with standard due diligence."
    elif verdict in (VERDICT_CONDITIONAL, _VERDICT_LEGACY_HOLD):
        rec = "Credible potential but needs targeted revision, clarification, and/or stronger evidence before a positive decision."
    else:
        rec = "Substantial strengthening required; not ready for a positive funding recommendation as written."

    lines.append(f"**Recommendation:** {rec}")
    lines.append("")
    return "\n".join(lines)


def run_review(file_path: Path, proposal_id: str | None = None, use_ocr: bool = True) -> Dict[str, Any]:
    file_path = file_path.resolve()
    proposal_id = proposal_id or _safe_name(file_path.stem)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    prep = prepare_text(file_path=file_path, proposal_id=proposal_id, use_ocr=use_ocr)
    pages = load_pages(proposal_id)
    full_text_path = Path(prep["full_text_path"])
    full_text = full_text_path.read_text(encoding="utf-8", errors="ignore")
    staleness = detect_document_staleness(full_text)

    run_dir = RUNS_DIR / proposal_id
    run_dir.mkdir(parents=True, exist_ok=True)

    profile, profiler_mode = profile_with_optional_llm(full_text, pages)
    (run_dir / "domain_profile.json").write_text(
        json.dumps(profile, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metric_report = build_metric_report(full_text, pages)
    (run_dir / "metric_report.json").write_text(
        json.dumps(metric_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metric_prompt_suffix = build_metric_prompt_suffix(metric_report)

    prompt_items = []
    task_results = []
    flags = feature_flags(pages, full_text)

    for task in REVIEW_TASKS:
        base_prompt_text = build_specialized_question(task, profile)
        form_suffix = get_document_form_prompt_suffix(profile)
        prompt_text = f"{base_prompt_text} {metric_prompt_suffix}{form_suffix}"
        top_k = 5 if task.task_id == "team" else 4
        evidences = select_evidence(task.task_id, pages, profile, top_k=top_k)
        score, conf = score_task(task.task_id, evidences, flags)

        result = build_task_assessment(
            task.task_id,
            task.title,
            prompt_text,
            evidences,
            score,
            conf,
            metric_report,
        )

        prompt_items.append({
            "task_id": task.task_id,
            "template_id": task.template_id,
            "dimension": task.dimension,
            "title": task.title,
            "prompt": prompt_text,
            "search_hints": QUESTION_SEARCH_HINTS.get(task.dimension, []),
        })
        task_results.append(result)

    (run_dir / "prompts.json").write_text(
        json.dumps(prompt_items, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "task_results.json").write_text(
        json.dumps(task_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    dim_scores = aggregate_dimension_scores(task_results)
    overall, confidence, verdict = compute_final_verdict(task_results)

    quality_notes = llm_review_quality_notes(task_results, staleness, metric_report)

    review_json = {
        "proposal_id": proposal_id,
        "source_file": str(file_path),
        "profiler_mode": profiler_mode,
        "overall_score_10": overall,
        "confidence": confidence,
        "verdict": verdict,
        "verdict_label_zh": verdict_label_zh(verdict),
        "dimension_scores": dim_scores,
        "domain_profile": profile,
        "document_form": profile.get("document_form", {}),
        "staleness": staleness,
        "metric_report": metric_report,
        "quality_notes": quality_notes,
        "tasks": task_results,
    }
    (run_dir / "review.json").write_text(
        json.dumps(review_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def _dim_bucket(dim: str) -> Dict[str, Any]:
        dim_tasks = [t for t in task_results if any(rt.dimension == dim and rt.task_id == t["task_id"] for rt in REVIEW_TASKS)]
        score_10 = dim_scores.get(dim, 0.0)
        summary = ""
        if dim_tasks:
            top = sorted(dim_tasks, key=lambda x: x.get("score_10", 0), reverse=True)[0]
            summary = top.get("judgment", "")[:120]
        return {
            "score": score_10 / 10.0,
            "summary": summary,
            "strengths": [s for t in dim_tasks for s in t.get("strengths", [])],
            "weaknesses": [w for t in dim_tasks for w in t.get("weaknesses", [])],
        }

    result = {
        "overall_score": overall / 10.0,
        "confidence": confidence,
        "verdict": verdict,
        "verdict_label_zh": verdict_label_zh(verdict),
        "team": _dim_bucket("team"),
        "objectives": _dim_bucket("objectives"),
        "strategy": _dim_bucket("strategy"),
        "innovation": _dim_bucket("innovation"),
        "feasibility": _dim_bucket("feasibility"),
    }
    domain = profile.get("domain", {}).get("primary", "Unknown")
    evidence_digest = build_evidence_digest_for_llm(task_results)
    doc_form = profile.get("document_form", {}) or {}
    final_report = llm_generate_review(
        result,
        proposal_id,
        domain,
        staleness=staleness,
        evidence_digest=evidence_digest,
        document_form_primary=str(doc_form.get("primary", "unknown")),
        quality_notes=quality_notes,
        task_results=task_results,
    )
    print(final_report)

    report_path = run_dir / f"{proposal_id}_review_report.md"
    report_path.write_text(final_report, encoding="utf-8")

    with open(f"report_{proposal_id}.md", "w", encoding="utf-8") as f:
        f.write(final_report)

    return {
        "proposal_id": proposal_id,
        "run_dir": str(run_dir),
        "report_path": str(report_path),
        "review_json_path": str(run_dir / "review.json"),
        "overall_score_10": overall,
        "confidence": confidence,
        "verdict": verdict,
        "verdict_label_zh": verdict_label_zh(verdict),
        "profiler_mode": profiler_mode,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Yangtze domain-adaptive review pipeline end to end.")
    ap.add_argument("--file", required=True, help="Path to PDF/DOCX/TXT/MD proposal file")
    ap.add_argument("--proposal_id", default="", help="Optional proposal ID for output directories")
    ap.add_argument("--no_ocr", action="store_true", help="Disable OCR fallback during PDF text preparation")
    args = ap.parse_args()

    result = run_review(
        Path(args.file), 
        proposal_id=args.proposal_id.strip() or None,
        use_ocr=not args.no_ocr,
    )


if __name__ == "__main__":
    main()