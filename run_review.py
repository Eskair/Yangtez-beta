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
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from dotenv import load_dotenv
from openai import OpenAI

from src.tools.domain_profiler import profile_with_optional_llm
from src.prompting.domain_adaptive import (
    QUESTION_SEARCH_HINTS,
    REVIEW_TASKS,
    build_specialized_question,
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


def _safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]+", "_", name).strip("_") or "proposal"


def load_pages(proposal_id: str) -> List[Dict[str, Any]]:
    path = DATA_DIR / "prepared" / proposal_id / "pages.json"
    return json.loads(path.read_text(encoding="utf-8"))


TASK_KEYWORDS = {
    "problem": ["背景", "需求", "意义", "motivation", "pain point", "problem", "行业背景"],
    "objectives": ["目标", "scope", "deliverable", "计划", "路径", "objective"],
    "methods": ["方法", "技术路线", "样机", "验证", "test", "prototype", "method"],
    "evidence": ["样机", "测试", "专利", "数据", "成果", "validation", "evidence"],
    "feasibility": ["交付", "timeline", "roadmap", "风险", "量产", "feasibility"],
    "innovation": ["唯一", "领先", "创新", "novel", "differentiation", "突破"],
    "risks": ["风险", "瓶颈", "dependency", "cost", "寿命", "竞争", "risk"],
    "team": ["团队", "依托单位", "博士", "研究员", "roles", "team"],
    "outcomes": ["市场", "订单", "收入", "profit", "impact", "evaluation", "融资"],
}


def score_page_for_task(task_id: str, page_text: str, profile: Dict[str, Any]) -> float:
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

    if task_id == "innovation" and any(x in text for x in ["唯一", "领先", "首台", "first", "unique"]):
        score += 1.5

    return score


def select_evidence(
    task_id: str,
    pages: Sequence[Dict[str, Any]],
    profile: Dict[str, Any],
    top_k: int = 4,
) -> List[EvidenceSnippet]:
    scored = []
    for page in pages:
        score = score_page_for_task(task_id, page.get("text", ""), profile)
        if score > 0:
            scored.append(
                EvidenceSnippet(
                    page_index=int(page["page_index"]),
                    text=_clean_text(page["text"])[:420],
                    score=score,
                )
            )
    scored.sort(key=lambda x: (x.score, -x.page_index), reverse=True)
    return scored[:top_k]


def feature_flags(pages: Sequence[Dict[str, Any]], full_text: str) -> Dict[str, bool]:
    return {
        "has_team": bool(re.search(r"团队|leader|博士|教授|研究员", full_text)),
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
    snippets = [ev.text[:120] for ev in evidences[:2] if ev.text]

    if snippets:
        evidence_text = " | ".join(snippets)
        judgment = f"Based on evidence: {evidence_text}"
    else:
        judgment = "Limited direct evidence found in the document."

    strengths: List[str] = []
    weaknesses: List[str] = []

    if evidences:
        strengths.append(f"Evidence located on pages {', '.join(map(str, pages[:3]))}.")
        if snippets:
            strengths.append("The proposal includes concrete supporting details rather than only high-level claims.")

    if score >= 7.5:
        strengths.append("This dimension is relatively strong compared to others.")

    # ❗减少模板味，增加针对性
    if score <= 6.0:
        weaknesses.append("Evidence is insufficient or not clearly connected to claims.")

    if task_id == "outcomes":
        weaknesses.append("Outcome claims are not tightly linked to measurable metrics.")
    elif task_id == "feasibility":
        weaknesses.append("Execution feasibility lacks detailed validation or timeline support.")
    elif task_id == "risks":
        weaknesses.append("Risk mitigation strategies are not sufficiently detailed.")
    elif task_id == "team":
        weaknesses.append("Team capability is not strongly evidenced by concrete roles or experience.")
    else:
        weaknesses.append("This aspect could be made more explicit and evidence-supported.")

    evidence_level = metric_report.get("metric_signals", {}).get("evidence_level", "weak")

    if evidence_level == "weak":
        weaknesses.append("Quantitative evidence is weak, reducing confidence.")
    elif evidence_level == "moderate":
        weaknesses.append("Quantitative support exists but is not fully convincing.")

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


def compute_final_verdict(task_results: Sequence[Dict[str, Any]]) -> Tuple[float, float, str]:
    scores = [float(r["score_10"]) for r in task_results]
    confs = [float(r["confidence"]) for r in task_results]
    overall = round(sum(scores) / len(scores), 2) if scores else 0.0
    confidence = round(sum(confs) / len(confs), 2) if confs else 0.0

    if overall >= 7.8 and confidence >= 0.7:
        verdict = "SUPPORT"
    elif overall >= 6.0:
        verdict = "HOLD"
    else:
        verdict = "CONCERN"

    return overall, confidence, verdict


def generate_final_review(result, proposal_id="Unknown", domain="Unknown"):
    from datetime import datetime

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
    verdict = result.get("verdict", "HOLD")

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
评审结论：{verdict}

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
综合以上分析，本项目整体达到{score_to_level(overall_score)}水平，建议：{verdict}。
"""

    return report


def llm_generate_review(result, proposal_id, domain):
    client = OpenAI()

    prompt = f"""
你是一位国家自然科学基金评审专家，请基于以下结构化评审结果，撰写一份正式、规范、严谨的科研项目评审报告。

【严格要求】
1. 必须使用中文
2. 必须使用正式评审报告语气（严谨、客观、有判断）
3. 不允许出现英文
4. 不允许出现“根据数据/系统/模型”等字样
5. 不允许输出JSON或解释过程
6. 必须使用标准科研评审结构（如下）
7. 最终结论必须与“评审结论”字段严格一致，不得出现前后矛盾。
8. 当评审结论为 HOLD 时，最终结论应表述为“建议暂缓资助”或“建议补充完善后再评估”，不得写为“建议资助”。

【报告结构（必须严格遵守）】

# 科研项目评审报告

## 一、项目基本信息
项目名称：{proposal_id}
申报领域：{domain}

## 二、总体评价
（必须为一段完整文字，总结目标、方法、创新、可行性，并给出整体判断）

## 三、总体评分
- 综合评分：X / 10（注意：必须转为10分制）
- 评审置信度：X.XX
- 评审结论：{result.get("verdict")}

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
（3–5条，必须是专家评价语气）

## 六、主要问题
（3–5条，必须具体、有判断）

## 七、修改建议
（必须与问题一一对应）

## 八、最终结论
（必须是正式评审结论语言，如：建议资助 / 有条件资助 / 暂缓资助 / 不建议资助）

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
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
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
    lines.append(f"- Verdict: **{verdict}**")
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

    if verdict == "SUPPORT":
        rec = "The proposal is supportable as presented, though normal diligence is still recommended."
    elif verdict == "HOLD":
        rec = "The proposal has credible potential but should move forward only with targeted clarification, validation, and execution-risk reduction."
    else:
        rec = "The proposal requires substantial strengthening before it would be ready for a positive funding or approval decision."

    lines.append(f"**Recommendation:** {rec}")
    lines.append("")
    return "\n".join(lines)


def run_review(file_path: Path, proposal_id: str | None = None, use_ocr: bool = True) -> Dict[str, Any]:
    file_path = file_path.resolve()
    proposal_id = proposal_id or _safe_name(file_path.stem)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    prep = prepare_text(file_path=file_path, proposal_id=proposal_id, use_ocr=use_ocr)
    print("=== DEBUG PAGE 3 TEXT ===")
    print(prep.get("reconstructed_full_text", "")[:2000])
    pages = load_pages(proposal_id)
    full_text_path = Path(prep["full_text_path"])
    full_text = full_text_path.read_text(encoding="utf-8", errors="ignore")

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
        prompt_text = f"{base_prompt_text} {metric_prompt_suffix}"
        evidences = select_evidence(task.task_id, pages, profile, top_k=4)
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

    review_json = {
        "proposal_id": proposal_id,
        "source_file": str(file_path),
        "profiler_mode": profiler_mode,
        "overall_score_10": overall,
        "confidence": confidence,
        "verdict": verdict,
        "dimension_scores": dim_scores,
        "domain_profile": profile,
        "metric_report": metric_report,
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
        "team": _dim_bucket("team"),
        "objectives": _dim_bucket("objectives"),
        "strategy": _dim_bucket("strategy"),
        "innovation": _dim_bucket("innovation"),
        "feasibility": _dim_bucket("feasibility"),
    }
    domain = profile.get("domain", {}).get("primary", "Unknown")
    final_report = llm_generate_review(result, proposal_id, domain)
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