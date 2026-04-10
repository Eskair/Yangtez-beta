# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from src.prompting.domain_adaptive import (
    PROFILER_SYSTEM_PROMPT,
    sanitize_document_form,
    sanitize_domain_profile,
)

try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def extract_keywords(text: str, max_terms: int = 18) -> List[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z\-\+\.]{2,}|[\u4e00-\u9fff]{2,}", text)
    stop = {
        "proposal", "project", "system", "technology", "market", "company", "team", "development",
        "研究", "项目", "公司", "系统", "技术", "团队", "市场", "发展", "方案", "计划", "应用",
        "产品", "核心", "页码", "介绍", "融资", "背景", "价值",
    }
    counts = Counter(tok.lower() for tok in tokens if tok.lower() not in stop)
    return [t for t, _ in counts.most_common(max_terms)]


def _heuristic_document_form(text: str) -> Dict[str, Any]:
    t = text[:80000]
    if re.search(r"融资计划|商业计划书|A轮|B轮|路演|股权|估值|BP\b", t, re.I):
        return {"primary": "business_plan", "confidence": 0.55, "rationale": "financing / BP keywords"}
    if re.search(r"国家自然科学基金|重点项目|面上项目|青年基金|科学问题属性|研究基础与可行性", t):
        return {"primary": "grant_proposal", "confidence": 0.55, "rationale": "NSFC-style keywords"}
    if re.search(r"可行性研究报告|可研报告|投资估算|建设规模|经济评价", t):
        return {"primary": "feasibility_study", "confidence": 0.5, "rationale": "feasibility-study keywords"}
    if re.search(r"技术白皮书|技术报告|试验结果|测试报告|附录A", t):
        return {"primary": "technical_report", "confidence": 0.4, "rationale": "technical-report keywords"}
    return {"primary": "unknown", "confidence": 0.0, "rationale": ""}


def heuristic_domain_profile(full_text: str, pages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Strictly derive a lightweight domain profile from the CURRENT document only.
    No hard-coded domain catalog, no historical/default domain injection.
    """
    text = _clean_text(full_text)
    keywords = extract_keywords(text)

    generic_terms = {
        "model", "models", "system", "systems", "method", "methods",
        "analysis", "data", "study", "research", "approach",
        "based", "using", "results", "validation", "evaluation"
    }

    top_keywords = [kw for kw in keywords if kw and kw.lower() not in generic_terms][:8]

    if not top_keywords:
        return sanitize_domain_profile({
            "document_form": _heuristic_document_form(full_text),
            "domain": {"primary": "unknown", "secondary": []},
            "evaluation_focus": {
                "problem": [],
                "objectives": [],
                "feasibility": [],
                "innovation": [],
                "team": [],
                "outcomes": [],
            },
            "methods": [],
            "risks": [],
            "terminology": [],
        })

    primary_domain = top_keywords[0]
    secondary_domains = top_keywords[1:4]

    profile = {
        "document_form": _heuristic_document_form(full_text),
        "domain": {
            "primary": primary_domain,
            "secondary": secondary_domains,
        },
        "evaluation_focus": {
            "problem": [
                "problem importance",
                "context clarity",
                "need definition",
            ],
            "objectives": [
                "objective clarity",
                "scope definition",
                "deliverable specificity",
            ],
            "feasibility": [
                "resource readiness",
                "timeline realism",
                "dependency management",
            ],
            "innovation": [
                "novelty",
                "differentiation",
                "comparative advantage",
            ],
            "team": [
                "relevant expertise",
                "role coverage",
                "execution capability",
            ],
            "outcomes": [
                "measurable outputs",
                "impact logic",
                "evaluation plan",
            ],
        },
        "methods": top_keywords[:4],
        "risks": [],
        "terminology": top_keywords[:8],
    }

    return sanitize_domain_profile(profile)


def profile_with_optional_llm(full_text: str, pages: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    heuristic = heuristic_domain_profile(full_text, pages)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return heuristic, "heuristic"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=DEFAULT_MODEL,
            messages=[
                {"role": "system", "content": PROFILER_SYSTEM_PROMPT},
                {"role": "user", "content": full_text[:16000]},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=900,
        )

        raw = response.choices[0].message.content or "{}"
        parsed = json.loads(raw)
        data = sanitize_domain_profile(parsed)
        if data.get("document_form", {}).get("primary") == "unknown":
            h_form = _heuristic_document_form(full_text)
            if h_form.get("primary") != "unknown":
                data = {**data, "document_form": sanitize_document_form(h_form)}

        primary = _clean_text(data.get("domain", {}).get("primary", ""))
        terminology = data.get("terminology", []) or []
        methods = data.get("methods", []) or []
        risks = data.get("risks", []) or []

        bad_primary = {
            "",
            "general",
            "general proposal",
            "proposal",
            "research proposal",
            "project proposal",
            "unknown",
        }

        def is_fragment_list(lst):
            if not lst:
                return True
            short = len([x for x in lst if len(x.strip()) <= 1])
            return short >= max(1, len(lst) * 0.6)

        def is_punctuation_noise(lst):
            return any(re.fullmatch(r"[、,;；\.\-]+", x.strip()) for x in lst)

        # 🚨 强力质量拦截
        if (
            primary.lower() in bad_primary
            or is_fragment_list(methods)
            or is_fragment_list(risks)
            or is_punctuation_noise(methods)
            or is_punctuation_noise(risks)
        ):
            return heuristic, "heuristic_fallback_bad_quality"

        return data, "llm"

    except Exception:
        return heuristic, "heuristic_fallback"