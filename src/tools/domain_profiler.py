# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from src.prompting.domain_adaptive import sanitize_domain_profile

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


LLM_PROFILER_SYSTEM = """
You are a strict domain profiler for proposal review.

Your task is ONLY to derive a clean domain profile from the CURRENT document text.

Hard rules:
- Use ONLY the provided document text.
- Ignore any prior tasks, prior files, templates, examples, cached memory, or default domains.
- Do NOT guess a domain from common proposal patterns.
- Do NOT mix domains unless the document clearly and repeatedly supports both.
- If the evidence is weak, return "unknown".
- Prefer short, concrete domain labels grounded in repeated terms from the text.
- Do NOT output aerospace, biomedical, or any specific field unless clearly supported by repeated explicit terms.
- Keep all lists short and text-grounded.
- Return valid JSON only.

Return JSON with exactly this structure:
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
  "terminology": ["string"]
}
""".strip()


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
                {"role": "system", "content": LLM_PROFILER_SYSTEM},
                {"role": "user", "content": full_text[:16000]},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=900,
        )

        raw = response.choices[0].message.content or "{}"
        data = sanitize_domain_profile(json.loads(raw))

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