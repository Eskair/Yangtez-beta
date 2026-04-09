# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Sequence


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _unique_keep_order(items: List[str], max_items: int = 20) -> List[str]:
    seen = set()
    out = []
    for item in items:
        key = item.strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item.strip())
        if len(out) >= max_items:
            break
    return out


def extract_numeric_spans(text: str) -> List[str]:
    """
    Extract common quantitative expressions from proposal text.
    Covers:
    - integers / decimals
    - percentages
    - money / financing
    - years / year-ranges
    - count-like targets
    """
    text = _clean_text(text)

    patterns = [
        r"\d+(?:\.\d+)?%",                         # 50%
        r"\d+(?:\.\d+)?\s*%",                     # 50 %
        r"\d+(?:\.\d+)?万",                       # 5500万
        r"\d+(?:\.\d+)?亿",                       # 3亿
        r"\d+(?:\.\d+)?万元",                     # 500万元
        r"\d+(?:\.\d+)?亿元",                     # 3亿元
        r"\d{4}\s*[-–—]\s*\d{4}",                 # 2018-2020
        r"\d{4}年(?:\d{1,2}月)?",                 # 2019年 / 2019年10月
        r"第[一二三四五六七八九十0-9]+年",          # 第一年
        r"\d+(?:\.\d+)?倍",                       # 2倍
        r"\d+(?:\.\d+)?种",                       # 20种
        r"\d+(?:\.\d+)?个",                       # 4个
        r"\d+(?:\.\d+)?项",                       # 70项
        r"\d+(?:\.\d+)?辆",                       # 5万辆
        r"\d+(?:\.\d+)?家",                       # 10家
        r"\d+(?:\.\d+)?台",                       # 3台
        r"\d+(?:\.\d+)?套",                       # 4套
        r"\d+(?:\.\d+)?人",                       # 20人
        r"\d+(?:\.\d+)?次",                       # 3次
        r"\d+(?:\.\d+)?(?:\.\d+)?",              # fallback numbers
    ]

    spans: List[str] = []
    for pat in patterns:
        spans.extend(re.findall(pat, text))

    return _unique_keep_order(spans, max_items=40)


def detect_metric_signals(text: str) -> Dict[str, Any]:
    """
    Detect whether the document contains useful measurable evidence.
    """
    text = _clean_text(text)
    lower = text.lower()

    numeric_spans = extract_numeric_spans(text)

    signal_patterns = {
        "has_numbers": r"\d",
        "has_money": r"(融资|预算|资金|投入|销售额|利润|成本|收入|亿元?|万元?|million|billion|revenue|profit|cost)",
        "has_timeline": r"(20\d{2}|timeline|roadmap|阶段|起步|腾飞|验证|年|月)",
        "has_percentages": r"\d+(?:\.\d+)?\s*%",
        "has_targets": r"(目标|达到|覆盖|完成|形成|实现|sales|target|milestone|deliverable)",
        "has_benchmark_terms": r"(领先|首个|第一|benchmark|sota|state of the art|对比|优于|提升)",
        "has_validation_terms": r"(验证|测试|试验|仿真|实验|prototype|validation|test|pilot)",
        "has_financial_table_terms": r"(销售额|利润|成本|投入|融资计划|财务预测|现金流|revenue|profit|cost)",
        "has_scale_terms": r"(规模化|场景|覆盖|批量|市场占有率|用户数|deployment|adoption|scale)",
    }

    flags = {name: bool(re.search(pattern, text, flags=re.IGNORECASE)) for name, pattern in signal_patterns.items()}

    score = 0
    score += 2 if flags["has_numbers"] else 0
    score += 1 if flags["has_money"] else 0
    score += 1 if flags["has_timeline"] else 0
    score += 1 if flags["has_percentages"] else 0
    score += 1 if flags["has_targets"] else 0
    score += 1 if flags["has_benchmark_terms"] else 0
    score += 1 if flags["has_validation_terms"] else 0
    score += 1 if flags["has_financial_table_terms"] else 0
    score += 1 if flags["has_scale_terms"] else 0

    if score >= 8:
        evidence_level = "strong"
    elif score >= 5:
        evidence_level = "moderate"
    else:
        evidence_level = "weak"

    missing = []
    if not flags["has_numbers"]:
        missing.append("no explicit numeric evidence")
    if not flags["has_targets"]:
        missing.append("no clear measurable targets")
    if not flags["has_timeline"]:
        missing.append("no clear timeline or milestone evidence")
    if not flags["has_validation_terms"]:
        missing.append("no explicit validation or testing evidence")

    return {
        "flags": flags,
        "numeric_spans": numeric_spans[:30],
        "numeric_count": len(numeric_spans),
        "evidence_level": evidence_level,
        "missing_metric_signals": missing,
    }


def detect_possible_numeric_conflicts(text: str) -> List[Dict[str, Any]]:
    """
    Lightweight conflict detector.
    It does not prove contradictions mathematically,
    but surfaces repeated metric categories with divergent values.
    """
    text = _clean_text(text)

    categories = {
        "financing": r"(\d+(?:\.\d+)?(?:万|亿|万元|亿元))",
        "scenarios": r"(\d+(?:\.\d+)?种(?:以上)?应用场景)",
        "sales": r"(预计年销售额\s*\d+(?:\.\d+)?)",
        "profit": r"(预计年利润\s*-?\d+(?:\.\d+)?)",
        "years": r"(20\d{2}\s*[-–—]\s*\d{4}|20\d{2}年(?:\d{1,2}月)?)",
    }

    conflicts: List[Dict[str, Any]] = []

    for category, pattern in categories.items():
        matches = re.findall(pattern, text)
        uniq = _unique_keep_order(matches, max_items=10)
        if len(uniq) >= 3:
            conflicts.append({
                "category": category,
                "values": uniq,
                "note": "multiple values detected; review for consistency",
            })

    return conflicts


def build_metric_report(full_text: str, pages: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Stage 1.5 output artifact.
    """
    text = _clean_text(full_text)
    metric_signals = detect_metric_signals(text)
    conflicts = detect_possible_numeric_conflicts(text)

    page_hits = []
    for page in pages:
        page_text = _clean_text(page.get("text", ""))
        page_metrics = extract_numeric_spans(page_text)
        if page_metrics:
            page_hits.append({
                "page_index": int(page.get("page_index", -1)),
                "metric_count": len(page_metrics),
                "sample_metrics": page_metrics[:8],
            })

    page_hits = sorted(page_hits, key=lambda x: x["metric_count"], reverse=True)[:8]

    return {
        "stage": "1.5_metric_checker",
        "metric_signals": metric_signals,
        "possible_conflicts": conflicts,
        "top_metric_pages": page_hits,
    }


def build_metric_prompt_suffix(metric_report: Dict[str, Any]) -> str:
    """
    Convert metric detection result into prompt guidance for downstream review.
    """
    signals = metric_report.get("metric_signals", {})
    flags = signals.get("flags", {})
    evidence_level = signals.get("evidence_level", "weak")
    missing = signals.get("missing_metric_signals", [])
    numeric_spans = signals.get("numeric_spans", [])[:10]
    conflicts = metric_report.get("possible_conflicts", [])

    lines = []
    lines.append("Metric-check guidance:")
    lines.append(f"- Quantitative evidence level: {evidence_level}.")
    lines.append(f"- Numeric evidence detected: {'yes' if flags.get('has_numbers') else 'no'}.")
    lines.append(f"- Timeline evidence detected: {'yes' if flags.get('has_timeline') else 'no'}.")
    lines.append(f"- Validation evidence detected: {'yes' if flags.get('has_validation_terms') else 'no'}.")
    lines.append(f"- Financial evidence detected: {'yes' if flags.get('has_money') else 'no'}.")

    if numeric_spans:
        lines.append(f"- Example metric spans: {'; '.join(numeric_spans)}.")

    if missing:
        lines.append(f"- Missing metric signals: {'; '.join(missing)}.")
        lines.append("- If a dimension relies on quantitative justification but explicit metrics are missing, mention this clearly as a limitation.")

    if conflicts:
        lines.append("- Possible numeric consistency issues were detected; review repeated values carefully before making strong claims.")

    return " ".join(lines)