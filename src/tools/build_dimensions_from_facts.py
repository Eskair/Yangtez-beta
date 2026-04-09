# -*- coding: utf-8 -*-
"""
Stage 2 · 维度构建器（build_dimensions_from_facts.py）
----------------------------------------------------
输入：
  - src/data/extracted/<proposal_id>/raw_facts.jsonl  （Stage 1 输出）

输出：
  - src/data/extracted/<proposal_id>/dimensions_v2.json      （新的五维度文件）
  - src/data/extracted/<proposal_id>/dimension_facts.json    （按维度分组的 facts，方便后续调试与复用）
  - src/data/parsed/parsed_dimensions.clean.llm.json         （供 llm_answering 使用的全局 parsed 文件）

核心职责：
  - 按 dimensions 标签把 facts 分桶到 team/objectives/strategy/innovation/feasibility
  - 对每个维度单独调用一次 LLM，只基于该维度的事实生成：
      summary / key_points / risks / mitigations
  - 严禁凭空造事实，所有内容必须能在事实列表中找到“影子”
  - 尽量覆盖该维度下出现过的不同 type（team_member/pipeline/market/risk/...）
"""

import os
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 假设本文件路径：<project_root>/src/tools/build_dimensions_from_facts.py
BASE_DIR = Path(__file__).resolve().parents[2]
EXTRACTED_DIR = BASE_DIR / "src" / "data" / "extracted"
PARSED_DIR = BASE_DIR / "src" / "data" / "parsed"   # 与 llm_answering 对齐

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

DIMENSION_NAMES = ["team", "objectives", "strategy", "innovation", "feasibility"]

# 全局 client 复用
client = OpenAI()

# 注意：这里用 {dimension_name} 标记占位，其它所有 { } 都是字面量 JSON 示例
# 后面用 .replace("{dimension_name}", xxx) 而不是 .format()

DIMENSION_PROMPT_TEMPLATE = """
你是一个严谨的项目评审助手，现在要基于【已经抽取好的事实列表】为某一个维度生成结构化摘要。

硬性约束：
1）只能使用我给你的 facts，不得凭空添加新的机构、人名、对象、方法、数字、时间、结果或结论。
2）允许压缩与合并，但每一个 key_point、risk、mitigation 都必须能在 facts 中找到依据。
3）只能总结【当前维度】的内容；若需要提及其他维度的信息，也只能在与当前维度强相关时简短引用。
4）不要写行业常识、通用背景或空泛判断，只写从 facts 能支持的内容。

【当前维度】：{dimension_name}

【维度含义提醒】
- team：团队、机构、角色分工、协作、治理
- objectives：问题背景、目标、范围、里程碑、交付物、评价指标
- strategy：方法、技术路线、实施流程、验证设计、外部协作路径
- innovation：新颖性、差异化、独特资源、已有证据、知识产权
- feasibility：资源、预算、时间安排、风险、限制、应对、执行条件

输入 payload 结构：
payload = {
  "all_facts": [{"text": "...", "dimensions": ["team"], "type": "team_member", "meta": {...}}, ...],
  "risk_facts": [...],
  "mitigation_facts": [...]
}

请输出 JSON：
{
  "summary": "2-4 句，概括该维度当前情况",
  "key_points": ["3-10 条，不重复、尽量覆盖不同主题"],
  "risks": ["基于 risk_facts 或相关 facts 总结；若信息不足，可明确写‘该维度风险信息较少/未详细说明’"],
  "mitigations": ["基于 mitigation_facts 或相关 facts 总结；若信息不足，可明确写‘该维度风险应对措施未详细说明’"]
}

补充要求：
- facts 足够多时，优先覆盖不同 type，而不是重复同一类信息。
- 不要输出任何 JSON 以外的文字。
"""


def load_raw_facts(proposal_id: str) -> List[Dict[str, Any]]:
    path = EXTRACTED_DIR / proposal_id / "raw_facts.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"raw_facts.jsonl 不存在，请先运行 extract_facts_by_chunk.py: {path}")

    facts: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                facts.append(obj)
    print(f"[INFO] 读取 facts 数量: {len(facts)} 来自 {path}")
    return facts


def group_facts_by_dimension(facts: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    按维度标签把 facts 分桶到五个维度。
    一条 fact 可能属于多个维度，会出现在多个桶里。
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {dim: [] for dim in DIMENSION_NAMES}

    for fact in facts:
        dims = fact.get("dimensions", [])
        if not isinstance(dims, list):
            continue
        for dim in dims:
            if dim in grouped:
                grouped[dim].append(fact)

    for dim in DIMENSION_NAMES:
        print(f"[INFO] 维度 {dim} 相关事实数: {len(grouped[dim])}")
    return grouped


def sort_facts_for_dimension(dimension_name: str, facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按维度定义 type 优先级排序，保证更关键信息排在前面。
    这是纯通用规则，不依赖具体提案内容。
    """
    priority_map = {
        "team": [
            # 先看具体成员履历，再看组织结构和协作模式
            "team_member", "org_structure", "collaboration", "resource", "other"
        ],
        "objectives": [
            # 目标、里程碑、验证设计与交付信息优先
            "milestone", "pipeline", "clinical_design", "product", "market", "other"
        ],
        "strategy": [
            # 方法路线、外部协作、应用环境与制度流程都属于 strategy 视角
            "tech_route", "product", "collaboration", "market",
            "funding_source", "regulatory", "other"
        ],
        "innovation": [
            "ip_asset", "evidence", "ai_model", "tech_route", "product", "other"
        ],
        "feasibility": [
            "resource", "budget_item", "funding_source",
            "risk", "mitigation", "regulatory", "other"
        ],
    }
    order = priority_map.get(dimension_name, ["other"])

    def type_rank(t: str) -> int:
        return order.index(t) if t in order else len(order)

    return sorted(
        facts,
        key=lambda f: type_rank(f.get("type", "other"))
    )


def truncate_facts_for_prompt(facts: List[Dict[str, Any]], max_chars: int = 10000) -> List[Dict[str, Any]]:
    """
    为了防止单次 prompt 爆 context，对 facts 做一个简单的字符长度截断。
    按顺序累加 text，超过 max_chars 就停（meta 仍然保留）。
    """
    kept: List[Dict[str, Any]] = []
    total = 0
    for fact in facts:
        t = fact.get("text", "") or ""
        t_len = len(t)
        if total + t_len > max_chars and kept:
            break
        kept.append(fact)
        total += t_len
    return kept


# ===== 辅助：基于文本再兜底识别 risk / mitigation =====

_RISK_CN = ["风险", "挑战", "瓶颈", "不确定性", "不足", "局限", "缺陷", "障碍", "难点"]
_RISK_EN = ["risk", "risks", "challenge", "challenges", "bottleneck", "bottlenecks",
            "uncertainty", "limitation", "limitations", "weakness", "weaknesses",
            "barrier", "barriers", "issue", "issues", "difficulty", "difficulties"]

_MITIG_CN = ["应对", "缓解", "降低", "减少", "解决", "克服", "应对措施", "改进", "优化", "管控"]
_MITIG_EN = ["mitigation", "mitigate", "mitigating", "address", "addresses", "addressing",
             "solve", "solves", "solving", "overcome", "overcoming",
             "reduce", "reduces", "reducing", "decrease", "decreases", "decreasing",
             "improve", "improves", "improving", "optimize", "optimizing", "optimization"]


def _looks_like_risk(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in text for k in _RISK_CN):
        return True
    if any(k in t for k in _RISK_EN):
        return True
    return False

def reclassify_risk_mitigation_global(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    在进入维度构建之前，对所有 facts 做一次全局的 risk/mitigation 纠偏：
    - 如果 type 不是 risk/mitigation，但文本明显是风险/挑战，就改成 risk
    - 如果 type 不是 risk/mitigation，但文本明显是应对/缓解方案，就改成 mitigation
    - 如果本身标成 risk/mitigation 但文本看起来不像风险/应对，则降级为 other
      （必要时在 risk 和 mitigation 之间互换）
    """
    new_facts: List[Dict[str, Any]] = []

    for f in facts:
        t = f.get("type", "other") or "other"
        txt = f.get("text", "") or ""

        # 优先纠错：如果已经标成 risk/mitigation，但文本不符合，就降级/互换
        if t == "risk":
            if not _looks_like_risk(txt):
                # 文本更像应对措施，就改成 mitigation；否则降级成 other
                if _looks_like_mitigation(txt):
                    t = "mitigation"
                else:
                    t = "other"
        elif t == "mitigation":
            if not _looks_like_mitigation(txt):
                # 文本更像风险描述，就改成 risk；否则降级成 other
                if _looks_like_risk(txt):
                    t = "risk"
                else:
                    t = "other"
        else:
            # 如果原始类型既不是 risk 也不是 mitigation，就尝试“升格”
            if _looks_like_risk(txt):
                t = "risk"
            elif _looks_like_mitigation(txt):
                t = "mitigation"

        f["type"] = t
        new_facts.append(f)

    return new_facts

def _looks_like_mitigation(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if any(k in text for k in _MITIG_CN):
        return True
    if any(k in t for k in _MITIG_EN):
        return True
    return False


def call_llm_for_dimension(dimension_name: str, facts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对单一维度调用一次 LLM，生成 summary/key_points/risks/mitigations。
    强约束：只能基于 facts，不得脑补。
    """

    # 1) 先按 type 排序，再截断，确保更重要的信息优先被看到
    sorted_facts = sort_facts_for_dimension(dimension_name, facts)
    all_facts_for_prompt = truncate_facts_for_prompt(sorted_facts, max_chars=10000)

    # 2) 识别风险 / 对策 facts：不仅看 type，还看文本关键词
    risk_facts: List[Dict[str, Any]] = []
    mitigation_facts: List[Dict[str, Any]] = []

    for f in all_facts_for_prompt:
        t = f.get("type", "")
        txt = f.get("text", "") or ""

        # 明确标成 risk 的，或者文本看起来是在描述风险/挑战，都归入
        if t == "risk" or _looks_like_risk(txt):
            risk_facts.append(f)

        # 明确标成 mitigation 的，或者文本看起来是在描述应对/解决方案，也归入
        if t == "mitigation" or _looks_like_mitigation(txt):
            mitigation_facts.append(f)

    payload = {
        "all_facts": all_facts_for_prompt,
        "risk_facts": risk_facts,
        "mitigation_facts": mitigation_facts,
    }
    facts_json_str = json.dumps(payload, ensure_ascii=False, indent=2)

    # 用 replace，而不是 format，避免 JSON 里的 { } 被当成占位符
    prompt = DIMENSION_PROMPT_TEMPLATE.replace("{dimension_name}", dimension_name)

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的项目评审助手，只能基于给定的事实列表进行总结，不得编造。",
        },
        {
            "role": "user",
            "content": prompt + "\n\n=== payload 开始 ===\n" + facts_json_str,
        },
    ]

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=2600,
    )
    raw = resp.choices[0].message.content

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"[WARN] 维度 {dimension_name} JSON 解析失败，原始内容如下：")
        print(raw)
        raise e

    # 兜底：保证四个字段存在
    summary = data.get("summary", "")
    if not isinstance(summary, str):
        summary = ""
    key_points = data.get("key_points", [])
    if not isinstance(key_points, list):
        key_points = []
    risks = data.get("risks", [])
    if not isinstance(risks, list):
        risks = []
    mitigations = data.get("mitigations", [])
    if not isinstance(mitigations, list):
        mitigations = []

    # 轻量兜底：facts 足够多但 key_points 太少，打日志提醒（先不强制重试）
    fact_count = len(all_facts_for_prompt)
    if fact_count >= 20 and len(key_points) < 6:
        print(
            f"[WARN] 维度 {dimension_name}: all_facts={fact_count} 但 key_points 只有 {len(key_points)} 条，"
            f"如有需要可以在此处加重试/补点逻辑。"
        )

    return {
        "summary": summary.strip(),
        "key_points": [str(x).strip() for x in key_points if str(x).strip()],
        "risks": [str(x).strip() for x in risks if str(x).strip()],
        "mitigations": [str(x).strip() for x in mitigations if str(x).strip()],
    }


def run_build(proposal_id: str):
    facts = load_raw_facts(proposal_id)
    # 全局先做一次 risk/mitigation 纠偏
    facts = reclassify_risk_mitigation_global(facts)
    grouped = group_facts_by_dimension(facts)

    # 额外输出一份 dimension_facts.json，方便后续人工检查和调试
    out_dir = EXTRACTED_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    dim_facts_path = out_dir / "dimension_facts.json"
    dim_facts_path.write_text(
        json.dumps(grouped, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 已写出按维度分组的 facts: {dim_facts_path}")

    dimensions_result: Dict[str, Dict[str, Any]] = {}

    for dim in DIMENSION_NAMES:
        dim_facts = grouped.get(dim, [])
        print(f"\n[INFO] 开始构建维度 {dim} ...")

        if not dim_facts:
            # 没有任何事实，给一个空壳（显式指出信息缺失）
            dimensions_result[dim] = {
                "summary": f"提案文本中关于 {dim} 维度的明确信息较少，无法做出详细总结。",
                "key_points": [],
                "risks": [f"提案中关于 {dim} 维度的细节信息较少，可能影响评估。"],
                "mitigations": ["提案未具体说明如何补充或缓解该维度信息不足的问题。"],
            }
            print(f"[INFO] 维度 {dim} 无事实，写入占位结果。")
            continue

        data = call_llm_for_dimension(dim, dim_facts)

        # ==== 风险覆盖度标记 ====
        risk_count = len(data.get("risks", []) or [])
        if risk_count == 0:
            level = "low"
            reason = "提案文本中几乎没有显式描述该维度相关的风险，系统无法进行充分的风险细化。"
        elif risk_count <= 2:
            level = "medium"
            reason = "该维度仅有少量风险相关描述，风险分析的粒度有限。"
        else:
            level = "high"
            reason = "该维度在提案中有较为丰富的风险相关描述，可以进行较细致的风险分析。"

        data["risk_coverage"] = {
            "level": level,
            "reason": reason,
            "risk_count": risk_count,
        }

        dimensions_result[dim] = data

        print(
            f"[INFO] 维度 {dim} 完成：summary_len={len(data['summary'])}, "
            f"key_points={len(data['key_points'])}, risks={len(data['risks'])}, "
            f"mitigations={len(data['mitigations'])}, "
            f"risk_coverage={level}"
        )

    # 1) 写入 per-proposal 维度文件
    out_path = out_dir / "dimensions_v2.json"
    out_path.write_text(
        json.dumps(dimensions_result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[OK] 新版五维度文件已生成: {out_path}")

    # 2) 同时写一份全局 parsed 文件给 llm_answering 用
    PARSED_DIR.mkdir(parents=True, exist_ok=True)
    parsed_path = PARSED_DIR / "parsed_dimensions.clean.llm.json"

    parsed_obj = {
        dim: {
            "summary": data.get("summary", ""),
            "key_points": data.get("key_points", []),
            "risks": data.get("risks", []),
            "mitigations": data.get("mitigations", []),
        }
        for dim, data in dimensions_result.items()
    }

    parsed_path.write_text(
        json.dumps(parsed_obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[OK] 已写出 parsed 维度文件供 llm_answering 使用: {parsed_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 2: 基于 raw_facts.jsonl 构建五个维度的 dimensions_v2.json"
    )
    parser.add_argument(
        "--proposal_id",
        required=False,
        help="提案 ID（对应 src/data/extracted/<proposal_id>）",
    )
    args = parser.parse_args()

    if args.proposal_id:
        pid = args.proposal_id
    else:
        # 默认用 extracted 里最新的一个子目录
        if not EXTRACTED_DIR.exists():
            raise FileNotFoundError(f"未找到 extracted 目录: {EXTRACTED_DIR}")
        candidates = [
            (d.stat().st_mtime, d.name)
            for d in EXTRACTED_DIR.iterdir()
            if d.is_dir()
        ]
        if not candidates:
            raise FileNotFoundError(f"extracted 目录下没有任何子目录: {EXTRACTED_DIR}")
        pid = max(candidates, key=lambda x: x[0])[1]
        print(f"[INFO] [auto] 选中最新提案 ID: {pid}")

    run_build(pid)


if __name__ == "__main__":
    main()
