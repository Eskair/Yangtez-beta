# -*- coding: utf-8 -*-
"""
Stage 1 · 块级事实抽取器（extract_facts_by_chunk.py）
----------------------------------------------------
输入：
  - src/data/prepared/<proposal_id>/full_text.txt  （由 prepare_proposal_text.py 生成）

输出：
  - src/data/extracted/<proposal_id>/raw_facts.jsonl  每行一个 JSON fact

职责：
  - 将长文按字符切块（带 overlap）
  - 对每个块，用 LLM 抽取“原子事实列表”
  - 每条事实附带：dimensions[], type, meta(chunk_index, char_range 由代码填充)
  - 后续 Stage 2 再用这些 facts 去构建五个维度的最终 dimensions_v2.json

本版优化要点：
  1）减小 chunk 大小、加大 overlap，提高单块信息覆盖率，避免一个块塞太多信息导致抽不全。
  2）加强 Prompt 中对“五维覆盖”的要求，显式提醒模型不要只抽单一维度的信息。
  3）更激进的 type→dimensions 映射，让跨维度事实被多个维度同时看到，避免某维度信息过于稀薄。
  4）扩充关键词推断逻辑 _infer_dims_from_text，补充 objectives / innovation / feasibility 等隐性表述。
  5）在运行结束时输出五个维度的事实数量分布，并对明显偏少的维度给出警告，便于调参与排错。
  6）新增：对“文本很长但抽取事实过少”的 chunk 自动再跑一轮 dense 模式，强化召回。
"""

import os
import json
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None

from .layout_reconstruction import semantic_pages_for_stage1

load_dotenv()

# ========= 路径配置 =========

BASE_DIR = Path(__file__).resolve().parents[2]
PREPARED_DIR = BASE_DIR / "src" / "data" / "prepared"
EXTRACTED_DIR = BASE_DIR / "src" / "data" / "extracted"

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# 全局复用一个 OpenAI client，避免每个 chunk 反复初始化
client = OpenAI() if OpenAI is not None else None

VALID_DIMENSIONS = ["team", "objectives", "strategy", "innovation", "feasibility"]
VALID_TYPES = [
    "team_member",
    "org_structure",
    "collaboration",
    "resource",
    "pipeline",
    "milestone",
    "market",
    "tech_route",
    "product",
    "ip_asset",
    "evidence",
    "budget_item",
    "funding_source",
    "risk",
    "mitigation",
    "ai_model",
    "clinical_design",
    "regulatory",
    "other",
]

# ⚠️ Prompt：既要防幻觉，又要尽量“捞干净”五维相关信息，并保证多维覆盖

FACT_PROMPT = """
你是一个“事实抽取器”，负责从一小段项目文本中逐条抽取可核对的原子事实。

硬性约束：
1）只处理当前文本块，不猜测其他页面或上下文。
2）不得编造任何文本中没有出现的机构、人物、对象、方法、数据、数字、时间、地点或结果。
3）每条 fact 必须能在原文中找到对应依据，可以轻微改写，但必须保留关键名词。
4）每条 fact 尽量只表达一个独立事实；若一句话包含多个主题，请拆开。
5）只输出 JSON，不输出解释。

适用范围：
- 任意科研、技术、教育、产业、治理或跨学科项目
- 不得预设具体领域

优先抽取的信息：
- 团队与组织：成员、角色、机构、职责、协作关系、治理安排
- 目标与范围：总体目标、阶段目标、任务边界、里程碑、交付物、评价指标
- 方法与路线：技术路线、研究方法、实施步骤、验证设计、工作流、平台或工具
- 创新与差异化：新颖点、独特资源、已有证据、比较优势、知识产权
- 可行性与风险：资源基础、数据/设备/资金/时间条件、风险、限制、应对措施
- 外部环境：用户/对象/场景/应用环境/政策约束/市场与需求信息（若文本明确提及）

抽取粒度要求：
- 信息丰富时尽量抽取 12–25 条 facts；信息较少时可少于 10 条；总数不要超过 25 条。
- 尽量覆盖不同主题，不要只集中在单一维度。

维度标签（dimensions）：
- "team": 团队、角色、机构、分工、协同、治理
- "objectives": 问题背景、目标、范围、里程碑、交付物、评价指标
- "strategy": 方法、技术路线、实施流程、实验/验证/落地路径、外部协作策略
- "innovation": 新颖性、差异化、独特资源、证据优势、知识产权
- "feasibility": 资源、预算、时间、风险、限制、依赖关系、应对措施

要求：
- 每条 fact 至少有 1 个维度标签；允许多标签。
- 若无法判断，使用 ["feasibility"] 兜底。

type 可选值：
- team_member, org_structure, collaboration, resource, pipeline, milestone,
  market, tech_route, product, ip_asset, evidence, budget_item, funding_source,
  risk, mitigation, ai_model, clinical_design, regulatory, other

type 选择规则：
- 选择最接近文本语义的类型即可；若不确定，用 other。
- market / regulatory / clinical_design 等类型可以用于任何“外部约束、验证设计、制度流程”场景，
  不代表你必须假设项目属于某个特定行业。

输出格式：
{
  "facts": [
    {
      "text": "一条具体、可核对的事实",
      "dimensions": ["team", "strategy"],
      "type": "team_member"
    }
  ]
}

如果当前文本块没有任何有用事实，返回 {"facts": []}。
现在开始处理我给你的文本块。
"""


def find_latest_prepared_proposal() -> str:
    if not PREPARED_DIR.exists():
        raise FileNotFoundError(f"未找到 prepared 目录: {PREPARED_DIR}")

    candidates = []
    for d in PREPARED_DIR.iterdir():
        if d.is_dir():
            candidates.append((d.stat().st_mtime, d.name))

    if not candidates:
        raise FileNotFoundError(f"prepared 目录下没有任何提案子目录: {PREPARED_DIR}")

    proposal_id = max(candidates, key=lambda x: x[0])[1]
    print(f"[INFO] [auto] 选中最新提案 ID: {proposal_id}")
    return proposal_id




def load_page_semantics(proposal_id: str) -> Dict[str, Any]:
    path = PREPARED_DIR / proposal_id / "page_semantics.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"[INFO] 读取 page_semantics: {path}")
        return data
    except Exception as e:
        print(f"[WARN] page_semantics.json 读取失败，退回 full_text 模式: {e}")
        return {}


def load_semantic_units(proposal_id: str) -> List[Dict[str, Any]]:
    page_sem = load_page_semantics(proposal_id)
    if not page_sem:
        return []
    units = semantic_pages_for_stage1(page_sem)
    return [u for u in units if (u.get("text") or "").strip()]

def load_full_text(proposal_id: str) -> str:
    path = PREPARED_DIR / proposal_id / "full_text.txt"
    if not path.exists():
        raise FileNotFoundError(f"full_text.txt 不存在: {path}")
    text = path.read_text(encoding="utf-8")
    print(f"[INFO] 读取 full_text: {path} (长度 {len(text)} 字符)")
    return text


def make_chunks(text: str, max_chars: int = 1800, overlap: int = 400) -> List[Dict[str, Any]]:
    """
    简单按字符切块，带 overlap；不做句子级别切分。
    默认 max_chars=1800, overlap=400，比之前更细、更密，有利于提高抽取覆盖率。
    返回列表，每个元素包含: chunk_text, start, end, index
    """
    chunks = []
    n = len(text)
    if n == 0:
        return chunks

    idx = 0
    chunk_idx = 0
    while idx < n:
        end = min(n, idx + max_chars)
        chunk_text = text[idx:end]
        chunks.append(
            {
                "index": chunk_idx,
                "start": idx,
                "end": end,
                "text": chunk_text,
            }
        )
        chunk_idx += 1
        if end == n:
            break
        idx = max(0, end - overlap)

    print(f"[INFO] 已切分为 {len(chunks)} 个 chunk (max_chars={max_chars}, overlap={overlap})")
    return chunks


def make_semantic_chunks(units: List[Dict[str, Any]], max_chars: int = 2200) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    current_parts: List[str] = []
    current_pages: List[int] = []
    current_len = 0
    chunk_idx = 0

    def flush():
        nonlocal current_parts, current_pages, current_len, chunk_idx
        if not current_parts:
            return
        text = "\n\n".join(current_parts).strip()
        chunks.append({
            "index": chunk_idx,
            "start": 0,
            "end": len(text),
            "text": text,
            "page_indices": sorted(set(current_pages)),
            "source": "page_semantics",
        })
        chunk_idx += 1
        current_parts = []
        current_pages = []
        current_len = 0

    for unit in units:
        text = (unit.get("text") or "").strip()
        if not text:
            continue
        page_idx = unit.get("page_index")
        if current_len and current_len + len(text) + 2 > max_chars:
            flush()
        current_parts.append(text)
        if page_idx is not None:
            current_pages.append(page_idx)
        current_len += len(text) + 2
    flush()
    print(f"[INFO] 基于 page_semantics 构建 {len(chunks)} 个语义 chunk")
    return chunks


def call_llm_for_chunk(chunk_text: str, attempt: int = 1, dense: bool = False) -> Dict[str, Any]:
    """
    调用 OpenAI，对单个 chunk 抽取 facts。
    - attempt > 1：用于 JSON 解析失败后的重试（提示“上一次 JSON 不合法”）。
    - dense = True：用于“当前 chunk 文本很长但事实过少”的第二轮密集抽取，会额外要求多抽一些 facts。
    """
    extra_hint_parts = []

    if attempt > 1:
        extra_hint_parts.append(
            "⚠️ 注意：上一次你返回的 JSON 因为太长或不合法导致解析失败。"
            "这一次请严格控制 facts 数量不超过 18 条，并且务必保证 JSON 语法完全正确。"
        )

    if dense:
        extra_hint_parts.append(
            "⚠️ 当前文本块信息非常丰富，你在本次抽取时应尽量覆盖文本中出现的所有与团队、目标、"
            "策略、创新、可行性相关的关键事实。请优先抽取 15–22 条 facts，"
            "并尽量覆盖不同维度和不同主题，不要只聚焦在单一方面。"
        )

    extra_hint = ""
    if extra_hint_parts:
        extra_hint = "\n\n" + "\n".join(extra_hint_parts)

    messages = [
        {
            "role": "system",
            "content": "你是一个严谨的事实抽取器，只能基于给定文本块抽取原子事实，不得编造。",
        },
        {
            "role": "user",
            "content": FACT_PROMPT
            + extra_hint
            + "\n\n=== 文本块开始 ===\n"
            + chunk_text.strip(),
        },
    ]

    if client is None:
        raise RuntimeError("OpenAI SDK 未安装或不可用，无法执行 Stage 1 facts 抽取。")

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=1800,
    )

    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        print("[WARN] JSON 解析失败，返回原始内容，便于你排查：")
        print(raw)
        if attempt == 1:
            print("[INFO] 尝试使用收缩版 prompt 重试该 chunk ...")
            return call_llm_for_chunk(chunk_text, attempt=2, dense=dense)
        # 第二次还失败就直接抛出
        raise e

    if not isinstance(data, dict):
        data = {"facts": []}
    if "facts" not in data or not isinstance(data["facts"], list):
        data["facts"] = []

    return data

# ===== 市场类关键词 & 识别函数 =====

_MARKET_CN = [
    "市场", "市场规模", "市场容量", "市场需求", "市场前景", "市场潜力",
    "目标市场", "细分市场", "市场份额", "渗透率",
    "客户", "客户群体", "目标客户", "目标人群",
    "患者群体", "目标患者",
    "销售", "销售额", "销量", "营收", "收入", "收益",
    "定价", "价格", "报销", "支付方", "医保", "保险",
    "商业化", "商业模式", "商业机会",
    "竞争", "竞品", "竞争对手", "竞争格局",
    "CAGR", "增长率"
]

_MARKET_EN = [
    "market", "market size", "market volume", "market demand", "market potential",
    "target market", "segment", "niche",
    "market share", "penetration",
    "customer", "customers", "client", "clients",
    "patient population", "target patients",
    "sales", "revenue", "turnover", "income",
    "pricing", "price", "reimbursement", "payer", "insurance",
    "commercialization", "commercialisation", "business model",
    "competition", "competitive", "competitor", "competitors",
    "cagr", "growth rate"
]


def _looks_like_market_fact(text: str) -> bool:
    """
    判断一条 fact 的文本是否“明显是市场/商业相关”的内容。
    注意：这里只用来纠偏 type，对 risk/mitigation 不覆盖。
    """
    if not text:
        return False
    t = text.lower()

    # 中文关键字
    if any(k in text for k in _MARKET_CN):
        return True

    # 英文关键字
    if any(k in t for k in _MARKET_EN):
        return True

    return False

def _infer_dims_from_text(text: str) -> List[str]:
    """
    当 LLM 没有给出 dimensions 且 type 也无法可靠映射时，
    用中英文关键词做一轮粗略的维度推断，尽量不要丢掉有用信息。
    这里只做“补充”，不会覆盖已有维度。
    """
    t_lower = (text or "").lower()
    t = text or ""

    candidates = set()

    # ---- team ----
    team_keywords = [
        "团队", "小组", "联合体", "合作单位", "合作方", "协作单位",
        "研究者", "研究团队", "专业团队", "项目组",
        "负责人", "项目负责人", "带头人",
        "教授", "副教授", "主任", "专家", "研究员", "博士", "博士后",
        "机构", "大学", "研究所", "中心", "实验室",
        "ceo", "coo", "cto", "cso", "vp", "vice president",
        "founder", "co-founder", "chief executive officer",
    ]
    if any(k in t for k in team_keywords) or any(k in t_lower for k in team_keywords):
        candidates.add("team")

    # ---- objectives ----
    obj_keywords_cn = [
        "目标", "总体目标", "阶段性目标", "里程碑", "阶段性里程碑",
        "计划", "任务", "工作包", "kpi", "终点", "主要终点", "次要终点",
        "本项目旨在", "本项目将", "本项目计划", "预期达到", "希望实现",
    ]
    obj_keywords_en = [
        "aim", "aims to", "aimed to",
        "objective", "objectives", "goal", "goals",
        "milestone", "milestones", "endpoint", "endpoints",
        "is designed to", "seeks to", "intends to", "in order to",
    ]
    if any(k in t for k in obj_keywords_cn) or any(k in t_lower for k in obj_keywords_en):
        candidates.add("objectives")

    # ---- strategy ----
    strat_keywords_cn = [
        "策略", "路径", "路线", "方案", "技术路线", "实施方案",
        "商业模式", "市场进入", "商业化", "推广策略",
        "合作模式", "运营模式", "联合开发", "授权引进",
        "市场", "市场规模", "市场需求", "市场前景", "市场潜力",
        "市场份额", "竞争格局", "竞品", "竞争对手"
    ]
    strat_keywords_en = [
        "strategy", "strategies", "pathway", "roadmap",
        "commercial", "commercialization", "business model",
        "market entry", "go-to-market", "go to market",
        "market", "market size", "market demand", "market potential",
        "market share", "competitive landscape", "competition", "competitor", "competitors",
        "partnership", "licensing", "co-development",
        "development plan", "regulatory strategy",
    ]

    if any(k in t for k in strat_keywords_cn) or any(k in t_lower for k in strat_keywords_en):
        candidates.add("strategy")

    # ---- innovation ----
    inno_keywords_cn = [
        "创新", "创新性", "差异化", "独特", "首创", "领先", "颠覆",
        "新一代", "新型", "原创", "填补空白", "突破性", "首个", "第一例",
    ]
    inno_keywords_en = [
        "novel", "novelty", "innovative", "innovation",
        "differentiated", "differentiation", "unique",
        "first-in-class", "best-in-class", "state-of-the-art",
        "cutting-edge", "breakthrough", "original", "disruptive",
        "fills the gap", "fill the gap",
    ]
    if any(k in t for k in inno_keywords_cn) or any(k in t_lower for k in inno_keywords_en):
        candidates.add("innovation")

    # ---- feasibility ----
    feas_keywords_cn = [
        "可行性", "可行", "可实施", "资源", "平台", "基础设施",
        "预算", "经费", "资金", "成本", "成本负担",
        "风险", "挑战", "瓶颈", "不确定性",
        "时间表", "进度", "周期", "排期",
        "入组难度", "依从性", "工作量", "实施复杂度",
    ]
    feas_keywords_en = [
        "feasibility", "feasible", "resource", "resources", "infrastructure",
        "budget", "funding", "cost", "costs", "cost-effectiveness",
        "risk", "risks", "challenge", "challenges", "bottleneck", "uncertainty",
        "timeline", "schedule", "timeframe",
        "enrollment", "recruitment", "compliance", "adherence", "burden",
    ]
    if any(k in t for k in feas_keywords_cn) or any(k in t_lower for k in feas_keywords_en):
        candidates.add("feasibility")

    # ---- 为“需求/应用/外部环境”类段落兜底：补充 strategy / objectives ----
    market_keywords_cn = [
        "市场", "市场规模", "市场分析", "市场份额", "市场占有率",
        "cagr", "复合年增长率", "销售额", "营收", "收入", "销售收入",
        "增长率", "增长幅度", "客户", "用户", "消费群体",
    ]
    market_keywords_en = [
        "market size", "market", "cagr", "market share", "share of",
        "sales", "revenue", "revenues", "turnover",
        "growth rate", "compound annual growth", "customer", "customers",
        "payer", "payers",
    ]
    if any(k in t for k in market_keywords_cn) or any(k in t_lower for k in market_keywords_en):
        candidates.add("strategy")
        candidates.add("objectives")

    return [d for d in candidates if d in VALID_DIMENSIONS]

def mark_numeric_suspect(fact: Dict[str, Any], chunk_text: str) -> Dict[str, Any]:
    """
    对包含数字的 fact 做简单校验：
    - 把 fact.text 里的数字片段（连续数字，不管是年份/金额）提取出来
    - 如果某个数字完全不出现在 chunk_text 中，则认为这条 fact 存在数字幻觉风险
    - 加 meta.suspect_numeric = True/False
    """
    text = fact.get("text", "") or ""
    nums = re.findall(r"\d+", text)
    if not nums:
        return fact  # 没数字，不管

    chunk_flat = (chunk_text or "").replace(" ", "")
    suspect = False
    for n in nums:
        if n not in chunk_flat:
            suspect = True
            break

    meta = fact.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta["suspect_numeric"] = suspect
    fact["meta"] = meta
    return fact

def normalize_fact(
    fact: Dict[str, Any],
    proposal_id: str,
    chunk_index: int,
    start: int,
    end: int,
) -> Dict[str, Any]:
    """
    给每条 fact 填上 meta 信息；清洗 dimensions / type。
    同时对维度标签做 type→dimensions 的通用“补标签”映射（支持多维度），
    尽量保证五个维度的信息都不会被漏掉。
    """
    text = fact.get("text", "")
    if not isinstance(text, str):
        text = str(text)

    # 原始维度标签清洗
    dims = fact.get("dimensions", [])
    if not isinstance(dims, list):
        dims = []
    dims_clean = [d for d in dims if isinstance(d, str) and d in VALID_DIMENSIONS]

    type_val = fact.get("type", "other")
    if type_val not in VALID_TYPES:
        type_val = "other"

    # ===== 外部环境/市场类事实的自动纠偏（在 type→dimensions 映射之前）=====
    # 如果 LLM 没有标成 market，但文本里明显是需求/应用/市场相关内容，则强制改为 "market"
    # （避免所有市场信息都被丢在 "other" 或 "product" 里）
    if type_val not in ["market", "risk", "mitigation"]:
        if _looks_like_market_fact(text):
            type_val = "market"

    # 先把已有维度放进一个 set，后面按 type / 文本内容补充
    dim_set = set(dims_clean)

    # ===== 1. 通用 type→dimensions 映射（更激进版本，优先保证信息被多个维度看到） =====
    if type_val in ["team_member", "org_structure"]:
        # 团队成员 / 组织结构 → 明确归入 team
        dim_set.add("team")

    elif type_val == "collaboration":
        # 协作既是团队协同，也是合作策略
        dim_set.update(["team", "strategy"])

    elif type_val == "pipeline":
        # 工作包/子任务：目标 + 路线，既体现 objectives，也体现 strategy
        dim_set.update(["objectives", "strategy"])

    elif type_val == "milestone":
        # 里程碑：目标的阶段性拆分 + 执行可行性
        dim_set.update(["objectives", "feasibility"])

    elif type_val == "clinical_design":
        # 验证/评估设计：目标路径 + 方案策略 + 执行可行性
        dim_set.update(["objectives", "strategy", "feasibility"])

    elif type_val in ["market", "product"]:
        # 产品/应用环境：实施策略 + 落地可行性
        dim_set.update(["objectives", "strategy", "feasibility"])

    elif type_val in ["tech_route", "regulatory"]:
        # 技术路线 / 制度流程：典型 strategy，但对可行性也有影响
        dim_set.update(["strategy", "feasibility"])

    elif type_val == "funding_source":
        # 资金来源：既体现策略布局，也影响可行性
        dim_set.update(["strategy", "feasibility"])

    elif type_val == "ip_asset":
        # IP 资产：创新优势 + 中长期可行性
        dim_set.update(["innovation", "feasibility"])

    elif type_val == "evidence":
        # 证据：创新价值 + 可行性（证据越强，可行性越高）
        dim_set.update(["innovation", "feasibility"])

    elif type_val == "ai_model":
        # AI 模型既是创新亮点，也是策略的一部分
        dim_set.update(["innovation", "strategy"])

    elif type_val in ["resource", "budget_item", "risk", "mitigation"]:
        # 资源、预算、风险与应对 → 可行性
        dim_set.add("feasibility")

    # ===== 2. 如果还是没维度，用文本关键词再判断一轮 =====
    if not dim_set:
        inferred = _infer_dims_from_text(text)
        for d in inferred:
            dim_set.add(d)

    # ===== 3. 最终兜底：还没有，就放到 feasibility，避免彻底丢失 =====
    if not dim_set:
        dim_set.add("feasibility")

    dims_final = [d for d in dim_set if d in VALID_DIMENSIONS]

    meta = fact.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    meta.update(
        {
            "proposal_id": proposal_id,
            "chunk_index": chunk_index,
            "char_start": start,
            "char_end": end,
        }
    )

    # ==== 计算 primary_dimension ====
    primary_dim = None
    if dims_final:
        # 简单的优先级规则：尽量按内容来，不行就取第一个
        # 你也可以自己按业务微调优先级
        priority = ["team", "objectives", "strategy", "innovation", "feasibility"]
        # 从 dims_final 里选出优先级最高的那一个
        for d in priority:
            if d in dims_final:
                primary_dim = d
                break
        if primary_dim is None:
            primary_dim = dims_final[0]
    else:
        primary_dim = "feasibility"  # 理论上不会走到这里，因为上面兜底了

    return {
        "text": text.strip(),
        "dimensions": dims_final,
        "type": type_val,
        "primary_dimension": primary_dim,
        "meta": meta,
    }

def run_extract(proposal_id: str, max_chars: int = 1800, overlap: int = 400):
    semantic_units = load_semantic_units(proposal_id)
    if semantic_units:
        chunks = make_semantic_chunks(semantic_units, max_chars=max(max_chars, 2200))
        print(f"[INFO] Stage 1 将优先使用 page_semantics 作为 facts 抽取输入")
    else:
        full_text = load_full_text(proposal_id)
        chunks = make_chunks(full_text, max_chars=max_chars, overlap=overlap)

    out_dir = EXTRACTED_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "raw_facts.jsonl"

    total_facts = 0
    # 统计每个维度的 fact 数量，便于 sanity check
    dim_counts = {dim: 0 for dim in VALID_DIMENSIONS}

    with out_path.open("w", encoding="utf-8") as f_out:
        for ch in chunks:
            idx = ch["index"]
            chunk_len = ch["end"] - ch["start"]
            print(f"\n[INFO] 处理 chunk {idx+1}/{len(chunks)} (chars={chunk_len})...")

            # 第一次正常抽取
            data = call_llm_for_chunk(ch["text"])
            facts = data.get("facts", [])
            if not isinstance(facts, list):
                facts = []

            # 如果 chunk 很长，但抽取的 facts 过少，则尝试 dense 模式重跑一次
            if chunk_len >= 1200 and len(facts) < 5:
                print(
                    f"[INFO] 当前 chunk 文本较长(chars={chunk_len})，但只抽取到 {len(facts)} 条事实，"
                    f"尝试使用 dense 模式重试以提高召回..."
                )
                dense_data = call_llm_for_chunk(ch["text"], dense=True)
                dense_facts = dense_data.get("facts", [])
                if isinstance(dense_facts, list) and len(dense_facts) > len(facts):
                    print(
                        f"[INFO] dense 模式抽取到 {len(dense_facts)} 条事实（优于原先的 {len(facts)} 条），"
                        f"采用 dense 结果。"
                    )
                    facts = dense_facts
                else:
                    print(
                        f"[INFO] dense 模式未显著提升（原 {len(facts)} 条，dense={len(dense_facts)} 条），"
                        f"保留原始抽取结果。"
                    )

            normalized_list = []
            for fact in facts:
                if not isinstance(fact, dict):
                    continue

                # 先基于 chunk_text 标记 suspect_numeric
                fact = mark_numeric_suspect(fact, ch["text"])

                norm = normalize_fact(
                    fact,
                    proposal_id=proposal_id,
                    chunk_index=idx,
                    start=ch.get("start", 0),
                    end=ch.get("end", len(ch.get("text", ""))),
                )
                norm.setdefault("meta", {})["source"] = ch.get("source", "full_text")
                if ch.get("page_indices"):
                    norm.setdefault("meta", {})["page_indices"] = ch.get("page_indices")

                # 过滤掉空文本
                if norm["text"]:
                    normalized_list.append(norm)

            for fact in normalized_list:
                f_out.write(json.dumps(fact, ensure_ascii=False) + "\n")
                total_facts += 1
                # 更新维度计数
                for d in fact.get("dimensions", []):
                    if d in dim_counts:
                        dim_counts[d] += 1

            print(
                f"[INFO] 该 chunk 最终写入 {len(normalized_list)} 条事实，"
                f"目前累计 {total_facts} 条。"
            )

    print(f"\n[OK] 已写出事实文件: {out_path} (总事实数={total_facts})")

    # ===== 全局维度分布检查 =====
    print("\n[SUMMARY] 维度分布统计（基于 raw_facts.jsonl）：")
    for dim in VALID_DIMENSIONS:
        print(f"  - {dim}: {dim_counts[dim]} facts")

    # 简单 sanity check：如果某个维度明显偏少，打个警告（这里只做提示，不终止）
    if total_facts > 0:
        avg = total_facts / len(VALID_DIMENSIONS)
        for dim in VALID_DIMENSIONS:
            if dim_counts[dim] < max(8, 0.25 * avg):
                print(
                    f"[WARN] 维度 {dim} 的事实数仅 {dim_counts[dim]}，"
                    f"显著低于平均值 {avg:.1f}，可能存在抽取不足或映射偏差，"
                    f"建议检查 raw_facts.jsonl 或适当调整 Prompt/映射。"
                )


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: 按 chunk 抽取原子事实（raw_facts.jsonl）"
    )
    parser.add_argument(
        "--proposal_id",
        required=False,
        help="提案 ID（对应 src/data/prepared/<proposal_id>）",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=1800,
        help="每个 chunk 最大字符数（默认 1800）",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=400,
        help="chunk 之间的字符重叠数（默认 400）",
    )
    args = parser.parse_args()

    if args.proposal_id:
        pid = args.proposal_id
    else:
        pid = find_latest_prepared_proposal()

    run_extract(pid, max_chars=args.max_chars, overlap=args.overlap)


if __name__ == "__main__":
    main()
