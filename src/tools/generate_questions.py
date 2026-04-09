# -*- coding: utf-8 -*-
"""
Stage 3 · Domain-adaptive question generation

Design goals:
- no domain-specific prompt text inside this stage
- deterministic question generation from:
    universal template + domain_profile.json + fixed task registry
- keep downstream compatibility with llm_answering.py
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

from src.prompting.domain_adaptive import (
    REVIEW_TASKS,
    QUESTION_SEARCH_HINTS,
    build_specialized_question,
    load_domain_profile,
)

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
EXTRACTED_DIR = DATA_DIR / "extracted"
QUESTIONS_DIR = DATA_DIR / "questions"
CONFIG_QS_DIR = DATA_DIR / "config" / "question_sets"

DIMENSION_NAMES = ["team", "objectives", "strategy", "innovation", "feasibility"]
QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_QS_DIR.mkdir(parents=True, exist_ok=True)


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def find_latest_extracted_proposal_id() -> str:
    cands = [d for d in EXTRACTED_DIR.iterdir() if d.is_dir()] if EXTRACTED_DIR.exists() else []
    if not cands:
        raise FileNotFoundError("No proposal found under src/data/extracted")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    pid = cands[0].name
    print(f"[INFO] [auto] selected latest proposal_id: {pid}")
    return pid


def load_dimensions(proposal_id: str) -> Dict[str, Any]:
    path = EXTRACTED_DIR / proposal_id / "dimensions_v2.json"
    if not path.exists():
        raise FileNotFoundError(f"dimensions_v2.json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dimension_summary(dimensions: Dict[str, Any], dim_name: str) -> Dict[str, Any]:
    block = dimensions.get(dim_name, {}) if isinstance(dimensions, dict) else {}
    if not isinstance(block, dict):
        block = {}
    return {
        "summary": block.get("summary", ""),
        "key_points": block.get("key_points", []) or [],
        "risks": block.get("risks", []) or [],
        "mitigations": block.get("mitigations", []) or [],
        "meta": block.get("meta", {}) or {},
    }


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_json_with_log(path: Path, payload: Dict[str, Any], label: str) -> None:
    _write_json(path, payload)
    print(f"✅ {label} -> {path}")


def build_question_record(task, profile: Dict[str, Any], dim_summary: Dict[str, Any], local_index: int) -> Dict[str, Any]:
    question_zh = build_specialized_question(task, profile)
    question_en = str(question_zh)

    links_to = {
        "key_points": list(range(min(3, len(dim_summary.get("key_points", []))))),
        "risks": list(range(min(2, len(dim_summary.get("risks", []))))),
        "mitigations": list(range(min(2, len(dim_summary.get("mitigations", []))))),
    }

    rating_tasks = {"team", "feasibility", "innovation", "outcomes", "objectives"}
    analysis_tasks = {"methods", "risks", "evidence", "problem"}
    answer_type = "rating" if task.task_id in rating_tasks else "analysis"

    return {
        "qid": f"{task.dimension}_Q{local_index}",
        "task_id": task.task_id,
        "template_id": task.template_id,
        "dimension": task.dimension,
        "aspect": task.task_id,
        "title": task.title,
        "question_zh": question_zh,
        "question_en": question_en,
        "answer_type": answer_type,
        "priority": 1,
        "links_to": links_to,
    }


def run_generate_questions(proposal_id: str):
    dimensions = load_dimensions(proposal_id)
    profile = load_domain_profile(EXTRACTED_DIR / proposal_id / "domain_profile.json")

    per_dim_counter = {dim: 0 for dim in DIMENSION_NAMES}

    all_dim_questions: Dict[str, Any] = {
        dim: {
            "dimension": dim,
            "questions": [],
            "search_hints": QUESTION_SEARCH_HINTS.get(dim, []),
            "source_proposal_id": proposal_id,
            "domain_profile": profile,
            "dimension_summary": _dimension_summary(dimensions, dim),
        }
        for dim in DIMENSION_NAMES
    }

    for task in REVIEW_TASKS:
        per_dim_counter[task.dimension] += 1
        dim_summary = _dimension_summary(dimensions, task.dimension)
        q = build_question_record(task, profile, dim_summary, per_dim_counter[task.dimension])
        all_dim_questions[task.dimension]["questions"].append(q)

    detail_out = {
        "proposal_id": proposal_id,
        "generated_at": now_str(),
        "mode": "domain_adaptive_deterministic",
        "template_version": "universal_v1",
        "profile_version": "domain_profile_v1",
        "dimensions": all_dim_questions,
    }

    out_dir = QUESTIONS_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_out_path = out_dir / "generated_questions.json"
    _write_json_with_log(detail_out_path, detail_out, "detailed questions")

    config_out: Dict[str, Any] = {
        "proposal_id": proposal_id,
        "generated_at": now_str(),
        "mode": "domain_adaptive_deterministic",
        "template_version": "universal_v1",
        "profile_version": "domain_profile_v1",
        "domain_profile": profile,
    }
    for dim in DIMENSION_NAMES:
        q_objs = all_dim_questions[dim]["questions"]
        config_out[dim] = {
            "dimension": dim,
            "questions": [q["question_zh"] for q in q_objs],
            "question_objects": q_objs,
            "search_hints": QUESTION_SEARCH_HINTS.get(dim, []),
            "source_proposal_id": proposal_id,
        }

    config_out_path = CONFIG_QS_DIR / "generated_questions.json"
    _write_json_with_log(config_out_path, config_out, "runtime question set")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate domain-adaptive questions from universal templates")
    ap.add_argument("--proposal_id", type=str, default="", help="proposal id under src/data/extracted/<proposal_id>")
    args = ap.parse_args()
    pid = args.proposal_id.strip() or find_latest_extracted_proposal_id()
    run_generate_questions(pid)
