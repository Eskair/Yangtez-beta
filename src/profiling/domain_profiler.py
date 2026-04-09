# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv
from openai import OpenAI

from src.prompting.domain_adaptive import PROFILER_SYSTEM_PROMPT, sanitize_domain_profile

load_dotenv()
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "src" / "data"
PREPARED_DIR = DATA_DIR / "prepared"
EXTRACTED_DIR = DATA_DIR / "extracted"
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def detect_latest_pid() -> str:
    if not PREPARED_DIR.exists():
        return "unknown"
    dirs = [d for d in PREPARED_DIR.iterdir() if d.is_dir()]
    if not dirs:
        return "unknown"
    dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return dirs[0].name


def load_source_text(pid: str) -> str:
    for candidate in [PREPARED_DIR / pid / "full_text.txt", EXTRACTED_DIR / pid / "dimensions_v2.json"]:
        if candidate.exists() and candidate.suffix == '.txt':
            return candidate.read_text(encoding='utf-8', errors='ignore')[:18000]
    dim_path = EXTRACTED_DIR / pid / "dimensions_v2.json"
    if dim_path.exists():
        return dim_path.read_text(encoding='utf-8', errors='ignore')[:18000]
    raise FileNotFoundError(f"No source text found for proposal_id={pid}")


def profile_with_llm(text: str) -> Dict[str, Any]:
    client = OpenAI()
    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": PROFILER_SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=900,
    )
    raw = resp.choices[0].message.content
    try:
        data = json.loads(raw)
    except Exception:
        data = {}
    return sanitize_domain_profile(data)


def run_domain_profiler(proposal_id: str) -> Path:
    text = load_source_text(proposal_id)
    profile = profile_with_llm(text)
    out_dir = EXTRACTED_DIR / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'domain_profile.json'
    out_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"🧭 Domain profile saved -> {out_path}")
    return out_path


def main():
    ap = argparse.ArgumentParser(description='Generate domain_profile.json for the current proposal')
    ap.add_argument('--proposal_id', type=str, default='')
    args = ap.parse_args()
    pid = args.proposal_id.strip() or detect_latest_pid()
    run_domain_profiler(pid)


if __name__ == '__main__':
    main()
