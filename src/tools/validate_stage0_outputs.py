# -*- coding: utf-8 -*-
"""Validate multimodal Stage 0 artifacts for a prepared proposal."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
PREPARED_DIR = BASE_DIR / "src" / "data" / "prepared"


def validate(proposal_id: str) -> int:
    out_dir = PREPARED_DIR / proposal_id
    issues = []
    page_sem_path = out_dir / "page_semantics.json"
    if not page_sem_path.exists():
        issues.append("missing page_semantics.json")
        print("[FAIL] missing page_semantics.json")
        return 1
    data = json.loads(page_sem_path.read_text(encoding="utf-8"))
    pages = data.get("pages") or []
    if data.get("num_pages") != len(pages):
        issues.append("num_pages mismatch")
    for page in pages:
        if not page.get("reconstructed_text"):
            issues.append(f"page {page.get('page_index')} missing reconstructed_text")
        if not page.get("blocks"):
            issues.append(f"page {page.get('page_index')} missing blocks")
    if issues:
        print("[WARN] Stage 0 validation issues:")
        for issue in issues:
            print(" -", issue)
        return 1
    print(f"[OK] Stage 0 artifacts look valid for proposal_id={proposal_id}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("proposal_id")
    args = parser.parse_args()
    raise SystemExit(validate(args.proposal_id))


if __name__ == "__main__":
    main()
