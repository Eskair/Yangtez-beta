# -*- coding: utf-8 -*-
"""
Stage 0: 提案文本准备（基础 OCR + 多模态页面重建）
"""
from __future__ import annotations

import os
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List

import pdfplumber
from pdf2image import convert_from_path
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "tesseract"
from docx import Document

from .layout_reconstruction import (
    build_document_semantics,
    rebuild_pages_json_from_semantics,
    save_document_semantics,
)

MIN_TEXT_CHARS_PER_PAGE = 30
TESSERACT_LANG = os.getenv("TESS_LANG", "chi_sim+eng")


def detect_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in [".docx", ".doc"]:
        return "docx"
    if suffix in [".txt", ".md"]:
        return "txt"
    raise ValueError(f"暂不支持的文件类型: {suffix}")


def find_latest_proposal() -> Path:
    base_dir = Path(__file__).resolve().parents[2]
    proposals_dir = base_dir / "src" / "data" / "proposals"
    if not proposals_dir.exists():
        raise FileNotFoundError(f"未找到提案目录: {proposals_dir}")
    candidates = [
        p for p in proposals_dir.iterdir()
        if p.is_file() and p.suffix.lower() in [".pdf", ".docx", ".doc", ".txt", ".md"]
    ]
    if not candidates:
        raise FileNotFoundError(f"提案目录中没有可用文件: {proposals_dir}")
    latest = max(candidates, key=lambda x: x.stat().st_mtime)
    print(f"[INFO] [auto] 选中最新提案文件: {latest}")
    return latest


def ocr_page_from_pdf(pdf_path: Path, page_index: int) -> str:
    try:
        images = convert_from_path(str(pdf_path), first_page=page_index + 1, last_page=page_index + 1)
    except Exception as e:
        print(f"[WARN] convert_from_path 失败 (page {page_index+1}): {e}")
        return ""
    if not images:
        return ""
    image: Image.Image = images[0]
    try:
        return pytesseract.image_to_string(image, lang=TESSERACT_LANG)
    except Exception as e:
        print(f"[WARN] OCR 失败 (page {page_index+1}): {e}")
        return ""


def extract_from_pdf(pdf_path: Path, use_ocr: bool = True):
    pages_text, page_sources = [], []
    with pdfplumber.open(pdf_path) as pdf:
        print(f"[INFO] PDF 页面数: {len(pdf.pages)}")
        for i, page in enumerate(pdf.pages):
            txt = (page.extract_text() or "").strip()
            if txt and len(txt) >= MIN_TEXT_CHARS_PER_PAGE:
                pages_text.append(txt)
                page_sources.append("pdf_text")
                print(f"  - 第 {i+1} 页: 使用 pdfplumber 文本，长度 {len(txt)}")
            else:
                if use_ocr:
                    print(f"  - 第 {i+1} 页: 文本太少({len(txt)} chars)，尝试 OCR...")
                    ocr_txt = (ocr_page_from_pdf(pdf_path, page_index=i) or "").strip()
                    if ocr_txt:
                        pages_text.append(ocr_txt)
                        page_sources.append("ocr")
                        print(f"    -> OCR 成功，长度 {len(ocr_txt)}")
                    else:
                        pages_text.append("")
                        page_sources.append("empty")
                        print("    -> OCR 也没有提取到文本")
                else:
                    pages_text.append(txt)
                    page_sources.append("pdf_text_empty")
    return pages_text, page_sources


def extract_from_docx(docx_path: Path):
    doc = Document(str(docx_path))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return ["\n".join(paragraphs)], ["docx"]


def extract_from_txt(txt_path: Path):
    content = Path(txt_path).read_text(encoding="utf-8", errors="ignore").strip()
    return [content], ["txt"]


def _build_pages_json(pages_text: List[str], page_sources: List[str]) -> List[Dict[str, Any]]:
    pages_data: List[Dict[str, Any]] = []
    offset = 0
    for i, (txt, src) in enumerate(zip(pages_text, page_sources)):
        char_len = len(txt)
        page_start = offset
        page_end = offset + char_len
        pages_data.append({
            "page_index": i + 1,
            "source": src,
            "char_len": char_len,
            "global_char_start": page_start,
            "global_char_end": page_end,
            "text": txt,
        })
        offset = page_end + 2
    return pages_data


def prepare_text(file_path: Path, proposal_id: str, use_ocr: bool = True):
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    file_type = detect_file_type(file_path)
    print(f"[INFO] 开始提取文本: {file_path} (type={file_type})")

    if file_type == "pdf":
        pages_text, page_sources = extract_from_pdf(file_path, use_ocr=use_ocr)
    elif file_type == "docx":
        pages_text, page_sources = extract_from_docx(file_path)
    elif file_type == "txt":
        pages_text, page_sources = extract_from_txt(file_path)
    else:
        raise ValueError(f"未知文件类型: {file_type}")

    base_dir = Path(__file__).resolve().parents[2]
    out_dir = base_dir / "src" / "data" / "prepared" / proposal_id
    out_dir.mkdir(parents=True, exist_ok=True)

    full_text = "\n\n".join(pages_text)
    full_text_path = out_dir / "full_text.txt"
    full_text_path.write_text(full_text, encoding="utf-8")
    print(f"[OK] full_text.txt 写入完成: {full_text_path} (长度 {len(full_text)} 字符)")

    pages_data = _build_pages_json(pages_text, page_sources)
    pages_json_path = out_dir / "pages.json"
    pages_json_path.write_text(json.dumps(pages_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[OK] pages.json 写入完成: {pages_json_path}")

    page_semantics_path = None
    reconstructed_full_text_path = None
    page_images_dir = None
    if file_type == "pdf":
        try:
            print("[INFO] 开始多模态 Stage 0 重建：pdf2image -> layout/table -> page_semantics")
            doc_sem = build_document_semantics(file_path, out_dir=out_dir)
            saved = save_document_semantics(doc_sem, out_dir=out_dir)
            page_semantics_path = saved["page_semantics_path"]
            reconstructed_full_text_path = saved["reconstructed_full_text_path"]
            page_images_dir = saved["page_images_dir"]
            reconstructed_full_text = doc_sem.get("reconstructed_full_text", "").strip()
            if reconstructed_full_text:
                full_text_path.write_text(reconstructed_full_text, encoding="utf-8")
                print(f"[OK] 已用 reconstructed_full_text 覆盖 full_text.txt，长度 {len(reconstructed_full_text)} 字符")
            # Keep pages.json aligned with per-page reconstructed/pdf choice (evidence selection)
            try:
                pages_data = rebuild_pages_json_from_semantics(doc_sem)
                pages_json_path.write_text(
                    json.dumps(pages_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                print(f"[OK] 已用 layout 结果同步 pages.json（{len(pages_data)} 页）")
            except Exception as sync_e:
                print(f"[WARN] pages.json 同步失败，保留初次提取: {sync_e}")
        except Exception as e:
            print(f"[WARN] 多模态 Stage 0 重建失败，继续使用基础文本: {e}")

    return {
        "proposal_id": proposal_id,
        "file_type": file_type,
        "out_dir": str(out_dir),
        "full_text_path": str(full_text_path),
        "pages_json_path": str(pages_json_path),
        "page_semantics_path": page_semantics_path,
        "reconstructed_full_text_path": reconstructed_full_text_path,
        "page_images_dir": page_images_dir,
        "num_pages": len(pages_text),
    }


def main():
    parser = argparse.ArgumentParser(description="Stage 0: 提案文本准备（含 OCR + 多模态页面重建 + 自动选最新提案）")
    parser.add_argument("--file", required=False, help="提案文件路径（PDF/DOCX/TXT）。不填则自动选最新提案")
    parser.add_argument("--proposal_id", required=False, help="提案 ID（用于输出目录名，不填则用文件名）")
    parser.add_argument("--no_ocr", action="store_true", help="禁用 OCR（仅调试用）")
    args = parser.parse_args()

    file_path = Path(args.file) if args.file else find_latest_proposal()
    if args.file:
        print(f"[INFO] 使用用户指定文件: {file_path}")
    proposal_id = args.proposal_id or file_path.stem
    info = prepare_text(file_path, proposal_id, use_ocr=not args.no_ocr)

    print("\n[SUMMARY]")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
