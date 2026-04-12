# -*- coding: utf-8 -*-
"""
Stage 0 升级模块：layout_reconstruction.py

目标：
- 将 PDF 逐页渲染成图片，保留页面视觉结构
- 使用 OCR / layout / table / vision-LLM 组合构建 page_semantics.json
- 生成更适合 Stage 1 的重建文本（reconstructed_text）

设计原则：
1) 默认轻量可运行：即使没有 LayoutLM / Table Transformer 依赖，也能使用 OCR + 几何规则回退
2) 可渐进增强：如果安装了 paddleocr / transformers / torch，则自动启用更强能力
3) 输出可验证：每个页面都保留 blocks/tables/reading_order/source，便于追溯
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from pdf2image import convert_from_path
from PIL import Image
import pdfplumber
import pytesseract
from pytesseract import Output

from .tesseract_configure import apply_pytesseract_cmd

apply_pytesseract_cmd()

try:
    from openai import OpenAI  # type: ignore
    from openai import RateLimitError as _OpenAIRateLimitError  # type: ignore
except Exception:  # pragma: no cover
    OpenAI = None
    _OpenAIRateLimitError = None  # type: ignore[misc, assignment]

logger = logging.getLogger(__name__)

VISION_MODEL = os.getenv("OPENAI_VISION_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
TESSERACT_LANG = os.getenv("TESS_LANG", "chi_sim+eng")
ENABLE_VISION_LLM = os.getenv("ENABLE_VISION_LLM", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_TABLE_MODEL = os.getenv("ENABLE_TABLE_MODEL", "1").strip().lower() not in {"0", "false", "no"}
ENABLE_LAYOUT_MODEL = os.getenv("ENABLE_LAYOUT_MODEL", "1").strip().lower() not in {"0", "false", "no"}
OCR_MIN_CONF = int(os.getenv("OCR_MIN_CONF", "35"))

# Table Transformer: load once per process; skip after first hard failure (e.g. missing timm).
_TT_PROC_MODEL: Optional[Tuple[Any, Any]] = None
_TT_SKIP_REASON: Optional[str] = None


@dataclass
class OCRWord:
    text: str
    x0: int
    y0: int
    x1: int
    y1: int
    conf: float

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return max(1, self.y1 - self.y0)


@dataclass
class Block:
    block_id: str
    block_type: str
    text: str
    bbox: List[int]
    source: str
    reading_order: int = -1
    confidence: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class TableCell:
    row: int
    col: int
    text: str
    bbox: List[int]


@dataclass
class PageSemantic:
    page_index: int
    image_path: str
    width: int
    height: int
    pdf_text: str
    ocr_text: str
    page_type: str
    title: str
    blocks: List[Dict[str, Any]]
    tables: List[Dict[str, Any]]
    reconstructed_text: str
    vision_summary: Optional[Dict[str, Any]] = None
    source_notes: Optional[List[str]] = None


def render_pdf_pages(pdf_path: Path, image_dir: Path, dpi: int = 220) -> List[Path]:
    image_dir.mkdir(parents=True, exist_ok=True)
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    paths: List[Path] = []
    for idx, page in enumerate(pages, start=1):
        out = image_dir / f"page_{idx:03d}.png"
        page.save(out, "PNG")
        paths.append(out)
    return paths


def _load_image_size(path: Path) -> Tuple[int, int]:
    with Image.open(path) as img:
        return img.size




def normalize_ocr_text(text: str) -> str:
    text = text or ""

    # 统一换行和空白
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)

    # 合并“逐字拆开”的中文，例如：
    # 自 场 性 主 景 技 术  -> 自场性主景技术（后续再靠块合并减少这类情况）
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", text)

    # 去掉中文与常见中文标点之间多余空格
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[，。；：！？、“”‘’（）《》])", "", text)
    text = re.sub(r"(?<=[，。；：！？、“”‘’（）《》])\s+(?=[\u4e00-\u9fff])", "", text)

    # 修复中文与项目符号之间的空格
    text = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[•▪◦·])", "", text)
    text = re.sub(r"(?<=[•▪◦·])\s+(?=[\u4e00-\u9fff])", "", text)

    # 只保留英文/数字之间单空格
    text = re.sub(r"(?<=[A-Za-z0-9]) {2,}(?=[A-Za-z0-9])", " ", text)
    # 清理多余空行
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

def _ocr_words(image_path: Path) -> List[OCRWord]:
    data = pytesseract.image_to_data(Image.open(image_path), lang=TESSERACT_LANG, output_type=Output.DICT)
    words: List[OCRWord] = []
    n = len(data.get("text", []))
    for i in range(n):
        text = (data["text"][i] or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0
        if conf < OCR_MIN_CONF:
            continue
        x0 = int(data["left"][i])
        y0 = int(data["top"][i])
        x1 = x0 + int(data["width"][i])
        y1 = y0 + int(data["height"][i])
        words.append(OCRWord(text=text, x0=x0, y0=y0, x1=x1, y1=y1, conf=conf))
    return words


def _group_words_into_lines(words: Sequence[OCRWord]) -> List[List[OCRWord]]:
    if not words:
        return []
    sorted_words = sorted(words, key=lambda w: (w.cy, w.x0))
    lines: List[List[OCRWord]] = []
    for word in sorted_words:
        placed = False
        for line in lines:
            ref = line[0]
            tol = max(ref.height * 0.65, 12)
            if abs(word.cy - ref.cy) <= tol:
                line.append(word)
                placed = True
                break
        if not placed:
            lines.append([word])
    for line in lines:
        line.sort(key=lambda w: w.x0)
    lines.sort(key=lambda l: min(w.y0 for w in l))
    return lines


def _merge_lines_to_blocks(lines: Sequence[Sequence[OCRWord]], page_w: int, page_h: int) -> List[Block]:
    blocks: List[Block] = []
    if not lines:
        return blocks

    current: List[Sequence[OCRWord]] = []
    block_counter = 0

    def _line_text(line: Sequence[OCRWord]) -> str:
        return "".join(w.text for w in line).strip()

    def _is_vertical_fragment(line: Sequence[OCRWord]) -> bool:
        text = _line_text(line)
        if not text:
            return False
        compact = re.sub(r"\s+", "", text)
        return len(compact) <= 2 and all(("\u4e00" <= ch <= "\u9fff") or ch in "、，。；：！？,.·•" for ch in compact)

    def flush() -> None:
        nonlocal current, block_counter
        if not current:
            return

        flat = [w for line in current for w in line]
        raw_lines = []
        for line in current:
            raw = "".join(w.text for w in line).strip()
            raw_lines.append(raw)

        # 关键：如果是一串“逐字拆行”的中文，直接拼成一行
        if raw_lines and sum(1 for x in raw_lines if len(re.sub(r"\s+", "", x)) <= 2) >= max(3, len(raw_lines) // 2):
            text = "".join(raw_lines).strip()
        else:
            text = "\n".join(raw_lines).strip()

        text = normalize_ocr_text(text)

        x0 = min(w.x0 for w in flat)
        y0 = min(w.y0 for w in flat)
        x1 = max(w.x1 for w in flat)
        y1 = max(w.y1 for w in flat)
        avg_conf = sum(w.conf for w in flat) / max(1, len(flat))

        block_type = classify_block(text, [x0, y0, x1, y1], page_w, page_h)
        blocks.append(Block(
            block_id=f"b{block_counter:03d}",
            block_type=block_type,
            text=text,
            bbox=[x0, y0, x1, y1],
            source="ocr_layout",
            confidence=round(avg_conf, 2),
            meta={"line_count": len(current)}
        ))
        block_counter += 1
        current = []

    prev_bottom = None
    prev_x0 = None
    prev_is_vertical = False

    for line in lines:
        flat = list(line)
        y0 = min(w.y0 for w in flat)
        x0 = min(w.x0 for w in flat)
        this_is_vertical = _is_vertical_fragment(line)

        if not current:
            current = [line]
            prev_bottom = max(w.y1 for w in flat)
            prev_x0 = x0
            prev_is_vertical = this_is_vertical
            continue

        gap = y0 - (prev_bottom or y0)

        # 普通段落合并
        same_paragraph = gap <= 18 and prev_x0 is not None and abs(x0 - prev_x0) <= 60

        # 关键：纵向碎字列，允许更强合并
        same_vertical_stream = (
            prev_x0 is not None
            and this_is_vertical
            and prev_is_vertical
            and abs(x0 - prev_x0) <= 40
            and gap <= 35
        )

        if same_paragraph or same_vertical_stream:
            current.append(line)
            prev_bottom = max(w.y1 for w in flat)
            prev_x0 = (prev_x0 + x0) / 2 if prev_x0 is not None else x0
            prev_is_vertical = this_is_vertical
        else:
            flush()
            current = [line]
            prev_bottom = max(w.y1 for w in flat)
            prev_x0 = x0
            prev_is_vertical = this_is_vertical

    flush()
    return blocks


def classify_block(text: str, bbox: List[int], page_w: int, page_h: int) -> str:
    stripped = re.sub(r"\s+", " ", text).strip()
    x0, y0, x1, y1 = bbox
    width = x1 - x0
    height = y1 - y0
    line_count = stripped.count("\n") + 1
    compact = stripped.replace(" ", "")
    if len(compact) <= 2 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9]", compact):
        return "noise"
    if y0 > page_h * 0.92:
        return "footer"
    if y0 < page_h * 0.18 and len(compact) <= 40:
        return "title"
    if re.fullmatch(r"[0-9]+([.,][0-9]+)?[A-Za-z%°·sN\.]*", compact):
        return "numeric_label"
    if line_count >= 2 and re.search(r"[|｜]", stripped):
        return "table_like"
    if width > page_w * 0.6 and line_count >= 2:
        return "paragraph"
    if height < 24 and len(compact) <= 3:
        return "noise"
    if len(stripped) <= 30:
        return "label"
    return "text"


def recover_reading_order(blocks: Sequence[Block]) -> List[Block]:
    if not blocks:
        return []
    sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    rows: List[List[Block]] = []
    for block in sorted_blocks:
        placed = False
        cy = (block.bbox[1] + block.bbox[3]) / 2
        for row in rows:
            ref = row[0]
            ref_cy = (ref.bbox[1] + ref.bbox[3]) / 2
            tol = max((ref.bbox[3] - ref.bbox[1]) * 0.7, 18)
            if abs(cy - ref_cy) <= tol:
                row.append(block)
                placed = True
                break
        if not placed:
            rows.append([block])
    rows.sort(key=lambda r: min(b.bbox[1] for b in r))
    ordered: List[Block] = []
    idx = 0
    for row in rows:
        row.sort(key=lambda b: b.bbox[0])
        for block in row:
            block.reading_order = idx
            ordered.append(block)
            idx += 1
    return ordered


def _extract_pdf_text_page(pdf_path: Path, page_index: int) -> str:
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_index < len(pdf.pages):
                return (pdf.pages[page_index].extract_text() or "").strip()
    except Exception as exc:
        logger.warning("pdfplumber page text failed: %s", exc)
    return ""


def scramble_score(text: str) -> float:
    """
    Heuristic 0..1: higher means text likely has wrong reading order / column merge noise
    (e.g. double-column PDF read as one stream).
    """
    if not text or len(text) < 50:
        return 0.0
    parts = re.split(r"[\s\u3000]+", text.strip())
    cjk_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    if cjk_chars < 12:
        return 0.0
    single_cjk_tokens = sum(1 for p in parts if len(p) == 1 and "\u4e00" <= p <= "\u9fff")
    return min(1.0, (single_cjk_tokens / max(cjk_chars, 1)) * 5.0)


def _extract_pdf_text_two_column(pdf_path: Path, page_index: int) -> str:
    """Left-then-right column merge for typical two-column proposal slides."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_index >= len(pdf.pages):
                return ""
            page = pdf.pages[page_index]
            w, h = float(page.width), float(page.height)
            mid = w * 0.5
            left = page.crop((0, 0, max(mid - 2, 1), h))
            right = page.crop((min(mid + 2, w - 1), 0, w, h))
            lt = (left.extract_text() or "").strip()
            rt = (right.extract_text() or "").strip()
            merged = f"{lt}\n\n{rt}".strip()
            return normalize_ocr_text(merged)
    except Exception as exc:
        logger.warning("two-column pdf extract failed: %s", exc)
    return ""


def best_pdf_text_for_page(pdf_path: Path, page_index: int) -> str:
    plain = _extract_pdf_text_page(pdf_path, page_index)
    if scramble_score(plain) > 0.22:
        col = _extract_pdf_text_two_column(pdf_path, page_index)
        if len(col) > 80 and scramble_score(col) + 0.06 < scramble_score(plain):
            return col
    return plain


def detect_tables_with_pdfplumber(pdf_path: Path, page_index: int) -> List[Dict[str, Any]]:
    tables: List[Dict[str, Any]] = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_index >= len(pdf.pages):
                return tables
            page = pdf.pages[page_index]
            for t_idx, table in enumerate(page.extract_tables() or []):
                if not table:
                    continue
                rows = []
                for r_idx, row in enumerate(table):
                    if row is None:
                        continue
                    cells = []
                    for c_idx, cell in enumerate(row):
                        txt = (cell or "").strip()
                        cells.append({"row": r_idx, "col": c_idx, "text": txt})
                    rows.append(cells)
                tables.append({
                    "table_id": f"pdf_table_{page_index+1}_{t_idx+1}",
                    "source": "pdfplumber",
                    "cells": rows,
                    "table_text": "\n".join(" | ".join(c["text"] for c in row) for row in rows if row),
                })
    except Exception as exc:
        logger.warning("pdfplumber table extraction failed: %s", exc)
    return tables


def detect_tables_with_transformer(image_path: Path) -> List[Dict[str, Any]]:
    global _TT_PROC_MODEL, _TT_SKIP_REASON
    if not ENABLE_TABLE_MODEL:
        return []
    if _TT_SKIP_REASON is not None:
        return []

    # TableTransformer loads timm inside `from_pretrained`; fail early to avoid
    # noisy stack traces on every page when timm is missing from this interpreter.
    try:
        import timm  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        _TT_SKIP_REASON = "timm not importable"
        logger.info(
            "Layout: Table Transformer skipped (optional `timm` not in this Python). "
            "Tables still use pdfplumber. To enable: `python -m pip install timm` in this env."
        )
        return []

    try:
        import torch  # type: ignore
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection  # type: ignore
    except Exception as exc:
        _TT_SKIP_REASON = f"import: {exc}"
        logger.warning("Table Transformer disabled (import error): %s", exc)
        return []

    model_name = os.getenv("TABLE_TRANSFORMER_MODEL", "microsoft/table-transformer-detection")

    if _TT_PROC_MODEL is None:
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = TableTransformerForObjectDetection.from_pretrained(model_name)
            _TT_PROC_MODEL = (processor, model)
        except Exception as exc:
            _TT_SKIP_REASON = str(exc)
            logger.warning(
                "Table Transformer disabled for this session (install `timm` if missing): %s",
                exc,
            )
            return []

    processor, model = _TT_PROC_MODEL
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=0.85, target_sizes=target_sizes)[0]
        tables = []
        for idx, (score, label, box) in enumerate(zip(results["scores"], results["labels"], results["boxes"])):
            x0, y0, x1, y1 = [int(v) for v in box.tolist()]
            label_name = model.config.id2label.get(int(label), str(label))
            sc = score.detach() if hasattr(score, "detach") else score
            tables.append({
                "table_id": f"tt_{idx+1}",
                "source": model_name,
                "label": label_name,
                "score": float(sc),
                "bbox": [x0, y0, x1, y1],
            })
        return tables
    except Exception as exc:
        logger.warning("Table Transformer inference failed on %s: %s", image_path.name, exc)
        return []


def _guess_page_type(blocks: Sequence[Block], tables: Sequence[Dict[str, Any]]) -> str:
    texts = " ".join(b.text for b in blocks)
    if tables:
        return "table_page"
    if any("目录" in b.text or "CONTENTS" in b.text.upper() for b in blocks):
        return "toc_slide"
    if re.search(r"20\d{2}年", texts) and len(re.findall(r"20\d{2}年", texts)) >= 3:
        return "timeline_slide"
    if len([b for b in blocks if b.block_type == "numeric_label"]) >= 3:
        return "data_visual_slide"
    if len(blocks) <= 4:
        return "cover_or_simple_slide"
    return "content_slide"


def _title_from_blocks(blocks: Sequence[Block]) -> str:
    titles = [b for b in blocks if b.block_type == "title"]
    if titles:
        return normalize_ocr_text(re.sub(r"\s+", " ", titles[0].text).strip())
    if blocks:
        return re.sub(r"\s+", " ", blocks[0].text.split("\n", 1)[0]).strip()
    return ""


def _reconstruct_table_text(table: Dict[str, Any]) -> str:
    cells = table.get("cells") or []
    lines = []
    for row in cells:
        if isinstance(row, list):
            line = " | ".join((c.get("text") or "").strip() for c in row)
            if line.strip(" |"):
                lines.append(line)
    return "\n".join(lines)


def _fallback_reconstructed_text(blocks: Sequence[Block], tables: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []

    for block in blocks:
        if block.block_type in {"footer", "noise"}:
            continue

        text = normalize_ocr_text(block.text.strip())
        if not text:
            continue

        parts.append(text)

    for table in tables:
        text = table.get("table_text") or _reconstruct_table_text(table)
        text = normalize_ocr_text(text)
        if text:
            parts.append(text)

    merged = "\n\n".join(p for p in parts if p).strip()
    return normalize_ocr_text(merged)


def _image_to_data_url(image_path: Path) -> str:
    raw = image_path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _is_openai_rate_limit(exc: BaseException) -> bool:
    if _OpenAIRateLimitError is not None and isinstance(exc, _OpenAIRateLimitError):
        return True
    s = str(exc).lower()
    return "429" in str(exc) or "rate_limit" in s or "rate limit" in s


def vision_llm_enrich_page(image_path: Path, page_payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not ENABLE_VISION_LLM:
        return None
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key or OpenAI is None:
        return None
    client = OpenAI(api_key=api_key)
    prompt = (
        "你是一个严格的多模态文档页面解析器。"
        "请基于页面图像本身作为第一依据，OCR/布局候选只作为辅助参考。"
        "目标：输出适合后续评审系统使用的高质量结构化 JSON。"

        "\n\n要求："
        "\n1. 优先恢复页面的自然阅读顺序。"
        "\n2. 不要保留明显错误的 OCR 垃圾文本。"
        "\n3. 如果页面中存在竖排字、装饰文字、旋转文本、图形中的碎字、难以连成自然句子的散字，请忽略它们。"
        "\n4. reconstructed_text 只保留对页面语义真正有贡献的内容。"
        "\n5. 对标题、正文、表格、关键标签分别理解，但不要编造。"
        "\n6. 如果 OCR 候选与图像冲突，以图像理解为准。"
        "\n7. 不要把单字散列、坐标噪声、页码、无意义重复片段写进 reconstructed_text。"

        "\n\n输出 JSON 格式："
        '{"page_type":"...",'
        '"title":"...",'
        '"reading_order_notes":["..."],'
        '"regions":[{"role":"title/text/table/chart/label","text":"..."}],'
        '"reconstructed_text":"..."}'
    )
    max_retries = max(1, int(os.getenv("VISION_LLM_MAX_RETRIES", "8")))
    base_wait = float(os.getenv("VISION_LLM_RETRY_BASE_SEC", "2.0"))
    page_gap = float(os.getenv("VISION_LLM_PAGE_GAP_SEC", "0.35"))
    last_exc: Optional[BaseException] = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=VISION_MODEL,
                temperature=0.0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "你是严谨的页面结构解析器，只能依据可见内容回答。"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt + "\n\nOCR/布局候选：\n" + json.dumps(page_payload, ensure_ascii=False)[:12000]},
                            {"type": "image_url", "image_url": {"url": _image_to_data_url(image_path)}},
                        ],
                    },
                ],
                max_tokens=1800,
            )
            raw = response.choices[0].message.content or "{}"
            out = json.loads(raw)
            if page_gap > 0:
                time.sleep(page_gap)
            return out
        except Exception as exc:
            last_exc = exc
            if _is_openai_rate_limit(exc) and attempt + 1 < max_retries:
                wait = base_wait * (1.55**attempt)
                logger.info(
                    "Vision LLM rate limited (%s); sleep %.1fs then retry %d/%d",
                    image_path.name,
                    wait,
                    attempt + 1,
                    max_retries,
                )
                time.sleep(wait)
                continue
            logger.warning("vision LLM enrichment failed on %s: %s", image_path, exc)
            return None
    logger.warning("vision LLM enrichment failed on %s after %d tries: %s", image_path, max_retries, last_exc)
    return None


def build_page_semantic(pdf_path: Path, image_path: Path, page_index: int) -> PageSemantic:
    width, height = _load_image_size(image_path)
    ocr_words = _ocr_words(image_path)
    lines = _group_words_into_lines(ocr_words)
    blocks = _merge_lines_to_blocks(lines, page_w=width, page_h=height)
    ordered_blocks = [b for b in recover_reading_order(blocks) if b.block_type != "noise"]
    pdf_text = best_pdf_text_for_page(pdf_path, page_index)
    ocr_text = normalize_ocr_text("\n".join(" ".join(w.text for w in line) for line in lines).strip())

    tables = detect_tables_with_pdfplumber(pdf_path, page_index)
    transformer_tables = detect_tables_with_transformer(image_path)
    if transformer_tables:
        tables.extend(transformer_tables)

    page_type = _guess_page_type(ordered_blocks, tables)
    title = _title_from_blocks(ordered_blocks)

    base_payload = {
        "page_index": page_index + 1,
        "page_type": page_type,
        "title": title,
        "blocks": [asdict(b) for b in ordered_blocks],
        "tables": tables,
        "pdf_text": pdf_text,
        "ocr_text": ocr_text,
    }
    vision_summary = vision_llm_enrich_page(image_path, base_payload)

    # Vision-LLM 优先：把 page_type / title / reconstructed_text / regions 作为主结果
    if vision_summary and isinstance(vision_summary, dict):
        reconstructed_text = normalize_ocr_text(
            vision_summary.get("reconstructed_text", "")
        )
    else:
        reconstructed_text = _fallback_reconstructed_text(ordered_blocks, tables)

    reconstructed_text = normalize_ocr_text(reconstructed_text)

    # Prefer cleaner PDF stream when OCR/vision still looks like column-scrambled garbage
    if pdf_text and len(pdf_text) > 100:
        rs, ps = scramble_score(reconstructed_text), scramble_score(pdf_text)
        if rs > 0.30 and ps + 0.08 < rs:
            reconstructed_text = normalize_ocr_text(pdf_text)
        elif len(reconstructed_text) < 45 and len(pdf_text) > len(reconstructed_text) * 2:
            reconstructed_text = normalize_ocr_text(pdf_text)

    source_notes = []
    if pdf_text:
        source_notes.append("pdf_text_available")
    if ocr_text:
        source_notes.append("ocr_layout_available")
    if vision_summary:
        source_notes.append("vision_llm_enriched")
    if transformer_tables:
        source_notes.append("table_transformer_detected")
    elif tables:
        source_notes.append("pdfplumber_tables_detected")

    return PageSemantic(
        page_index=page_index + 1,
        image_path=str(image_path),
        width=width,
        height=height,
        pdf_text=pdf_text,
        ocr_text=ocr_text,
        page_type=(vision_summary or {}).get("page_type", page_type),
        title=(vision_summary or {}).get("title", title),
        blocks=[asdict(b) for b in ordered_blocks],
        tables=tables,
        reconstructed_text=reconstructed_text,
        vision_summary=vision_summary,
        source_notes=source_notes,
    )


def build_document_semantics(pdf_path: Path, out_dir: Path, dpi: int = 220) -> Dict[str, Any]:
    image_dir = out_dir / "page_images"
    image_paths = render_pdf_pages(pdf_path, image_dir=image_dir, dpi=dpi)
    pages: List[Dict[str, Any]] = []
    for idx, image_path in enumerate(image_paths):
        page_sem = build_page_semantic(pdf_path=pdf_path, image_path=image_path, page_index=idx)
        pages.append(asdict(page_sem))
    reconstructed_pages = [p.get("reconstructed_text", "").strip() for p in pages]
    reconstructed_full_text = "\n\n".join([p for p in reconstructed_pages if p])
    return {
        "proposal_file": str(pdf_path),
        "num_pages": len(pages),
        "pages": pages,
        "reconstructed_full_text": reconstructed_full_text,
    }


def save_document_semantics(doc_semantics: Dict[str, Any], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    page_sem_path = out_dir / "page_semantics.json"
    page_sem_path.write_text(json.dumps(doc_semantics, ensure_ascii=False, indent=2), encoding="utf-8")
    reconstructed_text_path = out_dir / "reconstructed_full_text.txt"
    reconstructed_text_path.write_text(doc_semantics.get("reconstructed_full_text", ""), encoding="utf-8")
    return {
        "page_semantics_path": str(page_sem_path),
        "reconstructed_full_text_path": str(reconstructed_text_path),
        "page_images_dir": str(out_dir / "page_images"),
    }


def rebuild_pages_json_from_semantics(doc_semantics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Build pages.json rows aligned with layout pipeline output so run_review uses
    the same per-page text as reconstructed_full_text (fixes evidence selection drift).
    """
    pages = doc_semantics.get("pages") or []
    texts: List[str] = []
    sources: List[str] = []
    for p in pages:
        rt = (p.get("reconstructed_text") or "").strip()
        pdf_t = (p.get("pdf_text") or "").strip()
        if not rt and not pdf_t:
            texts.append("")
            sources.append("empty")
            continue
        if not rt:
            texts.append(pdf_t)
            sources.append("pdf_text")
            continue
        if not pdf_t:
            texts.append(rt)
            sources.append("reconstructed")
            continue
        sr, sp = scramble_score(rt), scramble_score(pdf_t)
        if sr > 0.30 and sp + 0.08 < sr:
            texts.append(pdf_t)
            sources.append("pdf_text_preferred")
        elif sp > 0.30 and sr + 0.08 < sp:
            texts.append(rt)
            sources.append("reconstructed_preferred")
        elif len(rt) >= len(pdf_t) * 1.25:
            texts.append(rt)
            sources.append("reconstructed")
        elif len(pdf_t) >= len(rt) * 1.25:
            texts.append(pdf_t)
            sources.append("pdf_text")
        else:
            texts.append(rt if sr <= sp else pdf_t)
            sources.append("merged_pick")

    offset = 0
    out: List[Dict[str, Any]] = []
    for i, (txt, src) in enumerate(zip(texts, sources)):
        char_len = len(txt)
        page_start = offset
        page_end = offset + char_len
        row: Dict[str, Any] = {
            "page_index": i + 1,
            "source": src,
            "char_len": char_len,
            "global_char_start": page_start,
            "global_char_end": page_end,
            "text": txt,
        }
        if i < len(pages):
            pt = (pages[i].get("page_type") or "").strip()
            if pt:
                row["page_type"] = pt
        out.append(row)
        offset = page_end + 2
    return out


def semantic_pages_for_stage1(page_semantics: Dict[str, Any]) -> List[Dict[str, Any]]:
    pages = page_semantics.get("pages") or []
    semantic_units: List[Dict[str, Any]] = []
    for page in pages:
        page_idx = page.get("page_index")
        title = (page.get("title") or "").strip()
        page_type = page.get("page_type") or "content_slide"
        text_parts: List[str] = []
        if title:
            text_parts.append(f"[PAGE_TITLE] {title}")
        text_parts.append(f"[PAGE_TYPE] {page_type}")
        reconstructed_text = (page.get("reconstructed_text") or "").strip()
        if reconstructed_text:
            text_parts.append(reconstructed_text)
        for table in page.get("tables") or []:
            ttext = table.get("table_text") or _reconstruct_table_text(table)
            if ttext:
                text_parts.append(f"[TABLE]\n{ttext}")
        unit_text = "\n\n".join([p for p in text_parts if p]).strip()
        semantic_units.append({
            "page_index": page_idx,
            "title": title,
            "page_type": page_type,
            "text": unit_text,
            "source": "page_semantics",
        })
    return semantic_units
