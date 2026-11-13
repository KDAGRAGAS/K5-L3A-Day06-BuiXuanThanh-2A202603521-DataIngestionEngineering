"""Fixed-size and structure-aware (heading/paragraph) chunking.

Every chunk is an exact slice page.text[char_start:char_end] of one page, so it can
always be traced back to document/page/offset. Chunks never cross page boundaries
(a paragraph continuing on the next page is split: a known coherence limitation).
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any

from .models import Chunk, DocumentPage

_PARAGRAPH = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", flags=re.S)
_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"”’)\]]*\s+(?=[\"“(\[]?[A-ZÀ-ỸĐ0-9])")
_HEADING_PREFIX = re.compile(r"^(chapter|chương|section|phần|mục|part|unit|bài|điều|appendix|phụ lục|\d+(\.\d+)*\.?)\s", re.I)


def _make_chunk(page: DocumentPage, start: int, end: int, index: int, kind: str) -> Chunk:
    text = page.text[start:end]
    # Trim whitespace but keep offsets exact.
    left = len(text) - len(text.lstrip())
    right = len(text.rstrip())
    start, end, text = start + left, start + right, text.strip()
    metadata: dict[str, Any] = {**page.metadata, "char_start": start, "char_end": end,
                                "chunk_index": index, "chunking": kind}
    chunk_id = f"{page.document_id}:p{page.page:04d}:c{index:03d}"
    return Chunk(chunk_id, page.document_id, page.source, page.page, text, metadata)


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Character windows of `size` with `overlap`, restarting on every page."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Cần size > 0 và 0 <= overlap < size")
    text = page.text
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if text[start:end].strip():
            spans.append((start, end))
        if end == len(text):
            break
        start = end - overlap
    return [_make_chunk(page, s, e, i, "fixed") for i, (s, e) in enumerate(spans)]


def is_heading(text: str) -> bool:
    """Short line without sentence-ending punctuation, or a numbered/labelled title."""
    stripped = text.strip()
    if not stripped or len(stripped) > 100 or "\n" in stripped:
        return False
    if _HEADING_PREFIX.match(stripped) and len(stripped.split()) <= 14:
        return True
    if stripped[-1] in ".!?:;,…" or stripped[0].islower():
        return False
    return len(stripped.split()) <= 12


@dataclass
class _Unit:
    start: int
    end: int
    heading: bool


def _split_long(text: str, start: int, end: int, max_size: int) -> list[_Unit]:
    """Split an oversized paragraph at sentence ends; hard-split very long sentences at spaces."""
    pieces: list[tuple[int, int]] = []
    cursor = start
    for match in _SENTENCE_END.finditer(text, start, end):
        pieces.append((cursor, match.start()))
        cursor = match.end()
    pieces.append((cursor, end))

    units: list[_Unit] = []
    for p_start, p_end in pieces:
        while p_end - p_start > max_size:
            cut = text.rfind(" ", p_start, p_start + max_size)
            cut = cut if cut > p_start else p_start + max_size
            units.append(_Unit(p_start, cut, False))
            p_start = cut + 1 if text[cut:cut + 1] == " " else cut
        if p_end > p_start:
            units.append(_Unit(p_start, p_end, False))
    return units


def by_paragraph(page: DocumentPage, max_size: int, min_size: int = 120) -> list[Chunk]:
    """Group paragraphs up to `max_size`; a heading always starts a new chunk and stays
    attached to the paragraph after it; oversized paragraphs are split by sentence.

    `max_size` is soft by at most one heading line (heading is kept with its body).
    """
    text = page.text
    units: list[_Unit] = []
    for match in _PARAGRAPH.finditer(text):
        p_start, p_end = match.start(), match.end()
        if p_end - p_start > max_size:
            units.extend(_split_long(text, p_start, p_end, max_size))
        else:
            units.append(_Unit(p_start, p_end, is_heading(text[p_start:p_end])))

    spans: list[list[int]] = []  # [start, end, has_body]
    for unit in units:
        if not spans:
            spans.append([unit.start, unit.end, int(not unit.heading)])
            continue
        current = spans[-1]
        too_long = unit.end - current[0] > max_size
        if current[2] and (unit.heading or too_long):
            spans.append([unit.start, unit.end, int(not unit.heading)])
        else:
            current[1] = unit.end
            current[2] = current[2] or int(not unit.heading)

    # Merge tiny leftovers (e.g. a caption) into a neighbour when the result still fits.
    merged: list[list[int]] = []
    for span in spans:
        if merged and (span[1] - span[0] < min_size) and (span[1] - merged[-1][0] <= max_size):
            merged[-1][1] = span[1]
            merged[-1][2] = merged[-1][2] or span[2]
        elif merged and (merged[-1][1] - merged[-1][0] < min_size) and (span[1] - merged[-1][0] <= max_size):
            merged[-1][1] = span[1]
            merged[-1][2] = merged[-1][2] or span[2]
        else:
            merged.append(span)
    return [_make_chunk(page, s, e, i, "paragraph") for i, (s, e, _b) in enumerate(merged)]


def chunk_pages(pages: list[DocumentPage], cfg: dict[str, Any]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        if cfg["chunking"] == "fixed":
            chunks.extend(fixed_size(page, int(cfg["chunk_size"]), int(cfg.get("overlap", 0))))
        else:
            chunks.extend(by_paragraph(page, int(cfg["max_size"]), int(cfg.get("min_size", 120))))
    return chunks


def length_stats(chunks: list[Chunk]) -> dict[str, Any]:
    lengths = sorted(len(c.text) for c in chunks)
    if not lengths:
        return {"count": 0}
    quartiles = statistics.quantiles(lengths, n=4) if len(lengths) > 1 else [lengths[0]] * 3
    return {
        "count": len(lengths),
        "total_chars": sum(lengths),
        "min": lengths[0],
        "p25": round(quartiles[0], 1),
        "median": statistics.median(lengths),
        "mean": round(statistics.fmean(lengths), 1),
        "p75": round(quartiles[2], 1),
        "max": lengths[-1],
    }
