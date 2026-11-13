"""Read the manifest, extract and normalize every page, keep a link back to source/page.

PDF: one DocumentPage per physical page (1-based), text built from PyMuPDF blocks so
paragraph breaks survive ("\n\n" between blocks). Section metadata comes from the PDF
outline (bookmarks), i.e. from the source itself, never guessed.

TXT/MD (no pagination): split in order into units of >= MIN_UNIT_CHARS normalized
characters at paragraph boundaries; the unit number is used as `page`.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import REPO_ROOT
from .models import DocumentPage

MIN_UNIT_CHARS = 1500
EMPTY_PAGE_CHARS = 30
MANIFEST_REQUIRED = ("source_id", "title", "source_path_or_url", "license_or_permission", "accessed_at")

_CHAR_MAP = str.maketrans({
    " ": " ", " ": " ", " ": " ", "\t": " ", "\f": " ", "\v": " ",
    "­": None, "​": None, "‌": None, "‍": None, "﻿": None,
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
})
_LOWER_START = re.compile(r"^[a-zà-ỹđ]")


def normalize_text(text: str) -> str:
    """NFC (important for Vietnamese), drop invisible chars/ligatures, tidy whitespace.

    Keeps paragraph structure: single newlines are kept, 3+ newlines collapse to one
    blank line, runs of spaces/tabs collapse to one space.
    """
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").translate(_CHAR_MAP)
    text = re.sub(r"[ ]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _join_block_lines(block_text: str) -> str:
    """Lines inside one PDF block are visual wraps of one paragraph: join them.

    A trailing hyphen is removed only when the next line starts with a lowercase letter.
    """
    lines = [line.strip() for line in block_text.split("\n") if line.strip()]
    if not lines:
        return ""
    out = lines[0]
    for line in lines[1:]:
        if out.endswith("-") and _LOWER_START.match(line):
            out = out[:-1] + line
        else:
            out = f"{out} {line}"
    return out


def _boilerplate_key(text: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.lower())).strip()


@dataclass
class Extraction:
    pages: list[DocumentPage]
    physical_pages: int
    removed_boilerplate: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _section_lookup(toc: list[list[Any]], section_level: int) -> tuple[dict[int, str], dict[int, str]]:
    """Map start page -> section / subsection title from the PDF outline."""
    sections: dict[int, str] = {}
    subsections: dict[int, str] = {}
    starts = sorted(
        ((int(entry[2]), int(entry[0]), normalize_text(str(entry[1]))) for entry in toc if int(entry[2]) > 0),
        key=lambda item: (item[0], item[1]),
    )
    for page, level, title in starts:
        if level <= section_level:
            sections[page] = title
        elif level == section_level + 1:
            subsections[page] = title
    return sections, subsections


def load_pdf(path: Path, document_id: str, source: str, *, section_level: int = 1,
             strip_repeated: bool = True) -> Extraction:
    """Extract each physical page; keep page number, page label and outline section.

    Pages with no text are kept in the result (status decided by load_corpus) so they
    can be logged; OCR is out of scope and must be reported as an error.
    """
    import pymupdf  # PyMuPDF; imported lazily so TXT/MD corpora do not need it

    errors: list[str] = []
    try:
        doc = pymupdf.open(path)
    except Exception as exc:  # noqa: BLE001 - surface any reader failure with the file name
        raise RuntimeError(f"Không mở được PDF {path}: {exc}") from exc

    with doc:
        page_blocks: list[list[tuple[str, bool]]] = []  # (text, in top/bottom margin)
        labels: list[str] = []
        for index in range(doc.page_count):
            page = doc.load_page(index)
            height = page.rect.height or 1
            blocks: list[tuple[str, bool]] = []
            try:
                for _x0, y0, _x1, y1, text, _no, block_type in page.get_text("blocks"):
                    if block_type != 0:  # image block
                        continue
                    joined = _join_block_lines(normalize_text(text))
                    if joined:
                        blocks.append((joined, y1 <= 0.1 * height or y0 >= 0.9 * height))
            except Exception as exc:  # noqa: BLE001
                errors.append(f"page {index + 1}: lỗi trích text: {exc}")
            page_blocks.append(blocks)
            try:
                labels.append(page.get_label() or "")
            except Exception:  # noqa: BLE001
                labels.append("")
        toc = doc.get_toc(simple=True) or []

    # Running headers/footers: short blocks in the top/bottom margin whose text (digits
    # ignored) repeats on many pages. Long books alternate odd/even headers and change them
    # per chapter, so the threshold is >= 3 pages and >= 2% of the pages, not a majority.
    removed: Counter[str] = Counter()
    boiler: set[str] = set()
    if strip_repeated and len(page_blocks) >= 4:
        seen: Counter[str] = Counter()
        for blocks in page_blocks:
            seen.update({_boilerplate_key(t) for t, margin in blocks if margin and len(t) <= 120})
        threshold = max(3, -(-len(page_blocks) * 2 // 100))
        boiler = {key for key, count in seen.items() if count >= threshold}

    sections, subsections = _section_lookup(toc, section_level)
    current_section, current_sub = "", ""
    pages: list[DocumentPage] = []
    for index, blocks in enumerate(page_blocks):
        number = index + 1
        if number in sections:
            current_section, current_sub = sections[number], ""
        if number in subsections:
            current_sub = subsections[number]
        kept = []
        for text, margin in blocks:
            key = _boilerplate_key(text)
            if margin and key in boiler:
                removed[key] += 1
                continue
            kept.append(text)
        metadata: dict[str, Any] = {"page_label": labels[index]}
        if toc:
            metadata["section"] = current_section
            metadata["subsection"] = current_sub
        pages.append(DocumentPage(document_id, source, number, normalize_text("\n\n".join(kept)), metadata))
    if not toc:
        errors.append("PDF không có outline/bookmark: không có metadata 'section' lấy từ nguồn")
    return Extraction(pages, len(page_blocks), dict(removed), errors)


def load_text(path: Path, document_id: str, source: str, min_chars: int = MIN_UNIT_CHARS) -> Extraction:
    """Split TXT/MD in order into units >= min_chars, cut only at paragraph boundaries."""
    raw = normalize_text(path.read_text(encoding="utf-8"))
    paragraphs = [(m.start(), m.end()) for m in re.finditer(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", raw, flags=re.S)]
    units: list[tuple[int, int]] = []
    start: int | None = None
    for p_start, p_end in paragraphs:
        start = p_start if start is None else start
        if p_end - start >= min_chars:
            units.append((start, p_end))
            start = None
    if start is not None and paragraphs:
        if units:  # short tail is merged into the previous unit so every unit stays >= min_chars
            units[-1] = (units[-1][0], paragraphs[-1][1])
        else:
            units.append((start, paragraphs[-1][1]))
    pages = []
    heading = ""
    is_md = path.suffix.lower() in {".md", ".markdown"}
    for number, (u_start, u_end) in enumerate(units, start=1):
        text = raw[u_start:u_end]
        metadata: dict[str, Any] = {"unit_char_start": u_start, "unit_char_end": u_end, "page_label": str(number)}
        if is_md:
            unit_section = heading
            for line in text.split("\n"):
                if re.match(r"^#{1,2}\s+\S", line):
                    heading = line.lstrip("#").strip()
                    unit_section = unit_section or heading
            metadata["section"] = unit_section
        pages.append(DocumentPage(document_id, source, number, text, metadata))
    return Extraction(pages, len(units))


def parse_page_spec(spec: Any) -> set[int] | None:
    """'1-3,7' or [1, 2, '5-6'] -> {1,2,3,7}. Empty -> None (= every non-empty page)."""
    if spec in (None, "", []):
        return None
    items = spec if isinstance(spec, list) else str(spec).split(",")
    pages: set[int] = set()
    for item in items:
        item = str(item).strip()
        if not item:
            continue
        if "-" in item:
            low, high = (int(x) for x in item.split("-", 1))
            pages.update(range(low, high + 1))
        else:
            pages.add(int(item))
    return pages


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chưa có manifest: {path} (copy từ data/manifest.example.json)")
    with path.open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    docs = manifest["documents"] if isinstance(manifest, dict) else manifest
    ids: set[str] = set()
    for doc in docs:
        missing = [key for key in MANIFEST_REQUIRED if not doc.get(key)]
        if missing:
            raise ValueError(f"Manifest: tài liệu {doc.get('source_id', '?')} thiếu {missing}")
        if not doc.get("local_path"):
            raise ValueError(f"Manifest: tài liệu {doc['source_id']} thiếu local_path (file đã tải về)")
        if doc["source_id"] in ids:
            raise ValueError(f"Manifest: source_id trùng {doc['source_id']}")
        ids.add(doc["source_id"])
    return docs


@dataclass
class Corpus:
    pages: list[DocumentPage]          # counted, non-empty content pages (the ones ingested)
    report: list[dict[str, Any]]       # one row per physical page/unit with its status
    documents: list[dict[str, Any]]    # manifest entries + computed counts
    errors: list[dict[str, Any]]

    def page_index(self) -> dict[tuple[str, int], DocumentPage]:
        return {(p.document_id, p.page): p for p in self.pages}


def load_corpus(manifest_path: Path) -> Corpus:
    docs = load_manifest(manifest_path)
    pages: list[DocumentPage] = []
    report: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    resolved_docs = []
    for doc in docs:
        doc_id = doc["source_id"]
        local = Path(doc["local_path"])
        local = local if local.is_absolute() else REPO_ROOT / local
        if not local.exists():
            raise FileNotFoundError(f"{doc_id}: không thấy file {local}")
        suffix = local.suffix.lower()
        if suffix == ".pdf":
            extraction = load_pdf(local, doc_id, doc["source_path_or_url"],
                                  section_level=int(doc.get("section_level", 1)),
                                  strip_repeated=bool(doc.get("strip_repeated_headers", True)))
        elif suffix in {".txt", ".md", ".markdown"}:
            extraction = load_text(local, doc_id, doc["source_path_or_url"])
        else:
            raise ValueError(f"{doc_id}: định dạng {suffix} chưa hỗ trợ (dùng PDF/TXT/MD)")
        for message in extraction.errors:
            errors.append({"source_id": doc_id, "page": None, "error": message})

        wanted = parse_page_spec(doc.get("content_pages"))
        doc_meta = dict(doc.get("metadata") or {})
        counted: list[int] = []
        chars = 0
        for page in extraction.pages:
            n_chars = len(page.text)
            if wanted is not None and page.page not in wanted:
                status = "excluded_by_manifest"
            elif n_chars < EMPTY_PAGE_CHARS:
                status = "empty_or_scanned"
                errors.append({"source_id": doc_id, "page": page.page,
                               "error": f"trang rỗng/scan ({n_chars} ký tự): không ingest, không tính"})
            else:
                status = "content"
            if status == "content":
                page.metadata = {"title": doc["title"], **doc_meta, **page.metadata}
                pages.append(page)
                counted.append(page.page)
                chars += n_chars
            report.append({
                "source_id": doc_id, "page": page.page, "page_label": page.metadata.get("page_label", ""),
                "status": status, "chars": n_chars, "section": page.metadata.get("section", ""),
            })
        if wanted is not None:
            beyond = sorted(p for p in wanted if p > extraction.physical_pages)
            if beyond:
                errors.append({"source_id": doc_id, "page": None,
                               "error": f"content_pages vượt số trang vật lý ({extraction.physical_pages}): {beyond[:10]}"})
        resolved_docs.append({
            **doc,
            "physical_pages_or_units": extraction.physical_pages,
            "counted_content_pages": counted,
            "counted_content_page_count": len(counted),
            "counted_chars": chars,
            "removed_repeated_header_footer": extraction.removed_boilerplate,
            "resolved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
    return Corpus(pages, report, resolved_docs, errors)
