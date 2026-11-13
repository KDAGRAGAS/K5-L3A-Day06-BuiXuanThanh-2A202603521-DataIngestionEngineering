"""Load and validate the benchmark against the normalized corpus BEFORE any retrieval run."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .evaluation import gold_options, matches_filter, normalize_for_match
from .ingestion import Corpus

REQUIRED = ("id", "split", "question", "expected_answer", "gold_source_id", "gold_page", "evidence_quote")


def load_questions(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Chưa có benchmark: {path} (copy từ benchmark/questions.example.json)")
    with path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    questions = data["questions"] if isinstance(data, dict) else data
    for q in questions:
        q["split"] = str(q.get("split", "")).strip().lower()
        if not q.get("filter"):
            q["filter"] = None
    return questions


def validate(questions: list[dict[str, Any]], corpus: Corpus) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    pages = corpus.page_index()
    page_text = {key: normalize_for_match(p.text) for key, p in pages.items()}
    meta_values: dict[str, set[str]] = {}
    for p in corpus.pages:
        for key, value in {**p.metadata, "document_id": p.document_id, "page": p.page}.items():
            if isinstance(value, (str, int, float)):
                meta_values.setdefault(key, set()).add(str(value))

    seen: set[str] = set()
    for q in questions:
        qid = q.get("id", "?")
        missing = [k for k in REQUIRED if q.get(k) in (None, "", [])]
        if missing:
            errors.append(f"{qid}: thiếu trường {missing}")
            continue
        if qid in seen:
            errors.append(f"{qid}: id trùng")
        seen.add(qid)
        if q["split"] not in {"dev", "test"}:
            errors.append(f"{qid}: split phải là dev/test, đang là {q['split']!r}")

        for option in gold_options(q):
            for page in option["pages"]:
                key = (option["source_id"], page)
                if key not in pages:
                    errors.append(f"{qid}: gold {option['source_id']} trang {page} không nằm trong các trang nội dung đã ingest")
                    continue
                for quote in option["quotes"]:
                    if normalize_for_match(quote) not in page_text[key]:
                        found = [p for (doc, p), text in page_text.items()
                                 if doc == option["source_id"] and normalize_for_match(quote) in text]
                        hint = f"; tìm thấy ở trang {found}" if found else "; không thấy ở trang nào (copy lại từ `show-page`)"
                        errors.append(f"{qid}: evidence không có trong {option['source_id']} trang {page} sau chuẩn hóa{hint}")

        q_filter = q.get("filter")
        if q_filter:
            if not isinstance(q_filter, dict):
                errors.append(f"{qid}: filter phải là object, ví dụ {{\"section\": \"...\"}}")
                continue
            for key, value in q_filter.items():
                values = value if isinstance(value, list) else [value]
                if key not in meta_values:
                    errors.append(f"{qid}: trường filter '{key}' không có trong metadata corpus")
                    continue
                unknown = [v for v in values if str(v) not in meta_values[key]]
                if unknown:
                    errors.append(f"{qid}: giá trị filter {key}={unknown} không tồn tại trong metadata")
                if key == "document_id" and q["gold_source_id"] in [str(v) for v in values]:
                    warnings.append(f"{qid}: filter document_id = gold source -> rò rỉ gold, chỉ hợp lệ nếu ngữ cảnh người dùng thật sự cung cấp")
            gold_ok = False
            for option in gold_options(q):
                for page in option["pages"]:
                    meta = pages.get((option["source_id"], page))
                    if meta is None:
                        continue
                    record = {"document_id": meta.document_id, "source": meta.source,
                              "page": meta.page, "metadata": meta.metadata}
                    gold_ok = gold_ok or matches_filter(record, q_filter)
            if not gold_ok:
                errors.append(f"{qid}: filter loại bỏ chính trang gold -> câu hỏi sai")
        if str(q["gold_source_id"]).lower() in str(q["question"]).lower():
            warnings.append(f"{qid}: câu hỏi chứa gold_source_id (rò rỉ nguồn?)")

    dev = [q for q in questions if q.get("split") == "dev"]
    test = [q for q in questions if q.get("split") == "test"]
    filtered = [q for q in questions if q.get("filter")]
    if len(questions) < 20:
        errors.append(f"Cần >= 20 câu, hiện có {len(questions)}")
    if len(dev) < 15 or len(test) < 5:
        errors.append(f"Cần >= 15 dev và >= 5 test, hiện có {len(dev)} dev / {len(test)} test")
    if len(filtered) < 3 or not any(q.get("split") == "test" for q in filtered):
        errors.append(f"Cần >= 3 câu có filter, >= 1 câu ở test; hiện có {len(filtered)} "
                      f"({sum(q.get('split') == 'test' for q in filtered)} ở test)")
    types: dict[str, int] = {}
    for q in questions:
        types[q.get("type", "") or "(không ghi)"] = types.get(q.get("type", "") or "(không ghi)", 0) + 1
    return {
        "ok": not errors, "questions": len(questions), "dev": len(dev), "test": len(test),
        "with_filter": len(filtered), "with_filter_test": sum(q.get("split") == "test" for q in filtered),
        "types": types, "filter_fields_available": sorted(k for k in meta_values if k not in {"page_label"}),
        "errors": errors, "warnings": warnings,
    }
