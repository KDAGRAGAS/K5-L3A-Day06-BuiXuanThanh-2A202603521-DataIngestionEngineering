"""Manual-audit sheets (CSV, opens in Excel) and the summary of the labels you fill in.

The code only PREPARES samples and helper columns marked `auto_*`; the labels
(`label_0_2`, `metadata_correct_0_1`, reasons) must be filled by a human.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .evaluation import normalize_for_match
from .ingestion import Corpus
from .models import Chunk


def _has_labels(path: Path, columns: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return any(any((row.get(c) or "").strip() for c in columns) for row in csv.DictReader(handle))


def write_sheet(path: Path, rows: list[dict[str, Any]], label_columns: tuple[str, ...]) -> Path:
    """Never overwrite a sheet that already has human labels: write *.new.csv instead."""
    target = path
    if _has_labels(path, label_columns):
        target = path.with_suffix(".new.csv")
        print(f"[audit] {path.name} đã có nhãn -> không ghi đè, bản mới: {target.name}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return target
    with target.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return target


def coherence_positions(corpus: Corpus, n: int) -> list[dict[str, Any]]:
    """n positions fixed BEFORE any score: pages sorted by (document_id, page), evenly
    spaced, offset = middle of the page text."""
    pages = sorted(corpus.pages, key=lambda p: (p.document_id, p.page))
    if not pages:
        return []
    n = min(n, len(pages))
    picks = sorted({round(i * (len(pages) - 1) / max(n - 1, 1)) for i in range(n)})
    positions = []
    for number, index in enumerate(picks, start=1):
        page = pages[index]
        offset = len(page.text) // 2
        positions.append({"position_id": f"pos{number:02d}", "document_id": page.document_id,
                          "page": page.page, "char_offset": offset,
                          "context": page.text[max(0, offset - 90):offset] + " ⟦▲⟧ " + page.text[offset:offset + 90]})
    return positions


def _covering_chunk(chunks: list[Chunk], document_id: str, page: int, offset: int) -> Chunk | None:
    same_page = [c for c in chunks if c.document_id == document_id and c.page == page]
    for chunk in same_page:
        if chunk.metadata["char_start"] <= offset < chunk.metadata["char_end"]:
            return chunk
    after = [c for c in same_page if c.metadata["char_start"] >= offset]
    return min(after, key=lambda c: c.metadata["char_start"]) if after else (same_page[-1] if same_page else None)


def coherence_sheet(path: Path, corpus: Corpus, chunks_by_strategy: dict[str, list[Chunk]],
                    index_of: dict[str, str], n: int) -> Path:
    rows = []
    for pos in coherence_positions(corpus, n):
        for strategy, chunks in chunks_by_strategy.items():
            chunk = _covering_chunk(chunks, pos["document_id"], pos["page"], pos["char_offset"])
            shared = [s for s, idx in index_of.items() if idx == index_of[strategy] and s < strategy]
            rows.append({
                "position_id": pos["position_id"], "strategy": strategy,
                "document_id": pos["document_id"], "page": pos["page"], "char_offset": pos["char_offset"],
                "context_around_position": pos["context"],
                "chunk_id": chunk.chunk_id if chunk else "", "chunk_chars": len(chunk.text) if chunk else 0,
                "chunk_text": chunk.text if chunk else "",
                "note": f"cùng chunk với {shared[0]} (dùng chung index) - để trống sẽ kế thừa nhãn" if shared else "",
                "label_0_2": "", "reason": "",
            })
    return write_sheet(path, rows, ("label_0_2", "reason"))


def metadata_sheet(path: Path, traces: list[dict[str, Any]], corpus: Corpus) -> Path:
    pages = {key: normalize_for_match(p.text) for key, p in corpus.page_index().items()}
    rows = []
    for trace in traces:
        for hit in trace["hits"]:
            meta = hit["metadata"]
            on_page = normalize_for_match(hit["text"]) in pages.get((hit["document_id"], int(hit["page"])), "")
            extra = {k: v for k, v in meta.items()
                     if k not in {"title", "section", "subsection", "page_label", "char_start", "char_end",
                                  "chunk_index", "chunking"}}
            rows.append({
                "strategy": trace["strategy"], "question_id": trace["id"], "rank": hit["rank"],
                "chunk_id": hit["chunk_id"], "document_id": hit["document_id"], "source": hit["source"],
                "page": hit["page"], "page_label": meta.get("page_label", ""), "title": meta.get("title", ""),
                "section": meta.get("section", ""), "subsection": meta.get("subsection", ""),
                "other_metadata": json.dumps(extra, ensure_ascii=False),
                "question_filter": json.dumps(trace["question_filter"], ensure_ascii=False) if trace["question_filter"] else "",
                "filter_applied": json.dumps(trace["filter_applied"], ensure_ascii=False) if trace["filter_applied"] else "",
                "auto_matches_question_filter": "" if hit["matches_question_filter"] is None else int(hit["matches_question_filter"]),
                "auto_text_found_on_stated_page": int(on_page),
                "chunk_text_preview": hit["text"][:300],
                "metadata_correct_0_1": "", "error_note": "",
            })
    return write_sheet(path, rows, ("metadata_correct_0_1", "error_note"))


def grounding_sheet(path: Path, traces: list[dict[str, Any]]) -> Path:
    rows = []
    for trace in traces:
        row: dict[str, Any] = {
            "strategy": trace["strategy"], "question_id": trace["id"], "question": trace["question"],
            "expected_answer": trace["expected_answer"],
            "gold": "; ".join(f"{g['source_id']} p{','.join(map(str, g['pages']))}" for g in trace["gold"]),
            "evidence_quote": " | ".join(q for g in trace["gold"] for q in g["evidence_quote"]),
            "filter_applied": json.dumps(trace["filter_applied"], ensure_ascii=False) if trace["filter_applied"] else "",
        }
        for rank in range(1, 4):
            hit = trace["hits"][rank - 1] if len(trace["hits"]) >= rank else None
            row[f"top{rank}_ref"] = f"{hit['document_id']} p{hit['page']} ({hit['chunk_id']}, {hit['score']:.3f})" if hit else ""
            row[f"top{rank}_text"] = hit["text"] if hit else ""
        row["auto_evidence_hit_at_3 (proxy)"] = int(trace["evidence_hit_at_3"])
        row["label_0_2"] = ""
        row["reason"] = ""
        rows.append(row)
    return write_sheet(path, rows, ("label_0_2", "reason"))


# ---------------------------------------------------------------------- summary
def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _label(value: str, allowed: set[int]) -> int | None:
    value = (value or "").strip()
    if not value:
        return None
    number = int(float(value))
    if number not in allowed:
        raise ValueError(f"Nhãn {value!r} ngoài thang {sorted(allowed)}")
    return number


def summarize_audit(audit_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    coherence = _read(audit_dir / "coherence.csv")
    labelled = {(r["position_id"], r["chunk_id"]): _label(r["label_0_2"], {0, 1, 2})
                for r in coherence if _label(r["label_0_2"], {0, 1, 2}) is not None}
    for row in coherence:
        label = _label(row["label_0_2"], {0, 1, 2})
        inherited = False
        if label is None and (row["position_id"], row["chunk_id"]) in labelled:
            label, inherited = labelled[(row["position_id"], row["chunk_id"])], True
        entry = summary.setdefault(row["strategy"], {}).setdefault("coherence", {"labels": [], "inherited": 0})
        if label is not None:
            entry["labels"].append(label)
            entry["inherited"] += inherited

    for row in _read(audit_dir / "metadata.csv"):
        entry = summary.setdefault(row["strategy"], {}).setdefault("metadata", {"ok": 0, "checked": 0, "errors": []})
        label = _label(row["metadata_correct_0_1"], {0, 1})
        if label is None:
            continue
        entry["checked"] += 1
        entry["ok"] += label
        if label == 0:
            entry["errors"].append(f"{row['question_id']}#{row['rank']}: {row['error_note']}")

    for row in _read(audit_dir / "grounding.csv"):
        entry = summary.setdefault(row["strategy"], {}).setdefault("grounding", {"0": 0, "1": 0, "2": 0})
        label = _label(row["label_0_2"], {0, 1, 2})
        if label is not None:
            entry[str(label)] += 1

    result: dict[str, Any] = {}
    for strategy, parts in sorted(summary.items()):
        coh = parts.get("coherence", {"labels": [], "inherited": 0})
        meta = parts.get("metadata", {"ok": 0, "checked": 0, "errors": []})
        ground = parts.get("grounding", {"0": 0, "1": 0, "2": 0})
        n_ground = sum(ground.values())
        test_metrics = metrics.get(strategy, {}).get("test", {})
        result[strategy] = {
            "coherence_mean_of_2": round(sum(coh["labels"]) / len(coh["labels"]), 2) if coh["labels"] else None,
            "coherence_n": len(coh["labels"]), "coherence_inherited_from_shared_index": coh["inherited"],
            "metadata_correct": f"{meta['ok']}/{meta['checked']}", "metadata_errors": meta["errors"],
            "filter_compliance_test": (test_metrics.get("top3_filter_compliance") or {}).get("text", "N/A")
            if isinstance(test_metrics.get("top3_filter_compliance"), dict) else "N/A",
            "grounding_counts_0_1_2": ground,
            "grounding_mean_of_2": round((ground["1"] + 2 * ground["2"]) / n_ground, 2) if n_ground else None,
            "grounding_n": n_ground,
        }
    return result
