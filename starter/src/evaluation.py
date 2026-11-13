"""Automatic retrieval checks (Hit@1, Evidence Hit@3) and full per-query traces.

A retrieved chunk is a hit when it comes from a gold (source, page) AND contains the
whole gold evidence (after match-normalization). Any documented alternative gold
counts. Human labels (coherence, grounding, metadata correctness) are NOT inferred
here; see audit.py for the manual sheets.
"""
from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from typing import Any

_PUNCT_MAP = str.maketrans({
    "“": '"', "”": '"', "„": '"', "‘": "'", "’": "'", "‚": "'", "–": "-", "—": "-", "−": "-",
    "…": "...", "­": None,
})


def normalize_for_match(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(_PUNCT_MAP).casefold()
    return re.sub(r"\s+", " ", text).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"\w+", normalize_for_match(text))


def gold_options(question: dict[str, Any]) -> list[dict[str, Any]]:
    """Main gold + `alternatives`; each option: source_id, pages (set), quotes (all required)."""
    def option(source: str, page: Any, quote: Any) -> dict[str, Any]:
        pages = page if isinstance(page, list) else [page]
        quotes = quote if isinstance(quote, list) else [quote]
        return {"source_id": str(source), "pages": {int(p) for p in pages},
                "quotes": [q for q in quotes if q and str(q).strip()]}

    options = [option(question["gold_source_id"], question["gold_page"], question["evidence_quote"])]
    for alt in question.get("alternatives") or []:
        options.append(option(alt["gold_source_id"], alt["gold_page"], alt["evidence_quote"]))
    return options


def evidence_in(text: str, quotes: list[str]) -> bool:
    haystack = normalize_for_match(text)
    return bool(quotes) and all(normalize_for_match(q) in haystack for q in quotes)


def evidence_coverage(text: str, quotes: list[str]) -> float:
    wanted = [w for q in quotes for w in _words(q)]
    if not wanted:
        return 0.0
    present = set(_words(text))
    return sum(w in present for w in wanted) / len(wanted)


def matches_filter(hit: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    for key, value in (filters or {}).items():
        allowed = {str(v) for v in (value if isinstance(value, list) else [value])}
        actual = hit.get(key) if key in {"document_id", "source", "page"} else (hit.get("metadata") or {}).get(key)
        if actual is None or str(actual) not in allowed:
            return False
    return True


def _merged_page_text(hits: list[dict[str, Any]]) -> str:
    """Join retrieved chunks of the same page in source order (overlaps removed) to test
    whether evidence was only split across chunks. Supplementary diagnostic only."""
    ordered = sorted(hits, key=lambda h: int(h["metadata"].get("char_start", 0)))
    text, end = "", None
    for hit in ordered:
        start = int(hit["metadata"].get("char_start", 0))
        stop = int(hit["metadata"].get("char_end", start + len(hit["text"])))
        if end is None:
            text, end = hit["text"], stop
        elif start < end:
            text += hit["text"][end - start:] if stop > end else ""
            end = max(end, stop)
        elif start - end <= 3:
            text, end = f"{text} {hit['text']}", stop
        else:
            text, end = f"{text} [...] {hit['text']}", stop
    return text


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]],
                   filter_applied: dict[str, Any] | None = None) -> dict[str, Any]:
    options = gold_options(question)
    ranked = []
    for rank, hit in enumerate(hits, start=1):
        source_match = any(hit["document_id"] == o["source_id"] and int(hit["page"]) in o["pages"] for o in options)
        evidence_match = any(hit["document_id"] == o["source_id"] and int(hit["page"]) in o["pages"]
                             and evidence_in(hit["text"], o["quotes"]) for o in options)
        coverage = max((evidence_coverage(hit["text"], o["quotes"]) for o in options
                        if hit["document_id"] == o["source_id"] and int(hit["page"]) in o["pages"]), default=0.0)
        ranked.append({
            "rank": rank, "chunk_id": hit["chunk_id"], "score": round(hit["score"], 6),
            "document_id": hit["document_id"], "source": hit["source"], "page": hit["page"],
            "metadata": hit["metadata"], "text": hit["text"],
            "gold_source_page": source_match, "evidence_hit": evidence_match,
            "evidence_word_coverage": round(coverage, 3),
            "matches_question_filter": matches_filter(hit, question.get("filter")) if question.get("filter") else None,
        })

    hit1 = bool(ranked) and ranked[0]["evidence_hit"]
    hit3 = any(r["evidence_hit"] for r in ranked)
    multi_chunk = False
    if not hit3:
        for option in options:
            same_page = [h for h in hits if h["document_id"] == option["source_id"] and int(h["page"]) in option["pages"]]
            by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
            for h in same_page:
                by_page[int(h["page"])].append(h)
            if any(len(group) > 1 and evidence_in(_merged_page_text(group), option["quotes"]) for group in by_page.values()):
                multi_chunk = True
    failure = None
    if not hit3:
        if multi_chunk:
            failure = "evidence_split_across_chunks"
        elif any(r["gold_source_page"] and r["evidence_word_coverage"] >= 0.5 for r in ranked):
            failure = "gold_page_partial_evidence"
        elif any(r["gold_source_page"] for r in ranked):
            failure = "gold_page_without_evidence"
        elif any(r["document_id"] in {o["source_id"] for o in options} for r in ranked):
            failure = "right_document_wrong_page"
        else:
            failure = "wrong_document"

    question_filter = question.get("filter")
    return {
        "id": question["id"], "split": question["split"], "type": question.get("type", ""),
        "question": question["question"], "expected_answer": question.get("expected_answer", ""),
        "gold": [{"source_id": o["source_id"], "pages": sorted(o["pages"]), "evidence_quote": o["quotes"]} for o in options],
        "question_filter": question_filter, "filter_applied": filter_applied,
        "hit_at_1": hit1, "evidence_hit_at_3": hit3,
        "multi_chunk_evidence_hit_at_3": hit3 or multi_chunk,
        "failure_type": failure,
        "top3_filter_compliant": (all(r["matches_question_filter"] for r in ranked) if question_filter else None),
        "hits": ranked,
    }


def _ratio(n: int, total: int) -> dict[str, Any]:
    return {"n": n, "N": total, "text": f"{n}/{total}", "rate": round(n / total, 3) if total else None}


def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate by strategy and split; always report numerator/denominator."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        groups[(trace["strategy"], trace["split"])].append(trace)
    summary: dict[str, Any] = {}
    for (strategy, split), items in sorted(groups.items()):
        filtered = [t for t in items if t["question_filter"]]
        failures: dict[str, int] = defaultdict(int)
        for t in items:
            if t["failure_type"]:
                failures[t["failure_type"]] += 1
        summary.setdefault(strategy, {})[split] = {
            "questions": len(items),
            "hit_at_1": _ratio(sum(t["hit_at_1"] for t in items), len(items)),
            "evidence_hit_at_3": _ratio(sum(t["evidence_hit_at_3"] for t in items), len(items)),
            "multi_chunk_evidence_hit_at_3 (bổ sung)": _ratio(sum(t["multi_chunk_evidence_hit_at_3"] for t in items), len(items)),
            "filter_questions": len(filtered),
            "top3_filter_compliance": (_ratio(sum(bool(t["top3_filter_compliant"]) for t in filtered), len(filtered))
                                       if filtered else "N/A"),
            "failure_types": dict(failures),
            "mean_search_ms": round(sum(t.get("search_ms", 0) for t in items) / len(items), 2),
        }
    return summary
