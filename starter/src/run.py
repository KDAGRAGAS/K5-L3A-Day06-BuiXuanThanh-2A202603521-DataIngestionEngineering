"""Day 06 pipeline CLI. Run from starter/:  python -m src.run <command>

  check-corpus        đọc manifest + tài liệu, đếm trang nội dung, log trang lỗi/rỗng
  show-page DOC PAGE  in text trang đã chuẩn hóa (để copy evidence_quote chính xác)
  validate-benchmark  kiểm tra >=20 câu, split, filter, evidence có thật ở trang gold
  ingest              chunk A/B, embed, lưu pgvector; chạy lại không nhân đôi
  evaluate --split dev|test|all   top-3 cho A/B/C, trace, Hit@1/Hit@3, sheet chấm tay
  query "câu hỏi" [--strategy B] [--filter key=value]   thử truy xuất
  audit-summary       tổng hợp nhãn thủ công đã điền trong results/audit/*.csv
  reset-db --yes      xóa bảng vector (khi đổi embedding model)
  all                 check-corpus -> validate-benchmark -> ingest -> evaluate --split all
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from . import audit
from .benchmark import load_questions, validate
from .chunking import chunk_pages, length_stats
from .config import PATHS, REPO_ROOT, indexes, load_strategies, open_store, vector_store_backend
from .evaluation import evaluate_query, summarize
from .ingestion import Corpus, load_corpus
from .models import Chunk

MIN_CONTENT_PAGES = 30


def _now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _versions() -> dict[str, str]:
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for package in ("psycopg", "pymupdf", "sentence-transformers", "torch", "transformers"):
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            pass
    return versions


# --------------------------------------------------------------------- corpus
def cmd_check_corpus(args: argparse.Namespace) -> Corpus:
    PATHS.ensure()
    corpus = load_corpus(PATHS.manifest)
    total = len(corpus.pages)
    if not corpus.report:
        sys.exit("Manifest không có trang nào đọc được.")
    _write_json(PATHS.manifest.with_name("manifest.resolved.json"), {"documents": corpus.documents})
    with (PATHS.results / "pages_report.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(corpus.report[0].keys()))
        writer.writeheader()
        writer.writerows(corpus.report)
    _write_jsonl(PATHS.results / "ingest_errors.jsonl", corpus.errors)
    _write_jsonl(REPO_ROOT / "data" / "processed" / "pages.jsonl",
                 [{"document_id": p.document_id, "page": p.page, "source": p.source, "metadata": p.metadata,
                   "chars": len(p.text), "text": p.text} for p in corpus.pages])
    summary = {
        "checked_at": _now(), "counted_content_pages": total, "minimum": MIN_CONTENT_PAGES,
        "meets_minimum": total >= MIN_CONTENT_PAGES,
        "documents": [{k: d[k] for k in ("source_id", "physical_pages_or_units", "counted_content_page_count",
                                          "counted_chars")} for d in corpus.documents],
        "errors": len(corpus.errors),
        "status_counts": {s: sum(r["status"] == s for r in corpus.report) for s in {r["status"] for r in corpus.report}},
    }
    _write_json(PATHS.results / "corpus_summary.json", summary)
    print(f"Tài liệu: {len(corpus.documents)} | trang nội dung được tính: {total} (cần >= {MIN_CONTENT_PAGES})")
    for doc in summary["documents"]:
        print(f"  - {doc['source_id']}: {doc['counted_content_page_count']}/{doc['physical_pages_or_units']} trang, "
              f"{doc['counted_chars']} ký tự")
    print(f"Trạng thái trang: {summary['status_counts']} | lỗi ghi nhận: {len(corpus.errors)} "
          f"(results/ingest_errors.jsonl)")
    sections = sorted({p.metadata.get('section', '') for p in corpus.pages} - {""})
    print(f"Giá trị 'section' (từ outline PDF): {len(sections)}" + (f", ví dụ: {sections[:5]}" if sections else ""))
    if total < MIN_CONTENT_PAGES:
        print(f"CẢNH BÁO: chưa đủ {MIN_CONTENT_PAGES} trang nội dung.")
    return corpus


def cmd_show_page(args: argparse.Namespace) -> None:
    corpus = load_corpus(PATHS.manifest)
    page = corpus.page_index().get((args.document_id, args.page))
    if page is None:
        sys.exit(f"Không có trang nội dung {args.document_id} p{args.page} (có thể bị loại hoặc rỗng)")
    print(json.dumps(page.metadata, ensure_ascii=False))
    print("-" * 80)
    print(page.text)


def cmd_validate(args: argparse.Namespace, corpus: Corpus | None = None) -> dict[str, Any]:
    PATHS.ensure()
    corpus = corpus or load_corpus(PATHS.manifest)
    report = validate(load_questions(PATHS.benchmark), corpus)
    report["validated_at"] = _now()
    _write_json(PATHS.results / "benchmark_validation.json", report)
    print(f"Câu hỏi: {report['questions']} ({report['dev']} dev / {report['test']} test), "
          f"có filter: {report['with_filter']} ({report['with_filter_test']} ở test)")
    print(f"Trường có thể dùng làm filter: {report['filter_fields_available']}")
    for warning in report["warnings"]:
        print(f"  [cảnh báo] {warning}")
    for error in report["errors"]:
        print(f"  [LỖI] {error}")
    print("Benchmark hợp lệ." if report["ok"] else "Benchmark CHƯA hợp lệ: sửa lỗi trên trước khi chạy.")
    return report


# --------------------------------------------------------------------- ingest
def _open_store(embedder: Any):
    store = open_store(embedder.dimensions, embedder.model_name)
    store.create_schema()
    return store


def cmd_ingest(args: argparse.Namespace) -> None:
    from .embeddings import make_embedder
    from .store import content_hash

    PATHS.ensure()
    config = load_strategies()
    corpus = load_corpus(PATHS.manifest)
    if len(corpus.pages) < MIN_CONTENT_PAGES and not args.allow_small:
        sys.exit(f"Chỉ có {len(corpus.pages)} trang nội dung (< {MIN_CONTENT_PAGES}). "
                 "Bổ sung tài liệu hoặc chạy với --allow-small để thử.")
    t0 = time.perf_counter()
    embedder = make_embedder()
    model_load_s = time.perf_counter() - t0
    print(f"Embedding: {embedder.model_name} ({embedder.dimensions} chiều), load {model_load_s:.1f}s")

    run: dict[str, Any] = {"run_at": _now(), "embedding": None, "versions": _versions(), "indexes": {},
                           "model_load_s": round(model_load_s, 2), "content_pages": len(corpus.pages)}
    chunks_by_index: dict[str, list[Chunk]] = {}
    with _open_store(embedder) as store:
        run["versions"].update(store.versions())
        for index_name, cfg in indexes(config).items():
            t_chunk = time.perf_counter()
            chunks = chunk_pages(corpus.pages, cfg)
            chunk_s = time.perf_counter() - t_chunk
            chunks_by_index[index_name] = chunks
            ids = [c.chunk_id for c in chunks]
            if len(ids) != len(set(ids)):
                sys.exit(f"Index {index_name}: chunk_id trùng trong cùng một lần chunk")

            if args.rebuild:  # full re-embed, used to measure end-to-end ingest time/cost
                store.delete_obsolete(index_name, set())
            existing = store.existing_hashes(index_name)
            todo = [c for c in chunks if existing.get(c.chunk_id) != (content_hash(c), embedder.model_name)]
            inserted = sum(c.chunk_id not in existing for c in todo)
            t_embed = time.perf_counter()
            vectors = embedder.embed_documents([c.text for c in todo])
            embed_s = time.perf_counter() - t_embed
            t_up = time.perf_counter()
            store.upsert(index_name, todo, vectors)
            deleted = store.delete_obsolete(index_name, set(ids))
            upsert_s = time.perf_counter() - t_up

            stats = length_stats(chunks)
            token_counts = embedder.count_tokens([c.text for c in chunks]) if hasattr(embedder, "count_tokens") else None
            if token_counts:
                limit = getattr(embedder, "max_seq_length", 0)
                stats["total_tokens"] = sum(token_counts)
                stats["max_tokens"] = max(token_counts)
                stats["chunks_over_model_limit"] = sum(t > limit for t in token_counts) if limit else None
                stats["model_max_seq_length"] = limit
            else:
                stats["approx_tokens_chars_div_4"] = stats.get("total_chars", 0) // 4
            dup = store.duplicate_report(index_name)
            run["indexes"][index_name] = {
                "config": cfg, "chunk_stats": stats,
                "upsert": {"embedded_now": len(todo), "inserted": inserted, "updated": len(todo) - inserted,
                           "unchanged_skipped": len(chunks) - len(todo), "deleted_obsolete": deleted},
                "db_check": {**dup, "expected_chunks": len(chunks), "rows_match_expected": dup["rows"] == len(chunks)},
                "time_s": {"chunk": round(chunk_s, 3), "embed": round(embed_s, 3), "store": round(upsert_s, 3),
                           "total": round(chunk_s + embed_s + upsert_s, 3)},
            }
            _write_jsonl(PATHS.results / "chunks" / f"{index_name}.jsonl",
                         [{"chunk_id": c.chunk_id, "document_id": c.document_id, "page": c.page,
                           "char_start": c.metadata["char_start"], "char_end": c.metadata["char_end"],
                           "chars": len(c.text), "section": c.metadata.get("section", ""), "text": c.text}
                          for c in chunks])
            print(f"Index {index_name}: {stats['count']} chunk | mới {inserted}, cập nhật {len(todo) - inserted}, "
                  f"giữ nguyên {len(chunks) - len(todo)}, xóa {deleted} | DB {dup['rows']} dòng, "
                  f"trùng {dup['duplicate_rows']} | embed {embed_s:.1f}s")
            if token_counts and stats.get("chunks_over_model_limit"):
                print(f"  CẢNH BÁO: {stats['chunks_over_model_limit']} chunk vượt {stats['model_max_seq_length']} token "
                      "-> bị model cắt bớt khi embed; ghi vào báo cáo hoặc giảm kích thước chunk.")
    run["embedding"] = embedder.describe()
    _write_json(PATHS.results / "ingest_stats.json", run)
    if all(v["upsert"]["embedded_now"] == v["chunk_stats"]["count"] for v in run["indexes"].values()):
        # Every chunk embedded in this run: the honest end-to-end time/cost for the report.
        _write_json(PATHS.results / "ingest_stats_full.json", run)
    with (PATHS.results / "ingest_history.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"run_at": run["run_at"], "model": embedder.model_name,
                                 **{k: {**v["upsert"], **v["db_check"]} for k, v in run["indexes"].items()}},
                                ensure_ascii=False) + "\n")

    strategies = config["strategies"]
    index_of = {name: cfg["index"] for name, cfg in strategies.items()}
    sheet = audit.coherence_sheet(PATHS.audit / "coherence.csv", corpus,
                                  {name: chunks_by_index[cfg["index"]] for name, cfg in strategies.items()},
                                  index_of, int(config["coherence_sample_size"]))
    print(f"Sheet chấm coherence (vị trí cố định trước khi có điểm): {_rel(sheet)}")
    print("Chạy lại `ingest` lần 2: kỳ vọng mới 0, cập nhật 0, xóa 0, trùng 0 (minh chứng nạp lại không nhân đôi).")


# ------------------------------------------------------------------- evaluate
def _load_all_traces() -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for path in sorted(PATHS.traces.glob("*.jsonl")):
        traces.extend(_read_jsonl(path))
    return traces


def _select_on_dev(summary: dict[str, Any], config: dict[str, Any], chunk_counts: dict[str, int]) -> dict[str, Any]:
    candidates = []
    for name, cfg in config["strategies"].items():
        dev = summary.get(name, {}).get("dev")
        if not dev:
            continue
        candidates.append((dev["evidence_hit_at_3"]["n"], dev["hit_at_1"]["n"], -chunk_counts.get(cfg["index"], 0), name))
    candidates.sort(reverse=True)
    return {"selected_strategy": candidates[0][3] if candidates else None,
            "rule": config.get("selection_rule", ""),
            "ranking": [{"strategy": c[3], "dev_hit3": c[0], "dev_hit1": c[1], "chunks": -c[2]} for c in candidates]}


def _update_experiment_log(config: dict[str, Any], summary: dict[str, Any], embedding: dict[str, Any]) -> None:
    path = PATHS.experiment_log
    existing: dict[str, dict[str, str]] = {}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            existing = {row["strategy"]: row for row in csv.DictReader(handle) if row.get("strategy")}
    fields = ["strategy", "chunking", "chunk_size", "overlap", "metadata", "filter", "vectorizer", "hypothesis",
              "dev_hit1", "dev_hit3", "test_hit1", "test_hit3", "notes"]
    rows = []
    for name, cfg in config["strategies"].items():
        old = existing.get(name, {})
        metrics = summary.get(name, {})
        rows.append({
            "strategy": name,
            "chunking": cfg["chunking"],
            "chunk_size": cfg.get("chunk_size") or f"max {cfg.get('max_size')}",
            "overlap": cfg.get("overlap", 0),
            "metadata": "document_id,source,page,page_label,title,section,subsection,char_start,char_end + manifest metadata",
            "filter": "có (khi câu hỏi có filter)" if cfg.get("use_filter") else "không",
            "vectorizer": f"{embedding.get('model')} ({embedding.get('dimensions')}d, {embedding.get('provider')}), pgvector cosine exact",
            "hypothesis": old.get("hypothesis") or cfg.get("hypothesis", ""),
            "dev_hit1": metrics.get("dev", {}).get("hit_at_1", {}).get("text", ""),
            "dev_hit3": metrics.get("dev", {}).get("evidence_hit_at_3", {}).get("text", ""),
            "test_hit1": metrics.get("test", {}).get("hit_at_1", {}).get("text", ""),
            "test_hit3": metrics.get("test", {}).get("evidence_hit_at_3", {}).get("text", ""),
            "notes": old.get("notes") or f"index {cfg['index']}; {cfg.get('description', '')}",
        })
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _metrics_markdown(summary: dict[str, Any], ingest: dict[str, Any], config: dict[str, Any]) -> str:
    cost = ingest.get("embedding", {}).get("cost_usd", 0)
    lines = ["| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Multi-chunk Hit@3 (bổ sung) | "
             "Filter tuân thủ top-3 | Chunk count | Thời gian ingest đầy đủ (s) | Search TB (ms) | Chi phí (USD) |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, cfg in config["strategies"].items():
        idx = ingest.get("indexes", {}).get(cfg["index"], {})
        for split in ("dev", "test"):
            m = summary.get(name, {}).get(split)
            if not m:
                continue
            compliance = m["top3_filter_compliance"]
            lines.append(
                f"| {name} | {split} | {m['hit_at_1']['text']} | {m['evidence_hit_at_3']['text']} | "
                f"{m['multi_chunk_evidence_hit_at_3 (bổ sung)']['text']} | "
                f"{compliance['text'] if isinstance(compliance, dict) else compliance} | "
                f"{idx.get('chunk_stats', {}).get('count', '')} | {idx.get('time_s', {}).get('total', '')} "
                f"(embed {idx.get('upsert', {}).get('embedded_now', '')} chunk) | "
                f"{m['mean_search_ms']} | {cost} |")
    return "\n".join(lines) + "\n"


def cmd_evaluate(args: argparse.Namespace) -> None:
    from .embeddings import make_embedder

    PATHS.ensure()
    config = load_strategies()
    corpus = load_corpus(PATHS.manifest)
    questions = load_questions(PATHS.benchmark)
    report = validate(questions, corpus)
    if not report["ok"] and not args.force:
        for error in report["errors"]:
            print(f"  [LỖI] {error}")
        sys.exit("Benchmark chưa hợp lệ (chạy validate-benchmark). Dùng --force chỉ khi thử nghiệm.")
    splits = ["dev", "test"] if args.split == "all" else [args.split]
    selection_path = PATHS.results / "selection.json"
    if "test" in splits and "dev" not in splits and not selection_path.exists():
        print("CẢNH BÁO: chưa có results/selection.json -> chưa chốt cấu hình trên dev trước khi xem test.")

    embedder = make_embedder()
    stats_path = next((p for p in (PATHS.results / "ingest_stats_full.json", PATHS.results / "ingest_stats.json")
                       if p.exists()), None)
    ingest_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path else {}
    with _open_store(embedder) as store:
        for index_name in indexes(config):
            if store.duplicate_report(index_name)["rows"] == 0:
                sys.exit(f"Index {index_name} trống: chạy `python -m src.run ingest` trước.")
        top_k = int(config["top_k"])
        for split in splits:
            subset = [q for q in questions if q["split"] == split]
            query_vectors = {}
            for q in subset:
                t = time.perf_counter()
                query_vectors[q["id"]] = (embedder.embed_query(q["question"]), (time.perf_counter() - t) * 1000)
            for name, cfg in config["strategies"].items():
                traces = []
                for q in subset:
                    vector, embed_ms = query_vectors[q["id"]]
                    applied = q["filter"] if (cfg["use_filter"] and q.get("filter")) else None
                    t = time.perf_counter()
                    hits = store.search(cfg["index"], vector, top_k=top_k, filters=applied)
                    search_ms = (time.perf_counter() - t) * 1000
                    trace = evaluate_query(q, hits, applied)
                    trace.update({"strategy": name, "index": cfg["index"], "embedding_model": embedder.model_name,
                                  "query_embed_ms": round(embed_ms, 2), "search_ms": round(search_ms, 2),
                                  "run_at": _now(),
                                  "filter_note": ("filter áp dụng" if applied else
                                                  "câu không có filter -> chạy không lọc" if cfg["use_filter"] else
                                                  "cấu hình không dùng filter")})
                    traces.append(trace)
                _write_jsonl(PATHS.traces / f"{name}_{split}.jsonl", traces)
                ok = sum(t["evidence_hit_at_3"] for t in traces)
                print(f"[{split}] {name}: Hit@1 {sum(t['hit_at_1'] for t in traces)}/{len(traces)} | "
                      f"Evidence Hit@3 {ok}/{len(traces)}")

    all_traces = _load_all_traces()
    summary = summarize(all_traces)
    chunk_counts = {k: v.get("chunk_stats", {}).get("count", 0) for k, v in ingest_stats.get("indexes", {}).items()}
    if "dev" in splits:
        test_seen = any(t["split"] == "test" for t in all_traces if "test" not in splits)
        selection = {**_select_on_dev(summary, config, chunk_counts), "decided_at": _now(),
                     "test_results_existed_before_this_selection": test_seen}
        _write_json(selection_path, selection)
        print(f"Chọn trên dev theo quy tắc đã chốt: {selection['selected_strategy']} (results/selection.json)")
        if test_seen:
            print("CẢNH BÁO: đã có kết quả test trước lần chọn này -> ghi rõ trong báo cáo, test không còn độc lập.")

    _write_json(PATHS.results / "metrics.json", {"generated_at": _now(), "embedding": embedder.describe(),
                                                 "top_k": config["top_k"], "summary": summary})
    (PATHS.results / "metrics.md").write_text(_metrics_markdown(summary, ingest_stats, config), encoding="utf-8")
    with (PATHS.results / "per_question.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["strategy", "split", "id", "type", "filter_applied", "hit_at_1", "evidence_hit_at_3",
                         "failure_type", "top1_ref", "top1_score", "gold"])
        for t in sorted(all_traces, key=lambda x: (x["strategy"], x["split"], x["id"])):
            top = t["hits"][0] if t["hits"] else {}
            writer.writerow([t["strategy"], t["split"], t["id"], t["type"],
                             json.dumps(t["filter_applied"], ensure_ascii=False) if t["filter_applied"] else "",
                             int(t["hit_at_1"]), int(t["evidence_hit_at_3"]), t["failure_type"] or "",
                             f"{top.get('document_id')} p{top.get('page')}" if top else "", top.get("score", ""),
                             "; ".join(f"{g['source_id']} p{g['pages']}" for g in t["gold"])])
    _update_experiment_log(config, summary, embedder.describe())
    _write_json(PATHS.results / "run_info.json", {"evaluated_at": _now(), "splits": splits,
                                                  "versions": _versions(), "embedding": embedder.describe(),
                                                  "strategies_file": _rel(PATHS.strategies),
                                                  "benchmark_file": _rel(PATHS.benchmark)})
    if "test" in splits:
        test_traces = [t for t in all_traces if t["split"] == "test"]
        audit.metadata_sheet(PATHS.audit / "metadata.csv", test_traces, corpus)
        audit.grounding_sheet(PATHS.audit / "grounding.csv", test_traces)
        print("Sheet chấm tay: results/audit/metadata.csv, results/audit/grounding.csv (+ coherence.csv từ ingest)")
    print("Kết quả: results/metrics.md, results/traces/, results/per_question.csv, artifacts/experiment_log.csv")


# ---------------------------------------------------------------------- misc
def cmd_query(args: argparse.Namespace) -> None:
    from .embeddings import make_embedder

    config = load_strategies()
    cfg = config["strategies"][args.strategy]
    filters: dict[str, Any] = {}
    for item in args.filter or []:
        key, _, value = item.partition("=")
        filters.setdefault(key, []).append(value)
    embedder = make_embedder()
    with _open_store(embedder) as store:
        hits = store.search(cfg["index"], embedder.embed_query(args.question), top_k=args.k, filters=filters or None)
    for rank, hit in enumerate(hits, start=1):
        print(f"#{rank} {hit['score']:.4f} {hit['chunk_id']} | {hit['document_id']} p{hit['page']} | "
              f"section={hit['metadata'].get('section', '')}")
        print("   " + hit["text"][:400].replace("\n", " "))


def cmd_audit_summary(args: argparse.Namespace) -> None:
    metrics = json.loads((PATHS.results / "metrics.json").read_text(encoding="utf-8")).get("summary", {}) \
        if (PATHS.results / "metrics.json").exists() else {}
    result = audit.summarize_audit(PATHS.audit, metrics)
    _write_json(PATHS.results / "audit_summary.json", {"generated_at": _now(), "by_strategy": result})
    lines = ["| Strategy | Coherence TB /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ test (n/N hoặc N/A) | "
             "Grounding 0/1/2 (số câu) | Grounding TB /2 |", "|---|---|---|---|---|---|"]
    for name, r in result.items():
        g = r["grounding_counts_0_1_2"]
        coherence = "chưa chấm" if r["coherence_mean_of_2"] is None else f"{r['coherence_mean_of_2']} ({r['coherence_n']})"
        grounding = "chưa chấm" if r["grounding_mean_of_2"] is None else r["grounding_mean_of_2"]
        lines.append(f"| {name} | {coherence} | {r['metadata_correct']} | "
                     f"{r['filter_compliance_test']} | {g['0']}/{g['1']}/{g['2']} | {grounding} |")
    (PATHS.results / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def cmd_reset(args: argparse.Namespace) -> None:
    if not args.yes:
        sys.exit("Thêm --yes để xác nhận xóa toàn bộ vector đã lưu.")
    # reset() drops everything regardless of dimensions/model, so no embedder load is needed.
    store = open_store(1, "reset")
    store.reset()
    store.close()
    print(f"Đã xóa toàn bộ vector ({vector_store_backend()}). Chạy lại `ingest`.")


def cmd_all(args: argparse.Namespace) -> None:
    corpus = cmd_check_corpus(args)
    if not cmd_validate(args, corpus)["ok"] and not args.force:
        sys.exit("Dừng: benchmark chưa hợp lệ.")
    cmd_ingest(args)
    args.split = "all"
    cmd_evaluate(args)


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="python -m src.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check-corpus")
    show = sub.add_parser("show-page")
    show.add_argument("document_id")
    show.add_argument("page", type=int)
    sub.add_parser("validate-benchmark")
    ingest = sub.add_parser("ingest")
    ingest.add_argument("--allow-small", action="store_true", help="cho phép < 30 trang (chỉ để thử)")
    ingest.add_argument("--rebuild", action="store_true", help="xóa index rồi embed lại toàn bộ (đo thời gian đầy đủ)")
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--split", choices=["dev", "test", "all"], default="dev")
    evaluate.add_argument("--force", action="store_true", help="chạy dù benchmark còn lỗi (chỉ để thử)")
    query = sub.add_parser("query")
    query.add_argument("question")
    query.add_argument("--strategy", default="B")
    query.add_argument("--filter", action="append", help="key=value, lặp lại được")
    query.add_argument("-k", type=int, default=3)
    sub.add_parser("audit-summary")
    reset = sub.add_parser("reset-db")
    reset.add_argument("--yes", action="store_true")
    everything = sub.add_parser("all")
    everything.add_argument("--allow-small", action="store_true")
    everything.add_argument("--rebuild", action="store_true")
    everything.add_argument("--force", action="store_true")

    args = parser.parse_args(argv)
    commands = {"check-corpus": cmd_check_corpus, "show-page": cmd_show_page, "validate-benchmark": cmd_validate,
                "ingest": cmd_ingest, "evaluate": cmd_evaluate, "query": cmd_query,
                "audit-summary": cmd_audit_summary, "reset-db": cmd_reset, "all": cmd_all}
    if args.command is None:
        parser.print_help()
        return
    try:
        commands[args.command](args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"LỖI: {exc}")


if __name__ == "__main__":
    main()
