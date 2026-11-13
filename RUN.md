# Cách setup và chạy bài làm

Pipeline: đọc PDF/TXT/MD → chuẩn hóa → chunk (A cố định, B theo heading/đoạn) → embedding thật → vector store (Chroma, lưu trên đĩa) → truy xuất top-3 cho A/B/C → Hit@1, Evidence Hit@3, trace → sheet chấm tay. Code nằm ở `starter/src/`, mọi lệnh chạy từ thư mục `starter/`.

Code **không** tự viết câu hỏi, nhãn chấm tay hay báo cáo. Đó là phần bạn phải làm và kiểm chứng.

## 0. Cài đặt (một lần)

Cần Python 3.11–3.14 và khoảng **2 GB trống** (PyTorch CPU khoảng 1 GB, model e5-small khoảng 0,5 GB). **Không cần Docker**: vector store mặc định là Chroma, chạy local và lưu xuống `results/chroma`.

```powershell
cd starter
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --timeout 300 --retries 10 -r requirements.txt -r requirements-local-embedding.txt
Copy-Item .env.example .env
```

Nếu `pip` bị timeout (mạng chậm), chạy lại đúng lệnh đó — nó tải tiếp phần còn thiếu chứ không tải lại từ đầu.

Embedding mặc định là `intfloat/multilingual-e5-small`: chạy local, 384 chiều, tối đa 512 token, hỗ trợ tiếng Việt và tiếng Anh, chi phí 0. Model được tải ở lần chạy đầu. Nếu muốn dùng API, sửa `EMBEDDING_PROVIDER=openai_compatible` trong `.env` (có ví dụ cho OpenAI, OpenRouter và Gemini). Dùng một model duy nhất cho cả ba cấu hình. Nếu đổi model, chạy `python -m src.run reset-db --yes` rồi ingest lại.

**Vector store:** mặc định `VECTOR_STORE=chroma` trong `.env` — Chroma lưu bền vững trên đĩa (`results/chroma`), metric cosine, search xấp xỉ (HNSW). Ở cỡ corpus này HNSW trả về đúng láng giềng gần nhất, nhưng trong báo cáo vẫn ghi là approximate search. Muốn dùng PostgreSQL + pgvector thay thế thì cần Docker: đổi `VECTOR_STORE=pgvector`, cài thêm `pip install "psycopg[binary]==3.3.8"`, chạy `docker compose -p day6_student_starter up -d --wait` rồi `python scripts/check_setup.py`.

## 1. Corpus (mốc 0:25)

1. Tải tài liệu được phép dùng vào `data/raw/`. Nên chọn PDF có **bookmark/outline** (ví dụ sách OpenStax), vì metadata `section` được lấy từ outline của chính PDF.
2. `data/manifest.json` (đã có): mô tả nguồn, giấy phép, phiên bản và trang được tính. `content_pages` là các trang PDF vật lý được tính; bỏ bìa, mục lục và tài liệu tham khảo. Trường `metadata` chỉ chứa giá trị có thật trong nguồn.
3. Chạy:

```powershell
python -m src.run check-corpus
```

Lệnh này tạo `data/manifest.resolved.json` (số trang, danh sách trang được tính, số ký tự), `results/pages_report.csv` (trạng thái từng trang), `results/ingest_errors.jsonl` (trang rỗng/scan, PDF thiếu outline…) và `results/corpus_summary.json`. Yêu cầu **≥ 30 trang nội dung**.

## 2. Benchmark (mốc 1:00, viết TRƯỚC khi chạy truy xuất)

```powershell
python -m src.run show-page <source_id> <so_trang>    # in text đã chuẩn hóa để copy evidence
```

`benchmark/questions.json` (đã có) gồm ≥ 20 câu: ≥ 15 dev, ≥ 5 test, ≥ 3 câu có `filter` (trong đó ≥ 1 câu ở test). Mỗi câu gồm:

| Trường | Ghi chú |
|---|---|
| `id`, `split` (`dev`/`test`), `type` | `type` tùy chọn: `truc_tiep`, `dien_dat_lai`, `ngoai_le`, `metadata_filter`… |
| `question`, `expected_answer` | Không đưa tên nguồn/đáp án vào câu hỏi |
| `gold_source_id`, `gold_page` | `gold_page` là số nguyên hoặc danh sách trang hợp lệ |
| `evidence_quote` | Một chuỗi, hoặc danh sách chuỗi (tất cả đều bắt buộc) |
| `alternatives` | Các gold khác cũng hợp lệ (tùy chọn) |
| `filter` | `{"section": "..."}` hoặc trường metadata khác; giá trị phải tồn tại. Ghi thêm `filter_context` để giải thích vì sao filter là input có sẵn trước khi truy xuất |

```powershell
python -m src.run validate-benchmark
```

Lệnh kiểm tra số câu, split, filter, và **evidence có thật ở trang gold sau chuẩn hóa** (nếu không thấy sẽ chỉ ra trang có chứa). Kết quả ghi ở `results/benchmark_validation.json`. Sau bước này **không sửa tập test nữa**.

## 3. Ingest ba chiến lược (mốc 2:40)

Cấu hình nằm ở `starter/config/strategies.json`:

| | Chunk | Truy xuất |
|---|---|---|
| A | Cố định 500 ký tự, overlap 50, trong từng trang | Không lọc |
| B | Theo heading/đoạn, tối đa 800 ký tự (heading đi cùng đoạn sau, đoạn dài tách theo câu) | Không lọc |
| C | **Dùng chung index với B** (cùng chunk, cùng vector) | Lọc metadata, chỉ khi câu hỏi có `filter` |

A và B chỉ khác cách chunk; B và C chỉ khác filter.

```powershell
python -m src.run ingest --rebuild    # embed lại toàn bộ để đo thời gian/chi phí đầy đủ
python -m src.run ingest              # chạy lần 2: kỳ vọng mới 0, cập nhật 0, xóa 0, trùng 0
```

Kết quả:
- `results/ingest_stats.json`: model, số chiều, phiên bản thư viện/PostgreSQL/pgvector, số chunk, phân bố độ dài, token, số chunk vượt giới hạn token, thời gian, chi phí.
- `results/ingest_history.jsonl`: mỗi lần ingest một dòng; là minh chứng nạp lại không nhân đôi.
- `results/chunks/A.jsonl`, `results/chunks/B.jsonl`.
- `results/audit/coherence.csv`: 10 vị trí cố định, chọn trước khi có điểm (trang rải đều theo thứ tự `document_id`, `page`; vị trí là giữa trang).

Khi một tài liệu thay đổi, chunk của nó được embed lại theo hash nội dung, và chunk không còn tồn tại bị xóa.

Store dùng exact search (không có index ANN), metric cosine, filter viết trong SQL bằng tham số.

## 4. Đánh giá (mốc 3:30)

```powershell
python -m src.run evaluate --split dev     # chọn cấu hình theo quy tắc đã chốt → results/selection.json
python -m src.run evaluate --split test    # chỉ chạy sau khi đã chốt
```

Kết quả:
- `results/traces/<A|B|C>_<dev|test>.jsonl`: câu hỏi, filter đã áp dụng, top-3 (id, text, score, metadata), cờ hit theo từng rank, loại lỗi.
- `results/metrics.md` và `results/metrics.json`: Hit@1 và Evidence Hit@3 dạng n/N theo strategy × split, multi-chunk Hit@3 (chỉ là số bổ sung), tỷ lệ tuân thủ filter.
- `results/per_question.csv`.
- `artifacts/experiment_log.csv`: được điền tự động; cột `hypothesis` và `notes` của bạn được giữ nguyên.
- `results/audit/metadata.csv`, `results/audit/grounding.csv`.

`failure_type` dùng để chọn ví dụ lỗi: `evidence_split_across_chunks`, `gold_page_partial_evidence`, `gold_page_without_evidence`, `right_document_wrong_page`, `wrong_document`.

Thử truy xuất nhanh:

```powershell
python -m src.run query "câu hỏi" --strategy C --filter "section=Tên chương"
```

## 5. Chấm tay rồi tổng hợp

Mở bằng Excel và điền các cột còn trống:

| File | Cột cần điền | Thang |
|---|---|---|
| `results/audit/coherence.csv` | `label_0_2`, `reason` | 0 = cắt giữa ý, 1 = thiếu ngữ cảnh, 2 = tự hiểu được. Dòng C giống B; để trống thì tự lấy nhãn của B |
| `results/audit/metadata.csv` | `metadata_correct_0_1`, `error_note` | Mở PDF đối chiếu source, page, section. Cột `auto_*` chỉ để hỗ trợ |
| `results/audit/grounding.csv` | `label_0_2`, `reason` | 0 = không có bằng chứng, 1 = thiếu điều kiện, 2 = đủ để trả lời và truy nguồn |

```powershell
python -m src.run audit-summary    # tạo results/audit_summary.md (bảng thứ 2 của mẫu báo cáo)
```

Chạy lại `evaluate` sẽ **không ghi đè** sheet đã có nhãn; bản mới được lưu thành `*.new.csv`.

## 6. Ghép vào báo cáo

| Mục trong báo cáo | Lấy từ |
|---|---|
| Nguồn, số trang, quy tắc tính | `data/manifest.json`, `data/manifest.resolved.json`, `results/corpus_summary.json` |
| Chuẩn hóa, lỗi | `results/ingest_errors.jsonl`, `results/pages_report.csv`, mục `removed_repeated_header_footer` trong manifest.resolved |
| Model, store, phiên bản, chi phí | `results/ingest_stats.json`, `results/run_info.json` |
| Bảng kết quả 1 | `results/metrics.md` |
| Bảng kết quả 2 | `results/audit_summary.md` |
| Ví dụ lỗi | `results/per_question.csv` và trace tương ứng |
| Tiêu chí chọn đã chốt | `starter/config/strategies.json` (`selection_rule`), `results/selection.json` |

`data/processed/pages.jsonl` chứa toàn văn đã chuẩn hóa nên bị `.gitignore` bỏ qua. Nếu giấy phép tài liệu không cho tái phân phối, đừng commit `data/raw/`; ghi cách tải trong manifest.
