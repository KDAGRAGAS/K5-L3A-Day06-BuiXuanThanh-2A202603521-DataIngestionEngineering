# Báo cáo cá nhân — Day 6

> Họ tên: `BÙI XUÂN THÀNH` · MSSV: `2A202603521` · Lớp: L3A  
> Chạy ngày 2026-11-13, Windows 10, Python 3.11.9. Số liệu lấy từ `results/` và `artifacts/experiment_log.csv`.

## Dữ liệu và pipeline

- **Chủ đề, nguồn/quyền, số trang và quy tắc tính trang:** Giáo trình *Introduction to Business 2e* của OpenStax (bản PDF WEB, ©2026 Rice University), tải từ https://openstax.org/details/books/introduction-business-2e. Giấy phép **CC BY-NC-SA 4.0** (ghi ở trang 4 của PDF), attribution "Access for free at openstax.org". Dùng phần nội dung chính của Chương 1 (trang 23–57) và Chương 2 (trang 67–83). Quy tắc tính: số trang PDF vật lý (1-based), bỏ bìa, mục lục, lời nói đầu và phần cuối mỗi chương (Key Terms, Summary, Ethics Activity, Working the Net, Critical Thinking Case, Hot Links). Kết quả: **52 trang nội dung** được tính / 663 trang vật lý, 169.626 ký tự (`results/corpus_summary.json`, `data/manifest.resolved.json`).
- **Chuẩn hóa, lỗi và giới hạn:** Đọc PDF bằng PyMuPDF theo block để giữ ranh giới đoạn văn; chuẩn hóa Unicode NFC, bỏ ký tự ẩn và ligature, gộp khoảng trắng, nối dòng bị ngắt bởi dấu gạch nối. Tự lọc header/footer lặp lại ("Access for free at openstax.org", số trang + tên chương) khi khối text ngắn nằm ở mép trang và lặp trên ≥2% số trang. Số lỗi ghi nhận: **0** (`results/ingest_errors.jsonl`); không có trang rỗng/scan trong phạm vi đã chọn. Giới hạn: metadata `section` lấy từ outline (bookmark) của PDF nên phụ thuộc chất lượng outline; chunk không vượt ranh giới trang nên một đoạn bắc qua 2 trang sẽ bị tách.
- **Embedding / vector store / version / lệnh chạy / chi phí:** Model `intfloat/multilingual-e5-small` (sentence-transformers 6.3.0), chạy **local**, **384 chiều**, giới hạn 512 token, vector chuẩn hóa, prefix `query:`  / `passage:` . Vector store **Chroma 1.5.9**, lưu bền vững trên đĩa tại `results/chroma`, metric **cosine**, search **xấp xỉ (HNSW)**. Dùng cùng một model và store cho cả ba cấu hình. **Chi phí: 0 USD** (chạy local). Lệnh: `python -m src.run check-corpus | validate-benchmark | ingest --rebuild | ingest | evaluate --split dev | evaluate --split test`. Phiên bản khác: torch 2.15.0, transformers 5.22.2, pymupdf 2.1.
- **Metadata bắt buộc, trường lọc, nguồn giá trị:** Mỗi chunk mang `document_id`, `source`, `page` (bắt buộc), cộng `title`, `section`, `subsection` (từ outline PDF), `char_start`, `char_end`, `chunk_index`, `chunking`, và `document_type`, `publisher`, `year` (từ manifest). Trường dùng để lọc: `section` (tên chương, lấy từ outline PDF). Lý do chọn `section`: có sẵn trong nguồn, phân biệt rõ hai chương, và khớp tình huống người dùng đang đọc một chương cụ thể.

## Cấu hình và benchmark

- **A / B / C (khác biệt duy nhất B–C):**
  - **A** — chunk cố định 500 ký tự, overlap 50, trong từng trang; không lọc. → 396 chunk.
  - **B** — chunk theo heading/đoạn, tối đa 800 ký tự (heading đi kèm đoạn sau, đoạn quá dài tách theo câu); không lọc. → 303 chunk.
  - **C** — **dùng chung index với B** (cùng chunk, cùng vector), chỉ khác: áp dụng **metadata filter** khi câu hỏi có `filter`. → 303 chunk (chung với B).
  - A khác B ở **cách chunk**; B khác C duy nhất ở **việc áp dụng filter**.
- **Benchmark:** 20 câu, **15 dev / 5 test**, 3 câu có `filter` theo `section` (1 câu ở test). Các loại: định nghĩa trực tiếp, diễn đạt lại, số liệu, điều kiện, ngoại lệ, lọc metadata. Mỗi câu có `evidence_quote` nguyên văn; công cụ `validate-benchmark` xác minh mọi câu trích tồn tại ở đúng trang gold sau chuẩn hóa, và mỗi câu trích chỉ xuất hiện ở trang gold (không trùng trang). Kết quả: benchmark hợp lệ (`results/benchmark_validation.json`).
- **Tiêu chí chọn chiến lược (chốt trước khi xem test):** chọn cấu hình có Evidence Hit@3 trên dev cao nhất; hòa thì Hit@1 dev cao hơn; hòa tiếp thì ít chunk hơn. Ghi ở `starter/config/strategies.json` (`selection_rule`). Kết quả chọn: **C** (`results/selection.json`, `test_results_existed_before_this_selection: false` — test chưa chạy khi chọn).

## Kết quả

**Retrieval (**`results/metrics.md`**):**

| Strategy | Split | Hit@1 (n/N) | Evidence Hit@3 (n/N) | Filter tuân thủ top-3 | Chunk count | Thời gian ingest (embed, s) | Search TB (ms) | Chi phí |
|---|---|---|---|---|---|---|---|---|
| A | dev | 11/15 | 13/15 | 2/2 | 396 | 17.20 | 2.26 | 0 |
| A | test | 1/5 | 3/5 | 1/1 | 396 | 17.20 | 4.13 | 0 |
| B | dev | 11/15 | 13/15 | 2/2 | 303 | 17.47 | 2.16 | 0 |
| B | test | 5/5 | 5/5 | 0/1 | 303 | 17.47 | 2.44 | 0 |
| C | dev | 11/15 | 13/15 | 2/2 | 303 | 17.47 | 2.53 | 0 |
| C | test | 5/5 | 5/5 | 1/1 | 303 | 17.47 | 4.21 | 0 |

Ghi chú: thời gian trên là thời gian embed + lưu khi dựng lại toàn bộ index (`ingest --rebuild`); lần chạy đầu tải model mất thêm ~219s (tải 471 MB) và ~22s nạp model các lần sau. Phân bố độ dài chunk: A trung bình 471,5 ký tự (median 500, max 500); B trung bình 558,3 (median 611, p25 432, p75 722, max 1.051). Token tối đa mỗi chunk: A 169, B 260 — đều dưới giới hạn 512 token, 0 chunk bị cắt. (Một số chunk B vượt 800 ký tự vì heading được giữ cùng đoạn — ghi nhận là đánh đổi có chủ đích.)

**Nạp lại không nhân đôi:** chạy `ingest` lần hai cho kết quả mới 0, cập nhật 0, xóa 0; DB giữ đúng 396/303 dòng, `duplicate_rows = 0`, `distinct_content_hashes = distinct_chunk_ids` (`results/ingest_stats.json`, `results/ingest_history.jsonl`). Upsert theo khóa ổn định `(strategy, chunk_id)`.

**Năm khía cạnh chấm tay (**`results/audit_summary.md`**):**

| Strategy | Coherence TB /2 (n) | Metadata đủ và đúng (n/N) | Filter tuân thủ test (n/N) | Grounding 0/1/2 (số câu) | Grounding TB /2 |
|---|---|---|---|---|---|
| A | 0.0 (10) | 15/15 | 1/1 | 1/1/3 | 1.4 |
| B | 1.7 (10) | 15/15 | 0/1 | 0/0/5 | 2.0 |
| C | 1.7 (10) | 15/15 | 1/1 | 0/0/5 | 2.0 |

Coherence của C kế thừa nhãn của B (dùng chung chunk). Trace và nhãn: `results/traces/<A|B|C>_<dev|test>.jsonl`, `results/per_question.csv`, `results/audit/*.csv`. Cách chọn mẫu: coherence chấm ở 10 vị trí cố định (trang rải đều theo thứ tự `document_id`, `page`; vị trí = giữa trang) được chốt trước khi có điểm; metadata và grounding chấm top-3 của 5 câu test.

## Phân tích và kết luận

**Dữ kiện so sánh A–B (2 ví dụ lỗi, từ** `results/per_question.csv` **và trace test):**

- **q16** (câu ngoại lệ, gold Chương 2 trang 71): A truy xuất đúng trang 71 nhưng chunk cố định cắt mất câu bằng chứng "can be disregarded only in extreme circumstances, such as during wartime" → `gold_page_without_evidence` (Hit@3 = 0). B/C lấy nguyên đoạn ở trang 71 → trúng.
- **q20** (gold trang 48, định nghĩa "supply curve"): A lấy trang 47 với bằng chứng một phần → `gold_page_partial_evidence` (Hit@3 = 0). B/C lấy trang 48 nơi câu định nghĩa nằm trọn → trúng.
- Tổng hợp: trên test A đạt 1/5 Hit@1 và 3/5 Hit@3, còn B/C đạt 5/5 và 5/5. Coherence tay: A 0.0 vs B 1.7.

**Dữ kiện so sánh B–C:**

- Câu lọc trong test là q17 (stakeholders, Chương 2). Cả B và C đều trúng Evidence Hit@3, nên filter **không** đổi tỷ lệ hit (cả hai 5/5). Khác biệt ở độ tuân thủ phạm vi: B đạt filter compliance test 0/1 (có chunk trong top-3 nằm ngoài Chương 2), C đạt 1/1 (mọi chunk top-3 đều thuộc Chương 2).

- So sánh A–B và B–C (ý nghĩa của các dữ kiện trên): A dễ cài đặt nhưng chunk 500 ký tự đôi khi cắt mất bằng chứng quan trọng; hai ví dụ q16 và q20 cho thấy A bỏ sót phần quyết định của câu trả lời. B giữ đoạn theo heading và đạt coherence tay cao hơn (1,7 so với 0,0), đồng thời đạt 5/5 trên test. C dùng chung chunk với B; ở câu có filter, C không tăng tỷ lệ hit nhưng giữ cả top-3 trong đúng Chương 2, còn B có một kết quả ngoài phạm vi.
- Chiến lược chọn, đánh đổi, điều kiện áp dụng: Báo cáo ghi nhận C là cấu hình được chọn. B và C có cùng số chunk và kết quả hit, nên khác biệt đáng kể là khả năng tuân thủ filter. Lần đo test này, search của C lâu hơn B (4,21 so với 2,44 ms). C phù hợp khi câu hỏi yêu cầu giới hạn theo chương; khi không cần lọc, B cho kết quả tương đương với thời gian tìm kiếm thấp hơn trong lần đo này.
- Hạn chế của dữ liệu, nhãn, test nhỏ (chỉ 5 câu), cách đo thời gian/chi phí: Test chỉ có 5 câu và chỉ 1 câu dùng filter, nên chưa đủ để khẳng định kết quả sẽ giữ nguyên trên bộ câu hỏi lớn hơn. Nhãn coherence và grounding cũng dựa trên mẫu nhỏ, có thể có khác biệt khi người khác chấm. Chi phí 0 USD phản ánh việc chạy local, chưa tính điện và phần cứng; thời gian search trên ít câu cũng dễ dao động.
- Nếu làm lại hoặc mở rộng, cải tiến gì? Mình sẽ mở rộng test với nhiều câu ở cả hai chương, nhất là câu có filter, câu hỏi nhiều điều kiện và bằng chứng nằm gần ranh giới trang. Nên có thêm người chấm độc lập cho coherence/grounding và đo thời gian qua nhiều lượt warm-up để so sánh ổn định hơn.

## Ghi chú về cách dùng AI

Tôi dùng trợ lý AI để hỗ trợ viết khung code pipeline, soạn nháp bộ câu hỏi benchmark và nháp phần số liệu của báo cáo. Tôi đã đọc lại code để hiểu từng bước (đọc PDF → chunk → embedding → Chroma → đánh giá), tự chạy toàn bộ trên máy, tự kiểm tra lại các câu hỏi và câu trích bằng chứng, và tự chấm tay ba bảng coherence/metadata/grounding mà không dùng AI. Phần nhận xét và kết luận trong báo cáo do tôi tự viết.
