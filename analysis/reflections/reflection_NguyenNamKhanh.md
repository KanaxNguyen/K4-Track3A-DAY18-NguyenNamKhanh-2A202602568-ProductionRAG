# Reflection — Lab 18: Production RAG

**Học viên:** Nguyễn Nam Khánh · **Mã học viên:** 2A202602568
**Khóa:** K4 — Track 3A · **Ngày kiểm chứng:** 2026-10-04

## Phần 1: Mapping bài giảng → triển khai

| Lecture concept | Module | Hàm cụ thể | Quan sát đã đo |
|---|---|---|---|
| Semantic, hierarchical, structure-aware chunking | M1 | chunk_semantic, chunk_hierarchical, chunk_structure_aware, compare_strategies | Pipeline chia từng tài liệu: baseline 57 chunks, production 106 children từ 26 tài liệu có text. A/B trên corpus nối chung: basic 51, semantic 364, hierarchical 104 children, structure 106. Hai cách đếm có scope khác nhau. |
| BM25 + dense fusion | M2 | segment_vietnamese, BM25Search, DenseSearch, reciprocal_rank_fusion | Dùng Underthesea, BGE-M3 và Qdrant server thật, không hashing fallback. RRF cộng reciprocal ranks; Context Recall tăng từ 0.9000 lên 0.9500. |
| Cross-encoder reranking | M3 | CrossEncoderReranker._load_model, rerank, benchmark_reranker | BGE reranker chạy CPU, top-20 → top-3. Warm benchmark 5 lần trung bình 2207.97 ms; reranking trong 20 query trung bình 2739.77 ms, gồm cold-start. |
| RAGAS 4 metrics | M4 | evaluate_ragas, failure_analysis, save_report | Hai pipeline đều completed với đủ 80 điểm. Production Faithfulness 0.8694, Relevancy 0.5301, Precision 0.9750, Recall 0.9500; còn lỗi số học và đa bước dù ba metrics đạt 0.70. |
| Contextual enrichment | M5 | _enrich_single_call, contextual_prepend, enrich_chunks | 106/106 chunks dùng combined qua OpenRouter, không fallback. Một request/chunk, cache theo text/source/model/prompt; lỗi combined chuyển sang xử lý local thay vì phát sinh thêm bốn API calls. |

Mốc so sánh bốn metrics nằm trong failure_analysis.md và reports/benchmark_summary.json. M1 thống kê độ dài theo ký tự, không phải token. Semantic threshold 0.85 tạo nhiều chunks ngắn; số lượng chunks tự nó chưa chứng minh chất lượng retrieval cao hơn.

## Phần 2: Khó khăn, kiểm tra và cách xử lý

1. **Chọn sai interpreter:** hệ thống Python 3.9 báo ImportError: cannot import name 'pairwise' from 'itertools'. Dự án đã có .venv Python 3.11.16 và dependencies đầy đủ; chạy trực tiếp interpreter trong .venv giải quyết lỗi môi trường. pip check xác nhận không có dependency bị hỏng.
2. **Tên key sai:** key có định dạng OpenRouter đã nằm dưới OPENAI_API_KEY. Chuyển giá trị cục bộ sang OPENROUTER_API_KEY; chat/embeddings/RAGAS dùng endpoint OpenRouter. Không đưa giá trị key vào source hoặc report; .env và .cache bị ignore.
3. **Docker trong sandbox:** permission denied while trying to connect to the docker API. Khi kiểm tra Docker daemon ngoài sandbox, dịch vụ hoạt động; docker compose up -d và healthz thành công. Production metadata xác nhận qdrant_server, không in-memory fallback.
4. **Summary phình dài:** test_summarize_shorter_than_original thất bại với 132 ký tự so với giới hạn 130 của một đoạn 65 ký tự. Đoạn ngắn dùng extractive summary; response dài hơn nguồn chuyển về fallback, giữ nguyên thông tin thay vì cắt tùy tiện.
5. **API ngân sách đồng thời:** HTTP 402, reason in_flight_budget_exhausted, remedy Retry after in-flight requests settle. Đây là lỗi đặt trước ngân sách theo request đồng thời; không kết luận chỉ từ phép trừ tổng credits đã mua/đã dùng rằng key vô hiệu. Giới hạn judge 2.048 output tokens và một worker; đo lại đúng điểm thiếu của câu 10 baseline, giữ nguyên các điểm còn lại. Production không thiếu điểm.
6. **Validator báo sẵn sàng quá sớm:** bản cũ chỉ kiểm tra file/keys và vẫn báo ready khi pytest không chạy. Validator mới kiểm tra exit code, collection errors, đủ 20 hàng × 4 điểm hữu hạn và evaluation_status completed. Bộ test cuối có 44/44 pass; thêm regression cho routing, summary, combined fallback, NaN và validator.

**Kiến thức cần củng cố:** Faithfulness đo grounding, không bảo đảm câu trả lời đủ ý; Recall/Precision cao không bảo đảm phép tính đúng. Câu Senior thiếu bảng lương, câu tạm ứng trả sai 5.000 thay vì 50.000 VNĐ, trong khi một số câu đáp án đúng vẫn có điểm judge thấp. Cần đọc answer/context/reference và giữ kiểm tra thủ công bên cạnh RAGAS.

## Phần 3: Action plan cho project cá nhân

### Project: Production RAG cho bộ chính sách nhân sự

**Hiện trạng đã kiểm chứng:** Markdown/PDF có text → parent/child chunks → combined enrichment → BM25+BGE-M3+RRF → BGE reranking → OpenRouter answer → RAGAS. Latency query trung bình 6068.14 ms trên máy này. Hai PDF scan chưa OCR; test set chỉ gồm 20 câu và đây là một lần đo, chưa có khoảng tin cậy hoặc kiểm thử tải.

**Plan cụ thể:**

1. **Chunking:** giữ hierarchical, thử child size theo token và metadata section; đo recall/latency thay vì chỉ so số chunks. OCR hai PDF scan rồi thêm Q&A tương ứng trước khi kết luận chất lượng toàn corpus.
2. **Search:** parse version/effective_date/superseded, lọc chính sách cũ mặc định; tách query đa bước và chọn các parent khác nguồn để không bỏ sót bảng lương.
3. **Reranking:** giữ BGE trên top-20 và kiểm soát đa dạng parent sau rerank; cân nhắc model nhẹ hơn nếu ngân sách latency dưới 6 giây/query, xác nhận lại chất lượng trước khi đổi.
4. **Generation:** thêm calculator Decimal cho phần trăm/pro-rata, chỉ rõ quy ước tháng 30 ngày khi cần. Tách dữ kiện trong câu hỏi khỏi quy tắc trong tài liệu để giải thích được phép tính.
5. **Evaluation/enrichment:** duy trì combined cache và cap output/concurrency. Tạo held-out test cho numeric/version/multi-hop; thử evaluator tiếng Việt, giữ kiểm tra manual cho các case metric và đáp án bất đồng. Không sửa test set hiện tại để nâng điểm báo cáo.

**Timeline dự kiến:**

- Tuần 1: OCR; calculator; version metadata/filter; thêm câu hỏi mới về các thay đổi.
- Tuần 2: query decomposition và parent diversity; A/B trên held-out set, đo latency nhiều lượt và phân tích bottom-5 mới.

Các mục trên là kế hoạch cải tiến tiếp theo, không được ghi như tính năng đã hoàn thành trong lần benchmark hiện tại.
