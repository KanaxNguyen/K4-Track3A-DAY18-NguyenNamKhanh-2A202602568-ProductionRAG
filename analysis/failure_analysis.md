# Failure Analysis — Lab 18: Production RAG

**Học viên:** Nguyễn Nam Khánh · **Mã học viên:** 2A202602568
**Ngày đo:** 2026-10-04 (GMT+7)

## Kết quả đo thật

| Metric | Naive baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8556 | 0.8694 | +0.0139 |
| Answer Relevancy | 0.5255 | 0.5301 | +0.0046 |
| Context Precision | 0.9333 | 0.9750 | +0.0417 |
| Context Recall | 0.9000 | 0.9500 | +0.0500 |

Cả hai report có trạng thái completed, mỗi pipeline gồm 20 câu hỏi và đủ 80 điểm hữu hạn. Production có ba metrics đạt ít nhất 0.70 theo rubric; Faithfulness đạt ít nhất 0.85. Answer Relevancy chưa đạt 0.70. Đây là một lần đo trên test set cố định, không phải bảo đảm mọi câu trả lời đều đúng.

- Baseline: 57 paragraph chunks, dense-only BGE-M3, không enrichment/reranking.
- Production: 106 child chunks, contextual enrichment combined qua OpenRouter, BM25 + BGE-M3 + RRF, BGE cross-encoder top-20 → top-3, trả parent context và bỏ context trùng.
- 26 tài liệu có text được index. Hai PDF scan không có text layer được bỏ qua đúng hành vi loader; cần OCR nếu muốn mở rộng corpus.
- Qdrant server thật được dùng. Embedding, reranking, enrichment và answer generation không dùng fallback trong production.
- Chat và RAGAS judge: openai/gpt-4o-mini qua OpenRouter; RAGAS embeddings: openai/text-embedding-3-small. Key chỉ nằm trong .env bị Git ignore.

Baseline có một điểm Answer Relevancy bị thiếu vì HTTP 402 in_flight_budget_exhausted khi chạy bốn job đồng thời, không giới hạn output token. Điểm thiếu được đo lại trên chính answer/context đã lưu, giữ nguyên mọi điểm hợp lệ. Production chạy evaluator với một worker và giới hạn 2.048 output token, hoàn tất không thiếu điểm. Không biến NaN thành điểm 0 rồi gắn trạng thái completed.

## Bottom-5 theo trung bình bốn metrics

| Thứ hạng | Câu hỏi | Trung bình | Metric thấp nhất |
|---:|---|---:|---|
| 1 | Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào? | 0.6250 | answer_relevancy |
| 2 | Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu? | 0.6537 | faithfulness |
| 3 | Nghỉ phép không lương 20 ngày cần ai phê duyệt? | 0.7272 | answer_relevancy |
| 4 | Nhân viên được tài trợ khóa học 25 triệu, nghỉ việc sau 8 tháng hoàn thành khóa học. Phải hoàn trả bao nhiêu? | 0.7313 | answer_relevancy |
| 5 | Mật khẩu phải có tối thiểu bao nhiêu ký tự? | 0.7506 | context_precision |

Thứ tự lấy trực tiếp từ reports/ragas_report.json. Bottom-5 có thể gồm câu trả lời đúng về nội dung; điểm thấp cần kiểm tra cùng context và ground truth.

### 1. Senior có 9 năm thâm niên: truy vấn đa bước thiếu khung lương

- **Điểm:** Faithfulness: 1.0000; Answer Relevancy: 0.0000; Context Precision: 1.0000; Context Recall: 0.5000.
- **Output thực tế:** đúng 18 ngày phép; nói không có thông tin lương.
- **Ground truth:** 18 ngày phép và lương Senior 20–35 triệu VNĐ/tháng.
- **Bằng chứng context:** chỉ có nghi_phep_nam_v2024.md và nghi_phep_nam_v2023.md; không có bang_luong_2024.md.
- **Error Tree:** Output đủ ý? Không → Context đủ hai nhánh phép/lương? Không → Query rõ? Có → lỗi coverage ở retrieval/reranking của truy vấn đa bước.
- **Diagnosis:** top-k ưu tiên các child cùng chủ đề nghỉ phép; lấy thêm parent cũ không bổ sung thông tin lương. Faithfulness cao vì phần được trả lời có căn cứ, nhưng Recall và Relevancy thấp vì bỏ một nhánh.
- **Suggested fix:** tách truy vấn thành nhánh phép và nhánh lương, hợp nhất rồi rerank; chọn các parent khác nhau theo nguồn và ưu tiên chính sách còn hiệu lực. Kiểm tra thêm truy vấn đa bước ngoài 20 câu hiện tại.

### 2. Phạt tạm ứng: lỗi số học dù retrieval đủ

- **Điểm:** Faithfulness: 0.2222; Answer Relevancy: 0.3927; Context Precision: 1.0000; Context Recall: 1.0000.
- **Output thực tế:** 5.000 VNĐ.
- **Ground truth:** khoảng 50.000 VNĐ khi dùng quy ước tháng 30 ngày.
- **Bằng chứng context:** tam_ung.md chứa đúng thời hạn 15 ngày và phí 2%/tháng; Precision và Recall đều đạt 1.0.
- **Error Tree:** Output đúng? Không → Context chứa quy tắc cần thiết? Có → Query đủ số tiền và thời gian? Có → lỗi generation/tính toán.
- **Diagnosis:** LLM áp dụng sai số học, lệch một bậc mười. Theo quy ước của ground truth: 15.000.000 × 0,02 × (20−15)/30 = 50.000 VNĐ. Văn bản không quy định số ngày của tháng, nên quy ước cần được nêu rõ.
- **Suggested fix:** dùng calculator với Decimal cho phần số học; chuẩn hóa phần trăm và đơn vị, trả phép tính có thể kiểm tra. Không sửa BM25/dense khi dữ kiện đã được truy xuất đủ.

### 3. Nghỉ không lương 20 ngày: đáp án đúng nhưng judge cho điểm thấp

- **Điểm:** Faithfulness: 0.5000; Answer Relevancy: 0.4087; Context Precision: 1.0000; Context Recall: 1.0000.
- **Output thực tế:** Giám đốc điều hành (CEO) phê duyệt.
- **Bằng chứng context:** nghi_phep_khong_luong.md ghi rõ 16–30 ngày cần CEO; cùng context có quy tắc tự đóng bảo hiểm khi nghỉ trên 14 ngày. Context thứ hai là chính sách phép năm cũ, không cần thiết cho câu hỏi này.
- **Error Tree:** Output trả đúng người phê duyệt? Có → Context chứa đúng quy định? Có → Query rõ? Có → cần kiểm tra sự phù hợp của evaluator/reference trước khi kết luận hallucination.
- **Diagnosis:** câu trả lời đúng phần người phê duyệt; ground truth còn thêm lưu ý bảo hiểm. Faithfulness 0.5 và Relevancy thấp không tương ứng rõ với lỗi nội dung quan sát được. Khả năng judge nhạy với cách diễn đạt/ngôn ngữ là giả thuyết cần kiểm chứng, không phải kết luận đã đo.
- **Suggested fix:** giữ case này trong kiểm tra thủ công, xem statement/NLI trace của RAGAS và thử prompt đánh giá tiếng Việt trên một bộ held-out. Làm rõ expected answer có bắt buộc nhắc phúc lợi hay chỉ cần người phê duyệt; tránh tự động coi mọi điểm thấp là lỗi trả lời.

### 4. Hoàn chi đào tạo: đúng số tiền, thiếu diễn giải điều kiện

- **Điểm:** Faithfulness: 0.5000; Answer Relevancy: 0.4251; Context Precision: 1.0000; Context Recall: 1.0000.
- **Output thực tế:** hoàn trả 100%, tức 25 triệu VNĐ.
- **Bằng chứng context:** hoan_chi_dao_tao.md yêu cầu làm việc ít nhất 1 năm sau khóa học; nghỉ trước hạn phải hoàn trả 100%. Giá trị 25 triệu và thời gian 8 tháng đến từ câu hỏi.
- **Error Tree:** Output đúng số tiền? Có → Context có quy tắc? Có → Query rõ? Có → diễn giải thiếu phép nối 8 tháng < 12 tháng; cần kiểm tra judge đối với dữ kiện nằm trong câu hỏi.
- **Diagnosis:** đáp án đúng khi đối chiếu thủ công. Phần 25 triệu không nằm trực tiếp trong policy mà là dữ kiện đầu vào; đây là một lý do có thể làm Faithfulness judge đánh giá thấp, cần trace để xác nhận.
- **Suggested fix:** trả lời thêm một câu: 8 tháng chưa đủ cam kết 12 tháng, nên hoàn 100% × 25 triệu = 25 triệu. Kiểm tra riêng số tiền, điều kiện và nguồn của dữ kiện thay vì chỉ dựa vào một metric tổng hợp.

### 5. Mật khẩu: trả đúng nhưng context còn chính sách superseded

- **Điểm:** Faithfulness: 1.0000; Answer Relevancy: 0.5025; Context Precision: 0.5000; Context Recall: 1.0000.
- **Output thực tế:** tối thiểu 12 ký tự, đúng chính sách hiện hành.
- **Bằng chứng context:** cả mat_khau_v1.md (8 ký tự, đã thay thế) và mat_khau_v2.md (12 ký tự, hiện hành) đều được trả về; Context Precision chỉ 0.5.
- **Error Tree:** Output đúng? Có → Context sạch, chỉ có quy định áp dụng? Không → Query yêu cầu lịch sử phiên bản? Không → lỗi precision/version filtering.
- **Diagnosis:** prompt giúp LLM chọn đúng phiên bản, nhưng retrieval vẫn đưa văn bản cũ vào context. Không nên phụ thuộc hoàn toàn vào LLM giải quyết xung đột phiên bản.
- **Suggested fix:** parse version, effective_date và superseded từ header; mặc định lọc tài liệu hết hiệu lực và chỉ giữ phiên bản cũ khi người dùng hỏi lịch sử/so sánh. Đo lại Precision và các câu hỏi version trên bộ mới.

## Latency và giới hạn phép đo

| Bước trên 20 truy vấn | Trung bình (ms) | Nhỏ nhất (ms) | Lớn nhất (ms) |
|---|---:|---:|---:|
| retrieval | 1369.96 | 80.04 | 3707.29 |
| reranking | 2739.77 | 1739.29 | 5739.20 |
| generation | 1958.30 | 1084.84 | 2551.36 |
| total_query | 6068.14 | 4619.25 | 9662.38 |

Warm reranker đo riêng 5 lần với 20 ứng viên: trung bình 2207.97 ms, không gồm load model. Trung bình query có cold-start của lần đầu; độ trễ API và phần cứng ảnh hưởng số đo. Setup enrichment 369.30 giây là lần gọi API, không phải thời gian dùng cache. RAGAS evaluation nằm ngoài latency mỗi query.

Case study ưu tiên: lỗi tạm ứng cho thấy retrieval đạt Precision/Recall 1.0 vẫn có thể trả số tiền sai. Nếu có thêm một giờ, bổ sung calculator và kiểm tra số học, sau đó đa dạng parent cho truy vấn đa bước. Answer Relevancy thấp trên cả baseline và production; cần đối chiếu thủ công và kiểm tra evaluator đa ngôn ngữ trước khi quy toàn bộ vấn đề cho retrieval.
