# Case Study: Xây dựng LLM Data-Agent tiếng Việt dưới ràng buộc production khắc nghiệt

*Cách tôi fine-tune một bộ model gọn nhẹ để đọc và trả lời câu hỏi trên dữ liệu kinh doanh
thực tế, lộn xộn — chạy hoàn toàn offline, tất định, và trong giới hạn tham số nghiêm ngặt.*

> **Tóm tắt** — Tôi xây một agent đầu-cuối trả lời câu hỏi ngôn ngữ tự nhiên **tiếng Việt**
> trên dữ liệu cửa hàng nhỏ thực tế (Excel / CSV / SQLite). Hệ thống phải gói trong
> **≤15B tham số**, chạy **hoàn toàn offline** trên **2 GPU**, và **tái lập được**: hai lần
> chạy cùng đầu vào lệch nhau ≤0.03 điểm. Kết quả: một hệ thống bền, đạt chuẩn production —
> không phải demo notebook. Đây là mức độ kỹ lưỡng tôi đưa vào mọi dự án fine-tuning.

---

## Bài toán

Một cuộc thi AI cấp quốc gia đặt ra một vấn đề khó một cách đánh lừa: cho một bộ file dữ liệu
tùy ý của cửa hàng nhỏ (bảng tính, CSV, cơ sở dữ liệu SQLite) và một câu hỏi tiếng Việt, hãy
trả về đáp án đúng — hoặc **bỏ trống** khi không chắc (trả lời sai bị trừ điểm).

Chính các ràng buộc mới biến nó thành một bài toán *kỹ thuật*, không phải bài tập prompt:

- **≤15B tham số tổng cộng** trên mọi model, adapter, router và classifier.
- **Hoàn toàn offline** trên máy chấm (2 GPU) — không internet, không API ngoài.
- **Tái lập**: hai lần chạy cùng đầu vào phải lệch ≤0.03 điểm.
- **Không train trên dữ liệu test** — khả năng tổng quát hóa phải là thật.
- **Đầu vào lộn xộn**: tên cột không nhất quán, mã hóa lẫn lộn, lỗi gõ, chữ mất dấu,
  và tiếng Việt "teencode".

## Kiến trúc

Thay vì ném một model khổng lồ vào bài toán, tôi chia việc thành một pipeline có kỷ luật:

```
file thô → nén + gợi ý vai trò cột/dòng (ML cổ điển, không dùng embedding)
         → model 9B viết bản đặc tả có cấu trúc của dữ liệu → kiểm tra & chọn bản tốt nhất
         → thực thi đặc tả → bảng chuẩn → data card
         → mỗi câu: 9B sinh intent (JSON) + 4B sinh code → thực thi cả hai
         → bỏ phiếu theo độ tin cậy + ngưỡng → trả lời hoặc bỏ trống
```

Hai model fine-tuned gánh phần chính: model **9B** sinh schema/intent và model **4B** tổng hợp
code — cộng lại vẫn dưới ngân sách 15B.

## Điểm nhấn kỹ thuật

- **Pipeline kiểm chất lượng nhãn 3 cổng.** Mọi nhãn train (đặc tả, intent, code) phải qua:
  (1) hợp schema + chạy trong giới hạn thời gian, (2) khớp đáp án vàng theo luật chuẩn hóa
  chính thức, (3) **vẫn đúng trên 5 biến thể làm bẩn của dữ liệu**. Hàng nghìn nhãn, mỗi
  nhãn kiểm ba lần — đây là lý do model tổng quát hóa thay vì học thuộc.
- **Bộ model fine-tuned gọn.** LoRA trên cặp model 9B + 4B, loại module ảnh/MTP khỏi phần
  thích nghi, gộp lại để chạm đúng ngân sách tham số.
- **Bền vững theo thiết kế.** Hệ thống xử lý được chữ mất dấu, lỗi gõ Telex, teencode, các
  chuỗi trạng thái lạ, dòng tổng/footer, và nguồn dữ liệu nhiều file — kiểm chứng bằng fuzzing
  tự động.
- **Tất định khi phục vụ.** Để hai lần chạy khớp nhau trong 0.03 trên các kernel không
  bất-biến-theo-batch đòi hỏi tuần tự hóa sinh ở khâu đặc tả và kiểm soát việc lấy mẫu —
  đây là khác biệt giữa một kết quả nghiên cứu và một thứ thực sự ship được.
- **Bỏ trống có hiệu chỉnh.** Độ tin cậy = mức đồng thuận giữa các lá phiếu độc lập (intent
  của model đối chiếu code sinh độc lập), với ngưỡng tăng lên khi dữ liệu trông không đáng
  tin — để hệ thống im lặng thay vì đoán sai.

## Kết quả

Một lời giải tất định, hoàn toàn offline, dưới 15B, đọc và trả lời trên những bộ dữ liệu cửa
hàng nhỏ **tùy ý** mà nó chưa từng thấy — với chất lượng đo được, tái lập được, và cơ chế bỏ
trống có nguyên tắc. Mọi quyết định thiết kế đều được kiểm chứng bằng đánh giá giữ lại
(held-out), không phải cảm tính.

## Từ cuộc thi đến doanh nghiệp của bạn

Chính pipeline này là thứ một công ty cần khi model đa dụng (GPT/Claude) hoạt động kém trên
lĩnh vực *của họ*:

- Bạn có dữ liệu riêng và muốn một model nói đúng định dạng, giọng điệu, hoặc schema của bạn.
- Bạn cần nó chạy **on-prem / offline** vì lý do bảo mật hoặc chi phí.
- Bạn cần chất lượng **đo được** và hành vi an toàn, không phải một prompt cầu may.

Tôi biến điều đó thành một model chạy được, đã đánh giá — chuẩn bị dữ liệu → fine-tune →
đánh giá → triển khai — và tôi làm chủ phần cứng (2×H100), nên dữ liệu của bạn không rời khỏi
máy do tôi kiểm soát.

**Bằng chứng đây là pipeline thật, không phải slide:** một phiên bản mini chạy được của luồng
này (load → đo trước → LoRA fine-tune → đo sau → giao adapter + báo cáo) có sẵn dưới dạng demo
trực tiếp — nó đưa một model 4B từ **0% lên 100%** exact-match trên một tác vụ sinh JSON tiếng
Việt trong **11 giây** trên một H100.

## Công nghệ sử dụng

PyTorch · Transformers · TRL (SFT, preference tuning) · PEFT (LoRA/QLoRA) · DeepSpeed ZeRO ·
vLLM · ML cổ điển (TF-IDF/LinearSVC/LightGBM) cho gợi ý đặc trưng · NVIDIA H100.

---

### Làm việc cùng tôi

Tôi fine-tune các LLM mở (Llama, Qwen, Mistral, Gemma) trên dữ liệu của bạn và giao model
đạt chuẩn production, đã đánh giá. Thế mạnh: **NLP tiếng Việt & đa ngôn ngữ, sinh có cấu trúc,
độ bền, triển khai on-prem.**

- 📫 [email / LinkedIn / Upwork / Fiverr của bạn]
- 🔗 Demo pipeline trực tiếp: [link repo demo / bài Viblo]

*(Bản tiếng Anh của bài viết này có sẵn cho LinkedIn / cộng đồng quốc tế.)*
