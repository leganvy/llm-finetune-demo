# Báo cáo Fine-tune — 2026-11-23

## Cấu hình

- Base model: `Qwen3-4B-Instruct-2507`
- Phương pháp: LoRA (r=16, alpha=32), completion-only loss
- Dữ liệu: 150 train / 20 eval (giữ lại) · epochs=3.0 · lr=0.0002
- Thời gian train: 10s trên 1×H100

## Kết quả (exact-match trên tập eval giữ lại)

| | Exact-match |
|---|---|
| **Base (chưa tune)** | 0.0% |
| **Sau fine-tune** | **100.0%** |

**Cải thiện: +100.0%**

## Ví dụ (sau fine-tune)

**1.** ✅ Input: `áo thun con hang k shop`
- Gold: `{"intent": "hoi_ton_kho", "san_pham": "áo thun"}`
- Pred: `{"intent": "hoi_ton_kho", "san_pham": "áo thun"}`

**2.** ✅ Input: `ok shop, thanks nhé`
- Gold: `{"intent": "cam_on"}`
- Pred: `{"intent": "cam_on"}`

**3.** ✅ Input: `đơn #37493 bị lỗi cho em đổi hàng nha`
- Gold: `{"intent": "doi_tra", "ma_don": "37493"}`
- Pred: `{"intent": "doi_tra", "ma_don": "37493"}`

**4.** ✅ Input: `hàng giao chậm quá mình không hài lòng chút nào`
- Gold: `{"intent": "khieu_nai"}`
- Pred: `{"intent": "khieu_nai"}`

**5.** ✅ Input: `shop giao sai mẫu rồi, bực mình ghê`
- Gold: `{"intent": "khieu_nai"}`
- Pred: `{"intent": "khieu_nai"}`

**6.** ✅ Input: `quần jean giá bao nhiêu vậy ad`
- Gold: `{"intent": "hoi_gia", "san_pham": "quần jean"}`
- Pred: `{"intent": "hoi_gia", "san_pham": "quần jean"}`
