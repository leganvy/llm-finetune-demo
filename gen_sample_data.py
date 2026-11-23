"""Sinh dataset mẫu: tin nhắn CSKH tiếng Việt -> JSON intent + slot.

Dùng làm dữ liệu demo cho job-runner fine-tune (FTaaS). Có biến thể nhiễu
(sai dấu, teencode, viết tắt) để minh hoạ khả năng robust của model sau khi tune.
Xuất: data/dataset.jsonl  (mỗi dòng: {"input": <tin nhắn>, "output": <json gọn>})
"""
import json, random, os

random.seed(7)
OUT = os.path.join(os.path.dirname(__file__), "data", "dataset.jsonl")

# (intent, template tin nhắn, builder slot). {x} được thay bằng giá trị ngẫu nhiên.
SP = ["áo thun", "quần jean", "giày sneaker", "balo", "nón lưỡi trai",
      "bình giữ nhiệt", "tai nghe", "ốp lưng", "sạc dự phòng", "đồng hồ"]

def rid():  # mã đơn ngẫu nhiên
    return str(random.randint(1000, 99999))

TEMPLATES = [
    ("hoi_tinh_trang_don", lambda: f"cho mình hỏi đơn #{(m:=rid())} bao giờ giao vậy shop", lambda m: {"ma_don": m}),
    ("hoi_tinh_trang_don", lambda: f"đơn {(m:=rid())} của em tới đâu rồi ạ", lambda m: {"ma_don": m}),
    ("hoi_tinh_trang_don", lambda: f"ship đơn so {(m:=rid())} toi chua shop oi", lambda m: {"ma_don": m}),  # noisy
    ("hoi_gia", lambda: f"{(s:=random.choice(SP))} giá bao nhiêu vậy ad", lambda s: {"san_pham": s}),
    ("hoi_gia", lambda: f"cho hỏi giá cái {(s:=random.choice(SP))} với", lambda s: {"san_pham": s}),
    ("hoi_ton_kho", lambda: f"shop còn {(s:=random.choice(SP))} không ạ", lambda s: {"san_pham": s}),
    ("hoi_ton_kho", lambda: f"{(s:=random.choice(SP))} con hang k shop", lambda s: {"san_pham": s}),  # noisy
    ("doi_tra", lambda: f"mình muốn đổi trả đơn {(m:=rid())} được không", lambda m: {"ma_don": m}),
    ("doi_tra", lambda: f"đơn #{(m:=rid())} bị lỗi cho em đổi hàng nha", lambda m: {"ma_don": m}),
    ("khieu_nai", lambda: "hàng giao chậm quá mình không hài lòng chút nào", lambda _: {}),
    ("khieu_nai", lambda: "shop giao sai mẫu rồi, bực mình ghê", lambda _: {}),
    ("hoi_van_chuyen", lambda: "ship về Đà Nẵng mất mấy ngày vậy shop", lambda _: {}),
    ("hoi_van_chuyen", lambda: "phí ship bao nhiêu tiền ạ", lambda _: {}),
    ("cam_on", lambda: "nhận được hàng rồi nha, cảm ơn shop nhiều", lambda _: {}),
    ("cam_on", lambda: "ok shop, thanks nhé", lambda _: {}),
]

def make(n=170):
    rows = []
    for _ in range(n):
        intent, msg_fn, slot_fn = random.choice(TEMPLATES)
        msg = msg_fn()
        # trích giá trị slot đã nhúng trong msg (số đơn hoặc sản phẩm)
        import re
        m = re.search(r"\d{4,5}", msg)
        s = next((x for x in SP if x in msg), None)
        key = m.group(0) if m else (s if s else None)
        slot = slot_fn(key) if key is not None else slot_fn(None)
        out = {"intent": intent}
        out.update(slot)
        rows.append({"input": msg, "output": json.dumps(out, ensure_ascii=False, sort_keys=True)})
    return rows

if __name__ == "__main__":
    rows = make()
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows -> {OUT}")
    for r in rows[:4]:
        print("  ", r["input"], "->", r["output"])
