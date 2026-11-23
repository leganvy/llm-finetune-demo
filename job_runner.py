"""FTaaS job-runner: fine-tune 1 model theo dataset của khách rồi xuất adapter + báo cáo eval.

Luồng: load base -> ĐO TRƯỚC (sinh trên tập eval, exact-match) -> LoRA SFT
       -> ĐO SAU -> lưu adapter + eval_report.md (trước/sau + ví dụ cạnh nhau).

Dataset: jsonl, mỗi dòng {"input": <prompt người dùng>, "output": <đáp án đích>}.
Đầu ra đích có thể là JSON (so khớp theo cấu trúc) hoặc văn bản (so khớp chuẩn hoá).

Ví dụ:
  CUDA_VISIBLE_DEVICES=1 python3 job_runner.py \
      --base /home/dungha/dungha_models/Qwen3-4B-Instruct-2507 \
      --data data/dataset.jsonl --out runs/demo --epochs 3
"""
import argparse, json, os, time, random

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base", required=True, help="đường dẫn base model (HF)")
    p.add_argument("--data", required=True, help="dataset jsonl {input,output}")
    p.add_argument("--out", required=True, help="thư mục xuất (adapter + report)")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--eval_n", type=int, default=20, help="số mẫu giữ lại để đo")
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--max_steps", type=int, default=-1, help=">0 để smoke-test nhanh")
    p.add_argument("--sys", default="Bạn là trợ lý phân loại tin nhắn CSKH. Trả về JSON gọn.",
                   help="system prompt")
    return p.parse_args()

def norm(s):
    """Chuẩn hoá để so khớp: thử parse JSON (so theo cấu trúc), nếu không thì so chuỗi."""
    s = (s or "").strip()
    try:
        return ("json", json.dumps(json.loads(s), sort_keys=True, ensure_ascii=False))
    except Exception:
        return ("text", " ".join(s.lower().split()))

def build_prompt(tok, sys_msg, user):
    msgs = [{"role": "system", "content": sys_msg}, {"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def generate(model, tok, prompt, max_new=96):
    import torch
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
    gen = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(gen, skip_special_tokens=True).strip()

def eval_set(model, tok, sys_msg, rows):
    """Trả về (accuracy, danh sách (input, gold, pred, hit))."""
    details, hit = [], 0
    for r in rows:
        pred = generate(model, tok, build_prompt(tok, sys_msg, r["input"]))
        ok = norm(pred) == norm(r["output"])
        hit += int(ok)
        details.append((r["input"], r["output"], pred, ok))
    return hit / max(len(rows), 1), details

def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTConfig, SFTTrainer

    rows = [json.loads(l) for l in open(a.data) if l.strip()]
    random.seed(42); random.shuffle(rows)
    ev = rows[:a.eval_n]; tr = rows[a.eval_n:]
    print(f"[data] train={len(tr)} eval={len(ev)}")

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("[load] base model ...")
    model = AutoModelForCausalLM.from_pretrained(a.base, torch_dtype=torch.bfloat16).to("cuda")
    model.config.use_cache = True

    print("[eval] ĐO TRƯỚC khi fine-tune ...")
    t0 = time.time()
    acc_before, det_before = eval_set(model, tok, a.sys, ev)
    print(f"[eval] before exact-match = {acc_before:.1%}  ({time.time()-t0:.0f}s)")

    # Dataset prompt/completion -> completion_only_loss chỉ tính loss trên đáp án
    ds = Dataset.from_list([{"prompt": build_prompt(tok, a.sys, r["input"]),
                             "completion": r["output"]} for r in tr])

    lora = LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      task_type="CAUSAL_LM")
    cfg = SFTConfig(output_dir=os.path.join(a.out, "trainer"), num_train_epochs=a.epochs,
                    per_device_train_batch_size=a.bs, gradient_accumulation_steps=a.grad_accum,
                    learning_rate=a.lr, warmup_steps=5, logging_steps=5, bf16=True,
                    max_length=a.max_len, completion_only_loss=True, packing=False,
                    save_strategy="no", report_to="none",
                    max_steps=a.max_steps, gradient_checkpointing=False)
    model.config.use_cache = False
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds,
                         peft_config=lora, processing_class=tok)
    print("[train] LoRA SFT ...")
    t0 = time.time(); trainer.train(); train_s = time.time() - t0
    print(f"[train] done in {train_s:.0f}s")

    tuned = trainer.model
    tuned.config.use_cache = True
    print("[eval] ĐO SAU khi fine-tune ...")
    acc_after, det_after = eval_set(tuned, tok, a.sys, ev)
    print(f"[eval] after exact-match = {acc_after:.1%}")

    adapter_dir = os.path.join(a.out, "adapter")
    tuned.save_pretrained(adapter_dir); tok.save_pretrained(adapter_dir)
    print(f"[save] adapter -> {adapter_dir}")

    write_report(a, len(tr), len(ev), acc_before, acc_after, det_after, train_s)
    print(f"[done] báo cáo -> {os.path.join(a.out,'eval_report.md')}")
    print(f"[KẾT QUẢ] exact-match {acc_before:.0%} -> {acc_after:.0%}")

def write_report(a, ntr, nev, before, after, details, train_s):
    import datetime
    lines = []
    lines.append(f"# Báo cáo Fine-tune — {datetime.date.today()}\n")
    lines.append("## Cấu hình\n")
    lines.append(f"- Base model: `{os.path.basename(a.base)}`")
    lines.append(f"- Phương pháp: LoRA (r={a.lora_r}, alpha={a.lora_alpha}), completion-only loss")
    lines.append(f"- Dữ liệu: {ntr} train / {nev} eval (giữ lại) · epochs={a.epochs} · lr={a.lr}")
    lines.append(f"- Thời gian train: {train_s:.0f}s trên 1×H100\n")
    lines.append("## Kết quả (exact-match trên tập eval giữ lại)\n")
    lines.append("| | Exact-match |")
    lines.append("|---|---|")
    lines.append(f"| **Base (chưa tune)** | {before:.1%} |")
    lines.append(f"| **Sau fine-tune** | **{after:.1%}** |")
    delta = after - before
    lines.append(f"\n**Cải thiện: {delta:+.1%}**\n")
    lines.append("## Ví dụ (sau fine-tune)\n")
    for i, (inp, gold, pred, ok) in enumerate(details[:6], 1):
        mark = "✅" if ok else "❌"
        lines.append(f"**{i}.** {mark} Input: `{inp}`")
        lines.append(f"- Gold: `{gold}`")
        lines.append(f"- Pred: `{pred}`\n")
    open(os.path.join(a.out, "eval_report.md"), "w").write("\n".join(lines))

if __name__ == "__main__":
    main()
