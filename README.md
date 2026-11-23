# LLM Fine-Tuning Service — Demo Pipeline

A self-contained, end-to-end **LoRA fine-tuning job runner** that turns a client's
dataset into a working, *measured* custom model. Built to run on 2×H100 (uses 1 GPU per job).

This repo is the live sample behind my freelance fine-tuning service — it is the exact
flow a paying job goes through: **load base → measure BEFORE → LoRA SFT → measure AFTER → deliver adapter + report.**

## Live demo result

Task: Vietnamese customer-support messages → compact JSON intent + slots (with noisy/typo inputs).
Base model: `Qwen3-4B-Instruct-2507`, LoRA (r=16), 150 train / 20 held-out eval.

| | Exact-match on held-out eval |
|---|---|
| Base model (no tuning) | **0%** |
| After fine-tune (**11 s** on 1×H100) | **100%** |

Full report with side-by-side examples: [`runs/demo/eval_report.md`](runs/demo/eval_report.md).
The base model produced valid-looking but off-schema answers; after tuning it emits the exact
required JSON — including on messy inputs like `áo thun con hang k shop` (missing diacritics/teencode).

## Run it

```bash
bash run_demo.sh
# or customize:
GPU=1 EPOCHS=3 OUT=runs/myjob bash run_demo.sh
```

## Live run output

Captured from an actual run (`bash run_demo.sh`) on one H100:

```text
[data] train=150 eval=20
[load] base model ...
[eval] ĐO TRƯỚC khi fine-tune ...
[eval] before exact-match = 0.0%  (17s)
[train] LoRA SFT ...
{'loss': 2.57,  'mean_token_accuracy': 0.69, 'epoch': 0.5}
{'loss': 0.35,  'mean_token_accuracy': 0.91, 'epoch': 1.0}
{'loss': 0.017, 'mean_token_accuracy': 0.99, 'epoch': 2.0}
{'loss': 0.0008,'mean_token_accuracy': 1.00, 'epoch': 3.0}
[train] done in 10s
[eval] ĐO SAU khi fine-tune ...
[eval] after exact-match = 100.0%
[KẾT QUẢ] exact-match 0% -> 100%
```

## Use your own data

Dataset = JSONL, one example per line:

```json
{"input": "<user message / prompt>", "output": "<target answer (JSON or text)>"}
```

Then:

```bash
CUDA_VISIBLE_DEVICES=1 ~/envs/llm/bin/python3 job_runner.py \
  --base /path/to/base-model \
  --data your_data.jsonl --out runs/your_job --epochs 3 --eval_n 30
```

Output: a LoRA `adapter/` (load on top of the base model) + `eval_report.md`
with before/after metrics. The eval metric auto-detects JSON targets (structural match)
vs text targets (normalized string match).

## Stack

PyTorch 2.13 · Transformers 5.19 · TRL 1.17 (SFT, completion-only loss) · PEFT 0.21 (LoRA) · on NVIDIA H100.

## What a real engagement adds (beyond this demo)

- Data cleaning, formatting, and synthetic augmentation
- Proper eval harness tailored to the task (not just exact-match)
- QLoRA for larger models (up to ~70B on 2×H100), or full SFT for ≤13B
- Deployment: merged weights or a Dockerized vLLM OpenAI-compatible endpoint
- NDA; client data stays on-prem and is deleted after delivery
