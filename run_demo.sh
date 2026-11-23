#!/usr/bin/env bash
# Chạy demo fine-tune đầu-cuối: sinh dữ liệu mẫu -> LoRA SFT -> báo cáo.
# Tuỳ biến qua biến môi trường: GPU, BASE, DATA, OUT, EPOCHS, EVAL_N
set -e
cd "$(dirname "$0")"
PY="$HOME/envs/llm/bin/python3"

"$PY" gen_sample_data.py

CUDA_VISIBLE_DEVICES="${GPU:-1}" "$PY" job_runner.py \
  --base "${BASE:-/home/dungha/dungha_models/Qwen3-4B-Instruct-2507}" \
  --data "${DATA:-data/dataset.jsonl}" \
  --out  "${OUT:-runs/demo}" \
  --epochs "${EPOCHS:-3}" \
  --eval_n "${EVAL_N:-20}"

echo
echo "Xong. Báo cáo: ${OUT:-runs/demo}/eval_report.md | Adapter: ${OUT:-runs/demo}/adapter"
