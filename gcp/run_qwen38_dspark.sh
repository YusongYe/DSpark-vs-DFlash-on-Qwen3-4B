#!/usr/bin/env bash
# Start SGLang with RadixArk Qwen3.8-27B-DSpark, then measure accept length.
# Run on the A100 (a2-highgpu-1g), inside tmux. Mac 24GB cannot do this.
set -euo pipefail

PORT="${PORT:-30000}"
MEM="${MEM_FRACTION:-0.80}"
OUT="${OUT:-accept_qwen38_dspark.json}"
EXTRA_MEASURE="${EXTRA_MEASURE:-}"   # e.g. --official

if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi missing; fix GPU drivers before measuring"
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv

if ! python -c "import sglang, transformers, requests" 2>/dev/null; then
  echo "installing sglang + transformers + requests …"
  pip install -U "sglang[all]" transformers requests
fi

echo "starting SGLang on :$PORT  (first launch downloads ~18 GB)"
sglang serve \
  --trust-remote-code \
  --model-path Qwen/Qwen3.8-27B-FP8 \
  --tp-size 1 \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path RadixArk/Qwen3.8-27B-DSpark \
  --speculative-dspark-block-size 7 \
  --speculative-draft-model-quantization unquant \
  --mamba-scheduler-strategy extra_buffer \
  --attention-backend fa3 \
  --mem-fraction-static "$MEM" \
  --host 0.0.0.0 \
  --port "$PORT" &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

HERE="$(cd "$(dirname "$0")" && pwd)"
python "$HERE/measure_qwen38_dspark.py" \
  --url "http://127.0.0.1:$PORT" \
  --out "$OUT" \
  $EXTRA_MEASURE
