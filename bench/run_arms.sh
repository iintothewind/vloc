#!/usr/bin/env bash
# Runs on spark2 in ~/Documents/vlocate. Native 2B/4B bf16 arms, then 4B Q4_K_M via patched llama-server
# with two image-token settings (native-matched 16..2025, and model-card 1024..1960).
set -u
cd ~/Documents/vlocate
mkdir -p arms
venv/bin/python cmp_arms.py native models/WeVisOne-VL-2B arms/2B_bf16.jsonl 2>&1 | grep -v 'Loading weights'
venv/bin/python cmp_arms.py native models/WeVisOne-VL-4B arms/4B_bf16.jsonl 2>&1 | grep -v 'Loading weights'

LS=llama.cpp/build-cuda/bin/llama-server
for cfg in "native 16 2025" "card 1024 1960"; do
  set -- $cfg
  name=$1; lo=$2; hi=$3
  $LS -m gguf/4B/WeVisOne-VL-4B-Q4_K_M.gguf --mmproj gguf/4B/vision-4b.gguf \
      --host 127.0.0.1 --port 18080 -c 8192 -ngl 99 --parallel 1 --special --no-warmup \
      --image-min-tokens $lo --image-max-tokens $hi > arms/llama_$name.log 2>&1 &
  pid=$!
  for i in $(seq 1 120); do curl -sf http://127.0.0.1:18080/health >/dev/null && break; sleep 1; done
  venv/bin/python cmp_arms.py llama http://127.0.0.1:18080 arms/4B_Q4KM_$name.jsonl
  grep -iE 'model buffer size|CUDA0 .*buffer|KV' arms/llama_$name.log | head -8
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep "^$pid" || true
  kill $pid; wait $pid 2>/dev/null
done
venv/bin/python score_arms.py arms/2B_bf16.jsonl arms/4B_bf16.jsonl arms/4B_Q4KM_native.jsonl arms/4B_Q4KM_card.jsonl
