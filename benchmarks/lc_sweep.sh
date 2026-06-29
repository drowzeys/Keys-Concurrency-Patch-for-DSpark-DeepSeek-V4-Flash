#!/bin/bash
# Long-context sweep on B: prompt 10k->256k, 5 iters each. Prefill TTFT + decode tok/s.
cd /home/keyspark/dspark-60/docker-stack
export MODEL=deepseek-v4-flash-dspark MODEL_DIR=/home/keyspark/models/dsv4-flash-dspark
export BASE_URL=http://10.100.10.2:8888 SCENARIO=context_confirm MAX_TOKENS=64 THINKING=false
for N in 10000 32000 64000 128000 192000 256000; do
  echo "===== prompt_tokens=$N (5 iters) ====="
  PROMPT_TOKENS=$N OUT_DIR=experiments/lc-$N bash scripts/run_dspark_benchmark_repeats.sh lc_$N 5 \
    > /tmp/lc_$N.log 2>&1
  echo "  done $N"
done
echo "LC_SWEEP_DONE"
