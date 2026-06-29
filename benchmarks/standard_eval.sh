#!/bin/bash
# Standard quality suite on B (.2:8888): GSM8K, MATH (minerva_math), HumanEval(+).
# Run AFTER the long-context sweep (single stack -> no concurrent contention).
set -uo pipefail
VENV=/home/keyspark/spark-cluster/benchmarks/eval-venv
PY=$VENV/bin/python
BASE=http://10.100.10.2:8888/v1
CHAT=$BASE/chat/completions
MODEL=deepseek-v4-flash-dspark
OUT=/home/keyspark/dspark-60/standard_eval_out
mkdir -p $OUT
export OPENAI_API_KEY=dummy HF_ALLOW_CODE_EVAL=1 TOKENIZERS_PARALLELISM=false

GSM_LIMIT=${GSM_LIMIT:-500}
MATH_LIMIT=${MATH_LIMIT:-500}

echo "########## GSM8K (limit $GSM_LIMIT) ##########"
$PY -m lm_eval --model local-chat-completions \
  --model_args "model=$MODEL,base_url=$CHAT,num_concurrent=8,max_retries=3,tokenized_requests=False" \
  --tasks gsm8k --limit $GSM_LIMIT --apply_chat_template --batch_size 8 \
  --output_path $OUT/gsm8k 2>&1 | tail -20

echo "########## MATH / minerva_math (limit $MATH_LIMIT) ##########"
$PY -m lm_eval --model local-chat-completions \
  --model_args "model=$MODEL,base_url=$CHAT,num_concurrent=8,max_retries=3,tokenized_requests=False" \
  --tasks minerva_math --limit $MATH_LIMIT --apply_chat_template --batch_size 8 \
  --output_path $OUT/math 2>&1 | tail -25

echo "########## HumanEval(+) via evalplus (164, greedy) ##########"
$PY -m evalplus.codegen "$MODEL" humaneval --greedy --backend openai \
  --base_url $BASE --root $OUT/he 2>&1 | tail -8
SAMPLES=$(ls -t $OUT/he/humaneval/*.jsonl 2>/dev/null | head -1)
echo "samples: $SAMPLES"
$PY -m evalplus.evaluate --dataset humaneval --samples "$SAMPLES" 2>&1 | tail -15
echo "STANDARD_EVAL_DONE"
