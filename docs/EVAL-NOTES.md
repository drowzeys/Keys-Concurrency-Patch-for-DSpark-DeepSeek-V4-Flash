# Eval notes — reproducing the benchmarks (gotchas & fixes)

The numbers in `RESULTS.md` come from the scripts in `benchmarks/`. Three
non-obvious issues had to be solved to get clean MATH/quality numbers on a
DeepSeek-V4 DSpark server. They're documented here so the results reproduce.

## 1. lm-eval MATH needs extra deps

`lm_eval --tasks minerva_math` fails with:
```
ModuleNotFoundError: `sympy`, `math_verify` and `antlr4-python3-runtime==4.11`
are required ...
```
Fix:
```bash
pip install sympy "antlr4-python3-runtime==4.11" math_verify
# (equivalently: pip install "lm-eval[math]")
```

## 2. DeepSeek reasoning parser → null `content` (the lm-eval MATH trap)

The DSpark server runs with `--reasoning-parser deepseek_v4` and
`--default-chat-template-kwargs '{"thinking":true}'`. With thinking on, the
model's reasoning is routed into the response's **`reasoning_content`** field and
the OpenAI `content` field can come back **null/empty**. lm-eval reads `content`,
so it logs:
```
WARNING ... API returned null content. Content filled with
LMEVAL_MODEL_NONE_ANSWER_PLACEHOLDER. Check reasoning_content field or generation limits.
```
and scores those items as wrong — depressing MATH (and few-shot GSM8K) accuracy.

**Fix used here:** evaluate MATH with a controlled runner
(`benchmarks/math_eval.py`) that sends `chat_template_kwargs={"thinking": false}`
so the answer lands in `content`, with `max_tokens=1024` for the chain-of-thought,
and extracts the final `\boxed{...}`. (Same reason GSM8K is reported from the
0-shot chat runner `benchmarks/gsm8k_eval.py` rather than lm-eval's 5-shot, whose
format mismatch with chat/reasoning models gives ~76% vs the representative ~95%.)

## 3. `math_verify` signal-timeout only works in the main thread (silent fail)

`math_verify.verify()` installs a **signal-based (SIGALRM) timeout**. Python only
allows signal handlers in the **main thread**, so calling `verify()` from a worker
thread raises `ValueError: signal only works in main thread` — which, if swallowed
by a bare `except`, silently marks **every** answer wrong (observed: 0/21 with
obviously-correct predictions).

**Fix used here (`math_eval.py`):** do generation in worker threads, but **score in
the main thread after join**:
```python
# workers only fetch + extract the \boxed{} prediction (no scoring)
res[idx] = {"pred": last_boxed(text), "gold": gold}
...
# after all threads join — score in the MAIN thread:
for i in res:
    p, g = res[i]["pred"], res[i]["gold"]
    res[i]["ok"] = bool(verify(parse("$"+g+"$"), parse("$"+p+"$"))) if p else False
```
Equivalent alternatives: run scoring single-threaded, or wrap `verify` to disable
its signal timeout.

## Scripts

| script | what it runs |
|---|---|
| `benchmarks/bench_concurrent.py` | static concurrency sweep (server-agg tok/s + acceptance) |
| `benchmarks/staggered_bench.py` | staggered/real-arrival concurrency (ragged path) |
| `benchmarks/correctness_test.py` | byte-identical-under-churn correctness |
| `benchmarks/gsm8k_eval.py` | GSM8K, 0-shot chat, seq-vs-concurrent agreement |
| `benchmarks/math_eval.py` | MATH (hendrycks), `\boxed{}` + `math_verify` (main-thread scoring) |
| `benchmarks/lc_sweep.sh` | long-context prompt sweep 10k→256k |
| `benchmarks/standard_eval.sh` | GSM8K + minerva_math + HumanEval(+) via lm-eval/evalplus |
