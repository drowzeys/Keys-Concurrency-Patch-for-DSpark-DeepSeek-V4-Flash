# Keys-Concurrency Patch for DSpark + DeepSeek-V4-Flash/Pro

**Correct in-server continuous-batch concurrency for DSpark speculative decoding on NVIDIA DGX Spark (GB10 / SM121).**

Stock DSpark serving forces `--max-num-seqs 1` — the upstream recipe states the DSpark draft path *"stalls under `max_num_seqs>1` in this build, so requests serialize. The win is depth (context), not width (concurrency)."*

This patch removes that limitation. With it, DSpark serves **multiple concurrent streams correctly under independent (staggered) request arrivals**, while keeping single-stream output **byte-identical** to the unpatched engine.

> Validated on **DeepSeek-V4-Flash-DSpark**, vLLM DSpark overlay (Rafael Caricio integration / TonyD2Wild packaging), TP=2 on 2× DGX Spark (GB10), 2026-06-29.
> The patch is in the model-agnostic DSpark proposer/model code, so it is expected to apply to **DeepSeek-V4-Pro-DSpark** as well — but Pro is **untested**; please report results.

---

## Results

Headline below; **full single + dual-stack tables in [`RESULTS.md`](RESULTS.md).**
One stack = 2× DGX Spark (GB10), TP=2, `kv-cache-dtype fp8`, DSpark `γ=5`,
`gpu-memory-utilization 0.80`, `max-num-seqs 16`.

### Concurrency unlocked (this patch) — one 2-Spark stack

| total concurrency | static (best-case) | staggered (real arrivals) | acceptance | errors |
|---:|---:|---:|---:|---:|
| 1  | 49 tok/s  | ~50 tok/s | ~0.6 | 0 |
| 4  | 122 tok/s | 104 tok/s | ~0.59 | 0 |
| 8  | 183 tok/s | 139 tok/s | ~0.56 | 0 |
| 16 | **290 tok/s** | **191 tok/s** | ~0.55 | 0 |

- **Single-stream unchanged**: ~50–54 tok/s, provably byte-identical output.
- **Zero errors** at every level; draft acceptance stays healthy (~0.55) — DSpark keeps accelerating under load.

### Scaling out — 2 stacks (4 Sparks) measured

| total concurrency | 1 stack | 2 stacks | scaling |
|---:|---:|---:|---:|
| 32 | — | **375 tok/s** (16+16, 32/32 ok) | **~1.96×** |

Replicas scale ~linearly (independent stacks behind a least-connections router).

### Correctness

A deterministic request (temperature 0, `ignore_eos`) produces **byte-identical** output whether run alone or while other requests start/finish around it (forcing continuous-batch *condense*). Run `benchmarks/correctness_test.py`.

---

## What the patch does

Two root causes blocked `max_num_seqs>1`:

**Patch 1 — request-stable KV slot.**
DSpark's only persistent per-request draft state (`DeepSeekV4DSparkAttention.main_kv_cache`) was indexed by **batch-row position**. Under vLLM-v1 continuous batching the running set is *condensed* when a request finishes (another request moves into its row), so the persistent sliding-window KV silently belonged to the **wrong request** → corrupted drafts → acceptance collapse. The patch keys `main_kv_cache` by a **stable per-request slot** (req-id → slot map, threaded from the runner), so reads/writes always hit the right request.

**Patch 2 — ragged context path.**
`prepare_context` / `prefill_main` / `store_main_kv` reshaped the flat batch into a rectangular `[B, seq, H]`, asserting every request contributed the same number of rows. With chunked prefill (required for long context) a step mixes prefill+decode → **non-uniform** rows → `ValueError: DSpark currently requires uniform flattened per-request inputs`. The patch handles requests **raggedly** via `query_start_loc` (per-request segment offsets), the same mechanism the rejection-trim path already used.

Both patches keep the original rectangular fast-path for uniform/static/single-stream batches, so those cases are **byte-identical**. Ragged/mixed steps run eager (never cudagraph-captured), so dynamic shapes are safe; the uniform decode-only graphed path is unchanged.

Files touched (overlay):
- `vllm/v1/spec_decode/dspark_proposer.py` (+158 / −10)
- `vllm/models/deepseek_v4/nvidia/dspark.py` (+110 / −12)
- `vllm/v1/worker/gpu_model_runner.py` (+10 / −0)

---

## Apply

This is a patch **against the DSpark vLLM overlay** (Rafael Caricio integration, as vendored by the TonyD2Wild packaged recipe — see Credits). It does not include the overlay itself.

```bash
# from the root of your DSpark overlay checkout (the dir containing vllm/)
git apply -p1 patches/keys-concurrency.patch
# then rebuild the runtime image on each node and restart, e.g.:
#   ./build-dspark-vllm-runtime.sh && ./start-deepseek-v4-flash-dspark.sh
```

Requirement: run with **`VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1`** (the ragged path is implemented for this mode), then set `--max-num-seqs` to your desired concurrency.

---

## Benchmark / verify

```bash
python3 benchmarks/correctness_test.py   http://<head>:<port>          # byte-identical under churn
python3 benchmarks/staggered_bench.py     http://<head>:<port> 16 0.4  # real staggered arrivals
python3 benchmarks/bench_concurrent.py    http://<head>:<port> 1,2,4,8,16  # static sweep
```

---

## Status & caveats (honest)

- Validated for **correctness** (byte-identical under churn), **stability** (0 errors at N≤16), **acceptance** (~0.55 under load), and **task quality** (GSM8K N=8 vs single-stream: quality-neutral, 97.5% per-question agreement — see `RESULTS.md`). Single-stream is a no-op.
- A multi-hour soak is still recommended before production.
- Only the `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` code path was made ragged; the legacy `_trim_rejected_target_context` path still assumes uniform.
- DeepSeek-**V4-Pro**-DSpark: expected to work (shared DSpark code) but **untested**.

## Credits

Built on public work — this repo is **only the concurrency patch** on top of it:
- DSpark vLLM integration — [rafaelcaricio/vllm#1](https://github.com/rafaelcaricio/vllm/pull/1), [rafaelcaricio/spark_vllm_docker#1](https://github.com/rafaelcaricio/spark_vllm_docker/pull/1)
- Two-node DGX Spark packaging / worker-first launch — MiaAI-Lab, and [tonyd2wild/DeepSeek-v4-Flash-DSpark-60-tok-s-900K-ctx-2x-DGX-Spark](https://github.com/tonyd2wild/DeepSeek-v4-Flash-DSpark-60-tok-s-900K-ctx-2x-DGX-Spark)
- [vLLM](https://github.com/vllm-project/vllm) (Apache-2.0)

Licensed under Apache-2.0 (see `LICENSE`, `NOTICE`). The patched lines are derivatives of Apache-2.0 vLLM/DSpark sources.
