# Aggregate results & summary

All measurements: one TP=2 replica (2× DGX Spark GB10), DeepSeek-V4-Flash-DSpark,
`kv-cache-dtype fp8`, DSpark `γ=5`, `gpu-memory-utilization 0.80`,
`max-model-len 262144`. Throughput is **server-side** (`/metrics`
`vllm:generation_tokens_total` delta); acceptance is
`spec_decode_num_accepted_tokens / num_draft_tokens`. Benchmark scripts in
`benchmarks/`.

## 1. Single-stream — A vs B (official harness, code_completion 512→256)

| replica | recipe | decode tok/s | acceptance | accepted/draft |
|---|---|---:|---:|---:|
| A | Rafael Caricio stack | 54.0 | 0.640 | 3.20 |
| B | TonyD2Wild packaged | 52.1 | 0.617 | 3.09 |

Same engine (B vendors A's overlay) → difference is run-to-run noise. Reference
upstream number is ~62; the gap is acceptance/content variance (GPU clocks and
warmup were ruled out — `0x0` throttle, 49–53 °C).

## 2. Concurrency — STATIC batch (all requests simultaneous; best-case overlap)

| concurrency | server aggregate tok/s | per-stream tok/s | acceptance |
|---:|---:|---:|---:|
| 1  | 52.1  | 52.1 | 0.59–0.64 |
| 2  | 82.6  | 41.3 | 0.60 |
| 4  | 123.9 | 31.0 | 0.58–0.63 |
| 8  | 212.3 | 26.5 | 0.58–0.62 |
| 16 | 301.2 | 18.8 | 0.59–0.61 |

## 3. Concurrency — STAGGERED arrivals (real, independent; the ragged path Patch 2 fixes)

| concurrency | success | server aggregate tok/s | acceptance |
|---:|---:|---:|---:|
| 4  | 4/4   | 94.6  | 0.551 |
| 8  | 8/8   | 129.5 | 0.541 |
| 16 | 16/16 | 190.2 | 0.568 |

Static is the upper bound (perfect overlap); staggered is the realistic floor for
that load shape. Production workloads land between them. **Zero errors at every
level; acceptance stays healthy (~0.55), i.e. DSpark keeps accelerating under
concurrency.**

## 4. Correctness under continuous-batch condense

| check | result |
|---|---|
| Deterministic victim output, alone vs under churn | **byte-identical** |
| Requests succeeding while others start/finish | 16/16, 0 errors |
| Single-stream output vs unpatched engine | byte-identical (no-op) |

## 5. Two replicas (4 nodes)

Two independent patched TP=2 replicas behind a least-connections router ≈ **double**
the aggregate and concurrency: order of **~380 tok/s @ 32 concurrent** staggered
(extrapolated from the single-replica curve; each request runs entirely on one
replica).

---

## Summary

| before this patch | after (Patch 1 + Patch 2) |
|---|---|
| DSpark locked to `max-num-seqs=1` (single stream) | correct concurrency at `max-num-seqs>1` |
| `max-num-seqs>1` → silent acceptance collapse (Patch 1) or HTTP 500 (Patch 2) | 0 errors, acceptance ~0.55, byte-identical correctness |
| 1 stream @ ~52 tok/s | up to **301 tok/s @16 static / 190 @16 staggered** per replica; single-stream unchanged |

The patch converts DSpark from a depth-only (long-context, single-stream) engine
into one that also scales in **width** (concurrency) — without regressing
single-stream speed or output.

### Caveats
- Certified for correctness, stability (N≤16), and acceptance. A task-quality eval
  at concurrency (GSM8K/HumanEval N=8 vs single-stream) and a multi-hour soak are
  recommended before production.
- Requires `VLLM_DSPARK_GPU_REJECTED_CONTEXT_MASK=1` (the patched ragged path).
- Validated on V4-Flash-DSpark; V4-Pro-DSpark expected to work (shared code) but
  untested.
