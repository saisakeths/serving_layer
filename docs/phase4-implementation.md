# Phase 4 implementation guide

Parent: [plan.md](plan.md) · Requires: [phase3-implementation.md](phase3-implementation.md) gate (including cloud `bench_forward`).

**Goal:** Token generation with **KV cache at model scope**, sampling, and `mini_infer` CLI on **synthetic** weights.

## LlamaCache (not in Phase 3)

| Piece | Description |
|-------|-------------|
| `LlamaCache` | Owns `n_layer` × [`KVCache`](../src/cache/kv_cache.hpp); `reset()`, shared `max_seq` |
| `LlamaModel::forward_prefill` | Embed prompt → each layer `TransformerBlock::forward_prefill` |
| `LlamaModel::decode_step` | Single token embed → per-layer `decode_step` → logits `[1, 1, vocab]` |

Block-level KV is already tested in `test_kv_cache.cpp`; model-level prefill-vs-decode equivalence tests are **optional**.

## Sampler and loop

| Piece | Description |
|-------|-------------|
| `Sampler` | Greedy, temperature, top-k, top-p; seeded RNG for tests |
| `GenerationLoop` | `forward_prefill` → repeat `decode_step` + sample until EOS or `max_tokens` |

## Deliverables

```text
mini_infer --model synthetic --prompt "Hello" --max-tokens 16 --greedy
```

Extend Phase 3 `mini_infer` with `--prompt`, `--max-tokens`, and sampler flags.

## Tests

- Sampler distributions with fixed seed
- Full loop token ids for **tiny** `LlamaConfig`
- (Optional) two decode steps match joint prefill at last position

## Benchmarks

- Decode tokens/s, synthetic **1B block shape**, `seq_out=64` — run on **cloud** (same RAM guidance as Phase 3).

## Gate

- `ctest` for sampler + generation on tiny config
- `mini_infer --model synthetic` produces non-empty token sequence

## After Phase 4

Phase 5 — GGUF load + real Llama-3.2-1B Q4_K_M per [plan.md](plan.md).
