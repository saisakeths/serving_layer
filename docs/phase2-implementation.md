# Phase 2 implementation guide

Parent: [plan.md](plan.md) · Style: [cpp-coding-style.md](cpp-coding-style.md)

Phase 2 is split so KV cache purpose is visible in benchmarks:

| Part | Focus | KV | Benchmarks |
|------|--------|----|------------|
| **2a** | RoPE, GQA attention, full `TransformerBlock` | None | Prefill seq 128/512 |
| **2b** | `KVCache`, incremental decode | Yes | Naive recompute vs KV decode |

## Prerequisites

- Phase 1 gate passed (`ctest`, `bench_ops`).
- Phase 2a gate passed.

## Build

```powershell
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
.\build\benchmarks\Release\bench_attention.exe
```

## Part A file checklist

| Path | Purpose |
|------|---------|
| `src/ops/ops.hpp`, `ops.cpp` | `linear` — `x[M,K] @ weight[N,K]^T → [M,N]` |
| `src/ops/rope.hpp`, `rope.cpp` | RoPE on Q/K `[batch, seq, heads, head_dim]` |
| `src/model/llama/block_config.hpp` | `dim`, `n_heads`, `n_kv_heads`, `head_dim`, `ffn_dim`, `rope_theta`, `rms_eps` |
| `src/model/llama/attention.hpp`, `attention.cpp` | GQA causal attention (full sequence) |
| `src/model/llama/transformer_block.hpp`, `transformer_block.cpp` | Pre-norm block + SwiGLU FFN |
| `tests/test_attention.cpp` | Scores vs naive reference |
| `tests/test_one_block_forward.cpp` | Block output shape / determinism |
| `benchmarks/bench_attention.cpp` | Prefill tokens/s |

### Gate (2a)

- All `ctest` targets pass (including attention tests).
- `bench_attention` runs prefill benchmarks without error.

## Part B file checklist

| Path | Purpose |
|------|---------|
| `src/cache/kv_cache.hpp`, `kv_cache.cpp` | Contiguous post-RoPE K/V `[batch, max_seq, kv_heads, head_dim]` |
| `src/model/llama/attention.hpp`, `attention.cpp` | `gqa_attention_prefill`, `gqa_attention_decode` |
| `src/model/llama/transformer_block.hpp`, `transformer_block.cpp` | `forward_prefill`, `decode_step` |
| `tests/test_kv_cache.cpp` | Prefill equivalence, decode vs joint forward |
| `benchmarks/bench_attention.cpp` | `BM_Block_DecodeNaive` vs `BM_Block_DecodeKV` |

### Gate (2b)

- `ctest` passes including `test_kv_cache`.
- `bench_attention` runs prefill plus decode suites.
- `src/cache/` holds contiguous KV only (paging is Phase 10).

### Decode benchmarks

- **DecodeNaive:** each new token re-runs `TransformerBlock::forward` on the growing sequence (recomputes all K/V).
- **DecodeKV:** one `forward_prefill` on the prompt, then `decode_step` per token.
- Compare `tokens/s` across `prompt_len` 64 / 128 / 256 with 8 decode steps; KV should pull ahead as context grows because naive work scales with sequence length per token.

## Status

- **2a:** Complete — prefill seq 128/512.
- **2b:** Complete — `test_kv_cache`; decode benchmarks on 1B-shaped block config.

## After 2b

Proceed to Phase 3 (`LlamaModel`, multi-layer cache handles) per [plan.md](plan.md).
