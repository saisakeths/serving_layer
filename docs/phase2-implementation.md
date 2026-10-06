# Phase 2 implementation guide

Parent: [plan.md](plan.md) · Style: [cpp-coding-style.md](cpp-coding-style.md)

Phase 2 is split so KV cache purpose is visible in benchmarks:

| Part | Focus | KV | Benchmarks |
|------|--------|----|------------|
| **2a** | RoPE, GQA attention, full `TransformerBlock` | None | Prefill seq 128/512 |
| **2b** | `KVCache`, incremental decode | Yes | Naive recompute vs KV decode |

This file tracks **Phase 2a** only.

## Prerequisites

- Phase 1 gate passed (`ctest`, `bench_ops`).

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

## Gate (2a)

- All `ctest` targets pass (including new attention tests).
- `bench_attention` runs prefill benchmarks without error.
- No `src/cache/` directory.

## Status

- **2a:** Complete — `ctest` 17/17; `bench_attention` prefill seq 128/512.
- **2b:** Not started (see [plan.md](plan.md) Phase 2b).

## After 2a

Complete gate above, then implement Phase 2b per `plan.md`.
