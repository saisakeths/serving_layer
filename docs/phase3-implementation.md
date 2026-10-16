# Phase 3 implementation guide

Parent: [plan.md](plan.md) · Style: [cpp-coding-style.md](cpp-coding-style.md)

**Goal:** Multi-layer Llama-style **prefill** forward: token embedding → `n_layer` blocks → output RMSNorm → logits. **No** `LlamaCache` (deferred to [phase4-implementation.md](phase4-implementation.md)).

## Prerequisites

- Phase 2b gate passed (`ctest`, including `test_kv_cache`).

## Build

```powershell
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
.\build\tools\Release\mini_infer.exe --synthetic --seq 8
```

## File checklist

| Path | Purpose |
|------|---------|
| `src/model/llama/llama_config.hpp` | `LlamaConfig::tiny()`, `llama_1b_shape()` |
| `src/model/llama/llama_weights.hpp`, `llama_weights.cpp` | Weights bundle + `make_random_llama_weights` |
| `src/model/llama/llama_model.hpp`, `llama_model.cpp` | `LlamaModel::forward` |
| `tests/test_llama_forward.cpp` | Shape + determinism (tiny only) |
| `benchmarks/bench_forward.cpp` | `BM_LlamaForward_1B` seq **1** and **128** |
| `tools/mini_infer/main.cpp` | One synthetic forward |

## Config reference

| | Tiny (tests) | 1B-shaped (cloud bench) |
|--|--------------|-------------------------|
| `n_layer` | 2 | 16 |
| `vocab_size` | 32 | 128256 |
| Block dims | 8 / 2 heads / ffn 16 | Same as `config_1b_shape()` in `bench_attention.cpp` |

## Cloud benchmark (mandatory gate)

Run **only** on a machine with **≥16 GB RAM** (not on 8 GB laptop).

**Option A — Modal (recommended):** see [modal-benchmark.md](modal-benchmark.md).

```powershell
pip install -r modal/requirements.txt
modal setup
modal run modal/main.py
```

**Option B — manual VM / cloud shell (Linux):**

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release -DSERVING_BUILD_TESTS=OFF
cmake --build build --parallel
./build/benchmarks/bench_forward --benchmark_filter=BM_LlamaForward_1B
```

**Option C — Windows Release build:**

```powershell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
.\build\benchmarks\Release\bench_forward.exe --benchmark_filter=BM_LlamaForward_1B
```

Save output under `benchmarks/results/` (e.g. `bench_forward_1b.txt`). The Modal wrapper writes timestamped `.txt`/`.json` plus `bench_forward_1b.txt` automatically.

**When:** After local `ctest` passes; **before** marking Phase 3 complete and starting Phase 4.

## Gate

- [ ] `ctest` includes `test_llama_forward`
- [ ] Cloud `bench_forward` 1B seq 1 & 128 recorded
- [ ] `mini_infer --synthetic` runs locally

## Status

- **Implementation:** In progress — code landed; cloud bench pending.
- **LlamaCache / decode:** Phase 4 per [phase4-implementation.md](phase4-implementation.md).

## After Phase 3

Proceed to Phase 4 (cache + sampling + generation loop).
