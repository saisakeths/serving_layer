# Phase 1 implementation guide

Parent: [plan.md](plan.md) Phase 1 · Style: [cpp-coding-style.md](cpp-coding-style.md)

## Coding rules (mandatory)

Follow [cpp-coding-style.md](cpp-coding-style.md): `using namespace std;` after includes; one-line `// name: ReturnType — description` before every function.

## Build (no Boost)

```powershell
cmake -B build
cmake --build build
ctest --test-dir build
.\build\benchmarks\Release\bench_ops.exe
```

Adjust config path (`Release` vs `Debug`) for your generator.

## File checklist

| Path | Purpose |
|------|---------|
| `CMakeLists.txt` | Root project, C++17, test/bench options |
| `cmake/Dependencies.cmake` | FetchContent GTest 1.14, benchmark 1.8.3 |
| `src/CMakeLists.txt` | Library `serving_core` |
| `src/core/dtypes.hpp`, `dtypes.cpp` | `DType` enum, `dtype_size` |
| `src/core/tensor.hpp`, `tensor.cpp` | Shape, strides, `zeros`, `from_vector`, `at`, `view` |
| `src/ops/ops.hpp`, `ops.cpp` | matmul, add/mul inplace, rmsnorm, softmax_last_dim, silu_inplace, embedding |
| `tests/CMakeLists.txt` | `test_tensor`, `test_ops` |
| `tests/test_tensor.cpp` | Shape, zeros, from_vector |
| `tests/test_ops.cpp` | 2×2 matmul, RMSNorm, softmax sum, SiLU(0), embedding |
| `benchmarks/CMakeLists.txt` | `bench_ops` |
| `benchmarks/bench_ops.cpp` | Matmul 128×2048×2048; RMSNorm [1,2048] and [32,2048] |
| `.cursor/rules/cpp-serving-layer-style.mdc` | Copy of style rule for Agent |

## Gate

- All `ctest` tests pass
- `bench_ops` runs without error
- No files under `model/`, `server/`, `io/` yet

**Status:** Complete (2026-09-28). Build: `cmake -B build`; `ctest --test-dir build -C Release` — 9/9 passed.

## After completion

Note completion date in this file or a short entry under Phase 1 in `plan.md`, then start Phase 2 per `plan.md`.
