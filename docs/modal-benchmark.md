# Modal cloud benchmark (`bench_forward`)

Run the **Llama 1B-shaped** CPU forward benchmark on a Modal worker (≥32 GiB RAM) and save artifacts under `benchmarks/results/`.

Parent: [phase3-implementation.md](phase3-implementation.md)

## Prerequisites

- [Modal account](https://modal.com/) and CLI: `pip install -r modal/requirements.txt`
- Authenticate: `modal setup` (or `modal token new`)
- Run from the **repository root**

## Run

```powershell
cd E:\AI_Systems\serving_layer
modal run modal/main.py
```

Optional filter (default `BM_LlamaForward_1B`):

```powershell
modal run modal/main.py --benchmark-filter BM_LlamaForward_1B
```

## What it does

1. Uploads the repo (excluding `build/`, `.git/`, prior `benchmarks/results/`).
2. On Modal: `cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DSERVING_BUILD_TESTS=OFF`, builds `bench_forward`.
3. Runs `./build/benchmarks/bench_forward --benchmark_filter=BM_LlamaForward_1B`.
4. Writes locally (gitignored):
   - `benchmarks/results/bench_forward_1b_<timestamp>.txt` — full report + host metadata + build log + benchmark output
   - `benchmarks/results/bench_forward_1b_<timestamp>.json` — structured metadata + raw benchmark stdout
   - `benchmarks/results/bench_forward_1b.txt` / `.json` — latest run copies

## Resource defaults

| Setting | Value | Why |
|---------|-------|-----|
| `memory` | 32768 MiB | Phase 3 gate: 1B-shaped random weights need ≥16 GiB; extra headroom for `seq=512` |
| `cpu` | 8 | Faster Release matmul on CPU |
| `timeout` | 21600 s (6 h) | 1B CPU forward can exceed 1 h with default benchmark timing |

Adjust in `modal/main.py` `@app.function(...)` if runs OOM or time out.

## Gate checklist

After a successful Modal run, confirm `bench_forward` output includes **seq 1** and **128** (and other registered args: 64, 256, 512). Archive the `.txt` file for the Phase 3 cloud benchmark gate.
