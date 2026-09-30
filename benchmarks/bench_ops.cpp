// Run: build/benchmarks/Release/bench_ops.exe  (or Debug)
#include "ops/ops.hpp"

#include <benchmark/benchmark.h>

using namespace std;

namespace serving {
namespace {

// make_random: Tensor — fills a tensor with a deterministic pseudo-random pattern.
Tensor make_random(vector<int64_t> shape, float scale = 0.01f) {
  Tensor t(move(shape));
  for (size_t i = 0; i < t.numel(); ++i) {
    t.data()[i] = scale * static_cast<float>(static_cast<int>(i % 97) - 48);
  }
  return t;
}

}  // namespace
}  // namespace serving

namespace {

// matmul_flop_count: int64_t — 2*M*K*N FLOPs for C[M,N] = A[M,K] * B[K,N].
int64_t matmul_flop_count(int64_t m, int64_t k, int64_t n) {
  return 2 * m * k * n;
}

// report_matmul_gflops_per_sec: void — sets GFLOP/s counter from timed matmul work.
void report_matmul_gflops_per_sec(benchmark::State& state, int64_t m, int64_t k,
                                  int64_t n) {
  const double total_gflops =
      static_cast<double>(state.iterations() * matmul_flop_count(m, k, n)) / 1e9;
  state.counters["GFLOP/s"] =
      benchmark::Counter(total_gflops, benchmark::Counter::kIsRate);
}

}  // namespace

static void BM_Matmul_128x2048x2048(benchmark::State& state) {
  using namespace serving;
  auto a = make_random({128, 2048});
  auto b = make_random({2048, 2048});
  Tensor out({128, 2048});
  for (auto _ : state) {
    matmul(a, b, out);
    benchmark::DoNotOptimize(out.data());
  }
  report_matmul_gflops_per_sec(state, 128, 2048, 2048);
}
BENCHMARK(BM_Matmul_128x2048x2048);

static void BM_Matmul_512x2048x8192(benchmark::State& state) {
  using namespace serving;
  auto a = make_random({512, 2048});
  auto b = make_random({2048, 8192});
  Tensor out({512, 8192});
  for (auto _ : state) {
    matmul(a, b, out);
    benchmark::DoNotOptimize(out.data());
  }
  report_matmul_gflops_per_sec(state, 512, 2048, 8192);
}
BENCHMARK(BM_Matmul_512x2048x8192);

static void BM_RmsNorm_1x2048(benchmark::State& state) {
  using namespace serving;
  auto x = make_random({1, 2048});
  auto w = Tensor::zeros({2048});
  for (int64_t i = 0; i < 2048; ++i) {
    w.at({i}) = 1.f;
  }
  Tensor out({1, 2048});
  for (auto _ : state) {
    rmsnorm(x, w, 1e-5f, out);
    benchmark::DoNotOptimize(out.data());
  }
}
BENCHMARK(BM_RmsNorm_1x2048);

static void BM_RmsNorm_32x2048(benchmark::State& state) {
  using namespace serving;
  auto x = make_random({32, 2048});
  auto w = Tensor::zeros({2048});
  for (int64_t i = 0; i < 2048; ++i) {
    w.at({i}) = 1.f;
  }
  Tensor out({32, 2048});
  for (auto _ : state) {
    rmsnorm(x, w, 1e-5f, out);
    benchmark::DoNotOptimize(out.data());
  }
}
BENCHMARK(BM_RmsNorm_32x2048);

BENCHMARK_MAIN();
