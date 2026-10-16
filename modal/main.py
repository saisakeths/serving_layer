"""Run bench_forward (Llama 1B-shaped) on Modal and save results under benchmarks/results/."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import modal

APP_NAME = "serving-layer-bench-forward"
REPO_ROOT = Path(__file__).resolve().parent.parent
REPO_MOUNT = "/src"
BUILD_DIR = f"{REPO_MOUNT}/build"
BENCH_BIN = f"{BUILD_DIR}/benchmarks/bench_forward"
DEFAULT_FILTER = "BM_LlamaForward_1B"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("build-essential", "cmake", "ninja-build", "git")
    .add_local_dir(
        str(REPO_ROOT),
        remote_path=REPO_MOUNT,
        ignore=[
            ".git/**",
            "build/**",
            "cmake-build-*/**",
            ".cursor/**",
            ".vs/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "benchmarks/results/**",
            "modal-run.log",
        ],
    )
)


def _run(cmd: list[str], cwd: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
    return result


def _read_mem_total_kb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1])
    except OSError:
        return None
    return None


def _collect_host_metadata() -> dict:
    mem_kb = _read_mem_total_kb()
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": platform.processor() or None,
        "logical_cpus": subprocess.check_output(["nproc"], text=True).strip(),
        "mem_total_gib": round(mem_kb / (1024 * 1024), 2) if mem_kb else None,
        "uname": subprocess.check_output(["uname", "-a"], text=True).strip(),
    }


def _format_report(metadata: dict, build_log: str, bench_stdout: str, bench_stderr: str) -> str:
    lines = [
        "serving_layer bench_forward (Modal)",
        "=" * 60,
        f"timestamp_utc: {metadata['timestamp_utc']}",
        f"benchmark_filter: {metadata['benchmark_filter']}",
        f"binary: {metadata['benchmark_binary']}",
        f"cmake_build_type: {metadata['cmake_build_type']}",
        "",
        "--- host ---",
        f"uname: {metadata['host'].get('uname')}",
        f"logical_cpus: {metadata['host'].get('logical_cpus')}",
        f"mem_total_gib: {metadata['host'].get('mem_total_gib')}",
        "",
        "--- cmake configure (stdout) ---",
        build_log.split("--- build ---", 1)[0].strip() if "--- build ---" in build_log else "",
        "",
        "--- cmake build (stdout) ---",
        build_log.split("--- build ---", 1)[1].strip() if "--- build ---" in build_log else build_log.strip(),
        "",
        "--- bench_forward stdout ---",
        bench_stdout.strip(),
    ]
    if bench_stderr.strip():
        lines.extend(["", "--- bench_forward stderr ---", bench_stderr.strip()])
    return "\n".join(lines) + "\n"


@app.function(
    image=image,
    cpu=8,
    memory=32768,
    timeout=6 * 60 * 60,
)
def run_bench_forward(benchmark_filter: str = DEFAULT_FILTER) -> dict:
    """Build serving_layer Release benchmarks and run bench_forward on Modal."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    configure = _run(
        [
            "cmake",
            "-B",
            "build",
            "-G",
            "Ninja",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DSERVING_BUILD_TESTS=OFF",
        ],
        cwd=REPO_MOUNT,
    )
    build = _run(["cmake", "--build", "build", "--parallel"], cwd=REPO_MOUNT)
    build_log = (
        configure.stdout
        + "\n--- build ---\n"
        + build.stdout
        + ("\n--- build stderr ---\n" + build.stderr if build.stderr else "")
    )

    bench = _run(
        [
            BENCH_BIN,
            f"--benchmark_filter={benchmark_filter}",
            "--benchmark_min_time=0.1s",
        ],
        cwd=REPO_MOUNT,
    )

    host = _collect_host_metadata()
    metadata = {
        "timestamp_utc": timestamp,
        "benchmark_filter": benchmark_filter,
        "benchmark_binary": BENCH_BIN,
        "cmake_build_type": "Release",
        "host": host,
    }
    report_text = _format_report(metadata, build_log, bench.stdout, bench.stderr)

    return {
        "metadata": metadata,
        "report_text": report_text,
        "bench_stdout": bench.stdout,
    }


def _write_local_results(payload: dict) -> tuple[Path, Path]:
    results_dir = REPO_ROOT / "benchmarks" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    ts_slug = payload["metadata"]["timestamp_utc"].replace(":", "").replace("-", "")
    txt_path = results_dir / f"bench_forward_1b_{ts_slug}.txt"
    json_path = results_dir / f"bench_forward_1b_{ts_slug}.json"
    latest_txt = results_dir / "bench_forward_1b.txt"
    latest_json = results_dir / "bench_forward_1b.json"

    txt_path.write_text(payload["report_text"], encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "metadata": payload["metadata"],
                "bench_stdout": payload["bench_stdout"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    latest_txt.write_text(payload["report_text"], encoding="utf-8")
    latest_json.write_text(json_path.read_text(encoding="utf-8"), encoding="utf-8")

    return txt_path, json_path


@app.local_entrypoint()
def main(benchmark_filter: str = DEFAULT_FILTER) -> None:
    """modal run modal/main.py [--benchmark-filter BM_LlamaForward_1B]"""
    print(f"Running bench_forward on Modal (filter={benchmark_filter}) ...", flush=True)
    payload = run_bench_forward.remote(benchmark_filter)
    txt_path, json_path = _write_local_results(payload)
    print(payload["bench_stdout"])
    print(f"\nSaved report: {txt_path}", flush=True)
    print(f"Saved JSON:   {json_path}", flush=True)
    print(f"Latest copy:  {REPO_ROOT / 'benchmarks' / 'results' / 'bench_forward_1b.txt'}", flush=True)


if __name__ == "__main__":
    print("Use: modal run modal/main.py", file=sys.stderr)
    sys.exit(1)
