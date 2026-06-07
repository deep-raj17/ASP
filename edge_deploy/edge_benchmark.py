"""
edge_deploy/edge_benchmark.py
────────────────────────────────────────────────────────
RPi5 benchmarking suite for ONNX models.

Tests:
  1. Latency profiling (p50/p95/p99)
  2. Throughput measurement
  3. Thermal soak test (multi-hour sustained inference)
  4. Memory profiling
  5. FP32 vs INT8 accuracy comparison
  6. Mel transform benchmark

Usage on RPi5:
    python edge_benchmark.py --model models/classifier_int8.onnx
    python edge_benchmark.py --model models/classifier_int8.onnx --soak-hours 4
    python edge_benchmark.py --validate --fp32 models/classifier_fp32.onnx --int8 models/classifier_int8.onnx
"""

from __future__ import annotations

import os
import sys
import time
import csv
import argparse
import tracemalloc
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class BenchmarkResult:
    name: str
    iterations: int
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    throughput_per_sec: float
    peak_memory_mb: float

    def __str__(self):
        return (
            f"\n{'─'*50}\n"
            f"  {self.name}\n"
            f"{'─'*50}\n"
            f"  Iterations:  {self.iterations}\n"
            f"  Mean:        {self.mean_ms:.2f} ms\n"
            f"  Std:         {self.std_ms:.2f} ms\n"
            f"  P50:         {self.p50_ms:.2f} ms\n"
            f"  P95:         {self.p95_ms:.2f} ms\n"
            f"  P99:         {self.p99_ms:.2f} ms\n"
            f"  Min:         {self.min_ms:.2f} ms\n"
            f"  Max:         {self.max_ms:.2f} ms\n"
            f"  Throughput:  {self.throughput_per_sec:.1f} samples/sec\n"
            f"  Peak Memory: {self.peak_memory_mb:.1f} MB"
        )


def _get_memory_mb() -> float:
    """Get current process RSS in MB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        try:
            with open(f"/proc/{os.getpid()}/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024
        except FileNotFoundError:
            pass
    return 0.0


def _get_temp() -> float:
    """Read CPU temp."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        return -1.0


def _make_input(shape) -> np.ndarray:
    """Generate synthetic mel-like input."""
    rng = np.random.RandomState(42)
    return rng.beta(2, 5, size=shape).astype(np.float32)


# ─────────────────────────────────────────────────────────
#  Latency Benchmark
# ─────────────────────────────────────────────────────────

def benchmark_latency(
    model_path: str,
    warmup: int = 20,
    iterations: int = 200,
    num_threads: int = 3,
) -> BenchmarkResult:
    """Measure inference latency with percentile breakdown."""
    import onnxruntime as ort

    # Setup
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = num_threads
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

    providers = []
    available = ort.get_available_providers()
    if "XNNPACKExecutionProvider" in available:
        providers.append("XNNPACKExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = [s if isinstance(s, int) else 1 for s in input_meta.shape]

    print(f"\n[Bench] Model: {os.path.basename(model_path)}")
    print(f"[Bench] Input shape: {shape}")
    print(f"[Bench] Providers: {session.get_providers()}")
    print(f"[Bench] Warmup: {warmup}, Iterations: {iterations}")

    x = _make_input(shape)

    # Warmup
    print("[Bench] Warming up...", end="", flush=True)
    for _ in range(warmup):
        session.run(None, {input_name: x})
    print(" done")

    # Benchmark
    latencies = []
    peak_mem = _get_memory_mb()

    for i in range(iterations):
        start = time.perf_counter()
        session.run(None, {input_name: x})
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)

        mem = _get_memory_mb()
        peak_mem = max(peak_mem, mem)

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{iterations}] last={elapsed:.1f}ms", flush=True)

    arr = np.array(latencies)
    total_sec = arr.sum() / 1000

    return BenchmarkResult(
        name=f"Latency: {os.path.basename(model_path)}",
        iterations=iterations,
        mean_ms=float(np.mean(arr)),
        std_ms=float(np.std(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        throughput_per_sec=iterations / total_sec,
        peak_memory_mb=peak_mem,
    )


# ─────────────────────────────────────────────────────────
#  Thermal Soak Test
# ─────────────────────────────────────────────────────────

def benchmark_soak(
    model_path: str,
    hours: float = 1.0,
    log_interval_sec: int = 30,
    num_threads: int = 3,
    output_csv: str = "soak_results.csv",
):
    """Sustained inference for thermal stability testing."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")

    providers = []
    if "XNNPACKExecutionProvider" in ort.get_available_providers():
        providers.append("XNNPACKExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, sess_options=opts, providers=providers)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    shape = [s if isinstance(s, int) else 1 for s in input_meta.shape]

    x = _make_input(shape)
    duration_sec = hours * 3600

    print(f"\n[Soak] Starting {hours}h soak test on {os.path.basename(model_path)}")
    print(f"[Soak] Logging every {log_interval_sec}s → {output_csv}")

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: x})

    start_time = time.time()
    last_log = start_time
    total_inferences = 0
    window_latencies = []

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "elapsed_min", "temp_c", "memory_mb",
            "mean_ms", "p95_ms", "p99_ms", "inferences"
        ])

        while (time.time() - start_time) < duration_sec:
            t0 = time.perf_counter()
            session.run(None, {input_name: x})
            elapsed_ms = (time.perf_counter() - t0) * 1000

            total_inferences += 1
            window_latencies.append(elapsed_ms)

            now = time.time()
            if now - last_log >= log_interval_sec:
                arr = np.array(window_latencies)
                elapsed_min = (now - start_time) / 60
                temp = _get_temp()
                mem = _get_memory_mb()

                row = [
                    f"{elapsed_min:.1f}",
                    f"{temp:.1f}",
                    f"{mem:.0f}",
                    f"{np.mean(arr):.1f}",
                    f"{np.percentile(arr, 95):.1f}",
                    f"{np.percentile(arr, 99):.1f}",
                    str(total_inferences),
                ]
                writer.writerow(row)
                csvfile.flush()

                print(
                    f"  [{elapsed_min:.0f}m] "
                    f"temp={temp:.1f}°C mem={mem:.0f}MB "
                    f"lat_mean={np.mean(arr):.0f}ms "
                    f"p95={np.percentile(arr, 95):.0f}ms "
                    f"n={total_inferences}"
                )

                window_latencies.clear()
                last_log = now

    print(f"\n[Soak] Complete: {total_inferences} inferences in {hours}h")
    print(f"[Soak] Results → {output_csv}")


# ─────────────────────────────────────────────────────────
#  Mel Transform Benchmark
# ─────────────────────────────────────────────────────────

def benchmark_mel_transform(iterations: int = 100):
    """Benchmark the numpy-based mel spectrogram transform."""
    _edge_dir = str(Path(__file__).resolve().parent)
    if _edge_dir not in sys.path:
        sys.path.insert(0, _edge_dir)
    from edge_streaming import MelTransform

    mel = MelTransform(sr=16000, n_fft=2048, hop_length=512,
                       n_mels=128, fmin=20.0, fmax=8000.0)

    audio = np.random.randn(160000).astype(np.float32) * 0.5

    # Warmup
    for _ in range(5):
        mel(audio)

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        mel(audio)
        latencies.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies)
    print(f"\n[Mel Bench] {iterations} iterations")
    print(f"  Mean:  {np.mean(arr):.2f} ms")
    print(f"  P95:   {np.percentile(arr, 95):.2f} ms")
    print(f"  P99:   {np.percentile(arr, 99):.2f} ms")


# ─────────────────────────────────────────────────────────
#  Accuracy Validation
# ─────────────────────────────────────────────────────────

def validate_accuracy(
    fp32_path: str,
    int8_path: str,
    n_samples: int = 100,
):
    """Compare FP32 vs INT8 output divergence."""
    import onnxruntime as ort

    print(f"\n[Validate] FP32: {os.path.basename(fp32_path)}")
    print(f"[Validate] INT8: {os.path.basename(int8_path)}")
    print(f"[Validate] Samples: {n_samples}")

    sess_fp32 = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    input_name = sess_fp32.get_inputs()[0].name
    shape = [s if isinstance(s, int) else 1 for s in sess_fp32.get_inputs()[0].shape]

    rng = np.random.RandomState(123)
    embed_maes = []
    logit_maes = []
    pooled_maes = []

    for i in range(n_samples):
        x = rng.beta(2, 5, size=shape).astype(np.float32)

        out_fp32 = sess_fp32.run(None, {input_name: x})
        out_int8 = sess_int8.run(None, {input_name: x})

        embed_maes.append(np.mean(np.abs(out_fp32[0] - out_int8[0])))
        logit_maes.append(np.mean(np.abs(out_fp32[1] - out_int8[1])))
        if len(out_fp32) > 2:
            pooled_maes.append(np.mean(np.abs(out_fp32[2] - out_int8[2])))

    print(f"\n  {'Metric':<25s} {'Mean MAE':>10s} {'Max MAE':>10s}")
    print(f"  {'─'*47}")
    print(f"  {'Embedding MAE':<25s} {np.mean(embed_maes):>10.6f} {np.max(embed_maes):>10.6f}")
    print(f"  {'Logit MAE':<25s} {np.mean(logit_maes):>10.6f} {np.max(logit_maes):>10.6f}")
    if pooled_maes:
        print(f"  {'Pooled Feat MAE':<25s} {np.mean(pooled_maes):>10.6f} {np.max(pooled_maes):>10.6f}")

    # Classification agreement
    agreements = 0
    for i in range(n_samples):
        x = rng.beta(2, 5, size=shape).astype(np.float32)
        o1 = sess_fp32.run(None, {input_name: x})
        o2 = sess_int8.run(None, {input_name: x})
        if (o1[1][0] > 0) == (o2[1][0] > 0):
            agreements += 1

    agreement_pct = agreements / n_samples * 100
    print(f"\n  Classification agreement: {agreement_pct:.1f}%")

    if agreement_pct < 95:
        print("  ⚠ WARNING: Agreement below 95% — consider excluding more ops")
    else:
        print("  ✓ Quantization quality: ACCEPTABLE")


# ─────────────────────────────────────────────────────────
#  Memory Profiling
# ─────────────────────────────────────────────────────────

def profile_memory(model_path: str, iterations: int = 50):
    """Profile memory allocations during inference."""
    import onnxruntime as ort

    tracemalloc.start()

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_meta = sess.get_inputs()[0]
    input_name = input_meta.name
    shape = [s if isinstance(s, int) else 1 for s in input_meta.shape]
    x = _make_input(shape)

    # Baseline
    snapshot_before = tracemalloc.take_snapshot()

    for _ in range(iterations):
        sess.run(None, {input_name: x})

    snapshot_after = tracemalloc.take_snapshot()

    print(f"\n[Memory] Top allocations after {iterations} inferences:")
    stats = snapshot_after.compare_to(snapshot_before, "lineno")
    for stat in stats[:15]:
        print(f"  {stat}")

    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"\n  Current: {current / 1024 / 1024:.1f} MB")
    print(f"  Peak:    {peak / 1024 / 1024:.1f} MB")


# ─────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RPi5 Edge Model Benchmarking")
    parser.add_argument("--model", help="ONNX model path for benchmarking")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--threads", type=int, default=3)

    parser.add_argument("--soak-hours", type=float, default=0,
                        help="Run thermal soak test for N hours")
    parser.add_argument("--soak-log-interval", type=int, default=30)

    parser.add_argument("--validate", action="store_true",
                        help="Compare FP32 vs INT8 accuracy")
    parser.add_argument("--fp32", help="FP32 model path (for validation)")
    parser.add_argument("--int8", help="INT8 model path (for validation)")

    parser.add_argument("--memory", action="store_true",
                        help="Run memory profiling")
    parser.add_argument("--mel", action="store_true",
                        help="Benchmark mel transform")

    parser.add_argument("--output", default="benchmark_edge_results.txt")
    args = parser.parse_args()

    results = []

    # Latency benchmark
    if args.model:
        result = benchmark_latency(
            args.model, args.warmup, args.iterations, args.threads
        )
        results.append(result)
        print(result)

    # Soak test
    if args.soak_hours > 0 and args.model:
        benchmark_soak(
            args.model, args.soak_hours,
            args.soak_log_interval, args.threads
        )

    # Validation
    if args.validate and args.fp32 and args.int8:
        validate_accuracy(args.fp32, args.int8)

    # Memory
    if args.memory and args.model:
        profile_memory(args.model, args.iterations)

    # Mel transform
    if args.mel:
        benchmark_mel_transform()

    # Save results
    if results:
        with open(args.output, "w") as f:
            f.write("MIMII Edge Benchmark Results\n")
            f.write("=" * 50 + "\n")
            for r in results:
                f.write(str(r) + "\n\n")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
