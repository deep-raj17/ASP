"""
benchmark.py
────────────────────────────────────────────────────────
Performance benchmarking suite for production optimization.

Measures:
  - Model inference latency (single and batch)
  - Throughput (samples/sec)
  - GPU memory utilization
  - Data loading speed
  - End-to-end pipeline performance
"""

from __future__ import annotations

import os
import sys
import time
import argparse
import statistics
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
from models.hybrid_model import HybridAnomalyModel
from inference.production_detector import ProductionDetector
from utils.audio_utils import AudioProcessor, pad_or_trim
from utils.gpu_utils import (
    setup_cuda_optimizations, get_memory_stats,
    compile_model, empty_cache
)
from utils.validation import setup_logging, logger


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    num_samples: int
    total_time_sec: float
    avg_latency_ms: float
    std_latency_ms: float
    throughput_samples_per_sec: float
    gpu_memory_peak_gb: float

    def __str__(self):
        return (
            f"{self.name}\n"
            f"  Samples: {self.num_samples}\n"
            f"  Total time: {self.total_time_sec:.2f}s\n"
            f"  Avg latency: {self.avg_latency_ms:.2f}ms (+/- {self.std_latency_ms:.2f}ms)\n"
            f"  Throughput: {self.throughput_samples_per_sec:.1f} samples/sec\n"
            f"  Peak GPU memory: {self.gpu_memory_peak_gb:.2f}GB"
        )


def generate_synthetic_audio(
    num_samples: int,
    duration_sec: float = 10.0,
    sample_rate: int = 16000,
) -> torch.Tensor:
    """Generate synthetic audio for benchmarking."""
    length = int(duration_sec * sample_rate)
    # Mix of sine waves and noise
    t = torch.linspace(0, duration_sec, length)
    audio = (
        0.5 * torch.sin(2 * np.pi * 440 * t)  # 440 Hz tone
        + 0.3 * torch.sin(2 * np.pi * 880 * t)  # 880 Hz harmonic
        + 0.1 * torch.randn(length)  # Noise
    )
    return audio.unsqueeze(0)  # Add channel dimension


def benchmark_single_inference(
    detector: ProductionDetector,
    processor: AudioProcessor,
    num_warmup: int = 10,
    num_runs: int = 100,
) -> BenchmarkResult:
    """Benchmark single-sample inference."""
    logger.info("Benchmarking single-sample inference...")

    device = detector.device
    target_len = int(cfg.data.sample_rate * cfg.data.audio_duration_sec)

    # Generate test input
    waveform = generate_synthetic_audio(1).to(device)
    waveform = pad_or_trim(waveform, target_len)
    mel, _, _ = processor(waveform, augment=False)

    # Warmup
    logger.info(f"Warming up with {num_warmup} iterations...")
    for _ in range(num_warmup):
        _ = detector.detect_batch(mel.unsqueeze(0))

    empty_cache()
    torch.cuda.synchronize() if device.type == "cuda" else None

    # Benchmark
    logger.info(f"Running {num_runs} iterations...")
    latencies = []
    peak_memory = 0.0

    for _ in tqdm(range(num_runs), desc="Single inference"):
        if device.type == "cuda":
            torch.cuda.synchronize()
            mem_before = torch.cuda.memory_allocated() / 1e9

        start = time.perf_counter()
        _ = detector.detect_batch(mel.unsqueeze(0))
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()

        latencies.append((end - start) * 1000)  # Convert to ms

        if device.type == "cuda":
            mem_after = torch.cuda.memory_allocated() / 1e9
            peak_memory = max(peak_memory, mem_after)

    total_time = sum(latencies) / 1000  # Convert back to seconds

    return BenchmarkResult(
        name="Single Sample Inference",
        num_samples=num_runs,
        total_time_sec=total_time,
        avg_latency_ms=statistics.mean(latencies),
        std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        throughput_samples_per_sec=num_runs / total_time,
        gpu_memory_peak_gb=peak_memory,
    )


def benchmark_batch_inference(
    detector: ProductionDetector,
    processor: AudioProcessor,
    batch_sizes: List[int] | None = None,
    num_runs_per_batch: int = 20,
) -> List[BenchmarkResult]:
    """Benchmark batch inference at different batch sizes."""
    if batch_sizes is None:
        batch_sizes = [1, 4, 8, 16, 32]
    logger.info("Benchmarking batch inference...")

    device = detector.device
    target_len = int(cfg.data.sample_rate * cfg.data.audio_duration_sec)

    results = []

    for batch_size in batch_sizes:
        logger.info(f"\nBatch size: {batch_size}")

        # Generate batch
        waveforms = []
        for _ in range(batch_size):
            w = generate_synthetic_audio(1)
            w = pad_or_trim(w, target_len)
            waveforms.append(w)

        batch = torch.stack([processor(w, augment=False)[0] for w in waveforms])
        batch = batch.to(device)

        # Warmup
        for _ in range(3):
            _ = detector.detect_batch(batch)

        empty_cache()
        torch.cuda.synchronize() if device.type == "cuda" else None

        # Benchmark
        latencies = []
        peak_memory = 0.0

        for _ in tqdm(range(num_runs_per_batch), desc=f"Batch {batch_size}"):
            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = detector.detect_batch(batch)
            if device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

            if device.type == "cuda":
                mem = torch.cuda.memory_allocated() / 1e9
                peak_memory = max(peak_memory, mem)

        total_time = sum(latencies) / 1000
        samples_processed = batch_size * num_runs_per_batch

        results.append(BenchmarkResult(
            name=f"Batch Inference (size={batch_size})",
            num_samples=samples_processed,
            total_time_sec=total_time,
            avg_latency_ms=statistics.mean(latencies),
            std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
            throughput_samples_per_sec=samples_processed / total_time,
            gpu_memory_peak_gb=peak_memory,
        ))

    return results


def benchmark_with_without_compilation(
    cfg,
    device: torch.device,
) -> Tuple[BenchmarkResult, BenchmarkResult]:
    """Compare performance with and without torch.compile()."""
    logger.info("\n=== Comiling vs Non-compiled Performance ===")

    processor = AudioProcessor(cfg.data)
    target_len = int(cfg.data.sample_rate * cfg.data.audio_duration_sec)

    # Without compilation
    logger.info("Testing WITHOUT torch.compile()...")
    model1 = HybridAnomalyModel(cfg.model).to(device)
    detector1 = ProductionDetector(model1, cfg, use_compiled=False)

    result_no_compile = benchmark_single_inference(detector1, processor, num_runs=50)

    del model1, detector1
    empty_cache()

    # With compilation
    logger.info("\nTesting WITH torch.compile()...")
    model2 = HybridAnomalyModel(cfg.model).to(device)
    detector2 = ProductionDetector(model2, cfg, use_compiled=True)

    result_compiled = benchmark_single_inference(detector2, processor, num_runs=50)

    return result_no_compile, result_compiled


def main():
    parser = argparse.ArgumentParser(description="Benchmark MIMII Anomaly Detection")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 4, 8, 16],
                        help="Batch sizes to test")
    parser.add_argument("--num-runs", type=int, default=100,
                        help="Number of runs for single inference")
    parser.add_argument("--test-compilation", action="store_true",
                        help="Compare with/without torch.compile()")
    parser.add_argument("--output", type=str, default="benchmark_results.txt",
                        help="Output file for results")
    args = parser.parse_args()

    setup_logging()
    setup_cuda_optimizations()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Initialize detector
    model = HybridAnomalyModel(cfg.model).to(device)
    detector = ProductionDetector(model, cfg, use_compiled=True)
    processor = AudioProcessor(cfg.data)

    # Run benchmarks
    results = []

    # Single inference
    single_result = benchmark_single_inference(detector, processor, num_runs=args.num_runs)
    results.append(single_result)

    # Batch inference
    batch_results = benchmark_batch_inference(
        detector, processor,
        batch_sizes=args.batch_sizes,
        num_runs_per_batch=20
    )
    results.extend(batch_results)

    # Optional: compare with/without compilation
    if args.test_compilation and hasattr(torch, "compile"):
        r1, r2 = benchmark_with_without_compilation(cfg, device)
        results.append(r1)
        results.append(r2)

    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)

    for r in results:
        print(f"\n{r}")

    # Save to file
    with open(args.output, "w") as f:
        f.write("MIMII Anomaly Detection Benchmark Results\n")
        f.write("=" * 60 + "\n\n")
        for r in results:
            f.write(str(r) + "\n\n")

    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
