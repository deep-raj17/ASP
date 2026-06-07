"""
utils/monitoring.py
────────────────────────────────────────────────────────
Production monitoring and metrics collection.

Features:
  - Prometheus-compatible metrics export
  - Performance profiling
  - Alerting thresholds
  - Request tracing
"""

from __future__ import annotations

import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
from functools import wraps

import torch
import numpy as np

from utils.gpu_utils import get_memory_stats
from utils.validation import logger


@dataclass
class MetricSample:
    """A single metric sample."""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricCollector:
    """Collects and aggregates metrics over time."""

    def __init__(self, max_history: int = 10000):
        self.max_history = max_history
        self._metrics: Dict[str, deque] = {}
        self._lock = threading.RLock()

    def record(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = deque(maxlen=self.max_history)

            self._metrics[name].append(MetricSample(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            ))

    def get_stats(self, name: str, window_seconds: Optional[float] = None) -> Dict[str, float]:
        """Get statistics for a metric."""
        with self._lock:
            if name not in self._metrics:
                return {}

            samples = list(self._metrics[name])

        if window_seconds:
            cutoff = time.time() - window_seconds
            samples = [s for s in samples if s.timestamp > cutoff]

        if not samples:
            return {}

        values = [s.value for s in samples]

        return {
            "count": len(values),
            "mean": np.mean(values),
            "std": np.std(values) if len(values) > 1 else 0,
            "min": np.min(values),
            "max": np.max(values),
            "p50": np.percentile(values, 50),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
        }

    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics for all metrics."""
        with self._lock:
            metric_names = list(self._metrics.keys())

        return {name: self.get_stats(name) for name in metric_names}

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        for name, samples in self._metrics.items():
            if not samples:
                continue

            # Replace non-alphanumeric characters
            metric_name = f"mimii_{name.replace('.', '_').replace('-', '_')}"

            lines.append(f"# TYPE {metric_name} gauge")

            for sample in samples:
                labels_str = ",".join(
                    f'{k}="{v}"' for k, v in sample.labels.items()
                )
                if labels_str:
                    lines.append(f"{metric_name}{{{labels_str}}} {sample.value}")
                else:
                    lines.append(f"{metric_name} {sample.value}")

        return "\n".join(lines)

    def clear(self):
        """Clear all metrics."""
        with self._lock:
            self._metrics.clear()


# Global collector instance
COLLECTOR = MetricCollector()


class PerformanceMonitor:
    """Context manager for timing operations."""

    def __init__(
        self,
        operation_name: str,
        collector: Optional[MetricCollector] = None,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.operation_name = operation_name
        self.collector = collector or COLLECTOR
        self.labels = labels or {}
        self.start_time: Optional[float] = None
        self.duration_ms: Optional[float] = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration_ms = (time.perf_counter() - self.start_time) * 1000

        # Record timing
        self.collector.record(
            f"{self.operation_name}_duration_ms",
            self.duration_ms,
            {**self.labels, "status": "error" if exc_val else "success"}
        )

        # Log slow operations
        if self.duration_ms > 1000:  # > 1 second
            logger.warning(
                f"Slow operation: {self.operation_name} took {self.duration_ms:.2f}ms"
            )

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds (only valid during context)."""
        if self.start_time is None:
            return 0.0
        return (time.perf_counter() - self.start_time) * 1000


@contextmanager
def timed(operation_name: str, **labels):
    """Convenience context manager for timing."""
    monitor = PerformanceMonitor(operation_name, labels=labels)
    with monitor:
        yield monitor


def track_execution(metric_name: str):
    """Decorator for tracking function execution time."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            with timed(metric_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


class AlertManager:
    """Simple alerting based on thresholds."""

    def __init__(self):
        self._thresholds: Dict[str, tuple] = {}
        self._handlers: List[Callable] = []
        self._cooldowns: Dict[str, float] = {}

    def add_threshold(
        self,
        metric_name: str,
        max_value: Optional[float] = None,
        min_value: Optional[float] = None,
        cooldown_seconds: float = 300,
    ):
        """Add an alert threshold."""
        self._thresholds[metric_name] = (min_value, max_value, cooldown_seconds)

    def add_handler(self, handler: Callable[[str, float, str], None]):
        """Add an alert handler."""
        self._handlers.append(handler)

    def check(self, metric_name: str, value: float):
        """Check if value triggers alert."""
        if metric_name not in self._thresholds:
            return

        min_val, max_val, cooldown = self._thresholds[metric_name]
        now = time.time()

        # Check cooldown
        last_alert = self._cooldowns.get(metric_name, 0)
        if now - last_alert < cooldown:
            return

        # Check threshold
        triggered = False
        message = ""

        if max_val is not None and value > max_val:
            triggered = True
            message = f"{metric_name} = {value:.2f} > max threshold {max_val}"
        elif min_val is not None and value < min_val:
            triggered = True
            message = f"{metric_name} = {value:.2f} < min threshold {min_val}"

        if triggered:
            self._cooldowns[metric_name] = now
            for handler in self._handlers:
                try:
                    handler(metric_name, value, message)
                except Exception as e:
                    logger.error(f"Alert handler failed: {e}")


class SystemMonitor:
    """Monitor system resources."""

    def __init__(self, interval_seconds: float = 60.0):
        self.interval = interval_seconds
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start background monitoring."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info(f"System monitor started (interval={self.interval}s)")

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                self._collect_metrics()
            except Exception as e:
                logger.error(f"System monitoring error: {e}")

            # Sleep with early exit check
            for _ in range(int(self.interval)):
                if not self._running:
                    break
                time.sleep(1)

    def _collect_metrics(self):
        """Collect system metrics."""
        import psutil

        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        COLLECTOR.record("system.cpu_percent", cpu_percent)

        # Memory
        mem = psutil.virtual_memory()
        COLLECTOR.record("system.memory_percent", mem.percent)
        COLLECTOR.record("system.memory_available_gb", mem.available / 1e9)

        # GPU
        if torch.cuda.is_available():
            gpu_stats = get_memory_stats()
            COLLECTOR.record("gpu.memory_allocated_gb", gpu_stats.get("allocated_gb", 0))
            COLLECTOR.record("gpu.memory_reserved_gb", gpu_stats.get("reserved_gb", 0))

        # Disk
        disk = psutil.disk_usage("/")
        COLLECTOR.record("system.disk_usage_percent", disk.percent)


# Convenience functions
def record_metric(name: str, value: float, **labels):
    """Record a metric to the global collector."""
    COLLECTOR.record(name, value, labels)


def get_metrics() -> Dict[str, Dict[str, float]]:
    """Get all metric statistics."""
    return COLLECTOR.get_all_stats()


def export_metrics() -> str:
    """Export metrics in Prometheus format."""
    return COLLECTOR.export_prometheus()
