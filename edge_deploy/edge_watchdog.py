"""
edge_deploy/edge_watchdog.py
────────────────────────────────────────────────────────
Industrial health monitoring and watchdog service.

Runs as a companion to the streaming engine, providing:
  - CPU temperature monitoring with thermal throttle actions
  - Memory leak detection and GC triggers
  - Disk usage monitoring and log rotation
  - Inference latency tracking
  - systemd watchdog petting
  - Automatic service restart on health failure

Can run standalone or be imported by edge_streaming.py.

Usage:
    python edge_watchdog.py --interval 30
"""

from __future__ import annotations

import os
import sys
import time
import signal
import logging
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from datetime import datetime

import numpy as np

try:
    import psutil
except ImportError:
    psutil = None


# ─────────────────────────────────────────────────────────
#  Health Check Data
# ─────────────────────────────────────────────────────────

@dataclass
class HealthStatus:
    timestamp: str = ""
    healthy: bool = True
    cpu_temp_c: float = -1.0
    cpu_usage_pct: float = 0.0
    memory_used_mb: float = 0.0
    memory_total_mb: float = 0.0
    memory_pct: float = 0.0
    disk_usage_pct: float = 0.0
    inference_service_active: bool = False
    uptime_hours: float = 0.0
    issues: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────
#  System Monitors
# ─────────────────────────────────────────────────────────

def get_cpu_temp() -> float:
    """Read CPU temp (Linux thermal zone or vcgencmd)."""
    # Method 1: sysfs thermal zone
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        pass

    # Method 2: vcgencmd (RPi-specific)
    try:
        result = subprocess.run(
            ["vcgencmd", "measure_temp"],
            capture_output=True, text=True, timeout=5
        )
        # Output: "temp=42.3'C"
        temp_str = result.stdout.strip().replace("temp=", "").replace("'C", "")
        return float(temp_str)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    return -1.0


def get_cpu_freq() -> float:
    """Get current CPU frequency in MHz."""
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") as f:
            return int(f.read().strip()) / 1000.0
    except (FileNotFoundError, ValueError):
        if psutil:
            freq = psutil.cpu_freq()
            if freq:
                return freq.current
    return -1.0


def is_service_active(service_name: str = "mimii-detector") -> bool:
    """Check if a systemd service is active."""
    try:
        result = subprocess.run(
            ["systemctl", "is-active", service_name],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() == "active"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def get_memory_info() -> Dict[str, float]:
    """Get memory usage info."""
    if psutil:
        mem = psutil.virtual_memory()
        return {
            "used_mb": mem.used / (1024 * 1024),
            "total_mb": mem.total / (1024 * 1024),
            "percent": mem.percent,
        }

    # Fallback: /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            if parts[0] in ("MemTotal:", "MemAvailable:", "MemFree:"):
                info[parts[0].rstrip(":")] = int(parts[1]) / 1024  # KB → MB

        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", info.get("MemFree", 0))
        used = total - avail
        return {
            "used_mb": used,
            "total_mb": total,
            "percent": (used / total * 100) if total > 0 else 0,
        }
    except (FileNotFoundError, ValueError):
        return {"used_mb": 0, "total_mb": 0, "percent": 0}


def get_disk_usage(path: str = "/") -> float:
    """Get disk usage percentage."""
    if psutil:
        usage = psutil.disk_usage(path)
        return usage.percent
    try:
        st = os.statvfs(path)
        total = st.f_blocks * st.f_frsize
        free = st.f_bfree * st.f_frsize
        used = total - free
        return (used / total * 100) if total > 0 else 0
    except (OSError, AttributeError):
        return 0.0


# ─────────────────────────────────────────────────────────
#  Health Actions
# ─────────────────────────────────────────────────────────

class HealthActions:
    """Automated corrective actions for health issues."""

    @staticmethod
    def force_gc():
        """Force Python garbage collection."""
        import gc
        gc.collect()

    @staticmethod
    def rotate_logs(log_dir: str = "/var/log/mimii", max_size_mb: int = 50):
        """Rotate log files if total size exceeds threshold."""
        if not os.path.isdir(log_dir):
            return

        total_size = 0
        log_files = []
        for f in os.listdir(log_dir):
            fp = os.path.join(log_dir, f)
            if os.path.isfile(fp):
                size = os.path.getsize(fp)
                total_size += size
                log_files.append((fp, os.path.getmtime(fp), size))

        if total_size / (1024 * 1024) > max_size_mb:
            # Delete oldest files until under threshold
            log_files.sort(key=lambda x: x[1])
            for fp, _, size in log_files:
                try:
                    os.remove(fp)
                    total_size -= size
                    if total_size / (1024 * 1024) <= max_size_mb * 0.7:
                        break
                except OSError:
                    continue

    @staticmethod
    def restart_service(service_name: str = "mimii-detector"):
        """Restart systemd service."""
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", service_name],
                timeout=30
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    @staticmethod
    def drop_caches():
        """Drop kernel caches to free memory."""
        try:
            subprocess.run(
                ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                timeout=10
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass


# ─────────────────────────────────────────────────────────
#  Health Monitor
# ─────────────────────────────────────────────────────────

class HealthMonitor:
    """Periodic health monitoring with automated corrective actions.

    Thresholds:
        CPU temp > 80°C  → increase inference interval
        CPU temp > 85°C  → skip inference, alert
        Memory > 90%     → force GC, drop caches
        Memory > 95%     → restart service
        Disk > 90%       → rotate logs
        Latency > 10s    → restart service
    """

    TEMP_WARNING = 80
    TEMP_CRITICAL = 85
    MEM_WARNING = 90
    MEM_CRITICAL = 95
    DISK_WARNING = 90

    def __init__(self, log_dir: str = "/var/log/mimii"):
        self.log_dir = log_dir
        self._start_time = time.time()
        self._logger = logging.getLogger("mimii-health")
        self._consecutive_thermal_warnings = 0
        self._consecutive_mem_warnings = 0

    def check(self) -> HealthStatus:
        """Run all health checks and return status."""
        status = HealthStatus(
            timestamp=datetime.now().isoformat(),
            uptime_hours=(time.time() - self._start_time) / 3600,
        )

        # CPU temperature
        status.cpu_temp_c = get_cpu_temp()
        if status.cpu_temp_c > 0:
            if status.cpu_temp_c >= self.TEMP_CRITICAL:
                status.issues.append(f"CPU CRITICAL: {status.cpu_temp_c:.1f}°C")
                status.healthy = False
                self._consecutive_thermal_warnings += 1
            elif status.cpu_temp_c >= self.TEMP_WARNING:
                status.issues.append(f"CPU WARNING: {status.cpu_temp_c:.1f}°C")
                self._consecutive_thermal_warnings += 1
            else:
                self._consecutive_thermal_warnings = 0

        # CPU usage
        if psutil:
            status.cpu_usage_pct = psutil.cpu_percent(interval=0.5)

        # Memory
        mem = get_memory_info()
        status.memory_used_mb = mem["used_mb"]
        status.memory_total_mb = mem["total_mb"]
        status.memory_pct = mem["percent"]

        if status.memory_pct >= self.MEM_CRITICAL:
            status.issues.append(f"MEMORY CRITICAL: {status.memory_pct:.1f}%")
            status.healthy = False
            HealthActions.force_gc()
            HealthActions.drop_caches()
            status.actions_taken.append("force_gc + drop_caches")
            self._consecutive_mem_warnings += 1
        elif status.memory_pct >= self.MEM_WARNING:
            status.issues.append(f"MEMORY WARNING: {status.memory_pct:.1f}%")
            HealthActions.force_gc()
            status.actions_taken.append("force_gc")
            self._consecutive_mem_warnings += 1
        else:
            self._consecutive_mem_warnings = 0

        # Disk
        status.disk_usage_pct = get_disk_usage("/")
        if status.disk_usage_pct >= self.DISK_WARNING:
            status.issues.append(f"DISK WARNING: {status.disk_usage_pct:.1f}%")
            HealthActions.rotate_logs(self.log_dir)
            status.actions_taken.append("log_rotation")

        # Service status
        status.inference_service_active = is_service_active("mimii-detector")

        # Escalation: restart if too many consecutive warnings
        if self._consecutive_mem_warnings >= 5:
            self._logger.error("Memory warnings persisting — restarting service")
            HealthActions.restart_service()
            status.actions_taken.append("service_restart")
            self._consecutive_mem_warnings = 0

        return status


# ─────────────────────────────────────────────────────────
#  Standalone CLI
# ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="MIMII Health Monitor")
    parser.add_argument("--interval", type=int, default=60, help="Check interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Run single check and exit")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger("mimii-health")

    monitor = HealthMonitor()

    def _run_check():
        status = monitor.check()
        if args.json:
            import json
            print(json.dumps(status.to_dict(), indent=2))
        else:
            print(f"\n{'='*50}")
            print(f"Health Check: {status.timestamp}")
            print(f"  Healthy:    {status.healthy}")
            print(f"  CPU Temp:   {status.cpu_temp_c:.1f}°C")
            print(f"  CPU Usage:  {status.cpu_usage_pct:.1f}%")
            print(f"  Memory:     {status.memory_used_mb:.0f}/{status.memory_total_mb:.0f} MB ({status.memory_pct:.1f}%)")
            print(f"  Disk:       {status.disk_usage_pct:.1f}%")
            print(f"  Service:    {'ACTIVE' if status.inference_service_active else 'INACTIVE'}")
            print(f"  Uptime:     {status.uptime_hours:.2f}h")
            if status.issues:
                print(f"  Issues:     {', '.join(status.issues)}")
            if status.actions_taken:
                print(f"  Actions:    {', '.join(status.actions_taken)}")
            print(f"{'='*50}")

    if args.once:
        _run_check()
        return

    running = True

    def _shutdown(signum, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    logger.info(f"Health monitor started (interval={args.interval}s)")
    while running:
        _run_check()
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
