"""
utils/data_pipeline.py
────────────────────────────────────────────────────────
Optimized data loading pipeline with caching and prefetching.

Features:
  - In-memory caching for small datasets
  - Prefetch buffer for streaming
  - Multi-threaded audio preprocessing
  - Feature caching to disk
"""

from __future__ import annotations

import os
import hashlib
import pickle
from pathlib import Path
from typing import Optional, List, Dict, Callable, Iterator
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import torch
import torchaudio
from torch.utils.data import IterableDataset, get_worker_info

from config import Config
from utils.audio_utils import AudioProcessor, pad_or_trim
from utils.validation import validate_audio_file, logger


class AudioCache:
    """Thread-safe LRU cache for preprocessed audio features."""

    def __init__(self, max_size: int = 1000, cache_dir: Optional[str] = None):
        self.max_size = max_size
        self.cache_dir = cache_dir
        self._memory_cache: Dict[str, torch.Tensor] = {}
        self._access_order: List[str] = []
        self._lock = threading.RLock()

        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _get_key(self, path: str, transform_hash: str = "") -> str:
        """Generate cache key from file path and transform parameters."""
        file_stat = os.stat(path)
        unique = f"{path}:{file_stat.st_size}:{file_stat.st_mtime}:{transform_hash}"
        return hashlib.md5(unique.encode()).hexdigest()

    def get(self, path: str, transform_hash: str = "") -> Optional[torch.Tensor]:
        """Get item from cache."""
        key = self._get_key(path, transform_hash)

        with self._lock:
            # Check memory cache
            if key in self._memory_cache:
                # Update access order
                self._access_order.remove(key)
                self._access_order.append(key)
                return self._memory_cache[key]

        # Check disk cache
        if self.cache_dir:
            disk_path = os.path.join(self.cache_dir, f"{key}.pt")
            if os.path.exists(disk_path):
                try:
                    tensor = torch.load(disk_path, map_location="cpu", weights_only=True)
                    self._put_memory(key, tensor)
                    return tensor
                except Exception as e:
                    logger.warning(f"Failed to load from disk cache: {e}")

        return None

    def put(self, path: str, tensor: torch.Tensor, transform_hash: str = ""):
        """Put item in cache."""
        key = self._get_key(path, transform_hash)

        with self._lock:
            self._put_memory(key, tensor)

        # Also save to disk cache
        if self.cache_dir:
            disk_path = os.path.join(self.cache_dir, f"{key}.pt")
            try:
                torch.save(tensor, disk_path)
            except Exception as e:
                logger.warning(f"Failed to save to disk cache: {e}")

    def _put_memory(self, key: str, tensor: torch.Tensor):
        """Put item in memory cache with LRU eviction."""
        with self._lock:
            if key in self._memory_cache:
                self._access_order.remove(key)

            # Evict if necessary
            while len(self._memory_cache) >= self.max_size:
                oldest = self._access_order.pop(0)
                del self._memory_cache[oldest]

            self._memory_cache[key] = tensor
            self._access_order.append(key)

    def clear(self):
        """Clear all caches."""
        with self._lock:
            self._memory_cache.clear()
            self._access_order.clear()


class PrefetchBuffer:
    """Async prefetch buffer for data loading."""

    def __init__(self, iterator: Iterator, buffer_size: int = 4):
        self.iterator = iterator
        self.buffer_size = buffer_size
        self._buffer: List = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._lock = threading.Lock()
        self._stopped = False

        # Start first prefetch
        self._prefetch_next()

    def _prefetch_next(self):
        """Start prefetching next item."""
        if not self._stopped:
            self._future = self._executor.submit(self._next_item)

    def _next_item(self):
        """Get next item from iterator."""
        try:
            return next(self.iterator)
        except StopIteration:
            return None

    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            if self._future is None:
                raise StopIteration

            # Wait for current prefetch
            item = self._future.result()
            self._future = None

            if item is None:
                self._stopped = True
                raise StopIteration

            # Start next prefetch
            self._prefetch_next()

            return item

    def close(self):
        """Shutdown the prefetch buffer."""
        self._stopped = True
        if self._future:
            self._future.cancel()
        self._executor.shutdown(wait=False)


class OptimizedAudioDataset(IterableDataset):
    """
    Optimized audio dataset with caching and multi-threaded loading.
    """

    def __init__(
        self,
        cfg: Config,
        file_paths: List[str],
        labels: List[int],
        cache: Optional[AudioCache] = None,
        num_threads: int = 4,
        augment: bool = False,
    ):
        self.cfg = cfg
        self.file_paths = file_paths
        self.labels = labels
        self.cache = cache or AudioCache()
        self.num_threads = num_threads
        self.augment = augment

        self.processor = AudioProcessor(cfg.data)
        self.target_len = int(cfg.data.sample_rate * cfg.data.audio_duration_sec)

        # Pre-validate files
        self._valid_indices = self._validate_files()

    def _validate_files(self) -> List[int]:
        """Pre-validate all files and return indices of valid ones."""
        valid = []
        for i, path in enumerate(self.file_paths):
            result = validate_audio_file(path)
            if result.valid:
                valid.append(i)
            else:
                logger.warning(f"Skipping invalid file: {path} - {result.error_message}")
        return valid

    def _load_and_process(self, idx: int) -> Optional[Dict]:
        """Load and process a single audio file."""
        if idx not in self._valid_indices:
            return None

        path = self.file_paths[idx]
        label = self.labels[idx]

        # Check cache
        cached = self.cache.get(path, transform_hash=f"aug_{self.augment}")
        if cached is not None:
            return {
                "mel": cached,
                "label": torch.tensor(label, dtype=torch.float32),
                "path": path,
            }

        # Load audio
        try:
            waveform, sr = torchaudio.load(path)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != self.cfg.data.sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sr, new_freq=self.cfg.data.sample_rate
                )

            # Normalize
            peak = waveform.abs().max()
            if peak > 1e-6:
                waveform = waveform / peak

            # Pad/trim
            waveform = pad_or_trim(waveform, self.target_len)

            # Extract features
            mel, _, _ = self.processor(waveform, augment=self.augment)

            # Cache result
            self.cache.put(path, mel, transform_hash=f"aug_{self.augment}")

            return {
                "mel": mel,
                "label": torch.tensor(label, dtype=torch.float32),
                "path": path,
            }

        except Exception as e:
            logger.error(f"Failed to process {path}: {e}")
            return None

    def __iter__(self):
        """Iterate over the dataset with multi-threaded loading."""
        worker_info = get_worker_info()

        if worker_info is None:
            # Single-process loading
            indices = self._valid_indices
        else:
            # Multi-process: split indices among workers
            per_worker = len(self._valid_indices) // worker_info.num_workers
            start = worker_info.id * per_worker
            end = start + per_worker if worker_info.id < worker_info.num_workers - 1 else len(self._valid_indices)
            indices = self._valid_indices[start:end]

        # Use thread pool for parallel loading within worker
        with ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = {executor.submit(self._load_and_process, idx): idx for idx in indices}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    yield result

    def __len__(self) -> int:
        return len(self._valid_indices)


class BatchCollator:
    """Custom collator for batching audio samples."""

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """Collate a batch of samples."""
        if not batch:
            return {}

        return {
            "mel": torch.stack([b["mel"] for b in batch]),
            "label": torch.stack([b["label"] for b in batch]),
            "path": [b["path"] for b in batch],
        }


def create_optimized_dataloader(
    cfg: Config,
    file_paths: List[str],
    labels: List[int],
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    cache_dir: Optional[str] = None,
    augment: bool = False,
) -> torch.utils.data.DataLoader:
    """
    Create an optimized DataLoader with caching and prefetching.

    Args:
        cfg: Configuration
        file_paths: List of audio file paths
        labels: List of labels
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of worker processes (default: auto)
        cache_dir: Directory for disk cache
        augment: Whether to apply augmentation

    Returns:
        Optimized DataLoader
    """
    if num_workers is None:
        num_workers = min(4, os.cpu_count() or 1)

    # Create cache
    cache = AudioCache(max_size=1000, cache_dir=cache_dir)

    # Create dataset
    dataset = OptimizedAudioDataset(
        cfg=cfg,
        file_paths=file_paths,
        labels=labels,
        cache=cache,
        num_threads=2,
        augment=augment,
    )

    # DataLoader with optimized settings
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # IterableDataset does not support shuffle
        num_workers=num_workers,
        collate_fn=BatchCollator(),
        pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    return loader
