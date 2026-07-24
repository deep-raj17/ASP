"""Immutable append-only research registries."""

from .sqlite import SQLiteRegistry
from .types import RegistryRecord

__all__ = ["RegistryRecord", "SQLiteRegistry"]
