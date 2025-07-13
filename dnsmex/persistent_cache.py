"""Persistent caching decorator for expensive model computations.

Saves results to disk in _ignore directory to persist between notebook restarts.
"""

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any, Callable, List, Tuple
import pandas as pd


class PersistentCache:
    """Persistent file-based cache for model results."""

    def __init__(self, cache_dir: str = "_ignore/persistent_cache"):
        # Find project root by looking for setup.py or pyproject.toml
        current_dir = Path.cwd()
        project_root = current_dir

        # Walk up the directory tree to find project root
        while project_root != project_root.parent:
            if (project_root / "setup.py").exists() or (
                project_root / "pyproject.toml"
            ).exists():
                break
            project_root = project_root.parent

        # Use absolute path from project root
        if cache_dir.startswith("_ignore/"):
            self.cache_dir = project_root / cache_dir
        else:
            self.cache_dir = Path(cache_dir)

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_key(
        self, antibody_group: str, model_name: str, sequences: List[Tuple[str, str]]
    ) -> str:
        """Generate cache key from antibody group, model name, and sequences."""
        # Hash the sequences to create a unique identifier
        seq_str = "".join(f"{h}|{l}" for h, l in sequences)
        combined = f"{antibody_group}_{model_name}_{seq_str}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get cache file path."""
        return self.cache_dir / f"{cache_key}.pkl"

    def get(
        self, antibody_group: str, model_name: str, sequences: List[Tuple[str, str]]
    ) -> List[float]:
        """Get cached results if available."""
        cache_key = self._get_cache_key(antibody_group, model_name, sequences)
        cache_path = self._get_cache_path(cache_key)

        if cache_path.exists():
            try:
                with open(cache_path, "rb") as f:
                    results = pickle.load(f)
                print(
                    f"✓ Loaded {len(results)} cached {model_name} results for {antibody_group}"
                )
                return results
            except Exception as e:
                print(f"Warning: Failed to load cache {cache_path}: {e}")
                return None
        return None

    def set(
        self,
        antibody_group: str,
        model_name: str,
        sequences: List[Tuple[str, str]],
        results: List[float],
    ):
        """Save results to cache."""
        cache_key = self._get_cache_key(antibody_group, model_name, sequences)
        cache_path = self._get_cache_path(cache_key)

        try:
            with open(cache_path, "wb") as f:
                pickle.dump(results, f)
            print(f"✓ Cached {len(results)} {model_name} results for {antibody_group}")
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_path}: {e}")

    def clear(self):
        """Clear all cached results."""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        print(f"✓ Cleared cache directory: {self.cache_dir}")

    def info(self):
        """Print cache information."""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        print(f"Cache directory: {self.cache_dir}")
        print(f"Files: {len(cache_files)}")
        print(f"Total size: {total_size / (1024 * 1024):.2f} MB")


def cached_model_wrapper(model_class, model_name: str, cache: PersistentCache = None):
    """Decorator to add persistent caching to any model class.

    Args:
        model_class: The model class to wrap
        model_name: Name for cache identification
        cache: Optional cache instance (creates default if None)

    Returns:
        Wrapped class with caching capabilities
    """
    if cache is None:
        cache = PersistentCache()

    class CachedModelWrapper:
        def __init__(self, *args, **kwargs):
            self.model = model_class(*args, **kwargs)
            self.model_name = model_name
            self.cache = cache

        def evaluate_antibodies(
            self, sequences: List[Tuple[str, str]], antibody_group: str = "default"
        ) -> List[float]:
            """Evaluate antibody sequences with persistent caching."""
            # Try to get from cache
            cached_results = self.cache.get(antibody_group, self.model_name, sequences)
            if cached_results is not None:
                return cached_results

            # Compute results using the wrapped model
            print(
                f"⏳ Computing {self.model_name} for {len(sequences)} sequences in {antibody_group}..."
            )

            # Call the appropriate evaluation method on the wrapped model
            if hasattr(self.model, "evaluate_antibodies"):
                results = self.model.evaluate_antibodies(sequences)
            elif hasattr(self.model, "evaluate_sequences"):
                results = self.model.evaluate_sequences(sequences)
            else:
                raise AttributeError(
                    f"Model {model_class} must have evaluate_antibodies or evaluate_sequences method"
                )

            # Cache results
            self.cache.set(antibody_group, self.model_name, sequences, results)

            return results

        def __getattr__(self, name):
            """Delegate other attributes to the wrapped model."""
            return getattr(self.model, name)

    return CachedModelWrapper


# Global cache instance
_global_cache = PersistentCache()


def get_global_cache() -> PersistentCache:
    """Get the global cache instance."""
    return _global_cache
