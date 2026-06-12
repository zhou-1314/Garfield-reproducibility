"""
Benchmarking utilities for measuring runtime and memory consumption of Garfield tasks.

This module provides utilities for:
- Measuring CPU and GPU memory usage
- Measuring execution time
- Generating synthetic spatial datasets
- Recording and saving benchmark results
"""

import time
import psutil
import os
import gc
import numpy as np
import pandas as pd
import anndata as ad
from typing import Dict, Callable, Any, Optional, Tuple
from contextlib import contextmanager
import json
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False


class MemoryTracker:
    """Track CPU and GPU memory usage during execution."""

    def __init__(self, device_id: int = 0):
        """
        Initialize memory tracker.

        Parameters
        ----------
        device_id : int
            GPU device ID to monitor (default: 0)
        """
        self.device_id = device_id
        self.process = psutil.Process(os.getpid())
        self.peak_cpu_memory = 0
        self.peak_gpu_memory = 0
        self.start_cpu_memory = 0
        self.start_gpu_memory = 0

    def get_cpu_memory_mb(self) -> float:
        """Get current CPU memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def get_gpu_memory_mb(self) -> float:
        """Get current GPU memory usage in MB."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.synchronize()
            return torch.cuda.max_memory_allocated(self.device_id) / 1024 / 1024
        elif GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if len(gpus) > self.device_id:
                    return gpus[self.device_id].memoryUsed
            except Exception:
                pass
        return 0.0

    def reset(self):
        """Reset memory tracking."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device_id)
            torch.cuda.empty_cache()
        gc.collect()

        self.start_cpu_memory = self.get_cpu_memory_mb()
        self.start_gpu_memory = self.get_gpu_memory_mb()
        self.peak_cpu_memory = self.start_cpu_memory
        self.peak_gpu_memory = self.start_gpu_memory

    def update_peak(self):
        """Update peak memory usage."""
        current_cpu = self.get_cpu_memory_mb()
        current_gpu = self.get_gpu_memory_mb()

        self.peak_cpu_memory = max(self.peak_cpu_memory, current_cpu)
        self.peak_gpu_memory = max(self.peak_gpu_memory, current_gpu)

    def get_peak_memory(self) -> Tuple[float, float]:
        """
        Get peak memory usage since last reset.

        Returns
        -------
        Tuple[float, float]
            (peak_cpu_memory_mb, peak_gpu_memory_mb)
        """
        self.update_peak()
        return (
            self.peak_cpu_memory - self.start_cpu_memory,
            self.peak_gpu_memory - self.start_gpu_memory,
        )


@contextmanager
def benchmark_task(
    task_name: str,
    device_id: int = 0,
    verbose: bool = True
):
    """
    Context manager for benchmarking a task.

    Parameters
    ----------
    task_name : str
        Name of the task being benchmarked
    device_id : int
        GPU device ID to monitor
    verbose : bool
        Whether to print progress information

    Yields
    ------
    dict
        Dictionary containing benchmark results (will be populated on exit)

    Examples
    --------
    >>> with benchmark_task("my_task") as results:
    ...     # perform task
    ...     pass
    >>> print(results['runtime_seconds'])
    """
    memory_tracker = MemoryTracker(device_id=device_id)
    results = {
        'task_name': task_name,
        'start_time': datetime.now().isoformat(),
    }

    # Reset memory tracking
    memory_tracker.reset()
    start_time = time.time()

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Starting benchmark: {task_name}")
        print(f"{'=' * 60}")

    try:
        yield results
    finally:
        # Record runtime
        runtime = time.time() - start_time
        results['runtime_seconds'] = runtime

        # Record peak memory
        peak_cpu_mb, peak_gpu_mb = memory_tracker.get_peak_memory()
        results['peak_cpu_memory_mb'] = peak_cpu_mb
        results['peak_gpu_memory_mb'] = peak_gpu_mb

        if verbose:
            print(f"\n{'-' * 60}")
            print(f"Benchmark completed: {task_name}")
            print(f"Runtime: {runtime:.2f} seconds")
            print(f"Peak CPU memory: {peak_cpu_mb:.2f} MB")
            if peak_gpu_mb > 0:
                print(f"Peak GPU memory: {peak_gpu_mb:.2f} MB")
            print(f"{'=' * 60}\n")


def generate_synthetic_spatial_data(
    n_cells: int,
    n_genes: int = 2000,
    spatial_dim: Tuple[int, int] = (100, 100),
    n_cell_types: int = 5,
    random_state: int = 42
) -> ad.AnnData:
    """
    Generate synthetic spatial transcriptomics data for benchmarking.

    Parameters
    ----------
    n_cells : int
        Number of cells/spots to generate
    n_genes : int
        Number of genes (features)
    spatial_dim : Tuple[int, int]
        Spatial dimensions (width, height) for random coordinate generation
    n_cell_types : int
        Number of distinct cell types
    random_state : int
        Random seed for reproducibility

    Returns
    -------
    ad.AnnData
        Synthetic spatial AnnData object with:
        - X: Count matrix (n_cells x n_genes)
        - obsm['spatial']: Spatial coordinates
        - obs['cell_type']: Cell type labels
        - obs['batch']: Batch labels (all set to 'batch_0')
    """
    np.random.seed(random_state)

    # Generate count matrix (negative binomial distribution)
    counts = np.random.negative_binomial(5, 0.3, size=(n_cells, n_genes))

    # Generate spatial coordinates (random uniform distribution)
    spatial_coords = np.random.uniform(
        low=0,
        high=spatial_dim,
        size=(n_cells, 2)
    )

    # Generate cell type labels
    cell_types = np.random.choice(
        [f"CellType_{i}" for i in range(n_cell_types)],
        size=n_cells
    )

    # Create AnnData object
    adata = ad.AnnData(
        X=counts.astype(np.float32),
        obs=pd.DataFrame({
            'cell_type': cell_types,
            'batch': 'batch_0'
        }),
        var=pd.DataFrame(index=[f"Gene_{i}" for i in range(n_genes)])
    )

    # Add spatial coordinates
    adata.obsm['spatial'] = spatial_coords

    return adata


def save_benchmark_results(
    results: Dict[str, Any],
    output_file: str,
    append: bool = True
):
    """
    Save benchmark results to JSON file.

    Parameters
    ----------
    results : Dict[str, Any]
        Dictionary containing benchmark results
    output_file : str
        Path to output JSON file
    append : bool
        If True, append to existing file; otherwise overwrite
    """
    # Load existing results if appending
    all_results = []
    if append and os.path.exists(output_file):
        try:
            with open(output_file, 'r') as f:
                all_results = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")
            all_results = []

    # Append new results
    all_results.append(results)

    # Save to file
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"Results saved to: {output_file}")


def load_benchmark_results(input_file: str) -> pd.DataFrame:
    """
    Load benchmark results from JSON file into DataFrame.

    Parameters
    ----------
    input_file : str
        Path to input JSON file

    Returns
    -------
    pd.DataFrame
        DataFrame containing benchmark results
    """
    with open(input_file, 'r') as f:
        results = json.load(f)

    return pd.DataFrame(results)


def benchmark_function(
    func: Callable,
    func_args: tuple = (),
    func_kwargs: dict = None,
    task_name: str = None,
    device_id: int = 0,
    n_repeats: int = 1,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Benchmark a function with multiple repeats.

    Parameters
    ----------
    func : Callable
        Function to benchmark
    func_args : tuple
        Positional arguments for the function
    func_kwargs : dict
        Keyword arguments for the function
    task_name : str
        Name of the task (defaults to function name)
    device_id : int
        GPU device ID to monitor
    n_repeats : int
        Number of times to repeat the function call
    verbose : bool
        Whether to print progress

    Returns
    -------
    Dict[str, Any]
        Aggregated benchmark results with mean and std of runtime and memory
    """
    if func_kwargs is None:
        func_kwargs = {}

    if task_name is None:
        task_name = func.__name__

    all_results = []

    for i in range(n_repeats):
        if verbose and n_repeats > 1:
            print(f"\nRepeat {i + 1}/{n_repeats}")

        with benchmark_task(
            f"{task_name}_repeat_{i}",
            device_id=device_id,
            verbose=verbose
        ) as results:
            _ = func(*func_args, **func_kwargs)

        all_results.append(results)

    # Aggregate results
    aggregated = {
        'task_name': task_name,
        'n_repeats': n_repeats,
        'runtime_seconds_mean': np.mean([r['runtime_seconds'] for r in all_results]),
        'runtime_seconds_std': np.std([r['runtime_seconds'] for r in all_results]),
        'peak_cpu_memory_mb_mean': np.mean([r['peak_cpu_memory_mb'] for r in all_results]),
        'peak_cpu_memory_mb_std': np.std([r['peak_cpu_memory_mb'] for r in all_results]),
        'peak_gpu_memory_mb_mean': np.mean([r['peak_gpu_memory_mb'] for r in all_results]),
        'peak_gpu_memory_mb_std': np.std([r['peak_gpu_memory_mb'] for r in all_results]),
    }

    return aggregated
