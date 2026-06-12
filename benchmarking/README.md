# Garfield Spatial Scalability Benchmarking

This directory contains comprehensive benchmarking scripts to evaluate Garfield's performance on large-scale spatial transcriptomics datasets, addressing the reviewer's request for scalability analysis.

## Overview

The benchmarking suite evaluates four major tasks:

1. **Dimension Reduction (PCA)** - Feature dimensionality reduction using PCA
2. **Graph Construction** - Spatial neighbor graph construction (KNN, Radius, mu_std, Squidpy)
3. **Model Training** - GNN-based model training and embedding generation
4. **Label Transfer (Mapping)** - Weighted KNN-based label transfer from reference to query

## Files

- `benchmark_utils.py` - Core utilities for memory tracking and timing
- `benchmark_spatial_scalability.py` - Main benchmarking script
- `plot_benchmark_results.py` - Visualization script for results
- `run_benchmark.sh` - Shell script to run the complete benchmark pipeline
- `README.md` - This file

## Requirements

```bash
pip install psutil GPUtil matplotlib seaborn
```

## Quick Start

### Option 1: Run Complete Benchmark (Recommended)

```bash
# Run on CPU
bash run_benchmark.sh

# Run on GPU (device 0)
bash run_benchmark.sh --device-id 0

# Run with custom dataset sizes
bash run_benchmark.sh --dataset-sizes 5000 10000 25000
```

### Option 2: Run Benchmark Manually

```bash
# Basic usage (tests 5k, 10k, 25k, 50k, 100k cells)
python benchmark_spatial_scalability.py

# Custom dataset sizes
python benchmark_spatial_scalability.py \
    --dataset-sizes 5000 10000 25000 50000 100000

# Test all graph construction methods
python benchmark_spatial_scalability.py --test-all-methods

# Use specific GPU device
python benchmark_spatial_scalability.py --device-id 1

# Specify output directory
python benchmark_spatial_scalability.py \
    --output-dir ./my_benchmark_results

# Adjust training epochs (default: 20)
python benchmark_spatial_scalability.py --n-epochs 50
```

### Option 3: Generate Plots from Existing Results

```bash
# Generate plots from benchmark results
python plot_benchmark_results.py \
    --results-file ./benchmark_results/benchmark_results.csv \
    --output-dir ./benchmark_results/plots
```

## Output

The benchmark generates the following outputs:

### Data Files

- `benchmark_results/benchmark_results.json` - Raw results in JSON format
- `benchmark_results/benchmark_results.csv` - Results in CSV format
- `benchmark_results/plots/summary_table.csv` - Summary statistics table

### Plots

- `runtime_by_task.png` - Runtime vs dataset size for each task (2×2 grid)
- `memory_by_task.png` - Memory consumption vs dataset size for each task (2×2 grid)
- `combined_overview.png` - Comprehensive overview with all tasks and efficiency metrics

## Benchmark Metrics

For each task and dataset size, the following metrics are recorded:

- **Runtime** (seconds) - Wall-clock execution time
- **Peak CPU Memory** (MB) - Maximum CPU memory usage
- **Peak GPU Memory** (MB) - Maximum GPU memory usage (if available)
- **Number of cells/spots** - Dataset size
- **Task-specific parameters** (e.g., number of edges, latent dimension, etc.)

## Customization

### Custom Dataset Sizes

```python
python benchmark_spatial_scalability.py \
    --dataset-sizes 1000 5000 10000 20000 50000 100000
```

### Custom Graph Construction Methods

By default, only KNN is tested. To test all methods:

```python
python benchmark_spatial_scalability.py --test-all-methods
```

This will benchmark:
- KNN (K-Nearest Neighbors)
- Radius (Radius-based neighbors)
- mu_std (Mean + Std based adaptive threshold)
- Squidpy (if installed, for datasets ≤50k cells)

### Custom Training Parameters

```python
python benchmark_spatial_scalability.py \
    --n-epochs 100 \
    --device-id 0
```

## Performance Optimizations

The benchmarking suite includes optimizations to Garfield's spatial graph construction:

### Optimized `mu_std` Method

**Before:**
- Used full pairwise distance matrix: O(n²) complexity
- Memory: O(n²) for distance matrix storage

**After:**
- Uses KNN for initial neighbor finding: O(n·k·log(n)) complexity
- Memory: O(n·k) for k-nearest neighbors storage
- **Result:** ~10-100× speedup for large datasets (>50k cells)

## Example Output

```
================================================================================
GARFIELD SPATIAL SCALABILITY BENCHMARK
================================================================================
Dataset sizes: [5000, 10000, 25000, 50000, 100000]
Output directory: ./benchmark_results
Device ID: 0
================================================================================

################################################################################
# BENCHMARKING: 5,000 cells
################################################################################

Generating synthetic spatial data with 5,000 cells...
Data shape: (5000, 2000)

--------------------------------------------------------------------------------
TASK 1: Dimension Reduction (PCA)
--------------------------------------------------------------------------------
============================================================
Starting benchmark: Dimension_Reduction_n5000
============================================================

------------------------------------------------------------
Benchmark completed: Dimension_Reduction_n5000
Runtime: 2.34 seconds
Peak CPU memory: 145.23 MB
Peak GPU memory: 0.00 MB
============================================================

...
```

## Interpreting Results

### Runtime Scalability

- **Linear scaling:** Runtime increases proportionally with dataset size (ideal)
- **Quadratic scaling:** Runtime increases with square of dataset size (needs optimization)

### Memory Consumption

- **CPU Memory:** Main memory usage on CPU
- **GPU Memory:** VRAM usage on GPU (for model training)

### Graph Construction Methods Comparison

- **KNN:** Balanced speed and connectivity
- **Radius:** Fixed-radius neighborhoods (can vary in size)
- **mu_std:** Adaptive thresholding (may find more neighbors)
- **Squidpy:** Reference implementation for comparison

## Troubleshooting

### Out of Memory Errors

If you encounter OOM errors:

1. **Reduce dataset sizes:**
   ```bash
   python benchmark_spatial_scalability.py --dataset-sizes 5000 10000 25000
   ```

2. **Reduce batch size** (edit script):
   ```python
   batch_size=min(128, n_cells // 20)  # Instead of 256
   ```

3. **Skip model training** (automatically skipped for >50k cells)

### GPU Not Detected

If GPU is not detected but available:

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Verify device ID
nvidia-smi
```

### Slow Graph Construction

For very large datasets (>100k), consider:
- Using KNN method (fastest)
- Reducing number of neighbors (default: 15)
- Increasing available CPU cores

## Citation

If you use these benchmarking scripts in your research, please cite:

```bibtex
@article{zhou2025graph,
  title={Graph-based Contrastive Learning Enables Unified Integration and Niche Transfer Across Single-Cell and Spatial Multi-Omics},
  author={Zhou, Weige and Fan, Xueying and Li, Lanxiang and Zheng, Jianrong and Liu, Xiaodong and Jin, Wenfei and Tian, Luyi},
  journal={bioRxiv},
  year={2025},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Contact

For issues or questions about benchmarking:
- GitHub Issues: https://github.com/zhou-1314/Garfield/issues
- Email: zhouwg1314@gmail.com
