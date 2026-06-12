#!/bin/bash

################################################################################
# Garfield Spatial Scalability Benchmark Runner
#
# This script runs the complete benchmarking pipeline:
# 1. Runs benchmarks on multiple dataset sizes
# 2. Generates visualization plots
# 3. Creates summary tables
#
# Usage:
#   bash run_benchmark.sh [options]
#
# Options:
#   --dataset-sizes    Dataset sizes to benchmark (default: 5000 10000 25000 50000 100000)
#   --output-dir       Output directory (default: ./benchmark_results)
#   --device-id        GPU device ID (default: 0)
#   --test-all-methods Test all graph construction methods (default: false)
#   --n-epochs         Number of training epochs (default: 20)
#   --help             Show this help message
################################################################################

set -e  # Exit on error

# Default parameters
DATASET_SIZES="5000 10000 25000 50000 100000"
OUTPUT_DIR="./benchmark_results"
DEVICE_ID=0
TEST_ALL_METHODS=""
N_EPOCHS=20

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset-sizes)
            DATASET_SIZES="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        --test-all-methods)
            TEST_ALL_METHODS="--test-all-methods"
            shift
            ;;
        --n-epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: bash run_benchmark.sh [options]"
            echo ""
            echo "Options:"
            echo "  --dataset-sizes SIZES     Dataset sizes to benchmark (space-separated)"
            echo "                           Default: 5000 10000 25000 50000 100000"
            echo "  --output-dir DIR         Output directory"
            echo "                           Default: ./benchmark_results"
            echo "  --device-id ID           GPU device ID"
            echo "                           Default: 0"
            echo "  --test-all-methods       Test all graph construction methods"
            echo "                           Default: false (KNN only)"
            echo "  --n-epochs N             Number of training epochs"
            echo "                           Default: 20"
            echo "  --help, -h               Show this help message"
            echo ""
            echo "Examples:"
            echo "  bash run_benchmark.sh"
            echo "  bash run_benchmark.sh --dataset-sizes \"5000 10000\""
            echo "  bash run_benchmark.sh --device-id 1 --test-all-methods"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print configuration
echo "================================================================================"
echo "GARFIELD SPATIAL SCALABILITY BENCHMARK RUNNER"
echo "================================================================================"
echo "Configuration:"
echo "  Dataset sizes: $DATASET_SIZES"
echo "  Output directory: $OUTPUT_DIR"
echo "  GPU device ID: $DEVICE_ID"
echo "  Test all methods: $([ -n "$TEST_ALL_METHODS" ] && echo "Yes" || echo "No")"
echo "  Training epochs: $N_EPOCHS"
echo "================================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.7+"
    exit 1
fi

# Check if required Python packages are available
echo "Checking dependencies..."
python -c "import numpy, pandas, matplotlib, seaborn, psutil, anndata, scanpy" 2>/dev/null || {
    echo "Error: Missing required Python packages."
    echo "Please install: numpy pandas matplotlib seaborn psutil anndata scanpy"
    echo ""
    echo "You can install them with:"
    echo "  pip install numpy pandas matplotlib seaborn psutil anndata scanpy"
    exit 1
}
echo "✓ All dependencies found"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/plots"

# Run benchmarks
echo "================================================================================"
echo "STEP 1: Running Benchmarks"
echo "================================================================================"
echo ""

python benchmark_spatial_scalability.py \
    --dataset-sizes $DATASET_SIZES \
    --output-dir "$OUTPUT_DIR" \
    --device-id "$DEVICE_ID" \
    $TEST_ALL_METHODS \
    --n-epochs "$N_EPOCHS"

BENCHMARK_EXIT_CODE=$?

if [ $BENCHMARK_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Benchmark failed with exit code $BENCHMARK_EXIT_CODE"
    echo "================================================================================"
    exit $BENCHMARK_EXIT_CODE
fi

echo ""
echo "✓ Benchmarks completed successfully"
echo ""

# Generate plots
echo "================================================================================"
echo "STEP 2: Generating Plots"
echo "================================================================================"
echo ""

python plot_benchmark_results.py \
    --results-file "$OUTPUT_DIR/benchmark_results.csv" \
    --output-dir "$OUTPUT_DIR/plots"

PLOT_EXIT_CODE=$?

if [ $PLOT_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Plot generation failed with exit code $PLOT_EXIT_CODE"
    echo "================================================================================"
    exit $PLOT_EXIT_CODE
fi

echo ""
echo "✓ Plots generated successfully"
echo ""

# Summary
echo "================================================================================"
echo "BENCHMARK COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to:"
echo "  • Raw data (JSON): $OUTPUT_DIR/benchmark_results.json"
echo "  • Tabular data (CSV): $OUTPUT_DIR/benchmark_results.csv"
echo "  • Summary table: $OUTPUT_DIR/plots/summary_table.csv"
echo ""
echo "Plots saved to $OUTPUT_DIR/plots/:"
echo "  • runtime_by_task.png - Runtime vs dataset size (2×2 grid)"
echo "  • memory_by_task.png - Memory consumption vs dataset size (2×2 grid)"
echo "  • combined_overview.png - Comprehensive overview with all metrics"
echo ""
echo "================================================================================"

# Display summary table if available
if [ -f "$OUTPUT_DIR/plots/summary_table.csv" ]; then
    echo ""
    echo "Summary Statistics:"
    echo "-------------------"
    python -c "import pandas as pd; df = pd.read_csv('$OUTPUT_DIR/plots/summary_table.csv'); print(df.to_string(index=False))"
    echo ""
fi

echo "================================================================================"
echo "You can view the plots and results in: $OUTPUT_DIR"
echo "================================================================================"
