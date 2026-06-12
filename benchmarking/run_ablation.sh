#!/bin/bash

################################################################################
# Garfield Ablation Study Runner
#
# This script runs comprehensive ablation studies for the denoised-graph branch
# and hyperparameter justification.
#
# Usage:
#   bash run_ablation.sh --data-path /path/to/data.h5ad [options]
#
# Required:
#   --data-path PATH       Path to input AnnData file (.h5ad)
#
# Options:
#   --output-dir DIR       Output directory (default: ./ablation_results)
#   --n-epochs N           Number of training epochs (default: 100)
#   --device-id ID         GPU device ID (default: 0)
#   --skip-augment         Skip augmentation type ablation
#   --skip-gnn-layer       Skip GNN layer ablation
#   --skip-svd-rank        Skip SVD rank ablation
#   --help, -h             Show this help message
################################################################################

set -e  # Exit on error

# Default parameters
DATA_PATH=""
OUTPUT_DIR="./ablation_results"
N_EPOCHS=100
DEVICE_ID=0
SKIP_AUGMENT=""
SKIP_GNN_LAYER=""
SKIP_SVD_RANK=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --n-epochs)
            N_EPOCHS="$2"
            shift 2
            ;;
        --device-id)
            DEVICE_ID="$2"
            shift 2
            ;;
        --skip-augment)
            SKIP_AUGMENT="--skip-augment"
            shift
            ;;
        --skip-gnn-layer)
            SKIP_GNN_LAYER="--skip-gnn-layer"
            shift
            ;;
        --skip-svd-rank)
            SKIP_SVD_RANK="--skip-svd-rank"
            shift
            ;;
        --help|-h)
            grep "^#" "$0" | grep -v "^#!/" | sed 's/^# //' | sed 's/^#//'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$DATA_PATH" ]; then
    echo "Error: --data-path is required"
    echo "Usage: bash run_ablation.sh --data-path /path/to/data.h5ad"
    echo "Use --help for more information"
    exit 1
fi

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "Error: Data file not found: $DATA_PATH"
    exit 1
fi

# Print configuration
echo "================================================================================"
echo "GARFIELD ABLATION STUDY RUNNER"
echo "================================================================================"
echo "Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Output directory: $OUTPUT_DIR"
echo "  Training epochs: $N_EPOCHS"
echo "  GPU device ID: $DEVICE_ID"
echo "  Skip augment ablation: $([ -n "$SKIP_AUGMENT" ] && echo "Yes" || echo "No")"
echo "  Skip GNN layer ablation: $([ -n "$SKIP_GNN_LAYER" ] && echo "Yes" || echo "No")"
echo "  Skip SVD rank ablation: $([ -n "$SKIP_SVD_RANK" ] && echo "Yes" || echo "No")"
echo "================================================================================"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.7+"
    exit 1
fi

# Check dependencies
echo "Checking dependencies..."
python -c "import numpy, pandas, matplotlib, seaborn, anndata, scanpy, sklearn" 2>/dev/null || {
    echo "Error: Missing required Python packages."
    echo "Please install: numpy pandas matplotlib seaborn anndata scanpy scikit-learn"
    exit 1
}

python -c "import Garfield" 2>/dev/null || {
    echo "Error: Garfield package not found."
    echo "Please install Garfield or run from the Garfield directory."
    exit 1
}
echo "✓ All dependencies found"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/plots"

# Run ablation studies
echo "================================================================================"
echo "STEP 1: Running Ablation Studies"
echo "================================================================================"
echo ""

python ablation_study.py \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --n-epochs "$N_EPOCHS" \
    --device-id "$DEVICE_ID" \
    $SKIP_AUGMENT \
    $SKIP_GNN_LAYER \
    $SKIP_SVD_RANK

ABLATION_EXIT_CODE=$?

if [ $ABLATION_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Ablation study failed with exit code $ABLATION_EXIT_CODE"
    echo "================================================================================"
    exit $ABLATION_EXIT_CODE
fi

echo ""
echo "✓ Ablation studies completed successfully"
echo ""

# Generate plots
echo "================================================================================"
echo "STEP 2: Generating Visualization Plots"
echo "================================================================================"
echo ""

python plot_ablation_results.py \
    --results-dir "$OUTPUT_DIR" \
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
echo "ABLATION STUDY COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to $OUTPUT_DIR/:"
echo "  • ablation_augment_type.csv - SVD vs Dropout comparison"
echo "  • ablation_gnn_layer.csv - GNN iteration analysis"
echo "  • ablation_svd_rank.csv - SVD rank parameter analysis"
echo ""
echo "Plots saved to $OUTPUT_DIR/plots/:"
echo "  • ablation_augment_type.png - SVD vs Dropout visualization"
echo "  • ablation_gnn_layer.png - GNN iteration impact"
echo "  • ablation_svd_rank.png - SVD rank parameter effect"
echo "  • ablation_combined_summary.png - Comprehensive overview"
echo ""
echo "Summary tables:"
for table in "$OUTPUT_DIR"/plots/summary_table_*.csv; do
    if [ -f "$table" ]; then
        echo "  • $(basename "$table")"
    fi
done
echo ""
echo "================================================================================"
echo "Use these results to answer the reviewer's questions about:"
echo "  1. Contribution of the denoised-graph branch"
echo "  2. Justification for GNN iteration steps (gnn_layer)"
echo "  3. Rationale for hyperparameters (svd_q, hidden_dims, etc.)"
echo "================================================================================"
