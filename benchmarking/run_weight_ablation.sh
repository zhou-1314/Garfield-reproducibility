#!/bin/bash

################################################################################
# Garfield Weight Parameter (ω) Ablation Study Runner
#
# This script runs ablation experiments for the weight parameter that controls
# the balance between spatial and expression connectivity.
#
# Usage:
#   bash run_weight_ablation.sh --data-path /path/to/data.h5ad [options]
#
# Required:
#   --data-path PATH       Path to input spatial AnnData file (.h5ad)
#
# Options:
#   --weight-values VALS   Weight values to test (space-separated)
#                         Default: 0.0 0.2 0.4 0.5 0.6 0.8 1.0
#   --output-dir DIR       Output directory (default: ./ablation_results)
#   --n-epochs N           Number of training epochs (default: 100)
#   --device-id ID         GPU device ID (default: 0)
#   --profile TYPE         Data profile (default: spatial)
#   --help, -h             Show this help message
################################################################################

set -e  # Exit on error

# Default parameters
DATA_PATH=""
WEIGHT_VALUES="0.0 0.2 0.4 0.5 0.6 0.8 1.0"
OUTPUT_DIR="./ablation_results"
N_EPOCHS=100
DEVICE_ID=0
PROFILE="spatial"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-path)
            DATA_PATH="$2"
            shift 2
            ;;
        --weight-values)
            WEIGHT_VALUES="$2"
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
        --profile)
            PROFILE="$2"
            shift 2
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
    echo "Usage: bash run_weight_ablation.sh --data-path /path/to/spatial_data.h5ad"
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
echo "GARFIELD WEIGHT PARAMETER (ω) ABLATION STUDY RUNNER"
echo "================================================================================"
echo "Configuration:"
echo "  Data path: $DATA_PATH"
echo "  Weight values: $WEIGHT_VALUES"
echo "  Output directory: $OUTPUT_DIR"
echo "  Training epochs: $N_EPOCHS"
echo "  GPU device ID: $DEVICE_ID"
echo "  Data profile: $PROFILE"
echo "================================================================================"
echo ""
echo "The weight parameter (ω) controls the connectivity matrix combination:"
echo "  Combined_adj = ω × spatial_adj + (1-ω) × expression_adj"
echo ""
echo "Testing range: ω ∈ [$(echo $WEIGHT_VALUES | awk '{print $1}'), $(echo $WEIGHT_VALUES | awk '{print $NF}')]"
echo "  ω = 0.0 → 100% expression-based (no spatial info)"
echo "  ω = 0.5 → 50% spatial + 50% expression (balanced)"
echo "  ω = 1.0 → 100% spatial-based (no expression info)"
echo "  ω = 0.8 → Default (80% spatial + 20% expression)"
echo "================================================================================"
echo ""

# Check Python and dependencies
if ! command -v python &> /dev/null; then
    echo "Error: Python not found. Please install Python 3.7+"
    exit 1
fi

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

# Run weight ablation study
echo "================================================================================"
echo "STEP 1: Running Weight Ablation Study"
echo "================================================================================"
echo ""

python ablation_weight_parameter.py \
    --data-path "$DATA_PATH" \
    --weight-values $WEIGHT_VALUES \
    --profile "$PROFILE" \
    --output-dir "$OUTPUT_DIR" \
    --n-epochs "$N_EPOCHS" \
    --device-id "$DEVICE_ID"

ABLATION_EXIT_CODE=$?

if [ $ABLATION_EXIT_CODE -ne 0 ]; then
    echo ""
    echo "================================================================================"
    echo "ERROR: Weight ablation study failed with exit code $ABLATION_EXIT_CODE"
    echo "================================================================================"
    exit $ABLATION_EXIT_CODE
fi

echo ""
echo "✓ Weight ablation study completed successfully"
echo ""

# Generate plots
echo "================================================================================"
echo "STEP 2: Generating Visualization Plots"
echo "================================================================================"
echo ""

python plot_weight_ablation.py \
    --results-file "$OUTPUT_DIR/ablation_weight.csv" \
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
echo "WEIGHT ABLATION STUDY COMPLETE!"
echo "================================================================================"
echo ""
echo "Results saved to $OUTPUT_DIR/:"
echo "  • ablation_weight.csv - Complete results table"
echo "  • ablation_weight.json - Raw results (JSON)"
echo ""
echo "Plots saved to $OUTPUT_DIR/plots/:"
echo "  • weight_ablation_comprehensive.png - All metrics vs weight"
echo "  • weight_tradeoff_analysis.png - Performance trade-offs"
echo "  • weight_heatmap_summary.png - Normalized metrics heatmap"
echo "  • weight_recommendations.csv - Optimal weights for objectives"
echo ""
echo "================================================================================"
echo "KEY FINDINGS (check the CSV files for exact values):"
echo "================================================================================"

# Display summary if CSV exists
if [ -f "$OUTPUT_DIR/ablation_weight.csv" ]; then
    echo ""
    python -c "
import pandas as pd
df = pd.read_csv('$OUTPUT_DIR/ablation_weight.csv')
print('Weight (ω) | Interpretation          | ARI    | Spatial Coherence')
print('-' * 70)
for _, row in df.iterrows():
    w = row['weight']
    spatial_pct = int(w * 100)
    expr_pct = 100 - spatial_pct
    interp = f'{spatial_pct:3d}% S + {expr_pct:3d}% E'
    ari = row.get('ARI', 0)
    sc = row.get('spatial_coherence_mean', 0)
    print(f'{w:5.1f}      | {interp:22s} | {ari:6.4f} | {sc:6.4f}')

# Find optimal
if 'ARI' in df.columns:
    opt_idx = df['ARI'].idxmax()
    opt_w = df.loc[opt_idx, 'weight']
    opt_ari = df.loc[opt_idx, 'ARI']
    print('\n' + '=' * 70)
    print(f'Optimal weight: ω = {opt_w:.1f} (ARI = {opt_ari:.4f})')
    print(f'Interpretation: {int(opt_w*100)}% spatial + {int((1-opt_w)*100)}% expression')
"
fi

echo ""
echo "================================================================================"
echo "NEXT STEPS:"
echo "================================================================================"
echo "1. Review the plots in $OUTPUT_DIR/plots/"
echo "2. Check the reviewer response: reviewer_response_weight_parameter.md"
echo "3. Consider adjusting the default weight if your optimal differs from 0.8"
echo ""
echo "For more information, see:"
echo "  • benchmarking/ABLATION_STUDY_README.md"
echo "  • reviewer_response_weight_parameter.md"
echo "================================================================================"
