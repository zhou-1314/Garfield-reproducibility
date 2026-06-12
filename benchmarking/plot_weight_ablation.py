"""
Visualization script for weight parameter (ω) ablation study results.

This script generates publication-quality figures showing:
- Performance vs weight value
- Spatial coherence vs weight value
- Graph connectivity statistics vs weight value
- Trade-offs between spatial and expression contributions
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set publication-quality plot style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.4)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


def load_weight_ablation_results(results_file: str) -> pd.DataFrame:
    """Load weight ablation results from CSV file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} weight configurations from {results_file}")

    return df


def plot_weight_ablation_comprehensive(df: pd.DataFrame, output_dir: str):
    """
    Create comprehensive figure showing all weight ablation results.

    Parameters
    ----------
    df : pd.DataFrame
        Weight ablation results
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    # Define metrics to plot
    metrics_config = [
        ('ARI', 'Adjusted Rand Index', True, '#2ecc71'),
        ('NMI', 'Normalized Mutual Info', True, '#3498db'),
        ('ASW_label', 'Silhouette Score', True, '#9b59b6'),
        ('spatial_coherence_mean', 'Spatial Coherence', True, '#e74c3c'),
        ('avg_degree', 'Average Node Degree', False, '#f39c12'),
        ('runtime_seconds', 'Runtime (seconds)', False, '#95a5a6')
    ]

    for idx, (metric, title, higher_is_better, color) in enumerate(metrics_config):
        ax = axes[idx]

        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{title}\n(Not Available)',
                   ha='center', va='center', fontsize=12)
            ax.set_title(title, fontweight='bold', fontsize=14)
            continue

        x = df['weight'].values
        y = df[metric].values

        # Line plot with markers
        ax.plot(x, y, marker='o', linewidth=3, markersize=12,
               color=color, markerfacecolor='white',
               markeredgewidth=3, markeredgecolor=color,
               label=title)

        # Highlight optimal point
        if higher_is_better:
            best_idx = np.argmax(y)
        else:
            best_idx = np.argmin(y)

        ax.plot(x[best_idx], y[best_idx], marker='*', markersize=25,
               color='gold', markeredgewidth=2.5, markeredgecolor='black',
               zorder=10, label=f'Optimal: ω={x[best_idx]:.1f}')

        # Annotate optimal point
        ax.annotate(f'ω={x[best_idx]:.1f}\n{y[best_idx]:.3f}',
                   xy=(x[best_idx], y[best_idx]),
                   xytext=(20, 20),
                   textcoords='offset points',
                   fontsize=11,
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                 lw=2, color='black'))

        # Shade extreme regions
        if metric in ['ARI', 'NMI', 'ASW_label', 'spatial_coherence_mean']:
            # Highlight problematic regions
            if x[0] == 0.0:  # Pure expression (no spatial)
                ax.axvspan(0.0, 0.2, alpha=0.1, color='red', label='Too little spatial')
            if x[-1] == 1.0:  # Pure spatial (no expression)
                ax.axvspan(0.9, 1.0, alpha=0.1, color='red', label='Too little expression')

        ax.set_xlabel('Weight (ω)', fontsize=13, fontweight='bold')
        ax.set_ylabel(title, fontsize=13, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=15, pad=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_xticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])
        ax.grid(True, alpha=0.3)

        # Add interpretation axis on top
        ax2 = ax.twiny()
        ax2.set_xlim(-0.05, 1.05)
        ax2.set_xticks([0.0, 0.5, 1.0])
        ax2.set_xticklabels(['100% Expr', '50%-50%', '100% Spatial'],
                           fontsize=10, style='italic')
        ax2.spines['top'].set_visible(False)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'weight_ablation_comprehensive.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_weight_tradeoff_analysis(df: pd.DataFrame, output_dir: str):
    """
    Create figure showing trade-offs between spatial and expression contributions.

    Parameters
    ----------
    df : pd.DataFrame
        Weight ablation results
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Performance metrics
    ax1 = axes[0]
    performance_metrics = ['ARI', 'NMI', 'ASW_label']
    colors = ['#2ecc71', '#3498db', '#9b59b6']

    for metric, color in zip(performance_metrics, colors):
        if metric in df.columns:
            # Normalize to 0-1 for comparison
            y = df[metric].values
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-8)

            ax1.plot(df['weight'], y_norm, marker='o', linewidth=2.5,
                    markersize=9, label=metric, color=color)

    ax1.set_xlabel('Weight (ω): Spatial ← → Expression', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Normalized Performance (0-1)', fontsize=13, fontweight='bold')
    ax1.set_title('Clustering Performance vs Weight', fontsize=15, fontweight='bold', pad=10)
    ax1.legend(loc='best', frameon=True, framealpha=0.9, fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_xticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])

    # Add interpretation
    ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=2)
    ax1.text(0.5, ax1.get_ylim()[1] * 0.95, 'Equal Balance',
            ha='center', va='top', fontsize=10, style='italic',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Right plot: Spatial coherence vs graph statistics
    ax2 = axes[1]

    if 'spatial_coherence_mean' in df.columns:
        ax2_left = ax2
        ax2_right = ax2.twinx()

        # Spatial coherence on left axis
        line1 = ax2_left.plot(df['weight'], df['spatial_coherence_mean'],
                             marker='o', linewidth=2.5, markersize=9,
                             color='#e74c3c', label='Spatial Coherence')
        ax2_left.set_ylabel('Spatial Coherence', fontsize=13, fontweight='bold', color='#e74c3c')
        ax2_left.tick_params(axis='y', labelcolor='#e74c3c')

        # Average degree on right axis
        if 'avg_degree' in df.columns:
            line2 = ax2_right.plot(df['weight'], df['avg_degree'],
                                  marker='s', linewidth=2.5, markersize=9,
                                  color='#f39c12', label='Avg Degree')
            ax2_right.set_ylabel('Average Node Degree', fontsize=13, fontweight='bold', color='#f39c12')
            ax2_right.tick_params(axis='y', labelcolor='#f39c12')

            # Combined legend
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            ax2_left.legend(lines, labels, loc='upper left', frameon=True, framealpha=0.9, fontsize=11)

        ax2_left.set_xlabel('Weight (ω): Spatial ← → Expression', fontsize=13, fontweight='bold')
        ax2_left.set_title('Spatial Properties vs Weight', fontsize=15, fontweight='bold', pad=10)
        ax2_left.grid(True, alpha=0.3)
        ax2_left.set_xlim(-0.05, 1.05)
        ax2_left.set_xticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0])

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'weight_tradeoff_analysis.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_weight_heatmap_summary(df: pd.DataFrame, output_dir: str):
    """
    Create heatmap showing all metrics across weight values.

    Parameters
    ----------
    df : pd.DataFrame
        Weight ablation results
    output_dir : str
        Output directory
    """
    # Select metrics for heatmap
    metric_cols = [c for c in ['ARI', 'NMI', 'ASW_label', 'spatial_coherence_mean',
                               'avg_degree', 'runtime_seconds']
                  if c in df.columns]

    if not metric_cols:
        print("Warning: No metrics available for heatmap")
        return

    # Create matrix (transpose for better visualization)
    heatmap_data = df[['weight'] + metric_cols].set_index('weight').T

    # Normalize each row to 0-1
    heatmap_data_norm = heatmap_data.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8), axis=1)

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot heatmap
    sns.heatmap(heatmap_data_norm, annot=True, fmt='.3f', cmap='RdYlGn',
               center=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Normalized Value (0-1)'},
               linewidths=1, linecolor='white', ax=ax, annot_kws={'fontsize': 10})

    ax.set_xlabel('Weight (ω)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=14, fontweight='bold')
    ax.set_title('Normalized Performance Across Weight Values\n(Green=Better, Red=Worse)',
                fontsize=16, fontweight='bold', pad=15)

    # Rotate x-axis labels
    ax.set_xticklabels([f'{float(x):.1f}' for x in heatmap_data_norm.columns],
                       rotation=0, ha='center', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)

    # Add interpretation labels
    plt.text(0.5, -0.15, '← More Expression-based',
            ha='left', va='top', transform=ax.transAxes,
            fontsize=11, style='italic')
    plt.text(0.5, -0.15, 'More Spatial-based →',
            ha='right', va='top', transform=ax.transAxes,
            fontsize=11, style='italic')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'weight_heatmap_summary.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def create_summary_table(df: pd.DataFrame, output_dir: str):
    """
    Create summary table with recommendations.

    Parameters
    ----------
    df : pd.DataFrame
        Weight ablation results
    output_dir : str
        Output directory
    """
    print("\n" + "=" * 100)
    print("WEIGHT PARAMETER ABLATION SUMMARY")
    print("=" * 100)

    # Display all results
    display_cols = ['weight']
    metric_cols = [c for c in df.columns if c not in ['weight', 'experiment_type', 'profile',
                                                       'n_epochs', 'peak_gpu_memory_mb', 'metrics',
                                                       'model', 'adata']]
    display_df = df[display_cols + metric_cols]

    print("\nComplete Results:")
    print(display_df.to_string(index=False))

    # Find optimal weights for different metrics
    print("\n" + "-" * 100)
    print("OPTIMAL WEIGHT VALUES FOR DIFFERENT OBJECTIVES")
    print("-" * 100)

    objectives = {
        'ARI': ('Best Clustering (ARI)', True),
        'spatial_coherence_mean': ('Best Spatial Coherence', True),
        'avg_degree': ('Moderate Connectivity', False),  # Usually want moderate, not extreme
        'runtime_seconds': ('Fastest Runtime', False)
    }

    recommendations = []
    for metric, (description, higher_is_better) in objectives.items():
        if metric in df.columns:
            if higher_is_better:
                optimal_idx = df[metric].idxmax()
            else:
                optimal_idx = df[metric].idxmin()

            optimal_weight = df.loc[optimal_idx, 'weight']
            optimal_value = df.loc[optimal_idx, metric]

            print(f"\n{description}:")
            print(f"  Optimal ω = {optimal_weight:.2f}")
            print(f"  {metric} = {optimal_value:.4f}")
            print(f"  Interpretation: {optimal_weight*100:.0f}% spatial + {(1-optimal_weight)*100:.0f}% expression")

            recommendations.append({
                'Objective': description,
                'Optimal_Weight': optimal_weight,
                'Metric_Value': optimal_value,
                'Interpretation': f"{optimal_weight*100:.0f}% spatial + {(1-optimal_weight)*100:.0f}% expression"
            })

    # Save recommendations
    if recommendations:
        rec_df = pd.DataFrame(recommendations)
        output_file = os.path.join(output_dir, 'weight_recommendations.csv')
        rec_df.to_csv(output_file, index=False)
        print(f"\n✓ Recommendations saved to: {output_file}")

    # General recommendation
    print("\n" + "=" * 100)
    print("GENERAL RECOMMENDATION")
    print("=" * 100)

    if 'ARI' in df.columns:
        # Find weight with best overall performance
        optimal_idx = df['ARI'].idxmax()
        optimal_weight = df.loc[optimal_idx, 'weight']

        print(f"\nFor spatial transcriptomics data, we recommend:")
        print(f"  ω = {optimal_weight:.1f} ({optimal_weight*100:.0f}% spatial + {(1-optimal_weight)*100:.0f}% expression)")
        print(f"\nRationale:")
        if optimal_weight >= 0.7:
            print("  - High spatial weight emphasizes physical tissue structure")
            print("  - Appropriate when spatial organization is biologically meaningful")
        elif optimal_weight >= 0.4:
            print("  - Balanced weight captures both spatial and molecular information")
            print("  - Recommended for most spatial transcriptomics applications")
        else:
            print("  - Low spatial weight prioritizes molecular similarity")
            print("  - May be appropriate for spatially heterogeneous tissues")

        print(f"\nNote: The default value (ω=0.8) is based on extensive benchmarking")
        print(f"      across multiple spatial transcriptomics datasets.")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize weight parameter ablation results"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="./ablation_results/ablation_weight.csv",
        help="Path to weight ablation results CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ablation_results/plots",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    print("Loading weight ablation results...")
    df = load_weight_ablation_results(args.results_file)

    # Generate plots
    print("\nGenerating plots...")
    plot_weight_ablation_comprehensive(df, args.output_dir)
    plot_weight_tradeoff_analysis(df, args.output_dir)
    plot_weight_heatmap_summary(df, args.output_dir)

    # Create summary table
    create_summary_table(df, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")
    print("\nGenerated plots:")
    print("  • weight_ablation_comprehensive.png - All metrics vs weight")
    print("  • weight_tradeoff_analysis.png - Performance trade-offs")
    print("  • weight_heatmap_summary.png - Heatmap of normalized metrics")
    print("  • weight_recommendations.csv - Optimal weights for different objectives")


if __name__ == "__main__":
    main()
