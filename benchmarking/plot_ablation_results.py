"""
Visualization script for Garfield ablation study results.

This script generates publication-quality figures showing:
- Performance comparison: SVD vs Dropout augmentation
- Impact of GNN iteration steps on performance and runtime
- Effect of SVD rank (svd_q) on denoising quality
- Trade-offs between performance and computational cost
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


def load_ablation_results(results_dir: str) -> dict:
    """Load all ablation study results."""
    results = {}

    # Augment type ablation
    augment_file = os.path.join(results_dir, "ablation_augment_type.csv")
    if os.path.exists(augment_file):
        results['augment_type'] = pd.read_csv(augment_file)

    # GNN layer ablation
    gnn_file = os.path.join(results_dir, "ablation_gnn_layer.csv")
    if os.path.exists(gnn_file):
        results['gnn_layer'] = pd.read_csv(gnn_file)

    # SVD rank ablation
    svd_file = os.path.join(results_dir, "ablation_svd_rank.csv")
    if os.path.exists(svd_file):
        results['svd_rank'] = pd.read_csv(svd_file)

    return results


def plot_augment_type_comparison(df: pd.DataFrame, output_dir: str):
    """
    Plot comparison between SVD and Dropout augmentation.

    Parameters
    ----------
    df : pd.DataFrame
        Ablation results for augment_type
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    metrics_to_plot = [
        ('ARI', 'Adjusted Rand Index'),
        ('NMI', 'Normalized Mutual Info'),
        ('ASW_label', 'Silhouette Score (Label)'),
        ('batch_mixing_score', 'Batch Mixing Score'),
        ('runtime_seconds', 'Runtime (seconds)'),
        ('peak_cpu_memory_mb', 'Peak CPU Memory (MB)')
    ]

    colors = {'svd': '#2ecc71', 'dropout': '#e74c3c'}

    for idx, (metric, title) in enumerate(metrics_to_plot):
        ax = axes[idx]

        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{title}\n(Not Available)',
                   ha='center', va='center', fontsize=12)
            ax.set_title(title, fontweight='bold', fontsize=14)
            continue

        # Bar plot
        x_pos = np.arange(len(df))
        values = df[metric].values
        augment_types = df['augment_type'].values

        bars = ax.bar(x_pos, values, color=[colors[a] for a in augment_types],
                     alpha=0.8, edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}' if val < 100 else f'{val:.1f}',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(augment_types, fontsize=12)
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14, pad=10)
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight better performance
        if metric in ['ARI', 'NMI', 'ASW_label', 'ASW_cluster', 'batch_mixing_score',
                     'label_transfer_accuracy']:
            best_idx = np.argmax(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)
        elif metric in ['runtime_seconds', 'peak_cpu_memory_mb', 'peak_gpu_memory_mb']:
            best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('gold')
            bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ablation_augment_type.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_gnn_layer_analysis(df: pd.DataFrame, output_dir: str):
    """
    Plot impact of GNN iteration steps.

    Parameters
    ----------
    df : pd.DataFrame
        Ablation results for gnn_layer
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    metrics_to_plot = [
        ('ARI', 'Adjusted Rand Index', True),
        ('NMI', 'Normalized Mutual Info', True),
        ('ASW_label', 'Silhouette Score', True),
        ('batch_mixing_score', 'Batch Mixing Score', True),
        ('runtime_seconds', 'Runtime (seconds)', False),
        ('peak_cpu_memory_mb', 'Peak CPU Memory (MB)', False)
    ]

    colors = sns.color_palette("viridis", len(df))

    for idx, (metric, title, higher_is_better) in enumerate(metrics_to_plot):
        ax = axes[idx]

        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{title}\n(Not Available)',
                   ha='center', va='center', fontsize=12)
            ax.set_title(title, fontweight='bold', fontsize=14)
            continue

        x = df['gnn_layer'].values
        y = df[metric].values

        # Line plot with markers
        ax.plot(x, y, marker='o', linewidth=2.5, markersize=10,
               color='#3498db', markerfacecolor='#e74c3c',
               markeredgewidth=2, markeredgecolor='#2c3e50')

        # Highlight optimal point
        if higher_is_better:
            best_idx = np.argmax(y)
        else:
            best_idx = np.argmin(y)

        ax.plot(x[best_idx], y[best_idx], marker='*', markersize=20,
               color='gold', markeredgewidth=2, markeredgecolor='black',
               zorder=10)

        # Annotate all points
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(f'{yi:.3f}' if yi < 100 else f'{yi:.1f}',
                       xy=(xi, yi), xytext=(0, 10),
                       textcoords='offset points', ha='center',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                               alpha=0.7 if i == best_idx else 0.3))

        ax.set_xlabel('Number of GNN Iterations (gnn_layer)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14, pad=10)
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ablation_gnn_layer.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_svd_rank_analysis(df: pd.DataFrame, output_dir: str):
    """
    Plot effect of SVD rank parameter.

    Parameters
    ----------
    df : pd.DataFrame
        Ablation results for svd_q
    output_dir : str
        Output directory
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    metrics_to_plot = [
        ('ARI', 'Adjusted Rand Index', True),
        ('NMI', 'Normalized Mutual Info', True),
        ('ASW_label', 'Silhouette Score', True),
        ('batch_mixing_score', 'Batch Mixing Score', True),
        ('runtime_seconds', 'Runtime (seconds)', False),
        ('peak_cpu_memory_mb', 'Peak CPU Memory (MB)', False)
    ]

    for idx, (metric, title, higher_is_better) in enumerate(metrics_to_plot):
        ax = axes[idx]

        if metric not in df.columns:
            ax.text(0.5, 0.5, f'{title}\n(Not Available)',
                   ha='center', va='center', fontsize=12)
            ax.set_title(title, fontweight='bold', fontsize=14)
            continue

        x = df['svd_q'].values
        y = df[metric].values

        # Line plot with markers
        ax.plot(x, y, marker='o', linewidth=2.5, markersize=10,
               color='#9b59b6', markerfacecolor='#e74c3c',
               markeredgewidth=2, markeredgecolor='#2c3e50')

        # Highlight optimal point
        if higher_is_better:
            best_idx = np.argmax(y)
        else:
            best_idx = np.argmin(y)

        ax.plot(x[best_idx], y[best_idx], marker='*', markersize=20,
               color='gold', markeredgewidth=2, markeredgecolor='black',
               zorder=10)

        # Annotate all points
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.annotate(f'{yi:.3f}' if yi < 100 else f'{yi:.1f}',
                       xy=(xi, yi), xytext=(0, 10),
                       textcoords='offset points', ha='center',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow',
                               alpha=0.7 if i == best_idx else 0.3))

        ax.set_xlabel('SVD Rank (svd_q)', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(title, fontweight='bold', fontsize=14, pad=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'ablation_svd_rank.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_combined_summary(results: dict, output_dir: str):
    """
    Create combined summary figure with key findings.

    Parameters
    ----------
    results : dict
        Dictionary of all ablation results
    output_dir : str
        Output directory
    """
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Top row: Augment type comparison (3 key metrics)
    if 'augment_type' in results:
        df_aug = results['augment_type']
        metrics = ['ARI', 'runtime_seconds', 'peak_cpu_memory_mb']
        titles = ['ARI (Higher is Better)', 'Runtime (Lower is Better)', 'Memory (Lower is Better)']
        colors = {'svd': '#2ecc71', 'dropout': '#e74c3c'}

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = fig.add_subplot(gs[0, i])
            if metric in df_aug.columns:
                x_pos = np.arange(len(df_aug))
                values = df_aug[metric].values
                augment_types = df_aug['augment_type'].values

                bars = ax.bar(x_pos, values, color=[colors[a] for a in augment_types],
                             alpha=0.8, edgecolor='black', linewidth=1.5)

                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.3f}' if val < 100 else f'{val:.1f}',
                           ha='center', va='bottom', fontsize=11, fontweight='bold')

                ax.set_xticks(x_pos)
                ax.set_xticklabels(augment_types, fontsize=11)
                ax.set_title(title, fontweight='bold', fontsize=13, pad=10)
                ax.grid(True, alpha=0.3, axis='y')

    # Middle row: GNN layer impact (3 key metrics)
    if 'gnn_layer' in results:
        df_gnn = results['gnn_layer']
        metrics = ['ARI', 'runtime_seconds', 'peak_cpu_memory_mb']
        titles = ['ARI vs GNN Iterations', 'Runtime vs GNN Iterations', 'Memory vs GNN Iterations']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = fig.add_subplot(gs[1, i])
            if metric in df_gnn.columns:
                x = df_gnn['gnn_layer'].values
                y = df_gnn[metric].values

                ax.plot(x, y, marker='o', linewidth=2.5, markersize=9,
                       color='#3498db', markerfacecolor='#e74c3c',
                       markeredgewidth=2)

                ax.set_xlabel('GNN Iterations', fontsize=11, fontweight='bold')
                ax.set_ylabel(metric, fontsize=11, fontweight='bold')
                ax.set_title(title, fontweight='bold', fontsize=13, pad=10)
                ax.set_xticks(x)
                ax.grid(True, alpha=0.3)

    # Bottom row: SVD rank impact (3 key metrics)
    if 'svd_rank' in results:
        df_svd = results['svd_rank']
        metrics = ['ARI', 'runtime_seconds', 'peak_cpu_memory_mb']
        titles = ['ARI vs SVD Rank', 'Runtime vs SVD Rank', 'Memory vs SVD Rank']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = fig.add_subplot(gs[2, i])
            if metric in df_svd.columns:
                x = df_svd['svd_q'].values
                y = df_svd[metric].values

                ax.plot(x, y, marker='o', linewidth=2.5, markersize=9,
                       color='#9b59b6', markerfacecolor='#e74c3c',
                       markeredgewidth=2)

                ax.set_xlabel('SVD Rank (q)', fontsize=11, fontweight='bold')
                ax.set_ylabel(metric, fontsize=11, fontweight='bold')
                ax.set_title(title, fontweight='bold', fontsize=13, pad=10)
                ax.grid(True, alpha=0.3)

    plt.suptitle('Garfield Ablation Study: Comprehensive Summary',
                fontsize=18, fontweight='bold', y=0.995)

    output_file = os.path.join(output_dir, 'ablation_combined_summary.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def create_summary_tables(results: dict, output_dir: str):
    """
    Create summary tables for all ablation studies.

    Parameters
    ----------
    results : dict
        Dictionary of all ablation results
    output_dir : str
        Output directory
    """
    print("\n" + "=" * 100)
    print("ABLATION STUDY SUMMARY TABLES")
    print("=" * 100)

    for study_name, df in results.items():
        print(f"\n{'=' * 100}")
        print(f"{study_name.upper().replace('_', ' ')} ABLATION")
        print(f"{'=' * 100}")

        # Select relevant columns
        display_cols = []
        if 'augment_type' in df.columns:
            display_cols.append('augment_type')
        if 'gnn_layer' in df.columns:
            display_cols.append('gnn_layer')
        if 'svd_q' in df.columns:
            display_cols.append('svd_q')

        metric_cols = [c for c in ['ARI', 'NMI', 'ASW_label', 'batch_mixing_score',
                                   'runtime_seconds', 'peak_cpu_memory_mb', 'peak_gpu_memory_mb']
                      if c in df.columns]

        display_df = df[display_cols + metric_cols]
        print(display_df.to_string(index=False))

        # Save to CSV
        output_file = os.path.join(output_dir, f"summary_table_{study_name}.csv")
        display_df.to_csv(output_file, index=False)
        print(f"\nSaved: {output_file}")

    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Garfield ablation study results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./ablation_results",
        help="Directory containing ablation results CSV files"
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
    print("Loading ablation study results...")
    results = load_ablation_results(args.results_dir)

    if not results:
        print(f"No ablation results found in {args.results_dir}")
        return

    print(f"Found {len(results)} ablation studies")

    # Generate plots
    print("\nGenerating plots...")

    if 'augment_type' in results:
        plot_augment_type_comparison(results['augment_type'], args.output_dir)

    if 'gnn_layer' in results:
        plot_gnn_layer_analysis(results['gnn_layer'], args.output_dir)

    if 'svd_rank' in results:
        plot_svd_rank_analysis(results['svd_rank'], args.output_dir)

    # Combined summary
    plot_combined_summary(results, args.output_dir)

    # Create summary tables
    create_summary_tables(results, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
