"""
Visualization script for Garfield spatial scalability benchmark results.

This script generates publication-quality figures showing:
- Runtime vs dataset size for all major tasks
- Memory consumption vs dataset size
- Comparison of different graph construction methods
- Breakdown by task category
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
sns.set_context("paper", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']


def load_results(results_file: str) -> pd.DataFrame:
    """Load benchmark results from CSV file."""
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")

    df = pd.read_csv(results_file)
    print(f"Loaded {len(df)} benchmark results from {results_file}")
    print(f"Task categories: {df['task_category'].unique()}")

    return df


def plot_runtime_by_task(df: pd.DataFrame, output_dir: str):
    """
    Plot runtime vs dataset size for each task category.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results
    output_dir : str
        Output directory for plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    task_categories = [
        'dimension_reduction',
        'graph_construction',
        'model_training',
        'label_transfer'
    ]

    task_names = {
        'dimension_reduction': 'Dimension Reduction (PCA)',
        'graph_construction': 'Graph Construction',
        'model_training': 'Model Training',
        'label_transfer': 'Label Transfer (Mapping)'
    }

    colors = sns.color_palette("husl", 5)

    for idx, task in enumerate(task_categories):
        ax = axes[idx]
        task_df = df[df['task_category'] == task]

        if len(task_df) == 0:
            ax.text(0.5, 0.5, f'No data for\n{task_names[task]}',
                   ha='center', va='center', fontsize=12)
            ax.set_title(task_names[task], fontsize=14, fontweight='bold')
            continue

        if task == 'graph_construction':
            # Plot different methods separately
            methods = task_df['method'].unique()
            for i, method in enumerate(methods):
                method_df = task_df[task_df['method'] == method]
                ax.plot(
                    method_df['n_cells'],
                    method_df['runtime_seconds'],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=method,
                    color=colors[i]
                )
            ax.legend(loc='best', frameon=True, framealpha=0.9)
        elif task == 'label_transfer':
            # Use query cells for x-axis
            ax.plot(
                task_df['n_query_cells'],
                task_df['runtime_seconds'],
                marker='o',
                linewidth=2,
                markersize=8,
                color=colors[0]
            )
        else:
            # Standard plot
            ax.plot(
                task_df['n_cells'],
                task_df['runtime_seconds'],
                marker='o',
                linewidth=2,
                markersize=8,
                color=colors[0]
            )

        ax.set_xlabel('Number of Cells', fontsize=12)
        ax.set_ylabel('Runtime (seconds)', fontsize=12)
        ax.set_title(task_names[task], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Format x-axis with thousands separator
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{int(x):,}')
        )

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'runtime_by_task.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_memory_by_task(df: pd.DataFrame, output_dir: str):
    """
    Plot memory consumption vs dataset size for each task category.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results
    output_dir : str
        Output directory for plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    task_categories = [
        'dimension_reduction',
        'graph_construction',
        'model_training',
        'label_transfer'
    ]

    task_names = {
        'dimension_reduction': 'Dimension Reduction (PCA)',
        'graph_construction': 'Graph Construction',
        'model_training': 'Model Training',
        'label_transfer': 'Label Transfer (Mapping)'
    }

    colors = sns.color_palette("husl", 5)

    for idx, task in enumerate(task_categories):
        ax = axes[idx]
        task_df = df[df['task_category'] == task]

        if len(task_df) == 0:
            ax.text(0.5, 0.5, f'No data for\n{task_names[task]}',
                   ha='center', va='center', fontsize=12)
            ax.set_title(task_names[task], fontsize=14, fontweight='bold')
            continue

        if task == 'graph_construction':
            # Plot different methods separately
            methods = task_df['method'].unique()
            for i, method in enumerate(methods):
                method_df = task_df[task_df['method'] == method]
                ax.plot(
                    method_df['n_cells'],
                    method_df['peak_cpu_memory_mb'],
                    marker='o',
                    linewidth=2,
                    markersize=8,
                    label=f'{method} (CPU)',
                    color=colors[i],
                    linestyle='-'
                )

                # Add GPU memory if available
                if 'peak_gpu_memory_mb' in method_df.columns:
                    gpu_mem = method_df['peak_gpu_memory_mb']
                    if gpu_mem.sum() > 0:
                        ax.plot(
                            method_df['n_cells'],
                            gpu_mem,
                            marker='s',
                            linewidth=2,
                            markersize=8,
                            label=f'{method} (GPU)',
                            color=colors[i],
                            linestyle='--'
                        )
            ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=9)
        elif task == 'label_transfer':
            # Use query cells for x-axis
            ax.plot(
                task_df['n_query_cells'],
                task_df['peak_cpu_memory_mb'],
                marker='o',
                linewidth=2,
                markersize=8,
                label='CPU',
                color=colors[0]
            )

            # Add GPU memory if available
            if 'peak_gpu_memory_mb' in task_df.columns:
                gpu_mem = task_df['peak_gpu_memory_mb']
                if gpu_mem.sum() > 0:
                    ax.plot(
                        task_df['n_query_cells'],
                        gpu_mem,
                        marker='s',
                        linewidth=2,
                        markersize=8,
                        label='GPU',
                        color=colors[0],
                        linestyle='--'
                    )
                    ax.legend(loc='best', frameon=True, framealpha=0.9)
        else:
            # Standard plot
            ax.plot(
                task_df['n_cells'],
                task_df['peak_cpu_memory_mb'],
                marker='o',
                linewidth=2,
                markersize=8,
                label='CPU',
                color=colors[0]
            )

            # Add GPU memory if available
            if 'peak_gpu_memory_mb' in task_df.columns:
                gpu_mem = task_df['peak_gpu_memory_mb']
                if gpu_mem.sum() > 0:
                    ax.plot(
                        task_df['n_cells'],
                        gpu_mem,
                        marker='s',
                        linewidth=2,
                        markersize=8,
                        label='GPU',
                        color=colors[0],
                        linestyle='--'
                    )
                    ax.legend(loc='best', frameon=True, framealpha=0.9)

        ax.set_xlabel('Number of Cells', fontsize=12)
        ax.set_ylabel('Peak Memory (MB)', fontsize=12)
        ax.set_title(task_names[task], fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Format x-axis with thousands separator
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{int(x):,}')
        )

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'memory_by_task.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def plot_combined_overview(df: pd.DataFrame, output_dir: str):
    """
    Create a combined overview figure with runtime and memory for all tasks.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results
    output_dir : str
        Output directory for plots
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    task_names = {
        'dimension_reduction': 'Dimension Reduction',
        'graph_construction': 'Graph Construction',
        'model_training': 'Model Training',
        'label_transfer': 'Label Transfer'
    }

    colors = {
        'dimension_reduction': '#1f77b4',
        'graph_construction': '#ff7f0e',
        'model_training': '#2ca02c',
        'label_transfer': '#d62728'
    }

    # Top row: Combined runtime plot
    ax_runtime = fig.add_subplot(gs[0, :])
    for task in df['task_category'].unique():
        if task == 'graph_construction':
            # Use KNN method as representative
            task_df = df[(df['task_category'] == task) & (df['method'] == 'KNN')]
        else:
            task_df = df[df['task_category'] == task]

        if len(task_df) == 0:
            continue

        if task == 'label_transfer':
            x_vals = task_df['n_query_cells']
        else:
            x_vals = task_df['n_cells']

        ax_runtime.plot(
            x_vals,
            task_df['runtime_seconds'],
            marker='o',
            linewidth=2.5,
            markersize=9,
            label=task_names.get(task, task),
            color=colors.get(task, '#333333')
        )

    ax_runtime.set_xlabel('Number of Cells', fontsize=13, fontweight='bold')
    ax_runtime.set_ylabel('Runtime (seconds)', fontsize=13, fontweight='bold')
    ax_runtime.set_title('Runtime Scalability Across All Tasks',
                         fontsize=15, fontweight='bold', pad=15)
    ax_runtime.legend(loc='best', frameon=True, framealpha=0.95, fontsize=11)
    ax_runtime.grid(True, alpha=0.3)
    ax_runtime.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{int(x):,}')
    )

    # Middle row: Combined memory plot
    ax_memory = fig.add_subplot(gs[1, :])
    for task in df['task_category'].unique():
        if task == 'graph_construction':
            # Use KNN method as representative
            task_df = df[(df['task_category'] == task) & (df['method'] == 'KNN')]
        else:
            task_df = df[df['task_category'] == task]

        if len(task_df) == 0:
            continue

        if task == 'label_transfer':
            x_vals = task_df['n_query_cells']
        else:
            x_vals = task_df['n_cells']

        ax_memory.plot(
            x_vals,
            task_df['peak_cpu_memory_mb'],
            marker='o',
            linewidth=2.5,
            markersize=9,
            label=task_names.get(task, task),
            color=colors.get(task, '#333333')
        )

    ax_memory.set_xlabel('Number of Cells', fontsize=13, fontweight='bold')
    ax_memory.set_ylabel('Peak CPU Memory (MB)', fontsize=13, fontweight='bold')
    ax_memory.set_title('Memory Consumption Across All Tasks',
                        fontsize=15, fontweight='bold', pad=15)
    ax_memory.legend(loc='best', frameon=True, framealpha=0.95, fontsize=11)
    ax_memory.grid(True, alpha=0.3)
    ax_memory.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{int(x):,}')
    )

    # Bottom row: Graph construction method comparison
    ax_graph_runtime = fig.add_subplot(gs[2, 0])
    graph_df = df[df['task_category'] == 'graph_construction']
    if len(graph_df) > 0:
        methods = graph_df['method'].unique()
        method_colors = sns.color_palette("Set2", len(methods))

        for i, method in enumerate(methods):
            method_df = graph_df[graph_df['method'] == method]
            ax_graph_runtime.plot(
                method_df['n_cells'],
                method_df['runtime_seconds'],
                marker='o',
                linewidth=2,
                markersize=7,
                label=method,
                color=method_colors[i]
            )

        ax_graph_runtime.set_xlabel('Number of Cells', fontsize=12, fontweight='bold')
        ax_graph_runtime.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
        ax_graph_runtime.set_title('Graph Construction Methods',
                                   fontsize=13, fontweight='bold', pad=10)
        ax_graph_runtime.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
        ax_graph_runtime.grid(True, alpha=0.3)
        ax_graph_runtime.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, p: f'{int(x):,}')
        )

    # Bottom right: Scalability efficiency (runtime per 1000 cells)
    ax_efficiency = fig.add_subplot(gs[2, 1])
    for task in df['task_category'].unique():
        if task == 'graph_construction':
            task_df = df[(df['task_category'] == task) & (df['method'] == 'KNN')]
        else:
            task_df = df[df['task_category'] == task]

        if len(task_df) == 0:
            continue

        if task == 'label_transfer':
            x_vals = task_df['n_query_cells']
            efficiency = task_df['runtime_seconds'] / (task_df['n_query_cells'] / 1000)
        else:
            x_vals = task_df['n_cells']
            efficiency = task_df['runtime_seconds'] / (task_df['n_cells'] / 1000)

        ax_efficiency.plot(
            x_vals,
            efficiency,
            marker='o',
            linewidth=2,
            markersize=7,
            label=task_names.get(task, task),
            color=colors.get(task, '#333333')
        )

    ax_efficiency.set_xlabel('Number of Cells', fontsize=12, fontweight='bold')
    ax_efficiency.set_ylabel('Runtime per 1K Cells (s)', fontsize=12, fontweight='bold')
    ax_efficiency.set_title('Scalability Efficiency',
                           fontsize=13, fontweight='bold', pad=10)
    ax_efficiency.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
    ax_efficiency.grid(True, alpha=0.3)
    ax_efficiency.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, p: f'{int(x):,}')
    )

    output_file = os.path.join(output_dir, 'combined_overview.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    print(f"Saved: {output_file}")
    plt.close()


def create_summary_table(df: pd.DataFrame, output_dir: str):
    """
    Create a summary table of benchmark results.

    Parameters
    ----------
    df : pd.DataFrame
        Benchmark results
    output_dir : str
        Output directory
    """
    summary_data = []

    for task in df['task_category'].unique():
        task_df = df[df['task_category'] == task]

        if task == 'graph_construction':
            for method in task_df['method'].unique():
                method_df = task_df[task_df['method'] == method]
                summary_data.append({
                    'Task': f'Graph Construction ({method})',
                    'Min Cells': method_df['n_cells'].min(),
                    'Max Cells': method_df['n_cells'].max(),
                    'Min Runtime (s)': method_df['runtime_seconds'].min(),
                    'Max Runtime (s)': method_df['runtime_seconds'].max(),
                    'Min Memory (MB)': method_df['peak_cpu_memory_mb'].min(),
                    'Max Memory (MB)': method_df['peak_cpu_memory_mb'].max(),
                })
        else:
            task_name = task.replace('_', ' ').title()
            summary_data.append({
                'Task': task_name,
                'Min Cells': task_df['n_cells'].min() if 'n_cells' in task_df.columns else task_df['n_query_cells'].min(),
                'Max Cells': task_df['n_cells'].max() if 'n_cells' in task_df.columns else task_df['n_query_cells'].max(),
                'Min Runtime (s)': task_df['runtime_seconds'].min(),
                'Max Runtime (s)': task_df['runtime_seconds'].max(),
                'Min Memory (MB)': task_df['peak_cpu_memory_mb'].min(),
                'Max Memory (MB)': task_df['peak_cpu_memory_mb'].max(),
            })

    summary_df = pd.DataFrame(summary_data)

    # Save to CSV
    output_file = os.path.join(output_dir, 'summary_table.csv')
    summary_df.to_csv(output_file, index=False)
    print(f"\nSaved summary table: {output_file}")

    # Print to console
    print("\n" + "=" * 100)
    print("BENCHMARK SUMMARY TABLE")
    print("=" * 100)
    print(summary_df.to_string(index=False))
    print("=" * 100 + "\n")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Garfield spatial scalability benchmark results"
    )
    parser.add_argument(
        "--results-file",
        type=str,
        default="./benchmark_results/benchmark_results.csv",
        help="Path to benchmark results CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results/plots",
        help="Output directory for plots"
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load results
    df = load_results(args.results_file)

    # Generate plots
    print("\nGenerating plots...")
    plot_runtime_by_task(df, args.output_dir)
    plot_memory_by_task(df, args.output_dir)
    plot_combined_overview(df, args.output_dir)

    # Create summary table
    create_summary_table(df, args.output_dir)

    print(f"\nAll plots saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
