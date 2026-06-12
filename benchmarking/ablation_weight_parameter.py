"""
Ablation Study for Weight Parameter (ω) in Connectivity Matrix Construction

This script addresses the reviewer's question:
"When generating the connectivity matrix, the weight (ω) represents the relative
contribution of inter and intra connection. Is ω calculated automatically, or
should it be predefined? If it is predefined, I suggest discussing the impact
of using different values in manuscript."

The weight parameter (ω) controls the combination of different adjacency matrices:
- For spatial data: Combined_adj = ω * spatial_adj + (1-ω) * expression_adj
- For multi-modal data: Combined_adj = ω * modality1_adj + (1-ω) * modality2_adj

This script tests different values of ω to determine optimal settings.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List
import time
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import Garfield as gf
from benchmark_utils import MemoryTracker, save_benchmark_results

# Import metrics
try:
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
    from sklearn.neighbors import NearestNeighbors
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Some metrics will be skipped.")


def compute_clustering_metrics(
    adata: ad.AnnData,
    cluster_key: str = 'leiden',
    label_key: str = 'cell_type',
    embedding_key: str = 'garfield_latent'
) -> Dict[str, float]:
    """Compute clustering quality metrics."""
    if not SKLEARN_AVAILABLE:
        return {}

    metrics = {}

    # Get labels
    clusters = adata.obs[cluster_key].values
    true_labels = adata.obs[label_key].values
    embeddings = adata.obsm[embedding_key]

    # Adjusted Rand Index
    metrics['ARI'] = adjusted_rand_score(true_labels, clusters)

    # Normalized Mutual Information
    metrics['NMI'] = normalized_mutual_info_score(true_labels, clusters)

    # Silhouette score (using true labels)
    metrics['ASW_label'] = silhouette_score(embeddings, true_labels, metric='euclidean')

    # Silhouette score (using clusters)
    metrics['ASW_cluster'] = silhouette_score(embeddings, clusters, metric='euclidean')

    return metrics


def compute_spatial_coherence(
    adata: ad.AnnData,
    label_key: str = 'cell_type',
    spatial_key: str = 'spatial',
    n_neighbors: int = 10
) -> Dict[str, float]:
    """
    Compute spatial coherence metrics.

    Measures how well cells of the same type cluster in physical space.
    """
    if not SKLEARN_AVAILABLE or spatial_key not in adata.obsm:
        return {}

    spatial_coords = adata.obsm[spatial_key]
    labels = adata.obs[label_key].values

    # Build spatial kNN graph
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(spatial_coords)
    _, indices = nbrs.kneighbors(spatial_coords)

    # Compute spatial coherence: fraction of neighbors with same label
    coherence_scores = []
    for i in range(len(labels)):
        neighbors_labels = labels[indices[i, 1:]]  # Exclude self
        same_label_fraction = (neighbors_labels == labels[i]).mean()
        coherence_scores.append(same_label_fraction)

    return {
        'spatial_coherence_mean': np.mean(coherence_scores),
        'spatial_coherence_std': np.std(coherence_scores)
    }


def compute_graph_statistics(
    adata: ad.AnnData,
    adj_key: str = 'connectivities'
) -> Dict[str, float]:
    """Compute graph connectivity statistics."""
    if adj_key not in adata.obsp:
        return {}

    adj = adata.obsp[adj_key]

    # Number of edges
    n_edges = adj.nnz // 2  # Undirected graph

    # Average degree
    degrees = np.array(adj.sum(axis=1)).flatten()
    avg_degree = degrees.mean()

    # Degree distribution statistics
    degree_std = degrees.std()
    degree_max = degrees.max()
    degree_min = degrees.min()

    return {
        'n_edges': n_edges,
        'avg_degree': avg_degree,
        'degree_std': degree_std,
        'degree_max': degree_max,
        'degree_min': degree_min
    }


def run_weight_experiment(
    adata: ad.AnnData,
    weight: float,
    profile: str = "spatial",
    n_epochs: int = 100,
    device_id: int = 0,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Run Garfield experiment with specified weight parameter.

    Parameters
    ----------
    adata : AnnData
        Input spatial data
    weight : float
        Weight parameter (0 to 1)
        - For spatial: weight * spatial_adj + (1-weight) * expr_adj
    profile : str
        Data profile type
    n_epochs : int
        Training epochs
    device_id : int
        GPU device
    seed : int
        Random seed
    verbose : bool
        Verbosity

    Returns
    -------
    Dict
        Results including metrics, runtime, memory
    """
    memory_tracker = MemoryTracker(device_id=device_id)
    memory_tracker.reset()
    start_time = time.time()

    print(f"\nRunning experiment with weight = {weight:.2f}")
    print(f"  Interpretation: {weight*100:.0f}% spatial + {(1-weight)*100:.0f}% expression")

    # Initialize model
    model = gf.model.Garfield(
        adata_list=adata.copy(),
        profile=profile,
        data_type="single-modal",
        weight=weight,  # Key parameter being tested
        graph_const_method="KNN",
        used_hvgs=True,
        rna_n_top_features=2000,
        n_components=50,
        n=15,
        augment_type="svd",
        svd_q=5,
        gnn_layer=2,
        hidden_dims=[128, 128],
        bottle_neck_neurons=20,
        conv_type="GAT",
        num_heads=4,
        dropout=0.2,
        edge_batch_size=128,
        node_batch_size=128,
        device_id=device_id,
        seed=seed,
        verbose=verbose
    )

    # Train model
    model.train(
        n_epochs=n_epochs,
        learning_rate=1e-3,
        lambda_edge_recon=1.0,
        lambda_gene_expr_recon=1.0,
        lambda_latent_contrastive_instanceloss=0.1,
        lambda_latent_contrastive_clusterloss=0.1,
        monitor=verbose
    )

    # Generate embeddings
    model.get_latent_representation()

    # Record time and memory
    runtime = time.time() - start_time
    peak_cpu_mb, peak_gpu_mb = memory_tracker.get_peak_memory()

    # Get processed adata
    adata_result = model.adata

    # Compute clusters
    sc.pp.neighbors(adata_result, use_rep='garfield_latent', n_neighbors=15)
    sc.tl.leiden(adata_result, resolution=0.5)

    # Compute metrics
    metrics = {}

    if 'cell_type' in adata_result.obs.columns:
        clustering_metrics = compute_clustering_metrics(
            adata_result,
            cluster_key='leiden',
            label_key='cell_type',
            embedding_key='garfield_latent'
        )
        metrics.update(clustering_metrics)

    # Spatial coherence
    if 'spatial' in adata_result.obsm:
        spatial_metrics = compute_spatial_coherence(
            adata_result,
            label_key='cell_type' if 'cell_type' in adata_result.obs.columns else 'leiden',
            spatial_key='spatial'
        )
        metrics.update(spatial_metrics)

    # Graph statistics
    graph_stats = compute_graph_statistics(adata_result, adj_key='connectivities')
    metrics.update(graph_stats)

    results = {
        'model': model,
        'adata': adata_result,
        'metrics': metrics,
        'weight': weight,
        'runtime_seconds': runtime,
        'peak_cpu_memory_mb': peak_cpu_mb,
        'peak_gpu_memory_mb': peak_gpu_mb,
        'n_epochs': n_epochs,
        'profile': profile
    }

    # Flatten metrics
    for k, v in metrics.items():
        results[k] = v

    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Peak CPU memory: {peak_cpu_mb:.2f} MB")
    if peak_gpu_mb > 0:
        print(f"  Peak GPU memory: {peak_gpu_mb:.2f} MB")
    if metrics:
        key_metrics = {k: v for k, v in metrics.items() if k in ['ARI', 'NMI', 'spatial_coherence_mean']}
        print(f"  Key metrics: {key_metrics}")

    return results


def run_weight_ablation_study(
    adata: ad.AnnData,
    weight_values: List[float] = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    profile: str = "spatial",
    n_epochs: int = 100,
    device_id: int = 0,
    output_dir: str = "./ablation_results",
    seed: int = 42
) -> pd.DataFrame:
    """
    Run ablation study for weight parameter.

    Parameters
    ----------
    adata : AnnData
        Input spatial data
    weight_values : List[float]
        List of weight values to test
    profile : str
        Data profile type
    n_epochs : int
        Training epochs
    device_id : int
        GPU device
    output_dir : str
        Output directory
    seed : int
        Random seed

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    os.makedirs(output_dir, exist_ok=True)
    results_list = []

    print("\n" + "=" * 80)
    print("WEIGHT PARAMETER (ω) ABLATION STUDY")
    print("=" * 80)
    print(f"Testing {len(weight_values)} weight values: {weight_values}")
    print(f"Dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"Profile: {profile}")
    print(f"Training epochs: {n_epochs}")
    print("=" * 80)

    for weight in weight_values:
        result = run_weight_experiment(
            adata=adata,
            weight=weight,
            profile=profile,
            n_epochs=n_epochs,
            device_id=device_id,
            seed=seed,
            verbose=False
        )
        result['experiment_type'] = 'weight_ablation'
        results_list.append(result)

        # Save intermediate results
        save_benchmark_results(
            {k: v for k, v in result.items() if k not in ['model', 'adata']},
            os.path.join(output_dir, "ablation_weight.json"),
            append=True
        )

    # Create dataframe
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'adata']}
                                for r in results_list])

    # Save to CSV
    output_file = os.path.join(output_dir, "ablation_weight.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("WEIGHT ABLATION SUMMARY")
    print("=" * 80)

    # Display key columns
    display_cols = ['weight']
    metric_cols = [c for c in ['ARI', 'NMI', 'ASW_label', 'spatial_coherence_mean',
                               'avg_degree', 'runtime_seconds']
                  if c in results_df.columns]
    display_df = results_df[display_cols + metric_cols]

    print(display_df.to_string(index=False))
    print("=" * 80)

    # Find optimal weight
    if 'ARI' in results_df.columns:
        optimal_idx = results_df['ARI'].idxmax()
        optimal_weight = results_df.loc[optimal_idx, 'weight']
        optimal_ari = results_df.loc[optimal_idx, 'ARI']
        print(f"\n✓ Optimal weight: {optimal_weight:.2f} (ARI = {optimal_ari:.4f})")
        print(f"  Interpretation: {optimal_weight*100:.0f}% spatial + {(1-optimal_weight)*100:.0f}% expression")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for weight parameter in connectivity matrix"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input spatial AnnData file (.h5ad)"
    )
    parser.add_argument(
        "--weight-values",
        nargs="+",
        type=float,
        default=[0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
        help="Weight values to test (default: 0.0 0.2 0.4 0.5 0.6 0.8 1.0)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="spatial",
        choices=["spatial", "RNA", "multi-modal"],
        help="Data profile type"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ablation_results",
        help="Output directory"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Check for spatial coordinates
    if args.profile == "spatial" and 'spatial' not in adata.obsm:
        raise ValueError("Spatial data must have 'spatial' coordinates in adata.obsm['spatial']")

    # Run ablation study
    results_df = run_weight_ablation_study(
        adata=adata,
        weight_values=args.weight_values,
        profile=args.profile,
        n_epochs=args.n_epochs,
        device_id=args.device_id,
        output_dir=args.output_dir,
        seed=args.seed
    )

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {args.output_dir}/ablation_weight.csv")
    print("\nNext steps:")
    print("  1. Visualize results:")
    print(f"     python plot_weight_ablation.py --results-file {args.output_dir}/ablation_weight.csv")
    print("  2. Check reviewer response document:")
    print("     cat reviewer_response_weight_parameter.md")
    print("=" * 80)


if __name__ == "__main__":
    main()
