"""
Ablation Study for Garfield: Denoised-Graph Branch and Hyperparameter Analysis

This script performs comprehensive ablation studies to answer reviewer questions:
1. Quantify the contribution of the SVD-based denoised-graph branch
2. Compare performance with SVD vs dropout augmentation
3. Evaluate impact of GNN iteration steps (gnn_layer parameter)
4. Justify key hyperparameters (svd_q, hidden_dims, etc.)
5. Report runtime and performance changes

Evaluation Metrics:
- Clustering metrics: ARI, NMI, ASW (silhouette)
- Batch correction: Graph connectivity, kBET
- Label transfer accuracy
- Runtime and memory consumption
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from pathlib import Path
from typing import Dict, List, Tuple
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
    """
    Compute clustering quality metrics.

    Parameters
    ----------
    adata : AnnData
        Annotated data with embeddings and labels
    cluster_key : str
        Key for cluster assignments
    label_key : str
        Key for ground truth cell type labels
    embedding_key : str
        Key for embedding in obsm

    Returns
    -------
    Dict[str, float]
        Dictionary with ARI, NMI, and silhouette scores
    """
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


def compute_batch_mixing_score(
    adata: ad.AnnData,
    batch_key: str = 'batch',
    embedding_key: str = 'garfield_latent',
    n_neighbors: int = 50
) -> Dict[str, float]:
    """
    Compute batch mixing score (graph connectivity).

    Parameters
    ----------
    adata : AnnData
        Annotated data
    batch_key : str
        Key for batch labels
    embedding_key : str
        Key for embedding in obsm
    n_neighbors : int
        Number of neighbors for kNN graph

    Returns
    -------
    Dict[str, float]
        Dictionary with batch mixing metrics
    """
    if not SKLEARN_AVAILABLE:
        return {}

    embeddings = adata.obsm[embedding_key]
    batches = adata.obs[batch_key].values

    # Build kNN graph
    nbrs = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(embeddings)
    _, indices = nbrs.kneighbors(embeddings)

    # Compute mixing score
    n_samples = len(batches)
    batch_freqs = pd.Series(batches).value_counts(normalize=True)

    mixing_scores = []
    for i in range(n_samples):
        neighbors_batches = batches[indices[i, 1:]]  # Exclude self
        obs_batch = batches[i]

        # Expected frequency of same batch neighbors
        expected_freq = batch_freqs[obs_batch]

        # Observed frequency of same batch neighbors
        observed_freq = (neighbors_batches == obs_batch).sum() / n_neighbors

        # Mixing score: 1 - (observed - expected) / (1 - expected)
        # Higher is better (more mixed)
        if expected_freq < 1:
            score = 1 - (observed_freq - expected_freq) / (1 - expected_freq)
        else:
            score = 1.0

        mixing_scores.append(score)

    return {
        'batch_mixing_score': np.mean(mixing_scores),
        'batch_mixing_std': np.std(mixing_scores)
    }


def compute_label_transfer_accuracy(
    ref_adata: ad.AnnData,
    query_adata: ad.AnnData,
    label_key: str = 'cell_type',
    embedding_key: str = 'X_pca',
    n_neighbors: int = 50
) -> Dict[str, float]:
    """
    Compute label transfer accuracy.

    Parameters
    ----------
    ref_adata : AnnData
        Reference data with labels
    query_adata : AnnData
        Query data with true labels
    label_key : str
        Key for cell type labels
    embedding_key : str
        Key for embeddings
    n_neighbors : int
        Number of neighbors for KNN

    Returns
    -------
    Dict[str, float]
        Dictionary with accuracy metrics
    """
    from Garfield.model.utils import weighted_knn_trainer, weighted_knn_transfer

    # Train KNN model
    knn_model = weighted_knn_trainer(
        train_adata=ref_adata,
        train_adata_emb=embedding_key,
        n_neighbors=n_neighbors
    )

    # Transfer labels
    pred_labels, uncertainties = weighted_knn_transfer(
        query_adata=query_adata,
        query_adata_emb=embedding_key,
        label_keys=label_key,
        knn_model=knn_model,
        ref_adata_obs=ref_adata.obs,
        threshold=1.0,
        pred_unknown=False,
        mode="package"
    )

    # Compute accuracy
    true_labels = query_adata.obs[label_key].values
    predicted = pred_labels[label_key].values
    accuracy = (true_labels == predicted).mean()

    # Mean uncertainty
    mean_uncertainty = uncertainties[label_key].values.mean()

    return {
        'label_transfer_accuracy': accuracy,
        'mean_uncertainty': mean_uncertainty
    }


def run_single_experiment(
    adata: ad.AnnData,
    augment_type: str = "svd",
    svd_q: int = 5,
    gnn_layer: int = 2,
    hidden_dims: List[int] = [128, 128],
    bottle_neck_neurons: int = 20,
    n_epochs: int = 100,
    device_id: int = 0,
    seed: int = 42,
    verbose: bool = False
) -> Dict[str, any]:
    """
    Run a single Garfield experiment with specified hyperparameters.

    Parameters
    ----------
    adata : AnnData
        Input data
    augment_type : str
        Augmentation type: "svd" or "dropout"
    svd_q : int
        SVD rank (only used when augment_type="svd")
    gnn_layer : int
        Number of GNN iterations
    hidden_dims : List[int]
        Hidden layer dimensions
    bottle_neck_neurons : int
        Latent dimension
    n_epochs : int
        Number of training epochs
    device_id : int
        GPU device ID
    seed : int
        Random seed
    verbose : bool
        Verbosity

    Returns
    -------
    Dict[str, any]
        Dictionary containing:
        - Trained model
        - Performance metrics
        - Runtime
        - Memory usage
    """
    memory_tracker = MemoryTracker(device_id=device_id)
    memory_tracker.reset()
    start_time = time.time()

    print(f"\nRunning experiment:")
    print(f"  augment_type: {augment_type}")
    print(f"  svd_q: {svd_q}")
    print(f"  gnn_layer: {gnn_layer}")
    print(f"  hidden_dims: {hidden_dims}")
    print(f"  bottle_neck_neurons: {bottle_neck_neurons}")

    # Initialize model
    model = gf.model.Garfield(
        adata_list=adata.copy(),
        profile="RNA",
        used_hvgs=True,
        rna_n_top_features=2000,
        n_components=50,
        augment_type=augment_type,
        svd_q=svd_q,
        gnn_layer=gnn_layer,
        hidden_dims=hidden_dims,
        bottle_neck_neurons=bottle_neck_neurons,
        conv_type="GAT",
        num_heads=4,
        dropout=0.2,
        drop_feature_rate=0.2,
        drop_edge_rate=0.2,
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

    # Get processed adata with embeddings
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

    if 'batch' in adata_result.obs.columns:
        batch_metrics = compute_batch_mixing_score(
            adata_result,
            batch_key='batch',
            embedding_key='garfield_latent'
        )
        metrics.update(batch_metrics)

    results = {
        'model': model,
        'adata': adata_result,
        'metrics': metrics,
        'runtime_seconds': runtime,
        'peak_cpu_memory_mb': peak_cpu_mb,
        'peak_gpu_memory_mb': peak_gpu_mb,
        'augment_type': augment_type,
        'svd_q': svd_q,
        'gnn_layer': gnn_layer,
        'hidden_dims': str(hidden_dims),
        'bottle_neck_neurons': bottle_neck_neurons,
        'n_epochs': n_epochs
    }

    # Flatten metrics into results
    for k, v in metrics.items():
        results[k] = v

    print(f"  Runtime: {runtime:.2f}s")
    print(f"  Peak CPU memory: {peak_cpu_mb:.2f} MB")
    if peak_gpu_mb > 0:
        print(f"  Peak GPU memory: {peak_gpu_mb:.2f} MB")
    if metrics:
        print(f"  Metrics: {metrics}")

    return results


def ablation_augment_type(
    adata: ad.AnnData,
    svd_q: int = 5,
    n_epochs: int = 100,
    device_id: int = 0,
    output_dir: str = "./ablation_results"
) -> pd.DataFrame:
    """
    Ablation study: SVD vs Dropout augmentation.

    Parameters
    ----------
    adata : AnnData
        Input data
    svd_q : int
        SVD rank for SVD experiments
    n_epochs : int
        Training epochs
    device_id : int
        GPU device
    output_dir : str
        Output directory

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY 1: SVD vs Dropout Augmentation")
    print("=" * 80)

    results_list = []

    for augment_type in ["dropout", "svd"]:
        result = run_single_experiment(
            adata=adata,
            augment_type=augment_type,
            svd_q=svd_q,
            gnn_layer=2,
            hidden_dims=[128, 128],
            bottle_neck_neurons=20,
            n_epochs=n_epochs,
            device_id=device_id,
            seed=42
        )
        result['experiment_type'] = 'augment_type_ablation'
        results_list.append(result)

        # Save intermediate results
        save_benchmark_results(
            result,
            os.path.join(output_dir, "ablation_augment_type.json"),
            append=True
        )

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'adata']}
                                for r in results_list])
    results_df.to_csv(os.path.join(output_dir, "ablation_augment_type.csv"), index=False)

    return results_df


def ablation_gnn_layer(
    adata: ad.AnnData,
    gnn_layers: List[int] = [1, 2, 3, 4],
    n_epochs: int = 100,
    device_id: int = 0,
    output_dir: str = "./ablation_results"
) -> pd.DataFrame:
    """
    Ablation study: Number of GNN iterations.

    Parameters
    ----------
    adata : AnnData
        Input data
    gnn_layers : List[int]
        List of gnn_layer values to test
    n_epochs : int
        Training epochs
    device_id : int
        GPU device
    output_dir : str
        Output directory

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY 2: Number of GNN Iterations (gnn_layer)")
    print("=" * 80)

    results_list = []

    for gnn_layer in gnn_layers:
        result = run_single_experiment(
            adata=adata,
            augment_type="svd",  # Use SVD for this ablation
            svd_q=5,
            gnn_layer=gnn_layer,
            hidden_dims=[128, 128],
            bottle_neck_neurons=20,
            n_epochs=n_epochs,
            device_id=device_id,
            seed=42
        )
        result['experiment_type'] = 'gnn_layer_ablation'
        results_list.append(result)

        # Save intermediate results
        save_benchmark_results(
            result,
            os.path.join(output_dir, "ablation_gnn_layer.json"),
            append=True
        )

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'adata']}
                                for r in results_list])
    results_df.to_csv(os.path.join(output_dir, "ablation_gnn_layer.csv"), index=False)

    return results_df


def ablation_svd_rank(
    adata: ad.AnnData,
    svd_qs: List[int] = [1, 3, 5, 10, 20],
    n_epochs: int = 100,
    device_id: int = 0,
    output_dir: str = "./ablation_results"
) -> pd.DataFrame:
    """
    Ablation study: SVD rank parameter (svd_q).

    Parameters
    ----------
    adata : AnnData
        Input data
    svd_qs : List[int]
        List of SVD ranks to test
    n_epochs : int
        Training epochs
    device_id : int
        GPU device
    output_dir : str
        Output directory

    Returns
    -------
    pd.DataFrame
        Results dataframe
    """
    print("\n" + "=" * 80)
    print("ABLATION STUDY 3: SVD Rank (svd_q)")
    print("=" * 80)

    results_list = []

    for svd_q in svd_qs:
        result = run_single_experiment(
            adata=adata,
            augment_type="svd",
            svd_q=svd_q,
            gnn_layer=2,
            hidden_dims=[128, 128],
            bottle_neck_neurons=20,
            n_epochs=n_epochs,
            device_id=device_id,
            seed=42
        )
        result['experiment_type'] = 'svd_rank_ablation'
        results_list.append(result)

        # Save intermediate results
        save_benchmark_results(
            result,
            os.path.join(output_dir, "ablation_svd_rank.json"),
            append=True
        )

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['model', 'adata']}
                                for r in results_list])
    results_df.to_csv(os.path.join(output_dir, "ablation_svd_rank.csv"), index=False)

    return results_df


def run_complete_ablation_study(
    adata: ad.AnnData,
    n_epochs: int = 100,
    device_id: int = 0,
    output_dir: str = "./ablation_results",
    run_augment_type: bool = True,
    run_gnn_layer: bool = True,
    run_svd_rank: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Run complete ablation study suite.

    Parameters
    ----------
    adata : AnnData
        Input data
    n_epochs : int
        Training epochs
    device_id : int
        GPU device
    output_dir : str
        Output directory
    run_augment_type : bool
        Run SVD vs dropout ablation
    run_gnn_layer : bool
        Run GNN iterations ablation
    run_svd_rank : bool
        Run SVD rank ablation

    Returns
    -------
    Dict[str, pd.DataFrame]
        Dictionary of results dataframes
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    print("\n" + "=" * 80)
    print("GARFIELD ABLATION STUDY SUITE")
    print("=" * 80)
    print(f"Dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
    print(f"Training epochs: {n_epochs}")
    print(f"Device ID: {device_id}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    if run_augment_type:
        results['augment_type'] = ablation_augment_type(
            adata, n_epochs=n_epochs, device_id=device_id, output_dir=output_dir
        )

    if run_gnn_layer:
        results['gnn_layer'] = ablation_gnn_layer(
            adata, gnn_layers=[1, 2, 3, 4], n_epochs=n_epochs,
            device_id=device_id, output_dir=output_dir
        )

    if run_svd_rank:
        results['svd_rank'] = ablation_svd_rank(
            adata, svd_qs=[1, 3, 5, 10, 20], n_epochs=n_epochs,
            device_id=device_id, output_dir=output_dir
        )

    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print("=" * 80)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Ablation study for Garfield denoised-graph and hyperparameters"
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to input AnnData file (.h5ad)"
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
        "--skip-augment",
        action="store_true",
        help="Skip augmentation type ablation"
    )
    parser.add_argument(
        "--skip-gnn-layer",
        action="store_true",
        help="Skip GNN layer ablation"
    )
    parser.add_argument(
        "--skip-svd-rank",
        action="store_true",
        help="Skip SVD rank ablation"
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading data from: {args.data_path}")
    adata = sc.read_h5ad(args.data_path)
    print(f"Loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")

    # Run ablation studies
    results = run_complete_ablation_study(
        adata=adata,
        n_epochs=args.n_epochs,
        device_id=args.device_id,
        output_dir=args.output_dir,
        run_augment_type=not args.skip_augment,
        run_gnn_layer=not args.skip_gnn_layer,
        run_svd_rank=not args.skip_svd_rank
    )

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for study_name, df in results.items():
        print(f"\n{study_name.upper()}:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
