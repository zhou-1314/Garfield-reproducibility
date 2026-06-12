"""
Comprehensive benchmarking script for Garfield spatial data scalability.

This script benchmarks all major tasks:
1. Dimension reduction (PCA)
2. Graph construction (KNN, Radius, mu_std, Squidpy)
3. Model training (embedding generation)
4. Label transfer (mapping)

Runs experiments on datasets ranging from ~5k to 100k cells/spots.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import Garfield as gf
from benchmark_utils import (
    benchmark_task,
    generate_synthetic_spatial_data,
    save_benchmark_results,
    MemoryTracker
)

import scanpy as sc
import warnings
warnings.filterwarnings('ignore')


def benchmark_dimension_reduction(
    adata,
    n_components: int = 50,
    device_id: int = 0
):
    """
    Benchmark PCA dimension reduction.

    Parameters
    ----------
    adata : AnnData
        Input data
    n_components : int
        Number of PCA components
    device_id : int
        GPU device ID

    Returns
    -------
    dict
        Benchmark results
    """
    with benchmark_task(
        f"Dimension_Reduction_n{adata.n_obs}",
        device_id=device_id,
        verbose=True
    ) as results:
        # Normalize and log-transform
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)

        # Highly variable genes
        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        adata_hvg = adata[:, adata.var['highly_variable']].copy()

        # PCA
        sc.tl.pca(adata_hvg, n_comps=n_components, svd_solver='arpack')

        results['n_cells'] = adata.n_obs
        results['n_genes'] = adata.n_vars
        results['n_hvgs'] = adata_hvg.n_vars
        results['n_components'] = n_components

    return results


def benchmark_graph_construction(
    adata,
    method: str = "KNN",
    n_neighbors: int = 15,
    device_id: int = 0
):
    """
    Benchmark spatial graph construction.

    Parameters
    ----------
    adata : AnnData
        Input data with spatial coordinates in obsm['spatial']
    method : str
        Graph construction method ('KNN', 'Radius', 'mu_std', 'Squidpy')
    n_neighbors : int
        Number of neighbors (or radius for 'Radius' method)
    device_id : int
        GPU device ID

    Returns
    -------
    dict
        Benchmark results
    """
    from Garfield.preprocessing.adj_construction import graph_construction

    with benchmark_task(
        f"Graph_Construction_{method}_n{adata.n_obs}",
        device_id=device_id,
        verbose=True
    ) as results:
        if method == "Squidpy":
            import squidpy as sq
            sq.gr.spatial_neighbors(
                adata,
                coord_type="generic",
                n_neighs=n_neighbors
            )
            # Make symmetric
            adata.obsp["spatial_connectivities"] = adata.obsp[
                "spatial_connectivities"
            ].maximum(adata.obsp["spatial_connectivities"].T)
        else:
            adj = graph_construction(
                adata,
                mode=method,
                k=n_neighbors,
                batch_key=None,
                verbose=False
            )
            adata.obsp["spatial_connectivities"] = adj

        results['n_cells'] = adata.n_obs
        results['method'] = method
        results['n_neighbors'] = n_neighbors
        results['n_edges'] = adata.obsp["spatial_connectivities"].nnz

    return results


def benchmark_model_training(
    adata,
    n_epochs: int = 50,
    device_id: int = 0,
    batch_size: int = 128
):
    """
    Benchmark Garfield model training.

    Parameters
    ----------
    adata : AnnData
        Preprocessed spatial data
    n_epochs : int
        Number of training epochs
    device_id : int
        GPU device ID
    batch_size : int
        Batch size for training

    Returns
    -------
    dict
        Benchmark results
    """
    with benchmark_task(
        f"Model_Training_n{adata.n_obs}",
        device_id=device_id,
        verbose=True
    ) as results:
        # Initialize model
        model = gf.model.Garfield(
            adata_list=adata,
            profile="spatial",
            data_type="single-modal",
            graph_const_method="KNN",
            used_hvgs=True,
            rna_n_top_features=2000,
            n_components=50,
            n=15,
            weight=0.8,
            hidden_dims=[512, 256],
            bottle_neck_neurons=64,
            conv_type="GAT",
            dropout=0.1,
            num_heads=4,
            edge_batch_size=batch_size,
            node_batch_size=batch_size,
            device_id=device_id,
            seed=42,
            verbose=False
        )

        # Train model
        model.train(
            n_epochs=n_epochs,
            learning_rate=1e-3,
            lambda_edge_recon=1.0,
            lambda_gene_expr_recon=1.0,
            lambda_latent_contrastive_instanceloss=0.1,
            lambda_latent_contrastive_clusterloss=0.1,
            monitor=False
        )

        # Generate embeddings
        model.get_latent_representation()

        results['n_cells'] = adata.n_obs
        results['n_genes'] = adata.n_vars
        results['n_epochs'] = n_epochs
        results['batch_size'] = batch_size
        results['latent_dim'] = 64

    return results


def benchmark_label_transfer(
    ref_adata,
    query_adata,
    device_id: int = 0,
    n_neighbors: int = 50
):
    """
    Benchmark label transfer (mapping).

    Parameters
    ----------
    ref_adata : AnnData
        Reference data with embeddings and labels
    query_adata : AnnData
        Query data with embeddings
    device_id : int
        GPU device ID
    n_neighbors : int
        Number of neighbors for KNN classifier

    Returns
    -------
    dict
        Benchmark results
    """
    from Garfield.model.utils import weighted_knn_trainer, weighted_knn_transfer

    with benchmark_task(
        f"Label_Transfer_ref{ref_adata.n_obs}_query{query_adata.n_obs}",
        device_id=device_id,
        verbose=True
    ) as results:
        # Train KNN classifier on reference
        knn_model = weighted_knn_trainer(
            train_adata=ref_adata,
            train_adata_emb="X_pca",
            n_neighbors=n_neighbors
        )

        # Transfer labels to query
        pred_labels, uncertainties = weighted_knn_transfer(
            query_adata=query_adata,
            query_adata_emb="X_pca",
            label_keys="cell_type",
            knn_model=knn_model,
            ref_adata_obs=ref_adata.obs,
            threshold=1.0,
            pred_unknown=False,
            mode="package"
        )

        results['n_ref_cells'] = ref_adata.n_obs
        results['n_query_cells'] = query_adata.n_obs
        results['n_neighbors'] = n_neighbors

    return results


def run_scalability_benchmark(
    dataset_sizes: list = [5000, 10000, 25000, 50000, 100000],
    output_dir: str = "./benchmark_results",
    device_id: int = 0,
    test_all_methods: bool = True,
    n_epochs_training: int = 20
):
    """
    Run complete scalability benchmark across multiple dataset sizes.

    Parameters
    ----------
    dataset_sizes : list
        List of dataset sizes (number of cells) to benchmark
    output_dir : str
        Output directory for results
    device_id : int
        GPU device ID
    test_all_methods : bool
        Whether to test all graph construction methods
    n_epochs_training : int
        Number of epochs for training benchmark

    Returns
    -------
    pd.DataFrame
        Complete benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    all_results = []

    print("\n" + "=" * 80)
    print("GARFIELD SPATIAL SCALABILITY BENCHMARK")
    print("=" * 80)
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Output directory: {output_dir}")
    print(f"Device ID: {device_id}")
    print("=" * 80 + "\n")

    for n_cells in dataset_sizes:
        print(f"\n{'#' * 80}")
        print(f"# BENCHMARKING: {n_cells:,} cells")
        print(f"{'#' * 80}\n")

        # Generate synthetic data
        print(f"Generating synthetic spatial data with {n_cells:,} cells...")
        adata = generate_synthetic_spatial_data(
            n_cells=n_cells,
            n_genes=2000,
            spatial_dim=(100, 100),
            n_cell_types=5,
            random_state=42
        )
        print(f"Data shape: {adata.shape}")

        # ========================================
        # 1. Dimension Reduction
        # ========================================
        print("\n" + "-" * 80)
        print("TASK 1: Dimension Reduction (PCA)")
        print("-" * 80)

        adata_pca = adata.copy()
        result = benchmark_dimension_reduction(
            adata_pca,
            n_components=50,
            device_id=device_id
        )
        result['task_category'] = 'dimension_reduction'
        all_results.append(result)

        # Save intermediate results
        save_benchmark_results(
            result,
            os.path.join(output_dir, "benchmark_results.json"),
            append=True
        )

        # ========================================
        # 2. Graph Construction
        # ========================================
        print("\n" + "-" * 80)
        print("TASK 2: Graph Construction")
        print("-" * 80)

        if test_all_methods:
            methods = ["KNN", "Radius", "mu_std"]
            # Add Squidpy if available and dataset is not too large
            if n_cells <= 50000:
                try:
                    import squidpy
                    methods.append("Squidpy")
                except ImportError:
                    print("Squidpy not available, skipping...")
        else:
            methods = ["KNN"]

        for method in methods:
            adata_graph = adata.copy()
            # Add minimal preprocessing for graph construction
            adata_graph.obsp["spatial"] = adata_graph.obsm["spatial"]

            result = benchmark_graph_construction(
                adata_graph,
                method=method,
                n_neighbors=15,
                device_id=device_id
            )
            result['task_category'] = 'graph_construction'
            all_results.append(result)

            save_benchmark_results(
                result,
                os.path.join(output_dir, "benchmark_results.json"),
                append=True
            )

        # ========================================
        # 3. Model Training & Embedding Generation
        # ========================================
        # Only run for datasets up to 50k cells (training can be slow)
        if n_cells <= 50000:
            print("\n" + "-" * 80)
            print("TASK 3: Model Training & Embedding Generation")
            print("-" * 80)

            adata_train = adata.copy()
            result = benchmark_model_training(
                adata_train,
                n_epochs=n_epochs_training,
                device_id=device_id,
                batch_size=min(256, n_cells // 10)
            )
            result['task_category'] = 'model_training'
            all_results.append(result)

            save_benchmark_results(
                result,
                os.path.join(output_dir, "benchmark_results.json"),
                append=True
            )
        else:
            print("\n" + "-" * 80)
            print("TASK 3: Model Training - SKIPPED (dataset too large)")
            print("-" * 80)

        # ========================================
        # 4. Label Transfer (Mapping)
        # ========================================
        print("\n" + "-" * 80)
        print("TASK 4: Label Transfer (Mapping)")
        print("-" * 80)

        # Use PCA embeddings for label transfer benchmark
        # Create reference (70%) and query (30%) split
        n_ref = int(n_cells * 0.7)
        indices = np.arange(n_cells)
        np.random.shuffle(indices)

        ref_adata = adata_pca[indices[:n_ref]].copy()
        query_adata = adata_pca[indices[n_ref:]].copy()

        result = benchmark_label_transfer(
            ref_adata,
            query_adata,
            device_id=device_id,
            n_neighbors=50
        )
        result['task_category'] = 'label_transfer'
        all_results.append(result)

        save_benchmark_results(
            result,
            os.path.join(output_dir, "benchmark_results.json"),
            append=True
        )

        print(f"\n{'#' * 80}")
        print(f"# COMPLETED: {n_cells:,} cells")
        print(f"{'#' * 80}\n")

    # Convert to DataFrame and save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(
        os.path.join(output_dir, "benchmark_results.csv"),
        index=False
    )

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE!")
    print(f"Results saved to: {output_dir}")
    print("=" * 80 + "\n")

    return results_df


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Garfield spatial data scalability"
    )
    parser.add_argument(
        "--dataset-sizes",
        nargs="+",
        type=int,
        default=[5000, 10000, 25000, 50000, 100000],
        help="Dataset sizes to benchmark (number of cells)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--test-all-methods",
        action="store_true",
        help="Test all graph construction methods (KNN, Radius, mu_std, Squidpy)"
    )
    parser.add_argument(
        "--n-epochs",
        type=int,
        default=20,
        help="Number of epochs for training benchmark"
    )

    args = parser.parse_args()

    results_df = run_scalability_benchmark(
        dataset_sizes=args.dataset_sizes,
        output_dir=args.output_dir,
        device_id=args.device_id,
        test_all_methods=args.test_all_methods,
        n_epochs_training=args.n_epochs
    )

    print("\nSummary:")
    print(results_df.groupby('task_category')['runtime_seconds'].describe())


if __name__ == "__main__":
    main()
