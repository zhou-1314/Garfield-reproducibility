"""
Multi-GPU Training Example for Garfield Model

This script demonstrates how to train the Garfield model using multiple GPUs
with PyTorch DistributedDataParallel (DDP) following PyTorch Geometric patterns.

Usage:
    python train_multi_gpu.py

Requirements:
    - Multiple CUDA-capable GPUs
    - PyTorch with CUDA support
    - PyTorch Geometric
    - Garfield package

References:
    - https://pytorch-geometric.readthedocs.io/en/latest/tutorial/multi_gpu_vanilla.html
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/multi_gpu/distributed_sampling.py
"""

import os
import os.path as osp
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import scanpy as sc
import Garfield as gf
from Garfield.trainer.distributed import setup_distributed, cleanup_distributed, is_main_process


def run_training(rank: int, world_size: int, adata, config: dict):
    """
    Training function that runs on each GPU process.

    Parameters
    ----------
    rank : int
        Rank of the current process (GPU ID).
    world_size : int
        Total number of processes (GPUs).
    adata : AnnData
        The annotated data matrix (should be in shared memory).
    config : dict
        Configuration dictionary with training parameters.
    """
    # Initialize distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.get('master_port', '12355')
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    if is_main_process(rank):
        print(f"\n{'=' * 60}")
        print(f"Starting Multi-GPU Training with {world_size} GPUs")
        print(f"{'=' * 60}\n")

    # Move features to device for faster access (following PyG pattern)
    # Note: AnnData can't be directly moved to GPU, but we prepare the data
    # The trainer will handle device placement internally

    # Initialize Garfield model
    # Note: Model initialization should use the same seed for all processes
    # but the trainer will handle DDP wrapping
    torch.manual_seed(config.get('seed', 2024))

    if is_main_process(rank):
        print(f"Initializing Garfield model on rank {rank}...")

    model = gf.model.Garfield(config)

    # The trainer will automatically handle DDP wrapping when rank and world_size are provided
    if is_main_process(rank):
        print(f"Starting training on rank {rank}...")

    model.train(rank=rank, world_size=world_size)

    # Compute latent representations (only on rank 0 to avoid duplication)
    if is_main_process(rank):
        print("\nComputing latent representations...")
        model.get_latent_representation(use_rep="X_gf")

    # Synchronize before cleanup
    dist.barrier()

    # Save model (only on rank 0)
    if is_main_process(rank):
        print(f"\nSaving model...")
        model.save(adata=adata)
        print("Training completed!")

    # Cleanup distributed training
    dist.destroy_process_group()


def main():
    """
    Main function that sets up the data and spawns training processes.
    """
    print("Loading data...")

    # Load your data here
    # Example: Using seqFISH Mouse Organogenesis dataset from the tutorial
    data_dir = "./data"
    adata_list = []

    # Load multiple batches (adjust paths as needed)
    for i in range(1, 7):
        adata_path = osp.join(data_dir, f"embryo{i}_filtered.h5ad")
        if osp.exists(adata_path):
            adata = sc.read_h5ad(adata_path)
            adata_list.append(adata)
        else:
            print(f"Warning: {adata_path} not found, skipping...")

    if len(adata_list) == 0:
        print("Error: No data files found. Please check your data directory.")
        print("Please update the data loading section with your dataset paths.")
        return

    # Concatenate batches
    adata = sc.concat(adata_list, label="batch", keys=[f"embryo{i}" for i in range(1, len(adata_list) + 1)])
    print(f"Total cells: {adata.n_obs}, Total genes: {adata.n_vars}")

    # Configure Garfield parameters
    print("\nSetting up Garfield configuration...")
    gf.settings.set_workdir("./garfield_multi_gpu_output")

    config = gf.settings.set_gf_params(
        # Data parameters
        adata_list=adata_list,
        profile="spatial",
        data_type="single",
        sub_data_type="RNA",
        sample_col="batch",

        # Preprocessing parameters
        single_n_top_genes=3000,
        n_pcs=50,
        graph_const_method="mu_std",

        # Model parameters
        hidden_dims=[512, 48],
        bottle_neck_neurons=10,
        cluster_num=None,  # Will be determined automatically
        used_hvg=True,
        used_DSBN=True,
        used_recon_exp=True,

        # Training parameters
        batch_size=512,
        val_split=0.1,
        test_split=0.0,
        epochs=100,
        learning_rate=1e-3,
        weight_decay=1e-6,
        gradient_clipping=5.0,

        # Loss weights
        lambda_edge_recon=1.0,
        lambda_gene_expr_recon=1.0,
        lambda_latent_adj_recon_loss=1.0,
        lambda_latent_contrastive_instanceloss=0.1,
        lambda_latent_contrastive_clusterloss=0.1,
        lambda_omics_recon_mmd_loss=0.0,

        # Other parameters
        monitor_only_val_losses=True,
        verbose=True,
        seed=2024,

        # Multi-GPU specific
        master_port='12355',  # Port for distributed communication
    )

    # Check available GPUs
    world_size = torch.cuda.device_count()
    if world_size < 2:
        print(f"\nWarning: Only {world_size} GPU(s) detected.")
        print("Multi-GPU training requires at least 2 GPUs.")
        print("The script will continue with single-GPU training.")
        if world_size == 1:
            # Fall back to single-GPU training
            print("\nRunning single-GPU training...")
            model = gf.model.Garfield(config)
            model.train()
            model.get_latent_representation(use_rep="X_gf")
            model.save(adata=adata)
        else:
            print("No GPUs available. Exiting.")
        return

    print(f"\nUsing {world_size} GPUs for distributed training!")

    # Important: Load the dataset BEFORE spawning processes to leverage shared memory
    # This is a key optimization from the PyG tutorial
    print("\nSpawning training processes...")

    # Spawn training processes (one per GPU)
    mp.spawn(
        run_training,
        args=(world_size, adata, config),
        nprocs=world_size,
        join=True
    )

    print("\n" + "=" * 60)
    print("Multi-GPU Training Completed Successfully!")
    print("=" * 60)


if __name__ == '__main__':
    # Set multiprocessing start method
    # 'spawn' is recommended for CUDA
    mp.set_start_method('spawn', force=True)

    main()
