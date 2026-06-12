"""
Multi-GPU Training for Garfield using torchrun (Recommended Method)

This script uses torchrun/torch.distributed.launch for better reliability
and compatibility with distributed training.

Usage:
    # For 4 GPUs:
    torchrun --nproc_per_node=4 train_multi_gpu_torchrun.py

    # Or with older PyTorch:
    python -m torch.distributed.launch --nproc_per_node=4 train_multi_gpu_torchrun.py

Advantages over mp.spawn:
    - No pickling issues
    - Better error messages
    - Industry standard
    - Works in all environments
"""

import os
import os.path as osp
import argparse
import torch
import torch.distributed as dist
import scanpy as sc
import Garfield as gf
from Garfield.trainer.distributed import is_main_process


def setup_distributed():
    """Initialize distributed training from environment variables."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not running in distributed mode")
        return None, None, None

    # Set device
    torch.cuda.set_device(local_rank)

    # Initialize process group
    dist.init_process_group(backend='nccl')

    return rank, local_rank, world_size


def get_config(adata, args):
    """Get Garfield configuration."""
    gf.settings.set_workdir(args.output_dir)

    ### modify parameter
    user_config = dict(
        # Data parameters
        adata_list=adata,
        profile='spatial',
        data_type='single-modal',
        sample_col=None,
        weight=0.9,

        # Preprocessing parameters
        graph_const_method='mu_std',  # mu_std, Radius, KNN, Squidpy
        used_hvg=True,
        min_cells=3,
        min_features=0,
        keep_mt=False,
        target_sum=1e4,
        rna_n_top_features=3000,
        n_components=50,
        n_neighbors=5,
        metric='euclidean',
        svd_solver='arpack',

        # Model parameters
        used_pca_feat=False,
        adj_key='connectivities',
        # data split parameters
        edge_val_ratio=0.1,
        edge_test_ratio=0.,
        node_val_ratio=0.1,
        node_test_ratio=0.,
        ## Model options
        augment_type='svd',
        svd_q=5,
        use_FCencoder=True,
        conv_type='GAT',  # GAT or GATv2Conv or GCN
        gnn_layer=2,
        hidden_dims=[128, 128],
        bottle_neck_neurons=20,
        cluster_num=20,
        drop_feature_rate=0.2,
        drop_edge_rate=0.2,
        num_heads=3,
        dropout=0.2,
        concat=True,
        used_edge_weight=True,
        used_DSBN=False,
        used_mmd=False,
        num_neighbors=5,
        loaders_n_hops=2,
        edge_batch_size=4096,
        node_batch_size=128,  # None

        # Training parameters
        batch_size=args.batch_size,
        val_split=0.1,
        test_split=0.0,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=1e-6,
        gradient_clipping=5.0,

        # Loss weights
        include_edge_recon_loss=True,
        include_gene_expr_recon_loss=True,
        lambda_latent_contrastive_instanceloss=1.0,
        lambda_latent_contrastive_clusterloss=0.5,
        lambda_gene_expr_recon=1.,  #
        lambda_edge_recon=1000.,  #
        lambda_latent_adj_recon_loss=2.,
        lambda_omics_recon_mmd_loss=0.2,

        # Other parameters
        latent_key='garfield_latent',
        reload_best_model=True,
        use_early_stopping=True,
        early_stopping_kwargs=None,
        monitor=True,
        verbose=True,
        seed=args.seed,
    )

    config = gf.settings.set_gf_params(
        user_config
    )

    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Garfield Multi-GPU Training')
    parser.add_argument('--data_dir', type=str, default='./data',
                       help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default='./garfield_multi_gpu_output',
                       help='Output directory')
    parser.add_argument('--batch_size', type=int, default=512,
                       help='Batch size per GPU')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    args = parser.parse_args()

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()

    if rank is None:
        print("ERROR: This script must be run with torchrun or torch.distributed.launch")
        print("\nUsage:")
        print("  torchrun --nproc_per_node=4 train_multi_gpu_torchrun.py")
        print("\nOr with older PyTorch:")
        print("  python -m torch.distributed.launch --nproc_per_node=4 train_multi_gpu_torchrun.py")
        return

    if is_main_process(rank):
        print(f"\n{'=' * 60}")
        print(f"Starting Multi-GPU Training with {world_size} GPUs")
        print(f"Current Rank: {rank}, Local Rank: {local_rank}")
        print(f"{'=' * 60}\n")

    # Load data (only log on rank 0)
    if is_main_process(rank):
        print("Loading data...")

    try:
        adata = sc.read_h5ad('/home/weige/zhouwg_data/Garfield_review/data/slideseqv2_mouse_hippocampus.h5ad')
        adata.X = adata.layers['counts'].copy()
        if is_main_process(rank):
            print(f"Total cells: {adata.n_obs}, Total genes: {adata.n_vars}")
    except Exception as e:
        if is_main_process(rank):
            print(f"Error loading data: {e}")
            print("\nPlease update the load_data() function with your dataset paths.")
        return

    # Get configuration
    if is_main_process(rank):
        print("\nSetting up Garfield configuration...")

    config = get_config(adata, args)

    # Set random seed
    torch.manual_seed(args.seed)

    # Initialize model
    if is_main_process(rank):
        print(f"\nInitializing Garfield model...")

    model = gf.model.Garfield(config)

    # Train with distributed parameters
    if is_main_process(rank):
        print(f"\nStarting training...")

    model.train(rank=rank, world_size=world_size)

    # Synchronize before post-processing
    dist.barrier()

    # Compute latent representations and save (only on rank 0)
    if is_main_process(rank):
        print(f"\nSaving model...")
        model_folder_path = f"{args.output_dir}/model"
        model.save(dir_path=model_folder_path,
                   overwrite=True,
                   save_adata=True,
                   adata_file_name="adata_ref.h5ad")

        print("\n" + "=" * 60)
        print("Multi-GPU Training Completed Successfully!")
        print("=" * 60)

    # Cleanup
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
