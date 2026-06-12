#!/usr/bin/env python3

# 使用 env_gf_test 环境
from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata as ad
import pandas as pd
import numpy as np
import scanpy as sc
import os
import time
import argparse

import pandas as pd
from scipy import io
from tqdm import tqdm

import sys
import warnings
sys.path.append('/data2/zhouwg_data/project/Garfield')

# load packages
import Garfield as gf
from mudata import MuData
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

gf.__version__

def check_file_exists(file_path):
    """Check if the file or directory exists and raise an error if not."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file or directory {file_path} does not exist.")


def read_rna_data(rna_dir):
    """Read RNA data including barcodes, features, and matrix from the specified directory."""
    rna_barcodes = pd.read_csv(f'{rna_dir}/barcodes.tsv', header=None, sep='\t', dtype=str)
    rna_features = pd.read_csv(f'{rna_dir}/features.tsv', header=None, sep='\t', dtype=str)
    rna_matrix = io.mmread(f'{rna_dir}/matrix.mtx').T.tocsr()

    rna = sc.AnnData(X=rna_matrix)
    rna.var['gene_symbols'] = rna_features.iloc[:, 0].values
    rna.obs['barcodes'] = rna_barcodes.iloc[:, 0].values
    rna.var_names = rna.var['gene_symbols']  # Set feature names
    rna.obs_names = rna.obs['barcodes']  # Set barcode names
    return rna


def read_atac_data(atac_dir):
    """Read ATAC data from the specified CSV file and return it as an AnnData object."""
    atac_barcodes = pd.read_csv(f'{atac_dir}/barcodes.tsv', header=None, sep='\t', dtype=str)
    atac_features = pd.read_csv(f'{atac_dir}/features.tsv', header=None, sep='\t', dtype=str)
    atac_matrix = io.mmread(f'{atac_dir}/matrix.mtx').T.tocsr()

    atac = sc.AnnData(X=atac_matrix)
    atac.var['gene_symbols'] = atac_features.iloc[:, 0].values
    atac.obs['barcodes'] = atac_barcodes.iloc[:, 0].values
    atac.var_names = atac.var['gene_symbols']  # Set feature names
    atac.obs_names = atac.obs['barcodes']  # Set barcode names
    return atac


def read_metadata(metadata_file):
    """Read metadata from a CSV file and return it as a DataFrame."""
    return pd.read_csv(metadata_file, index_col=0)


def read_multiomics_data(rna_dir, atac_dir, metadata_file):
    """
    Reads and processes RNA and ADT data, adding metadata to both.

    Parameters:
        rna_dir (str): Path to the directory containing RNA matrix, barcodes, and features.
        atac_file (str): Path to the CSV file containing ATAC data.
        metadata_file (str): Path to the CSV file containing metadata.

    Returns:
        dict: A dictionary with keys "rna" and "adt" containing the corresponding AnnData objects.
    """
    # Ensure all paths exist
    check_file_exists(rna_dir)
    check_file_exists(atac_dir)
    check_file_exists(metadata_file)

    # Use progress feedback to track steps
    with tqdm(total=3, desc="Reading multiomics data") as pbar:
        # Read RNA data
        rna = read_rna_data(rna_dir)
        pbar.update(1)

        # Read ADT data
        atac = read_atac_data(atac_dir)
        pbar.update(1)

        # Read metadata
        metadata = read_metadata(metadata_file)
        pbar.update(1)

    # Add metadata to RNA and ADT objects
    rna.obs = rna.obs.join(metadata, how="left")  # Ensure all metadata entries align
    atac.obs = atac.obs.join(metadata, how="left")

    return {"rna": rna, "atac": atac}

# Function to process data
def process_data(dataset, genome):
    # 定义路径
    rna_dir = f'/data2/zhouwg_data/project/Garfield_benchmark/datasets/MultiomicsBenchmark/Raw_data/RNA_ATAC/{dataset}/RNA'
    atac_dir = f'/data2/zhouwg_data/project/Garfield_benchmark/datasets/MultiomicsBenchmark/Raw_data/RNA_ATAC/{dataset}/ATAC'
    metadata_file = f'/data2/zhouwg_data/project/Garfield_benchmark/datasets/MultiomicsBenchmark/Raw_data/RNA_ATAC/{dataset}/meta_data.csv'
    res_dir = f'/data2/zhouwg_data/project/Garfield_benchmark/results/Vertical/RNA_ATAC/{dataset}'

    # 判断结果目录是否存在，不存在则创建
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # 读取数据
    data = read_multiomics_data(rna_dir, atac_dir, metadata_file)

    # 访问 RNA 和 ATAC 数据
    rna = data["rna"]
    atac = data["atac"]
    rna.var_names_make_unique()
    atac.var_names_make_unique()
    rna.layers['counts'] = rna.X.copy()
    atac.layers['counts'] = atac.X.copy()

    rna.var['feature_types'] = 'RNA'
    atac.var['feature_types'] = 'ATAC'

    # split each element for gene score calculation
    split_data = atac.var_names.str.split('-', expand=True).to_frame(index=False)
    split_data.index = atac.var_names
    split_data.columns = ['chr', 'start', 'end']
    atac.var[['chr', 'start', 'end']] = split_data

    mdata = MuData({"rna": rna, "atac": atac})
    del rna, atac

    # set workdir
    workdir = f'/data2/zhouwg_data/project/Garfield_benchmark/results/Vertical/RNA_ATAC/{dataset}'
    gf.settings.set_workdir(workdir)

    ### modify parameter
    user_config = dict(
        ## Input options
        adata_list=mdata,
        profile='multi-modal',  # if it is 'ATAC' or 'ADT', please adjust it.
        data_type='Paired',
        sub_data_type=['rna', 'atac'],
        sample_col=None,  # Specify columns for batch, only one batch, so set `None`
        weight=0.8,

        ## Preprocessing options
        genome=genome,
        use_gene_weight=True,
        use_top_pcs=False,
        used_hvg=True,
        min_cells=3,
        min_features=0,
        keep_mt=False,
        target_sum=1e4,
        rna_n_top_features=3000,
        atac_n_top_features=10000,  # if data belongs to 'ATAC', please specify it.
        n_components=50,
        n_neighbors=5,
        metric='euclidean',
        svd_solver='arpack',
        # datasets
        used_pca_feat=True,
        adj_key='connectivities',

        # data split parameters
        edge_val_ratio=0.1,
        edge_test_ratio=0.,
        node_val_ratio=0.1,
        node_test_ratio=0.,

        ## Model options
        augment_type='svd',
        svd_q=5,
        use_FCencoder=False,
        conv_type='GATv2Conv',  # GAT or GATv2Conv or GCN
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
        # data loader parameters
        num_neighbors=5,
        loaders_n_hops=2,
        edge_batch_size=4096,
        node_batch_size=128,  # None
        # loss parameters
        include_edge_recon_loss=True,
        include_gene_expr_recon_loss=True,
        lambda_latent_contrastive_instanceloss=1.0,
        lambda_latent_contrastive_clusterloss=0.5,
        lambda_gene_expr_recon=1.,
        # To make the model more focused on learning expression features, increase this parameter.
        lambda_edge_recon=5.,
        # To make the model more focused on learning Adjacency graph features, increase this parameter.
        lambda_latent_adj_recon_loss=1.0,
        lambda_omics_recon_mmd_loss=3.,  # If the integration is not strong enough, increase it.
        # train parameters
        n_epochs_no_edge_recon=0,
        learning_rate=0.001,
        weight_decay=1e-05,
        gradient_clipping=5,
        # other parameters
        latent_key='garfield_latent',
        reload_best_model=True,
        use_early_stopping=True,
        early_stopping_kwargs=None,
        monitor=True,
        device_id=1,
        seed=2024,
        verbose=True
    )
    dict_config = gf.settings.set_gf_params(user_config)

    from Garfield.model import Garfield

    # Initialize model
    model = Garfield(dict_config)

    start_time = time.time()
    # Train model
    model.train()
    # Compute latent neighbor graph
    latent_key = 'garfield_latent'
    sc.pp.neighbors(model.adata,
                    use_rep=latent_key,
                    key_added=latent_key)
    # Compute UMAP embedding
    sc.tl.umap(model.adata,
               neighbors_key=latent_key)
    # Save trained model
    model_folder_path = f"{workdir}/model"
    os.makedirs(model_folder_path, exist_ok=True)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"running time: {execution_time} s")

    # 获取潜在表示
    latent = model.adata.obsm['garfield_latent'].copy()
    np.savetxt(os.path.join(res_dir, "Garfield.csv"), latent, delimiter=',')

    del model.adata.obsm['feat']
    model.save(dir_path=model_folder_path,
               overwrite=True,
               save_adata=True,
               adata_file_name="adata_ref.h5ad")



if __name__ == '__main__':
    # 设置参数解析
    parser = argparse.ArgumentParser(description='Process RNA-ATAC data for a specific dataset.')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., 1_ShareSeq_Skin)')
    parser.add_argument('genome', type=str, help='reference genome (e.g., mm10)')

    # 获取命令行参数
    args = parser.parse_args()

    # 执行数据处理
    process_data(args.dataset, args.genome)