#!/usr/bin/env python3

# 使用 scarches 环境
import scarches as sca
import scanpy as sc
import anndata as ad
import numpy as np
import os
import time
import argparse
from multigrate.data import organize_multiome_anndatas
from multigrate.model import MultiVAE

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from scipy import io
from tqdm import tqdm


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
def process_data(dataset):
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
    adata_RNA = data["rna"]
    adata_ATAC = data["atac"]
    adata_RNA.var_names_make_unique()
    adata_ATAC.var_names_make_unique()
    adata_RNA.layers['counts'] = adata_RNA.X.copy()
    adata_ATAC.layers['counts'] = adata_ATAC.X.copy()

    # 归一化和对数化 RNA 数据
    sc.pp.normalize_total(adata_RNA)
    sc.pp.log1p(adata_RNA)
    sc.pp.highly_variable_genes(
        adata_RNA,
        flavor="seurat_v3",
        n_top_genes=3000,
        subset=False
    )
    adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable].copy()

    # 归一化和对数化 ATAC 数据
    sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
    sc.pp.log1p(adata_ATAC)
    adata_ATAC.layers['log-norm'] = adata_ATAC.X.copy()
    sc.pp.highly_variable_genes(
        adata_ATAC,
        flavor="seurat_v3",
        n_top_genes=10000,
        subset=False
    )
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable].copy()

    start_time = time.time()

    # 使用 multigrate 组织多模态数据
    adata = organize_multiome_anndatas(
        adatas=[[adata_RNA], [adata_ATAC]],    # RNA-seq 总是第一个
        layers=[['counts'], ['log-norm']], # 如果需要使用 .layers 数据，否则使用 .X
    )

    # 设置 MultiVAE 模型
    MultiVAE.setup_anndata(
        adata,
        rna_indices_end=3000,
    )
    model = MultiVAE(
        adata,
        losses=['nb', 'mse'],
        loss_coefs={'kl': 1e-1, 'integ': 3000},
    )

    # 训练模型
    model.train()

    # 获取潜在表示
    model.get_latent_representation()
    latent = adata.obsm['latent'].copy()
    np.savetxt(os.path.join(res_dir, "Multigrate.csv"), latent, delimiter=',')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"running time: {execution_time} s")

if __name__ == '__main__':
    # 设置参数解析
    parser = argparse.ArgumentParser(description='Process RNA-ATAC data for a specific dataset.')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., 1_ShareSeq_Skin)')

    # 获取命令行参数
    args = parser.parse_args()

    # 执行数据处理
    process_data(args.dataset)