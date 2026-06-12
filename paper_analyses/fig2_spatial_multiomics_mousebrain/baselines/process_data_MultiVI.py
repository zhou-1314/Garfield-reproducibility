#!/usr/bin/env python3

# 使用 scVI 环境
from scipy.io import mmread
from scipy.sparse import csr_matrix
import anndata as ad
import pandas as pd
import scvi
import numpy as np
import scanpy as sc
import os
import time
import argparse

scvi.settings.seed = 420

import pandas as pd
from scipy import io
from tqdm import tqdm

def check_file_exists(file_path):
    """Check if the file or directory exists and raise an error if not."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file or directory {file_path} does not exist.")


def read_rna_data(rna_dir):
    """Read RNA data including barcodes, features, and matrix from the specified directory."""
    # dataset = 'spatial_atac_rna_seq_mouse_brain'
    # file_fold = os.path.join(rna_dir, str(dataset))
    rna = sc.read_h5ad(rna_dir + '/spatial_atac_rna_seq_mouse_brain_batch2.h5ad')
    rna.obs = rna.obs.drop(['ATAC_clusters'], axis=1)
    return rna


def read_atac_data(atac_dir):
    """Read ATAC data from the specified CSV file and return it as an AnnData object."""
    # dataset = 'spatial_atac_rna_seq_mouse_brain'
    # file_fold = os.path.join(atac_dir, str(dataset))
    atac = sc.read_h5ad(atac_dir + '/spatial_atac_rna_seq_mouse_brain_batch2_atac.h5ad')
    return atac


def read_metadata(meta_dir, metadata_file):
    """Read metadata from a CSV file and return it as a DataFrame."""
    # dataset = 'spatial_atac_rna_seq_mouse_brain'
    # file_fold = os.path.join(meta_dir, str(dataset))
    return pd.read_csv(os.path.join(meta_dir, metadata_file), index_col=0)


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
    # check_file_exists(metadata_file)

    # Use progress feedback to track steps
    with tqdm(total=3, desc="Reading multiomics data") as pbar:
        # Read RNA data
        rna = read_rna_data(rna_dir)
        pbar.update(1)

        # Read ADT data
        atac = read_atac_data(atac_dir)
        pbar.update(1)

        # Read metadata
        metadata = read_metadata(rna_dir, metadata_file)
        pbar.update(1)

    # Add metadata to RNA and ADT objects
    rna.obs = rna.obs.join(metadata, how="left")  # Ensure all metadata entries align
    # atac.obs = atac.obs.join(metadata, how="left")

    return {"rna": rna, "atac": atac}

# Function to process the data
def process_data(dataset):
    # 定义路径
    rna_dir = f'/pri_exthome/zhouwg/project/spatial_data/spatial_multi-omics/{dataset}'
    atac_dir = f'/pri_exthome/zhouwg/project/spatial_data/spatial_multi-omics/{dataset}'
    metadata_file = 'spatial_atac_rna_seq_mouse_brain_batch2_cell_type_annotations.csv'
    res_dir = '/pri_exthome/zhouwg/project/Garfield_benchmark/results/benchmark_multimodal/mouse_brain'

    # 判断结果目录是否存在，不存在则创建
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # 读取数据
    data = read_multiomics_data(rna_dir, atac_dir, metadata_file)

    # 访问 RNA 和 ATAC 数据
    adata_RNA = data["rna"]
    adata_ATAC = data["atac"]
    adata_RNA.var_names_make_unique()
    adata_RNA.layers['counts'] = adata_RNA.X.copy()
    adata_ATAC.layers['counts'] = adata_ATAC.X.copy()

    sc.pp.filter_cells(adata_RNA, min_genes=10)
    sc.pp.filter_genes(adata_RNA, min_cells=20)
    sc.pp.filter_cells(adata_ATAC, min_genes=10)
    sc.pp.filter_genes(adata_ATAC, min_cells=1)

    from scipy import sparse
    if not sparse.issparse(adata_ATAC.X):
        adata_ATAC.X = sparse.csr_matrix(adata_ATAC.X)
    if not sparse.issparse(adata_RNA.X):
        adata_RNA.X = sparse.csr_matrix(adata_RNA.X)

    sc.pp.normalize_total(adata_RNA, target_sum=1e4)
    sc.pp.log1p(adata_RNA)
    sc.pp.highly_variable_genes(adata_RNA, flavor="seurat_v3", n_top_genes=3000, subset=False)
    adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable].copy()
    adata_RNA.X = adata_RNA.layers['counts'].copy()

    sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
    sc.pp.log1p(adata_ATAC)
    sc.pp.highly_variable_genes(adata_ATAC, flavor="seurat_v3", n_top_genes=10000, subset=False)
    adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable].copy()
    adata_ATAC.X = adata_ATAC.layers['counts'].copy()

    # 合并 RNA 和 ATAC 数据
    adata_paired = ad.concat([adata_RNA, adata_ATAC], merge="same", axis=1)
    adata_paired.var['modality'] = ['Gene Expression'] * adata_RNA.shape[1] + ['Peaks'] * adata_ATAC.shape[1]

    start_time = time.time()

    # 使用 scvi 来组织数据
    adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired)
    del adata_paired
    adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()

    # 设置并训练 MULTIVI 模型
    scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
    model = scvi.model.MULTIVI(
        adata_mvi,
        n_genes=(adata_mvi.var["modality"] == "Gene Expression").sum(),
        n_regions=(adata_mvi.var["modality"] == "Peaks").sum(),
    )
    model.view_anndata_setup()
    model.train()

    # 获取模型的潜在表示
    latent = model.get_latent_representation()
    np.savetxt(os.path.join(res_dir, "MultiVI.csv"), latent, delimiter=',')

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