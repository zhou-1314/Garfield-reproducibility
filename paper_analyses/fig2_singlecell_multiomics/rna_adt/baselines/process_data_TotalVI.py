import anndata as ad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mudata as md
# import muon
from scipy.io import mmread
from scipy.sparse import csr_matrix
import scanpy as sc
import scvi
import torch
import time
import os
import sys

## 使用 scVI 环境
import argparse
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


def read_adt_data(adt_dir):
    """Read ADT data from the specified CSV file and return it as an AnnData object."""
    adata_ADT = pd.read_csv(f'{adt_dir}/ADT.csv', index_col=0)
    adata_ADT.columns = adata_ADT.columns.str.replace('.', '-')
    adata_ADT.index = adata_ADT.index.str.replace('.', '_')
    adata_ADT.index = adata_ADT.index.str.replace('-', '_')
    adata_ADT = ad.AnnData(adata_ADT.T)
    adata_ADT.X = adata_ADT.X.astype(np.float64)
    adata_ADT.layers["counts"] = adata_ADT.X.copy()
    return adata_ADT

def read_metadata(metadata_file):
    """Read metadata from a CSV file and return it as a DataFrame."""
    return pd.read_csv(metadata_file, index_col=0)


def read_multiomics_data(rna_dir, adt_dir, metadata_file):
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
    check_file_exists(adt_dir)
    check_file_exists(metadata_file)

    # Use progress feedback to track steps
    with tqdm(total=3, desc="Reading multiomics data") as pbar:
        # Read RNA data
        rna = read_rna_data(rna_dir)
        pbar.update(1)

        # Read ADT data
        adt = read_adt_data(adt_dir)
        pbar.update(1)

        # Read metadata
        metadata = read_metadata(metadata_file)
        pbar.update(1)

    # Add metadata to RNA and ADT objects
    rna.obs = rna.obs.join(metadata, how="left")  # Ensure all metadata entries align
    # adt.obs = adt.obs.join(metadata, how="left")

    return {"rna": rna, "adt": adt}

# Function to process data
def process_data(dataset):
    # 定义路径
    rna_dir = f'/data2/zhouwg_data/project/Garfield_benchmark/datasets/MultiomicsBenchmark/Raw_data/RNA_Protein/{dataset}/RNA'
    adt_dir = f'/data2/zhouwg_data/project/Garfield_benchmark/datasets/MultiomicsBenchmark/Raw_data/RNA_Protein/{dataset}'
    metadata_file = f'/data2/zhouwg_data/project/Garfield_benchmark/datasets/MultiomicsBenchmark/Raw_data/RNA_Protein/{dataset}/metadata.csv'
    res_dir = f'/data2/zhouwg_data/project/Garfield_benchmark/results/Vertical/RNA_Protein/{dataset}'

    # 判断结果目录是否存在，不存在则创建
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    # 读取数据
    data = read_multiomics_data(rna_dir, adt_dir, metadata_file)

    # 访问 RNA 和 ADT 数据
    adata_RNA = data["rna"]
    adata_ADT = data["adt"]
    adata_RNA.var_names_make_unique()
    adata_ADT.var_names_make_unique()
    adata_RNA.layers['counts'] = adata_RNA.X.copy()
    adata_ADT.layers['counts'] = adata_ADT.X.copy()

    # 归一化和对数化 RNA 数据
    sc.pp.normalize_total(adata_RNA)
    sc.pp.log1p(adata_RNA)

    # 归一化和对数化 ADT 数据
    # muon.prot.pp.clr(adata_ADT)
    # adata_ADT.layers['clr'] = adata_ADT.X.copy()

    modality = ['batch1'] * adata_RNA.shape[0]
    adata_RNA.obs['batch'] = modality

    start_time = time.time()

    mdata = md.MuData({"rna": adata_RNA, "adt": adata_ADT})
    # Place subsetted counts in a new modality
    sc.pp.highly_variable_genes(
        mdata.mod["rna"],
        n_top_genes=3000,
        flavor="seurat_v3",
        batch_key="batch",
        layer="counts",
    )
    mdata.mod["rna_subset"] = mdata.mod["rna"][
                              :, mdata.mod["rna"].var["highly_variable"]].copy()
    mdata.update()

    scvi.model.TOTALVI.setup_mudata(
        mdata,
        rna_layer="counts",
        protein_layer=None,
        batch_key="batch",
        modalities={
            "rna_layer": "rna_subset",
            "protein_layer": "adt",
            "batch_key": "rna_subset",
        },
    )

    # run TotalVI
    model = scvi.model.TOTALVI(mdata)
    model.train()

    adata_RNA.obsm["X_totalVI"] = model.get_latent_representation()
    latent = adata_RNA.obsm['X_totalVI'].copy()
    np.savetxt(os.path.join(res_dir, "TotalVI.csv"), latent, delimiter=',')

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"running time: {execution_time} s")

if __name__ == '__main__':
    # 设置参数解析
    parser = argparse.ArgumentParser(description='Process RNA-ADT data for a specific dataset.')
    parser.add_argument('dataset', type=str, help='Dataset name (e.g., 1_ShareSeq_Skin)')

    # 获取命令行参数
    args = parser.parse_args()

    # 执行数据处理
    process_data(args.dataset)


