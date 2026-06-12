#!/usr/bin/env python
"""Run MultiGATE on the P22 spatial ATAC-RNA benchmark."""

from __future__ import print_function

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse

from common import (
    ATAC_PATH,
    DEFAULT_RESULTS_DIR,
    RNA_PATH,
    attach_common_obs,
    duration_seconds,
    ensure_dir,
    evaluate_embedding,
    now_seconds,
    read_labels,
    write_failure,
    write_status,
)


METHOD = "MultiGATE"
DEFAULT_GTF = Path(
    "/data2/zhouwg_data/project/Garfield-reproducibility/"
    "benchmark_unimodal/nichecompass/data/gene_annotations/"
    "gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR / "multigate"))
    parser.add_argument("--rna", default=str(RNA_PATH))
    parser.add_argument("--atac", default=str(ATAC_PATH))
    parser.add_argument("--gtf", default=str(DEFAULT_GTF))
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-obs", type=int, default=None)
    parser.add_argument("--rna-hvg", type=int, default=3000)
    parser.add_argument("--atac-hvg", type=int, default=8000)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-dim", type=int, default=30)
    parser.add_argument("--knn", type=int, default=6)
    return parser.parse_args()


def maybe_subset(adata, max_obs, seed):
    if max_obs is None or adata.n_obs <= max_obs:
        return adata
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_obs, replace=False))
    return adata[idx].copy()


def sanitize_x(adata):
    adata = adata.copy()
    if sparse.issparse(adata.X):
        adata.X = adata.X.astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    return adata


def preprocess_rna(adata, n_top_genes):
    adata = sanitize_x(adata)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars), flavor="seurat_v3")
    return adata


def preprocess_atac(adata, n_top_genes):
    adata = sanitize_x(adata)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.highly_variable_genes(adata, n_top_genes=min(n_top_genes, adata.n_vars), flavor="seurat_v3")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def parse_gtf_genes(gtf_path, gene_names):
    columns = ["seqname", "source", "feature", "start", "end", "score", "strand", "frame", "attribute"]
    gtf = pd.read_csv(gtf_path, sep="\t", header=None, comment="#", names=columns)
    gtf = gtf.loc[gtf["feature"] == "gene"].copy()
    attrs = gtf["attribute"].str.extract(r'gene_name "([^"]+)"')
    gtf["gene_name"] = attrs[0]
    gtf = gtf.dropna(subset=["gene_name"])
    gtf = gtf.drop_duplicates("gene_name", keep="last").set_index("gene_name")
    gtf = gtf.reindex(pd.Index(gene_names))
    gene_df = pd.DataFrame(
        {
            "Gene": gtf.index.astype(str),
            "chrom": gtf["seqname"].astype(str).values,
            "chromStart": pd.to_numeric(gtf["start"], errors="coerce").values,
            "chromEnd": pd.to_numeric(gtf["end"], errors="coerce").values,
            "strand": gtf["strand"].astype(str).values,
        },
        index=gtf.index.astype(str),
    )
    gene_df = gene_df.dropna(subset=["chromStart", "chromEnd"])
    gene_df["chromStart"] = gene_df["chromStart"].astype(int)
    gene_df["chromEnd"] = gene_df["chromEnd"].astype(int)
    return gene_df


def parse_peaks(peak_names):
    split = pd.Series(peak_names, index=peak_names).str.extract(r"^(?P<chrom>[^:]+):(?P<chromStart>\d+)-(?P<chromEnd>\d+)$")
    split["Peak"] = split.index.astype(str)
    split = split.dropna()
    split["chromStart"] = split["chromStart"].astype(int)
    split["chromEnd"] = split["chromEnd"].astype(int)
    return split


def build_gene_peak_net(rna, atac, gtf_path, window=2000, states=("Top",), min_edges_per_gene=1):
    rna_vars = rna.var_names[rna.var["highly_variable"].values]
    atac_vars = atac.var_names[atac.var["highly_variable"].values]
    genes = parse_gtf_genes(gtf_path, rna_vars)
    peaks = parse_peaks(pd.Index(atac_vars))
    rows = []
    states = set(states)
    for chrom, chrom_genes in genes.groupby("chrom", sort=False):
        chrom_peaks = peaks.loc[peaks["chrom"] == chrom]
        if chrom_peaks.empty:
            continue
        peak_starts = chrom_peaks["chromStart"].to_numpy()
        peak_ends = chrom_peaks["chromEnd"].to_numpy()
        peak_names = chrom_peaks["Peak"].to_numpy()
        for gene, row in chrom_genes.iterrows():
            gene_start = int(row["chromStart"])
            gene_end = int(row["chromEnd"])
            distance = np.abs(peak_starts - gene_start)
            overlap = ((peak_starts <= gene_end) & (peak_starts >= gene_start)) | (
                (peak_ends >= gene_start) & (peak_ends <= gene_end)
            )
            top = (peak_starts < gene_start) & (peak_ends < gene_start)
            state = np.where(overlap, "Overlap", np.where(top, "Top", "Bottom"))
            mask = (distance <= window) & np.isin(state, list(states))
            selected = np.where(mask)[0]
            if selected.size == 0 and min_edges_per_gene:
                order = np.argsort(distance)
                selected = order[:min(min_edges_per_gene, order.size)]
            for idx in selected:
                rows.append({"Gene": str(gene), "Peak": str(peak_names[idx]), "Distance": float(distance[idx])})
    gene_peak_net = pd.DataFrame(rows).drop_duplicates()
    if gene_peak_net.empty:
        raise ValueError("No gene-peak edges were constructed from the provided GTF and ATAC peaks.")
    rna.uns["gene_peak_Net"] = gene_peak_net
    atac.uns["gene_peak_Net"] = gene_peak_net.copy()
    return gene_peak_net


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    status_path = out_dir / "multigate_status.json"
    started = now_seconds()
    try:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
        from MultiGATE import Cal_Spatial_Net, train_MultiGATE

        labels = read_labels()
        rna = sc.read_h5ad(args.rna)
        atac = sc.read_h5ad(args.atac)
        rna = attach_common_obs(rna, labels)
        atac = attach_common_obs(atac, labels)
        rna = maybe_subset(rna, args.max_obs, args.seed)
        atac = atac[rna.obs_names].copy()

        rna = preprocess_rna(rna, args.rna_hvg)
        atac = preprocess_atac(atac, args.atac_hvg)

        Cal_Spatial_Net(rna, k_cutoff=args.knn, model="KNN", verbose=True)
        atac.uns["Spatial_Net"] = rna.uns["Spatial_Net"].copy()
        gene_peak_net = build_gene_peak_net(rna, atac, args.gtf, window=2000, states=("Top",), min_edges_per_gene=1)
        gene_peak_net.to_csv(out_dir / "gene_peak_net.csv", index=False)

        rna, atac = train_MultiGATE(
            rna,
            atac,
            hidden_dims=[512, args.hidden_dim],
            n_epochs=args.epochs,
            lr=0.0001,
            key_added="MultiGATE",
            random_seed=args.seed,
            save_loss=True,
        )
        embedding = np.asarray(rna.obsm["MultiGATE_clip_all"], dtype=np.float32)
        np.save(out_dir / "multigate_clip_all.npy", embedding)
        labels_df = rna.obs[["predicted.celltype", "ATAC_clusters"]].copy()
        labels_df.to_csv(out_dir / "labels_used.csv")
        evaluate_embedding(embedding, labels_df, out_dir / "multigate_metrics.csv", method=METHOD)
        if "MultiGATE_loss_united" in rna.uns:
            pd.DataFrame({"epoch": np.arange(len(rna.uns["MultiGATE_loss_united"])), "loss": rna.uns["MultiGATE_loss_united"]}).to_csv(
                out_dir / "loss.csv", index=False
            )
        rna.write(out_dir / "adata_rna_multigate.h5ad")
        atac.write(out_dir / "adata_atac_multigate.h5ad")
        write_status(
            status_path,
            METHOD,
            "success",
            "MultiGATE completed using package defaults with P22-derived spatial and gene-peak graphs.",
            {
                "duration_seconds": duration_seconds(started),
                "n_obs": int(rna.n_obs),
                "used_full_p22": args.max_obs is None,
                "defaults": "RNA HVG=3000; ATAC HVG=8000; KNN spatial graph=6; gene-peak Top within 2kb; hidden_dims=[512,30]; epochs=500; lr=1e-4",
                "gtf": str(args.gtf),
                "gene_peak_edges": int(gene_peak_net.shape[0]),
            },
        )
    except Exception as exc:
        write_failure(status_path, METHOD, exc, {"duration_seconds": duration_seconds(started)})
        raise


if __name__ == "__main__":
    main()
