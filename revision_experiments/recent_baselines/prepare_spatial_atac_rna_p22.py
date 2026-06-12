#!/usr/bin/env python
"""Prepare metadata and audit files for the P22 spatial ATAC-RNA benchmark."""

from __future__ import print_function

import argparse
from pathlib import Path

import pandas as pd
import scanpy as sc

from common import (
    ANNOTATION_PATH,
    ATAC_PATH,
    DEFAULT_RESULTS_DIR,
    RNA_PATH,
    attach_common_obs,
    ensure_dir,
    read_labels,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--rna", default=str(RNA_PATH))
    parser.add_argument("--atac", default=str(ATAC_PATH))
    parser.add_argument("--annotations", default=str(ANNOTATION_PATH))
    parser.add_argument("--write-labeled-h5ad", action="store_true")
    return parser.parse_args()


def summarize_adata(name, path, labels):
    adata = sc.read_h5ad(path)
    adata = attach_common_obs(adata, labels)
    rows = []
    for label_key in ["predicted.celltype", "ATAC_clusters", "RNA_clusters", "batch"]:
        rows.append(
            {
                "modality": name,
                "n_obs": int(adata.n_obs),
                "n_vars": int(adata.n_vars),
                "label_key": label_key,
                "present": label_key in adata.obs,
                "n_missing": int(adata.obs[label_key].isna().sum()) if label_key in adata.obs else None,
                "n_unique": int(adata.obs[label_key].nunique(dropna=True)) if label_key in adata.obs else None,
                "has_spatial": "spatial" in adata.obsm,
                "has_counts_layer": "counts" in adata.layers,
            }
        )
    return adata, rows


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    labels = read_labels(args.annotations)
    labels.to_csv(out_dir / "labels.csv")

    rna, rna_rows = summarize_adata("RNA", args.rna, labels)
    atac, atac_rows = summarize_adata("ATAC", args.atac, labels)
    summary = pd.DataFrame(rna_rows + atac_rows)
    summary.to_csv(out_dir / "data_summary.csv", index=False)

    protocol = pd.DataFrame(
        [
            {
                "dataset": "P22 spatial ATAC-RNA mouse brain",
                "rna_path": str(Path(args.rna)),
                "atac_path": str(Path(args.atac)),
                "annotation_path": str(Path(args.annotations)),
                "primary_label": "predicted.celltype",
                "secondary_label": "ATAC_clusters",
                "evaluation": "Leiden resolution sweep; select resolution with cluster count closest to label count, then report ARI/NMI/AMI/HOM/ASW",
            }
        ]
    )
    protocol.to_csv(out_dir / "protocol_summary.csv", index=False)

    if args.write_labeled_h5ad:
        rna.write(out_dir / "rna_labeled.h5ad")
        atac.write(out_dir / "atac_labeled.h5ad")

    print("Wrote", out_dir / "data_summary.csv")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
