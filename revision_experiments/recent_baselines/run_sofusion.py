#!/usr/bin/env python
"""Run soFusion on the P22 spatial ATAC-RNA benchmark using tutorial defaults."""

from __future__ import print_function

import argparse
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

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


METHOD = "soFusion"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR / "sofusion"))
    parser.add_argument("--rna", default=str(RNA_PATH))
    parser.add_argument("--atac", default=str(ATAC_PATH))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-obs", type=int, default=None)
    return parser.parse_args()


def maybe_subset(adata, max_obs, seed):
    if max_obs is None or adata.n_obs <= max_obs:
        return adata
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_obs, replace=False))
    return adata[idx].copy()


def add_spatial_obs(adata):
    coords = np.asarray(adata.obsm["spatial"])
    adata.obs["array_col"] = coords[:, 0]
    adata.obs["array_row"] = coords[:, 1]
    adata.obs["x"] = coords[:, 0]
    adata.obs["y"] = coords[:, 1]
    return adata


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    status_path = out_dir / "sofusion_status.json"
    started = now_seconds()
    try:
        if "munkres" not in sys.modules:
            stub = types.ModuleType("munkres")

            class _Munkres(object):
                def compute(self, matrix):
                    raise RuntimeError("munkres is not installed; soFusion training does not require this helper.")

            stub.Munkres = _Munkres
            sys.modules["munkres"] = stub
        if "ot" not in sys.modules:
            ot_stub = types.ModuleType("ot")

            def _missing_ot(*_args, **_kwargs):
                raise RuntimeError("POT/ot is not installed; this soFusion runner does not use OT utilities.")

            ot_stub.dist = _missing_ot
            sys.modules["ot"] = ot_stub

        from soFusion import train_model
        from soFusion.process import prefilter_genes
        from soFusion.utils import calculate_adj_matrix

        labels = read_labels()
        rna = sc.read_h5ad(args.rna)
        atac = sc.read_h5ad(args.atac)
        rna = attach_common_obs(rna, labels)
        atac = attach_common_obs(atac, labels)
        rna = maybe_subset(rna, args.max_obs, args.seed)
        atac = atac[rna.obs_names].copy()
        rna = add_spatial_obs(rna)
        atac = add_spatial_obs(atac)

        atac.X = atac.X.astype(float)
        prefilter_genes(atac, min_cells=3)
        sc.pp.highly_variable_genes(atac, flavor="seurat_v3", n_top_genes=8000)
        sc.pp.normalize_per_cell(atac)
        sc.pp.log1p(atac)

        prefilter_genes(rna, min_cells=3)
        sc.pp.normalize_per_cell(rna)
        sc.pp.log1p(rna)
        sc.pp.highly_variable_genes(rna, flavor="seurat_v3", n_top_genes=3000)

        adj = calculate_adj_matrix(atac, "array_col", "array_row")
        atac, rna = train_model.train(
            [atac, rna],
            adj,
            ["ATAC", "RNA"],
            k=14,
            n_epochs=20,
            h=[3000, 3000],
            l=1,
            a=1,
            b=1,
            c=1,
            d=1,
            weight=[1, 5, 5],
            device=args.device,
            random_seed=args.seed,
        )

        embedding = np.asarray(atac.obsm["emb_pca"], dtype=np.float32)
        np.save(out_dir / "emb_pca.npy", embedding)
        labels_df = atac.obs[["predicted.celltype", "ATAC_clusters"]].copy()
        labels_df.to_csv(out_dir / "labels_used.csv")
        evaluate_embedding(embedding, labels_df, out_dir / "sofusion_metrics.csv", method=METHOD)
        atac.write(out_dir / "adata_atac_sofusion.h5ad")
        rna.write(out_dir / "adata_rna_sofusion.h5ad")
        write_status(
            status_path,
            METHOD,
            "success",
            "soFusion completed using README spatial ATAC-RNA tutorial defaults.",
            {
                "duration_seconds": duration_seconds(started),
                "n_obs": int(atac.n_obs),
                "used_full_p22": args.max_obs is None,
                "device": str(args.device),
                "defaults": "ATAC HVG=8000; RNA HVG=3000; k=14; epochs=20; h=[3000,3000]; l=1; a=b=c=d=1; weight=[1,5,5]",
            },
        )
    except Exception as exc:
        write_failure(status_path, METHOD, exc, {"duration_seconds": duration_seconds(started)})
        raise


if __name__ == "__main__":
    main()
