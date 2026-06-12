#!/usr/bin/env python
"""Run SpaMosaic on the P22 spatial ATAC-RNA benchmark using demo defaults.

For a single paired RNA+ATAC slice both modalities are the two omics of the
same spots, so they live in a single batch (batch 0) of the modBatch_dict the
way the SpaMosaic demo expects:

    input_dict = {'rna': [rna_ad], 'atac': [atac_ad]}

Preprocessing (RNA_preprocess / Epigenome_preprocess) -> SpaMosaic graph model
-> infer_emb produces one joint embedding vector per spot, aligned to obs order.
"""

from __future__ import print_function

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc

# Deterministic CuBLAS for SpaMosaic (matches demo); harmless on CPU.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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


SPAMOSAIC_SRC = Path("/data2/zhouwg_data/project/Garfield_benchmark/SpaMosaic")
if str(SPAMOSAIC_SRC) not in sys.path:
    sys.path.insert(0, str(SPAMOSAIC_SRC))


METHOD = "SpaMosaic"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR / "spamosaic"))
    parser.add_argument("--rna", default=str(RNA_PATH))
    parser.add_argument("--atac", default=str(ATAC_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--max-obs", type=int, default=None)
    parser.add_argument("--n-hvg", type=int, default=5000)
    parser.add_argument("--n-peak", type=int, default=100000)
    parser.add_argument("--n-comps", type=int, default=50)
    parser.add_argument("--n-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--intra-knns", type=int, default=10)
    parser.add_argument("--inter-knn-base", type=int, default=10)
    return parser.parse_args()


def maybe_subset(adata, max_obs, seed):
    if max_obs is None or adata.n_obs <= max_obs:
        return adata
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_obs, replace=False))
    return adata[idx].copy()


def prep_modality(adata):
    """Make X hold raw counts and add the batch/src column the demo expects."""
    adata = adata.copy()
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    adata.obs["src"] = "p22"
    adata.obs["batch"] = "p22"
    adata.var_names_make_unique()
    return adata


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    status_path = out_dir / "spamosaic_status.json"
    started = now_seconds()
    try:
        from spamosaic.framework import SpaMosaic
        from spamosaic.preprocessing import Epigenome_preprocess, RNA_preprocess

        labels = read_labels()
        rna = sc.read_h5ad(args.rna)
        atac = sc.read_h5ad(args.atac)
        rna = attach_common_obs(rna, labels)
        atac = attach_common_obs(atac, labels)
        rna = maybe_subset(rna, args.max_obs, args.seed)
        atac = atac[rna.obs_names].copy()

        obs_order = rna.obs_names.astype(str).to_numpy()

        rna = prep_modality(rna)
        atac = prep_modality(atac)

        # Single paired slice: RNA and ATAC are the two modalities of one batch.
        input_key = "dimred_bc"
        input_dict = {"rna": [rna], "atac": [atac]}

        # batch_corr=False: only one batch, nothing to correct across.
        RNA_preprocess(
            input_dict["rna"],
            batch_corr=False,
            favor="scanpy",
            n_hvg=args.n_hvg,
            batch_key="src",
            key=input_key,
        )
        Epigenome_preprocess(
            input_dict["atac"],
            batch_corr=False,
            n_peak=args.n_peak,
            n_comps=args.n_comps,
            batch_key="src",
            key=input_key,
        )

        model = SpaMosaic(
            modBatch_dict=input_dict,
            input_key=input_key,
            batch_key="src",
            intra_knns=args.intra_knns,
            inter_knn_base=args.inter_knn_base,
            w_g=0.8,
            seed=args.seed,
            device=args.device,
        )
        model.train(net="wlgcn", lr=args.lr, T=0.01, n_epochs=args.n_epochs)

        ad_embs = model.infer_emb(input_dict, emb_key="emb", final_latent_key="merged_emb")
        ad_mosaic = sc.concat(ad_embs)
        embedding = np.asarray(ad_mosaic.obsm["merged_emb"], dtype=np.float32)

        # Align rows back to the P22 obs order.
        emb_index = ad_mosaic.obs_names.astype(str).to_numpy()
        if not np.array_equal(emb_index, obs_order):
            order = pd.Index(emb_index).get_indexer(obs_order)
            if (order < 0).any():
                raise RuntimeError("Embedding obs_names do not cover the P22 obs order.")
            embedding = embedding[order]

        if embedding.shape[0] != len(obs_order):
            raise RuntimeError(
                "Embedding has {} rows but {} spots expected.".format(embedding.shape[0], len(obs_order))
            )

        np.save(out_dir / "spamosaic_emb.npy", embedding)
        pd.DataFrame({"obs_name": obs_order}).to_csv(out_dir / "obs_names.csv", index=False)

        labels_df = pd.DataFrame(index=pd.Index(obs_order, name="obs_name"))
        for col in ("predicted.celltype", "ATAC_clusters"):
            if col in labels.columns:
                labels_df[col] = labels.reindex(obs_order)[col].values
        labels_df.to_csv(out_dir / "labels_used.csv")

        # Quick sanity score only; the real niche scoring happens in a separate
        # scorer, so a failure here must not lose the saved embedding.
        sanity_metrics = "ok"
        try:
            evaluate_embedding(embedding, labels_df, out_dir / "spamosaic_metrics.csv", method=METHOD)
        except Exception as eval_exc:  # noqa: BLE001
            sanity_metrics = "skipped: {}: {}".format(type(eval_exc).__name__, eval_exc)
            print("WARNING: evaluate_embedding sanity score skipped:", sanity_metrics)

        write_status(
            status_path,
            METHOD,
            "success",
            "SpaMosaic completed on P22 paired RNA+ATAC slice using demo defaults.",
            {
                "duration_seconds": duration_seconds(started),
                "n_obs": int(embedding.shape[0]),
                "embedding_dim": int(embedding.shape[1]),
                "used_full_p22": args.max_obs is None,
                "device": str(args.device),
                "sanity_metrics": sanity_metrics,
                "source": str(SPAMOSAIC_SRC),
                "defaults": (
                    "RNA scanpy HVG={}; ATAC peaks={}/LSI={}; intra_knns={}; "
                    "inter_knn_base={}; w_g=0.8; net=wlgcn; lr={}; T=0.01; epochs={}".format(
                        args.n_hvg,
                        args.n_peak,
                        args.n_comps,
                        args.intra_knns,
                        args.inter_knn_base,
                        args.lr,
                        args.n_epochs,
                    )
                ),
            },
        )
    except Exception as exc:
        write_failure(status_path, METHOD, exc, {"duration_seconds": duration_seconds(started)})
        raise


if __name__ == "__main__":
    main()
