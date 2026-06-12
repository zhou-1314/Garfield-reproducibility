#!/usr/bin/env python
"""Reproduce the Garfield-P22 notebook with the RELEASED Garfield (v1.0.0) to
produce the pseudo ground truth (niche_type_sub, 14 niches) for the recent-
baseline benchmark.

Strategy (per user decision): the released Garfield (1.0.0) clustering is the
pseudo-GT; the dev Garfield (1.0.1) embedding is the "Garfield" prediction row.
Using two different versions breaks the trivial self-referential ARI=1.0 that
arises when GT == the same run's leiden_0.5 partition.

Config is taken verbatim from
  Garfield_code/spatial-multi-modal-code/spatial_tissue_niches_mouse_brain.ipynb
Run on a single pinned GPU:
  CUDA_VISIBLE_DEVICES=1 python run_garfield_p22_pseudo_gt.py --device 0
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time
import traceback

# Limit CPU threads BEFORE importing numpy/torch (80-core shared box).
_GF_THREADS = os.environ.get("GF_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _GF_THREADS)

REPO = "/data2/zhouwg_data/project/Garfield-reproducibility"
RELEASED = os.path.join(REPO, "Garfield")  # v1.0.0
sys.path.insert(0, RELEASED)

import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import scanpy as sc

DATA = "/data2/zhouwg_data/project/Garfield_tutorials/data"
RNA_PATH = os.path.join(DATA, "spatial_atac_rna_seq_mouse_brain_rna.h5ad")
ATAC_PATH = os.path.join(DATA, "spatial_atac_rna_seq_mouse_brain_atac.h5ad")
ANN_PATH = os.path.join(DATA, "spatial_atac_rna_seq_mouse_brain_cell_type_annotations.csv")
OUT_DIR = os.path.join(REPO, "revision_experiments", "results",
                       "recent_baselines", "spatial_atac_rna_p22", "pseudo_gt")

# 1-to-1 relabel of the released-Garfield leiden_0.5 (14 clusters) -> niche names
# (verbatim from the notebook). niche_type_sub keeps all 14; niche_type is coarser.
CLUSTER2SUB = {
    '0': '0-cp', '1': '1-ctx', '2': '2-ccg/aco', '3': '3-aca', '4': '4-ctx',
    '5': '5-aca', '6': '6-ctx', '7': '7-acb', '8': '8-cp', '9': '9-ls',
    '10': '10-ctx', '11': '11-ctx/aca', '12': '12-vl', '13': '13-islm',
}
CLUSTER2NICHE = {
    '0': 'cp', '1': 'ctx', '2': 'ccg/aco', '3': 'aca', '4': 'ctx', '5': 'aca',
    '6': 'ctx', '7': 'acb', '8': 'cp', '9': 'ls', '10': 'ctx', '11': 'ctx/aca',
    '12': 'vl', '13': 'islm',
}


def log(msg, t0):
    print("[{:7.1f}s] {}".format(time.time() - t0, msg), flush=True)


def search_res_for_k(adata, latent_key, k_target=14, start=0.3, end=0.9, step=0.02):
    """Find the leiden resolution (near 0.5) that yields exactly k_target clusters."""
    best_res, best_gap, best_n = 0.5, 1e9, None
    r = start
    while r <= end + 1e-9:
        sc.tl.leiden(adata, resolution=round(r, 3), key_added="_tmp",
                     random_state=0, neighbors_key=latent_key)
        n = adata.obs["_tmp"].nunique()
        gap = abs(n - k_target)
        if gap < best_gap:
            best_gap, best_res, best_n = gap, round(r, 3), n
        if n == k_target:
            return round(r, 3), n
        r += step
    return best_res, best_n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    status_path = os.path.join(OUT_DIR, "pseudo_gt_status.json")
    t0 = time.time()
    try:
        import Garfield as gf
        from Garfield.model import Garfield
        from mudata import MuData
        log("released Garfield %s" % gf.__version__, t0)

        # ---- load data (mirror notebook) ----
        adata_rna = sc.read_h5ad(RNA_PATH)
        adata_rna.var_names_make_unique()
        if "ATAC_clusters" in adata_rna.obs:
            adata_rna.obs = adata_rna.obs.drop(["ATAC_clusters"], axis=1)
        meta = pd.read_csv(ANN_PATH, index_col=0)
        adata_rna.obs = adata_rna.obs.join(meta)

        adata_atac = sc.read_h5ad(ATAC_PATH)
        adata_atac.var_names_make_unique()
        split = adata_atac.var_names.str.split(":|-", expand=True).to_frame(index=False)
        split.index = adata_atac.var_names
        split.columns = ["chr", "start", "end"]
        adata_atac.var[["chr", "start", "end"]] = split
        mdata = MuData({"rna": adata_rna, "atac": adata_atac})
        log("data loaded: %s" % str(mdata.shape), t0)

        workdir = os.path.join(OUT_DIR, "_gf_workdir")
        os.makedirs(workdir, exist_ok=True)
        gf.settings.set_workdir(workdir)

        user_config = dict(
            adata_list=mdata, profile="spatial", data_type="multi-modal",
            sub_data_type=["rna", "atac"], sample_col=None, weight=0.8,
            graph_const_method="Squidpy", genome="mm10", used_hvg=True,
            min_cells=3, min_features=0, keep_mt=False, target_sum=1e4,
            rna_n_top_features=3000, atac_n_top_features=10000, n_components=50,
            n_neighbors=5, metric="euclidean", svd_solver="arpack",
            used_pca_feat=True, adj_key="connectivities",
            edge_val_ratio=0.1, edge_test_ratio=0., node_val_ratio=0.1, node_test_ratio=0.,
            augment_type="svd", svd_q=5, use_FCencoder=False, conv_type="GATv2Conv",
            gnn_layer=2, hidden_dims=[128, 128], bottle_neck_neurons=20, cluster_num=20,
            drop_feature_rate=0.2, drop_edge_rate=0.2, num_heads=3, dropout=0.2,
            concat=True, used_edge_weight=False, used_DSBN=False, used_mmd=False,
            num_neighbors=5, loaders_n_hops=2, edge_batch_size=4096, node_batch_size=256,
            include_edge_recon_loss=True, include_gene_expr_recon_loss=True,
            lambda_latent_contrastive_instanceloss=1.0,
            lambda_latent_contrastive_clusterloss=0.5,
            lambda_gene_expr_recon=1.0, lambda_edge_recon=50.,
            lambda_latent_adj_recon_loss=200., lambda_omics_recon_mmd_loss=0.2,
            n_epochs_no_edge_recon=0, learning_rate=0.001, weight_decay=1e-05,
            gradient_clipping=5, latent_key="garfield_latent",
            reload_best_model=True, use_early_stopping=True, early_stopping_kwargs=None,
            monitor=True, device_id=args.device, seed=args.seed, verbose=True,
            n_epochs=args.epochs,
        )
        dict_config = gf.settings.set_gf_params(user_config)
        log("building released Garfield model (preprocessing)...", t0)
        model = Garfield(dict_config)
        log("preprocessing done; training...", t0)
        model.train()
        log("training done", t0)

        # ---- latent -> neighbors -> leiden 0.5 -> 14 niches ----
        latent_key = "garfield_latent"
        sc.pp.neighbors(model.adata, use_rep=latent_key, key_added=latent_key)
        sc.tl.leiden(model.adata, resolution=0.5, key_added="latent_leiden_0.5",
                     neighbors_key=latent_key, random_state=0)
        n05 = model.adata.obs["latent_leiden_0.5"].nunique()
        log("leiden 0.5 -> %d clusters" % n05, t0)
        if n05 == 14:
            used_res, used_n = 0.5, 14
            part_key = "latent_leiden_0.5"
        else:
            used_res, used_n = search_res_for_k(model.adata, latent_key, 14)
            part_key = "latent_leiden_gt14"
            sc.tl.leiden(model.adata, resolution=used_res, key_added=part_key,
                         neighbors_key=latent_key, random_state=0)
            log("searched res=%s -> %d clusters (target 14)" % (used_res, used_n), t0)

        labels = model.adata.obs[part_key].astype(str)
        # Only apply the notebook's niche NAMES when leiden_0.5 reproduced the canonical
        # 14-cluster partition. If we had to search a different resolution to reach 14
        # clusters, the named map would mislabel; use generic names (only the PARTITION
        # matters for ARI/NMI scoring, not the niche names).
        canonical = (n05 == 14 and part_key == "latent_leiden_0.5"
                     and set(labels.unique()) <= set(CLUSTER2SUB.keys()))
        if canonical:
            model.adata.obs["niche_type_sub"] = labels.map(CLUSTER2SUB).astype("category")
            model.adata.obs["niche_type"] = labels.map(CLUSTER2NICHE).astype("category")
        else:
            model.adata.obs["niche_type_sub"] = ("niche_" + labels).astype("category")
            model.adata.obs["niche_type"] = ("niche_" + labels).astype("category")

        # ---- save pseudo-GT artifacts (robust: raw arrays first, then a CLEAN adata) ----
        latent = np.asarray(model.adata.obsm[latent_key], dtype=np.float32)
        spatial = np.asarray(model.adata.obsm["spatial"], dtype=np.float32)
        np.save(os.path.join(OUT_DIR, "garfield_latent_released.npy"), latent)
        np.save(os.path.join(OUT_DIR, "spatial.npy"), spatial)

        def _get_obs(name):
            for k in (name, "rna:" + name, "atac:" + name):
                if k in model.adata.obs:
                    return model.adata.obs[k].astype(str).values
            return None

        import anndata as ad
        obs_df = pd.DataFrame(index=model.adata.obs_names.astype(str))
        obs_df["niche_type_sub"] = model.adata.obs["niche_type_sub"].astype(str).values
        obs_df["niche_type"] = model.adata.obs["niche_type"].astype(str).values
        for c in ("predicted.celltype", "ATAC_clusters", "RNA_clusters"):
            v = _get_obs(c)
            if v is not None:
                obs_df[c] = v
        obs_df.to_csv(os.path.join(OUT_DIR, "pseudo_gt_labels.csv"))

        # clean minimal AnnData for the scorers (avoids the int-keyed obsm['feat']
        # DataFrame that anndata.write rejects)
        clean = ad.AnnData(
            X=np.zeros((model.adata.n_obs, 1), dtype=np.float32),
            obs=obs_df.astype("category"),
        )
        clean.obsm["garfield_latent"] = latent
        clean.obsm["spatial"] = spatial
        out_h5ad = os.path.join(OUT_DIR, "adata_pseudo_gt.h5ad")
        clean.write(out_h5ad)

        with open(status_path, "w") as fh:
            json.dump({
                "status": "success", "released_gf_version": gf.__version__,
                "n_obs": int(model.adata.n_obs),
                "niche_type_sub_k": int(model.adata.obs["niche_type_sub"].nunique()),
                "niche_type_k": int(model.adata.obs["niche_type"].nunique()),
                "leiden_resolution_used": float(used_res),
                "n_clusters_used": int(used_n),
                "duration_seconds": round(time.time() - t0, 1),
                "out_h5ad": out_h5ad,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
            }, fh, indent=2)
        log("DONE -> %s (niche_type_sub k=%d)" % (
            out_h5ad, model.adata.obs["niche_type_sub"].nunique()), t0)
    except Exception as exc:
        with open(status_path, "w") as fh:
            json.dump({"status": "failed", "error": str(exc),
                       "traceback": traceback.format_exc(),
                       "duration_seconds": round(time.time() - t0, 1)}, fh, indent=2)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
