#!/usr/bin/env python
"""Run the DEV Garfield (v1.0.1, garfield_dev) on P22 to produce the "Garfield"
prediction embedding for the recent-baseline benchmark. Same notebook config as
the released pseudo-GT run, but a different package version -> its clustering is
scored against the released pseudo-GT niche_type_sub (no trivial ARI=1.0).

  CUDA_VISIBLE_DEVICES=2 python run_garfield_p22_pred.py --device 0
Saves garfield_latent + obs_names so it can be scored later by score_vs_pseudo_gt.py.
"""
from __future__ import print_function

import argparse
import json
import os
import sys
import time
import traceback

_GF_THREADS = os.environ.get("GF_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _GF_THREADS)

REPO = "/data2/zhouwg_data/project/Garfield-reproducibility"
DEVPKG = os.path.join(REPO, "Garfield-garfield_dev")  # v1.0.1
sys.path.insert(0, DEVPKG)

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
                       "recent_baselines", "spatial_atac_rna_p22", "garfield_dev_pred")


def log(msg, t0):
    print("[{:7.1f}s] {}".format(time.time() - t0, msg), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    status_path = os.path.join(OUT_DIR, "pred_status.json")
    t0 = time.time()
    try:
        import Garfield as gf
        from Garfield.model import Garfield
        from mudata import MuData
        log("dev Garfield %s" % gf.__version__, t0)

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
            # isolate the ATAC gene-activity cache so it never collides with the
            # released pseudo-GT run (which writes a cwd-relative adata_ATAC_cache.h5ad)
            user_cache_path=os.path.join(OUT_DIR, "_cache"),
        )
        os.makedirs(os.path.join(OUT_DIR, "_cache"), exist_ok=True)
        # dev package supports use_lightning; force single-GPU
        try:
            user_config["use_lightning"] = False
        except Exception:
            pass
        dict_config = gf.settings.set_gf_params(user_config)
        log("building dev Garfield model (preprocessing)...", t0)
        model = Garfield(dict_config)
        log("preprocessing done; training...", t0)
        model.train()
        log("training done", t0)

        latent = np.asarray(model.adata.obsm["garfield_latent"], dtype=np.float32)
        np.save(os.path.join(OUT_DIR, "garfield_latent_dev.npy"), latent)
        pd.DataFrame({"obs_name": model.adata.obs_names.astype(str)}).to_csv(
            os.path.join(OUT_DIR, "obs_names.csv"), index=False)
        np.save(os.path.join(OUT_DIR, "spatial.npy"),
                np.asarray(model.adata.obsm["spatial"], dtype=np.float32))
        with open(status_path, "w") as fh:
            json.dump({"status": "success", "dev_gf_version": gf.__version__,
                       "n_obs": int(model.adata.n_obs), "latent_dim": int(latent.shape[1]),
                       "duration_seconds": round(time.time() - t0, 1),
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z")}, fh, indent=2)
        log("DONE -> garfield_latent_dev.npy %s" % str(latent.shape), t0)
    except Exception as exc:
        with open(status_path, "w") as fh:
            json.dump({"status": "failed", "error": str(exc),
                       "traceback": traceback.format_exc(),
                       "duration_seconds": round(time.time() - t0, 1)}, fh, indent=2)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
