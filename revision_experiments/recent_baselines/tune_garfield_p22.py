#!/usr/bin/env python
"""Hyperparameter tuning for DEV Garfield (1.0.1) on P22, to maximise external-GT
niche recovery. A FULL preprocessed-adata pickle cache is shared across all configs,
so only model/loss params are swept (they don't change preprocessing) and each run is
just training + latent dump. Score the saved latents externally with score_scib.py.

  CUDA_VISIBLE_DEVICES=1 python tune_garfield_p22.py --tag cfgA \
      --overrides '{"lambda_edge_recon":5,"lambda_latent_adj_recon_loss":1,"n_epochs":200}'
"""
from __future__ import print_function
import argparse, json, os, sys, time, traceback, pickle, importlib

_GF_THREADS = os.environ.get("GF_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _GF_THREADS)

REPO = "/data2/zhouwg_data/project/Garfield-reproducibility"
sys.path.insert(0, os.path.join(REPO, "Garfield-garfield_dev"))
import warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, scanpy as sc

DATA = "/data2/zhouwg_data/project/Garfield_tutorials/data"
RNA_PATH = os.path.join(DATA, "spatial_atac_rna_seq_mouse_brain_rna.h5ad")
ATAC_PATH = os.path.join(DATA, "spatial_atac_rna_seq_mouse_brain_atac.h5ad")
ANN_PATH = os.path.join(DATA, "spatial_atac_rna_seq_mouse_brain_cell_type_annotations.csv")
OUT_DIR = os.path.join(REPO, "revision_experiments", "results", "recent_baselines",
                       "spatial_atac_rna_p22", "garfield_tune")
CACHE_DIR = os.path.join(OUT_DIR, "_shared_cache")  # full-preprocessing pickle + ATAC gene-activity


def log(m, t0): print("[{:7.1f}s] {}".format(time.time() - t0, m), flush=True)


def base_config(mdata, device):
    # notebook P22 config (paper's tuned spatial-multimodal settings)
    return dict(
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
        monitor=True, device_id=device, seed=2024, verbose=True, n_epochs=100,
        use_lightning=False, user_cache_path=CACHE_DIR,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--overrides", default="{}")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--cache-key", default="shared",
                    help="separate full-preprocessing pickle per key; use a new key when "
                         "an override changes PREPROCESSING (weight/svd_q/n_neighbors/used_pca_feat/HVG)")
    args = ap.parse_args()
    overrides = json.loads(args.overrides)
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    status = os.path.join(OUT_DIR, f"tune_{args.tag}_status.json")
    try:
        import Garfield as gf
        from Garfield.model import Garfield
        from mudata import MuData
        log(f"dev Garfield {gf.__version__} tag={args.tag} overrides={overrides}", t0)

        adata_rna = sc.read_h5ad(RNA_PATH); adata_rna.var_names_make_unique()
        if "ATAC_clusters" in adata_rna.obs:
            adata_rna.obs = adata_rna.obs.drop(["ATAC_clusters"], axis=1)
        adata_rna.obs = adata_rna.obs.join(pd.read_csv(ANN_PATH, index_col=0))
        adata_atac = sc.read_h5ad(ATAC_PATH); adata_atac.var_names_make_unique()
        sp = adata_atac.var_names.str.split(":|-", expand=True).to_frame(index=False)
        sp.index = adata_atac.var_names; sp.columns = ["chr", "start", "end"]
        adata_atac.var[["chr", "start", "end"]] = sp
        mdata = MuData({"rna": adata_rna, "atac": adata_atac})

        gf.settings.set_workdir(os.path.join(OUT_DIR, "_gf_workdir"))
        cfg = base_config(mdata, args.device)
        cfg.update(overrides)

        # ---- shared FULL-preprocessing pickle cache (model/loss sweeps reuse it) ----
        _gmod = importlib.import_module("Garfield.model.Garfield")
        pp_cache = os.path.join(CACHE_DIR, f"preprocessed_adata_{args.cache_key}.pkl"
                                if args.cache_key != "shared" else "preprocessed_adata.pkl")
        if not hasattr(_gmod, "_orig_DataProcess"):
            _gmod._orig_DataProcess = _gmod.DataProcess

        def _cached_DataProcess(*a, **k):
            if os.path.exists(pp_cache):
                log("loading cached preprocessed adata", t0)
                with open(pp_cache, "rb") as fh:
                    return pickle.load(fh)
            ad_ = _gmod._orig_DataProcess(*a, **k)
            tmp = pp_cache + f".tmp.{os.getpid()}"
            with open(tmp, "wb") as fh:
                pickle.dump(ad_, fh, protocol=4)
            os.replace(tmp, pp_cache)
            log("cached preprocessed adata", t0)
            return ad_
        _gmod.DataProcess = _cached_DataProcess

        log("building model (preprocess or cache-load)...", t0)
        model = Garfield(gf.settings.set_gf_params(cfg))
        log("training...", t0)
        model.train()
        log("training done", t0)

        latent = np.asarray(model.adata.obsm["garfield_latent"], dtype=np.float32)
        np.save(os.path.join(OUT_DIR, f"latent_{args.tag}.npy"), latent)
        pd.DataFrame({"obs_name": model.adata.obs_names.astype(str)}).to_csv(
            os.path.join(OUT_DIR, f"obs_names_{args.tag}.csv"), index=False)
        with open(status, "w") as fh:
            json.dump({"status": "success", "tag": args.tag, "overrides": overrides,
                       "latent_dim": int(latent.shape[1]), "n_obs": int(latent.shape[0]),
                       "duration_seconds": round(time.time() - t0, 1)}, fh, indent=2)
        log(f"DONE latent_{args.tag}.npy {latent.shape}", t0)
    except Exception as exc:
        with open(status, "w") as fh:
            json.dump({"status": "failed", "tag": args.tag, "error": str(exc),
                       "traceback": traceback.format_exc()}, fh, indent=2)
        traceback.print_exc(); sys.exit(1)


if __name__ == "__main__":
    main()
