#!/usr/bin/env python
"""Hippocampus module-ablation on the VALIDATED foundation:
garfield_dev + exact benchmark config (notebook train_garfield_models) +
TRUE niche GT obs['niches']. One run per call; writes one CSV row.

  python ablate_hippo.py --tag no_gene_recon --overrides '{"lambda_gene_expr_recon":0.0}' \
      --flags '{}' --epochs 50 --device 0 --out results/hippo_abl/no_gene_recon.csv
"""
import os, sys, json, time, argparse, csv, warnings
warnings.simplefilter("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ["GARFIELD_USE_OPTIMIZED_GRAPH"] = "1"
_thr = os.environ.get("GF_THREADS", "8")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, _thr)
REPO = "/data2/zhouwg_data/project/Garfield-reproducibility"
sys.path.insert(0, os.path.join(REPO, "Garfield-garfield_dev"))  # garfield_dev version

import numpy as np, scanpy as sc, anndata as ad, sklearn.metrics as skm
import Garfield as gf
from Garfield.model import Garfield


def faithful_config(n_neighbors=8):
    return dict(
        profile="spatial", data_type="single-modal", sample_col=None, weight=0.5,
        graph_const_method="Squidpy", used_hvg=True, min_cells=3, min_features=0,
        keep_mt=False, target_sum=1e4, rna_n_top_features=3000, n_components=50,
        n_neighbors=n_neighbors, metric="euclidean", svd_solver="arpack",
        used_pca_feat=False, adj_key="connectivities",
        edge_val_ratio=0.1, edge_test_ratio=0.0, node_val_ratio=0.1, node_test_ratio=0.0,
        augment_type="dropout", svd_q=5, use_FCencoder=True, conv_type="GAT", gnn_layer=2,
        hidden_dims=[128, 128], bottle_neck_neurons=20, cluster_num=20,
        drop_feature_rate=0.2, drop_edge_rate=0.2, num_heads=3, dropout=0.2, concat=True,
        used_edge_weight=False, used_DSBN=False, used_mmd=False,
        num_neighbors=5, loaders_n_hops=2, edge_batch_size=4096, node_batch_size=256,
        include_edge_recon_loss=True, include_gene_expr_recon_loss=True,
        lambda_latent_contrastive_instanceloss=1.0, lambda_latent_contrastive_clusterloss=0.5,
        lambda_gene_expr_recon=1.0, lambda_edge_recon=1.0, lambda_latent_adj_recon_loss=2.0,
        lambda_omics_recon_mmd_loss=0.2,
        n_epochs_no_edge_recon=0, learning_rate=0.001, weight_decay=1e-05, gradient_clipping=5,
        latent_key="garfield_latent", reload_best_model=True, use_early_stopping=True,
        early_stopping_kwargs=None, monitor=True, verbose=False,
        use_lightning=False,  # legacy single-GPU trainer (avoid DDP spawning across all GPUs)
    )


def evaluate(Z, spatial, niches, res=0.5):
    """UNSUPERVISED metrics (no GT) for spatial niche quality + niches ARI as reference.
    Leiden at a FIXED resolution so all ablation variants are compared at the same
    clustering granularity (fair). Metrics:
      - ASW_cluster: silhouette of the latent vs predicted clusters (latent separation)
      - spatial_coherence: fraction of spatial neighbors sharing the predicted cluster
      - (reference) niches ARI/NMI at best resolution
    """
    from sklearn.neighbors import NearestNeighbors
    Z = np.ascontiguousarray(np.asarray(Z, dtype=np.float64))
    b = ad.AnnData(X=Z)
    sc.pp.neighbors(b, use_rep="X", key_added="nn", random_state=0)
    sc.tl.leiden(b, resolution=res, key_added="clu", random_state=0,
                 neighbors_key="nn", flavor="igraph", n_iterations=2, directed=False)
    clusters = b.obs["clu"].values.astype(str)
    nclu = len(np.unique(clusters))
    out = {"leiden_res": res, "n_clusters": int(nclu)}
    # --- UNSUPERVISED ---
    out["ASW_cluster"] = float(skm.silhouette_score(Z, clusters)) if nclu > 1 else float("nan")
    sp = np.asarray(spatial)
    nbrs = NearestNeighbors(n_neighbors=11).fit(sp)
    _, idx = nbrs.kneighbors(sp)
    neigh_same = (clusters[idx[:, 1:]] == clusters[:, None]).mean(axis=1)
    out["spatial_coherence"] = float(neigh_same.mean())
    # Calinski-Harabasz / Davies-Bouldin (unsupervised latent cluster quality)
    if nclu > 1:
        out["calinski_harabasz"] = float(skm.calinski_harabasz_score(Z, clusters))
        out["davies_bouldin"] = float(skm.davies_bouldin_score(Z, clusters))
    # --- reference (vs niches GT, best-res) ---
    y = np.asarray(niches).astype(str)
    msk = ~np.isin(y, ["nan", "NaN", "None", ""])
    best = 0.0
    for r in np.arange(0.1, 2.01, 0.2):
        sc.tl.leiden(b, resolution=float(r), key_added="t", random_state=0,
                     neighbors_key="nn", flavor="igraph", n_iterations=2, directed=False)
        best = max(best, skm.adjusted_rand_score(y[msk], b.obs["t"].values[msk]))
    out["niches_ARI_bestres"] = float(best)
    return out, clusters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--overrides", default="{}")
    p.add_argument("--flags", default="{}")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--n_neighbors", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t0 = time.time()

    adata = ad.read_h5ad(os.path.join(REPO, "benchmark_unimodal/adata_with_niches.h5ad"))
    adata.X = adata.layers["counts"].copy()
    niches = adata.obs["niches"].astype(str).values

    gf.settings.set_workdir(f"/tmp/abl_hippo_{a.tag}_s{a.seed}")
    cfg = faithful_config(a.n_neighbors)
    cfg.update(json.loads(a.overrides))
    cfg["adata_list"] = adata
    cfg["n_epochs"] = a.epochs
    cfg["seed"] = a.seed
    cfg["device_id"] = a.device
    dict_config = gf.settings.set_gf_params(cfg)

    model = Garfield(dict_config)
    for k, v in json.loads(a.flags).items():
        setattr(model.model, k, v)
    model.train()

    Z = model.adata.obsm["garfield_latent"]
    spatial = model.adata.obsm["spatial"]
    gt = adata.obs["niches"].reindex(model.adata.obs_names).astype(str).values
    metrics, clusters = evaluate(Z, spatial, gt, res=0.5)
    # save latent + spatial + niches so metrics can be recomputed without retraining
    np.savez(a.out.replace(".csv", "_latent.npz"),
             Z=np.asarray(Z), spatial=np.asarray(spatial),
             niches=gt.astype(str), clusters=clusters.astype(str))
    row = dict(dataset="hippo", tag=a.tag, seed=a.seed, epochs=a.epochs,
               overrides=a.overrides, flags=a.flags, n_obs=int(model.adata.n_obs),
               train_sec=round(time.time() - t0, 1), **metrics)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    wh = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if wh: w.writeheader()
        w.writerow(row)
    print(f"[DONE] hippo/{a.tag} ASW_clu={metrics['ASW_cluster']:.3f} "
          f"spatial_coh={metrics['spatial_coherence']:.3f} "
          f"niches_ARI={metrics['niches_ARI_bestres']:.3f} nclu={metrics['n_clusters']} "
          f"({row['train_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
