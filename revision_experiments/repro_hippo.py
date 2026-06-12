#!/usr/bin/env python
"""Faithful hippocampus reproduction using garfield_dev + the exact benchmark
config (notebook spatial_tissue_niches_..._multi_GPUs.ipynb, train_garfield_models).
Validates that we reproduce the paper-level niche ARI (~0.59) before ablating.
"""
import os, sys, time
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ["GARFIELD_USE_OPTIMIZED_GRAPH"] = "1"
_thr = os.environ.get("GF_THREADS", "16")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, _thr)
import warnings; warnings.simplefilter("ignore")
REPO = "/data2/zhouwg_data/project/Garfield-reproducibility"
sys.path.insert(0, os.path.join(REPO, "Garfield-garfield_dev"))

import numpy as np, scanpy as sc, anndata as ad, sklearn.metrics as skm
import Garfield as gf
from Garfield.model import Garfield

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# ---- data: raw counts + niche_type GT (paper Fig.3 benchmark label) ----
log(f"garfield_dev version {gf.__version__}; loading data")
adata = ad.read_h5ad(os.path.join(REPO, "benchmark_unimodal/adata_with_niches.h5ad"))
adata.X = adata.layers["counts"].copy()
# GT = obs['niches'] (the TRUE niche annotation in adata_with_niches.h5ad)
GT_KEY = "niches"
log(f"adata {adata.shape}; {GT_KEY} classes={adata.obs[GT_KEY].nunique()}")

gf.settings.set_workdir("/tmp/repro_hippo_dev")
n_neighbors = int(sys.argv[1]) if len(sys.argv) > 1 else 8
user_config = dict(
    adata_list=adata, profile="spatial", data_type="single-modal", sample_col=None,
    weight=0.5, graph_const_method="Squidpy", used_hvg=True, min_cells=3, min_features=0,
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
    early_stopping_kwargs=None, monitor=True, device_id=int(os.environ.get("GF_DEVICE", "0")),
    seed=2024, verbose=True,
)
dict_config = gf.settings.set_gf_params(user_config)
log("building model + preprocessing")
model = Garfield(dict_config)
log("training")
model.train()
log("training done")

Z = model.adata.obsm["garfield_latent"]
np.save("/tmp/repro_hippo_latent.npy", np.asarray(Z))
model.adata.obs[[GT_KEY]].to_csv("/tmp/repro_hippo_gt.csv")
y = model.adata.obs[GT_KEY].astype(str).values
m = ~np.isin(y, ["nan", "NaN", "None", ""]); nt = len(np.unique(y[m]))
b = ad.AnnData(X=np.ascontiguousarray(np.asarray(Z, dtype=np.float64)))
sc.pp.neighbors(b, use_rep="X", key_added="nn", random_state=0)
best = 0; matched = None
for r in np.arange(0.1, 2.01, 0.1):
    sc.tl.leiden(b, resolution=float(r), key_added="t", random_state=0, neighbors_key="nn",
                 flavor="igraph", n_iterations=2, directed=False)
    a = skm.adjusted_rand_score(y[m], b.obs["t"].values[m])
    if a > best: best = a
    if matched is None and b.obs["t"].nunique() >= nt: matched = a
log(f"RESULT n_neighbors={n_neighbors}: best-res ARI={best:.3f} | matched-count ARI={matched}")
print(f"REPRO_HIPPO_DONE best_ari={best:.4f} matched_ari={matched}", flush=True)
