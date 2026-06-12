#!/usr/bin/env python
"""Evaluate a saved latent (.npy) against the TRUE niche GT = obs['niches']
(17 classes) in benchmark_unimodal/adata_with_niches.h5ad, aligned by row order."""
import os, sys, warnings
warnings.simplefilter("ignore")
REPO = "/data2/zhouwg_data/project/Garfield-reproducibility"
import numpy as np, scanpy as sc, anndata as ad, sklearn.metrics as skm
latent_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/repro_hippo_latent.npy"
Z = np.load(latent_path)
gt = ad.read_h5ad(os.path.join(REPO, "benchmark_unimodal/adata_with_niches.h5ad"))
y = gt.obs["niches"].astype(str).values
assert len(y) == Z.shape[0], f"latent {Z.shape} vs niches {len(y)}"
m = ~np.isin(y, ["nan", "NaN", "None", ""]); nt = len(np.unique(y[m]))
b = ad.AnnData(X=np.ascontiguousarray(Z.astype(np.float64)))
sc.pp.neighbors(b, use_rep="X", key_added="nn", random_state=0)
best = bn = 0; matched = None
for r in np.arange(0.1, 2.01, 0.1):
    sc.tl.leiden(b, resolution=float(r), key_added="t", random_state=0, neighbors_key="nn",
                 flavor="igraph", n_iterations=2, directed=False)
    a = skm.adjusted_rand_score(y[m], b.obs["t"].values[m])
    if a > best:
        best = a; bn = skm.normalized_mutual_info_score(y[m], b.obs["t"].values[m])
    if matched is None and b.obs["t"].nunique() >= nt:
        matched = a
print(f"vs niches (TRUE GT, {nt} classes): best-res ARI={best:.3f} NMI={bn:.3f} | matched-count ARI={matched}", flush=True)
