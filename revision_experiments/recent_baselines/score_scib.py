#!/usr/bin/env python
"""Score a method embedding against the P22 pseudo ground-truth (obs['niche_type_sub'])
using the EXACT scib protocol from metrics_benchmark.ipynb (count_metrics_for_modality /
count_metrics_for_best_resolution), so new-method numbers are directly comparable to the
already-published SpatialGlue/MOFA/NicheCompass/MultiVI 'best resolution' table:

    sc.pp.neighbors(adata, use_rep=emb)
    scib.metrics.cluster_optimal_resolution(adata, cluster_key='cluster', label_key=GT)
    ARI = scib.metrics.ari ; NMI = scib.metrics.nmi
    AMI = sklearn.adjusted_mutual_info_score ; HOM = sklearn.homogeneity_score
    Silhouette = scib.metrics.silhouette(..., embed=emb, scale=True)   # rescaled to [0,1]

Run in the 'scib' conda env (scib 1.1.5).
  conda run -n scib --no-capture-output python score_scib.py \
      --emb EMB.npy --obs-names OBS.csv --pseudo-gt ADATA.h5ad --method NAME --out OUT.csv
"""
from __future__ import print_function
import argparse, os, sys, warnings
warnings.simplefilter("ignore")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scib
from sklearn import metrics as skm


def load_obs_names(path):
    df = pd.read_csv(path)
    col = "obs_name" if "obs_name" in df.columns else df.columns[0]
    return df[col].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--obs-names", required=True)
    ap.add_argument("--pseudo-gt", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gt-key", default="niche_type_sub")
    args = ap.parse_args()

    emb = np.load(args.emb)
    obs_names = load_obs_names(args.obs_names)
    if emb.shape[0] != len(obs_names):
        raise ValueError("emb rows %d != obs_names %d" % (emb.shape[0], len(obs_names)))
    emb_df = pd.DataFrame(np.asarray(emb, dtype=np.float32), index=[str(x) for x in obs_names])

    gt = sc.read_h5ad(args.pseudo_gt)
    gt.obs_names = gt.obs_names.astype(str)
    if args.gt_key not in gt.obs:
        raise KeyError("%s not in pseudo-GT obs: %s" % (args.gt_key, list(gt.obs.columns)))

    # align by obs_name -> pseudo-GT order, intersection only
    common = [n for n in gt.obs_names if n in set(emb_df.index)]
    if len(common) < 0.5 * gt.n_obs:
        print("WARN only %d/%d cells overlap" % (len(common), gt.n_obs), file=sys.stderr)
    E = emb_df.loc[common].values.astype(np.float32)

    a = ad.AnnData(X=np.zeros((len(common), 1), dtype=np.float32),
                   obs=pd.DataFrame(index=common))
    a.obsm["emb"] = E
    lab = gt.obs.loc[common, args.gt_key].astype(str)
    # scib needs a categorical label with no NaN
    lab = lab.fillna("NaN")
    a.obs[args.gt_key] = pd.Categorical(lab.values)

    sc.pp.neighbors(a, use_rep="emb")
    scib.metrics.cluster_optimal_resolution(a, cluster_key="cluster", label_key=args.gt_key)
    ari = float(scib.metrics.ari(a, cluster_key="cluster", label_key=args.gt_key))
    nmi = float(scib.metrics.nmi(a, cluster_key="cluster", label_key=args.gt_key))
    label = a.obs[args.gt_key].values
    pred = a.obs["cluster"].values
    ami = float(np.round(skm.adjusted_mutual_info_score(label, pred), 4))
    hom = float(np.round(skm.homogeneity_score(label, pred), 4))
    sht = float(scib.metrics.silhouette(a, label_key=args.gt_key, embed="emb", scale=True))

    row = dict(method=args.method, n_obs_scored=int(len(common)),
               n_target=int(len(np.unique(label))), n_pred=int(len(np.unique(pred))),
               ARI=round(ari, 4), NMI=round(nmi, 4), AMI=round(ami, 4),
               Homogeneity=round(hom, 4), Silhouette=round(sht, 4), gt_key=args.gt_key)
    df = pd.DataFrame([row])
    write_header = not os.path.exists(args.out)
    df.to_csv(args.out, mode="a", header=write_header, index=False)
    print("[SCORED] %s  ARI=%.4f NMI=%.4f AMI=%.4f HOM=%.4f SHT=%.4f (n=%d, k=%d->%d)" % (
        args.method, ari, nmi, ami, hom, sht, len(common),
        len(np.unique(label)), len(np.unique(pred))))


if __name__ == "__main__":
    main()
