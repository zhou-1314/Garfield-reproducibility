#!/usr/bin/env python
"""Score a method's embedding against the P22 pseudo ground-truth partition.

This standalone scorer reproduces, verbatim in spirit, the metric protocol used
by the project's existing benchmark notebook

    Garfield_code/spatial-multi-modal-code/metrics_benchmark.ipynb

so that numbers produced for a NEW recent-baseline method are directly
comparable to the already-published SpatialGlue / MOFA / NicheCompass / MultiVI
numbers on the spatial ATAC+RNA mouse-brain (P22) benchmark.

Protocol
--------
1. Load a pseudo ground-truth AnnData that carries obs[niche_type_sub] (~14
   niches) and obsm['spatial'].
2. Align the supplied embedding to the pseudo-GT cells BY obs_name (some methods
   reorder / subset cells), keep the intersection, order by the pseudo-GT order.
3. Build a working AnnData with obsm['emb'] = aligned embedding,
   obsm['spatial'] = pseudo-GT spatial, obs['niche_type_sub'].
4. n_clusters target = number of unique niche_type_sub categories (expect 14).
   sc.pp.neighbors(use_rep='emb'); search_res to that target; sc.tl.leiden at
   the found resolution -> predicted clusters.
5. Compute label metrics (ARI / NMI / AMI / HOM via sklearn), ASW
   (silhouette_score(embedding, niche_type_sub)) and the spatial metrics
   (CHAOS, PAS, Moran_I, spatial_coherence) replicated from the notebook.
6. Append one row to the output CSV (create with header, otherwise append
   without header).

The metric helper functions below are copied verbatim from the notebook so that
behaviour matches exactly.
"""

from __future__ import print_function

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import scanpy as sc
from sklearn import metrics
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import StandardScaler
from pandas import get_dummies


# ---------------------------------------------------------------------------
# Spatial-metric helpers (copied verbatim from metrics_benchmark.ipynb)
# ---------------------------------------------------------------------------
def _get_spatial_entropy(C, C_sum):
    H = 0
    for i in range(len(C)):
        for j in range(len(C)):
            z = C[i, j]
            if z != 0:
                H += -(z / C_sum) * np.log(z / C_sum)
    return H


def spatial_entropy(k_neighbors, labels, degree=4):
    """
    Calculates spatial entropy of graph
    """
    S = np.broadcast_to(labels[:, None], (len(labels), degree))
    N = labels[k_neighbors]
    cluster_names = np.unique(labels)
    cluster_nums = len(cluster_names)
    C = np.zeros((cluster_nums, cluster_nums))
    for i in range(cluster_nums):
        for j in range(cluster_nums):
            C[i, j] = np.sum(np.logical_and(S == cluster_names[i], N == cluster_names[j]))
    C_sum = C.sum()
    return _get_spatial_entropy(C, C_sum)


def sigmoid_score(raw_score):
    return 1 / (1 + np.exp(-raw_score))


def spatial_coherence_score(adata, annotation_key, degree=4, rep_time=1000, seed=0):
    spatial_coords = adata.obsm['spatial']
    origin_labels = adata.obs[annotation_key].values
    # Use kneighbors_graph to get the adjacency matrix
    neigh = NearestNeighbors(n_neighbors=degree, metric='euclidean').fit(spatial_coords)
    k_neighbors = neigh.kneighbors(n_neighbors=degree, return_distance=False)
    true_entropy = spatial_entropy(k_neighbors, origin_labels, degree=degree)
    entropies = []
    rng = np.random.default_rng(seed)
    shuffled_labels = origin_labels.copy()
    for _ in range(rep_time):
        rng.shuffle(shuffled_labels)
        entropies.append(spatial_entropy(k_neighbors, shuffled_labels, degree=degree))

    raw_score = (true_entropy - np.mean(entropies)) / np.std(entropies)
    normalized_score = sigmoid_score(raw_score)

    return normalized_score, true_entropy, entropies


def CHAOS_score(X, pred_labels):
    """
    Calculate the CHAOS score for a given set of spatial coordinates and predicted labels.

    param: X - spatial coordinates
    param: pred_labels - predicted labels

    return: CHAOS score
    """
    # Standardize the spatial coordinates
    X = StandardScaler().fit_transform(X)

    # Get the unique cluster labels
    cluster_labels = np.unique(pred_labels)

    # Initialize the distance value and count
    dist_val = 0.
    count = 0

    # Iterate through each cluster
    for k in cluster_labels:
        # Get the spatial coordinates for the current cluster
        cluster_coords = X[pred_labels == k, :]

        # Check if there are at least 2 spatial coordinates in the cluster
        if len(cluster_coords) <= 2:
            continue
        else:
            count += len(cluster_coords)

        # Calculate the distance to the nearest neighbor for each spatial coordinate in the cluster
        nbrs = NearestNeighbors(n_neighbors=1).fit(cluster_coords)
        distances, _ = nbrs.kneighbors()

        # Sum the distances
        dist_val = dist_val + np.sum(distances)

    # Calculate the CHAOS score
    return dist_val / count


def PAS_score(X, pred_labels, k=6):
    """
    Calculate the PAS score for a given set of spatial coordinates and predicted labels.

    param: X - spatial coordinates
    param: pred_labels - predicted labels
    param: k - number of nearest neighbors to consider

    return: PAS score
    """
    # Use NearestNeighbors to find the nearest neighbors
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    indices = nbrs.kneighbors(return_distance=False)
    # Calculate the PAS score
    return ((pred_labels.reshape(-1, 1) != pred_labels[indices]).sum(1) > k / 2).mean()


def moranI_score(adata, key):
    g = kneighbors_graph(adata.obsm['spatial'], 6, mode='connectivity', metric='euclidean')
    one_hot = get_dummies(adata.obs[key])
    moranI = sc.metrics.morans_i(g, one_hot.values.T).mean()
    return moranI


def search_res(adata, n_clusters, use_rep='emb', cluster_method='leiden', start=0.5, end=2.0,
               increment=0.05, expansion_factor=1.75, max_iter=50):
    """Search the Leiden/Louvain resolution whose cluster count is closest to
    n_clusters, expanding the search range if no exact match is found.

    Copied verbatim (in spirit) from metrics_benchmark.ipynb. The neighbor graph
    is built here under `key_added=use_rep` so that downstream `sc.tl.leiden`
    calls can reference it via `neighbors_key=use_rep`.
    """
    print('Searching resolution...')
    label = 0
    method = use_rep
    method_cluster = method + '_cluster'

    current_start, current_end = start, end
    best_res = None
    best_diff = float('inf')  # Track the best resolution so far
    best_count = 0
    iteration = 0  # Track number of iterations

    # build graph
    sc.pp.neighbors(adata, use_rep=use_rep, key_added=method)

    while label == 0 and iteration < max_iter:
        iteration += 1
        for res in sorted(list(np.arange(current_start, current_end, increment)), reverse=True):
            if cluster_method == 'leiden':
                sc.tl.leiden(adata, random_state=0, key_added=method_cluster,
                             resolution=res, neighbors_key=method)
                count_unique = adata.obs[method_cluster].nunique()
            elif cluster_method == 'louvain':
                sc.tl.louvain(adata, random_state=0, key_added=method_cluster,
                              resolution=res, neighbors_key=method)
                count_unique = adata.obs[method_cluster].nunique()

            print(f'resolution={res}, cluster number={count_unique}')

            # Update the best resolution if it's closer to the target
            diff = abs(count_unique - n_clusters)
            if diff < best_diff:
                best_diff = diff
                best_res = res
                best_count = count_unique

            if count_unique == n_clusters:
                label = 1
                break

        if label == 0:
            if best_res is not None:
                # Instead of starting from scratch, continue from the closest found resolution
                if best_count > n_clusters:
                    current_start = min(best_res, current_start * 0.9)  # Avoid too small a search range
                    current_end = min(current_start * expansion_factor, end * 2)  # Limit max range
                    print(f"Expanding search range: new range ({current_start}, {current_end})")
                else:
                    current_start = max(best_res, current_start * 1.2)  # Avoid too small a search range
                    current_end = min(current_start * expansion_factor, end * 2)  # Limit max range
                    print(f"Expanding search range: new range ({current_start}, {current_end})")
            else:
                print("Warning: No valid resolution found yet, expanding the search range.")
                current_start *= expansion_factor
                current_end *= expansion_factor

    if iteration >= max_iter:
        print("Warning: Reached maximum iteration limit.")

    return best_res if label == 1 else best_res


# ---------------------------------------------------------------------------
# Alignment + scoring
# ---------------------------------------------------------------------------
def load_obs_names(obs_csv):
    """Read OBS.csv and return obs_name list in EMB row order."""
    df = pd.read_csv(obs_csv)
    if "obs_name" not in df.columns:
        # Fall back to first column for robustness.
        df = df.rename(columns={df.columns[0]: "obs_name"})
    return df["obs_name"].astype(str).values


def build_aligned_adata(emb, emb_obs_names, pseudo_gt, gt_key):
    """Align embedding rows to pseudo-GT cells by obs_name.

    Returns an AnnData with obsm['emb'], obsm['spatial'], obs[gt_key], ordered by
    the pseudo-GT order, restricted to obs_names present in BOTH.
    """
    from anndata import AnnData

    emb_obs_names = np.asarray(emb_obs_names).astype(str)
    if emb.shape[0] != emb_obs_names.shape[0]:
        raise ValueError(
            "EMB rows ({}) and OBS.csv rows ({}) disagree.".format(emb.shape[0], emb_obs_names.shape[0])
        )

    emb_index = {name: i for i, name in enumerate(emb_obs_names)}

    gt_names = pseudo_gt.obs_names.astype(str).values
    keep_mask = np.array([name in emb_index for name in gt_names])
    if keep_mask.sum() == 0:
        raise ValueError("No obs_name overlap between EMB/OBS.csv and the pseudo-GT AnnData.")

    gt_order_names = gt_names[keep_mask]
    emb_rows = np.array([emb_index[name] for name in gt_order_names])

    aligned_emb = np.asarray(emb, dtype=np.float32)[emb_rows]
    gt_sub = pseudo_gt[keep_mask].copy()

    spatial = np.asarray(gt_sub.obsm["spatial"])
    niche = gt_sub.obs[gt_key].values

    adata = AnnData(X=np.zeros((aligned_emb.shape[0], 1), dtype=np.float32))
    adata.obs_names = gt_order_names
    adata.obsm["emb"] = aligned_emb
    adata.obsm["spatial"] = spatial
    adata.obs[gt_key] = pd.Categorical(np.asarray(niche).astype(str))
    return adata


def score(emb, emb_obs_names, pseudo_gt, method, gt_key):
    adata = build_aligned_adata(emb, emb_obs_names, pseudo_gt, gt_key)

    n_target = int(adata.obs[gt_key].nunique())

    # Build neighbor graph on the embedding and search resolution -> leiden.
    res = search_res(adata, n_target, use_rep="emb", cluster_method="leiden")
    sc.tl.leiden(adata, random_state=0, key_added="pred_cluster",
                 resolution=res, neighbors_key="emb")

    pred_labels = adata.obs["pred_cluster"].values
    y_true = adata.obs[gt_key].astype(str).values
    y_pred = np.asarray(pred_labels).astype(str)

    n_pred = int(adata.obs["pred_cluster"].nunique())

    # Label metrics via sklearn.
    ari = float(metrics.adjusted_rand_score(y_true, y_pred))
    nmi = float(metrics.normalized_mutual_info_score(y_true, y_pred))
    ami = float(metrics.adjusted_mutual_info_score(y_true, y_pred))
    hom = float(metrics.homogeneity_score(y_true, y_pred))

    # ASW = silhouette of the embedding against the niche_type_sub labels.
    asw = float(silhouette_score(adata.obsm["emb"], y_true))

    # Spatial metrics (replicated from notebook).
    chaos = float(CHAOS_score(X=adata.obsm["spatial"], pred_labels=pred_labels))
    pas = float(PAS_score(X=adata.obsm["spatial"], pred_labels=pred_labels))
    moran = float(moranI_score(adata, key="pred_cluster"))
    coherence = float(spatial_coherence_score(adata, annotation_key="pred_cluster")[0])

    row = {
        "method": method,
        "n_obs_scored": int(adata.n_obs),
        "n_target": n_target,
        "n_pred": n_pred,
        "leiden_res": float(res) if res is not None else np.nan,
        "ARI": ari,
        "NMI": nmi,
        "AMI": ami,
        "HOM": hom,
        "ASW": asw,
        "CHAOS": chaos,
        "PAS": pas,
        "Moran_I": moran,
        "spatial_coherence": coherence,
    }
    return row


COLUMNS = [
    "method", "n_obs_scored", "n_target", "n_pred", "leiden_res",
    "ARI", "NMI", "AMI", "HOM", "ASW",
    "CHAOS", "PAS", "Moran_I", "spatial_coherence",
]


def write_row(out_csv, row):
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=COLUMNS)
    if out_csv.exists():
        df.to_csv(out_csv, mode="a", header=False, index=False)
    else:
        df.to_csv(out_csv, index=False)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--emb", required=True, help="(n_cells, d) float embedding .npy")
    parser.add_argument("--obs-names", required=True,
                        help="CSV with a column 'obs_name' giving the obs_name for each EMB row (same order as EMB)")
    parser.add_argument(
        "--pseudo-gt",
        default="/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/results/"
                "recent_baselines/spatial_atac_rna_p22/pseudo_gt/adata_pseudo_gt.h5ad",
        help="pseudo-GT AnnData with obs[niche_type_sub] and obsm['spatial']")
    parser.add_argument("--method", required=True, help="method name written into the row")
    parser.add_argument("--out", required=True, help="output CSV (append-or-create)")
    parser.add_argument("--gt-key", default="niche_type_sub", help="obs key of the pseudo ground-truth partition")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.pseudo_gt):
        raise FileNotFoundError("pseudo-GT AnnData not found: {}".format(args.pseudo_gt))

    emb = np.load(args.emb)
    emb_obs_names = load_obs_names(args.obs_names)
    pseudo_gt = sc.read_h5ad(args.pseudo_gt)
    if args.gt_key not in pseudo_gt.obs.columns:
        raise KeyError("gt-key '{}' not in pseudo-GT obs columns: {}".format(
            args.gt_key, list(pseudo_gt.obs.columns)))
    if "spatial" not in pseudo_gt.obsm:
        raise KeyError("pseudo-GT AnnData has no obsm['spatial'].")

    row = score(emb, emb_obs_names, pseudo_gt, args.method, args.gt_key)
    df = write_row(args.out, row)
    print("Wrote row to {}:".format(args.out))
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
