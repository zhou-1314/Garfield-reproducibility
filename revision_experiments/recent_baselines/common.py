#!/usr/bin/env python
"""Shared utilities for recent spatial multi-omics baseline reruns."""

from __future__ import print_function

import json
import os
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = Path("/data2/zhouwg_data/project/Garfield_tutorials/data")
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "revision_experiments" / "results" / "recent_baselines" / "spatial_atac_rna_p22"

RNA_PATH = DEFAULT_DATA_DIR / "spatial_atac_rna_seq_mouse_brain_rna.h5ad"
ATAC_PATH = DEFAULT_DATA_DIR / "spatial_atac_rna_seq_mouse_brain_atac.h5ad"
ANNOTATION_PATH = DEFAULT_DATA_DIR / "spatial_atac_rna_seq_mouse_brain_cell_type_annotations.csv"


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def now_seconds():
    return time.time()


def duration_seconds(started_at):
    return round(time.time() - started_at, 3)


def read_labels(annotation_path=ANNOTATION_PATH):
    ann = pd.read_csv(annotation_path)
    if "Unnamed: 0" in ann.columns:
        ann = ann.rename(columns={"Unnamed: 0": "obs_name"})
    if "obs_name" not in ann.columns:
        ann = ann.rename(columns={ann.columns[0]: "obs_name"})
    ann["obs_name"] = ann["obs_name"].astype(str)
    ann = ann.set_index("obs_name", drop=True)
    return ann


def attach_common_obs(adata, labels):
    labels = labels.copy()
    labels.index = labels.index.astype(str)
    obs_names = adata.obs_names.astype(str)
    for col in labels.columns:
        adata.obs[col] = labels.reindex(obs_names)[col].values
    return adata


def dense_array(x, dtype=np.float32):
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x, dtype=dtype)


def write_status(path, method, status, message="", extra=None):
    payload = {
        "method": method,
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    if extra:
        payload.update(extra)
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return payload


def write_failure(path, method, exc, extra=None):
    detail = {
        "exception_type": type(exc).__name__,
        "traceback": traceback.format_exc(),
    }
    if extra:
        detail.update(extra)
    return write_status(path, method, "failed", str(exc), detail)


def write_blocked(path, method, message, extra=None):
    return write_status(path, method, "blocked", message, extra)


def _safe_silhouette(embedding, labels):
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(uniq) >= len(labels):
        return np.nan
    try:
        return float(metrics.silhouette_score(embedding, labels, metric="euclidean"))
    except Exception:
        return np.nan


def compute_label_metrics(y_true, y_pred, embedding=None):
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    out = {
        "ARI": float(metrics.adjusted_rand_score(y_true, y_pred)),
        "NMI": float(metrics.normalized_mutual_info_score(y_true, y_pred)),
        "AMI": float(metrics.adjusted_mutual_info_score(y_true, y_pred)),
        "HOM": float(metrics.homogeneity_score(y_true, y_pred)),
        "COM": float(metrics.completeness_score(y_true, y_pred)),
        "VMEASURE": float(metrics.v_measure_score(y_true, y_pred)),
        "n_clusters_pred": int(len(np.unique(y_pred))),
        "n_labels": int(len(np.unique(y_true))),
        "n_obs": int(len(y_true)),
    }
    if embedding is not None:
        out["ASW_pred"] = _safe_silhouette(embedding, y_pred)
        out["ASW_label"] = _safe_silhouette(embedding, y_true)
    return out


def evaluate_embedding(
    embedding,
    labels_df,
    out_csv,
    method,
    representation_key="embedding",
    label_keys=("predicted.celltype", "ATAC_clusters"),
    resolutions=None,
    n_neighbors=50,
    random_state=2024,
):
    import scanpy as sc
    from anndata import AnnData

    if resolutions is None:
        resolutions = np.round(np.arange(0.1, 2.05, 0.05), 3)
    embedding = np.asarray(embedding, dtype=np.float32)
    labels_df = labels_df.copy()
    labels_df.index = labels_df.index.astype(str)

    adata = AnnData(X=np.zeros((embedding.shape[0], 1), dtype=np.float32), obs=labels_df.iloc[: embedding.shape[0]].copy())
    adata.obsm[representation_key] = embedding
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=representation_key, random_state=random_state)

    rows = []
    cluster_keys = []
    for resolution in resolutions:
        key = "leiden_r{:.3f}".format(float(resolution)).replace(".", "_")
        sc.tl.leiden(adata, resolution=float(resolution), key_added=key, random_state=random_state)
        cluster_keys.append((key, float(resolution), int(adata.obs[key].nunique())))

    for label_key in label_keys:
        if label_key not in adata.obs:
            continue
        mask = adata.obs[label_key].notna().values
        if mask.sum() == 0:
            continue
        y_true = adata.obs.loc[mask, label_key].astype(str).values
        target_n = len(np.unique(y_true))
        best = None
        for cluster_key, resolution, n_pred in cluster_keys:
            y_pred = adata.obs.loc[mask, cluster_key].astype(str).values
            row = compute_label_metrics(y_true, y_pred, embedding=embedding[mask])
            row.update(
                {
                    "method": method,
                    "label_key": label_key,
                    "resolution": resolution,
                    "cluster_key": cluster_key,
                    "representation": representation_key,
                    "target_n_labels": target_n,
                    "abs_cluster_delta": abs(n_pred - target_n),
                }
            )
            rows.append(row)
            rank_key = (row["abs_cluster_delta"], -row["ARI"], -row["NMI"])
            if best is None or rank_key < best[0]:
                best = (rank_key, row)

    result = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    ensure_dir(out_csv.parent)
    result.to_csv(out_csv, index=False)

    best_rows = []
    if not result.empty:
        for label_key, grp in result.groupby("label_key"):
            grp = grp.sort_values(["abs_cluster_delta", "ARI", "NMI"], ascending=[True, False, False])
            best_rows.append(grp.iloc[0])
    best_df = pd.DataFrame(best_rows)
    best_csv = out_csv.with_name(out_csv.stem + "_best.csv")
    best_df.to_csv(best_csv, index=False)
    return result, best_df
