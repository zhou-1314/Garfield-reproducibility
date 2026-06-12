#!/usr/bin/env python
"""Run SpaMI on the P22 spatial ATAC-RNA benchmark using tutorial defaults."""

from __future__ import print_function

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.decomposition
import sklearn.metrics
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix
from tqdm import tqdm

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


SPAMI_SRC = Path("/data2/zhouwg_data/project/Garfield_benchmark/SpaMI/SpaMI")
sys.path.insert(0, str(SPAMI_SRC))

import episcanpy  # noqa: E402
from model import SpaMI  # noqa: E402
from utils import fix_seed  # noqa: E402


METHOD = "SpaMI"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_RESULTS_DIR / "spami"))
    parser.add_argument("--rna", default=str(RNA_PATH))
    parser.add_argument("--atac", default=str(ATAC_PATH))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max-obs", type=int, default=None)
    return parser.parse_args()


def maybe_subset(adata, max_obs, seed):
    if max_obs is None or adata.n_obs <= max_obs:
        return adata
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(np.arange(adata.n_obs), size=max_obs, replace=False))
    return adata[idx].copy()


def matrix_to_array(x):
    if isinstance(x, csc_matrix) or isinstance(x, csr_matrix):
        return x.toarray()
    return np.asarray(x)


def tfidf(x):
    idf = x.shape[0] / x.sum(axis=0)
    if hasattr(x, "multiply"):
        tf = x.multiply(1 / x.sum(axis=1))
        return tf.multiply(idf)
    tf = x / x.sum(axis=1, keepdims=True)
    return tf * idf


def construct_adj(adata, n_neighbors=3):
    coordinate = np.asarray(adata.obsm["spatial"])
    distance_matrix = sklearn.metrics.pairwise_distances(coordinate, metric="euclidean")
    n_spot = distance_matrix.shape[0]
    interaction = np.zeros((n_spot, n_spot), dtype=np.float32)
    for i in range(n_spot):
        distance = distance_matrix[i, :].argsort()
        for t in range(1, n_neighbors + 1):
            interaction[i, distance[t]] = 1
    adj = interaction + interaction.T
    adj = np.where(adj > 1, 1, adj)
    return adj, interaction


def add_contrastive_label(adata):
    n_spot = adata.n_obs
    return np.concatenate([np.ones((n_spot, 1)), np.zeros((n_spot, 1))], axis=1)


def permutation(feature):
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    return feature[ids]


def preprocess_rna_adata(adata):
    adata = adata.copy()
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000)
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    feat = matrix_to_array(adata_hvg.X)
    pca = sklearn.decomposition.PCA(n_components=50, random_state=2024)
    feat = pca.fit_transform(feat)
    adj, graph_neigh = construct_adj(adata, n_neighbors=6)
    return {
        "feat": feat,
        "adj": adj,
        "graph_neigh": graph_neigh,
        "label_CSL": add_contrastive_label(adata),
    }


def preprocess_atac_adata(adata):
    adata = adata.copy()
    adata_copy = adata.copy()
    episcanpy.pp.binarize(adata_copy)
    episcanpy.pp.filter_features(adata_copy, min_cells=np.ceil(0.005 * adata.shape[0]))
    count_mat = adata_copy.X.copy()
    x = tfidf(count_mat)
    x_norm = sklearn.preprocessing.Normalizer(norm="l1").fit_transform(x)
    x_norm = np.log1p(x_norm * 1e4)
    x_lsi = sklearn.utils.extmath.randomized_svd(x_norm, 51, random_state=2024)[0]
    x_lsi -= x_lsi.mean(axis=1, keepdims=True)
    x_lsi /= x_lsi.std(axis=1, ddof=1, keepdims=True)
    feat = matrix_to_array(x_lsi[:, 1:])
    adj, graph_neigh = construct_adj(adata, n_neighbors=6)
    return {
        "feat": feat,
        "adj": adj,
        "graph_neigh": graph_neigh,
        "label_CSL": add_contrastive_label(adata),
    }


def train_spami(omics1_data, omics2_data, out_dir, device, random_seed=2024):
    fix_seed(random_seed)
    out_dir = ensure_dir(out_dir)
    device = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    omics1_feat = torch.FloatTensor(omics1_data["feat"]).to(device)
    omics1_label_csl = torch.FloatTensor(omics1_data["label_CSL"]).to(device)
    omics1_graph_neigh = torch.FloatTensor(omics1_data["graph_neigh"] + np.eye(omics1_data["adj"].shape[0])).to(device)
    omics1_adj = torch.FloatTensor(omics1_data["adj"] + np.eye(omics1_data["adj"].shape[0])).to(device)

    omics2_feat = torch.FloatTensor(omics2_data["feat"]).to(device)
    omics2_label_csl = torch.FloatTensor(omics2_data["label_CSL"]).to(device)
    omics2_graph_neigh = torch.FloatTensor(omics2_data["graph_neigh"] + np.eye(omics2_data["adj"].shape[0])).to(device)
    omics2_adj = torch.FloatTensor(omics2_data["adj"] + np.eye(omics2_data["adj"].shape[0])).to(device)

    learning_rate = 0.001
    epochs = 400
    out_dim = 64
    dropout_rate = 0.15
    weight_decay = 0.0
    factors = [1, 1, 10, 15, 5]

    model = SpaMI(
        omics1_feat.shape[1],
        omics2_feat.shape[1],
        out_dim,
        dropout_rate,
        omics1_feat,
        omics1_adj,
        omics1_graph_neigh,
        omics1_label_csl,
        omics2_feat,
        omics2_adj,
        omics2_graph_neigh,
        omics2_label_csl,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)
    loss_rec = nn.MSELoss()
    loss_csl = nn.BCEWithLogitsLoss()
    losses = []
    omics1_feat_shuffle = None
    omics2_feat_shuffle = None

    for _epoch in tqdm(range(epochs), desc="SpaMI"):
        model.train()
        omics1_feat_shuffle = permutation(omics1_feat)
        omics2_feat_shuffle = permutation(omics2_feat)
        (
            omics1_emb,
            omics1_rec,
            omics1_ret,
            omics1_ret_a,
            omics2_emb,
            omics2_rec,
            omics2_ret,
            omics2_ret_a,
            _combine_emb,
            _omics_weight,
        ) = model(omics1_feat_shuffle, omics2_feat_shuffle)

        loss_omics1_csl = loss_csl(omics1_ret, omics1_label_csl) + loss_csl(omics1_ret_a, omics1_label_csl)
        loss_omics2_csl = loss_csl(omics2_ret, omics2_label_csl) + loss_csl(omics2_ret_a, omics2_label_csl)
        loss_omics1_rec = loss_rec(omics1_feat, omics1_rec)
        loss_omics2_rec = loss_rec(omics2_feat, omics2_rec)
        loss_cosine = 1 - F.cosine_similarity(omics1_emb, omics2_emb, dim=1).mean()
        loss = (
            factors[0] * loss_omics1_csl
            + factors[1] * loss_omics2_csl
            + factors[2] * loss_omics1_rec
            + factors[3] * loss_omics2_rec
            + factors[4] * loss_cosine
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

    with torch.no_grad():
        model.eval()
        torch.save(model.state_dict(), str(out_dir / "model.pt"))
        omics1_emb, omics1_rec, _, _, omics2_emb, omics2_rec, _, _, combine_emb, _ = model(
            omics1_feat_shuffle, omics2_feat_shuffle
        )
        emb_combine = combine_emb.detach().cpu().numpy()

    np.save(out_dir / "combine_emb.npy", emb_combine)
    pd.DataFrame({"epoch": np.arange(len(losses)), "loss": losses}).to_csv(out_dir / "loss.csv", index=False)
    plt.figure(figsize=(4, 3))
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(out_dir / "loss_plot.png", dpi=150)
    plt.close()
    return emb_combine, str(device)


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    status_path = out_dir / "spami_status.json"
    started = now_seconds()
    try:
        labels = read_labels()
        rna = sc.read_h5ad(args.rna)
        atac = sc.read_h5ad(args.atac)
        rna = attach_common_obs(rna, labels)
        atac = attach_common_obs(atac, labels)
        rna = maybe_subset(rna, args.max_obs, args.seed)
        atac = atac[rna.obs_names].copy()

        omics1_data = preprocess_rna_adata(rna)
        omics2_data = preprocess_atac_adata(atac)
        embedding, actual_device = train_spami(omics1_data, omics2_data, out_dir, args.device, args.seed)

        labels_df = rna.obs[["predicted.celltype", "ATAC_clusters"]].copy()
        labels_df.to_csv(out_dir / "labels_used.csv")
        evaluate_embedding(embedding, labels_df, out_dir / "spami_metrics.csv", method=METHOD)
        write_status(
            status_path,
            METHOD,
            "success",
            "SpaMI completed using Tutorial 2 P22 defaults.",
            {
                "duration_seconds": duration_seconds(started),
                "n_obs": int(rna.n_obs),
                "used_full_p22": args.max_obs is None,
                "device": actual_device,
                "source": str(SPAMI_SRC),
                "defaults": "RNA HVG=3000/PCA=50; ATAC TF-IDF/LSI=50; spatial neighbors=6; epochs=400; out_dim=64; factors=[1,1,10,15,5]",
            },
        )
    except Exception as exc:
        write_failure(status_path, METHOD, exc, {"duration_seconds": duration_seconds(started)})
        raise


if __name__ == "__main__":
    main()
