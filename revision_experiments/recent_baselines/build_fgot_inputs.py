#!/usr/bin/env python
"""Reconstruct the preprocessed FGOT P22 inputs locally from the raw AnnData + GTF.

The published FGOT P22 pipeline consumes preprocessed artifacts that are NOT on this
box (feature_selected_RNA.txt, feature_selected_ATAC.txt, P22_wsnn.txt, mm10_TSS.txt,
P22_putative_clusters.txt). This script rebuilds faithful local stand-ins from:

  * raw paired AnnData (9215 spots; counts in layers['counts'])
  * the local Ensembl GRCm38.100 GTF (for per-gene TSS)

Methodological approximations (MUST be disclosed in the paper):

  1. Feature selection. The published files used a Seurat pipeline (RNA: vst HVG;
     ATAC: TFIDF + FindTopFeatures) on metacell-aggregated counts. We replicate the
     spirit with scanpy seurat_v3 HVG directly on the raw per-spot counts. We then
     restrict genes/peaks to those that actually participate in at least one
     peak-gene edge of the prior feature graph (FGOT requires a connected bipartite
     graph), which naturally trims the gene set toward the published ~503 / ~5061.

  2. TSS. Per-gene TSS taken from the GTF gene records: start if strand '+', else end.
     Chromosomes are 'chr'-prefixed to match the ATAC peak naming.

  3. Cross-modality cost (the Seurat WNN P22_wsnn). We approximate the WNN cost with a
     joint paired-spot similarity built from PCA(RNA) + LSI(ATAC): for each pair of
     spots we average the RNA-graph and ATAC-graph SNN connectivities (a python
     stand-in for Seurat's weighted nearest-neighbor graph), then follow the tutorial
     exactly: cost = exp(1 - wnn); cost = cost - cost.min(). This is an approximation
     of Seurat's learned modality weights, not an exact reproduction.

  4. Putative clusters. The published P22_putative_clusters.txt came from Seurat WNN
     Louvain clustering. We use the dataset-provided RNA_clusters as the paired
     cluster label (used only for FGOT mini-batch stratified sampling, not evaluation).
"""

from __future__ import print_function

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData

RNA_PATH = "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_rna.h5ad"
ATAC_PATH = "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_atac.h5ad"
GTF_PATH = "/data2/zhouwg_data/project/Garfield-reproducibility/Mus_musculus.GRCm38.100.gtf.gz"
OUT_DIR = Path("/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/recent_baselines/fgot_inputs")

# tutorial reference target sizes (for reporting only)
TARGET_RNA = 503
TARGET_ATAC = 5061


def parse_gtf_tss(gtf_path):
    """Return DataFrame with columns chr, starts, ends, genes (one TSS per gene name)."""
    rows = []
    with gzip.open(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] != "gene":
                continue
            chrom, start, end, strand, attrs = f[0], int(f[3]), int(f[4]), f[6], f[8]
            name = None
            for field in attrs.split(";"):
                field = field.strip()
                if field.startswith("gene_name"):
                    name = field.split('"')[1]
                    break
            if name is None:
                continue
            tss = start if strand == "+" else end
            rows.append((("chr" + chrom) if not chrom.startswith("chr") else chrom, tss, tss, name))
    df = pd.DataFrame(rows, columns=["chr", "starts", "ends", "genes"])
    # one TSS per gene name (drop dup gene names, keep first)
    df = df.drop_duplicates("genes").reset_index(drop=True)
    return df


def peak_colon_to_dash(names):
    """chr1:3094734-3095650 -> chr1-3094734-3095650 (FGOT splits on '-' expecting 3 parts)."""
    out = []
    for n in names:
        chrom, rng = n.split(":")
        start, end = rng.split("-")
        out.append("{}-{}-{}".format(chrom, start, end))
    return out


def select_hvg_counts(adata, n_top, layer="counts"):
    a = adata.copy()
    a.X = a.layers[layer].copy()
    sc.pp.filter_genes(a, min_cells=3)
    n_top = min(n_top, a.n_vars)
    sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_top, layer=None)
    return list(a.var_names[a.var["highly_variable"].values])


def lsi(counts, n_comps=30):
    """TF-IDF + SVD (Signac-style LSI). Returns spots x (n_comps-1) embedding, dropping comp 0."""
    from scipy import sparse
    from sklearn.utils.extmath import randomized_svd

    X = counts
    if not sparse.issparse(X):
        X = sparse.csr_matrix(X)
    X = X.astype(np.float64)
    # TF-IDF (Signac default: log(TF*IDF))
    npeaks = np.asarray(X.sum(axis=1)).ravel()
    npeaks[npeaks == 0] = 1.0
    tf = X.multiply(1.0 / npeaks[:, None]).tocsr()
    ncells = X.shape[0]
    colsum = np.asarray((X > 0).sum(axis=0)).ravel().astype(np.float64)
    colsum[colsum == 0] = 1.0
    idf = ncells / colsum
    tfidf = tf.multiply(idf[None, :]).tocsr()
    tfidf.data = np.log1p(tfidf.data * 1e4)
    U, S, Vt = randomized_svd(tfidf, n_components=n_comps, random_state=0)
    emb = U * S
    return emb[:, 1:]  # drop first component (depth-correlated)


def snn_connectivities(emb, n_neighbors=20):
    a = AnnData(np.ascontiguousarray(emb.astype(np.float32)))
    sc.pp.neighbors(a, n_neighbors=n_neighbors, use_rep="X", random_state=0)
    return a.obsp["connectivities"]


def build_wnn_similarity(rna_emb, atac_emb, n_neighbors=20):
    """Paired-spot WNN-like similarity: mean of RNA and ATAC SNN connectivities, [0,1]."""
    c_rna = snn_connectivities(rna_emb, n_neighbors).toarray()
    c_atac = snn_connectivities(atac_emb, n_neighbors).toarray()
    n = c_rna.shape[0]
    # normalise each to [0,1] then average -> WNN-like joint affinity
    def norm01(m):
        mx = m.max()
        return m / mx if mx > 0 else m
    wnn = 0.5 * (norm01(c_rna) + norm01(c_atac))
    np.fill_diagonal(wnn, 1.0)
    return wnn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rna", default=RNA_PATH)
    ap.add_argument("--atac", default=ATAC_PATH)
    ap.add_argument("--gtf", default=GTF_PATH)
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--n-hvg-rna", type=int, default=3000)
    ap.add_argument("--n-hvg-atac", type=int, default=8000)
    ap.add_argument("--scope", type=int, default=150000)
    ap.add_argument("--max-obs", type=int, default=None, help="subset spots for smoke test")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading raw AnnData")
    rna = sc.read_h5ad(args.rna)
    atac = sc.read_h5ad(args.atac)
    # align spot order
    common = [n for n in rna.obs_names if n in set(atac.obs_names)]
    rna = rna[common].copy()
    atac = atac[common].copy()

    if args.max_obs is not None and rna.n_obs > args.max_obs:
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(np.arange(rna.n_obs), size=args.max_obs, replace=False))
        rna = rna[idx].copy()
        atac = atac[idx].copy()
    spots = list(rna.obs_names)
    n_spots = len(spots)
    print("    spots:", n_spots)

    print("[2/6] building TSS table from GTF")
    promoters = parse_gtf_tss(args.gtf)
    promoters.to_csv(out / "mm10_TSS.txt", sep="\t", index=False)
    print("    TSS genes:", len(promoters))

    print("[3/6] feature selection (HVG)")
    hvg_rna = select_hvg_counts(rna, args.n_hvg_rna)
    # restrict RNA HVG to genes that have a TSS (needed for the feature graph)
    tss_genes = set(promoters["genes"])
    hvg_rna = [g for g in hvg_rna if g in tss_genes]
    hvg_atac = select_hvg_counts(atac, args.n_hvg_atac)
    print("    RNA HVG (with TSS):", len(hvg_rna), " ATAC HVG:", len(hvg_atac))

    # prior feature graph to trim to connected genes/peaks
    import sys
    sys.path.insert(0, "/data2/zhouwg_data/project/Garfield_benchmark/FGOT")
    from FGOT import preprocess as pre

    peak_dash = peak_colon_to_dash(hvg_atac)
    prom2 = promoters[promoters["genes"].isin(hvg_rna)].copy()
    print("    building prior feature graph (scope=%d) ..." % args.scope)
    feat = pre.prior_feature_graph(prom2, peak_dash, hvg_rna, scope=args.scope)
    # feat: peaks x genes, inf where no edge. keep only connected nodes.
    finite = np.isfinite(feat.values)
    keep_peaks_mask = finite.any(axis=1)
    keep_genes_mask = finite.any(axis=0)
    sel_peaks_dash = [p for p, k in zip(feat.index, keep_peaks_mask) if k]
    sel_genes = [g for g, k in zip(feat.columns, keep_genes_mask) if k]
    print("    connected -> genes:", len(sel_genes), " peaks:", len(sel_peaks_dash))

    # map selected dashed peaks back to original colon names (preserve hvg_atac order)
    dash2colon = {d: c for d, c in zip(peak_dash, hvg_atac)}
    sel_peaks_colon = [dash2colon[d] for d in sel_peaks_dash]

    # final feature matrix restricted to connected nodes (index=dashed peaks, cols=genes)
    feat_final = feat.loc[sel_peaks_dash, sel_genes]
    feat_final.to_pickle(out / "feature_matrix.pkl")

    print("[4/6] writing feature-selected expression (normalised, genes/peaks x spots)")
    # RNA: log-normalised
    rna_sel = rna[:, sel_genes].copy()
    rna_sel.X = rna_sel.layers["counts"].copy()
    sc.pp.normalize_total(rna_sel, target_sum=1e4)
    sc.pp.log1p(rna_sel)
    rna_mat = pd.DataFrame(
        np.asarray(rna_sel.X.todense() if hasattr(rna_sel.X, "todense") else rna_sel.X).T,
        index=sel_genes, columns=spots,
    )
    rna_mat.to_pickle(out / "feature_selected_RNA.pkl")

    # ATAC: TF-IDF-style normalisation -> use log1p of normalized counts (per-spot)
    atac_sel = atac[:, sel_peaks_colon].copy()
    atac_sel.X = atac_sel.layers["counts"].copy()
    sc.pp.normalize_total(atac_sel, target_sum=1e4)
    sc.pp.log1p(atac_sel)
    atac_mat = pd.DataFrame(
        np.asarray(atac_sel.X.todense() if hasattr(atac_sel.X, "todense") else atac_sel.X).T,
        index=sel_peaks_dash, columns=spots,  # dashed names so they match feature_matrix index
    )
    atac_mat.to_pickle(out / "feature_selected_ATAC.pkl")

    print("[5/6] building WNN-approx cross-modality cost")
    # PCA(RNA) on log-normalised full HVG; LSI(ATAC) on counts
    rna_full = rna.copy()
    rna_full.X = rna_full.layers["counts"].copy()
    sc.pp.normalize_total(rna_full, target_sum=1e4)
    sc.pp.log1p(rna_full)
    sc.pp.highly_variable_genes(rna_full, n_top_genes=min(2000, rna_full.n_vars - 1))
    rna_full = rna_full[:, rna_full.var["highly_variable"].values].copy()
    sc.pp.scale(rna_full, max_value=10)
    sc.tl.pca(rna_full, n_comps=min(30, rna_full.n_obs - 1, rna_full.n_vars - 1))
    rna_emb = rna_full.obsm["X_pca"]

    from scipy import sparse as _sp
    atac_counts = atac.layers["counts"]
    if not _sp.issparse(atac_counts):
        atac_counts = _sp.csr_matrix(atac_counts)
    atac_emb = lsi(atac_counts, n_comps=min(31, atac.n_obs - 1))

    wnn = build_wnn_similarity(rna_emb, atac_emb, n_neighbors=min(20, n_spots - 1))
    cost = np.exp(1 - wnn)
    cost = cost - cost.min()
    cost_df = pd.DataFrame(cost, index=spots, columns=spots)
    cost_df.to_pickle(out / "P22_wsnn_cost.pkl")

    print("[6/6] writing clusters + manifest")
    clusters = pd.DataFrame(
        {"cell": spots, "cluster": rna.obs["RNA_clusters"].astype(str).values}
    )
    clusters.to_csv(out / "P22_putative_clusters.txt", sep="\t", index=False)

    spatial = np.asarray(rna.obsm["spatial"])
    np.save(out / "spatial.npy", spatial)
    pd.Series(spots, name="obs_name").to_csv(out / "obs_names.csv", index=False)

    manifest = {
        "n_spots": int(n_spots),
        "n_genes_selected": int(len(sel_genes)),
        "n_peaks_selected": int(len(sel_peaks_dash)),
        "target_rna": TARGET_RNA,
        "target_atac": TARGET_ATAC,
        "n_tss_genes": int(len(promoters)),
        "scope": int(args.scope),
        "n_hvg_rna": int(args.n_hvg_rna),
        "n_hvg_atac": int(args.n_hvg_atac),
        "cost_min": float(cost.min()),
        "cost_max": float(cost.max()),
        "subset_max_obs": args.max_obs,
        "approximations": [
            "scanpy seurat_v3 HVG on raw per-spot counts instead of Seurat metacell HVG",
            "genes/peaks trimmed to prior-feature-graph connected nodes",
            "WNN cost approximated by mean RNA/ATAC SNN connectivity (PCA+LSI), cost=exp(1-wnn)-min",
            "RNA_clusters used as paired mini-batch cluster label",
        ],
    }
    with (out / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    print("manifest:", json.dumps(manifest, indent=2))
    print("DONE")


if __name__ == "__main__":
    main()
