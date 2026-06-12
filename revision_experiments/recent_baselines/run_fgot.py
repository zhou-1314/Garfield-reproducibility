#!/usr/bin/env python
"""Run FGOT on the P22 spatial ATAC-RNA mouse-brain benchmark.

FGOT (Feature-Guided Optimal Transport) is run with the canonical P22 pipeline
(FGOT.fgot_sparse_tensor -> fgot_tol -> align). The published pipeline consumes
preprocessed Seurat artifacts that are NOT distributed; this runner consumes the
locally reconstructed stand-ins produced by build_fgot_inputs.py. See that file's
docstring (and fgot_inputs/manifest.json) for the methodological approximations,
which MUST be disclosed in the paper.

Final embedding: following the tutorial, after alignment we concatenate
(X2_aligned [RNA], X1_aligned [ATAC]), run sc.tl.pca, and the RNA-batch rows are
the 9215 per-spot embedding used for niche clustering. We save that RNA half.

Outputs (under --out-dir):
  fgot_emb.npy      : (n_spots, n_pcs) RNA-aligned PCA embedding
  obs_names.csv     : spot order matching fgot_emb rows
  labels_used.csv   : RNA_clusters / ATAC_clusters for the spots
  fgot_status.json  : run status + parameters + approximations
"""

from __future__ import print_function

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/data2/zhouwg_data/project/Garfield_benchmark/FGOT")

HERE = Path(__file__).resolve().parent
DEFAULT_INPUTS = HERE / "fgot_inputs"
DEFAULT_OUT = HERE / "fgot_out"
METHOD = "FGOT"


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=str(DEFAULT_INPUTS))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batchsize", type=int, default=600)
    ap.add_argument("--eps-p", type=float, default=1e-1)
    ap.add_argument("--rho-mu", type=float, default=10.0)
    ap.add_argument("--rho-nu", type=float, default=10.0)
    ap.add_argument("--n-pcs", type=int, default=50)
    return ap.parse_args()


def main():
    args = parse_args()
    inputs = Path(args.inputs)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    status_path = out / "fgot_status.json"
    started = time.time()

    try:
        import FGOT
        from FGOT import fgot_sparse_tensor, fgot_tol, align

        # ---- load reconstructed inputs -------------------------------------
        RNA_data = pd.read_pickle(inputs / "feature_selected_RNA.pkl")     # genes x spots
        ATAC_data = pd.read_pickle(inputs / "feature_selected_ATAC.pkl")   # peaks x spots
        feature_matrix = pd.read_pickle(inputs / "feature_matrix.pkl")     # peaks x genes (inf=no edge)
        cost_df = pd.read_pickle(inputs / "P22_wsnn_cost.pkl")             # spots x spots
        clusters = pd.read_csv(inputs / "P22_putative_clusters.txt", sep="\t")
        spatial = np.load(inputs / "spatial.npy")

        spots = list(RNA_data.columns)
        # align everything to the RNA spot order
        ATAC_data = ATAC_data[spots]
        cost_df = cost_df.loc[spots, spots]
        clusters = clusters.set_index("cell").loc[spots].reset_index()

        # feature_matrix: rows must equal ATAC peak order, cols equal RNA gene order
        feature_matrix = feature_matrix.loc[list(ATAC_data.index), list(RNA_data.index)]

        # ---- assemble FGOT inputs (mirrors tutorial cells 9, 16, 19) -------
        X1 = ATAC_data.T  # spots x peaks
        X2 = RNA_data.T   # spots x genes
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        peak_names = list(ATAC_data.index)
        gene_names = list(RNA_data.index)
        cell_names = spots

        ATAC_cluster = clusters.rename(columns={"index": "cell"})[["cell", "cluster"]].copy()
        RNA_cluster = ATAC_cluster.copy()

        cost = cost_df.copy()
        cost.index = cell_names
        cost.columns = cell_names

        scaler = StandardScaler()
        X1v = scaler.fit_transform(X1)
        X2v = scaler.fit_transform(X2)
        X1 = pd.DataFrame(X1v, index=cell_names, columns=peak_names)
        X2 = pd.DataFrame(X2v, index=cell_names, columns=gene_names)

        print("Dimensions: X1(ATAC) =", X1.shape, " X2(RNA) =", X2.shape,
              " feature_matrix =", feature_matrix.shape, flush=True)

        # ---- FGOT solve (tutorial cell 21-23) ------------------------------
        # pair=True -> source/dest indexed by identical spots (paired data)
        P_tensor = fgot_sparse_tensor(
            X1, X2, feature_matrix, cost,
            ATAC_cluster, RNA_cluster,
            minibatch=1, batchsize=min(args.batchsize, n1), pair=True,
            device=args.device, eps_p=args.eps_p, rho_mu=args.rho_mu, rho_nu=args.rho_nu,
        )
        P = fgot_tol(P_tensor)
        print("P shape:", P.shape, " P sum:", float(np.sum(P)), flush=True)

        # ---- align + embed (tutorial cell 25-26) ---------------------------
        X1_aligned, X2_aligned = align(X1, X2, P, mode="RNA2ATAC")
        data_aligned = np.concatenate((X2_aligned, X1_aligned), axis=0)
        adata_aligned = AnnData(np.asarray(data_aligned, dtype=np.float32))
        adata_aligned.obs["batch"] = np.array(["RNA"] * n2 + ["ATAC"] * n1)
        sc.tl.pca(adata_aligned, n_comps=min(args.n_pcs, data_aligned.shape[1] - 1,
                                             data_aligned.shape[0] - 1))

        # RNA half = the 9215 per-spot embedding for niche clustering
        rna_mask = (adata_aligned.obs["batch"] == "RNA").values
        emb = np.asarray(adata_aligned.obsm["X_pca"][rna_mask], dtype=np.float32)

        np.save(out / "fgot_emb.npy", emb)
        pd.Series(cell_names, name="obs_name").to_csv(out / "obs_names.csv", index=False)

        labels = pd.DataFrame(index=cell_names)
        labels.index.name = "obs_name"
        rna_ref = sc.read_h5ad(
            "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_rna.h5ad"
        )
        for col in ["RNA_clusters", "ATAC_clusters"]:
            labels[col] = rna_ref.obs.reindex(cell_names)[col].astype(str).values
        labels.to_csv(out / "labels_used.csv")

        manifest = {}
        man_path = inputs / "manifest.json"
        if man_path.exists():
            manifest = json.loads(man_path.read_text())

        status = {
            "method": METHOD,
            "status": "success",
            "n_spots": int(n2),
            "n_genes": int(len(gene_names)),
            "n_peaks": int(len(peak_names)),
            "n_feature_edges": int(np.isfinite(feature_matrix.values).sum()),
            "emb_shape": list(emb.shape),
            "P_sum": float(np.sum(P)),
            "P_nnz": int(np.count_nonzero(P)),
            "duration_seconds": round(time.time() - started, 2),
            "device": args.device,
            "params": {
                "batchsize": int(min(args.batchsize, n1)),
                "eps_p": args.eps_p, "rho_mu": args.rho_mu, "rho_nu": args.rho_nu,
                "n_pcs": int(min(args.n_pcs, data_aligned.shape[1] - 1)),
            },
            "input_manifest": manifest,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
        with status_path.open("w") as fh:
            json.dump(status, fh, indent=2)
        print("STATUS:", json.dumps(status, indent=2))
        print("DONE")

    except Exception as exc:
        fail = {
            "method": METHOD,
            "status": "failed",
            "message": str(exc),
            "exception_type": type(exc).__name__,
            "traceback": traceback.format_exc(),
            "duration_seconds": round(time.time() - started, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        }
        with status_path.open("w") as fh:
            json.dump(fail, fh, indent=2)
        print(fail["traceback"], file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
