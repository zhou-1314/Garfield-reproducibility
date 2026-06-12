#!/usr/bin/env python3
"""Run MOFA+ on the P22 spatial ATAC-RNA mouse-brain data and save the joint latent embedding.

Adapted from:
    Garfield_code/spatial-multi-modal-code/process_data_MOFA.py

That reference script targets this exact dataset but uses old /pri_exthome paths and the
"batch2" filenames. Here we point at the LOCAL P22 h5ad files and write a fixed set of
outputs (mofa_emb.npy + obs_names.csv + mofa_status.json) under the results dir.

MOFA+ (muon mu.tl.mofa) factorizes a MuData of {rna, atac} into a shared set of factors;
mdata.obsm['X_mofa'] is the per-spot joint latent embedding. Both modalities are the two
omics of the SAME 9215 spots (paired), so the embedding is a joint per-spot representation.

CPU only: gpu_mode=False, no CUDA needed.

The RNA preprocessing can drop spots (sc.pp.filter_cells), and MOFA only keeps spots present
in both views. To honour the contract "all 9215 spots in RNA-obs order", we DISABLE cell
filtering, capture the original full RNA obs_names, and re-align the MOFA output back to that
full order, raising if any spot is missing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

# Limit CPU threads BEFORE importing numpy/scipy-backed libs.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "8")

import numpy as np
import pandas as pd
import scanpy as sc
import muon as mu
from mudata import MuData
from scipy import sparse


# --- Local P22 data paths (NOT the batch2 filenames in the reference script) ---
RNA_PATH = "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_rna.h5ad"
ATAC_PATH = "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_atac.h5ad"

OUT_DIR = Path(
    "/data2/zhouwg_data/project/Garfield-reproducibility/"
    "revision_experiments/results/recent_baselines/spatial_atac_rna_p22/mofa"
)

METHOD = "MOFA+"
ENV = "Garfield_benchmark"

# Match the reference script's HVG / HVP feature-selection sizes.
N_TOP_GENES = 3000
N_TOP_PEAKS = 10000


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    status_path = OUT_DIR / "mofa_status.json"
    emb_path = OUT_DIR / "mofa_emb.npy"
    obs_path = OUT_DIR / "obs_names.csv"
    hdf5_path = OUT_DIR / "MOFA.hdf5"

    start_time = time.time()
    try:
        # --- Read RNA + ATAC (paired spots, identical obs_names) ---
        adata_RNA = sc.read_h5ad(RNA_PATH)
        adata_ATAC = sc.read_h5ad(ATAC_PATH)

        # The full RNA obs order is the contract for the output rows.
        full_obs_names = adata_RNA.obs_names.astype(str).to_numpy()
        n_obs_full = full_obs_names.shape[0]

        # Make sure ATAC is in the same spot order as RNA before MOFA.
        adata_ATAC = adata_ATAC[adata_RNA.obs_names].copy()

        adata_RNA.var_names_make_unique()
        adata_ATAC.var_names_make_unique()

        # Use raw counts as the starting point (layers['counts'] exists for both).
        if "counts" in adata_RNA.layers:
            adata_RNA.X = adata_RNA.layers["counts"].copy()
        if "counts" in adata_ATAC.layers:
            adata_ATAC.X = adata_ATAC.layers["counts"].copy()

        # Gene/peak filtering is harmless and follows the reference, but we must NOT
        # drop any spot (would break the 9215-spots-in-RNA-order contract).
        sc.pp.filter_genes(adata_RNA, min_cells=20)
        sc.pp.filter_genes(adata_ATAC, min_cells=1)

        if not sparse.issparse(adata_ATAC.X):
            adata_ATAC.X = sparse.csr_matrix(adata_ATAC.X)
        if not sparse.issparse(adata_RNA.X):
            adata_RNA.X = sparse.csr_matrix(adata_RNA.X)

        # RNA: normalize + log + HVG (seurat_v3 on counts uses the raw, fine here).
        sc.pp.normalize_total(adata_RNA, target_sum=1e4)
        sc.pp.log1p(adata_RNA)
        sc.pp.highly_variable_genes(
            adata_RNA, flavor="seurat_v3", n_top_genes=N_TOP_GENES, subset=False
        )
        adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable].copy()

        # ATAC: normalize + log + HVP.
        sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
        sc.pp.log1p(adata_ATAC)
        sc.pp.highly_variable_genes(
            adata_ATAC, flavor="seurat_v3", n_top_genes=N_TOP_PEAKS, subset=False
        )
        adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable].copy()

        # Build the MuData and run MOFA+ (CPU).
        mdata = MuData({"rna": adata_RNA, "atac": adata_ATAC})

        mu.tl.mofa(
            mdata,
            gpu_mode=False,
            outfile=str(hdf5_path),
            use_var=None,
        )

        # mdata.obsm['X_mofa'] is the per-spot joint latent embedding.
        x_mofa = np.asarray(mdata.obsm["X_mofa"], dtype=np.float32)
        mofa_obs_names = mdata.obs_names.astype(str).to_numpy()

        # Re-align embedding rows to the FULL RNA obs order (all 9215 spots).
        if not np.array_equal(mofa_obs_names, full_obs_names):
            order = pd.Index(mofa_obs_names).get_indexer(full_obs_names)
            if (order < 0).any():
                missing = int((order < 0).sum())
                raise RuntimeError(
                    f"MOFA embedding is missing {missing} of {n_obs_full} RNA spots; "
                    "cannot row-align to the full RNA obs order."
                )
            x_mofa = x_mofa[order]

        if x_mofa.shape[0] != n_obs_full:
            raise RuntimeError(
                f"Embedding has {x_mofa.shape[0]} rows but {n_obs_full} spots expected."
            )
        if not np.isfinite(x_mofa).all():
            raise RuntimeError("MOFA embedding contains non-finite values.")

        np.save(emb_path, x_mofa)
        pd.DataFrame({"obs_name": full_obs_names}).to_csv(obs_path, index=False)

        duration = time.time() - start_time
        status = {
            "method": METHOD,
            "status": "success",
            "n_obs": int(x_mofa.shape[0]),
            "emb_shape": [int(x_mofa.shape[0]), int(x_mofa.shape[1])],
            "duration_seconds": float(duration),
            "env": ENV,
        }
        with open(status_path, "w") as fh:
            json.dump(status, fh, indent=2)

        print(f"MOFA+ done: emb shape={x_mofa.shape}, time={duration:.1f}s")
        print(f"Saved: {emb_path}")
        print(f"Saved: {obs_path}")
        print(f"Saved: {status_path}")

    except Exception as exc:
        duration = time.time() - start_time
        status = {
            "method": METHOD,
            "status": "failed",
            "n_obs": None,
            "emb_shape": None,
            "duration_seconds": float(duration),
            "env": ENV,
            "error": f"{type(exc).__name__}: {exc}",
        }
        with open(status_path, "w") as fh:
            json.dump(status, fh, indent=2)
        raise


if __name__ == "__main__":
    main()
