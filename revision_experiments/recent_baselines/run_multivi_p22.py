#!/usr/bin/env python3
"""Run scVI MULTIVI on the P22 spatial ATAC-RNA mouse-brain paired data.

Adapted from
  Garfield_code/spatial-multi-modal-code/process_data_MultiVI.py
but pointed at the LOCAL P22 files (single paired RNA+ATAC slice) and writing
the joint latent as multivi_emb.npy in RNA obs_names order, plus obs_names.csv
and multivi_status.json.

Pipeline (matches the reference):
  1. read RNA + ATAC h5ad (paired, identical obs_names)
  2. raw counts into .X, light filtering, seurat_v3 HVG on each modality
  3. concat into one paired AnnData with a per-feature `modality` column
  4. scvi.data.organize_multiome_anndatas -> setup_anndata(batch_key="modality")
  5. train MULTIVI, get_latent_representation()
  6. realign the per-spot latent to the original RNA obs_names order and save

Run (GPU 1 only):
  export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
  CUDA_VISIBLE_DEVICES=1 conda run -n scVI --no-capture-output \
      python revision_experiments/recent_baselines/run_multivi_p22.py
"""

from __future__ import annotations

import json
import os
import time

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
from scipy import sparse

scvi.settings.seed = 420

# Local P22 paired data (use THESE, not the batch2 reference filenames).
RNA_PATH = "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_rna.h5ad"
ATAC_PATH = "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_atac.h5ad"

OUT_DIR = (
    "/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/"
    "results/recent_baselines/spatial_atac_rna_p22/multivi"
)

METHOD = "MultiVI"
ENV = "scVI"


def _counts_to_X(adata):
    """Put raw integer counts into .X as CSR (MULTIVI expects counts)."""
    if "counts" in adata.layers:
        adata.X = adata.layers["counts"].copy()
    if not sparse.issparse(adata.X):
        adata.X = sparse.csr_matrix(adata.X)
    adata.layers["counts"] = adata.X.copy()
    return adata


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    status_path = os.path.join(OUT_DIR, "multivi_status.json")
    started = time.time()

    status = {
        "method": METHOD,
        "status": "failed",
        "n_obs": None,
        "emb_shape": None,
        "duration_seconds": None,
        "env": ENV,
    }

    try:
        # ---- 1. read paired RNA + ATAC -------------------------------------
        adata_RNA = sc.read_h5ad(RNA_PATH)
        adata_ATAC = sc.read_h5ad(ATAC_PATH)

        # Canonical row order = RNA obs_names; ATAC must follow the same order.
        rna_obs_order = adata_RNA.obs_names.astype(str).to_numpy()
        assert adata_RNA.n_obs == adata_ATAC.n_obs, "RNA/ATAC obs count mismatch"
        adata_ATAC = adata_ATAC[adata_RNA.obs_names].copy()
        assert (adata_RNA.obs_names == adata_ATAC.obs_names).all(), "obs_names not aligned"

        adata_RNA.var_names_make_unique()
        adata_ATAC.var_names_make_unique()
        adata_RNA = _counts_to_X(adata_RNA)
        adata_ATAC = _counts_to_X(adata_ATAC)

        # ---- 2. filter + HVG (same knobs as the reference script) ----------
        sc.pp.filter_cells(adata_RNA, min_genes=10)
        sc.pp.filter_genes(adata_RNA, min_cells=20)
        sc.pp.filter_cells(adata_ATAC, min_genes=10)
        sc.pp.filter_genes(adata_ATAC, min_cells=1)

        # Filtering may drop spots per-modality; keep only spots present in BOTH
        # so the paired AnnData stays paired, then re-derive the common order.
        common = adata_RNA.obs_names.intersection(adata_ATAC.obs_names)
        # Preserve original RNA order for the kept spots.
        keep_mask = pd.Index(rna_obs_order).isin(common)
        kept_order = rna_obs_order[keep_mask]
        adata_RNA = adata_RNA[kept_order].copy()
        adata_ATAC = adata_ATAC[kept_order].copy()

        sc.pp.normalize_total(adata_RNA, target_sum=1e4)
        sc.pp.log1p(adata_RNA)
        sc.pp.highly_variable_genes(
            adata_RNA, flavor="seurat_v3", n_top_genes=3000, subset=False
        )
        adata_RNA = adata_RNA[:, adata_RNA.var.highly_variable].copy()
        adata_RNA.X = adata_RNA.layers["counts"].copy()

        sc.pp.normalize_total(adata_ATAC, target_sum=1e4)
        sc.pp.log1p(adata_ATAC)
        sc.pp.highly_variable_genes(
            adata_ATAC, flavor="seurat_v3", n_top_genes=10000, subset=False
        )
        adata_ATAC = adata_ATAC[:, adata_ATAC.var.highly_variable].copy()
        adata_ATAC.X = adata_ATAC.layers["counts"].copy()

        # ---- 3. concat into one paired multiome AnnData --------------------
        # Drop layers/obsm before concat to avoid merge conflicts on axis=1.
        rna_part = ad.AnnData(X=adata_RNA.X.copy(), obs=adata_RNA.obs.copy())
        rna_part.var_names = adata_RNA.var_names
        atac_part = ad.AnnData(X=adata_ATAC.X.copy(), obs=adata_ATAC.obs.copy())
        atac_part.var_names = adata_ATAC.var_names

        adata_paired = ad.concat([rna_part, atac_part], merge="same", axis=1)
        adata_paired.obs_names = rna_part.obs_names  # explicit paired obs order
        adata_paired.var["modality"] = (
            ["Gene Expression"] * rna_part.shape[1] + ["Peaks"] * atac_part.shape[1]
        )

        # remember the order we will align the latent back to
        paired_obs_order = adata_paired.obs_names.astype(str).to_numpy()

        # ---- 4. organize + setup + 5. train MULTIVI ------------------------
        adata_mvi = scvi.data.organize_multiome_anndatas(adata_paired)
        del adata_paired
        adata_mvi = adata_mvi[:, adata_mvi.var["modality"].argsort()].copy()

        scvi.model.MULTIVI.setup_anndata(adata_mvi, batch_key="modality")
        model = scvi.model.MULTIVI(
            adata_mvi,
            n_genes=int((adata_mvi.var["modality"] == "Gene Expression").sum()),
            n_regions=int((adata_mvi.var["modality"] == "Peaks").sum()),
        )
        model.view_anndata_setup()
        model.train()

        latent = np.asarray(model.get_latent_representation(), dtype=np.float32)

        # ---- 6. align latent rows to RNA obs_names order -------------------
        # organize_multiome_anndatas suffixes paired cells (e.g. "<bc>_paired").
        # Map each latent row back to its original spot barcode, then reorder
        # to the kept RNA obs order.
        latent_index = adata_mvi.obs_names.astype(str).to_numpy()

        def _strip_suffix(name):
            for suf in ("_paired", "_accessibility", "_expression"):
                if name.endswith(suf):
                    return name[: -len(suf)]
            return name

        stripped = np.array([_strip_suffix(n) for n in latent_index])

        # We expect exactly the paired spots, one row each.
        if not set(paired_obs_order).issubset(set(stripped)):
            missing = set(paired_obs_order) - set(stripped)
            raise RuntimeError(
                f"{len(missing)} paired spots missing from MULTIVI latent index"
            )

        order = pd.Index(stripped).get_indexer(paired_obs_order)
        if (order < 0).any():
            raise RuntimeError("Could not align all latent rows to paired obs order")
        latent = latent[order]
        emb_obs_order = paired_obs_order

        assert latent.shape[0] == len(emb_obs_order), "latent/obs row mismatch"
        assert np.isfinite(latent).all(), "non-finite values in latent"

        # ---- save ----------------------------------------------------------
        emb_path = os.path.join(OUT_DIR, "multivi_emb.npy")
        np.save(emb_path, latent.astype(np.float32))
        pd.DataFrame({"obs_name": emb_obs_order}).to_csv(
            os.path.join(OUT_DIR, "obs_names.csv"), index=False
        )

        status.update(
            {
                "status": "success",
                "n_obs": int(latent.shape[0]),
                "emb_shape": [int(latent.shape[0]), int(latent.shape[1])],
                "duration_seconds": round(time.time() - started, 2),
                "n_obs_input_rna": int(len(rna_obs_order)),
                "n_genes_hvg": int(rna_part.shape[1]),
                "n_peaks_hvg": int(atac_part.shape[1]),
            }
        )
        print(f"MULTIVI latent saved: {latent.shape} -> {emb_path}")

    except Exception as exc:  # noqa: BLE001
        status["status"] = "failed"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["duration_seconds"] = round(time.time() - started, 2)
        with open(status_path, "w") as fh:
            json.dump(status, fh, indent=2)
        raise

    with open(status_path, "w") as fh:
        json.dump(status, fh, indent=2)


if __name__ == "__main__":
    main()
