"""
Run SpatialGlue on the P22 spatial ATAC-RNA mouse-brain data.

Modality 1 = RNA, Modality 2 = ATAC (datatype 'Spatial-epigenome-transcriptome').
Saves the joint embedding output['SpatialGlue'] in RNA-obs order.

ENV (GPU 2 only):
  CUDA_VISIBLE_DEVICES=2 conda run -p /home/zhouweige/zhouwg_data/conda_env/SpatialGlue \
      --no-capture-output python run_spatialglue_p22.py

Outputs (to results/.../spatialglue/):
  spatialglue_emb.npy      (9215, d) float32, row-aligned to RNA obs_names
  obs_names.csv            one column 'obs_name' in the same row order
  spatialglue_status.json  run metadata
"""
import os

# Limit CPU threads BEFORE importing numpy/torch-backed libs.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import json
import time
import traceback

import numpy as np
import pandas as pd
import scanpy as sc
import torch

from SpatialGlue.preprocess import pca, lsi, construct_neighbor_graph, fix_seed
from SpatialGlue.SpatialGlue_pyG import Train_SpatialGlue

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
RNA_PATH = "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_rna.h5ad"
ATAC_PATH = "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_atac.h5ad"

OUT_DIR = (
    "/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/"
    "results/recent_baselines/spatial_atac_rna_p22/spatialglue"
)
EMB_PATH = os.path.join(OUT_DIR, "spatialglue_emb.npy")
OBS_PATH = os.path.join(OUT_DIR, "obs_names.csv")
STATUS_PATH = os.path.join(OUT_DIR, "spatialglue_status.json")

DATATYPE = "Spatial-epigenome-transcriptome"  # spatial ATAC + RNA
N_COMPS = 50  # per-modality reduced dim (RNA PCA / ATAC LSI)
RANDOM_SEED = 2022


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fix_seed(RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    t0 = time.time()

    # ------------------------------------------------------------------
    # Load. RNA = modality 1, ATAC = modality 2. They share obs_names.
    # ------------------------------------------------------------------
    adata_omics1 = sc.read_h5ad(RNA_PATH)   # RNA
    adata_omics2 = sc.read_h5ad(ATAC_PATH)  # ATAC

    adata_omics1.var_names_make_unique()
    adata_omics2.var_names_make_unique()

    # Record the canonical RNA obs order; everything is saved in this order.
    rna_obs_names = adata_omics1.obs_names.astype(str).copy()

    # Sanity: paired spots, identical obs order.
    assert (adata_omics1.obs_names == adata_omics2.obs_names).all(), \
        "RNA and ATAC obs_names are not aligned"

    # ------------------------------------------------------------------
    # Modality 1: RNA  (official Spatial-epigenome-transcriptome recipe)
    #   normalize_total -> log1p -> HVG -> scale -> PCA(50)
    # X currently holds raw counts (verified); use it directly.
    # ------------------------------------------------------------------
    sc.pp.highly_variable_genes(
        adata_omics1, flavor="seurat_v3", n_top_genes=3000
    )
    sc.pp.normalize_total(adata_omics1, target_sum=1e4)
    sc.pp.log1p(adata_omics1)
    sc.pp.scale(adata_omics1)

    adata_omics1_high = adata_omics1[:, adata_omics1.var["highly_variable"]]
    adata_omics1.obsm["feat"] = pca(adata_omics1_high, n_comps=N_COMPS)

    # ------------------------------------------------------------------
    # Modality 2: ATAC  (official recipe)
    #   HVG on peaks -> LSI (TF-IDF + SVD) on HVPeaks -> 'feat'
    # X currently holds raw peak counts (verified).
    # ------------------------------------------------------------------
    sc.pp.highly_variable_genes(
        adata_omics2, flavor="seurat_v3", n_top_genes=3000
    )
    lsi(adata_omics2, use_highly_variable=False, n_components=N_COMPS + 1)
    adata_omics2.obsm["feat"] = adata_omics2.obsm["X_lsi"].copy()

    # ------------------------------------------------------------------
    # Build neighbor graphs + train.
    # ------------------------------------------------------------------
    data = construct_neighbor_graph(adata_omics1, adata_omics2, datatype=DATATYPE)

    model = Train_SpatialGlue(data, datatype=DATATYPE, device=device,
                              random_seed=RANDOM_SEED)
    output = model.train()

    emb = np.asarray(output["SpatialGlue"], dtype=np.float32)

    # ------------------------------------------------------------------
    # Re-align to RNA obs order (adata_omics1 order was preserved end-to-end,
    # but reindex defensively to guarantee row alignment).
    # ------------------------------------------------------------------
    emb_df = pd.DataFrame(emb, index=adata_omics1.obs_names.astype(str))
    emb_df = emb_df.reindex(rna_obs_names)
    emb = emb_df.to_numpy(dtype=np.float32)

    assert emb.shape[0] == len(rna_obs_names), "row count mismatch after reindex"
    assert np.isfinite(emb).all(), "non-finite values in embedding"

    # ------------------------------------------------------------------
    # Save.
    # ------------------------------------------------------------------
    np.save(EMB_PATH, emb)
    pd.DataFrame({"obs_name": rna_obs_names}).to_csv(OBS_PATH, index=False)

    duration = time.time() - t0

    status = {
        "method": "SpatialGlue",
        "status": "success",
        "n_obs": int(emb.shape[0]),
        "emb_shape": list(emb.shape),
        "duration_seconds": round(duration, 2),
        "env": "conda -p /home/zhouweige/zhouwg_data/conda_env/SpatialGlue, "
               "CUDA_VISIBLE_DEVICES=2, datatype=%s" % DATATYPE,
    }
    with open(STATUS_PATH, "w") as fh:
        json.dump(status, fh, indent=2)

    # Reload validation.
    reloaded = np.load(EMB_PATH)
    print("Saved embedding:", reloaded.shape, reloaded.dtype,
          "finite:", bool(np.isfinite(reloaded).all()))
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        os.makedirs(OUT_DIR, exist_ok=True)
        status = {
            "method": "SpatialGlue",
            "status": "failed",
            "n_obs": None,
            "emb_shape": None,
            "duration_seconds": None,
            "env": "conda -p /home/zhouweige/zhouwg_data/conda_env/SpatialGlue, "
                   "CUDA_VISIBLE_DEVICES=2, datatype=%s" % DATATYPE,
            "error": traceback.format_exc(),
        }
        with open(STATUS_PATH, "w") as fh:
            json.dump(status, fh, indent=2)
        print(status["error"])
        raise
