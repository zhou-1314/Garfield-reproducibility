"""
Run NicheCompass on the P22 spatial ATAC-RNA mouse-brain data (RNA modality).

NicheCompass models spatial transcriptomics with gene-program (GP) masks built
from prior knowledge (ligand-receptor + metabolite-sensor interactions). This is
how the paper benchmarked it, so we run the *RNA* modality of the paired dataset
with the squidpy spatial graph.

Outputs (results/recent_baselines/spatial_atac_rna_p22/nichecompass/):
  nichecompass_emb.npy       latent, float32, (9215, d), row-aligned to RNA obs_names
  obs_names.csv              column 'obs_name' in the same row order
  nichecompass_status.json   run metadata
"""

import os
import json
import time
import traceback

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import scipy.sparse as sp
import anndata as ad
import squidpy as sq

from nichecompass.models import NicheCompass
from nichecompass.utils import (
    extract_gp_dict_from_omnipath_lr_interactions,
    extract_gp_dict_from_nichenet_lrt_interactions,
    extract_gp_dict_from_mebocost_ms_interactions,
    filter_and_combine_gp_dict_gps,
    add_gps_from_gp_dict_to_adata,
)

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
RNA_PATH = "/data2/zhouwg_data/project/Garfield_tutorials/data/spatial_atac_rna_seq_mouse_brain_rna.h5ad"

GP_DATA_DIR = "/data2/zhouwg_data/project/Garfield-reproducibility/NicheCompass/data/gene_programs"
OMNIPATH_LR = os.path.join(GP_DATA_DIR, "omnipath_lr_network.csv")
NICHENET_LR = os.path.join(GP_DATA_DIR, "nichenet_lr_network_v2_mouse.csv")
NICHENET_LTM = os.path.join(GP_DATA_DIR, "nichenet_ligand_target_matrix_v2_mouse.csv")
MEBOCOST_DIR = os.path.join(GP_DATA_DIR, "metabolite_enzyme_sensor_gps")
GENE_ORTHOLOGS = "/data2/zhouwg_data/project/Garfield-reproducibility/NicheCompass/data/gene_annotations/human_mouse_gene_orthologs.csv"

OUT_DIR = "/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/results/recent_baselines/spatial_atac_rna_p22/nichecompass"
os.makedirs(OUT_DIR, exist_ok=True)

EMB_PATH = os.path.join(OUT_DIR, "nichecompass_emb.npy")
OBS_PATH = os.path.join(OUT_DIR, "obs_names.csv")
STATUS_PATH = os.path.join(OUT_DIR, "nichecompass_status.json")

# Mask / adata keys (NicheCompass defaults)
COUNTS_KEY = "counts"
ADJ_KEY = "spatial_connectivities"
GP_NAMES_KEY = "nichecompass_gp_names"
ACTIVE_GP_NAMES_KEY = "nichecompass_active_gp_names"
GP_TARGETS_MASK_KEY = "nichecompass_gp_targets"
GP_SOURCES_MASK_KEY = "nichecompass_gp_sources"
LATENT_KEY = "nichecompass_latent"

# Hyperparameters (match NicheCompass single-sample tutorial defaults)
N_NEIGHS = 6              # squidpy spatial neighbors (tutorial default for generic coords)
N_EPOCHS = 100
N_EPOCHS_ALL_GPS = 25
LR = 0.001
LAMBDA_EDGE_RECON = 500000.0
LAMBDA_GENE_EXPR_RECON = 300.0
LAMBDA_L1_MASKED = 0.0
EDGE_BATCH_SIZE = 256
SEED = 0

ENV = "NicheCompass (conda env 'NicheCompass'), nichecompass 0.2.0"


def main():
    t0 = time.time()
    status = {
        "method": "NicheCompass",
        "status": "failed",
        "n_obs": None,
        "emb_shape": None,
        "duration_seconds": None,
        "env": ENV,
        "gp_choice": None,
    }
    try:
        # ----------------------------------------------------------------- #
        # 1. Load RNA adata (keep counts), record obs_names order
        # ----------------------------------------------------------------- #
        adata = ad.read_h5ad(RNA_PATH)
        print(f"[load] RNA adata: {adata.shape}", flush=True)
        assert COUNTS_KEY in adata.layers, "RNA adata missing layers['counts']"
        assert "spatial" in adata.obsm, "RNA adata missing obsm['spatial']"

        orig_obs_names = adata.obs_names.to_numpy().copy()  # canonical row order
        status["n_obs"] = int(adata.n_obs)

        # NicheCompass reads raw counts from layers[counts_key]; use uppercase
        # gene symbols consistently with the GP dict (genes_uppercase=True).

        # ----------------------------------------------------------------- #
        # 2. Build spatial neighbor graph (squidpy) -> adata.obsp[ADJ_KEY]
        # ----------------------------------------------------------------- #
        sq.gr.spatial_neighbors(
            adata,
            coord_type="generic",
            n_neighs=N_NEIGHS,
        )
        # Symmetrize the adjacency (NicheCompass expects an undirected graph).
        conns = adata.obsp[ADJ_KEY]
        adata.obsp[ADJ_KEY] = ((conns + conns.T) > 0).astype(np.float32)
        n_edges = int(adata.obsp[ADJ_KEY].nnz)
        print(f"[graph] spatial_neighbors n_neighs={N_NEIGHS}, "
              f"nnz(symmetrized)={n_edges}", flush=True)

        # ----------------------------------------------------------------- #
        # 3. Build gene-program mask from local prior-knowledge resources
        #    (OmniPath LR + NicheNet LR-target + MEBOCOST metabolite-sensor),
        #    all shipped offline under NicheCompass/data/gene_programs.
        # ----------------------------------------------------------------- #
        # OmniPath: this installed version queries the OmniPath web API when
        # load_from_disk=False, and the locally shipped omnipath_lr_network.csv
        # is NOT compatible with the load_from_disk=True reader (it was saved
        # with index=False but is re-read with index_col=0, which would corrupt
        # the columns). So we attempt the web API and, if it is unreachable
        # offline, we simply skip OmniPath -- NicheNet + MEBOCOST (both fully
        # offline) still provide a large GP mask. This keeps the run robust.
        print("[gp] extracting OmniPath LR GPs (web API; optional) ...", flush=True)
        omnipath_used = False
        try:
            omnipath_gp_dict = extract_gp_dict_from_omnipath_lr_interactions(
                species="mouse",
                min_curation_effort=2,
                load_from_disk=False,
                save_to_disk=False,
                lr_network_file_path=OMNIPATH_LR,
                gene_orthologs_mapping_file_path=GENE_ORTHOLOGS,
                plot_gp_gene_count_distributions=False,
            )
            omnipath_used = len(omnipath_gp_dict) > 0
            print(f"[gp] OmniPath GPs: {len(omnipath_gp_dict)}", flush=True)
        except Exception as omni_err:
            omnipath_gp_dict = {}
            print(f"[gp] OmniPath skipped (offline/unreachable): "
                  f"{type(omni_err).__name__}: {omni_err}", flush=True)

        print("[gp] extracting NicheNet LR-target GPs (load_from_disk) ...", flush=True)
        # IMPORTANT: load_from_disk=True so NicheCompass reads the local CSVs
        # (lr network + 593MB ligand-target matrix) instead of downloading the
        # .rds files from Zenodo. The download path is flaky offline
        # (IncompleteRead) and must be avoided for a deterministic offline run.
        nichenet_gp_dict = extract_gp_dict_from_nichenet_lrt_interactions(
            species="mouse",
            version="v2",
            keep_target_genes_ratio=1.0,
            max_n_target_genes_per_gp=250,
            load_from_disk=True,
            save_to_disk=False,
            lr_network_file_path=NICHENET_LR,
            ligand_target_matrix_file_path=NICHENET_LTM,
            gene_orthologs_mapping_file_path=GENE_ORTHOLOGS,
            plot_gp_gene_count_distributions=False,
        )
        print(f"[gp] NicheNet GPs: {len(nichenet_gp_dict)}", flush=True)

        print("[gp] extracting MEBOCOST metabolite-sensor GPs ...", flush=True)
        mebocost_gp_dict = extract_gp_dict_from_mebocost_ms_interactions(
            species="mouse",
            dir_path=MEBOCOST_DIR,
            plot_gp_gene_count_distributions=False,
        )
        print(f"[gp] MEBOCOST GPs: {len(mebocost_gp_dict)}", flush=True)

        # Combine + filter overlapping GPs (default subset/superset filtering off,
        # combine_overlap_gps=True with 1.0 thresholds == dedup identical GPs).
        combined_gp_dict = {}
        combined_gp_dict.update(omnipath_gp_dict)
        combined_gp_dict.update(nichenet_gp_dict)
        combined_gp_dict.update(mebocost_gp_dict)
        print(f"[gp] combined (raw) GPs: {len(combined_gp_dict)}", flush=True)

        combined_gp_dict = filter_and_combine_gp_dict_gps(
            combined_gp_dict,
            gp_filter_mode="subset",
            combine_overlap_gps=True,
            overlap_thresh_source_genes=0.9,
            overlap_thresh_target_genes=0.9,
            overlap_thresh_genes=0.9,
            verbose=False,
        )
        print(f"[gp] filtered/combined GPs: {len(combined_gp_dict)}", flush=True)

        sources_used = []
        if omnipath_used:
            sources_used.append("OmniPath LR (web API)")
        sources_used.append("NicheNet LR-target v2 mouse (local CSV)")
        sources_used.append("MEBOCOST metabolite-enzyme-sensor mouse (local TSV)")
        gp_choice = (
            "Prior-knowledge gene-program mask built from "
            + " + ".join(sources_used)
            + ", combined and filtered (subset mode, overlap thresh 0.9). "
            + f"n_GPs_after_filter={len(combined_gp_dict)}. "
            + "(NicheCompass run on the RNA modality with the squidpy spatial "
            + "graph, matching the paper's benchmarking setup.)"
        )
        status["gp_choice"] = gp_choice
        print(f"[gp] gp_choice: {gp_choice}", flush=True)

        # Add GP masks to adata (targets/sources masks, gp names, gene idx)
        add_gps_from_gp_dict_to_adata(
            gp_dict=combined_gp_dict,
            adata=adata,
            genes_uppercase=True,
            gp_targets_mask_key=GP_TARGETS_MASK_KEY,
            gp_sources_mask_key=GP_SOURCES_MASK_KEY,
            gp_names_key=GP_NAMES_KEY,
            min_genes_per_gp=1,
            min_source_genes_per_gp=0,
            min_target_genes_per_gp=0,
            max_genes_per_gp=None,
            max_source_genes_per_gp=None,
            max_target_genes_per_gp=None,
            filter_genes_not_in_masks=False,
        )
        n_gps = len(adata.uns[GP_NAMES_KEY])
        print(f"[gp] GPs added to adata: {n_gps}", flush=True)
        assert n_gps > 0, "No gene programs added to adata; cannot build GP mask."

        # ----------------------------------------------------------------- #
        # 4. Initialize + train NicheCompass (RNA-only: no chrom access loss)
        # ----------------------------------------------------------------- #
        model = NicheCompass(
            adata,
            counts_key=COUNTS_KEY,
            adj_key=ADJ_KEY,
            gp_names_key=GP_NAMES_KEY,
            active_gp_names_key=ACTIVE_GP_NAMES_KEY,
            gp_targets_mask_key=GP_TARGETS_MASK_KEY,
            gp_sources_mask_key=GP_SOURCES_MASK_KEY,
            latent_key=LATENT_KEY,
            include_edge_recon_loss=True,
            include_gene_expr_recon_loss=True,
            include_chrom_access_recon_loss=False,  # RNA-only run
            gene_expr_recon_dist="nb",
            node_label_method="one-hop-norm",
            active_gp_thresh_ratio=0.01,
            n_addon_gp=100,
            conv_layer_encoder="gatv2conv",
            use_cuda_if_available=True,
            seed=SEED,
        )
        print("[model] initialized; training ...", flush=True)

        model.train(
            n_epochs=N_EPOCHS,
            n_epochs_all_gps=N_EPOCHS_ALL_GPS,
            lr=LR,
            lambda_edge_recon=LAMBDA_EDGE_RECON,
            lambda_gene_expr_recon=LAMBDA_GENE_EXPR_RECON,
            lambda_l1_masked=LAMBDA_L1_MASKED,
            edge_batch_size=EDGE_BATCH_SIZE,
            use_cuda_if_available=True,
            verbose=True,
        )
        print("[model] training complete", flush=True)

        # ----------------------------------------------------------------- #
        # 5. Latent representation
        # ----------------------------------------------------------------- #
        latent = model.get_latent_representation(
            adata=adata,
            counts_key=COUNTS_KEY,
            adj_key=ADJ_KEY,
            only_active_gps=True,
            return_mu_std=False,
            dtype=np.float64,
        )
        latent = np.asarray(latent, dtype=np.float32)
        print(f"[latent] shape={latent.shape}", flush=True)

        # ----------------------------------------------------------------- #
        # 6. Row-align to original RNA obs_names and save
        # ----------------------------------------------------------------- #
        # NicheCompass does not reorder rows, but align defensively.
        cur_obs_names = adata.obs_names.to_numpy()
        if not np.array_equal(cur_obs_names, orig_obs_names):
            order = {n: i for i, n in enumerate(cur_obs_names)}
            idx = np.array([order[n] for n in orig_obs_names])
            latent = latent[idx]
            print("[align] reordered latent to original RNA obs_names", flush=True)

        assert latent.shape[0] == len(orig_obs_names), "Row count mismatch after align"
        assert np.isfinite(latent).all(), "Non-finite values in latent embedding"

        np.save(EMB_PATH, latent)
        with open(OBS_PATH, "w") as f:
            f.write("obs_name\n")
            for n in orig_obs_names:
                f.write(f"{n}\n")

        status["status"] = "success"
        status["emb_shape"] = list(latent.shape)
        status["duration_seconds"] = round(time.time() - t0, 2)
        print(f"[save] embedding -> {EMB_PATH} shape={latent.shape}", flush=True)

    except Exception as e:
        status["status"] = "failed"
        status["duration_seconds"] = round(time.time() - t0, 2)
        status["error"] = f"{type(e).__name__}: {e}"
        status["traceback"] = traceback.format_exc()
        print("[ERROR]", status["error"], flush=True)
        traceback.print_exc()
    finally:
        with open(STATUS_PATH, "w") as f:
            json.dump(status, f, indent=2)
        print(f"[status] -> {STATUS_PATH}: {status['status']}", flush=True)

    return status["status"] == "success"


if __name__ == "__main__":
    ok = main()
    raise SystemExit(0 if ok else 1)
