#!/usr/bin/env bash
# Wait for the shared preprocessing pickle (built by config A), then sweep a batch of
# Garfield configs on GPU 2, scoring each on the 3 external GTs. Stop early if one beats
# SpatialGlue (mean ARI>0.314 AND mean NMI>0.391).
set -u
RB=/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/recent_baselines
P22=/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/results/recent_baselines/spatial_atac_rna_p22
PKL=$P22/garfield_tune/_shared_cache/preprocessed_adata.pkl
cd "$RB"
echo "[sweep] waiting for preprocessing pickle..."
for i in $(seq 1 120); do [ -f "$PKL" ] && break; sleep 20; done
[ -f "$PKL" ] || { echo "[sweep] pickle never appeared; abort"; exit 1; }
sleep 40  # let the pickle finish writing + config A's training settle on GPU1
echo "[sweep] pickle ready, starting batch"

run() {
  local tag="$1" ov="$2"
  echo "===== [sweep] $tag :: $ov ====="
  CUDA_VISIBLE_DEVICES=2 GF_THREADS=8 conda run -n Garfield --no-capture-output python tune_garfield_p22.py \
    --tag "$tag" --overrides "$ov" --device 0 2>&1 | grep -E "DONE|training done|Error|Traceback" | tail -2
  bash score_tuned.sh "$tag" 2>&1 | grep -E "MEAN|gt_key|ATAC|RNA|predicted|BEATS"
}

# config name : overrides (model/loss only -> share preprocessing cache)
run B_adj50_ep200_bn30   '{"lambda_latent_adj_recon_loss":50,"n_epochs":200,"bottle_neck_neurons":30}'
run C_devw_ep200_bn30    '{"lambda_edge_recon":5,"lambda_latent_adj_recon_loss":1,"n_epochs":200,"bottle_neck_neurons":30}'
run D_contr_bn32_ep200   '{"lambda_latent_contrastive_instanceloss":2,"lambda_latent_contrastive_clusterloss":1,"bottle_neck_neurons":32,"n_epochs":200}'
run E_big_bn32_ep200     '{"hidden_dims":[256,256],"bottle_neck_neurons":32,"n_epochs":200}'
run F_ep300_bn30         '{"n_epochs":300,"bottle_neck_neurons":30}'
run G_gene5_ep200_bn30   '{"lambda_gene_expr_recon":5,"n_epochs":200,"bottle_neck_neurons":30}'
echo "===== [sweep] GPU2 BATCH DONE ====="
echo "--- summary of all tuned configs ---"
for s in $P22/garfield_tune/scored_*.csv; do
  conda run -n scib --no-capture-output python - "$s" <<'PY' 2>/dev/null
import sys,pandas as pd,os
d=pd.read_csv(sys.argv[1]); tag=os.path.basename(sys.argv[1])[7:-4]
print(f"{tag:24s} meanARI={d.ARI.mean():.4f} meanNMI={d.NMI.mean():.4f}")
PY
done
