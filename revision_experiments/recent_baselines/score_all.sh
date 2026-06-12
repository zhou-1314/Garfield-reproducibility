#!/usr/bin/env bash
# Score every method's saved embedding against the P22 pseudo-GT (niche_type_sub)
# using BOTH the scib protocol (comparable to the published best-resolution table)
# and the spatial-metric scorer (CHAOS/PAS/Moran/coherence). Reuses published numbers
# for SpatialGlue/MOFA/NicheCompass/MultiVI separately (not here).
set -u
RB=/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/recent_baselines
P22=/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/results/recent_baselines/spatial_atac_rna_p22
PGT=$P22/pseudo_gt/adata_pseudo_gt.h5ad
SCIB_OUT=$P22/scib_combined.csv
SPAT_OUT=$P22/spatial_combined.csv
cd "$RB" || exit 1

if [ ! -f "$PGT" ]; then echo "PSEUDO-GT not found: $PGT"; exit 1; fi
rm -f "$SCIB_OUT" "$SPAT_OUT"

# obs_names for the released self-score (pseudo-GT adata order)
conda run -n Garfield --no-capture-output python - "$PGT" "$P22/pseudo_gt/obs_names.csv" <<'PY' 2>/dev/null
import sys, scanpy as sc, pandas as pd
a=sc.read_h5ad(sys.argv[1]); pd.DataFrame({"obs_name":a.obs_names.astype(str)}).to_csv(sys.argv[2],index=False)
print("pseudo-GT obs_names written", a.n_obs, "| obs cols:", [c for c in a.obs.columns if 'clust' in c.lower() or 'celltype' in c.lower() or 'niche' in c.lower()])
PY

# method  emb_path  obs_names_path
METHODS=(
  "Garfield_released|$P22/pseudo_gt/garfield_latent_released.npy|$P22/pseudo_gt/obs_names.csv"
  "Garfield|$P22/garfield_dev_pred/garfield_latent_dev.npy|$P22/garfield_dev_pred/obs_names.csv"
  "soFusion|$P22/sofusion/emb_pca.npy|$P22/sofusion/obs_names.csv"
  "SpaMI|$P22/spami_full/combine_emb.npy|$P22/spami_full/obs_names.csv"
  "SpaMosaic|$P22/spamosaic/spamosaic_emb.npy|$P22/spamosaic/obs_names.csv"
  "FGOT|$P22/fgot_out/fgot_emb.npy|$P22/fgot_out/obs_names.csv"
)

for entry in "${METHODS[@]}"; do
  IFS='|' read -r m emb obs <<< "$entry"
  if [ ! -f "$emb" ]; then echo "SKIP $m (no emb: $emb)"; continue; fi
  if [ ! -f "$obs" ]; then echo "SKIP $m (no obs: $obs)"; continue; fi
  echo "=== scoring $m ==="
  conda run -n scib --no-capture-output python score_scib.py \
      --emb "$emb" --obs-names "$obs" --pseudo-gt "$PGT" --method "$m" --out "$SCIB_OUT" \
      2>/dev/null | grep -E "SCORED|Error|Trace"
  conda run -n Garfield --no-capture-output python score_vs_pseudo_gt.py \
      --emb "$emb" --obs-names "$obs" --pseudo-gt "$PGT" --method "$m" --out "$SPAT_OUT" \
      2>/dev/null | grep -E "method|Error|Trace" || true
done

echo; echo "===== SCIB (niche_type_sub, best-resolution protocol) ====="
column -t -s, "$SCIB_OUT" 2>/dev/null || cat "$SCIB_OUT"
echo; echo "===== SPATIAL metrics ====="
column -t -s, "$SPAT_OUT" 2>/dev/null || cat "$SPAT_OUT"
