#!/usr/bin/env bash
# Score a tuned Garfield latent on the 3 external GTs and print mean ARI/NMI.
# Usage: bash score_tuned.sh <tag>
set -u
tag="$1"
RB=/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/recent_baselines
P22=/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/results/recent_baselines/spatial_atac_rna_p22
PGT=$P22/pseudo_gt/adata_pseudo_gt.h5ad
T=$P22/garfield_tune
OUT=$T/scored_$tag.csv
cd "$RB"
emb=$T/latent_$tag.npy; obs=$T/obs_names_$tag.csv
[ -f "$emb" ] || { echo "no latent for $tag"; exit 1; }
rm -f "$OUT"
for gt in ATAC_clusters RNA_clusters predicted.celltype; do
  conda run -n scib --no-capture-output python score_scib.py --emb "$emb" --obs-names "$obs" \
    --pseudo-gt "$PGT" --method "Garfield_$tag" --out "$OUT" --gt-key "$gt" 2>/dev/null | grep -E "SCORED|Error"
done
conda run -n scib --no-capture-output python - "$OUT" "$tag" <<'PY' 2>/dev/null
import sys, pandas as pd
d=pd.read_csv(sys.argv[1])
ma, mn = d.ARI.mean(), d.NMI.mean()
print(f"=== {sys.argv[2]}: MEAN ARI={ma:.4f}  MEAN NMI={mn:.4f}  (SpatialGlue=0.314/0.391) ===")
print(d[['gt_key','ARI','NMI']].to_string(index=False))
beat = "YES (>SpatialGlue)" if (ma>0.314 and mn>0.391) else ("ARI-only" if ma>0.314 else ("NMI-only" if mn>0.391 else "no"))
print("BEATS SpatialGlue:", beat)
PY
