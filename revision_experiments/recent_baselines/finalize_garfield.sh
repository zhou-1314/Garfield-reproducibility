#!/usr/bin/env bash
# Finalize the benchmark with a chosen tuned Garfield config.
# Usage: bash finalize_garfield.sh <tune_tag>
# Replaces the "Garfield" row in the 3 external-GT scib CSVs with the tuned latent's
# scores, regenerates the LaTeX quant table, and copies it into the latex dir.
set -u
tag="$1"
RB=/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/recent_baselines
P22=/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/results/recent_baselines/spatial_atac_rna_p22
PGT=$P22/pseudo_gt/adata_pseudo_gt.h5ad
T=$P22/garfield_tune
cd "$RB"
emb=$T/latent_$tag.npy; obs=$T/obs_names_$tag.csv
[ -f "$emb" ] || { echo "no latent for $tag"; exit 1; }

for pair in ATAC_clusters:scib_atac_clusters RNA_clusters:scib_RNA_clusters predicted.celltype:scib_predicted_celltype; do
  k=${pair%%:*}; csv=$P22/${pair##*:}.csv
  # drop existing Garfield row
  conda run -n scib --no-capture-output python - "$csv" <<'PY' 2>/dev/null
import sys,pandas as pd
c=sys.argv[1]; d=pd.read_csv(c); d=d[d.method!="Garfield"]; d.to_csv(c,index=False)
PY
  # score tuned latent as "Garfield"
  conda run -n scib --no-capture-output python score_scib.py --emb "$emb" --obs-names "$obs" \
    --pseudo-gt "$PGT" --method Garfield --out "$csv" --gt-key "$k" 2>/dev/null | grep -E "SCORED|Error"
done

echo "=== regenerate quant table ==="
conda run -n Garfield --no-capture-output python make_quant_table.py 2>/dev/null
cp quant_table.tex /data2/zhouwg_data/project/Garfield-reproducibility/Garfield_latex/recent_quant_table.tex
echo "=== final quant table ==="
conda run -n scib --no-capture-output python - <<'PY' 2>/dev/null
import pandas as pd,glob,os
P="/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/results/recent_baselines/spatial_atac_rna_p22"
gts={"ATAC":"scib_atac_clusters.csv","RNA":"scib_RNA_clusters.csv","CT":"scib_predicted_celltype.csv"}
dfs={g:pd.read_csv(f"{P}/{f}").set_index("method") for g,f in gts.items()}
ms=["Garfield","SpatialGlue","NicheCompass","MultiVI","SpaMosaic","SpaMI","soFusion","FGOT"]
import numpy as np
rows=[{"method":m,"mARI":round(np.mean([dfs[g].loc[m,"ARI"] for g in gts]),4),
       "mNMI":round(np.mean([dfs[g].loc[m,"NMI"] for g in gts]),4)} for m in ms if all(m in dfs[g].index for g in gts)]
print(pd.DataFrame(rows).sort_values("mARI",ascending=False).to_string(index=False))
PY
echo "tag=$tag finalized"
