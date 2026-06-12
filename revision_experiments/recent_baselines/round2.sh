#!/usr/bin/env bash
set -u
gpu="$1"; cfgfile="$2"
RB=/data2/zhouwg_data/project/Garfield-reproducibility/revision_experiments/recent_baselines
cd "$RB"
while IFS='|' read -r tag ov; do
  [ -z "$tag" ] && continue
  echo "===== [g$gpu] $tag :: $ov ====="
  CUDA_VISIBLE_DEVICES=$gpu GF_THREADS=8 conda run -n Garfield --no-capture-output python tune_garfield_p22.py \
    --tag "$tag" --overrides "$ov" --device 0 2>&1 | grep -E "DONE|training done|Error|Traceback" | tail -1
  bash score_tuned.sh "$tag" 2>&1 | grep -E "MEAN|BEATS"
done < "$cfgfile"
echo "[g$gpu] BATCH DONE"
