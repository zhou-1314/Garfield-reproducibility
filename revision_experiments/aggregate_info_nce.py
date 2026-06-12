#!/usr/bin/env python
"""Aggregate the per-run InfoNCE result JSONs into info_nce_summary.csv.

Reads results/info_nce/{variant}__s{seed}.json for the two variants and the two
seeds, and writes cross-seed means / standard deviations of the fixed-N held-out
InfoNCE bound (latent primary, projector secondary, shuffled control). Keeping the
summary generation in a committed script (rather than hand-editing the CSV) makes
the reported R1.1 numbers reproducible from the raw runs.

Usage:  python aggregate_info_nce.py
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "info_nce")
OUT = os.path.join(RES, "info_nce_summary.csv")
VARIANTS = ["full", "no_contrastive"]
SEEDS = [2024, 2025]


def main():
    runs = {(v, s): json.load(open(os.path.join(RES, f"{v}__s{s}.json")))
            for v in VARIANTS for s in SEEDS}
    cols = ["variant", "I_NCE_latent_mean", "I_NCE_latent_sd",
            "I_NCE_latent_shuffled_mean", "I_NCE_projector_mean", "I_NCE_projector_sd",
            "seeds", "n_obs_train", "eval_candidate_count", "fixed_K", "temperature"]
    lines = [",".join(cols)]
    for v in VARIANTS:
        lat = [runs[(v, s)]["I_NCE_latent_mean"] for s in SEEDS]
        shuf = [runs[(v, s)]["I_NCE_latent_shuffled_mean"] for s in SEEDS]
        proj = [runs[(v, s)]["I_NCE_projector_mean"] for s in SEEDS]
        r0 = runs[(v, SEEDS[0])]
        # sample SD (ddof=1) to match the ablation summaries' convention
        row = [v,
               f"{np.mean(lat):.4f}", f"{np.std(lat, ddof=1):.4f}",
               f"{np.mean(shuf):.4f}",
               f"{np.mean(proj):.4f}", f"{np.std(proj, ddof=1):.4f}",
               '"' + ",".join(str(s) for s in SEEDS) + '"',
               str(int(r0["n_obs"])),
               str(int(r0["eval_candidate_count"])), str(int(r0["fixed_K"])),
               str(r0["temperature"])]
        lines.append(",".join(row))
    with open(OUT, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", os.path.relpath(OUT, os.path.join(HERE, "..")))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
