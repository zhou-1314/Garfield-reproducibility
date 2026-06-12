#!/usr/bin/env python
"""Regenerate the Supplementary InfoNCE figure (Garfield_latex/Figs/FigInfoNCE.pdf)
deterministically from the saved per-run results in results/info_nce/.

Usage:  python make_infonce_figure.py
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results", "info_nce")
OUT = os.path.join(HERE, "..", "Garfield_latex", "Figs", "FigInfoNCE.pdf")
SEEDS = [2024, 2025]
COL = {"full": "#1f77b4", "no_contrastive": "#d62728"}


def main():
    R = {(v, s): json.load(open(os.path.join(RES, f"{v}__s{s}.json")))
         for v in COL for s in SEEDS}
    fig, ax = plt.subplots(1, 2, figsize=(8.2, 3.3))
    # Panel a: per-epoch training-time contrastive InfoNCE bound (projection space).
    for v in COL:
        for s in SEEDS:
            y = R[(v, s)]["traj_INCE"]
            ax[0].plot(range(1, len(y) + 1), y, color=COL[v], alpha=0.85, lw=1.6,
                       label=(v if s == SEEDS[0] else None))
    ax[0].set_xlabel("epoch")
    ax[0].set_ylabel("training InfoNCE bound (proj. space, nats)")
    ax[0].set_title("a  Bound rises during training", loc="left", fontsize=10)
    ax[0].legend(frameon=False, fontsize=8)
    # Panel b: converged held-out I_NCE (latent / projector / shuffled), per-seed dots.
    keys = ["I_NCE_latent_mean", "I_NCE_projector_mean", "I_NCE_latent_shuffled_mean"]
    x = np.arange(len(keys)); w = 0.36
    for i, v in enumerate(COL):
        means = [np.mean([R[(v, s)][k] for s in SEEDS]) for k in keys]
        ax[1].bar(x + (i - 0.5) * w, means, w, color=COL[v], alpha=0.55, label=v)
        for j, k in enumerate(keys):
            pts = [R[(v, s)][k] for s in SEEDS]
            ax[1].scatter([x[j] + (i - 0.5) * w] * len(pts), pts, color=COL[v], s=16,
                          zorder=3, edgecolor="k", lw=0.3)
    ax[1].axhline(0, color="grey", lw=0.7)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(["latent\n(non-circular)", "projector\n(diagnostic)",
                           "shuffled\n(control)"], fontsize=8)
    ax[1].set_ylabel("converged held-out I_NCE (nats)")
    ax[1].set_title("b  full vs no-contrastive", loc="left", fontsize=10)
    ax[1].legend(frameon=False, fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    plt.savefig(OUT, bbox_inches="tight")
    print("wrote", os.path.relpath(OUT, os.path.join(HERE, "..")))


if __name__ == "__main__":
    main()
