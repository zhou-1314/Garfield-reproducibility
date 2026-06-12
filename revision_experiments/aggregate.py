#!/usr/bin/env python
"""Aggregate per-job CSV rows into summary tables + print pivots."""
import os, glob, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(HERE, "results", "rows_dev")
OUT = os.path.join(HERE, "results")


def main():
    rows = sys.argv[1] if len(sys.argv) > 1 else ROWS
    files = sorted(glob.glob(os.path.join(rows, "*.csv")))
    if not files:
        print("no row CSVs yet")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    out_name = "all_rows_dev.csv" if rows.endswith("rows_dev") else "all_rows.csv"
    df.to_csv(os.path.join(OUT, out_name), index=False)

    metrics = [m for m in [
        "ARI", "NMI", "AMI", "HOM", "ASW", "ASW_batch", "batch_knn_entropy",
        "train_sec", "peak_gpu_gb"
    ] if m in df.columns]
    for ds, g in df.groupby("dataset"):
        print(f"\n================= {ds}  (n_runs={len(g)}) =================")
        piv = (g.groupby("tag")[metrics]
                 .mean().round(3).sort_index())
        cnt = g.groupby("tag").size().rename("n")
        piv = piv.join(cnt)
        with pd.option_context("display.width", 200, "display.max_rows", 200):
            print(piv.to_string())
    df.to_csv(os.path.join(OUT, out_name), index=False)
    print(f"\n[saved] {os.path.join(OUT, out_name)}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
