#!/usr/bin/env python
"""Aggregate recent-baseline status and metric files."""

from __future__ import print_function

import argparse
import json
from pathlib import Path

import pandas as pd

from common import DEFAULT_RESULTS_DIR, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR))
    return parser.parse_args()


def load_json(path):
    with open(path) as fh:
        return json.load(fh)


def main():
    args = parse_args()
    results_dir = ensure_dir(args.results_dir)
    status_rows = []
    for path in sorted(results_dir.glob("*/*_status.json")):
        row = load_json(path)
        row["status_file"] = str(path)
        status_rows.append(row)
    status_df = pd.DataFrame(status_rows)
    if not status_df.empty:
        status_df.to_csv(results_dir / "method_status.csv", index=False)

    metric_rows = []
    for path in sorted(results_dir.glob("*/*_metrics_best.csv")):
        df = pd.read_csv(path)
        df["metrics_file"] = str(path)
        metric_rows.append(df)
    if metric_rows:
        metrics_df = pd.concat(metric_rows, ignore_index=True)
        metrics_df.to_csv(results_dir / "recent_baselines_metrics_best.csv", index=False)
        print(metrics_df.to_string(index=False))
    else:
        print("No metric files found under", results_dir)

    if not status_df.empty:
        print(status_df[["method", "status", "message"]].to_string(index=False))


if __name__ == "__main__":
    main()
