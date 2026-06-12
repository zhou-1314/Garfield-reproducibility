#!/usr/bin/env python
"""Build the ablation + sensitivity job grid and run it across GPUs.

Each job runs harness.py in a subprocess, writing one row to results/rows/<id>.csv.
Resumable: jobs whose output CSV already exists are skipped.

Usage:
  python launch.py --suite ablation --datasets brain10x --max-parallel 4 --epochs 100
  python launch.py --suite sensitivity --datasets brain10x --max-parallel 4
  python launch.py --suite all --datasets brain10x,hippo
"""
import os, sys, json, time, argparse, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/zhouweige/anaconda3/envs/Garfield/bin/python"
ROWS_DEFAULT = os.path.join(HERE, "results", "rows_dev")
DEVICES = [0, 1, 2, 3]


def ablation_jobs(dataset, seeds):
    # (tag, overrides, flags)
    base = [
        ("baseline",            {},                                      {}),
        ("no_contrastive",      {},  {"include_instance_loss": False, "include_cluster_loss": False}),
        ("no_instance_loss",    {},                  {"include_instance_loss": False}),
        ("no_cluster_loss",     {},                  {"include_cluster_loss": False}),
        ("no_entropy_HY",       {},                  {"include_cluster_entropy": False}),
        ("no_adj_recon",        {"lambda_latent_adj_recon_loss": 0.0},   {}),
        ("no_edge_recon",       {"include_edge_recon_loss": False},      {}),
        ("no_gene_recon",       {"lambda_gene_expr_recon": 0.0},         {}),
    ]
    jobs = []
    for tag, ov, fl in base:
        for s in seeds:
            jobs.append((dataset, tag, ov, fl, s))
    return jobs


def sensitivity_jobs(dataset, seeds):
    jobs = []
    # n_sam (message-passing neighbor sampling) — R2.B2
    for n in (1, 3, 5, 10, 15):
        for s in seeds:
            jobs.append((dataset, f"nsam_{n}", {"num_neighbors": n}, {}, s))
    # svd_q (ApproxSVD rank) — R2.B3
    for q in (1, 3, 5, 10, 20):
        for s in seeds:
            jobs.append((dataset, f"svdq_{q}", {"augment_type": "svd", "svd_q": q}, {}, s))
    # per-lambda sensitivity (linear-weight terms) — R3.2
    lam = {"lambda_edge_recon": 5.0, "lambda_gene_expr_recon": 1.0,
           "lambda_latent_adj_recon_loss": 1.0}
    for key, dflt in lam.items():
        for mult in (0.1, 10.0):
            short = key.replace("lambda_", "").replace("_recon", "").replace("latent_", "")
            for s in seeds:
                jobs.append((dataset, f"{short}_x{mult}", {key: dflt * mult}, {}, s))
    return jobs


def mmd_jobs(dataset, seeds):
    jobs = []
    for s in seeds:
        jobs.append((dataset, "baseline_no_mmd", {"used_mmd": False}, {}, s))
        jobs.append((dataset, "with_mmd", {"used_mmd": True}, {}, s))
    return jobs


def build(suite, datasets, seeds):
    jobs = []
    for d in datasets:
        if suite in ("ablation", "all"):
            jobs += ablation_jobs(d, seeds)
        if suite in ("sensitivity", "all"):
            jobs += sensitivity_jobs(d, seeds)
        if suite in ("mmd", "all"):
            jobs += mmd_jobs(d, seeds)
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["ablation", "sensitivity", "mmd", "all"], required=True)
    ap.add_argument("--datasets", default="brain10x")
    ap.add_argument("--seeds", default="2024,2025")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--rows-dir", default=ROWS_DEFAULT)
    ap.add_argument("--devices", default=",".join(map(str, DEVICES)),
                    help="Comma-separated physical CUDA devices exposed one per subprocess")
    ap.add_argument("--threads", type=int, default=8,
                    help="CPU threads per job; keep max_parallel*threads <= n_cores")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = a.rows_dir
    os.makedirs(rows, exist_ok=True)
    datasets = a.datasets.split(",")
    seeds = [int(x) for x in a.seeds.split(",")]
    devices = [int(x) for x in a.devices.split(",") if x.strip()]
    if not devices:
        raise ValueError("--devices must include at least one GPU id")
    jobs = build(a.suite, datasets, seeds)

    pending = []
    for (ds, tag, ov, fl, seed) in jobs:
        out = os.path.join(rows, f"{ds}__{tag}__s{seed}.csv")
        if os.path.exists(out):
            continue
        pending.append((ds, tag, ov, fl, seed, out))
    print(f"suite={a.suite} datasets={datasets} total={len(jobs)} pending={len(pending)}", flush=True)
    if a.dry_run:
        for j in pending:
            print("  ", j[0], j[1], "seed", j[4])
        return

    running = []  # (proc, label, t0)
    qi = 0
    while qi < len(pending) or running:
        while len(running) < a.max_parallel and qi < len(pending):
            ds, tag, ov, fl, seed, out = pending[qi]
            dev = devices[qi % len(devices)]
            label = f"{ds}/{tag}/s{seed}@gpu{dev}"
            cmd = [PY, os.path.join(HERE, "harness.py"),
                   "--dataset", ds, "--tag", tag,
                   "--overrides", json.dumps(ov), "--flags", json.dumps(fl),
                   "--epochs", str(a.epochs), "--seed", str(seed),
                   "--device", "0", "--out", out]
            log = open(os.path.join(rows, f"{ds}__{tag}__s{seed}.log"), "w")
            child_env = {**os.environ, "GF_THREADS": str(a.threads),
                         "OMP_NUM_THREADS": str(a.threads),
                         "MKL_NUM_THREADS": str(a.threads),
                         "OPENBLAS_NUM_THREADS": str(a.threads),
                         "NUMEXPR_NUM_THREADS": str(a.threads),
                         "CUDA_VISIBLE_DEVICES": str(dev)}
            p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=child_env)
            running.append((p, label, time.time(), log))
            print(f"[LAUNCH] {label}", flush=True)
            qi += 1
        time.sleep(5)
        still = []
        for p, label, t0, log in running:
            if p.poll() is None:
                still.append((p, label, t0, log))
            else:
                log.close()
                ok = "OK" if p.returncode == 0 else f"FAIL({p.returncode})"
                print(f"[{ok}] {label} ({time.time()-t0:.0f}s)", flush=True)
        running = still
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
