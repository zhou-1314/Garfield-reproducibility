#!/usr/bin/env python
"""Launch the 8-config hippo module-ablation across GPUs (garfield_dev foundation)."""
import os, json, time, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = "/home/zhouweige/anaconda3/envs/Garfield/bin/python"
OUTDIR = os.path.join(HERE, "results", "hippo_abl")
DEVICES = [0, 1, 2, 3]
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 50
SEEDS = [int(s) for s in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["2024"])]
MAXP = int(sys.argv[3]) if len(sys.argv) > 3 else 4
THREADS = 8

CONFIGS = [
    ("baseline",          {},                                      {}),
    ("no_contrastive",    {},   {"include_instance_loss": False, "include_cluster_loss": False}),
    ("no_instance_loss",  {},                  {"include_instance_loss": False}),
    ("no_cluster_loss",   {},                  {"include_cluster_loss": False}),
    ("no_entropy_HY",     {},                  {"include_cluster_entropy": False}),
    ("no_adj_recon",      {"lambda_latent_adj_recon_loss": 0.0},   {}),
    ("no_edge_recon",     {"include_edge_recon_loss": False},      {}),
    ("no_gene_recon",     {"lambda_gene_expr_recon": 0.0},         {}),
]

os.makedirs(OUTDIR, exist_ok=True)
jobs = []
for seed in SEEDS:
    for tag, ov, fl in CONFIGS:
        out = os.path.join(OUTDIR, f"{tag}__s{seed}.csv")
        if os.path.exists(out):
            continue
        jobs.append((tag, ov, fl, seed, out))
print(f"pending hippo ablation jobs: {len(jobs)} (epochs={EPOCHS}, seeds={SEEDS})", flush=True)

running = []
qi = 0
while qi < len(jobs) or running:
    while len(running) < MAXP and qi < len(jobs):
        tag, ov, fl, seed, out = jobs[qi]
        dev = DEVICES[qi % len(DEVICES)]
        # Isolate each job to ONE physical GPU via CUDA_VISIBLE_DEVICES so it sees
        # only cuda:0 (prevents DDP/auto from grabbing all GPUs -> OOM).
        env = {**os.environ, "GF_THREADS": str(THREADS), "OMP_NUM_THREADS": str(THREADS),
               "MKL_NUM_THREADS": str(THREADS), "OPENBLAS_NUM_THREADS": str(THREADS),
               "CUDA_VISIBLE_DEVICES": str(dev)}
        cmd = [PY, os.path.join(HERE, "ablate_hippo.py"), "--tag", tag,
               "--overrides", json.dumps(ov), "--flags", json.dumps(fl),
               "--epochs", str(EPOCHS), "--seed", str(seed), "--device", "0", "--out", out]
        log = open(os.path.join(OUTDIR, f"{tag}__s{seed}.log"), "w")
        p = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        running.append((p, f"{tag}/s{seed}@gpu{dev}", time.time(), log))
        print(f"[LAUNCH] {tag}/s{seed}@gpu{dev}", flush=True)
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
print("HIPPO ABLATION ALL DONE", flush=True)
