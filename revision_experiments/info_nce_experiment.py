#!/usr/bin/env python
"""Direct InfoNCE mutual-information lower bound between the two augmented views.

Measures I_NCE(z1; z2) = log(K) - L_ins on Slide-seqV2 hippocampus for the full
model versus the contrastive-disabled model, to validate the contrastive design's
information-theoretic justification on the quantity the theory actually concerns
(view-view MI), rather than on downstream label MI.

Key design choices (locked before running):
  * variants: "full" and "no_contrastive" (include_instance_loss=False,
    include_cluster_loss=False); identical architecture/config otherwise.
  * estimator: InfoNCE lower bound I_NCE = log(K) - L_ins, scored on a FIXED
    predeclared candidate count -- the node_batch_size seed anchors per validation
    batch, giving K = 2*node_batch_size - 1 (=511) for every scored batch; the
    incomplete final batch is dropped. Using a single fixed K (rather than the
    per-batch subgraph size) makes the averaged bound a clean fixed-N estimate; the
    constant still cancels in the full-vs-no_contrastive comparison and the shuffled
    control.
  * measurement space: the ENCODER latents z1, z2 (pre-projection). The
    instance_projector receives gradients only from the instance loss, so it is
    untrained under no_contrastive; measuring in projector space would confound
    "untrained projector" with "lower latent MI". Latent space is non-circular.
    Projector-space I_NCE is also reported as a clearly-labeled secondary.
  * "converged" value: taken from the early-stopped reload_best_model checkpoint
    (the same selection rule used for every other Garfield result); the full
    per-epoch L_ins trajectory is read from trainer.epoch_logs.
  * fixed evaluation: the dropout augmentation is made reproducible by seeding the
    RNG before each forward; the same fixed eval seed is used for both variants.
  * shuffled-positive-pair control: the positive pairing is permuted before
    scoring; a well-behaved bound collapses toward / below the no_contrastive level.

Usage (one process per GPU, matching the single-GPU protocol):
  CUDA_VISIBLE_DEVICES=0 python info_nce_experiment.py --variant full --seed 2024 \
      --device 0 --epochs 50 --out results/info_nce/full__s2024.json
"""
import os, sys, json, time, argparse, warnings, hashlib
warnings.simplefilter("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
os.environ["GARFIELD_USE_OPTIMIZED_GRAPH"] = "1"
_thr = os.environ.get("GF_THREADS", "8")
for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(v, _thr)

REPO = "/data2/zhouwg_data/project/Garfield-reproducibility"
sys.path.insert(0, os.path.join(REPO, "Garfield-garfield_dev"))  # garfield_dev (v1.0.1)
sys.path.insert(0, os.path.join(REPO, "revision_experiments"))

import numpy as np
import torch
import torch.nn.functional as F
import anndata as ad
import Garfield as gf
from Garfield.model import Garfield
from Garfield.modules.loss import compute_contrastive_instanceloss

# Reuse the validated hippocampus config + downstream evaluator.
from ablate_hippo import faithful_config, evaluate

EVAL_SEED = 12345  # fixes the (stochastic) dropout augmentation at measurement time

# --- per-epoch L_ins trajectory recorder ---
# The legacy trainer only stores global/optim loss in epoch_logs, not the per-term
# instance loss. We therefore record every L_ins call during training, tagged by the
# current epoch. compute_contrastive_instanceloss is resolved in the GNNModelVAE module
# namespace (it is imported there), so we patch that reference; print_progress fires
# exactly once per epoch (monitor=True), which we use as the epoch counter.
_REC = {"epoch": 0, "rows": [], "active": False, "n_anchor": None}


def install_trajectory_recorder():
    import importlib
    # importlib returns the module objects reliably (the package __init__ re-exports
    # the GNNModelVAE *class* under the same dotted name, which shadows plain import).
    _vae = importlib.import_module("Garfield.modules.GNNModelVAE")
    _trn = importlib.import_module("Garfield.trainer.trainer")
    _orig_ince = _vae.compute_contrastive_instanceloss

    def _rec_ince(z_i, z_j, temperature):
        out = _orig_ince(z_i, z_j, temperature)  # real (full-batch) training loss
        if _REC["active"]:
            # Record the trajectory L_ins on the SAME fixed-anchor basis as the
            # held-out scoring: recompute over the first n_anchor seed rows so the
            # per-epoch bound uses K = 2*n_anchor - 1 (a valid lower bound), rather
            # than the full neighbour-expanded batch (whose larger, variable K would
            # make log(2*n_anchor-1) - L_ins a spuriously negative, inconsistent
            # quantity).
            na = _REC["n_anchor"]
            if na is not None and z_i.size(0) >= na:
                l_seed = float(_orig_ince(z_i[:na], z_j[:na], temperature).item())
                _REC["rows"].append((_REC["epoch"], l_seed))
        return out

    _vae.compute_contrastive_instanceloss = _rec_ince

    _orig_pp = _trn.print_progress

    def _rec_pp(*args, **kwargs):
        _REC["epoch"] += 1  # bump after each completed epoch
        return _orig_pp(*args, **kwargs)

    _trn.print_progress = _rec_pp


def epoch_trajectory_Lins():
    """Mean L_ins per epoch over all recorded (train+val) batches of that epoch."""
    by_epoch = {}
    for ep, v in _REC["rows"]:
        by_epoch.setdefault(ep, []).append(v)
    return [float(np.mean(by_epoch[e])) for e in sorted(by_epoch)]


def infonce_bound(z_i, z_j, temperature):
    """InfoNCE MI lower bound in nats for L2-normalised view embeddings.

    Returns (I_NCE, L_ins, K) where K = 2*B - 1 contrasted candidates per anchor.
    """
    zi = F.normalize(z_i, dim=1)
    zj = F.normalize(z_j, dim=1)
    l_ins = compute_contrastive_instanceloss(zi, zj, temperature).item()
    B = z_i.size(0)
    K = 2 * B - 1
    return float(np.log(K) - l_ins), float(l_ins), int(K)


@torch.no_grad()
def measure_views(model, temperature, device, n_anchor):
    """Score I_NCE on the fixed validation loader with seeded dropout augmentation.

    The bound is scored on a FIXED number of seed/anchor nodes per validation batch:
    the first ``n_anchor`` rows of each batch are the mini-batch seed nodes (PyG
    NeighbourLoader convention; the model uses the same ``[:node_batch.batch_size]``
    slice elsewhere), and their encoder latents are computed with full neighbour
    message passing. Slicing to a constant ``n_anchor`` fixes the candidate count at
    K = 2*n_anchor - 1 for every scored batch, so the averaged InfoNCE lower bound
    uses a single predeclared N rather than the per-batch subgraph size. Validation
    batches with fewer than ``n_anchor`` seed nodes (the incomplete final batch) are
    dropped. The same anchor slice is applied to the latent, shuffled-control, and
    projector scores and to the saved view embeddings.

    The model is put in train() mode so the dropout augmentation is active, but
    seeded for reproducibility and wrapped in no_grad so no parameters change.
    """
    net = model.model
    was_training = net.training
    net.train()  # activate dropout augmentation; seeded below for determinism
    augment_type = "dropout"
    lat, lat_shuf, proj = [], [], []
    z1_all, z2_all = [], []
    scored, skipped, fixed_K = 0, 0, None
    loader = model.trainer.node_val_loader
    if loader is None:
        loader = model.trainer.node_train_loader
    for bi, node_batch in enumerate(loader):
        seed_n = int(getattr(node_batch, "batch_size", n_anchor))
        if seed_n < n_anchor:           # incomplete final batch -> drop
            skipped += 1
            continue
        torch.manual_seed(EVAL_SEED + bi)          # fixed eval augmentation
        np.random.seed(EVAL_SEED + bi)
        node_batch = node_batch.to(device)
        out = net(node_batch, "omics", augment_type)
        z1 = out["z1_latent"][:n_anchor]
        z2 = out["z2_latent"][:n_anchor]
        z1_all.append(z1.detach().cpu().numpy())
        z2_all.append(z2.detach().cpu().numpy())
        i_lat, _, fixed_K = infonce_bound(z1, z2, temperature)
        # shuffled positive pairing: permute view-2 rows -> destroys correspondence
        perm = torch.randperm(z2.size(0), device=z2.device)
        i_shuf, _, _ = infonce_bound(z1, z2[perm], temperature)
        # projector-space (secondary; favours full trivially since it trains that space)
        i_proj, _, _ = infonce_bound(out["z_1"][:n_anchor], out["z_2"][:n_anchor], temperature)
        lat.append(i_lat); lat_shuf.append(i_shuf); proj.append(i_proj)
        scored += 1
    if was_training:
        net.train()
    else:
        net.eval()
    return dict(latent=lat, latent_shuffled=lat_shuf, projector=proj,
                scored_batches=scored, skipped_incomplete_batches=skipped,
                eval_candidate_count=int(n_anchor), fixed_K=fixed_K,
                z1=np.concatenate(z1_all) if z1_all else None,
                z2=np.concatenate(z2_all) if z2_all else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["full", "no_contrastive"])
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=8)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    t0 = time.time()

    adata = ad.read_h5ad(os.path.join(REPO, "benchmark_unimodal/adata_with_niches.h5ad"))
    adata.X = adata.layers["counts"].copy()
    niches_gt = adata.obs["niches"].astype(str).values

    gf.settings.set_workdir(f"/tmp/infonce_{a.variant}_s{a.seed}")
    cfg = faithful_config(a.n_neighbors)
    cfg["adata_list"] = adata
    cfg["n_epochs"] = a.epochs
    cfg["seed"] = a.seed
    cfg["device_id"] = a.device
    temperature = float(cfg["lambda_latent_contrastive_instanceloss"])  # 1.0
    dict_config = gf.settings.set_gf_params(cfg)

    # Fixed scoring basis (predeclared): the node_batch_size seed anchors per batch,
    # K = 2*n_anchor - 1. Set before training so the trajectory recorder scores L_ins
    # on the same fixed-anchor basis as the converged held-out value.
    n_anchor = int(cfg.get("node_batch_size", 256))
    _REC["n_anchor"] = n_anchor

    install_trajectory_recorder()
    model = Garfield(dict_config)
    if a.variant == "no_contrastive":
        model.model.include_instance_loss = False
        model.model.include_cluster_loss = False
    _REC["active"] = True
    model.train()
    _REC["active"] = False
    device = next(model.model.parameters()).device

    # --- per-epoch L_ins trajectory (recorded via patched instance-loss calls) ---
    # Same fixed candidate count as the held-out scoring below.
    assert int(getattr(model.trainer, "node_batch_size_", n_anchor)) == n_anchor, \
        "node_batch_size changed at train time; fixed-anchor scoring basis inconsistent"
    traj_K = 2 * n_anchor - 1
    def to_ince(curve):
        return [float(np.log(traj_K) - v) for v in curve] if curve else []
    traj_lins = epoch_trajectory_Lins()

    # --- converged value on the fixed eval loader (best/reloaded checkpoint) ---
    # Scored on a FIXED predeclared candidate count (n_anchor seed nodes per batch,
    # fixed K = 2*n_anchor - 1); the incomplete final validation batch is dropped.
    m = measure_views(model, temperature, device, n_anchor)

    # --- downstream sanity (compared to known hippo baseline ~0.220/0.804/0.451) ---
    Z = np.asarray(model.adata.obsm["garfield_latent"])
    spatial = np.asarray(model.adata.obsm["spatial"])
    gt = adata.obs["niches"].reindex(model.adata.obs_names).astype(str).values
    metrics, _ = evaluate(Z, spatial, gt, res=0.5)

    result = dict(
        variant=a.variant, seed=a.seed, epochs=a.epochs, n_obs=int(model.adata.n_obs),
        temperature=temperature, eval_seed=EVAL_SEED,
        # fixed-N held-out scoring protocol (predeclared)
        eval_candidate_count=m["eval_candidate_count"], fixed_K=m["fixed_K"],
        scored_batches=m["scored_batches"],
        skipped_incomplete_batches=m["skipped_incomplete_batches"],
        # PRIMARY: latent-space view-view InfoNCE bound (mean over scored batches)
        I_NCE_latent_mean=float(np.mean(m["latent"])) if m["latent"] else None,
        I_NCE_latent_per_batch=m["latent"],
        # control: shuffled positive pairing (should collapse toward/below baseline)
        I_NCE_latent_shuffled_mean=float(np.mean(m["latent_shuffled"])) if m["latent_shuffled"] else None,
        # SECONDARY: projector-space (favours full trivially; reported transparently)
        I_NCE_projector_mean=float(np.mean(m["projector"])) if m["projector"] else None,
        # trajectory (per-epoch L_ins and derived I_NCE; same fixed K as held-out scoring)
        traj_K=traj_K, n_epochs_recorded=len(traj_lins),
        traj_Lins=traj_lins, traj_INCE=to_ince(traj_lins),
        # downstream sanity
        downstream=metrics,
        train_sec=round(time.time() - t0, 1),
    )
    base = a.out[:-5] if a.out.endswith(".json") else a.out
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    # sidecar 1: resolved config (drop the AnnData object)
    cfg_save = {k: v for k, v in cfg.items() if k != "adata_list"}
    cfg_path = base + "_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg_save, f, indent=2, default=str)
    # sidecar 2: per-epoch L_ins / I_NCE trajectory
    log_path = base + "_epoch_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,L_ins,I_NCE\n")
        for ep, (li, ii) in enumerate(zip(traj_lins, to_ince(traj_lins)), 1):
            f.write(f"{ep},{li:.6f},{ii:.6f}\n")
    # sidecar 3: the fixed-anchor view embeddings actually scored for the bound.
    # The .npz binary is git-ignored in this text repo; the committed, reproducible
    # record is the meta sidecar (shapes + checksums + node-selection rule) together
    # with the per-batch scores stored in the result JSON.
    z1, z2 = m.get("z1"), m.get("z2")
    npz_path, meta_path = base + "_views.npz", base + "_views_meta.json"
    artifacts = dict(
        config=dict(file=os.path.basename(cfg_path), sha256=_sha256(cfg_path)),
        epoch_log=dict(file=os.path.basename(log_path), sha256=_sha256(log_path)),
    )
    if z1 is not None:
        np.savez_compressed(npz_path, z1=z1, z2=z2)
        meta = dict(
            description="fixed-anchor view embeddings scored for I_NCE(z1;z2)",
            node_selection="first eval_candidate_count seed nodes of each scored "
                           "validation batch, in loader order; incomplete final "
                           "batch dropped",
            eval_candidate_count=m["eval_candidate_count"], fixed_K=m["fixed_K"],
            scored_batches=m["scored_batches"],
            skipped_incomplete_batches=m["skipped_incomplete_batches"],
            z1_shape=list(z1.shape), z2_shape=list(z2.shape),
            z1_abs_sum=float(np.abs(z1).sum()), z2_abs_sum=float(np.abs(z2).sum()),
            npz=os.path.basename(npz_path), npz_sha256=_sha256(npz_path),
        )
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        artifacts["views_npz"] = dict(file=os.path.basename(npz_path),
                                      sha256=meta["npz_sha256"], git_tracked=False)
        artifacts["views_meta"] = dict(file=os.path.basename(meta_path),
                                       sha256=_sha256(meta_path), git_tracked=True)
    result["artifacts"] = artifacts
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[DONE] {a.variant} s{a.seed}: "
          f"I_NCE_latent={result['I_NCE_latent_mean']:.4f} "
          f"shuffled={result['I_NCE_latent_shuffled_mean']:.4f} "
          f"proj={result['I_NCE_projector_mean']:.4f} | "
          f"downstream ASW/coh/ARI={metrics['ASW_cluster']:.3f}/"
          f"{metrics['spatial_coherence']:.3f}/{metrics['niches_ARI_bestres']:.3f} "
          f"({result['train_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
