#!/usr/bin/env python
"""Direct InfoNCE mutual-information lower bound between the two augmented views.

Measures I_NCE(z1; z2) = log(K) - L_ins on Slide-seqV2 hippocampus for the full
model versus the contrastive-disabled model, to validate the contrastive design's
information-theoretic justification on the quantity the theory actually concerns
(view-view MI), rather than on downstream label MI.

This file was revised (round-3) to address the reviewer's measurement-validity
concerns. Each numbered concern is handled by an explicit, self-checking step:

  (1) TEMPERATURE IS VERIFIED, NOT ASSUMED. In Garfield's loss the scalar passed to
      ``compute_contrastive_instanceloss(z_1, z_2, temperature)`` is the config field
      ``lambda_latent_contrastive_instanceloss`` (modules/GNNModelVAE.py); that term is
      then added to the optimisation loss with coefficient 1 (it is NOT used as a
      linear loss weight), so the field *is* the InfoNCE temperature tau (=1.0). The
      naming is a legacy artefact of the original code. We do not merely re-read the
      config key: a recorder hook captures the temperature actually passed to the loss
      during training, and ``main`` asserts that the scoring temperature equals that
      training-time temperature (``temperature_verified`` in the output).

  (2) EVALUATION DOES NOT MODIFY MODEL STATE. Scoring runs in ``net.eval()`` (NOT
      ``train()``): BatchNorm running statistics are frozen, GAT attention-dropout is
      disabled, and the VAE ``reparameterize`` returns the mean deterministically. The
      two views are still genuinely augmented because Garfield's view augmentation
      (``drop_feature`` + ``dropout_adj``) is driven by the ``augment_type`` argument
      and PyG/functional defaults, not by ``module.training``; we assert the two views
      are not identical. The full ``state_dict`` is snapshotted before scoring and
      asserted byte-identical afterwards (``model_state_unchanged``), and scoring is
      wrapped in ``torch.no_grad``.

  (3) NO SILENT FALLBACK TO THE TRAINING LOADER. If no held-out validation split
      exists the run raises; it never scores on training data.

  (4) THE BEST CHECKPOINT IS VERIFIED RELOADED. We assert ``reload_best_model`` is
      enabled, that a best checkpoint was saved, and that the live weights equal
      ``trainer.best_model_state_dict`` (the early-stopped epoch is recorded as
      ``best_epoch``).

  (5) DETERMINISTIC FIXED VALIDATION BATCHES. The legacy trainer builds its val loader
      with ``shuffle=True``; we therefore do NOT reuse it for scoring. Instead we build
      a dedicated ``shuffle=False`` NeighbourLoader over the *same* held-out
      ``val_mask`` (same neighbour fan-out as training), materialise all batches once
      under a fixed seed, and record the global node ids of every scored anchor.
      ``main`` asserts those anchors lie in ``val_mask`` and are disjoint from
      ``train_mask`` (genuinely held out), and the ids are saved for inspection.

Other locked design choices (unchanged from the pre-registered protocol):
  * variants: "full" and "no_contrastive" (include_instance_loss=False,
    include_cluster_loss=False); identical architecture/config otherwise.
  * estimator: InfoNCE lower bound I_NCE = log(K) - L_ins with a FIXED predeclared
    candidate count. Each scored batch contributes its first ``n_anchor`` seed nodes,
    giving K = 2*n_anchor - 1 (=511); the incomplete final batch is dropped. The single
    fixed K makes the averaged bound a clean fixed-N estimate, and the constant cancels
    in the full-vs-no_contrastive comparison and the shuffled control.
  * measurement space: the ENCODER latents z1, z2 (pre-projection) are PRIMARY. The
    instance_projector is trained only by the instance loss, so it is untrained under
    no_contrastive; projector-space I_NCE is reported as a clearly-labelled secondary
    optimisation diagnostic that is partly definitional.
  * shuffled-positive-pair control: the positive pairing is permuted before scoring; a
    well-behaved bound collapses toward / below the no_contrastive level.

Usage (one process per GPU, matching the single-GPU protocol):
  CUDA_VISIBLE_DEVICES=0 python info_nce_experiment.py --variant full --seed 2024 \
      --device 0 --epochs 50 --out results/info_nce/full__s2024.json
"""
import os, sys, json, time, copy, argparse, warnings, hashlib
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
from torch_geometric.loader import NeighborLoader
import Garfield as gf
from Garfield.model import Garfield
from Garfield.modules.loss import compute_contrastive_instanceloss

# Reuse the validated hippocampus config + downstream evaluator.
from ablate_hippo import faithful_config, evaluate

EVAL_SEED = 12345  # fixes the (stochastic) dropout augmentation at measurement time

# --- per-epoch L_ins trajectory recorder (projection space, during training) ---
# The legacy trainer only stores global/optim loss in epoch_logs, not the per-term
# instance loss. We record every L_ins call during training, tagged by the current
# epoch, and ALSO capture the temperature actually passed to the loss so that the
# scoring temperature can be verified against it (concern 1). compute_contrastive_-
# instanceloss is resolved in the GNNModelVAE module namespace (imported there), so we
# patch that reference; print_progress fires exactly once per epoch (monitor=True),
# which we use as the epoch counter.
_REC = {"epoch": 0, "rows": [], "active": False, "n_anchor": None,
        "train_temperature": None, "train_temperature_inconsistent": False}


def install_trajectory_recorder():
    import importlib
    # importlib returns the module objects reliably (the package __init__ re-exports
    # the GNNModelVAE *class* under the same dotted name, which shadows plain import).
    _vae = importlib.import_module("Garfield.modules.GNNModelVAE")
    _trn = importlib.import_module("Garfield.trainer.trainer")
    _orig_ince = _vae.compute_contrastive_instanceloss

    def _rec_ince(z_i, z_j, temperature):
        out = _orig_ince(z_i, z_j, temperature)  # real (full-batch) training loss
        # Concern (1): capture the temperature actually used by the training loss, and
        # flag if any call ever uses a different value (the experiment asserts it does not).
        t = float(temperature)
        if _REC["train_temperature"] is None:
            _REC["train_temperature"] = t
        elif abs(_REC["train_temperature"] - t) > 1e-12:
            _REC["train_temperature_inconsistent"] = True
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


def build_eval_loader(trainer, batch_size, num_neighbors, n_hops):
    """Build a DETERMINISTIC held-out validation loader (concerns 3 & 5).

    Uses the trainer's own split (``node_masked_data`` + ``val_mask``) and the same
    neighbour fan-out as training, but with ``shuffle=False`` so the seed/anchor nodes
    are yielded in a fixed order across variants and runs. Raises if there is no
    held-out validation split, so scoring can never silently fall back to training
    data. Returns ``(loader, data, n_val_nodes)``.
    """
    data = getattr(trainer, "node_masked_data", None)
    if data is None or not hasattr(data, "val_mask"):
        raise RuntimeError("trainer has no node_masked_data/val_mask; cannot build a "
                           "held-out validation loader for InfoNCE scoring.")
    n_val = int(data.val_mask.sum().item())
    if n_val <= 0:
        raise RuntimeError("node_val_ratio produced an EMPTY validation split; refusing "
                           "to score the InfoNCE bound on training data.")
    loader = NeighborLoader(
        data,
        num_neighbors=[num_neighbors] * n_hops,
        batch_size=batch_size,
        directed=False,
        shuffle=False,                     # deterministic, fixed seed-node order
        input_nodes=data.val_mask,         # validation-split seed nodes (disjoint from train seeds)
    )
    return loader, data, n_val


@torch.no_grad()
def measure_views(model, temperature, device, n_anchor, eval_loader, masked_data):
    """Score I_NCE on a deterministic held-out loader without modifying model state.

    The bound is scored on a FIXED number of seed/anchor nodes per validation batch:
    the first ``n_anchor`` rows of each batch are the mini-batch seed nodes (PyG
    NeighbourLoader convention; the model uses the same ``[:batch_size]`` slice
    elsewhere), and their encoder latents are computed with full neighbour message
    passing. Slicing to a constant ``n_anchor`` fixes the candidate count at
    K = 2*n_anchor - 1 for every scored batch. Batches with fewer than ``n_anchor``
    seed nodes (the incomplete final batch) are dropped.

    Concern (2): the network is put in ``eval()`` mode (BatchNorm frozen, attention
    dropout off, VAE reparameterise -> mean), so no parameter or buffer changes. The
    contrastive view augmentation still fires because it is driven by the
    ``augment_type`` argument, not by ``module.training``. We snapshot the full state
    before and assert it is byte-identical afterwards.
    """
    net = model.model
    was_training = net.training
    # concern (2)/(4): prove measurement leaves BOTH the model tensors AND the global RNG
    # untouched -- snapshot model state and RNG now, restore the RNG after scoring.
    state_before = copy.deepcopy(net.state_dict())
    rng_cpu = torch.get_rng_state()
    rng_cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rng_np = np.random.get_state()
    net.eval()                                       # NOT train(): freeze BN, no dropout
    augment_type = "dropout"
    lat, lat_shuf, proj = [], [], []
    z1_all, z2_all, anchor_ids_all = [], [], []
    scored, skipped, fixed_K = 0, 0, None

    # concern (5): make the held-out batches deterministic and identical across variants by
    # seeding the RNG immediately before the loader is materialised, independent of any RNG
    # consumed during training; with shuffle=False this fixes both the seed-node order and
    # the sampled neighbourhoods. The per-batch augmentation is reseeded inside the loop.
    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    batches = list(eval_loader)  # materialise once -> fixed, inspectable batches
    for bi, node_batch in enumerate(batches):
        seed_n = int(getattr(node_batch, "batch_size", n_anchor))
        if seed_n < n_anchor:           # incomplete final batch -> drop
            skipped += 1
            continue
        # global node ids of the scored anchors (held-out verification + provenance)
        anchor_ids = node_batch.n_id[:n_anchor].detach().cpu().numpy().astype(np.int64)
        torch.manual_seed(EVAL_SEED + bi)          # fixed eval augmentation (per batch)
        np.random.seed(EVAL_SEED + bi)
        node_batch = node_batch.to(device)
        out = net(node_batch, "omics", augment_type)
        z1 = out["z1_latent"][:n_anchor]
        z2 = out["z2_latent"][:n_anchor]
        if scored == 0:
            # the two views must genuinely differ, else the augmentation did not fire
            assert not torch.equal(z1, z2), \
                "the two views are identical -> augmentation inactive in eval mode"
        z1_all.append(z1.detach().cpu().numpy())
        z2_all.append(z2.detach().cpu().numpy())
        anchor_ids_all.append(anchor_ids)
        i_lat, _, fixed_K = infonce_bound(z1, z2, temperature)
        # shuffled positive pairing: permute view-2 rows -> destroys correspondence
        perm = torch.randperm(z2.size(0), device=z2.device)
        i_shuf, _, _ = infonce_bound(z1, z2[perm], temperature)
        # projector-space (secondary; favours full trivially since it trains that space)
        i_proj, _, _ = infonce_bound(out["z_1"][:n_anchor], out["z_2"][:n_anchor], temperature)
        lat.append(i_lat); lat_shuf.append(i_shuf); proj.append(i_proj)
        scored += 1

    # restore original mode and the global RNG, then prove the model was not modified
    net.train(was_training)
    torch.set_rng_state(rng_cpu)                     # concern (2)/(4): no process-state leak
    if rng_cuda is not None:
        torch.cuda.set_rng_state_all(rng_cuda)
    np.random.set_state(rng_np)
    state_after = net.state_dict()
    state_unchanged = (set(state_before) == set(state_after)) and all(
        torch.equal(state_before[k].cpu(), state_after[k].cpu()) for k in state_before
    )

    anchors = np.concatenate(anchor_ids_all) if anchor_ids_all else np.array([], np.int64)
    # concern (5)+(3): every scored anchor is a VALIDATION-split seed node, disjoint from the
    # training seed nodes (the underlying graph is shared, as is standard for inductive GNN
    # evaluation; this proves we never score on training-seed nodes).
    val_mask = masked_data.val_mask.detach().cpu().numpy().astype(bool)
    train_mask = masked_data.train_mask.detach().cpu().numpy().astype(bool)
    anchors_held_out = bool(anchors.size and val_mask[anchors].all()
                            and not train_mask[anchors].any())

    return dict(latent=lat, latent_shuffled=lat_shuf, projector=proj,
                scored_batches=scored, skipped_incomplete_batches=skipped,
                eval_candidate_count=int(n_anchor), fixed_K=fixed_K,
                model_state_unchanged=bool(state_unchanged),
                anchors_held_out=anchors_held_out,
                n_anchor_nodes_scored=int(anchors.size),
                anchor_node_ids=anchors,
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
    # Concern (1): in Garfield's loss the field below is passed as the `temperature`
    # argument of compute_contrastive_instanceloss and added to optim_loss with
    # coefficient 1 (it is NOT a linear loss weight) -- i.e. it IS the InfoNCE
    # temperature tau. We verify this against the training-time value after training.
    temperature = float(cfg["lambda_latent_contrastive_instanceloss"])  # tau = 1.0
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

    # --- concern (1): verify the scoring temperature == the training-time temperature
    train_temp = _REC["train_temperature"]
    assert train_temp is not None, "training-time temperature was never observed"
    assert not _REC["train_temperature_inconsistent"], \
        "the training loss used more than one contrastive temperature"
    temperature_verified = abs(temperature - train_temp) < 1e-9
    assert temperature_verified, (
        f"scoring temperature {temperature} != training temperature {train_temp}")

    # --- concern (4): verify the early-stopped best checkpoint is the live model
    trn = model.trainer
    assert getattr(trn, "reload_best_model_", False), "reload_best_model is disabled"
    best_sd = getattr(trn, "best_model_state_dict", None)
    assert best_sd is not None, "no best checkpoint was saved (validation never ran?)"
    cur_sd = model.model.state_dict()
    best_checkpoint_reloaded = (set(best_sd) == set(cur_sd)) and all(
        torch.equal(best_sd[k].cpu(), cur_sd[k].cpu()) for k in best_sd)
    assert best_checkpoint_reloaded, "live weights do not match the best checkpoint"
    best_epoch = int(getattr(trn, "best_epoch", -1)) + 1  # 1-indexed, as printed
    # concern (3): record the actual ablation toggles in effect (the cfg dict alone does
    # not capture them, since they are set on the model after construction).
    variant_flags = {"include_instance_loss": bool(model.model.include_instance_loss),
                     "include_cluster_loss": bool(model.model.include_cluster_loss)}

    # --- per-epoch L_ins trajectory (recorded via patched instance-loss calls) ---
    # Same fixed candidate count as the held-out scoring below.
    assert int(getattr(model.trainer, "node_batch_size_", n_anchor)) == n_anchor, \
        "node_batch_size changed at train time; fixed-anchor scoring basis inconsistent"
    traj_K = 2 * n_anchor - 1
    def to_ince(curve):
        return [float(np.log(traj_K) - v) for v in curve] if curve else []
    traj_lins = epoch_trajectory_Lins()

    # --- concerns (3)+(5): deterministic, held-out validation loader (shuffle=False) ---
    # Seed before constructing the loader so any construction-time sampler RNG is fixed
    # (measure_views reseeds again before materialising the batches).
    torch.manual_seed(EVAL_SEED)
    np.random.seed(EVAL_SEED)
    eval_loader, masked_data, n_val_nodes = build_eval_loader(
        model.trainer, batch_size=n_anchor,
        num_neighbors=int(cfg.get("num_neighbors", 5)),
        n_hops=int(cfg.get("loaders_n_hops", 2)))

    # --- converged value on the fixed held-out loader (best/reloaded checkpoint) ---
    m = measure_views(model, temperature, device, n_anchor, eval_loader, masked_data)
    # hard guarantees surfaced to the caller / artefacts
    assert m["model_state_unchanged"], "measurement modified model state"
    assert m["anchors_held_out"], "scored anchors are not strictly held out"

    # --- downstream sanity (compared to known hippo baseline ~0.220/0.804/0.451) ---
    Z = np.asarray(model.adata.obsm["garfield_latent"])
    spatial = np.asarray(model.adata.obsm["spatial"])
    gt = adata.obs["niches"].reindex(model.adata.obs_names).astype(str).values
    metrics, _ = evaluate(Z, spatial, gt, res=0.5)

    result = dict(
        variant=a.variant, seed=a.seed, epochs=a.epochs, n_obs=int(model.adata.n_obs),
        # --- protocol-validity verifications (concerns 1-5) ---
        temperature=temperature, train_temperature=train_temp,
        temperature_verified=bool(temperature_verified),
        temperature_source=("lambda_latent_contrastive_instanceloss, which Garfield "
                            "passes as the `temperature` arg of "
                            "compute_contrastive_instanceloss and adds to optim_loss "
                            "with coefficient 1 (it is the InfoNCE temperature, not a "
                            "linear weight); verified == training-time value"),
        temperature_consistent_across_calls=(not _REC["train_temperature_inconsistent"]),
        eval_mode="eval", model_state_unchanged=bool(m["model_state_unchanged"]),
        best_checkpoint_reloaded=bool(best_checkpoint_reloaded), best_epoch=best_epoch,
        variant_flags=variant_flags,
        used_DSBN=bool(cfg.get("used_DSBN", False)),
        val_loader_shuffle=False, n_val_nodes=int(n_val_nodes),
        anchors_held_out=bool(m["anchors_held_out"]),
        n_anchor_nodes_scored=int(m["n_anchor_nodes_scored"]),
        eval_seed=EVAL_SEED,
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

    # sidecar 1: resolved config (drop the AnnData object); record the actual ablation
    # toggles, which are set on the model after construction and are not in cfg.
    cfg_save = {k: v for k, v in cfg.items() if k != "adata_list"}
    cfg_save["variant_flags"] = variant_flags
    cfg_path = base + "_config.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg_save, f, indent=2, default=str)
    # sidecar 2: per-epoch L_ins / I_NCE trajectory
    log_path = base + "_epoch_log.csv"
    with open(log_path, "w") as f:
        f.write("epoch,L_ins,I_NCE\n")
        for ep, (li, ii) in enumerate(zip(traj_lins, to_ince(traj_lins)), 1):
            f.write(f"{ep},{li:.6f},{ii:.6f}\n")
    # sidecar 3: the fixed-anchor view embeddings actually scored for the bound + the
    # global ids of the held-out anchor nodes. The .npz binary is git-ignored in this
    # text repo; the committed, reproducible record is the meta sidecar (shapes +
    # checksums + node-selection rule + held-out node ids) together with the per-batch
    # scores stored in the result JSON.
    z1, z2 = m.get("z1"), m.get("z2")
    npz_path, meta_path = base + "_views.npz", base + "_views_meta.json"
    artifacts = dict(
        config=dict(file=os.path.basename(cfg_path), sha256=_sha256(cfg_path)),
        epoch_log=dict(file=os.path.basename(log_path), sha256=_sha256(log_path)),
    )
    if z1 is not None:
        np.savez_compressed(npz_path, z1=z1, z2=z2,
                            anchor_node_ids=m["anchor_node_ids"])
        meta = dict(
            description="fixed-anchor view embeddings scored for I_NCE(z1;z2)",
            node_selection="first eval_candidate_count seed nodes of each scored "
                           "validation batch, in shuffle=False loader order; incomplete "
                           "final batch dropped",
            eval_candidate_count=m["eval_candidate_count"], fixed_K=m["fixed_K"],
            scored_batches=m["scored_batches"],
            skipped_incomplete_batches=m["skipped_incomplete_batches"],
            anchors_held_out=bool(m["anchors_held_out"]),
            n_anchor_nodes_scored=int(m["n_anchor_nodes_scored"]),
            model_state_unchanged=bool(m["model_state_unchanged"]),
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
          f"tau={temperature} (verified={temperature_verified}) "
          f"best_epoch={best_epoch} state_unchanged={m['model_state_unchanged']} "
          f"held_out={m['anchors_held_out']} | "
          f"downstream ASW/coh/ARI={metrics['ASW_cluster']:.3f}/"
          f"{metrics['spatial_coherence']:.3f}/{metrics['niches_ARI_bestres']:.3f} "
          f"({result['train_sec']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
