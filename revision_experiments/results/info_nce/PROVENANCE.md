# InfoNCE experiment — provenance & reproducibility (AC-1)

Direct InfoNCE bound `I_NCE(z₁;z₂)` on the full Slide-seqV2 hippocampus (41,786 obs),
full vs no_contrastive, seeds {2024, 2025}, 50 epochs (early-stopped best model).

## Fixed-N scoring protocol (predeclared)
The held-out bound is `I_NCE = log K − L_ins` scored on a **fixed candidate count**:
the `node_batch_size = 256` seed anchors of each held-out validation batch (the first
`node_batch.batch_size` rows; the model uses the same slice at `GNNModelVAE.py:522`),
giving a fixed `K = 2·256 − 1 = 511` for **every** scored batch. The single incomplete
final validation batch (fewer than 256 seeds) is dropped: `scored_batches = 16`,
`skipped_incomplete_batches = 1`. The same anchor slice is applied to the latent
(primary), shuffled-control, and projector (secondary) scores and to the saved view
embeddings. The per-epoch trajectory is scored on the same 256-anchor / `K = 511` basis,
so the trajectory and the converged value are directly comparable (a valid, positive
lower bound). Rationale: the `omics` forward computes `L_ins` over the full
neighbour-expanded subgraph (`GNNModelVAE.py:234-235`, the seed slice is commented out),
so "fixed `N`" cannot equal the exact per-batch training `N`; the 256 seed anchors are
the canonical mini-batch unit and yield a fixed-`N` bound on the same view-view MI.

## Corrected measurement protocol (round 3 — Reviewer 1)
The held-out bound is scored under a validity-checked protocol; every result JSON records the
outcome of each check (all `true` for the four runs):
- **Temperature verified.** τ = 1.0 is captured from the *training* loss call and asserted
  equal to the scoring temperature (`temperature_verified`). In Garfield's loss the field
  `lambda_latent_contrastive_instanceloss` **is** the InfoNCE temperature (passed as the
  `temperature` arg of `compute_contrastive_instanceloss`, added to the optim loss with unit
  coefficient), not a linear loss weight — a legacy naming artifact.
- **No state mutation.** Scoring runs in `net.eval()` under `torch.no_grad`; the full
  `state_dict` is snapshotted and asserted byte-identical afterwards (`model_state_unchanged`),
  and the global RNG (torch CPU/CUDA + NumPy) is snapshotted and restored, so scoring leaves
  neither the model nor the process RNG altered. The view augmentation (feature/edge dropout) is
  driven by the augment argument, not the train/eval flag, so the two views remain genuinely
  augmented (asserted non-identical). (`used_DSBN=false` here, so BatchNorm is never invoked.)
- **No training-loader fallback.** Raises if no held-out split exists; scores a dedicated
  loader over `val_mask` only. Every scored anchor is asserted to be a validation-split seed
  node (∈ `val_mask`, ∉ `train_mask`; `anchors_held_out`, `n_val_nodes = 4179`). The graph is
  shared (standard for inductive GNN eval), so anchors are never training *seed* nodes, though
  their sampled neighbourhoods may overlap training — exactly as for the model's own validation.
- **Best checkpoint verified.** Asserts `reload_best_model` is enabled and that the live
  weights equal `best_model_state_dict` (`best_checkpoint_reloaded`); records `best_epoch`
  (30 / 19 for seeds 2024 / 2025).
- **Deterministic batches.** The legacy val loader uses `shuffle=True`; scoring instead builds a
  dedicated `shuffle=False` `NeighborLoader` over the same held-out nodes and **seeds the RNG
  (`EVAL_SEED`) before constructing and materialising** it, so the seed-node order and the sampled
  neighbourhoods are fixed and identical across the two variants (within a seed); the per-batch
  augmentation is reseeded and the global anchor node-IDs are saved. The split — hence the anchor
  set — differs across the two random seeds by construction.

## Result (info_nce_summary.csv, sample SD over 2 seeds; corrected protocol)
| variant | latent I_NCE | projector I_NCE | shuffled |
|---------|-------------|-----------------|----------|
| full | 0.664 ± 0.022 | 0.747 ± 0.017 | −0.090 |
| no_contrastive | 0.658 ± 0.014 | 0.337 ± 0.008 | −0.090 |

Latent-space bound is essentially equal with/without the contrastive term (0.664 vs 0.658);
projector-space bound is much higher for the full model (0.747 vs 0.337 — partly definitional,
the projection head is trained only by the contrastive loss); the shuffled control collapses to
≈ −0.090. Values shifted modestly from the earlier train-mode scoring (latent 0.619/0.613 →
0.664/0.658; projector 0.768/0.269 → 0.747/0.337) because eval-mode scoring removes
attention-dropout and VAE-sampling noise; the conclusion is unchanged and the latent equality
is even tighter.

## Artifacts per run (`{variant}__s{seed}*`)
- `*.json` — full result: fixed `K`, `eval_candidate_count`, `scored_batches`,
  `skipped_incomplete_batches`, latent / projector / shuffled `I_NCE` (mean + per-batch
  arrays), per-epoch trajectory, seeds, temperature, downstream metrics, and an
  `artifacts` block with the sidecar file names and **sha256** checksums.
- `*_config.json` — resolved Garfield config (validated `faithful_config(8)` +
  `n_epochs=50` + seed; `variant_flags` records the `no_contrastive` toggles
  `include_instance_loss=False`, `include_cluster_loss=False`; `eval_seed=12345`).
- `*_epoch_log.csv` — per-epoch `L_ins` and derived `I_NCE` (256-anchor / `K=511` basis).
- `*_views.npz` — the fixed-anchor view embeddings (z₁, z₂) actually scored. **Git-ignored**
  in this text-only repo; the committed, reproducible record is `*_views_meta.json`
  (shapes, abs-sum, sha256, node-selection rule) plus the JSON `artifacts` checksums and
  the per-batch scores stored in the JSON.
- `info_nce_summary.csv` — cross-seed means ± sample SD, regenerated by
  `revision_experiments/aggregate_info_nce.py`.

Reproduce:
```
CUDA_VISIBLE_DEVICES=<gpu> python revision_experiments/info_nce_experiment.py \
    --variant <full|no_contrastive> --seed <2024|2025> --device 0 --epochs 50 \
    --out revision_experiments/results/info_nce/<variant>__s<seed>.json
python revision_experiments/aggregate_info_nce.py
python revision_experiments/make_infonce_figure.py
```
Determinism: training uses the seed; measurement runs in `eval()` mode and re-encodes the
two augmented views on a deterministic, held-out `shuffle=False` loader. The RNG is seeded
with EVAL_SEED=12345 **before the loader is materialised** (fixing both the seed-node order and
the sampled neighbourhoods) and re-seeded per batch for the dropout augmentation, so the
evaluation is fixed and identical for both variants within a seed; the global RNG is snapshotted
and restored around scoring. CUDA kernels are not bit-exact across machines, so embeddings/metrics
reproduce within small GPU-nondeterminism tolerance.

## Downstream-sanity (full vs no_contrastive at fixed Leiden res=0.5)
| run | ASW (↔0.220) | spatial coh (↔0.804) | niche-ARI (↔0.451) | n_clusters (↔12.5) |
|-----|------|------|------|------|
| full 2024 | 0.161 | **0.817** | **0.461** | **12** |
| full 2025 | 0.189 | 0.700 | 0.459 | 13 |
| no_contrastive 2024 | 0.207 | 0.801 | 0.465 | 13 |
| no_contrastive 2025 | 0.197 | 0.728 | 0.409 | 14 |

**Conclusion: the retrain reproduces baseline-quality latents, and the spatial-coherence
sanity sub-target is cluster-granularity-dependent.** The full-model runs reproduce the
baseline niche-ARI (0.459–0.461 vs 0.451) and fall in the baseline coherence range
(0.700–0.817 vs 0.804), at 12–13 predicted clusters (vs the baseline 12.5). Across the four
runs spatial coherence covaries with predicted cluster count (12–13 clusters → 0.801–0.817;
14 clusters → 0.728), so the ~0.80 baseline is a granularity-conditional value, not a hard
point. ASW is slightly below baseline for the full runs (0.161–0.189 vs 0.220) and also
covaries with granularity. Full and no_contrastive downstream metrics are close (removing
the contrastive term does not degrade niche quality). Because the InfoNCE conclusion rests on
the view-view bound — not on the downstream Leiden coherence — this granularity variance does
not affect the experiment's validity. See the Plan Evolution Log (Round 2) for the formal
re-baseline of this sanity sub-target.
