# Garfield — Reproducibility

Code, configurations and result tables for reproducing the analyses in the
**Garfield** manuscript (VGAE + SVD-guided graph contrastive learning for
spatial and single-cell multi-omics integration).

The Garfield package itself lives at
[zhou-1314/Garfield](https://github.com/zhou-1314/Garfield). This repository
contains the experiment harness, ablation/sensitivity studies, baseline
comparisons and the notebooks used to generate the figures and tables.

## Layout

| Path | Contents |
|------|----------|
| `Garfield-garfield_dev/Garfield/` | Package source snapshot (v1.0.1, `garfield_dev`) used for every experiment here. |
| `revision_experiments/` | Experiment harness: module ablation, hyperparameter sensitivity, InfoNCE / mutual-information analysis, runtime/memory, and recent-method baselines, plus the result CSVs and the scripts that build the tables/figures. |
| `revision_experiments/recent_baselines/` | Recent-method comparison (SpaMI / FGOT / soFusion / SpaMosaic and established baselines) on the P22 spatial ATAC–RNA mouse brain, scored against the external `ATAC_clusters` ground truth. |
| `benchmarking/` | Standalone ablation / weight-ablation / scalability drivers. |
| `*.ipynb` | Reproduction notebooks: data preparation, spatial tissue-niche analysis, the niche benchmark, and a lightweight train→save→reload→transfer tutorial. |

## Environment

- conda environment `Garfield`, PyTorch 2.1 (CUDA 12.1)
- GPUs: experiments were run on NVIDIA A800 80GB
- Install the package from the [Garfield repo](https://github.com/zhou-1314/Garfield),
  or add the bundled `Garfield-garfield_dev/` to `sys.path` (as the notebooks do).

## Data

Raw and preprocessed datasets (`*.h5ad`) and trained checkpoints are **not**
tracked here (they are large). Place the datasets each experiment expects under
the paths referenced by the scripts/notebooks; only code, configurations and the
small result tables (CSV/JSON/logs) are versioned.

## Reproducing

Each experiment family is driven from `revision_experiments/` (see `launch.py`,
`harness.py`, `repro_hippo.py`, `ablate_hippo.py`, `info_nce_experiment.py`) and
aggregated with `aggregate.py` / `aggregate_info_nce.py`. The `make_*` scripts
regenerate the tables and the InfoNCE figure from the committed result files.
