# Garfield — Reproducibility

Reproducibility resources for **Garfield** (*Graph-based Contrastive Learning
Enables Fast Single-Cell Embedding*), a geometric deep-learning framework that
co-embeds single-cell and spatial multi-omics data—transcriptomics, epigenomics
and proteomics—into a shared, spatially aware latent space for tissue-niche
identification, reference-atlas construction and cross-sample niche transfer.

This repository accompanies the Garfield manuscript. It contains the analysis
notebooks behind each figure, the peer-review (revision) experiment harness, and
the baseline-comparison and benchmarking code needed to reproduce the reported
results.

- **Method / package:** https://github.com/zhou-1314/Garfield
- **Architecture:** variational graph autoencoder (VGAE) with a graph
  contrastive-learning objective and SVD-guided graph augmentation.
- **Keywords:** spatial omics · tissue niches · VGAE · tumor microenvironment

---

## Repository structure

| Path | Contents |
|------|----------|
| [`paper_analyses/`](paper_analyses/) | Analysis notebooks and baseline-method scripts behind the manuscript figures, organized **per figure** (Fig. 2–5) with an index README. |
| [`revision_experiments/`](revision_experiments/) | Peer-review experiment harness: module ablation, hyperparameter sensitivity, information-theoretic (InfoNCE / mutual-information) analysis, runtime & memory profiling, and recent-method baselines, with the result tables (CSV/JSON) and the scripts that build the figures/tables. |
| [`benchmarking/`](benchmarking/) | Standalone drivers for module ablation, loss-weight ablation and spatial scalability profiling (see its own README). |
| [`Garfield-garfield_dev/Garfield/`](Garfield-garfield_dev/Garfield/) | Frozen snapshot of the Garfield package source (v1.0.1, `garfield_dev` branch) used for **every** experiment in this repository, for exact reproducibility. |
| `*.ipynb` (top level) | Stand-alone reproduction notebooks: dataset preparation, the Slide-seqV2 hippocampus niche analysis and benchmark, and a lightweight train → save → reload → query-transfer tutorial. |

### Mapping analyses to figures

`paper_analyses/` is grouped so each manuscript figure maps to one folder:

| Folder | Figure | Analysis |
|--------|--------|----------|
| `fig2_singlecell_multiomics/` | Fig. 2a–f | Single-cell RNA+ATAC and RNA+ADT integration benchmarks vs. Seurat V4, MultiVI, MOFA+, Multigrate, TotalVI, scArches. |
| `fig2_spatial_multiomics_mousebrain/` | Fig. 2g–j | Spatial epigenome–transcriptome integration on mouse brain vs. NicheCompass, SpatialGlue, MultiVI, MOFA+. |
| `fig3_hippocampus_niche/` | Fig. 3a–e | Tissue-niche detection on the Slide-seqV2 mouse hippocampus vs. NicheCompass, CellCharter, GraphST. |
| `fig3_olfactorybulb_crossplatform/` | Fig. 3f–m | Mouse olfactory-bulb niches across Stereo-seq and Slide-seqV2, with cross-platform niche transfer. |
| `fig4_nsclc/` | Fig. 4 | NSCLC tumor-microenvironment spatial atlas and reference → query niche mapping. |
| `fig5_breastcancer_xenium/` | Fig. 5 | Human breast-cancer 10x Xenium niche, neighborhood-composition and cell–cell interaction analysis. |
| `additional_spatial_analyses/` | Supplementary / related | seqFISH mouse-organogenesis and additional single-modality spatial-niche analyses. |

See [`paper_analyses/README.md`](paper_analyses/README.md) for the full file-level map.

---

## Installation & environment

Experiments were run with:

- **OS / hardware:** Linux, NVIDIA A800 80 GB GPUs.
- **Python:** conda environment, PyTorch 2.1 (CUDA 12.1).
- **Package:** Garfield v1.0.1 (`garfield_dev`).

```bash
# 1. Clone this repository
git clone https://github.com/zhou-1314/Garfield-reproducibility.git
cd Garfield-reproducibility

# 2. Install the Garfield package (either from the package repo …)
pip install git+https://github.com/zhou-1314/Garfield.git
# … or use the frozen snapshot bundled here, as the notebooks do:
#   import sys; sys.path.insert(0, "Garfield-garfield_dev")

# 3. Extra dependencies for the scalability benchmark
pip install psutil GPUtil matplotlib seaborn
```

> The pinned `Garfield-garfield_dev/` snapshot reproduces the exact code used in
> the paper; installing the latest package may give slightly different numbers as
> the method evolves.

---

## Data availability

Raw and preprocessed datasets (`*.h5ad`, `*.h5`, `*.csv`) and trained model
checkpoints are **not** version-controlled (they are large); only code,
configurations and small result tables are tracked. Point each notebook/script
at your local copies of the public datasets below.

| Analysis | Platform / data | Accession |
|----------|-----------------|-----------|
| sc RNA+ATAC integration | paired scRNA + scATAC (9 datasets) | see manuscript Supplementary Table 1 |
| sc RNA+ADT integration | CITE-seq scRNA + scADT | GSE128639, GSE193181, Zenodo 6368128 (LUNG); see Supplementary Table 2 |
| spatial multi-omics | mouse-brain spatial ATAC + RNA | see manuscript |
| hippocampus niches | Slide-seqV2 mouse hippocampus | Stickels et al., 2021 |
| olfactory bulb | Stereo-seq + Slide-seqV2; scRNA reference | GSE121891 |
| NSCLC | NanoString CosMx, 8 sections / 5 donors (702,199 cells) | see manuscript |
| breast cancer | 10x Xenium human breast | 10x Genomics |

Full accessions and processing details are given in the manuscript Methods and
Supplementary Tables.

---

## Reproducing the results

**Figures (paper analyses).** Open the notebook for the figure of interest under
`paper_analyses/<figure>/`, set the dataset paths, and run top to bottom.
Baseline methods are under each `baselines/` subfolder. The two largest
notebooks are committed with cell outputs cleared to keep the repository small;
re-running them regenerates the figures.

**Peer-review experiments.** The harness lives in `revision_experiments/`:

```bash
cd revision_experiments
python launch.py             # orchestrate the experiment grid
python repro_hippo.py        # hippocampus reproduction
python ablate_hippo.py       # module ablation
python info_nce_experiment.py  # InfoNCE / mutual-information analysis
python aggregate.py          # aggregate run outputs
python aggregate_info_nce.py # aggregate InfoNCE runs
# make_*.py regenerate the result tables and the InfoNCE figure from committed CSVs
```

**Scalability benchmark.** See [`benchmarking/README.md`](benchmarking/README.md)
for runtime/memory profiling across dataset sizes and graph-construction methods.

---

## Citation

If you use Garfield or these resources, please cite:

```bibtex
@article{zhou2025graph,
  title   = {Graph-based Contrastive Learning Enables Unified Integration and
             Niche Transfer Across Single-Cell and Spatial Multi-Omics},
  author  = {Zhou, Weige and Fan, Xueying and Li, Lanxiang and Zheng, Jianrong
             and Liu, Xiaodong and Jin, Wenfei and Tian, Luyi},
  journal = {bioRxiv},
  year    = {2025},
  publisher = {Cold Spring Harbor Laboratory}
}
```

## Contact

- Issues: https://github.com/zhou-1314/Garfield-reproducibility/issues
- Package: https://github.com/zhou-1314/Garfield
- Email: zhouwg1314@gmail.com

## License

See [`LICENSE`](LICENSE).
