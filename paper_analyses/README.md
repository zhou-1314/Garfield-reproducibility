# Paper analyses

Analysis notebooks and baseline-method scripts behind the figures of the
Garfield manuscript, organized by figure. Each folder holds the Garfield
analysis notebook(s) and, where applicable, a `baselines/` subfolder with the
scripts used to run the competing methods.

> Datasets (`*.h5ad` etc.) are not tracked here; point the notebooks at your
> local copies. The two largest notebooks
> (`fig3_olfactorybulb_crossplatform/integrated_spatial_dataset.ipynb`,
> `fig4_nsclc/spatial_niche_NSCLC.ipynb`) are committed with cell outputs
> cleared to keep the repository small; re-run them to regenerate the figures.

## Map to figures

| Folder | Manuscript | Analysis |
|--------|-----------|----------|
| `fig2_singlecell_multiomics/rna_atac/` | Fig. 2a–b | Single-cell RNA+ATAC integration benchmark (vs Seurat V4, MultiVI, MOFA+, Multigrate) across the paired scRNA+scATAC datasets, with the representative `pbmc` cross-modality analysis. |
| `fig2_singlecell_multiomics/rna_adt/` | Fig. 2c,e–f | Single-cell RNA+ADT integration benchmark (vs Seurat V4, TotalVI, scArches, Multigrate). |
| `fig2_singlecell_multiomics/metrics_bench_plot.ipynb` | Fig. 2a–f | Aggregated metric computation and plotting for the single-cell benchmarks. |
| `fig2_spatial_multiomics_mousebrain/` | Fig. 2g–j | Spatial multi-omics integration on the mouse-brain epigenome–transcriptome dataset (vs NicheCompass, SpatialGlue, MultiVI, MOFA+). |
| `fig3_hippocampus_niche/` | Fig. 3a–e | Tissue-niche detection on the Slide-seqV2 mouse hippocampus (vs NicheCompass, CellCharter, GraphST). |
| `fig3_olfactorybulb_crossplatform/` | Fig. 3f–m | Mouse olfactory-bulb niches across Stereo-seq and Slide-seqV2, cross-platform niche transfer, and GSE121891 cell-type enrichment. |
| `fig4_nsclc/` | Fig. 4 | NSCLC tumor-microenvironment spatial atlas and reference→query niche mapping. |
| `fig5_breastcancer_xenium/` | Fig. 5 | Human breast cancer 10x Xenium niche analysis, neighborhood composition and cell-cell interaction. |
| `additional_spatial_analyses/` | Supplementary / related | seqFISH mouse-organogenesis niches and additional single-modality spatial-niche analyses. |
| `data_preparation/` | — | Spatial dataset preparation. |
| `utils/` | — | Helper scripts (e.g. conda environment transfer). |

The Garfield package itself: https://github.com/zhou-1314/Garfield
