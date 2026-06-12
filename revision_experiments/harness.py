#!/usr/bin/env python
"""Garfield revision experiment harness.

Config-driven single run: load dataset -> build Garfield config (base + overrides) ->
apply ablation flags on the inner GNNModelVAE -> train -> embed -> matched-resolution
Leiden -> ARI/NMI/AMI/HOM/ASW vs ground-truth label -> append one CSV row.

CLI:
  python harness.py --dataset brain10x --tag baseline \
      --overrides '{}' --flags '{}' --epochs 100 --seed 2024 --device 0 \
      --out results/ablation.csv
"""
import os, sys, json, time, argparse, traceback, warnings, gc
warnings.simplefilter("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
# Limit per-process CPU threads BEFORE importing numpy/torch to avoid
# oversubscription when many runs execute in parallel (80-core box -> thrash).
_GF_THREADS = os.environ.get("GF_THREADS", "8")
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, _GF_THREADS)

REPO = "/data2/zhouwg_data/project/Garfield-reproducibility"
sys.path.insert(0, os.path.join(REPO, "Garfield-garfield_dev"))

import numpy as np

# ---------------------------------------------------------------- datasets ----
def load_dataset(name):
    """Return (data_obj, base_overrides, label_candidates)."""
    import anndata as ad
    import glob
    if name == "brain10x":
        from mudata import MuData
        rna = ad.read_h5ad(os.path.join(REPO, "mouse_brain_10x_RNA.h5ad"))
        atac = ad.read_h5ad(os.path.join(REPO, "mouse_brain_10x_ATAC.h5ad"))
        for a in (rna, atac):
            if hasattr(a.X, "toarray"):
                a.X = a.X.toarray().astype(float)
            else:
                a.X = np.asarray(a.X, dtype=float)
        rna.var["feature_types"] = "RNA"
        atac.var["feature_types"] = "ATAC"
        # peaks are 'chr-start-end' -> split for gene-score calc
        sp = atac.var_names.str.split("-", expand=True).to_frame(index=False)
        sp.index = atac.var_names
        sp.columns = ["chr", "start", "end"]
        atac.var[["chr", "start", "end"]] = sp
        data = MuData({"rna": rna, "atac": atac})
        base = dict(
            profile="multi-modal", data_type="Paired", sub_data_type=["rna", "atac"],
            sample_col=None, weight=0.8, genome="mm10",
            use_gene_weight=True, used_hvg=True, rna_n_top_features=3000,
            atac_n_top_features=10000, n_components=50, n_neighbors=5,
            used_pca_feat=True, adj_key="connectivities",
            augment_type="svd", svd_q=5, use_FCencoder=False, conv_type="GATv2Conv",
            gnn_layer=2, hidden_dims=[128, 128], bottle_neck_neurons=20, cluster_num=20,
            num_heads=3, dropout=0.2, concat=True, used_edge_weight=True,
            used_DSBN=False, used_mmd=False, num_neighbors=5, loaders_n_hops=2,
            edge_batch_size=4096, node_batch_size=128,
            include_edge_recon_loss=True, include_gene_expr_recon_loss=True,
            lambda_latent_contrastive_instanceloss=1.0,
            lambda_latent_contrastive_clusterloss=0.5,
            lambda_gene_expr_recon=1.0, lambda_edge_recon=5.0,
            lambda_latent_adj_recon_loss=1.0, lambda_omics_recon_mmd_loss=3.0,
            n_epochs_no_edge_recon=0, learning_rate=0.001, weight_decay=1e-05,
            gradient_clipping=5, use_lightning=False,
        )
        return data, base, ["celltype", "rna:celltype", "cell_type"]
    if name == "nsclc1pct":
        files = sorted(glob.glob(os.path.join(
            REPO, "datasets/st_data/gold/nanostring_cosmx_human_nsclc_subsample_1pct_batch*.h5ad"
        )))
        if not files:
            raise FileNotFoundError("No NSCLC 1pct batch files found")
        data = [ad.read_h5ad(f) for f in files]
        base = dict(
            profile="RNA", data_type=None, sub_data_type=None, sample_col="batch",
            weight=0.8, genome=None, used_hvg=True, min_cells=3, min_features=0,
            keep_mt=False, target_sum=1e4, rna_n_top_features=960, n_components=50,
            n_neighbors=15, metric="euclidean", svd_solver="arpack",
            used_pca_feat=True, adj_key="connectivities",
            augment_type="svd", svd_q=5, use_FCencoder=False, conv_type="GATv2Conv",
            gnn_layer=2, hidden_dims=[128, 128], bottle_neck_neurons=20, cluster_num=20,
            num_heads=3, dropout=0.2, concat=True, used_edge_weight=True,
            used_DSBN=False, used_mmd=False, num_neighbors=5, loaders_n_hops=2,
            edge_batch_size=4096, node_batch_size=128,
            include_edge_recon_loss=True, include_gene_expr_recon_loss=True,
            lambda_latent_contrastive_instanceloss=1.0,
            lambda_latent_contrastive_clusterloss=0.5,
            lambda_gene_expr_recon=1.0, lambda_edge_recon=5.0,
            lambda_latent_adj_recon_loss=1.0, lambda_omics_recon_mmd_loss=3.0,
            n_epochs_no_edge_recon=0, learning_rate=0.001, weight_decay=1e-05,
            gradient_clipping=5, use_lightning=False,
        )
        return data, base, ["cell_type", "niche"]
    if name == "hippo":
        # adata_with_niches.h5ad carries the TRUE niche ground truth (obs['niches'],
        # 17 niches) + raw counts; config matches the paper's hippocampus run
        # (notebook 02.Garfield_spatial_niche_analysis2.ipynb) exactly.
        adata = ad.read_h5ad(os.path.join(REPO, "benchmark_unimodal/adata_with_niches.h5ad"))
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"].copy()  # Garfield normalizes raw counts
        # Transfer the paper's benchmark niche label (niche_type, 10 classes) from
        # adata_slideseqv2.h5ad (identical obs_names) so we can evaluate against the
        # SAME ground truth the paper's Fig.3 used, in addition to the finer niches.
        try:
            _sl = ad.read_h5ad(os.path.join(REPO, "benchmark_unimodal/adata_slideseqv2.h5ad"))
            if (_sl.obs_names == adata.obs_names).all() and "niche_type" in _sl.obs:
                adata.obs["niche_type"] = _sl.obs["niche_type"].values
        except Exception:
            pass
        base = dict(
            profile="spatial", data_type="single-modal", sub_data_type=None,
            sample_col=None, weight=1.0, genome=None, graph_const_method="mu_std",
            used_hvg=True, rna_n_top_features=3000, n_components=50, n_neighbors=5,
            metric="euclidean", used_pca_feat=False, adj_key="connectivities",
            augment_type="svd", svd_q=5, use_FCencoder=False, conv_type="GATv2Conv",
            gnn_layer=2, hidden_dims=[128, 128], bottle_neck_neurons=20, cluster_num=20,
            num_heads=3, dropout=0.2, concat=True, used_edge_weight=False,
            used_DSBN=False, used_mmd=False,
            num_neighbors=5, loaders_n_hops=2, edge_batch_size=4096, node_batch_size=128,
            lambda_latent_contrastive_clusterloss=1.0,
            lambda_latent_adj_recon_loss=0.5, lambda_omics_recon_mmd_loss=1.0,
        )
        # Primary eval label = niche_type (paper Fig.3 benchmark GT, 10 classes);
        # 'niches' (17 classes) is the finer annotation, reported secondarily.
        return adata, base, ["niche_type", "niches", "cell_type"]
    if name == "hippo_sparse":
        # Same as 'hippo' but with artificially increased sparsity: binomial
        # down-sampling of raw counts (keep ~30%) to emulate a shallow/sparse regime
        # for the ApproxSVD-rank (svd_q) robustness check requested in R2.B3/R2.D2.
        import scipy.sparse as _sp
        adata = ad.read_h5ad(os.path.join(REPO, "benchmark_unimodal/adata_with_niches.h5ad"))
        if "counts" in adata.layers:
            adata.X = adata.layers["counts"].copy()
        try:
            _sl = ad.read_h5ad(os.path.join(REPO, "benchmark_unimodal/adata_slideseqv2.h5ad"))
            if (_sl.obs_names == adata.obs_names).all() and "niche_type" in _sl.obs:
                adata.obs["niche_type"] = _sl.obs["niche_type"].values
        except Exception:
            pass
        import scanpy as _sc
        _rng = np.random.default_rng(2024)
        X = adata.X
        if _sp.issparse(X):
            X = X.tocsr().astype(np.float64)
            X.data = _rng.binomial(np.rint(X.data).astype(np.int64), 0.65).astype(np.float64)
            X.eliminate_zeros()
        else:
            X = _rng.binomial(np.rint(np.asarray(X)).astype(np.int64), 0.65).astype(np.float64)
        adata.X = X
        # drop cells/genes left empty by down-sampling (keeps obs labels aligned via subsetting)
        _sc.pp.filter_cells(adata, min_counts=1)
        _sc.pp.filter_genes(adata, min_cells=1)
        if "counts" in adata.layers:
            adata.layers["counts"] = adata.X.copy()
        base = dict(
            profile="spatial", data_type="single-modal", sub_data_type=None,
            sample_col=None, weight=1.0, genome=None, graph_const_method="mu_std",
            used_hvg=True, rna_n_top_features=3000, n_components=50, n_neighbors=5,
            metric="euclidean", used_pca_feat=False, adj_key="connectivities",
            augment_type="svd", svd_q=5, use_FCencoder=False, conv_type="GATv2Conv",
            gnn_layer=2, hidden_dims=[128, 128], bottle_neck_neurons=20, cluster_num=20,
            num_heads=3, dropout=0.2, concat=True, used_edge_weight=False,
            used_DSBN=False, used_mmd=False,
            num_neighbors=5, loaders_n_hops=2, edge_batch_size=4096, node_batch_size=128,
            lambda_latent_contrastive_clusterloss=1.0,
            lambda_latent_adj_recon_loss=0.5, lambda_omics_recon_mmd_loss=1.0,
        )
        return adata, base, ["niche_type", "niches", "cell_type"]
    raise ValueError(f"unknown dataset {name}")


# ---------------------------------------------------------------- metrics -----
def _search_res(adata, n_target, use_rep, start=0.1, end=3.0, step=0.1):
    import scanpy as sc
    best_res, best_gap = start, 1e9
    r = start
    while r <= end + 1e-9:
        sc.tl.leiden(adata, resolution=r, key_added="_tmp", random_state=0,
                     neighbors_key="_nn")
        ncl = adata.obs["_tmp"].nunique()
        gap = abs(ncl - n_target)
        if gap < best_gap:
            best_gap, best_res = gap, r
        if ncl >= n_target:
            break
        r = round(r + step, 4)
    return best_res


def cluster_and_score(latent, labels):
    import scanpy as sc, anndata as ad
    import sklearn.metrics as skm
    labels = np.asarray(labels).astype(str)
    mask = ~np.isin(labels, ["nan", "NaN", "None", ""])
    n_target = len(np.unique(labels[mask]))
    a = ad.AnnData(X=np.asarray(latent, dtype=float))
    a.obsm["X_lat"] = a.X.copy()
    sc.pp.neighbors(a, use_rep="X_lat", key_added="_nn", random_state=0)
    res = _search_res(a, n_target, "X_lat")
    sc.tl.leiden(a, resolution=res, key_added="pred", random_state=0, neighbors_key="_nn")
    pred = a.obs["pred"].values
    lt, pt = labels[mask], pred[mask]
    out = dict(
        n_target=int(n_target), n_pred=int(a.obs["pred"].nunique()), leiden_res=float(res),
        ARI=float(skm.adjusted_rand_score(lt, pt)),
        NMI=float(skm.normalized_mutual_info_score(lt, pt)),
        AMI=float(skm.adjusted_mutual_info_score(lt, pt)),
        HOM=float(skm.homogeneity_score(lt, pt)),
    )
    try:
        out["ASW"] = float(skm.silhouette_score(np.asarray(latent)[mask], lt))
    except Exception:
        out["ASW"] = float("nan")
    return out


def batch_mixing_score(latent, batches):
    import sklearn.metrics as skm
    from sklearn.neighbors import NearestNeighbors
    labels = np.asarray(batches).astype(str)
    Z = np.asarray(latent, dtype=float)
    uniq = np.unique(labels)
    out = {"batch_n": int(len(uniq))}
    if len(uniq) <= 1:
        out.update(ASW_batch=float("nan"), batch_knn_entropy=float("nan"))
        return out
    out["ASW_batch"] = float(skm.silhouette_score(Z, labels))
    k = min(31, len(labels))
    idx = NearestNeighbors(n_neighbors=k).fit(Z).kneighbors(Z, return_distance=False)
    denom = np.log(len(uniq))
    ent = []
    for neigh in idx[:, 1:]:
        vals, cnts = np.unique(labels[neigh], return_counts=True)
        p = cnts / cnts.sum()
        ent.append(float(-(p * np.log(p)).sum() / denom))
    out["batch_knn_entropy"] = float(np.mean(ent))
    return out


# ---------------------------------------------------------------- run ---------
def run(dataset, tag, overrides, flags, epochs, seed, device, out_csv):
    import torch, Garfield as gf
    from Garfield.model import Garfield
    try:
        torch.set_num_threads(int(os.environ.get("GF_THREADS", "8")))
    except Exception:
        pass
    t0 = time.time()
    def _log(m):
        print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)
    _log(f"loading dataset {dataset}")
    data, base, label_cands = load_dataset(dataset)
    _log("dataset loaded")

    workdir = os.path.join(REPO, "revision_experiments", "_runs", f"{dataset}_{tag}_s{seed}")
    os.makedirs(workdir, exist_ok=True)
    gf.settings.set_workdir(workdir)
    gf.settings.set_gf_params({})
    cfg = dict(gf.settings.gf_params)  # complete default-merged dict
    cfg.update(base); cfg.update(overrides)
    cfg["adata_list"] = data
    # Shared per-dataset preprocessing cache (ATAC gene-activity etc.) so the
    # expensive preprocessing is computed once and reused across all ablation/
    # sensitivity runs (they share identical preprocessing; only model/loss vary).
    shared_cache = os.path.join(REPO, "revision_experiments", "_cache", f"{dataset}_dev")
    os.makedirs(shared_cache, exist_ok=True)
    cfg["user_cache_path"] = shared_cache
    cfg["n_epochs"] = epochs
    cfg["seed"] = seed
    cfg["device_id"] = device
    cfg["latent_key"] = "garfield_latent"

    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
        except Exception:
            pass
    # ---- Cache the FULL preprocessed adata (gene-scores + cell-matching +
    # HVG/PCA/graph). All ablation/sensitivity runs share identical preprocessing
    # (overrides only touch model/loss params), so compute it once and reuse.
    import pickle, importlib
    # NB: attribute `Garfield.model.Garfield` is shadowed by the *class*; grab the
    # actual module object from sys.modules via importlib to patch its globals.
    _gmod = importlib.import_module("Garfield.model.Garfield")
    pp_cache = os.path.join(shared_cache, "preprocessed_adata.pkl")
    if not hasattr(_gmod, "_orig_DataProcess"):
        _gmod._orig_DataProcess = _gmod.DataProcess

    def _cached_DataProcess(*a, **k):
        import fcntl
        lock_path = pp_cache + ".lock"
        with open(lock_path, "w") as lock_fh:
            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            if os.path.exists(pp_cache):
                _log("loading cached preprocessed adata")
                with open(pp_cache, "rb") as fh:
                    return pickle.load(fh)
            ad_ = _gmod._orig_DataProcess(*a, **k)
            tmp_cache = f"{pp_cache}.tmp.{os.getpid()}"
            try:
                with open(tmp_cache, "wb") as fh:
                    pickle.dump(ad_, fh, protocol=4)
                os.replace(tmp_cache, pp_cache)
                _log(f"cached preprocessed adata -> {pp_cache}")
            except Exception as e:  # pragma: no cover
                try:
                    if os.path.exists(tmp_cache):
                        os.remove(tmp_cache)
                except Exception:
                    pass
                _log(f"WARN could not cache preprocessed adata: {e}")
            return ad_

    _gmod.DataProcess = _cached_DataProcess
    _log("building Garfield model (runs preprocessing)")
    model = Garfield(cfg)
    _log("preprocessing done; starting training")
    # apply ablation flags on inner module
    for k, v in (flags or {}).items():
        setattr(model.model, k, v)
    model.train()
    _log("training done")
    train_s = time.time() - t0

    adata = model.adata
    latent = adata.obsm[cfg["latent_key"]]
    label_key = next((k for k in label_cands if k in adata.obs.columns), None)
    if label_key is None:
        raise RuntimeError(f"no label among {label_cands}; obs={list(adata.obs.columns)[:30]}")
    metrics = cluster_and_score(latent, adata.obs[label_key].values)
    batch_key = None
    for k in ("batch", "rna:batch", "sample", "rna:sample"):
        if k in adata.obs.columns:
            batch_key = k
            break
    if batch_key is not None:
        metrics.update(batch_mixing_score(latent, adata.obs[batch_key].values))

    try:
        peak_mem_gb = (torch.cuda.max_memory_allocated(device) / 1e9
                       if torch.cuda.is_available() else float("nan"))
    except Exception:
        peak_mem_gb = float("nan")
    row = dict(dataset=dataset, tag=tag, seed=seed, epochs=epochs,
               overrides=json.dumps(overrides), flags=json.dumps(flags),
               label_key=label_key, batch_key=batch_key or "", n_obs=int(adata.n_obs),
               train_sec=round(train_s, 1), peak_gpu_gb=round(peak_mem_gb, 2),
               **metrics)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    import csv
    write_header = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)
    print(f"[DONE] {dataset}/{tag} seed={seed} "
          f"ARI={metrics['ARI']:.3f} NMI={metrics['NMI']:.3f} "
          f"({train_s:.0f}s, {peak_mem_gb:.1f}GB)", flush=True)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--overrides", default="{}")
    p.add_argument("--flags", default="{}")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=2024)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    try:
        run(a.dataset, a.tag, json.loads(a.overrides), json.loads(a.flags),
            a.epochs, a.seed, a.device, a.out)
    except Exception:
        traceback.print_exc()
        sys.exit(1)
