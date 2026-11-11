"""
Classical baselines on EXACTLY the LLM cells of the v2 design.

For each dataset and ladder size N of data/splits/<ds>_v2.json: train on the first N molecules
of `train_order_ids`, predict the 150 `test_ids`. Methods:

  tabpfn   TabPFNRegressor v3.5 (in-context tabular foundation model) on RDKit 2D descriptors
           (features.cell_descriptors: training-row column selection + median imputation).
           prediction = mean of the predictive distribution; uncertainty (1 sigma) =
           (q84 - q16) / 2; quantiles q05 ... q95 stored as predicted (interval metrics can use
           them directly instead of the Gaussian assumption). Extra column sd_full = standard
           deviation of the full predictive bar distribution (diagnostic only).
  rf       RandomForestRegressor, 200 trees, Morgan r=2 / 2048 bits (as v1); prediction = forest
           mean, uncertainty = std over the 200 tree predictions (ddof=1).
  rf_desc  the same RF (200 trees, sigma = tree std) on the SAME descriptor matrix as tabpfn
           (features.cell_descriptors), so learner and feature effects can be separated.
  trivial  training-mean predictor, uncertainty = training std (ddof=1); sanity floor.

Added 2026-10-13 (principled-sigma baselines; all on the cached Morgan r=2 / 2048-bit fingerprints
of features.morgan_matrix except chemprop_ens, which reads the SMILES):
  gp_tanimoto   exact GP regression, k(x, x') = a * Tanimoto(x, x') (bit vectors) + s2 * delta,
                constant mean = training mean (not fitted); a and s2 maximise the log marginal
                likelihood (scipy L-BFGS-B on log a, log s2 with the analytic gradient; bounds
                a in [1e-3, 1e2] x var(y), s2 in [1e-4, 1e1] x var(y); 14 starts: (var, 0.1 var), a
                3 x 3 grid a in {0.01, 0.3, 10} x s2 in {0.001, 0.05, 1} (x var(y)) and 4 uniform in
                the log-bounds from default_rng(seed); best NLML wins -- at N = 30 the likelihood can
                be bimodal, interpolating (s2 -> lower bound) vs noise-only (a -> lower bound, i.e.
                the trivial predictor), so a single start is not enough). prediction =
                posterior mean, uncertainty = posterior predictive SD INCLUDING the noise s2.
                numpy / scipy only.
  mlp_ens       deep ensemble of 5 MLPs (torch, CPU) on the raw 2048 bits: 2 x 256 ReLU, dropout 0.1,
                MSE on standardised targets, Adam lr 1e-3, batch min(64, N); v1's stopping rule
                (sklearn MLPRegressor defaults: <= 500 epochs, stop after 10 epochs without a
                training-loss gain > 1e-4; no validation split); member k seeded 100*seed + k
                (init, dropout, shuffling). prediction = member mean, uncertainty = member SD
                (ddof=1). Deviations from v1 (archive/v1_2026-09_contaminated/src/baselines_rf_mlp.py:
                sklearn MLPRegressor, 5 different architectures, bootstrap bagging, StandardScaler on
                the bits, unscaled targets): one architecture, no bagging (standard deep ensemble),
                raw bits, standardised targets.
  chemprop_ens  v1 Chemprop D-MPNN deep ensemble (archive/.../baseline_chemprop.py, recipe kept):
                chemprop 2.0 defaults (BondMessagePassing d_h 300 depth 3, MeanAggregation,
                RegressionFFN 1 x 300, batch norm, Noam schedule 1e-4 -> 1e-3 -> 1e-4, warm-up 2),
                targets standardised on the training rows, batch min(64, N), 50 epochs, no
                validation split, 5 members with seed 100*seed + k, Lightning deterministic=True
                (bit-reproducible on CPU; without it two runs differ by up to 1e-3). prediction = member mean,
                uncertainty = member SD (ddof=1). Needs chemprop (base env: chemprop 2.0.0,
                torch 2.2.2 CPU, lightning 2.2.4); imported lazily.
  knn_tanimoto  k = 5 nearest training molecules by Tanimoto similarity (ties at the k-th place
                broken by training order, stable sort); prediction = similarity-weighted mean of
                their labels (plain mean if all 5 similarities are 0), uncertainty = max(SD of
                the 5 labels (ddof=1), 0.01). Deterministic (the seed has no effect).

Leakage guard: the label vector handed to every fit is built from a copy of all labels in which
the 150 test labels are set to NaN, and the fit asserts finite labels -- a test label reaching a
fit would fail loudly. Test truths are taken from data/test_sets/<ds>_v2test150.csv after
prediction and cross-checked against data/<ds>.csv.

Output per cell: results/baselines/<method>__<ds>__N<size>.csv (seed 0) or
<method>__<ds>__N<size>__s<seed>.csv (other seeds), columns
  dataset, method, size, seed, mol_id, smiles, truth, prediction, uncertainty,
  q05, q16, q25, q50, q75, q84, q95, sd_full, wall_s
plus <same stem>.meta.json (training ids, feature count, versions, device, wall clock).
Existing cells are skipped unless --force.

Context draws (2026-10-13): --draw d (default 1) trains on the first N molecules of
data/splits/<ds>_v2_draw<d>.json (same test set and pool, independently permuted pool order;
src/make_splits.py --draws 3). Draw d >= 2 cells are written as <stem>__d<d>.csv (+ .meta.json),
e.g. tabpfn__bace__N30__d2.csv, with an extra column `draw` (after `seed`) and "draw" in the meta;
draw-1 files keep their names and columns.

Environment: TabPFN needs the `uc44` conda env (torch >= 2.5); rf / trivial run in any env with
rdkit + scikit-learn. tabpfn is imported lazily inside its method function. gp_tanimoto,
knn_tanimoto, mlp_ens were run in `uc44` (Morgan cache of rdkit 2026.03.6, as rf); chemprop_ens in
the base env (chemprop 2.0.0 is only installed there; the `chemprop` conda env is empty).

    python src/baselines/run_baselines.py --methods gp_tanimoto,knn_tanimoto --sizes all --seeds 0,1,2  # uc44
    python src/baselines/run_baselines.py --methods mlp_ens --sizes all --seeds 0                       # uc44
    python src/baselines/run_baselines.py --methods chemprop_ens --sizes all --seeds 0                  # base env

    conda activate uc44
    python src/baselines/run_baselines.py --methods tabpfn,rf,rf_desc,trivial --datasets all --sizes all --seeds 0
    python src/baselines/run_baselines.py --methods tabpfn,rf,rf_desc,trivial --datasets all --sizes 30,60,120 --draw 2

Weights: TabPFN v3.5 is downloaded once from Hugging Face (Prior-Labs/tabpfn_3_5) into the
TabPFN cache (%APPDATA%/tabpfn) after a license check that needs TABPFN_TOKEN. The token is read
from the environment or, if absent, from the git-ignored .env in the repository root at runtime; it is never
printed or written into this repository.
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, HERE)
from common import DATASET_KEYS, RESULTS, dotenv_value, load_dataset, load_split, load_test_set  # noqa: E402
import features  # noqa: E402

OUT = os.path.join(RESULTS, "baselines")
METHODS = ["tabpfn", "rf", "rf_desc", "trivial", "gp_tanimoto", "mlp_ens", "chemprop_ens", "knn_tanimoto"]
QUANTILES = [0.05, 0.16, 0.25, 0.5, 0.75, 0.84, 0.95]
QCOLS = ["q05", "q16", "q25", "q50", "q75", "q84", "q95"]
COLUMNS = ["dataset", "method", "size", "seed", "mol_id", "smiles", "truth", "prediction",
           "uncertainty"] + QCOLS + ["sd_full", "wall_s"]


# --------------------------------------------------------------------------------------
# Cell definition
# --------------------------------------------------------------------------------------

class Cell:
    """Row indices, ids and masked labels of one (dataset, size) cell."""

    def __init__(self, ds, size, df=None, split=None, check_truth=True, draw=1):
        self.ds, self.size, self.draw = ds, int(size), int(draw)
        self.df = load_dataset(ds) if df is None else df
        self.split = load_split(ds, self.draw) if split is None else split
        assert int(self.split.get("draw", 1)) == self.draw, f"{ds}: split draw != {self.draw}"
        ids = self.df.mol_id.astype(str).values
        pos = {m: i for i, m in enumerate(ids)}
        self.train_ids = [str(m) for m in self.split["train_order_ids"][:self.size]]
        self.test_ids = [str(m) for m in self.split["test_ids"]]
        assert len(self.train_ids) == self.size, f"{ds}: pool smaller than N={self.size}"
        assert len(set(self.train_ids)) == self.size and len(set(self.test_ids)) == len(self.test_ids)
        assert not set(self.train_ids) & set(self.test_ids), f"{ds}: train/test overlap"
        self.train_rows = np.array([pos[m] for m in self.train_ids])
        self.test_rows = np.array([pos[m] for m in self.test_ids])
        y = self.df.value.to_numpy(float)
        y_masked = y.copy()
        y_masked[self.test_rows] = np.nan          # test labels never reach a fit
        self.y_train = y_masked[self.train_rows]
        assert np.all(np.isfinite(self.y_train))
        # truths from the frozen test-set file, cross-checked against the dataset file
        T = load_test_set(ds).assign(mol_id=lambda t: t.mol_id.astype(str)).set_index("mol_id")
        assert set(T.index) == set(self.test_ids), f"{ds}: test-set file != split test_ids"
        self.truth = T.loc[self.test_ids, "value"].to_numpy(float)
        if check_truth:                            # off only in the selfcheck's label-poisoning test
            assert np.allclose(self.truth, y[self.test_rows])
        self.smiles_test = self.df.smiles.values[self.test_rows]


def checked_fit_labels(y, n):
    y = np.asarray(y, float)
    if len(y) != n or not np.all(np.isfinite(y)):
        raise AssertionError("fit labels are not the N finite training labels")
    return y


# --------------------------------------------------------------------------------------
# Methods: each returns dict(prediction, uncertainty, quantiles (n,7) or None, sd_full, n_features, info)
# --------------------------------------------------------------------------------------

def load_tabpfn_token():
    """Put TABPFN_TOKEN into os.environ: from the environment, else from the line TABPFN_TOKEN=... of the
    git-ignored .env file in the repository root (common.dotenv_value). Never prints it."""
    os.environ.setdefault("TABPFN_NO_BROWSER", "1")      # never open a browser login
    if os.environ.get("TABPFN_TOKEN", "").strip():
        return "env"
    v = dotenv_value("TABPFN_TOKEN")
    if v:
        os.environ["TABPFN_TOKEN"] = v
        return "repository .env"
    return None


_TABPFN = {}


def tabpfn_setup(device="auto", version="v3.5"):
    if _TABPFN:
        return _TABPFN
    import torch
    import tabpfn
    from tabpfn.constants import ModelVersion
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    token_src = load_tabpfn_token()
    _TABPFN.update(module=tabpfn, ModelVersion=ModelVersion, device=device,
                   version=ModelVersion(version), token_source=token_src,
                   versions=dict(torch=torch.__version__, tabpfn=getattr(tabpfn, "__version__", "?")))
    return _TABPFN


def fit_tabpfn(cell, seed, device="auto", version="v3.5"):
    st = tabpfn_setup(device, version)
    from tabpfn import TabPFNRegressor
    X, names, _ = features.descriptor_matrix(cell.ds)
    Xtr, Xte, kept = features.cell_descriptors(X, names, cell.train_rows, cell.test_rows)
    y = checked_fit_labels(cell.y_train, cell.size)
    m = TabPFNRegressor.create_default_for_version(st["version"], random_state=seed, device=st["device"])
    m.fit(Xtr.astype(np.float32), y)
    out = m.predict(Xte.astype(np.float32), output_type="full", quantiles=list(QUANTILES))
    q = np.stack([np.asarray(a, float) for a in out["quantiles"]], axis=1)
    sd_full = np.full(len(Xte), np.nan)
    try:
        import torch
        with torch.no_grad():
            var = out["criterion"].variance(out["logits"])
        sd_full = np.sqrt(np.clip(var.detach().cpu().numpy().astype(float), 0, None)).reshape(-1)
    except Exception as e:  # diagnostic column only
        print(f"  [warn] sd_full unavailable: {e}")
    return dict(prediction=np.asarray(out["mean"], float), uncertainty=(q[:, 5] - q[:, 1]) / 2,
                quantiles=q, sd_full=sd_full, n_features=len(kept),
                info=dict(model_version=st["version"].value, device=st["device"], **st["versions"],
                          n_estimators=str(getattr(m, "n_estimators", "?"))))


def fit_rf(cell, seed, n_trees=200):
    from sklearn.ensemble import RandomForestRegressor
    F, _ = features.morgan_matrix(cell.ds)
    y = checked_fit_labels(cell.y_train, cell.size)
    rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    rf.fit(F[cell.train_rows], y)
    Xte = F[cell.test_rows]
    trees = np.stack([t.predict(Xte) for t in rf.estimators_], axis=0)
    import sklearn
    return dict(prediction=trees.mean(axis=0), uncertainty=trees.std(axis=0, ddof=1),
                quantiles=None, sd_full=None, n_features=F.shape[1],
                info=dict(n_trees=n_trees, features="morgan_r2_2048", sklearn=sklearn.__version__))


def fit_rf_desc(cell, seed, n_trees=200):
    from sklearn.ensemble import RandomForestRegressor
    import sklearn
    X, names, _ = features.descriptor_matrix(cell.ds)
    Xtr, Xte, kept = features.cell_descriptors(X, names, cell.train_rows, cell.test_rows)
    y = checked_fit_labels(cell.y_train, cell.size)
    rf = RandomForestRegressor(n_estimators=n_trees, n_jobs=-1, random_state=seed)
    rf.fit(Xtr, y)
    trees = np.stack([t.predict(Xte) for t in rf.estimators_], axis=0)
    return dict(prediction=trees.mean(axis=0), uncertainty=trees.std(axis=0, ddof=1),
                quantiles=None, sd_full=None, n_features=len(kept),
                info=dict(n_trees=n_trees, features="rdkit2d (cell_descriptors, as tabpfn)",
                          sklearn=sklearn.__version__))


def fit_trivial(cell, seed):
    y = checked_fit_labels(cell.y_train, cell.size)
    n = len(cell.test_rows)
    return dict(prediction=np.full(n, y.mean()), uncertainty=np.full(n, y.std(ddof=1)),
                quantiles=None, sd_full=None, n_features=0, info={})


def tanimoto(A, B):
    """Tanimoto (Jaccard) similarity of bit-vector rows: |a & b| / (|a| + |b| - |a & b|).
    Two empty fingerprints have similarity 1 (identical), an empty and a non-empty one 0."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    inter = A @ B.T                                   # exact: integer counts <= 2048 in float64
    denom = A.sum(axis=1)[:, None] + B.sum(axis=1)[None, :] - inter
    with np.errstate(invalid="ignore", divide="ignore"):
        T = np.where(denom > 0, inter / np.where(denom > 0, denom, 1.0), 1.0)
    return T


GP_JITTER = 1e-8


def _gp_nlml(theta, T, yc):
    """Negative log marginal likelihood and gradient w.r.t. (log a, log s2) of
    K = a * T + (s2 + jitter) I, zero-mean residuals yc."""
    from scipy.linalg import cho_factor, cho_solve
    a, s2 = np.exp(theta)
    n = len(yc)
    K = a * T + (s2 + GP_JITTER) * np.eye(n)
    try:
        c = cho_factor(K, lower=True, check_finite=False)
    except np.linalg.LinAlgError:
        return 1e25, np.zeros(2)
    alpha = cho_solve(c, yc, check_finite=False)
    nll = 0.5 * yc @ alpha + np.sum(np.log(np.diag(c[0]))) + 0.5 * n * np.log(2 * np.pi)
    Kinv = cho_solve(c, np.eye(n), check_finite=False)
    W = np.outer(alpha, alpha) - Kinv                 # d nll / d theta_j = -0.5 tr(W dK/dtheta_j)
    g = np.array([-0.5 * a * np.sum(W * T), -0.5 * s2 * np.trace(W)])
    return float(nll), g


GP_GRID_A = (1e-2, 0.3, 10.0)        # start grid, relative to var(y)
GP_GRID_S2 = (1e-3, 0.05, 1.0)


def fit_gp_tanimoto(cell, seed, n_random=4):
    from scipy.linalg import cho_factor, cho_solve
    from scipy.optimize import minimize
    import scipy
    F, _ = features.morgan_matrix(cell.ds)
    y = checked_fit_labels(cell.y_train, cell.size)
    Xtr, Xte = F[cell.train_rows], F[cell.test_rows]
    T = tanimoto(Xtr, Xtr)
    m = float(y.mean())
    yc = y - m
    v = float(np.var(y)) if np.var(y) > 0 else 1.0
    bounds = [(np.log(1e-3 * v), np.log(1e2 * v)), (np.log(1e-4 * v), np.log(1e1 * v))]
    rng = np.random.default_rng(seed)
    starts = [np.array([np.log(v), np.log(0.1 * v)])]
    starts += [np.array([np.log(ga * v), np.log(gs * v)]) for ga in GP_GRID_A for gs in GP_GRID_S2]
    starts += [np.array([rng.uniform(*bounds[0]), rng.uniform(*bounds[1])]) for _ in range(n_random)]
    best = None
    fits = []
    for k, x0 in enumerate(starts):
        r = minimize(_gp_nlml, x0, args=(T, yc), jac=True, method="L-BFGS-B", bounds=bounds,
                     options=dict(maxiter=500))
        fits.append(dict(start=k, nlml=float(r.fun), log_a=float(r.x[0]), log_s2=float(r.x[1]),
                         nit=int(r.nit), success=bool(r.success)))
        if best is None or r.fun < best.fun - 1e-9:
            best = r
    a, s2 = np.exp(best.x)
    K = a * T + (s2 + GP_JITTER) * np.eye(len(y))
    c = cho_factor(K, lower=True, check_finite=False)
    alpha = cho_solve(c, yc, check_finite=False)
    Ks = a * tanimoto(Xte, Xtr)                       # (n_test, N)
    mu = m + Ks @ alpha
    V = cho_solve(c, Ks.T, check_finite=False)
    prior = a * np.diag(tanimoto(Xte, Xte))           # = a (self-similarity 1)
    var_f = np.clip(prior - np.sum(Ks.T * V, axis=0), 0.0, None)
    sigma = np.sqrt(var_f + s2)
    return dict(prediction=mu, uncertainty=sigma, quantiles=None, sd_full=None, n_features=F.shape[1],
                info=dict(features="morgan_r2_2048 bits", kernel="a*Tanimoto + s2*I", mean="training mean",
                          gp_amplitude=float(a), gp_noise_var=float(s2), gp_mean=m,
                          gp_nlml=float(best.fun), gp_var_y=v, gp_best_start=int(np.argmin([f["nlml"] for f in fits])),
                          gp_starts=fits, gp_bounds_rel_var_y=dict(a=[1e-3, 1e2], s2=[1e-4, 1e1]),
                          rdkit_fp=features._rdkit_version(), numpy=np.__version__, scipy=scipy.__version__))


def fit_knn_tanimoto(cell, seed, k=5, floor=0.01):
    F, _ = features.morgan_matrix(cell.ds)
    y = checked_fit_labels(cell.y_train, cell.size)
    S = tanimoto(F[cell.test_rows], F[cell.train_rows])     # (n_test, N)
    idx = np.argsort(-S, axis=1, kind="stable")[:, :k]      # ties: earlier training position first
    sims = np.take_along_axis(S, idx, axis=1)
    lab = y[idx]
    wsum = sims.sum(axis=1)
    pred = np.where(wsum > 0, (sims * lab).sum(axis=1) / np.where(wsum > 0, wsum, 1.0), lab.mean(axis=1))
    sd = lab.std(axis=1, ddof=1)
    return dict(prediction=pred, uncertainty=np.maximum(sd, floor), quantiles=None, sd_full=None,
                n_features=F.shape[1],
                info=dict(features="morgan_r2_2048 bits", k=k, sigma_floor=floor,
                          n_sigma_floored=int(np.sum(sd < floor)),
                          mean_nn_similarity=float(sims.mean()),
                          median_top1_similarity=float(np.median(sims[:, 0])),
                          rdkit_fp=features._rdkit_version(), numpy=np.__version__))


MLP_HIDDEN, MLP_DROPOUT, MLP_LR, MLP_MAX_EPOCHS, MLP_TOL, MLP_PATIENCE = (256, 256), 0.1, 1e-3, 500, 1e-4, 10


def fit_mlp_ens(cell, seed, n_members=5, threads=4):
    """Deep ensemble (Lakshminarayanan et al. 2017) of 5 MLPs on raw Morgan bits.
    2048 -> 256 -> 256 -> 1, ReLU, dropout 0.1 after each hidden layer; MSE on targets standardised
    with the training mean / SD; Adam lr 1e-3, batch min(64, N), reshuffled every epoch.
    Stopping rule kept from v1 (sklearn MLPRegressor defaults, no validation split): at most
    500 epochs, stop when the epoch-mean training loss has not improved by > 1e-4 for 10
    consecutive epochs. Member k: torch.manual_seed(100*seed + k) for init, dropout and shuffling."""
    import torch
    F, _ = features.morgan_matrix(cell.ds)
    y = checked_fit_labels(cell.y_train, cell.size)
    torch.set_num_threads(threads)
    mu_y, sd_y = float(y.mean()), float(y.std(ddof=0)) or 1.0
    Xtr = torch.tensor(F[cell.train_rows], dtype=torch.float32)
    Xte = torch.tensor(F[cell.test_rows], dtype=torch.float32)
    yt = torch.tensor((y - mu_y) / sd_y, dtype=torch.float32).reshape(-1, 1)
    n, bs = len(y), min(64, len(y))
    preds, epochs = [], []
    for k in range(n_members):
        ms = 100 * seed + k
        g = torch.Generator().manual_seed(ms)
        torch.manual_seed(ms)
        h1, h2 = MLP_HIDDEN
        net = torch.nn.Sequential(torch.nn.Linear(Xtr.shape[1], h1), torch.nn.ReLU(), torch.nn.Dropout(MLP_DROPOUT),
                                  torch.nn.Linear(h1, h2), torch.nn.ReLU(), torch.nn.Dropout(MLP_DROPOUT),
                                  torch.nn.Linear(h2, 1))
        opt = torch.optim.Adam(net.parameters(), lr=MLP_LR)
        best, stall, ep = np.inf, 0, 0
        for ep in range(1, MLP_MAX_EPOCHS + 1):
            net.train()
            perm = torch.randperm(n, generator=g)
            tot = 0.0
            for b in range(0, n, bs):
                idx = perm[b:b + bs]
                opt.zero_grad()
                loss = torch.nn.functional.mse_loss(net(Xtr[idx]), yt[idx])
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(idx)
            tot /= n
            if tot > best - MLP_TOL:
                stall += 1
                if stall >= MLP_PATIENCE:
                    break
            else:
                stall = 0
            best = min(best, tot)
        epochs.append(ep)
        net.eval()
        with torch.no_grad():
            preds.append(net(Xte).numpy().reshape(-1).astype(np.float64) * sd_y + mu_y)
    P = np.stack(preds, axis=0)
    return dict(prediction=P.mean(axis=0), uncertainty=P.std(axis=0, ddof=1), quantiles=None, sd_full=None,
                n_features=F.shape[1],
                info=dict(features="morgan_r2_2048 bits (raw 0/1)", hidden=list(MLP_HIDDEN), dropout=MLP_DROPOUT,
                          optimizer="Adam", lr=MLP_LR, batch_size=bs, target_scaling="training mean / SD",
                          stopping=f"max {MLP_MAX_EPOCHS} epochs; stop after {MLP_PATIENCE} epochs without "
                                   f"training-loss improvement > {MLP_TOL} (v1 / sklearn rule, no validation split)",
                          member_seeds=[100 * seed + k for k in range(n_members)], epochs=epochs,
                          torch=torch.__version__, torch_threads=threads,
                          rdkit_fp=features._rdkit_version(), numpy=np.__version__))


CHEMPROP_EPOCHS = 50


def fit_chemprop_ens(cell, seed, n_members=5, max_epochs=None):
    import logging
    import warnings
    max_epochs = int(max_epochs or CHEMPROP_EPOCHS)
    import lightning
    import lightning.pytorch as pl
    import torch
    import chemprop
    for name in ("lightning.pytorch", "lightning.fabric", "pytorch_lightning", "lightning"):
        logging.getLogger(name).setLevel(logging.ERROR)
    from chemprop import models, nn
    from chemprop.data import MoleculeDatapoint, MoleculeDataset, build_dataloader
    from chemprop.nn.transforms import UnscaleTransform
    y = checked_fit_labels(cell.y_train, cell.size)
    smis_tr = [str(s) for s in cell.df.smiles.values[cell.train_rows]]
    smis_te = [str(s) for s in cell.smiles_test]
    preds, epoch_s = [], []
    for k in range(n_members):
        ms = 100 * seed + k
        pl.seed_everything(ms)
        tr = MoleculeDataset([MoleculeDatapoint.from_smi(s, np.array([v], dtype=float)) for s, v in zip(smis_tr, y)])
        te = MoleculeDataset([MoleculeDatapoint.from_smi(s, np.array([0.0])) for s in smis_te])  # no test labels
        scaler = tr.normalize_targets()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tl = build_dataloader(tr, batch_size=min(64, len(tr)), shuffle=True, seed=ms)
            el = build_dataloader(te, shuffle=False)
            model = models.MPNN(nn.BondMessagePassing(), nn.MeanAggregation(),
                                nn.RegressionFFN(output_transform=UnscaleTransform.from_standard_scaler(scaler)),
                                batch_norm=True)
            trainer = pl.Trainer(max_epochs=max_epochs, accelerator="cpu", devices=1, enable_progress_bar=False,
                                 enable_model_summary=False, enable_checkpointing=False, logger=False,
                                 deterministic=True)
            t0 = time.perf_counter()
            trainer.fit(model, tl)
            epoch_s.append((time.perf_counter() - t0) / max_epochs)
            out = trainer.predict(model, el)
        preds.append(torch.cat(list(out), dim=0).numpy().reshape(-1).astype(float))
    P = np.stack(preds, axis=0)
    return dict(prediction=P.mean(axis=0), uncertainty=P.std(axis=0, ddof=1), quantiles=None, sd_full=None,
                n_features=0,
                info=dict(features="molecular graph (chemprop 2 default featurizer)", n_members=n_members,
                          max_epochs=max_epochs, batch_size=min(64, len(y)), aggregation="mean", batch_norm=True,
                          deterministic=True,
                          member_seeds=[100 * seed + k for k in range(n_members)],
                          mean_epoch_s=float(np.mean(epoch_s)), chemprop=chemprop.__version__,
                          torch=torch.__version__, lightning=lightning.__version__,
                          torch_threads=torch.get_num_threads(), numpy=np.__version__))



# --------------------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------------------

def cell_path(method, ds, size, seed, draw=1):
    stem = f"{method}__{ds}__N{size}" + ("" if seed == 0 else f"__s{seed}") + ("" if draw == 1 else f"__d{draw}")
    return os.path.join(OUT, stem + ".csv")


def run_cell(method, cell, seed, device="auto", tabpfn_version="v3.5", chemprop_epochs=None):
    t0 = time.perf_counter()
    if method == "tabpfn":
        r = fit_tabpfn(cell, seed, device, tabpfn_version)
    elif method == "rf":
        r = fit_rf(cell, seed)
    elif method == "rf_desc":
        r = fit_rf_desc(cell, seed)
    elif method == "trivial":
        r = fit_trivial(cell, seed)
    elif method == "gp_tanimoto":
        r = fit_gp_tanimoto(cell, seed)
    elif method == "knn_tanimoto":
        r = fit_knn_tanimoto(cell, seed)
    elif method == "mlp_ens":
        r = fit_mlp_ens(cell, seed)
    elif method == "chemprop_ens":
        r = fit_chemprop_ens(cell, seed, max_epochs=chemprop_epochs)
    else:
        raise ValueError(method)
    wall = time.perf_counter() - t0
    n = len(cell.test_ids)
    q = r["quantiles"] if r["quantiles"] is not None else np.full((n, len(QCOLS)), np.nan)
    sd = r["sd_full"] if r["sd_full"] is not None else np.full(n, np.nan)
    D = pd.DataFrame(dict(dataset=cell.ds, method=method, size=cell.size, seed=seed,
                          mol_id=cell.test_ids, smiles=cell.smiles_test, truth=cell.truth,
                          prediction=r["prediction"], uncertainty=r["uncertainty"]))
    for j, c in enumerate(QCOLS):
        D[c] = q[:, j]
    D["sd_full"] = sd
    D["wall_s"] = wall
    cols = list(COLUMNS)
    if cell.draw != 1:                             # draw-1 files keep their original columns
        D["draw"] = cell.draw
        cols.insert(cols.index("seed") + 1, "draw")
    assert len(D) == n and np.all(np.isfinite(D.prediction)) and np.all(np.isfinite(D.uncertainty)), \
        f"non-finite output in {method} {cell.ds} N={cell.size}"
    meta = dict(dataset=cell.ds, method=method, size=cell.size, seed=seed, draw=cell.draw, wall_s=wall,
                n_train=len(cell.train_ids), n_test=n, n_features=r["n_features"],
                train_ids=cell.train_ids, test_ids=cell.test_ids, host=platform.node(),
                python=platform.python_version(), **r["info"],
                sigma_definition={"tabpfn": "(q84 - q16) / 2 of the predictive distribution",
                                  "rf": "std over the 200 tree predictions (ddof=1)",
                                  "rf_desc": "std over the 200 tree predictions (ddof=1)",
                                  "trivial": "training std (ddof=1)",
                                  "gp_tanimoto": "GP posterior predictive SD incl. noise, sqrt(var_f + s2)",
                                  "mlp_ens": "SD over the 5 MLP members (ddof=1)",
                                  "chemprop_ens": "SD over the 5 D-MPNN members (ddof=1)",
                                  "knn_tanimoto": "max(SD of the 5 neighbour labels (ddof=1), 0.01)"}[method],
                prediction_definition={"tabpfn": "mean of the predictive distribution",
                                       "rf": "mean over trees", "rf_desc": "mean over trees",
                                       "trivial": "training mean", "gp_tanimoto": "GP posterior mean",
                                       "mlp_ens": "mean over members", "chemprop_ens": "mean over members",
                                       "knn_tanimoto": "Tanimoto-weighted mean of the 5 nearest labels"}[method])
    return D[cols], meta


def parse_list(s, all_values):
    if s in (None, "", "all"):
        return list(all_values)
    return [x.strip() for x in s.split(",") if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--methods", default="tabpfn,rf,rf_desc,trivial")
    ap.add_argument("--datasets", default="all")
    ap.add_argument("--sizes", default="all", help="'all' = the split's ladder, or e.g. 30,1000; "
                    "'max' = the top rung; sizes above the pool are capped to the ladder's top rung")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--tabpfn-version", default="v3.5")
    ap.add_argument("--draw", type=int, default=1, help="context draw: 1 = main split, 2 / 3 = <ds>_v2_draw<d>.json")
    ap.add_argument("--chemprop-epochs", type=int, default=None,
                    help=f"chemprop_ens epochs (default {CHEMPROP_EPOCHS}; recorded in the meta)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    methods = parse_list(a.methods, METHODS)
    for m in methods:
        assert m in METHODS, f"unknown method {m}"
    dsets = parse_list(a.datasets, DATASET_KEYS)
    seeds = [int(s) for s in a.seeds.split(",")]
    os.makedirs(OUT, exist_ok=True)
    timing = []
    for ds in dsets:
        df, split = load_dataset(ds), load_split(ds, a.draw)
        ladder = [int(s) for s in split["sizes"]]
        if a.sizes == "all":
            sizes = ladder
        elif a.sizes == "max":
            sizes = [ladder[-1]]
        elif a.sizes == "min":
            sizes = [ladder[0]]
        else:
            sizes = sorted({min(int(s), ladder[-1]) for s in a.sizes.split(",")})
        for size in sizes:
            cell = Cell(ds, size, df, split, draw=a.draw)
            for method in methods:
                for seed in (seeds if method != "trivial" else [0]):
                    path = cell_path(method, ds, size, seed, a.draw)
                    if os.path.exists(path) and not a.force:
                        print(f"skip {os.path.basename(path)} (exists)")
                        continue
                    D, meta = run_cell(method, cell, seed, a.device, a.tabpfn_version, a.chemprop_epochs)
                    D.to_csv(path, index=False)
                    with open(path[:-4] + ".meta.json", "w", encoding="utf8") as f:
                        json.dump(meta, f, indent=1)
                    rmse = float(np.sqrt(np.mean((D.prediction - D.truth) ** 2)))
                    print(f"{method:12s} {ds:14s} N={size:5d} seed={seed} draw={a.draw}  RMSE={rmse:.3f}  "
                          f"mean_sigma={D.uncertainty.mean():.3f}  wall={meta['wall_s']:.1f}s"
                          + (f"  [{meta.get('device')}]" if method == "tabpfn" else ""), flush=True)
                    timing.append(dict(method=method, dataset=ds, size=size, seed=seed, wall_s=meta["wall_s"]))
    if timing:
        T = pd.DataFrame(timing)
        print("\nwall-clock (s) per method:")
        print(T.groupby("method").wall_s.agg(["count", "sum", "max"]).round(1).to_string())


if __name__ == "__main__":
    main()
