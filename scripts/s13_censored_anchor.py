"""Censoring-aware (survival-ML) anchors for detection-limited elements.

Anchors (all on dual features, spatial-group 5-fold CV, natural sites):
  DL/2 substitution + LightGBM  |  drop non-detects + LightGBM  |  XGBoost-AFT (interval-censored
  likelihood; left-censored = (0, DL]) with normal and logistic error.
(A) Synthetic censoring with known truth: fully detected Pb, Cu, Zn, As are censored at their
    30/50/70 % quantiles in training labels; R2/RMSE against the hidden true values on all
    outer-test sites, plus RMSE on the censored test sites only.
(B) Real non-detects (Se, Cd, Hg, In, S): left-censored concordance (Harrell C on -log y, with
    non-detects as censored) and R2 on detected test values.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
sys.path.insert(0, str(Path(__file__).parent))
from common import *

LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1)


def aft(Xtr, lo, hi, Xte, dist):
    d = xgb.DMatrix(Xtr, missing=np.nan)
    d.set_float_info("label_lower_bound", lo)
    d.set_float_info("label_upper_bound", hi)
    p = dict(objective="survival:aft", eval_metric="aft-nloglik", aft_loss_distribution=dist,
             aft_loss_distribution_scale=0.5, tree_method="hist", learning_rate=0.05, max_depth=5,
             subsample=0.8, colsample_bytree=0.7, min_child_weight=3, nthread=1, verbosity=0)
    b = xgb.train(p, d, num_boost_round=400)
    return np.log10(b.predict(xgb.DMatrix(Xte, missing=np.nan)))


def fit_all(X, yt, cens, dl, folds):
    """yt: log10 label (DL for censored); returns OOF predictions per method (log10)."""
    out = {m: np.zeros(len(yt)) for m in ["DL/2 + LightGBM", "Drop non-detects + LightGBM",
                                          "XGB-AFT normal", "XGB-AFT logistic"]}
    for tr, te in folds:
        y_sub = np.where(cens[tr], dl[tr] - np.log10(2), yt[tr])
        out["DL/2 + LightGBM"][te] = lgb.LGBMRegressor(**LGB).fit(X.iloc[tr], y_sub).predict(X.iloc[te])
        k = tr[~cens[tr]]
        out["Drop non-detects + LightGBM"][te] = lgb.LGBMRegressor(**LGB).fit(X.iloc[k], yt[k]).predict(X.iloc[te])
        lo = np.where(cens[tr], 0.0, 10 ** yt[tr])
        hi = 10 ** np.where(cens[tr], dl[tr], yt[tr])
        for dist in ["normal", "logistic"]:
            out[f"XGB-AFT {dist}"][te] = aft(X.iloc[tr], lo, hi, X.iloc[te], dist)
    return out


def cindex_left(pred, y, cens):
    """Harrell C for left-censored data: flip sign so non-detects become right-censored."""
    from sksurv.metrics import concordance_index_censored
    return concordance_index_censored(~cens, -y, pred)[0]  # higher pred = "earlier event" on the flipped axis


def synthetic(el, q, X, nat, folds):
    y, c0 = target(nat, el)
    y = y.to_numpy()
    keep = ~c0.to_numpy()
    dl_val = np.quantile(y[keep], q)
    cens = y < dl_val
    dl = np.full(len(y), dl_val)
    yt = np.where(cens, dl_val, y)
    preds = fit_all(X, yt, cens, dl, folds)
    rows = []
    for m, p in preds.items():
        rows.append(dict(part="synthetic", element=el, cens_level=q, method=m,
                         R2_true=r2_score(y[keep], p[keep]),
                         RMSE_true=np.sqrt(np.mean((y[keep] - p[keep]) ** 2)),
                         RMSE_censored_sites=np.sqrt(np.mean((y[cens & keep] - p[cens & keep]) ** 2))))
    return rows


def real(el, X, nat, folds):
    y, cens = target(nat, el)
    y, cens = y.to_numpy(), cens.to_numpy()
    preds = fit_all(X, y, cens, y.copy(), folds)
    return [dict(part="real", element=el, cens_level=cens.mean(), method=m,
                 C_index=cindex_left(p, y, cens), R2_detected=r2_score(y[~cens], p[~cens]))
            for m, p in preds.items()]


if __name__ == "__main__":
    df = load()
    nat = df[df["LandCover1"].isin(NATURAL) & df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    X = dual_features(nat)
    g = spatial_groups(nat, 120)
    folds = list(GroupKFold(5).split(X, g, g))
    jobs = [delayed(synthetic)(el, q, X, nat, folds) for el in ["Pb", "Cu", "Zn", "As"] for q in [0.3, 0.5, 0.7]]
    jobs += [delayed(real)(el, X, nat, folds) for el in ["Se", "Cd", "Hg", "In", "S"]]
    res = pd.DataFrame([r for rows in Parallel(n_jobs=8)(jobs) for r in rows])
    res.to_csv(RES / "s13_censored_anchor.csv", index=False)
    syn = res[res.part == "synthetic"]
    print(syn.pivot_table(index=["element", "cens_level"], columns="method", values="RMSE_censored_sites").round(3).to_string())
    print(syn.pivot_table(index=["element", "cens_level"], columns="method", values="R2_true").round(3).to_string())
    print(res[res.part == "real"].pivot_table(index="element", columns="method", values="C_index").round(3).to_string())
