"""20-algorithm benchmark + nested spatial stacking for the profile anchor f(C horizon) -> topsoil.

Reference population: natural land-cover sites. Outer 5-fold spatial-group CV; stacking uses
inner 3-fold spatial-group CV on each outer-training set only (no outer-test labels touched).
Non-detects in the target: DL/2 for training, scored on detected outer-test sites only.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet, BayesianRidge,
                                  HuberRegressor)
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor, AdaBoostRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import xgboost as xgb, lightgbm as lgb, catboost as cb
sys.path.insert(0, str(Path(__file__).parent))
from common import *

warnings.filterwarnings("ignore")
ELS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["Pb", "As", "Cd", "Ni", "Zn", "Hg"]


def lin(m):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), m)


def tree(m):
    return make_pipeline(SimpleImputer(strategy="median"), m)


def models():
    return {
        "OLS": lin(LinearRegression()),
        "Ridge": lin(Ridge(alpha=10.0)),
        "Lasso": lin(Lasso(alpha=0.003, max_iter=5000)),
        "ElasticNet": lin(ElasticNet(alpha=0.005, l1_ratio=0.5, max_iter=5000)),
        "BayesRidge": lin(BayesianRidge()),
        "Huber": lin(HuberRegressor(alpha=1e-3, max_iter=1000)),
        "PLS": lin(PLSRegression(n_components=12)),
        "KNN": lin(KNeighborsRegressor(n_neighbors=15, weights="distance")),
        "SVR": lin(SVR(C=1.0, epsilon=0.05, gamma="scale")),
        "KRR": lin(KernelRidge(alpha=1.0, kernel="rbf", gamma=1.0 / 300)),
        "GPR": lin(GaussianProcessRegressor(ConstantKernel(1.0) * Matern(length_scale=10.0, nu=1.5)
                                            + WhiteKernel(0.05), normalize_y=True, n_restarts_optimizer=0)),
        "CART": tree(DecisionTreeRegressor(max_depth=8, min_samples_leaf=10)),
        "RF": tree(RandomForestRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.4, n_jobs=1)),
        "ExtraTrees": tree(ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, max_features=0.5, n_jobs=1)),
        "AdaBoost": tree(AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=200, learning_rate=0.05)),
        "GBDT": tree(GradientBoostingRegressor(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.8)),
        "HistGB": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=31, l2_regularization=1.0),
        "XGBoost": xgb.XGBRegressor(n_estimators=500, learning_rate=0.03, max_depth=5, subsample=0.8,
                                    colsample_bytree=0.7, min_child_weight=3, reg_lambda=1.0, n_jobs=1, verbosity=0),
        "LightGBM": lgb.LGBMRegressor(n_estimators=400, learning_rate=0.04, num_leaves=31, min_child_samples=20,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1),
        "CatBoost": cb.CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, verbose=0, thread_count=1),
        "MLP": lin(MLPRegressor(hidden_layer_sizes=(128, 64), alpha=1e-2, learning_rate_init=1e-3,
                                max_iter=400, early_stopping=True, random_state=0)),
    }


def fit_pred(name, Xtr, ytr, Xte):
    t = time.time()
    m = models()[name].fit(Xtr, ytr)
    return np.ravel(m.predict(Xte)), np.ravel(m.predict(Xtr)), time.time() - t


def run_element(el, nat, X, g):
    y, cens = target(nat, el)
    y = np.where(cens, y - np.log10(2), y)
    det = ~cens.to_numpy()
    names = list(models())
    outer = list(GroupKFold(5).split(X, y, g))
    jobs = []
    for k, (tr, te) in enumerate(outer):
        jobs += [("outer", k, None, n, tr, te) for n in names]
        for j, (a, b) in enumerate(GroupKFold(3).split(X.iloc[tr], y[tr], g[tr])):
            jobs += [("inner", k, j, n, tr[a], tr[b]) for n in names]
    out = Parallel(n_jobs=10, verbose=0)(delayed(fit_pred)(n, X.iloc[a], y[a], X.iloc[b])
                                         for _, _, _, n, a, b in jobs)
    oof = {n: np.full(len(y), np.nan) for n in names}
    inner = {}
    rows = []
    for (kind, k, j, n, a, b), (p, ptr, sec) in zip(jobs, out):
        if kind == "outer":
            oof[n][b] = p
            rows.append(dict(element=el, model=n, fold=k, sec=sec,
                             R2_train=r2_score(y[a][det[a]], ptr[det[a]])))
        else:
            inner.setdefault(k, {}).setdefault(n, {})[j] = (b, p)
    # nested stacking: meta-learner fitted on inner OOF predictions of the outer-train set
    stack, top5 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for k, (tr, te) in enumerate(outer):
        Z = np.full((len(y), len(names)), np.nan)
        for c, n in enumerate(names):
            for j, (b, p) in inner[k][n].items():
                Z[b, c] = p
        Ztr = Z[tr]
        meta = LinearRegression(positive=True).fit(Ztr, y[tr])
        Zte = np.column_stack([oof[n][te] for n in names])
        stack[te] = meta.predict(Zte)
        inner_r2 = [r2_score(y[tr][det[tr]], Ztr[det[tr], c]) for c in range(len(names))]
        best = np.argsort(inner_r2)[-5:]
        top5[te] = Zte[:, best].mean(axis=1)
    oof["Stack-NNLS (nested)"] = stack
    oof["Top-5 mean (nested)"] = top5
    tr_df = pd.DataFrame(rows).groupby("model")[["R2_train", "sec"]].mean()
    res = []
    for n, p in oof.items():
        fold_r2 = [r2_score(y[te][det[te]], p[te][det[te]]) for _, te in outer]
        res.append(dict(element=el, model=n, R2=r2_score(y[det], p[det]),
                        RMSE=np.sqrt(mean_squared_error(y[det], p[det])),
                        MAE=mean_absolute_error(y[det], p[det]),
                        R2_fold_mean=np.mean(fold_r2), R2_fold_min=np.min(fold_r2),
                        R2_train=tr_df["R2_train"].get(n, np.nan), sec=tr_df["sec"].get(n, np.nan)))
    pd.DataFrame(oof).assign(SiteID=nat.index, y=y, detected=det).to_parquet(RES / f"s05_oof_{el}.parquet")
    return pd.DataFrame(res)


if __name__ == "__main__":
    df = load()
    nat = df[df["LandCover1"].isin(NATURAL) & df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    X = anchor_features(nat, with_xy=True)
    g = spatial_groups(nat, 120)
    allres = []
    for el in ELS:
        t = time.time()
        r = run_element(el, nat, X, g)
        allres.append(r)
        print(el, f"{time.time() - t:.0f}s", flush=True)
        print(r.sort_values("R2", ascending=False).round(3).to_string(index=False), flush=True)
        pd.concat(allres).to_csv(RES / "s05_benchmark20.csv", index=False)
