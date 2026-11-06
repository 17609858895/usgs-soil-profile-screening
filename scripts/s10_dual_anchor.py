"""Dual anchoring: vertical anchor (C horizon) + in-sample immobile anchor (topsoil lithogenic
elements whose land-use ATT is ~0 in the DML negative-control test).

Spatial-block CV R2 on natural sites for each feature set, and the residual correlation matrix
across contaminant elements (lower correlation -> larger gain from multi-element signatures).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent))
from common import *

IMMOBILE = ["Al", "Ti", "Nb", "Th", "Sc", "La", "Ce", "Y", "Ga"]
ELS = ["As", "Cd", "Cr", "Cu", "Hg", "Ni", "P", "Pb", "Sb", "Se", "U", "Zn"]
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1)


def immobile_features(df):
    X = pd.DataFrame(index=df.index)
    for el in IMMOBILE:
        X[f"Top5_{el}"], _ = log_el(df, "Top5", el)
    return X


if __name__ == "__main__":
    df = load()
    nat = df[df["LandCover1"].isin(NATURAL) & df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    XC = anchor_features(nat, with_xy=True)
    XI = immobile_features(nat)
    sets = {"Profile (C horizon)": XC, "Immobile topsoil only": pd.concat([XI, XC[["lat", "lon"]]], axis=1),
            "Dual (C horizon + immobile topsoil)": pd.concat([XC, XI], axis=1)}
    g = spatial_groups(nat, 120)
    folds = list(GroupKFold(5).split(XC, g, g))
    def one(el):
        y, cens = target(nat, el)
        y = np.where(cens, y - np.log10(2), y)
        det = ~cens.to_numpy()
        pr = np.zeros(len(y))
        for tr, te in folds:
            pr[te] = LinearRegression().fit(XC[[f"C_{el}"]].iloc[tr], y[tr]).predict(XC[[f"C_{el}"]].iloc[te])
        r, res = dict(element=el, R2_ratio=r2_score(y[det], pr[det])), {(el, "Ratio"): y - pr}
        for name, X in sets.items():
            p = np.zeros(len(y))
            for tr, te in folds:
                p[te] = lgb.LGBMRegressor(**LGB).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
            r[f"R2_{name}"] = r2_score(y[det], p[det])
            res[(el, name)] = y - p
        return r, res

    outs = Parallel(n_jobs=8)(delayed(one)(el) for el in ELS)
    rows = [o[0] for o in outs]
    resid = {k: v for o in outs for k, v in o[1].items()}
    out = pd.DataFrame(rows)
    out.to_csv(RES / "s10_dual_anchor_R2.csv", index=False)
    print(out.round(3).to_string(index=False))
    for name in ["Ratio", "Profile (C horizon)", "Dual (C horizon + immobile topsoil)"]:
        R = pd.DataFrame({el: resid[(el, name)] for el in ELS}).corr()
        R.to_csv(RES / f"s10_residual_corr_{name.split(' ')[0]}.csv")
        off = R.to_numpy()[np.triu_indices(len(ELS), 1)]
        print(name, "mean off-diagonal residual corr:", off.mean().round(3))
