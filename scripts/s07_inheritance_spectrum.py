"""Geochemical inheritance spectrum: share of topsoil (0-5 cm) variance explained by the
site's own parent material (C horizon) under spatial-block CV, for every analysed element.

Two anchors per element (natural land-cover sites, detected outer-test values only):
  ratio  : log Top5 ~ log C (same element)      -> conventional enrichment-factor logic
  profile: LightGBM on full C chemistry + XRD + depth + lat/lon
Inheritance index = R2_profile; the profile gain = R2_profile - R2_ratio.
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

LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1)
df = load()
nat = df[df["LandCover1"].isin(NATURAL) & df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
X = anchor_features(nat, with_xy=True)
g = spatial_groups(nat, 120)
folds = list(GroupKFold(5).split(X, g, g))


def one(el):
    y, cens = target(nat, el)
    y = np.where(cens, y - np.log10(2), y)
    det = ~cens.to_numpy()
    pr, pp = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in folds:
        pr[te] = LinearRegression().fit(X[[f"C_{el}"]].iloc[tr], y[tr]).predict(X[[f"C_{el}"]].iloc[te])
        pp[te] = lgb.LGBMRegressor(**LGB).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    return dict(element=el, cens_frac=1 - det.mean(), R2_ratio=r2_score(y[det], pr[det]),
                R2_profile=r2_score(y[det], pp[det]),
                slope_ratio=LinearRegression().fit(X[[f"C_{el}"]][det], y[det]).coef_[0])


rows = Parallel(n_jobs=8)(delayed(one)(el) for el in [e for e in ELEMENTS if e not in DROP_EL])
out = pd.DataFrame(rows)
out["gain"] = out["R2_profile"] - out["R2_ratio"]
out.sort_values("R2_profile", ascending=False).to_csv(RES / "s07_inheritance_spectrum.csv", index=False)
print(out.sort_values("R2_profile", ascending=False).round(3).to_string(index=False))
