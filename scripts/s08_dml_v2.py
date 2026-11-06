"""Profile-anchored DML v2: refined land-use treatments, overlap trimming, placebo tests.

Treatments (vs natural cover: forest, shrubland, grassland, barren), from NLCD 1992 codes:
  cropland = Row crops + Small grains + Fallow; pasture = Pasture/Hay;
  urban    = Developed + Urban/Recreational Grasses.
Primary outcome: log10(Top5 X / Top5 Ti) (removes organic-matter dilution); raw log10 X as SI.
Confounders: C-horizon chemistry, XRD, C depth, lat/lon (pre-treatment parent material).
ATT by cross-fitted AIPW; spatial-group folds; propensity clipped to [0.01, 0.99] and sites
outside the control support (e > 0.95) dropped; cluster-robust SE by spatial group.
Placebo: treatment labels permuted within spatial groups (5 draws) -> ATT should be ~0.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from common import *

OUTCOMES = ["As", "Cd", "Co", "Cr", "Cu", "Hg", "Mo", "Ni", "P", "Pb", "Sb", "Se", "Sn", "Tl",
            "U", "V", "Zn", "Mn", "S", "Ca", "Mg", "K", "Na", "Ba", "Sr", "Li", "Bi", "W", "In"]
NEG_CTRL = ["Nb", "Th", "Sc", "La", "Ce", "Y", "Ga", "Al", "Fe"]
TREAT = {"cropland": lambda d: d["LandCover2"].isin(["Row Crops", "Row crops", "Small Grains", "Fallow"]),
         "pasture": lambda d: d["LandCover2"].eq("Pasture/Hay"),
         "urban": lambda d: d["LandCover1"].eq("Developed") | d["LandCover2"].eq("Urban/Recreational Grasses")}
LGB_R = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=30,
             subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1)
LGB_C = dict(LGB_R, objective="binary")


def propensity(d, X, folds):
    e = np.zeros(len(d))
    for tr, te in folds:
        e[te] = lgb.LGBMClassifier(**LGB_C).fit(X.iloc[tr], d[tr]).predict_proba(X.iloc[te])[:, 1]
    return np.clip(e, 0.01, 0.99)


def att_aipw(y, d, X, e, folds, groups):
    ok = np.isfinite(y)
    m0 = np.full(len(y), np.nan)
    for tr, te in folds:
        t0 = tr[(d[tr] == 0) & ok[tr]]
        m0[te] = lgb.LGBMRegressor(**LGB_R).fit(X.iloc[t0], y[t0]).predict(X.iloc[te])
    y, d, e, m0, groups = y[ok], d[ok], e[ok], m0[ok], groups[ok]
    p = d.mean()
    psi = (d * (y - m0) - (1 - d) * e / (1 - e) * (y - m0)) / p
    att = psi.mean()
    infl = psi - d * att / p
    se = np.sqrt((pd.Series(infl).groupby(groups).sum() ** 2).sum()) / len(y)
    return att, se, y[d == 1].mean() - y[d == 0].mean()


def outcome(sub, el, norm):
    y, cens = target(sub, el)
    y = np.where(cens.to_numpy(), y.to_numpy() - np.log10(2), y.to_numpy())
    if norm == "Ti-normalised":
        y = y - np.log10(sub["Top5_Ti"].to_numpy())
    return y, float(cens.mean())


def one(treat, el, norm, sub, d, X, e, folds, g, placebo=None):
    y, cf = outcome(sub, el, norm)
    dd = d if placebo is None else placebo
    if placebo is not None:  # labels permuted within groups: propensity = group treated share
        e = np.clip(pd.Series(dd).groupby(g).transform("mean").to_numpy(), 0.01, 0.99)
    att, se, naive = att_aipw(y, dd, X, e, folds, g)
    return dict(treatment=treat, element=el, outcome=norm, placebo=placebo is not None,
                role="negative control" if el in NEG_CTRL else "target",
                ATT_log10=att, SE=se, z=att / se, ATT_pct=(10 ** att - 1) * 100,
                lo_pct=(10 ** (att - 1.96 * se) - 1) * 100, hi_pct=(10 ** (att + 1.96 * se) - 1) * 100,
                naive_pct=(10 ** naive - 1) * 100, cens_frac=cf,
                n_treated=int(dd.sum()), n_control=int((1 - dd).sum()))


if __name__ == "__main__":
    df = load()
    df = df[df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    natural = df["LandCover1"].isin(NATURAL)
    rows, diag = [], []
    rng = np.random.default_rng(7)
    for treat, fn in TREAT.items():
        tmask = fn(df)
        sub = df[natural | tmask].copy()
        d = tmask[sub.index].astype(int).to_numpy()
        X = anchor_features(sub, with_xy=True)
        g = spatial_groups(sub, 120)
        folds = list(GroupKFold(5).split(X, d, g))
        e = propensity(d, X, folds)
        keep = ~((d == 1) & (e > 0.95))  # treated sites without comparable natural controls
        sub, d, X, g, e = sub[keep], d[keep], X[keep].reset_index(drop=True), g[keep], e[keep]
        folds = list(GroupKFold(5).split(X, d, g))
        diag.append(dict(treatment=treat, n_treated=int(d.sum()), n_control=int((1 - d).sum()),
                         dropped_no_overlap=int((~keep).sum()),
                         e_treated_p05=np.percentile(e[d == 1], 5), e_treated_p50=np.median(e[d == 1]),
                         e_treated_p95=np.percentile(e[d == 1], 95)))
        pd.DataFrame({"SiteID": sub.index, "d": d, "e": e}).to_csv(RES / f"s08_propensity_{treat}.csv", index=False)
        tasks = [(el, nm, None) for el in OUTCOMES + NEG_CTRL for nm in ["Ti-normalised", "raw"]]
        for b in range(5):
            perm = d.copy()
            for gg in np.unique(g):
                ix = np.nonzero(g == gg)[0]
                perm[ix] = rng.permutation(perm[ix])
            tasks += [(el, "Ti-normalised", perm) for el in ["P", "Pb", "Cu", "Zn", "U", "Hg", "Cd", "Nb", "Th"]]
        rows += Parallel(n_jobs=8)(delayed(one)(treat, el, nm, sub, d, X, e, folds, g, pl)
                                   for el, nm, pl in tasks)
        print(treat, "done", diag[-1], flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "s08_dml_v2_att.csv", index=False)
    pd.DataFrame(diag).to_csv(RES / "s08_overlap_diagnostics.csv", index=False)
    main = res[~res.placebo & (res.outcome == "Ti-normalised")]
    print(main.pivot_table(index=["role", "element"], columns="treatment", values="ATT_pct").round(1).to_string())
    print(main.pivot_table(index=["role", "element"], columns="treatment", values="z").round(1).to_string())
    pl = res[res.placebo]
    print("placebo |z|>1.96 share:", (pl.z.abs() > 1.96).mean().round(3), "mean |ATT%|:", pl.ATT_pct.abs().mean().round(2))
