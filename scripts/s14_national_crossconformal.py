"""National PACD v2: cross-conformal calibration on all natural sites + land-use families.

Why: with split conformal (n_cal ~ 1,200) and m ~ 1,765 tests, the smallest attainable p-value
1/(n_cal+1) exceeds the BH rank-1 threshold q/m, so BH needs >= ~15 sites beating every
calibration score before it can reject anything (s09 returned no discoveries).
Fix:
  - K = 5 spatial-group folds over natural sites. Fold-k anchor/scale models are fitted
    without fold k; natural sites in fold k get calibration scores from the fold-k model,
    and every test site is scored by all K models.
  - Reference scores are the pooled out-of-fold natural-site scores; test sites use the mean
    score over the K fold models (an approximation to cross-conformal, Vovk 2015; the exact
    fold-wise version is kept in cross_conformal_p but not used).
  - The reference is self-cleaned (pacd.clean_reference) before calibration; "plain conformal"
    and "cleaned + EVT" variants are both written; s16 adds the cleaned-only main variant.
  - BH within pre-specified land-use families (urban, cropland, pasture, orchard). Natural sites
    form their own family, each fold tested against the other folds' reference scores.
Signature and agnostic scores use the same fold-wise construction.
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
from s06_injection_v2 import SIGS, ELS
from pacd import conf_p, evt_p, bh, clean_reference, CLEAN_V1
from s09_national_discovery import landuse

Q = 0.10
K = 5
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1)


def fold_scores(el, df, X, g, nat_idx, folds):
    """Return S[k] = full-length score vector from the fold-k model (NaN where not defined)."""
    y, cens = target(df, el)
    y = np.where(cens, y - np.log10(2), y)
    S = np.full((K, len(y)), np.nan)
    for k, (tr, te) in enumerate(folds):
        a = nat_idx[tr]
        f = lgb.LGBMRegressor(**LGB).fit(X.iloc[a], y[a])
        oof = np.zeros(len(a))
        for i, j in GroupKFold(3).split(a, groups=g[a]):
            oof[j] = y[a[j]] - lgb.LGBMRegressor(**LGB).fit(X.iloc[a[i]], y[a[i]]).predict(X.iloc[a[j]])
        s = np.clip(lgb.LGBMRegressor(**{**LGB, "n_estimators": 150}).fit(X.iloc[a], np.abs(oof)).predict(X), 0.03, None)
        S[k] = (y - f.predict(X)) / s
    return el, S, y


def cross_conformal_p(S, cal_sets, test_idx):
    n = sum(len(c) for c in cal_sets)
    cnt = np.zeros(len(test_idx))
    for k, c in enumerate(cal_sets):
        cs = np.sort(S[k, c])
        cnt += len(cs) - np.searchsorted(cs, S[k, test_idx], side="left")
    return (1 + cnt) / (n + 1)


if __name__ == "__main__":
    df = load()
    df = df[df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()].copy()
    df["landuse"] = landuse(df)
    df = df[df["landuse"] != "other"]
    nat = (df["landuse"] == "natural").to_numpy()
    nat_idx = np.nonzero(nat)[0]
    X = dual_features(df)
    g = spatial_groups(df, 150)
    folds = list(GroupKFold(K).split(nat_idx, groups=g[nat_idx]))
    cal_sets = [nat_idx[te] for _, te in folds]
    res = Parallel(n_jobs=6)(delayed(fold_scores)(el, df, X, g, nat_idx, folds) for el in ELS)
    S = {el: s for el, s, _ in res}
    Y = {el: y for el, _, y in res}
    for sig, els in SIGS.items():
        S[f"sig_{sig}"] = np.mean([S[e] for e in els], axis=0)
    S["any_max"] = np.max([S[e] for e in ELS], axis=0)
    out = df[["Latitude", "Longitude", "StateID", "LandCover2", "landuse"] + [f"Top5_{e}" for e in ELS]].copy()
    fold_of = np.full(len(df), -1)
    for k, c in enumerate(cal_sets):
        fold_of[c] = k
    fam_rows = []
    for name, Sk in S.items():
        oof = np.array([Sk[fold_of[i], i] if fold_of[i] >= 0 else np.nan for i in range(len(df))])
        sbar = np.nanmean(Sk, axis=0)  # test sites: mean score over the K fold models
        for variant in ["plain conformal", "cleaned + EVT (PACD)"]:
            p = np.full(len(df), np.nan)
            disc = np.zeros(len(df), bool)
            # natural family (negative control + legacy-contamination search): leave-fold-out
            for k, c in enumerate(cal_sets):
                ref = oof[np.concatenate([cc for j, cc in enumerate(cal_sets) if j != k])]
                if variant == "plain conformal":
                    p[c] = conf_p(ref, oof[c])
                else:
                    p[c] = evt_p(ref[clean_reference(ref, **CLEAN_V1)], oof[c])
            disc[nat_idx] = bh(p[nat_idx], Q)
            ref = oof[nat_idx]
            keep = clean_reference(ref, **CLEAN_V1) if variant != "plain conformal" else np.ones(len(ref), bool)
            for fam in ["urban", "cropland", "pasture", "orchard"]:
                idx = np.nonzero((df["landuse"] == fam).to_numpy())[0]
                p[idx] = conf_p(ref, sbar[idx]) if variant == "plain conformal" else evt_p(ref[keep], sbar[idx])
                disc[idx] = bh(p[idx], Q)
            tag = "plain" if variant == "plain conformal" else "pacd"
            out[f"p_{tag}_{name}"] = p
            out[f"disc_{tag}_{name}"] = disc
            for fam in ["natural", "urban", "cropland", "pasture", "orchard"]:
                m = (df["landuse"] == fam).to_numpy()
                fam_rows.append(dict(score=name, variant=variant, family=fam, n=int(m.sum()),
                                     n_disc=int(disc[m].sum()), rate=disc[m].mean(), min_p=np.nanmin(p[m]),
                                     ref_removed=int((~keep).sum())))
        out[f"score_{name}"] = np.where(nat, oof, sbar)
    out.to_csv(RES / "s14_site_pvalues_discoveries.csv")
    fam = pd.DataFrame(fam_rows)
    fam.to_csv(RES / "s14_discoveries_by_family.csv", index=False)
    print(fam.pivot_table(index="score", columns=["variant", "family"], values="n_disc").to_string())
    for name in ["Pb", "Zn", "Cu", "As", "P", "sig_urban", "sig_agricultural", "sig_mining", "any_max"]:
        d = out[out[f"disc_pacd_{name}"]]
        if len(d):
            print(name, len(d), d.groupby(["landuse", "StateID"]).size().sort_values(ascending=False).head(8).to_dict())
