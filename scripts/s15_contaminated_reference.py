"""Validation: PACD when the reference (calibration) set itself contains contaminated sites.

Mirrors the real data: the most Pb/Zn-enriched sites of the survey lie in natural land cover
(legacy mining/smelting). Per rep (dual anchor, spatial 40/40/20 split):
  - 3 % of calibration sites receive a strong enrichment (k_ref = 10);
  - 10 % of test sites receive k in {3, 5, 10};
  - methods: plain conformal | EVT-extended | self-cleaned reference | cleaned + EVT.
Reported: test-set FDR and power; also the k = 1 (all-null test set) false-discovery count.
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
from pacd import conf_p, evt_p, bh, clean_reference, CLEAN_V1

ELS = ["As", "Cd", "Cr", "Cu", "Hg", "Ni", "P", "Pb", "Sb", "U", "Zn"]
SIGS = {"urban": ["Pb", "Zn", "Cu", "Sb"], "mining": ["As", "Sb", "Pb", "Zn", "Cd", "Hg"]}
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1)


def ev(flag, truth):
    d = flag.sum()
    return dict(n_disc=int(d), FDR=((flag & ~truth).sum() / d) if d else 0.0,
                power=(flag & truth).sum() / max(truth.sum(), 1))


def run(rep, nat, X, g):
    rng = np.random.default_rng(300 + rep)
    ug = rng.permutation(np.unique(g))
    n = len(ug)
    tr = np.isin(g, ug[: int(.4 * n)])
    ca = np.isin(g, ug[int(.4 * n): int(.8 * n)])
    te = ~(tr | ca)
    Z0 = {}
    for el in ELS:
        y, cens = target(nat, el)
        y = np.where(cens, y - np.log10(2), y)
        f = lgb.LGBMRegressor(**LGB).fit(X[tr], y[tr]).predict(X)
        oof = np.zeros(tr.sum())
        for a, b in GroupKFold(3).split(X[tr], y[tr], g[tr]):
            oof[b] = y[tr][b] - lgb.LGBMRegressor(**LGB).fit(X[tr].iloc[a], y[tr][a]).predict(X[tr].iloc[b])
        s = np.clip(lgb.LGBMRegressor(**{**LGB, "n_estimators": 150}).fit(X[tr], np.abs(oof)).predict(X), 0.03, None)
        Z0[el] = (y - f, s)
    ca_idx, te_idx = np.nonzero(ca)[0], np.nonzero(te)[0]
    rows = []
    targets = [(el, [el]) for el in ELS] + [(f"sig:{k}", v) for k, v in SIGS.items()]
    for name, els in targets:
        for k in [1, 3, 5, 10]:
            ref_bad = rng.choice(ca_idx, int(.03 * len(ca_idx)), replace=False)
            inj = rng.choice(te_idx, int(.10 * len(te_idx)), replace=False) if k > 1 else np.array([], int)
            Zs = []
            for el in els:
                r, s = Z0[el]
                r = r.copy()
                r[ref_bad] += np.log10(10)
                r[inj] += np.log10(k)
                Zs.append(r / s)
            Z = np.mean(Zs, axis=0)
            truth = np.isin(te_idx, inj)
            cal, tst = Z[ca], Z[te]
            keep = clean_reference(cal, use_evt=True, **CLEAN_V1)
            ps = {"plain conformal": conf_p(cal, tst), "EVT-extended": evt_p(cal, tst),
                  "self-cleaned reference": conf_p(cal[keep], tst),
                  "cleaned + EVT (PACD)": evt_p(cal[keep], tst)}
            for mname, p in ps.items():
                rows.append(dict(rep=rep, target=name, k=k, method=mname,
                                 ref_removed=int((~keep).sum()),
                                 ref_bad_recovered=float(np.isin(ca_idx[~keep], ref_bad).sum() / len(ref_bad)),
                                 **ev(bh(p), truth)))
    return rows


if __name__ == "__main__":
    df = load()
    nat = df[df["LandCover1"].isin(NATURAL) & df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    X = dual_features(nat)
    g = spatial_groups(nat, 120)
    out = Parallel(n_jobs=6)(delayed(run)(r, nat, X, g) for r in range(6))
    res = pd.DataFrame([row for rows in out for row in rows])
    res.to_csv(RES / "s15_contaminated_reference_raw.csv", index=False)
    res["kind"] = np.where(res.target.str.startswith("sig"), "signature", "single")
    print(res.groupby(["kind", "k", "method"])[["FDR", "power", "n_disc", "ref_removed", "ref_bad_recovered"]].mean().round(3).to_string())
