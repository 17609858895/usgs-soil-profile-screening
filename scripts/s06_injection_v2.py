"""Semi-synthetic validation v2 of profile-anchored conformal discovery (PACD).

Changes vs s03: train/cal/test = 40/40/20 % of spatial groups (larger calibration); k up to 10;
multi-element source signatures; agnostic max-residual score; covariate-shift scenario in which
null test sites are resampled to resemble cultivated land (propensity odds from s04), with
unweighted vs propensity-weighted conformal p-values.
Injection touches only test-site outcomes, so anchor predictions are computed once per rep.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from common import *

Q = 0.10
ELS = ["As", "Cd", "Cr", "Cu", "Hg", "Ni", "P", "Pb", "Sb", "Se", "U", "Zn"]
SIGS = {"urban": ["Pb", "Zn", "Cu", "Sb"], "agricultural": ["P", "Cd", "U", "Zn", "Cu"],
        "mining": ["As", "Sb", "Pb", "Zn", "Cd", "Hg"]}
KS = [1.5, 2, 3, 5, 10]
UCC = dict(As=4.8, Cd=0.09, Cr=92, Cu=28, Hg=0.05, Ni=47, P=655, Pb=17, Sb=0.4, Se=0.09, U=2.7, Zn=67)
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1)


def conf_p(cal, test, wc=None, wt=None):
    if wc is None:
        cal = np.sort(cal)
        return (1 + len(cal) - np.searchsorted(cal, test, side="left")) / (len(cal) + 1)
    o = np.argsort(cal)
    cs, ws = cal[o], wc[o]
    tail = np.concatenate([np.cumsum(ws[::-1])[::-1], [0.0]])
    ge = tail[np.searchsorted(cs, test, side="left")]
    return (ge + wt) / (ws.sum() + wt)


def bh(p, q=Q):
    m = len(p)
    o = np.argsort(p)
    ok = np.nonzero(p[o] <= q * np.arange(1, m + 1) / m)[0]
    rej = np.zeros(m, bool)
    if len(ok):
        rej[o[: ok.max() + 1]] = True
    return rej


def ev(flag, truth):
    d = flag.sum()
    return dict(n_disc=int(d), FDR=((flag & ~truth).sum() / d) if d else 0.0,
                power=(flag & truth).sum() / max(truth.sum(), 1))


def anchors(rep, nat, X, g):
    rng = np.random.default_rng(100 + rep)
    ug = rng.permutation(np.unique(g))
    n = len(ug)
    tr = np.isin(g, ug[: int(.4 * n)])
    ca = np.isin(g, ug[int(.4 * n): int(.8 * n)])
    te = ~(tr | ca)
    out = {"tr": tr, "ca": ca, "te": te}
    ti = np.log10(nat["Top5_Ti"].to_numpy()) - np.log10(nat["C_Ti"].to_numpy())
    for el in ELS:
        y, cens = target(nat, el)
        y = np.where(cens, y - np.log10(2), y)
        yC = X[f"C_{el}"].to_numpy()
        f = lgb.LGBMRegressor(**LGB).fit(X[tr], y[tr]).predict(X)
        oof = np.zeros(tr.sum())
        for a, b in GroupKFold(3).split(X[tr], y[tr], g[tr]):
            oof[b] = y[tr][b] - lgb.LGBMRegressor(**LGB).fit(X[tr].iloc[a], y[tr][a]).predict(X[tr].iloc[b])
        s = np.clip(lgb.LGBMRegressor(**{**LGB, "n_estimators": 150}).fit(X[tr], np.abs(oof)).predict(X), 0.03, None)
        lr = LinearRegression().fit(yC[tr, None], y[tr]).predict(yC[:, None])
        out[el] = dict(y=y, cens=cens.to_numpy(), f=f, s=s, lr=lr, ef=yC + ti)
    return out


def scenarios(rep, A, odds):
    rng = np.random.default_rng(900 + rep)
    ca, te = A["ca"], A["te"]
    te_idx = np.nonzero(te)[0]
    rows = []

    def score_set(Y, el):
        a = A[el]
        return {"Unconditional": Y, "EF (own C, Ti)": Y - a["ef"], "Ratio anchor": Y - a["lr"],
                "Profile anchor": Y - a["f"], "PACD (profile + scale)": (Y - a["f"]) / a["s"]}

    # (i) single-element injections
    for el in ELS:
        for k in KS:
            Y = A[el]["y"].copy()
            pick = rng.choice(te_idx, int(.10 * len(te_idx)), replace=False)
            inj = np.zeros(len(Y), bool)
            inj[pick] = True
            Y[inj] += np.log10(k)
            truth = inj[te]
            for name, s in score_set(Y, el).items():
                # left-censored calibration sites: score is an upper bound -> conservative p
                rows.append(dict(rep=rep, scenario="single", element=el, k=k, method=name,
                                 **ev(bh(conf_p(s[ca], s[te])), truth)))
            conc = 10 ** Y[te]
            mad = np.median(np.abs(Y[ca] - np.median(Y[ca]))) * 1.4826
            rules = {"Igeo > 1": conc > 3 * UCC[el],
                     "EF > 2": (Y - A[el]["ef"])[te] > np.log10(2),
                     "Median + 2MAD": Y[te] > np.median(Y[ca]) + 2 * mad}
            for name, flag in rules.items():
                rows.append(dict(rep=rep, scenario="single", element=el, k=k, method=name, **ev(flag, truth)))
    # (ii) multi-element signatures
    for sig, els in SIGS.items():
        for k in KS:
            pick = rng.choice(te_idx, int(.10 * len(te_idx)), replace=False)
            inj = np.zeros(len(te), bool)
            inj[pick] = True
            truth = inj[te]
            Z = {}
            for el in ELS:
                Y = A[el]["y"].copy()
                if el in els:
                    Y[inj] += np.log10(k)
                Z[el] = (Y - A[el]["f"]) / A[el]["s"]
            pmat = np.column_stack([conf_p(Z[el][ca], Z[el][te]) for el in els])
            cauchy = np.mean(np.tan((0.5 - pmat) * np.pi), axis=1)
            p_cauchy = 0.5 - np.arctan(cauchy) / np.pi
            s_sig = np.mean([Z[el] for el in els], axis=0)
            s_max = np.max([Z[el] for el in ELS], axis=0)
            ps = {"Best single element (Bonferroni)": np.minimum(pmat.min(1) * len(els), 1),
                  "Cauchy combination": p_cauchy,
                  "PACD signature score": conf_p(s_sig[ca], s_sig[te]),
                  "PACD agnostic max score": conf_p(s_max[ca], s_max[te])}
            for name, p in ps.items():
                rows.append(dict(rep=rep, scenario=f"signature:{sig}", element="+".join(els), k=k,
                                 method=name, **ev(bh(p), truth)))
    # (iii) covariate shift: null test sites resampled toward cultivated-land covariates
    pr = odds[te] / odds[te].sum()
    sub = rng.choice(te_idx, int(.5 * len(te_idx)), replace=False, p=pr)
    for el in ELS:
        for k in [1, 3]:
            Y = A[el]["y"].copy()
            inj = np.zeros(len(Y), bool)
            if k > 1:
                inj[rng.choice(sub, int(.10 * len(sub)), replace=False)] = True
                Y[inj] += np.log10(k)
            truth = inj[sub]
            s = (Y - A[el]["f"]) / A[el]["s"]
            pu = conf_p(s[ca], s[sub])
            pw = conf_p(s[ca], s[sub], wc=odds[ca], wt=odds[sub])
            for name, p in {"PACD unweighted": pu, "PACD propensity-weighted": pw}.items():
                rows.append(dict(rep=rep, scenario="covariate shift", element=el, k=k, method=name,
                                 null_p05=float((p[~truth] < 0.05).mean()), **ev(bh(p), truth)))
    return rows


def run(rep, nat, X, g, odds):
    return scenarios(rep, anchors(rep, nat, X, g), odds)


if __name__ == "__main__":
    df = load()
    nat = df[df["LandCover1"].isin(NATURAL) & df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    X = anchor_features(nat, with_xy=True)
    g = spatial_groups(nat, 120)
    ps = pd.read_csv(RES / "s04_propensity_Plan.csv")
    ps["SiteID"] = ps["SiteID"].astype(str)
    ps = ps.set_index("SiteID")
    e = ps["e"].reindex(nat.index.astype(str)).fillna(ps["e"].median()).to_numpy()
    odds = np.clip(e / (1 - e), 0.02, 50)
    out = Parallel(n_jobs=6)(delayed(run)(r, nat, X, g, odds) for r in range(10))
    res = pd.DataFrame([row for rows in out for row in rows])
    res.to_csv(RES / "s06_injection_v2_raw.csv", index=False)
    for sc in ["single", "covariate shift"]:
        sub = res[res.scenario == sc]
        cols = ["FDR", "power"] + (["null_p05"] if sc != "single" else [])
        print("=====", sc)
        print(sub.groupby(["k", "method"])[cols].mean().round(3).to_string())
    sub = res[res.scenario.str.startswith("signature")]
    print(sub.groupby(["scenario", "k", "method"])[["FDR", "power"]].mean().round(3).to_string())
