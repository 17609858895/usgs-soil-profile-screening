"""Head-to-head: PACD self-cleaning vs the contaminated-reference methods of Bashari, Sesia &
Romano (ICML 2025, arXiv:2502.04807) inside the s15 injection design (dual anchor, same splits).

Reference contamination regimes: rate r in {0, 1, 3, 5} % of calibration sites, strength
k_ref in {3, 10} (r = 0 measures the clean-side cost of trimming, cf. Wang 2026,
arXiv:2605.06204). Test sets: 10 % of test sites enriched by k in {3, 5, 10}; k = 1 is all-null.
Methods (all use the same scores):
  Standard     plain conformal on the contaminated reference
  Oracle       injected reference contamination removed (infeasible)
  Naive-Trim   top r % of reference scores removed (r known, as in Bashari et al.)
  Label-Trim   top m = 50 reference scores annotated, true injected outliers among them removed
  Small-Clean  m = 50 random reference points annotated, inliers kept
  PACD         label-free self-cleaning (LOO conformal + robust GPD tail, BH q = 0.10)
  PACD + EVT   PACD with the EVT tail beyond the reference maximum (sensitivity)
Metrics: BH (q = 0.10) FDR and power; per-test type-I error at alpha = 0.01 / 0.02 on null test
sites; numbers of clean and contaminated reference points removed.
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
from s15_contaminated_reference import ELS, SIGS, LGB, ev

N_REP = 10
M_LABEL = 50
REGIMES = [(0.0, 10)] + [(r, kr) for r in (0.01, 0.03, 0.05) for kr in (3, 10)]
ALPHAS = (0.01, 0.02)


def fit_scores(rep, nat, X, g):
    """Same split and anchor fits as s15 (seed 300 + rep)."""
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
    return ca, te, Z0


def methods(cal, bad, r, rng):
    """Calibration subsets (boolean keep-masks or index arrays) for every method."""
    n = len(cal)
    order = np.argsort(cal)
    keep = {"Standard": np.ones(n, bool), "Oracle": ~bad}
    k_naive = int(round(r * n))
    kn = np.ones(n, bool)
    if k_naive:
        kn[order[-k_naive:]] = False
    keep["Naive-Trim"] = kn
    top = order[-M_LABEL:]
    kl = np.ones(n, bool)
    kl[top[bad[top]]] = False
    keep["Label-Trim"] = kl
    sub = rng.choice(n, M_LABEL, replace=False)
    ks = np.zeros(n, bool)
    ks[sub[~bad[sub]]] = True
    keep["Small-Clean"] = ks
    keep["PACD"] = clean_reference(cal, use_evt=True, **CLEAN_V1)
    return keep


def run(rep, nat, X, g):
    ca, te, Z0 = fit_scores(rep, nat, X, g)
    rng = np.random.default_rng(1900 + rep)
    ca_idx, te_idx = np.nonzero(ca)[0], np.nonzero(te)[0]
    targets = [(el, [el]) for el in ELS] + [(f"sig:{k}", v) for k, v in SIGS.items()]
    rows = []
    for r, k_ref in REGIMES:
        for name, els in targets:
            ref_bad = rng.choice(ca_idx, int(round(r * len(ca_idx))), replace=False)
            for k in [1, 3, 5, 10]:
                inj = rng.choice(te_idx, int(.10 * len(te_idx)), replace=False) if k > 1 else np.array([], int)
                Zs = []
                for el in els:
                    res, s = Z0[el]
                    res = res.copy()
                    res[ref_bad] += np.log10(k_ref)
                    res[inj] += np.log10(k)
                    Zs.append(res / s)
                Z = np.mean(Zs, axis=0)
                truth = np.isin(te_idx, inj)
                cal, tst = Z[ca], Z[te]
                bad = np.isin(ca_idx, ref_bad)
                keep = methods(cal, bad, r, rng)
                ps = {m: conf_p(cal[kp], tst) for m, kp in keep.items()}
                ps["PACD + EVT"] = evt_p(cal[keep["PACD"]], tst)
                for m, p in ps.items():
                    kp = keep["PACD" if m == "PACD + EVT" else m]
                    row = dict(rep=rep, r=r, k_ref=k_ref, target=name, k=k, method=m, n_ref=int(kp.sum()),
                               bad_removed=int((~kp & bad).sum()) if m != "Small-Clean" else np.nan,
                               clean_removed=int((~kp & ~bad).sum()) if m != "Small-Clean" else np.nan,
                               **ev(bh(p), truth))
                    for a in ALPHAS:
                        row[f"typeI_{a}"] = (p[~truth] <= a).mean()
                    rows.append(row)
    return rows


if __name__ == "__main__":
    df = load()
    nat = df[df["LandCover1"].isin(NATURAL) & df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    X = dual_features(nat)
    g = spatial_groups(nat, 120)
    out = Parallel(n_jobs=6)(delayed(run)(i, nat, X, g) for i in range(N_REP))
    res = pd.DataFrame([row for rows in out for row in rows])
    res["kind"] = np.where(res["target"].str.startswith("sig"), "signature", "single")
    res.to_csv(RES / "s19_trim_comparison_raw.csv", index=False)

    pd.set_option("display.width", 250)
    order = ["Oracle", "Standard", "Naive-Trim", "Label-Trim", "Small-Clean", "PACD", "PACD + EVT"]
    null = res[res["k"] == 1]
    t1 = null.groupby(["r", "k_ref", "method"])[["typeI_0.01", "typeI_0.02"]].mean().unstack("method")
    print("== per-test type-I error on all-null test sets (single + signature)\n"
          + t1.reindex(columns=order, level=1).round(4).to_string())
    gn = null.assign(any=null["n_disc"] > 0).groupby(["r", "k_ref", "kind", "method"])["any"].mean().unstack("method")
    print("\n== global null: P(any BH discovery)\n" + gn[order].round(3).to_string())
    alt = res[res["k"] > 1].groupby(["kind", "r", "k_ref", "k", "method"])[["FDR", "power"]].mean()
    for kind in ["single", "signature"]:
        a = alt.loc[kind]
        print(f"\n== {kind}: BH FDR\n" + a["FDR"].unstack("method")[order].round(3).to_string())
        print(f"\n== {kind}: BH power\n" + a["power"].unstack("method")[order].round(3).to_string())
    rm = res[(res["k"] == 1) & res["method"].isin(["Naive-Trim", "Label-Trim", "PACD"])]
    print("\n== reference points removed (mean per test; clean = wrongly trimmed inliers)\n"
          + rm.groupby(["r", "k_ref", "method"])[["bad_removed", "clean_removed", "n_ref"]].mean().round(2).unstack("method").to_string())
