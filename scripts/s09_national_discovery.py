"""National application of profile-anchored conformal discovery (PACD, dual anchor).

Per split: natural sites are divided by spatial group into anchor-training and calibration
halves; every non-natural site is a test site. One-sided conformal p-values + BH (q = 0.10)
per element, plus multi-element signature scores. Repeated over 20 random splits; a site's
discovery frequency across splits measures stability. Conventional rules (Igeo > 1, EF > 2 vs
own C horizon, median + 2 MAD of natural topsoil) are applied to every site for comparison.
Geogenic negative control: held-out natural sites whose C-horizon concentration of the
element is in the top 10 % of natural sites (high-background parent material).
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
from s06_injection_v2 import conf_p, bh, UCC, SIGS, ELS

N_SPLIT = 20
LGB = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=20,
           subsample=0.8, subsample_freq=1, colsample_bytree=0.7, verbose=-1, n_jobs=1)


def landuse(df):
    lu = pd.Series("other", index=df.index)
    lu[df["LandCover1"].isin(NATURAL)] = "natural"
    lu[df["LandCover2"].isin(["Row Crops", "Row crops", "Small Grains", "Fallow"])] = "cropland"
    lu[df["LandCover2"].eq("Pasture/Hay")] = "pasture"
    lu[df["LandCover1"].eq("Developed") | df["LandCover2"].eq("Urban/Recreational Grasses")] = "urban"
    lu[df["LandCover2"].eq("Orchards/Vineyards/Tree Farm")] = "orchard"
    return lu


def split_run(sp, df, X, g, nat):
    rng = np.random.default_rng(500 + sp)
    ug = rng.permutation(np.unique(g[nat]))
    tr = nat & np.isin(g, ug[: int(.4 * len(ug))])
    ca = nat & np.isin(g, ug[int(.4 * len(ug)): int(.8 * len(ug))])
    nt = nat & ~(tr | ca)  # held-out natural sites: negative-control test set
    te = ~nat
    Z, out = {}, {}
    for el in ELS:
        y, cens = target(df, el)
        y = np.where(cens, y - np.log10(2), y)
        f = lgb.LGBMRegressor(**LGB).fit(X[tr], y[tr]).predict(X)
        oof = np.zeros(tr.sum())
        for a, b in GroupKFold(3).split(X[tr], y[tr], g[tr]):
            oof[b] = y[tr][b] - lgb.LGBMRegressor(**LGB).fit(X[tr].iloc[a], y[tr][a]).predict(X[tr].iloc[b])
        s = np.clip(lgb.LGBMRegressor(**{**LGB, "n_estimators": 150}).fit(X[tr], np.abs(oof)).predict(X), 0.03, None)
        Z[el] = (y - f) / s
        p = conf_p(Z[el][ca], Z[el][te])
        out[f"disc_{el}"] = bh(p)
        out[f"p_{el}"] = p
        out[f"excess_{el}"] = y[te] - f[te]  # log10 excess over the anchored baseline
        dn = np.full(nat.sum(), np.nan)
        dn[nt[nat]] = bh(conf_p(Z[el][ca], Z[el][nt]))
        out[f"natdisc_{el}"] = dn
    for sig, els in SIGS.items():
        sc = np.mean([Z[el] for el in els], axis=0)
        out[f"disc_sig_{sig}"] = bh(conf_p(sc[ca], sc[te]))
    smax = np.max([Z[el] for el in ELS], axis=0)
    out["disc_any_max"] = bh(conf_p(smax[ca], smax[te]))
    return {k: np.asarray(v, float) for k, v in out.items()}


if __name__ == "__main__":
    df = load()
    df = df[df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()].copy()
    df["landuse"] = landuse(df)
    df = df[df["landuse"] != "other"]
    nat = (df["landuse"] == "natural").to_numpy()
    X = dual_features(df)  # dual anchor: C horizon + immobile topsoil suite
    g = spatial_groups(df, 150)
    outs = Parallel(n_jobs=8)(delayed(split_run)(sp, df, X, g, nat) for sp in range(N_SPLIT))
    test = df[~nat][["Latitude", "Longitude", "StateID", "LandCover1", "LandCover2", "landuse"]].copy()
    natt = df[nat][["Latitude", "Longitude", "StateID", "LandCover2"] + [f"C_{el}" for el in ELS]].copy()
    for k in outs[0]:
        if k.startswith("natdisc_"):
            natt[k[3:]] = np.nanmean([o[k] for o in outs], axis=0)  # freq when held out
        else:
            test[k] = np.mean([o[k] for o in outs], axis=0)  # discovery frequency / mean p / excess
    natt.to_csv(RES / "s09_natural_heldout_discoveries.csv")
    test.to_csv(RES / "s09_test_site_discoveries.csv")
    # conventional rules on every site
    ti = np.log10(df["Top5_Ti"]) - np.log10(df["C_Ti"])
    rule_rows = []
    for el in ELS:
        y, cens = target(df, el)
        y = np.where(cens, y - np.log10(2), y)
        ef = y - (X[f"C_{el}"].to_numpy() + ti.to_numpy())
        ynat = y[nat]
        mad = 1.4826 * np.median(np.abs(ynat - np.median(ynat)))
        flags = {"Igeo > 1": 10 ** y > 3 * UCC[el], "EF > 2": ef > np.log10(2),
                 "Median + 2MAD": y > np.median(ynat) + 2 * mad}
        for name, fl in flags.items():
            for lu in ["natural", "cropland", "pasture", "urban", "orchard"]:
                m = (df["landuse"] == lu).to_numpy()
                rule_rows.append(dict(element=el, method=name, landuse=lu, rate=fl[m].mean(), n=int(m.sum())))
            hb = nat & (df[f"C_{el}"] >= df.loc[nat, f"C_{el}"].quantile(0.9)).to_numpy()
            rule_rows.append(dict(element=el, method=name, landuse="natural, high-background C horizon",
                                  rate=fl[hb].mean(), n=int(hb.sum())))
        rule_rows.append(dict(element=el, method="PACD (BH q=0.10, freq>=0.5)", landuse="natural",
                              rate=(natt[f"disc_{el}"] >= 0.5).mean(), n=len(natt)))
        hbn = natt[f"C_{el}"] >= natt[f"C_{el}"].quantile(0.9)
        rule_rows.append(dict(element=el, method="PACD (BH q=0.10, freq>=0.5)", landuse="natural, high-background C horizon",
                              rate=(natt.loc[hbn, f"disc_{el}"] >= 0.5).mean(), n=int(hbn.sum())))
        for lu in ["cropland", "pasture", "urban", "orchard"]:
            m = test["landuse"] == lu
            rule_rows.append(dict(element=el, method="PACD (BH q=0.10, freq>=0.5)", landuse=lu,
                                  rate=(test.loc[m, f"disc_{el}"] >= 0.5).mean(), n=int(m.sum())))
    rules = pd.DataFrame(rule_rows)
    rules.to_csv(RES / "s09_flag_rates_by_method_landuse.csv", index=False)
    print(rules.pivot_table(index=["element", "method"], columns="landuse", values="rate").round(3).to_string())
    cols = [c for c in test.columns if c.startswith("disc_sig") or c == "disc_any_max"]
    print((test.groupby("landuse")[cols].apply(lambda t: (t >= 0.5).mean())).round(3).to_string())
