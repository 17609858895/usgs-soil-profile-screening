"""DML robustness: normalise topsoil outcomes by a lithogenic-matrix index instead of Ti alone.

L = mean log10 of nine immobile elements (Al, Ti, Nb, Th, Sc, La, Ce, Y, Ga) in the same
topsoil sample; a negative-control element j is normalised by L computed without j.
Ti-only normalisation left cropland Al/Ti and Sc/Ti at about +6 % (texture change); a
multi-element matrix should absorb organic-matter dilution and texture shifts jointly.
Propensities are reused from s08 (same samples, folds rebuilt with the same seed).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from s08_dml_v2 import OUTCOMES, NEG_CTRL, TREAT, att_aipw

IMMOBILE = ["Al", "Ti", "Nb", "Th", "Sc", "La", "Ce", "Y", "Ga"]


def lith_index(sub, drop=None):
    cols = [np.log10(sub[f"Top5_{e}"].astype(float).clip(lower=1e-4)) for e in IMMOBILE if e != drop]
    return np.mean(cols, axis=0)


def one(treat, el, sub, d, X, e, folds, g):
    y, cens = target(sub, el)
    y = np.where(cens.to_numpy(), y.to_numpy() - np.log10(2), y.to_numpy())
    y = y - lith_index(sub, drop=el if el in IMMOBILE else None)
    att, se, naive = att_aipw(y, d, X, e, folds, g)
    return dict(treatment=treat, element=el, outcome="lithogenic-index normalised",
                role="negative control" if el in NEG_CTRL else "target",
                ATT_log10=att, SE=se, z=att / se, ATT_pct=(10 ** att - 1) * 100,
                lo_pct=(10 ** (att - 1.96 * se) - 1) * 100, hi_pct=(10 ** (att + 1.96 * se) - 1) * 100,
                naive_pct=(10 ** naive - 1) * 100, cens_frac=float(cens.mean()))


if __name__ == "__main__":
    df = load()
    df = df[df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    natural = df["LandCover1"].isin(NATURAL)
    rows = []
    for treat, fn in TREAT.items():
        ps = pd.read_csv(RES / f"s08_propensity_{treat}.csv")
        ps["SiteID"] = ps["SiteID"].astype(str)
        sub = df.loc[ps["SiteID"]].copy()
        d, e = ps["d"].to_numpy(), ps["e"].to_numpy()
        X = anchor_features(sub, with_xy=True).reset_index(drop=True)
        g = spatial_groups(sub, 120)
        folds = list(GroupKFold(5).split(X, d, g))
        rows += Parallel(n_jobs=8)(delayed(one)(treat, el, sub, d, X, e, folds, g)
                                   for el in OUTCOMES + NEG_CTRL)
        print(treat, "done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(RES / "s11_dml_lithogenic_index.csv", index=False)
    res["cell"] = res.apply(lambda x: f"{x.ATT_pct:+.1f} ({x.z:+.1f})", axis=1)
    print(res.pivot_table(index=["role", "element"], columns="treatment", values="cell", aggfunc="first").to_string())
