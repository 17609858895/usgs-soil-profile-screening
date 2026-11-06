"""Shared feature construction and spatial CV for the DS801 profile-anchoring study."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_derived" / "ds801_sites.parquet"
RES = ROOT / "tables"
RES.mkdir(exist_ok=True)

ELEMENTS = ["Al", "As", "Ba", "Be", "Bi", "Ca", "Cd", "Ce", "Co", "Cr", "Cs", "Cu", "Fe", "Ga",
            "Hg", "In", "K", "La", "Li", "Mg", "Mn", "Mo", "Na", "Nb", "Ni", "P", "Pb", "Rb",
            "S", "Sb", "Sc", "Se", "Sn", "Sr", "Te", "Th", "Ti", "Tl", "U", "V", "W", "Y", "Zn"]
DROP_EL = ["Ag", "Te", "Cs"]  # >70 % non-detects in every horizon
MINERALS = ["Quartz", "Tot_K_fs", "Tot_Plag", "Tot_14A", "Tot_10A", "Kaolinit", "Tot_Clay",
            "Gibbsite", "Calcite", "Dolomite", "Tot_Carb", "Tot_Zeol", "Gypsum", "Talc", "Hornbl",
            "Serpent", "Hematite", "Goethite", "Pyroxene", "Amorph"]
NATURAL = ["Forested Upland", "Shrubland", "Herbaceous Upland", "Barren"]
TARGETS = ["As", "Cd", "Co", "Cr", "Cu", "Hg", "Mo", "Ni", "P", "Pb", "Sb", "Se", "Tl", "U", "V", "Zn"]


def load():
    return pd.read_parquet(DATA)


def log_el(df, layer, el):
    """log10 concentration with non-detects at DL/2 plus the censor flag."""
    v = df[f"{layer}_{el}"].astype(float).copy()
    c = df.get(f"{layer}_{el}__cens", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    v[c] = v[c] / 2.0
    return np.log10(v.clip(lower=1e-4)), c.astype(int)


def anchor_features(df, layer="C", with_xy=True):
    """Parent-horizon descriptors: C-horizon chemistry, XRD, carbon, depth."""
    X = pd.DataFrame(index=df.index)
    for el in ELEMENTS:
        if el in DROP_EL:
            continue
        X[f"{layer}_{el}"], c = log_el(df, layer, el)
        if c.mean() > 0.02:
            X[f"{layer}_{el}__cens"] = c
    for m in MINERALS:
        X[f"{layer}_{m}"] = df[f"{layer}_{m}"]
    for k in ["C_Tot", "C_Inorg", "C_Org"]:
        X[f"{layer}_{k}"] = np.log10(df[f"{layer}_{k}"].clip(lower=0.005))
    X[f"{layer}_depth_top"] = df[f"{layer}_depth_top"]
    if with_xy:
        X["lat"] = df["Latitude"]
        X["lon"] = df["Longitude"]
    return X


def target(df, el, layer="Top5"):
    y = df[f"{layer}_{el}"].astype(float)
    c = df.get(f"{layer}_{el}__cens", pd.Series(0, index=df.index)).fillna(0).astype(bool)
    return np.log10(y.clip(lower=1e-4)), c


def spatial_groups(df, n_clusters=120, seed=0):
    xy = df[["Latitude", "Longitude"]].to_numpy()
    km = KMeans(n_clusters=n_clusters, n_init=4, random_state=seed).fit(xy)
    return km.labels_


IMMOBILE = ["Al", "Ti", "Nb", "Th", "Sc", "La", "Ce", "Y", "Ga"]


def dual_features(df):
    """Vertical anchor (C horizon) + in-sample anchor (immobile lithogenic elements in the same
    topsoil sample; examined separately using negative-control outcomes)."""
    X = anchor_features(df, with_xy=True)
    for el in IMMOBILE:
        X[f"Top5_{el}"], _ = log_el(df, "Top5", el)
    return X
