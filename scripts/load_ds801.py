"""Load USGS DS801 Top5 / A / C horizon tables into tidy frames.

Values reported as "<x" are kept as left-censored observations: the numeric
column holds the detection limit and a parallel boolean marks censoring.
"N.S." and blanks are missing (not analysed / no sample).
"""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "data_raw"
FILES = {
    "Top5": ROOT / "Appendix_2b_Top5_18Sept2013.txt",
    "A": ROOT / "Appendix_3b_Ahorizon_18Sept2013.txt",
    "C": ROOT / "Appendix_4b_Chorizon_18Sept2013.txt",
}
META = ["LabID", "SiteID", "StateID", "Latitude", "Longitude", "CollDate",
        "LandCover1", "LandCover2", "Depth"]
MINERALS = ["Quartz", "Tot_K_fs", "Tot_Plag", "Tot_Flds", "Tot_14A", "Tot_10A",
            "Kaolinit", "Tot_Clay", "Gibbsite", "Calcite", "Dolomite", "Aragon",
            "Tot_Carb", "Analcime", "Heuland", "Tot_Zeol", "Gypsum", "Talc",
            "Hornbl", "Serpent", "Hematite", "Goethite", "Pyroxene", "Pyrite",
            "Other", "Amorph"]


def _find_header(path):
    with open(path, encoding="latin-1") as fh:
        for i, line in enumerate(fh):
            if "SiteID" in line and "Latitude" in line:
                return i
    raise ValueError(path)


def read_horizon(layer):
    path = FILES[layer]
    h = _find_header(path)
    df = pd.read_csv(path, sep="\t", skiprows=h, header=0, dtype=str,
                     encoding="latin-1")
    df = df.iloc[1:]  # unit row
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    prefix = f"{layer}_"
    df.columns = [c[len(prefix):] if c.startswith(prefix) else c for c in df.columns]
    df = df[df["SiteID"].notna() & (df["SiteID"].str.strip() != "")]
    return df.reset_index(drop=True)


def parse_values(df):
    """Return (values, censored_flag) frames for analyte columns."""
    analytes = [c for c in df.columns if c not in META]
    vals = pd.DataFrame(index=df.index)
    cens = pd.DataFrame(index=df.index)
    for c in analytes:
        s = df[c].fillna("").astype(str).str.strip()
        is_lt = s.str.startswith("<")
        num = pd.to_numeric(s.str.lstrip("<").str.strip(), errors="coerce")
        vals[c] = num
        cens[c] = is_lt
    return vals, cens


def load_all():
    out = {}
    for layer in FILES:
        raw = read_horizon(layer)
        vals, cens = parse_values(raw)
        meta = raw[[c for c in META if c in raw.columns]].copy()
        meta["Latitude"] = pd.to_numeric(meta["Latitude"], errors="coerce")
        meta["Longitude"] = pd.to_numeric(meta["Longitude"], errors="coerce")
        out[layer] = dict(meta=meta, vals=vals, cens=cens)
    return out


if __name__ == "__main__":
    d = load_all()
    for k, v in d.items():
        print(k, v["meta"].shape, v["vals"].shape)
