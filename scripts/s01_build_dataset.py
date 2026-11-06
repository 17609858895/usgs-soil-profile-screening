"""Build one row per site with Top5 / A / C horizon chemistry, mineralogy and metadata.

Conventions
- "<DL" values: numeric column = DL, flag column <layer>_<el>__cens = 1 (left-censored).
- XRD minerals (A, C only): blank = not detected -> 0 when the sample was scanned (Quartz present).
- C_Inorg blank with C_Tot measured -> 0 (no carbonate carbon).
- Depth strings "a-b" (cm) parsed to top/bottom; "N.S." -> NaN (no sample).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from load_ds801 import load_all, MINERALS

OUT = Path(__file__).resolve().parents[1] / "data_derived"


def parse_depth(s):
    s = s.fillna("").astype(str).str.strip()
    parts = s.str.extract(r"^\s*(\d+\.?\d*)\s*-\s*(\d+\.?\d*)\s*$")
    return pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")


def main():
    d = load_all()
    base = d["Top5"]["meta"][["SiteID", "StateID", "Latitude", "Longitude", "CollDate",
                              "LandCover1", "LandCover2"]].copy()
    base["SiteID"] = base["SiteID"].astype(str).str.strip()
    frames = [base.set_index("SiteID")]
    for layer, v in d.items():
        meta, vals, cens = v["meta"].copy(), v["vals"].copy(), v["cens"].copy()
        sid = meta["SiteID"].astype(str).str.strip()
        if layer in ("A", "C"):
            scanned = vals["Quartz"].notna()
            for m in MINERALS:
                vals.loc[scanned & vals[m].isna(), m] = 0.0
            ctot = vals["C_Tot"].notna()
            vals.loc[ctot & vals["C_Inorg"].isna(), "C_Inorg"] = 0.0
        else:
            vals = vals.drop(columns=[m for m in MINERALS] + ["C_Tot", "C_Inorg", "C_Org"])
            cens = cens[vals.columns]
        top, bot = parse_depth(meta["Depth"])
        f = vals.add_prefix(f"{layer}_")
        f[f"{layer}_depth_top"] = top.values
        f[f"{layer}_depth_bot"] = bot.values
        cc = cens.loc[:, cens.any()].astype(int)
        cc.columns = [f"{layer}_{c}__cens" for c in cc.columns]
        f = pd.concat([f, cc], axis=1)
        f.index = sid.values
        frames.append(f)
    df = pd.concat(frames, axis=1)
    df.index.name = "SiteID"
    df["CollDate"] = pd.to_datetime(df["CollDate"], format="%m/%d/%y", errors="coerce")
    OUT.mkdir(exist_ok=True, parents=True)
    df.to_parquet(OUT / "ds801_sites.parquet")
    print(df.shape)
    print(df[["Top5_Pb", "A_Pb", "C_Pb"]].describe().T)
    print("A depth bottom:", df["A_depth_bot"].describe().round(1).to_dict())
    print("C depth top:", df["C_depth_top"].describe().round(1).to_dict())
    full = df[["Top5_Pb", "A_Pb", "C_Pb", "C_Quartz"]].notna().all(axis=1).sum()
    print("sites with Top5+A+C chemistry and C XRD:", full)


if __name__ == "__main__":
    main()
