"""Harmonise the EPA evidence sources used to extend the s22 labels (rule: contaminated_reference_comparison.md
section 10).

Inputs (downloaded 2026-11-06, docs/evidence_record.md):
  data_external/epa_smelting/406370.pdf     SEMS FOIA-001 Sites with Potential Smelting-Related Operations
                                            (run 2026-10-26); the source table is the attached 406370_attached.xlsx
                                            (extracted with poppler pdfdetach).
  data_external/epa_coc/406315.xlsx         Contaminant of Concern Data for Decision Documents by Media, FY1981-2024.
  data_external/frs_sems/sems_points.json   EPA FRS SEMS points (ArcGIS FRS_INTERESTS_SEMS) for sites without
                                            coordinates in the tables above; fetched by this script (--fetch).
  data_external/tri/tri_{1988,1995,2000,2005,2010}_us.csv   TRI Basic national files.

Output: data_derived/evidence_sources.parquet, one row per (source, site, element):
  source   SMELT (smelting-related SEMS site) | NPL (soil/sediment COC at an NPL or SAA site) | TRI (air release)
  site_id, name, state, latitude, longitude, element (DS801 symbol), detail.
Run: python -I scripts/s23_external_sources.py [--fetch]
"""
import sys
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent))
from common import *


sys.stdout.reconfigure(encoding="utf-8")
EXT = ROOT / "data_external"
FRS_URL = ("https://services.arcgis.com/cJ9YHowT8TU7DUyn/arcgis/rest/services/"
           "FRS_INTERESTS_SEMS/FeatureServer/0/query")
FRS_JSON = EXT / "frs_sems" / "sems_points.json"
TRI_YEARS = [1988, 1995, 2000, 2005, 2010]
NAMES = dict(As="ARSENIC", Cd="CADMIUM", Cr="CHROMIUM", Cu="COPPER", Hg="MERCURY", Ni="NICKEL",
             P="PHOSPHORUS", Pb="LEAD", Sb="ANTIMONY", Se="SELENIUM", U="URANIUM", Zn="ZINC")
SMELT_SUITE = ["Pb", "Zn", "Cd", "Cu", "Sb", "As"]  # smelting site without a listed metal contaminant
COC_MEDIA = ["Soil", "Sediment"]
BIDI = re.compile("[‎‏‪-‮]")


def elements_in(name):
    """DS801 elements named in a contaminant / chemical string. Lead isotopes (Pb-210/212/214, radon progeny)
    are excluded."""
    s = str(name).upper()
    out = []
    for el, nm in NAMES.items():
        if re.search(rf"\b{nm}\b", s) and not (el == "Pb" and re.search(r"LEAD-\d", s)):
            out.append(el)
    return out


def clean(df):
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: BIDI.sub("", v).strip() if isinstance(v, str) else v).replace("", np.nan)
    return df


def read_smelting():
    d = pd.read_excel(EXT / "epa_smelting" / "406370_attached.xlsx", header=4).iloc[1:]
    d.columns = [str(c).strip() for c in d.columns]
    d = clean(d[d["EPA ID"].notna()].copy())
    for c in ["Latitude", "Longitude"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def read_coc():
    d = pd.read_excel(EXT / "epa_coc" / "406315.xlsx", header=1)
    d = clean(d)
    for c in ["Latitude", "Longitude"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d


def fetch_frs(ids):
    feats = []
    ids = sorted(set(ids))
    for i in range(0, len(ids), 150):
        where = "PGM_SYS_ID IN (" + ",".join(f"'{x}'" for x in ids[i:i + 150]) + ")"
        body = urllib.parse.urlencode(dict(where=where, f="json", returnGeometry="false",
                                           outFields="PGM_SYS_ID,PRIMARY_NAME,STATE_CODE,LATITUDE83,LONGITUDE83,"
                                                     "ACCURACY_VALUE,COLLECT_MTH_DESC,ACTIVE_STATUS")).encode()
        with urllib.request.urlopen(urllib.request.Request(FRS_URL, data=body), timeout=60) as r:
            js = json.load(r)
        if "error" in js:
            raise RuntimeError(js["error"])
        feats += [f["attributes"] for f in js["features"]]
    FRS_JSON.write_text(json.dumps(dict(source=FRS_URL, fetched="2026-11-06", n_requested=len(ids),
                                        features=feats), indent=1), encoding="utf-8")
    return feats


def frs_table():
    f = pd.DataFrame(json.loads(FRS_JSON.read_text(encoding="utf-8"))["features"])
    f = f.dropna(subset=["LATITUDE83", "LONGITUDE83"]).drop_duplicates("PGM_SYS_ID")
    return f.set_index("PGM_SYS_ID")[["LATITUDE83", "LONGITUDE83", "COLLECT_MTH_DESC"]]


def add_coords(d, frs):
    own = d["Latitude"].notna() & d["Longitude"].notna()
    j = d["EPA ID"].map(frs["LATITUDE83"]), d["EPA ID"].map(frs["LONGITUDE83"])
    d["lat"] = np.where(own, d["Latitude"], j[0])
    d["lon"] = np.where(own, d["Longitude"], j[1])
    d["coord_src"] = np.where(own, "table", np.where(j[0].notna(), "FRS", "none"))
    return d


def build_smelt(sm, frs):
    sm = add_coords(sm, frs)
    rows = []
    for eid, g in sm.groupby("EPA ID"):
        els = sorted({e for n in g["Contaminant Name"].dropna() for e in elements_in(n)})
        basis = "listed CoC" if els else "smelter suite"
        r0 = g.iloc[0]
        sub = r0.get("Primary Site -Sub Type")
        for el in (els or SMELT_SUITE):
            rows.append(dict(source="SMELT", site_id=eid, name=r0["Site Name"], state=r0["State"],
                             latitude=r0["lat"], longitude=r0["lon"], element=el,
                             detail=f"{basis}; {r0['NPL']}; {sub}; coords {r0['coord_src']}"))
    return pd.DataFrame(rows)


def build_npl(coc, frs):
    coc = add_coords(coc.copy(), frs)
    m = coc[coc["Media"].isin(COC_MEDIA)].copy()
    m["els"] = m["Contaminant Name"].map(elements_in)
    m = m.explode("els").dropna(subset=["els"])
    g = m.groupby(["EPA ID", "els"]).agg(name=("Site Name", "first"), state=("State", "first"),
                                         latitude=("lat", "first"), longitude=("lon", "first"),
                                         media=("Media", lambda v: "/".join(sorted(set(v)))),
                                         npl=("NPL Status", "first"), fy=("Fiscal Year", "min")).reset_index()
    return pd.DataFrame(dict(source="NPL", site_id=g["EPA ID"], name=g["name"], state=g["state"],
                             latitude=g["latitude"], longitude=g["longitude"], element=g["els"],
                             detail=g["media"] + "; " + g["npl"] + "; first FY " + g["fy"].astype(str)))


def read_tri(year):
    t = pd.read_csv(EXT / "tri" / f"tri_{year}_us.csv", low_memory=False, encoding="latin-1")
    t.columns = [re.sub(r"^\d+\.\s*", "", c).strip().upper() for c in t.columns]
    return t


def build_tri():
    rows = []
    for y in TRI_YEARS:
        t = read_tri(y)
        col = lambda key: next(c for c in t.columns if c.endswith(key))
        air = t[col("FUGITIVE AIR")].fillna(0) + t[col("STACK AIR")].fillna(0)
        t = t.assign(air_lb=pd.to_numeric(air, errors="coerce").fillna(0), els=t["CHEMICAL"].map(elements_in))
        t = t[t["air_lb"] > 0].explode("els").dropna(subset=["els"])
        rows.append(pd.DataFrame(dict(site_id=t["TRIFD"], name=t["FACILITY NAME"], state=t["ST"],
                                      latitude=pd.to_numeric(t["LATITUDE"], errors="coerce"),
                                      longitude=pd.to_numeric(t["LONGITUDE"], errors="coerce"),
                                      element=t["els"], air_lb=t["air_lb"], year=y)))
    t = pd.concat(rows)
    g = t.groupby(["site_id", "element"]).agg(name=("name", "last"), state=("state", "last"),
                                              latitude=("latitude", "median"), longitude=("longitude", "median"),
                                              air_lb=("air_lb", "max"),
                                              years=("year", lambda v: ",".join(map(str, sorted(set(v)))))).reset_index()
    return pd.DataFrame(dict(source="TRI", site_id=g["site_id"], name=g["name"], state=g["state"],
                             latitude=g["latitude"], longitude=g["longitude"], element=g["element"],
                             detail="max air " + g["air_lb"].round(0).astype(int).astype(str) + " lb; years " + g["years"]))


if __name__ == "__main__":
    sm, coc = read_smelting(), read_coc()
    if "--fetch" in sys.argv or not FRS_JSON.exists():
        need = set(sm.loc[sm["Latitude"].isna(), "EPA ID"]) | set(coc.loc[coc["Latitude"].isna(), "EPA ID"])
        feats = fetch_frs(need)
        print(f"FRS: requested {len(need)} IDs, received {len(feats)} records")
    frs = frs_table()
    src = pd.concat([build_smelt(sm, frs), build_npl(coc, frs), build_tri()], ignore_index=True)
    conus = src["latitude"].between(24, 50) & src["longitude"].between(-125, -66)
    print("rows without usable CONUS coordinates:\n" + src[~conus].groupby("source").site_id.nunique().to_string())
    src = src[conus].reset_index(drop=True)
    src.to_parquet(ROOT / "data_derived" / "evidence_sources.parquet", index=False)
    print("\nsites per source and element\n" + src.groupby(["element", "source"]).site_id.nunique().unstack().to_string())
    s = src[src.source == "SMELT"].drop_duplicates("site_id")
    print("\nsmelting sites: listed CoC", s.detail.str.startswith("listed").sum(), "| smelter suite",
          s.detail.str.startswith("smelter").sum())
