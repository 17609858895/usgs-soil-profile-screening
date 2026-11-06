"""External validation of the national PACD discoveries against USGS MRDS deposit locations.

Mineral-occurrence proximity is a descriptive geological comparison. It does not independently
identify anthropogenic inputs: mineralisation can raise both parent-horizon concentrations and
the probability of a nearby mineral record. Conventional flags and geology-only rankings are
compared on the same sites.
Inference: flags are permuted within strata (C-horizon decile x coarse region), so enrichment is
measured beyond what parent material and broad geography predict. The stratified permutation null
is a sum of independent hypergeometric draws, sampled directly.
MRDS: https://mrdata.usgs.gov/mrds/mrds-csv.zip (downloaded 2026-11-05; docs/evidence_record.md).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from s06_injection_v2 import UCC, SIGS, ELS
from s09_national_discovery import landuse

MRDS = ROOT / "data_external" / "mrds" / "mrds.csv"
R_EARTH = 6371.0
RADII = [5, 10, 25]
R0 = 10  # headline radius (km)
K_LIST = 50  # length of the ranked comparison lists (fixed, independent of the PACD discovery count)
B = 10000
BINS = [0, 5, 10, 25, 50, 100, np.inf]
COMMOD = dict(Pb="Lead", Zn="Zinc|Cadmium", Cd="Zinc|Cadmium", Cu="Copper", Sb="Antimony",
              Hg="Mercury", As="Arsenic", Ni="Nickel", Cr="Chromium", U="Uranium", P="Phosph",
              Se="Selenium")
TARGET_ELS = {"Pb": ["Pb"], "Zn": ["Zn"], "Cd": ["Cd"], "Sb": ["Sb"], "Cu": ["Cu"],
              "sig_mining": SIGS["mining"], "any_max": ELS}
WORKED = ["Producer", "Past Producer", "Plant"]
UNWORKED = ["Occurrence", "Prospect"]


def load_mrds():
    cols = ["dep_id", "site_name", "latitude", "longitude", "country", "state", "commod1",
            "commod2", "dev_stat", "prod_size"]
    m = pd.read_csv(MRDS, usecols=cols, low_memory=False)
    m = m[(m.country == "United States") & ~m.state.isin(["Alaska", "Hawaii"])
          & m.latitude.between(24, 50) & m.longitude.between(-125, -66)].copy()
    m["commod"] = m.commod1.fillna("") + ", " + m.commod2.fillna("")  # major + minor commodities
    return m


def commod_regex(t):
    return "|".join(sorted({COMMOD[e] for e in TARGET_ELS[t]}))


def dist_count(dep, site_rad):
    """Nearest-deposit distance (km), its row position, and deposit counts within RADII."""
    if len(dep) == 0:
        return np.full(len(site_rad), np.inf), np.full(len(site_rad), -1), {r: np.zeros(len(site_rad)) for r in RADII}
    tree = BallTree(np.radians(dep[["latitude", "longitude"]].to_numpy()), metric="haversine")
    dd, ii = tree.query(site_rad, k=1)
    cnt = {r: tree.query_radius(site_rad, r / R_EARTH, count_only=True) for r in RADII}
    return dd[:, 0] * R_EARTH, ii[:, 0], cnt


def strat_test(flag, near, strata, rng):
    """Observed flagged-and-near count, its expectation and upper p under within-stratum permutation."""
    obs, exp, draws = int((flag & near).sum()), 0.0, np.zeros(B)
    for s in np.unique(strata):
        m = strata == s
        n_s, N_s, K_s = int(flag[m].sum()), int(m.sum()), int(near[m].sum())
        if n_s == 0:
            continue
        exp += n_s * K_s / N_s
        draws += rng.hypergeometric(K_s, N_s - K_s, n_s, size=B)
    return obs, exp, (1 + (draws >= obs).sum()) / (B + 1)


def z(a):
    return (a - np.nanmean(a)) / np.nanstd(a)


if __name__ == "__main__":
    rng = np.random.default_rng(17)
    d = pd.read_csv(RES / "s16_site_discoveries_variants.csv", index_col=0)
    df = load()
    df = df[df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()].copy()
    df["landuse"] = landuse(df)
    df = df[df["landuse"] != "other"]
    assert (df.index.astype(str) == d.index.astype(str)).all()
    nat = (df["landuse"] == "natural").to_numpy()
    n = len(df)
    region = spatial_groups(df, 6)

    # per-element topsoil, C-horizon and EF (Ti-normalised Top5/C) on the log10 scale
    ti = (log_el(df, "Top5", "Ti")[0] - log_el(df, "C", "Ti")[0]).to_numpy()
    Y, C, EF, RULE = {}, {}, {}, {}
    for el in ELS:
        y, cens = target(df, el)
        y = np.where(cens, y - np.log10(2), y).astype(float)
        c = log_el(df, "C", el)[0].to_numpy()
        ynat = y[nat]
        mad = 1.4826 * np.median(np.abs(ynat - np.median(ynat)))
        Y[el], C[el], EF[el] = y, c, y - (c + ti)
        RULE[el] = {"Igeo > 1": 10 ** y > 3 * UCC[el], "EF > 2": EF[el] > np.log10(2),
                    "Median + 2MAD": y > np.median(ynat) + 2 * mad}

    mrds = load_mrds()
    site_rad = np.radians(df[["Latitude", "Longitude"]].to_numpy())
    keep_cols = ["dep_id", "site_name", "latitude", "longitude", "state", "commod1", "dev_stat", "prod_size"]
    mrds[keep_cols + ["commod"]].to_parquet(ROOT / "data_derived" / "mrds_conus.parquet")
    print(f"MRDS conterminous US records: {len(mrds)}")

    enr_rows, decay_rows, topk_rows, disc_rows, geo_rows = [], [], [], [], []
    for t, els in TARGET_ELS.items():
        dep = mrds[mrds.commod.str.contains(commod_regex(t), regex=True)]
        cls = {"worked": dep[dep.dev_stat.isin(WORKED)],
               "major worked (M/L)": dep[dep.dev_stat.isin(WORKED) & dep.prod_size.isin(["M", "L"])],
               "unworked": dep[dep.dev_stat.isin(UNWORKED)]}
        geo = {k: dist_count(v, site_rad) for k, v in cls.items()}
        near = {k: geo[k][0] <= R0 for k in geo}
        near_sets = {}
        for r in RADII:
            w = geo["worked"][0] <= r
            near_sets[("worked", r)] = w
            near_sets[("major worked (M/L)", r)] = geo["major worked (M/L)"][0] <= r
            near_sets[("unworked only", r)] = (geo["unworked"][0] <= r) & ~w

        # flag sets and rankings for this target
        agg = np.nanmax if t == "any_max" else np.nanmean
        rank = {"PACD score": d[f"score_{t}"].to_numpy(),
                "Concentration (Igeo/MAD order)": agg(np.vstack([z(Y[e]) for e in els]), axis=0),
                "EF": agg(np.vstack([EF[e] for e in els]), axis=0),
                "C horizon (geology only)": agg(np.vstack([z(C[e]) for e in els]), axis=0)}
        pacd = d[f"disc_main_{t}"].to_numpy().astype(bool)
        k0 = K_LIST
        flags = {"PACD (BH q=0.10)": pacd}
        for rule in ["Igeo > 1", "EF > 2", "Median + 2MAD"]:
            flags[rule] = np.any(np.vstack([RULE[e][rule] for e in els]), axis=0)
        for name, sc in rank.items():
            top = np.zeros(n, bool)
            top[np.argsort(-np.nan_to_num(sc, nan=-np.inf))[:k0]] = True
            flags[f"top-{k0}: {name}"] = top
        cstrat = np.nan_to_num(pd.qcut(rank["C horizon (geology only)"], 10, labels=False, duplicates="drop"),
                               nan=-1).astype(int)
        strata = {"unconditional": np.zeros(n, int), "C decile x region": cstrat * 10 + region}

        geogenic = np.any(np.vstack([10 ** C[e] > 3 * UCC[e] for e in els]), axis=0)  # C horizon alone has Igeo > 1
        ratio = np.nanmean(np.vstack([Y[e] - C[e] for e in els]), axis=0)  # log10 Top5/C
        for method, fl in flags.items():
            for sub, m in [("all flagged", fl), (f"flagged within {R0} km of worked", fl & near["worked"])]:
                geo_rows.append(dict(target=t, method=method, subset=sub, n=int(m.sum()),
                                     geogenic_share=geogenic[m].mean() if m.any() else np.nan,
                                     median_top5_over_C=10 ** np.nanmedian(ratio[m]) if m.any() else np.nan))
            for (dcls, r), nr in near_sets.items():
                row = dict(target=t, method=method, deposit=dcls, radius_km=r, n_flag=int(fl.sum()),
                           frac_all=nr.mean(), frac_flag=nr[fl].mean() if fl.any() else np.nan)
                row["enrichment"] = row["frac_flag"] / row["frac_all"] if row["frac_all"] > 0 else np.nan
                for sname, st in strata.items():
                    obs, exp, p = strat_test(fl, nr, st, rng)
                    tag = "uncond" if sname == "unconditional" else "cond"
                    row[f"exp_{tag}"], row[f"enr_{tag}"], row[f"p_{tag}"] = exp, obs / exp if exp > 0 else np.nan, p
                row["n_near_flag"] = int((fl & nr).sum())
                enr_rows.append(row)
            if not method.startswith("top-"):
                b = pd.cut(geo["worked"][0], BINS, right=False)
                for lab, g in pd.Series(fl).groupby(b, observed=False):
                    decay_rows.append(dict(target=t, method=method, dist_bin_km=str(lab), n_sites=len(g),
                                           n_flag=int(g.sum()), rate=g.mean() if len(g) else np.nan))
        for name, sc in rank.items():
            order = np.argsort(-np.nan_to_num(sc, nan=-np.inf))
            for k in range(10, 310, 10):
                topk_rows.append(dict(target=t, ranking=name, k=k, frac_near_worked=near["worked"][order[:k]].mean(),
                                      frac_near_unworked_only=near_sets[("unworked only", R0)][order[:k]].mean()))
        # per-discovery nearest worked deposit (SI table)
        dw, iw, cw = geo["worked"]
        wk = cls["worked"].reset_index(drop=True)
        for i in np.nonzero(pacd)[0]:
            j = iw[i]
            row = dict(target=t, SiteID=df.index[i], state=df["StateID"].iat[i], landuse=df["landuse"].iat[i],
                       lat=df["Latitude"].iat[i], lon=df["Longitude"].iat[i], score=rank["PACD score"][i],
                       nearest_worked_km=dw[i], n_worked_25km=int(cw[25][i]),
                       nearest_unworked_km=geo["unworked"][0][i])
            if len(els) == 1:
                row.update(top5_mgkg=10 ** Y[els[0]][i], C_mgkg=10 ** C[els[0]][i])
            if j >= 0:
                row.update(mine=wk.site_name[j], mine_state=wk.state[j], mine_status=wk.dev_stat[j],
                           mine_commod=wk.commod1[j], mine_size=wk.prod_size[j])
            disc_rows.append(row)

    enr = pd.DataFrame(enr_rows)
    enr.to_csv(RES / "s17_mrds_enrichment.csv", index=False)
    pd.DataFrame(decay_rows).to_csv(RES / "s17_mrds_distance_decay.csv", index=False)
    pd.DataFrame(topk_rows).to_csv(RES / "s17_mrds_topk_curve.csv", index=False)
    disc = pd.DataFrame(disc_rows)
    disc.to_csv(RES / "s17_mrds_discoveries_nearest_mine.csv", index=False)
    geo = pd.DataFrame(geo_rows)
    geo.to_csv(RES / "s17_mrds_geogenic_share.csv", index=False)

    pd.set_option("display.width", 250)
    h = enr[enr.radius_km == R0]
    for dcls in ["worked", "major worked (M/L)", "unworked only"]:
        print(f"\n== {dcls} deposits within {R0} km: enrichment (unconditional | within C decile x region), p_cond")
        s = h[h.deposit == dcls].set_index(["target", "method"])
        print(s[["n_flag", "frac_all", "frac_flag", "enrichment", "enr_cond", "p_cond"]].round(3).to_string())
    w = h[h.deposit == "worked"].set_index(["target", "method"]).enrichment
    u = h[h.deposit == "unworked only"].set_index(["target", "method"]).enrichment
    print(f"\n== specificity (worked / unworked-only enrichment, {R0} km)\n" + (w / u).round(2).unstack(0).to_string())
    g0 = geo[~geo.method.str.startswith("top-") | geo.method.str.contains("C horizon|PACD")]
    print("\n== geogenic share (C horizon alone has Igeo > 1) and median Top5/C among flags\n"
          + g0.set_index(["target", "method", "subset"]).round(3).unstack("subset").to_string())
    print("\n== PACD discoveries: share within 10 / 25 km of a worked deposit")
    print(disc.groupby("target").agg(n=("SiteID", "size"), within10=("nearest_worked_km", lambda x: (x <= 10).mean()),
                                     within25=("nearest_worked_km", lambda x: (x <= 25).mean()),
                                     median_km=("nearest_worked_km", "median")).round(3).to_string())
