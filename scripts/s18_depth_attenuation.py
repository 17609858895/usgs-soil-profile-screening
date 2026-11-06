"""Vertical-attenuation test of the national PACD discoveries (internal validation, no new data).

The A-horizon sample (0 to A_depth_bot = D cm, median 15 cm) is not used by the anchor; by the
NASGLP field protocol it is a composite of the entire A horizon (whole Ap in ploughed soils). With
lithogenic (immobile-element) normalisation on the log10 scale,
    TEI = Top5 vs C excess = SEI (Top5 vs A) + AEI (A vs C)   (exact decomposition).
If the A sample integrates 0-D cm and an excess E_T (relative to the natural expectation) is held
uniformly to depth h with background below, the A excess is E_A = (h E_T + D - h) / D, so
    h = D (E_A - 1) / (E_T - 1)            (apparent penetration depth, cm).
Any input confined within the 0-5 cm layer gives h = 5 whatever its internal profile; plough
mixing or a whole-A pedogenic excess gives h = D; Top5-only sampling noise gives h = 0.
Predictions for surface-deposited inputs: (P1) SEI exceeds the natural-site distribution while the
geology-only list (top C horizon) does not; (P2) SEI grows with D (dilution); (P3) tillage erases
the gradient (cropland h ~ D).
Selection placebo: the same top-k selection on Top5-over-C excess applied to immobile elements
(leave-one-out lithogenic index), which have no surface source except dust or heavy-mineral lags.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from s06_injection_v2 import UCC, ELS
from s09_national_discovery import landuse

LAYERS = ["Top5", "A", "C"]
UNTILLED = ["natural", "pasture", "urban"]
K_PLACEBO = 50
N_BOOT = 2000


def build_excess(df, nat):
    """Per-layer log10 values and the lithogenic-normalised excess indices (also used by the figures)."""
    els_all = sorted(set(ELS) | set(IMMOBILE))
    L, CEN = {}, {}
    for lay in LAYERS:
        L[lay], CEN[lay] = {}, {}
        for el in els_all:
            v, c = log_el(df, lay, el)
            L[lay][el], CEN[lay][el] = v.to_numpy(), c.to_numpy().astype(bool)

    def li(a, b, excl=None):
        return np.nanmean([L[a][e] - L[b][e] for e in IMMOBILE if e != excl], axis=0)

    def excess(el):
        """Lithogenic-normalised log10 ratios minus the natural-site median (natural expectation)."""
        excl = el if el in IMMOBILE else None
        raw = {"SEI": L["Top5"][el] - L["A"][el] - li("Top5", "A", excl),
               "AEI": L["A"][el] - L["C"][el] - li("A", "C", excl)}
        cens = CEN["Top5"][el] | CEN["A"][el] | CEN["C"][el]
        out = {}
        for k, v in raw.items():
            v = np.where(cens, np.nan, v)
            out[k] = v - np.nanmedian(v[nat])
        out["TEI"] = out["SEI"] + out["AEI"]
        return out

    return excess, li, L


def boot_ci(x, rng, q=(2.5, 97.5)):
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    b = np.median(rng.choice(x, (N_BOOT, len(x))), axis=1)
    return tuple(np.percentile(b, q))


def summarise(name, el, m, ex, D, valid_h, rng, ref_sei):
    sei, aei, tei = ex["SEI"][m], ex["AEI"][m], ex["TEI"][m]
    ok = np.isfinite(sei)
    row = dict(element=el, group=name, n=int(m.sum()), n_valid=int(ok.sum()))
    if ok.sum() == 0:
        return row
    lo, hi = boot_ci(sei, rng)
    row.update(median_TEI_x=10 ** np.nanmedian(tei), median_AEI_x=10 ** np.nanmedian(aei),
               median_SEI_x=10 ** np.nanmedian(sei), SEI_ci_lo=10 ** lo, SEI_ci_hi=10 ** hi,
               p_SEI_gt_ref=mannwhitneyu(sei[ok], ref_sei[np.isfinite(ref_sei)], alternative="greater",
                                         method="asymptotic").pvalue
               if ok.sum() >= 3 else np.nan)
    hm = m & valid_h
    h = D[hm] * (10 ** ex["AEI"][hm] - 1) / (10 ** ex["TEI"][hm] - 1)
    row.update(n_h=int(hm.sum()), median_D=np.median(D[hm]) if hm.any() else np.nan,
               median_h=np.median(h) if hm.any() else np.nan,
               h_q25=np.percentile(h, 25) if hm.any() else np.nan, h_q75=np.percentile(h, 75) if hm.any() else np.nan,
               share_h_lt_2p5=(h < 2.5).mean() if hm.any() else np.nan,
               share_h_gt_0p8D=(h > 0.8 * D[hm]).mean() if hm.any() else np.nan)
    return row, h


if __name__ == "__main__":
    rng = np.random.default_rng(18)
    d = pd.read_csv(RES / "s16_site_discoveries_variants.csv", index_col=0)
    df = load()
    df = df[df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()].copy()
    df["landuse"] = landuse(df)
    df = df[df["landuse"] != "other"]
    assert (df.index.astype(str) == d.index.astype(str)).all()
    lu = df["landuse"].to_numpy()
    nat = lu == "natural"
    D = df["A_depth_bot"].to_numpy(float)
    a_top0 = (df["A_depth_top"].fillna(np.nan) == 0).to_numpy()

    excess, li, L = build_excess(df, nat)

    rows, site_rows, h_pool, dose_rows, till_rows = [], [], {}, [], []
    for el in ELS:
        ex = excess(el)
        y = L["Top5"][el]
        ynat = y[nat]
        mad = 1.4826 * np.median(np.abs(ynat - np.median(ynat)))
        ti = L["Top5"]["Ti"] - L["C"]["Ti"]
        pacd = d[f"disc_main_{el}"].to_numpy().astype(bool)
        valid_h = a_top0 & (D > 5) & (ex["TEI"] >= np.log10(2)) & np.isfinite(ex["AEI"])
        groups = {
            "PACD discoveries": pacd,
            "PACD discoveries, untilled": pacd & np.isin(lu, UNTILLED),
            "PACD discoveries, cropland": pacd & (lu == "cropland"),
            "Igeo > 1": 10 ** y > 3 * UCC[el],
            "Igeo > 1, not PACD": (10 ** y > 3 * UCC[el]) & ~pacd,
            "Median + 2MAD": y > np.median(ynat) + 2 * mad,
            "EF > 2": y - (L["C"][el] + ti) > np.log10(2),
            "EF > 2, not PACD": (y - (L["C"][el] + ti) > np.log10(2)) & ~pacd,
            "Geology only (top 5% C horizon)": L["C"][el] >= np.nanquantile(L["C"][el], 0.95),
        }
        ref = nat & ~pacd
        forest = (df["LandCover1"] == "Forested Upland").to_numpy()
        groups["PACD discoveries, natural forest"] = pacd & nat & forest
        groups["PACD discoveries, natural non-forest"] = pacd & nat & ~forest
        # Top5 can include the humified H horizon while a thick A sample is mineral only, and Top5 has
        # no carbon data; the Top5/A immobile-element dilution (li) is the organic-matter proxy.
        # Regress SEI on it over natural reference sites and remove that part (organic-adjusted).
        dil = li("Top5", "A", el if el in IMMOBILE else None)
        ok = ref & np.isfinite(ex["SEI"]) & np.isfinite(dil)
        beta = np.polyfit(dil[ok] - np.median(dil[ok]), ex["SEI"][ok], 1)[0]
        ex_adj = dict(ex)
        ex_adj["SEI"] = ex["SEI"] - beta * (dil - np.median(dil[ok]))
        ex_adj["TEI"] = ex_adj["SEI"] + ex["AEI"]
        valid_adj = a_top0 & (D > 5) & (ex_adj["TEI"] >= np.log10(2)) & np.isfinite(ex["AEI"])
        out = summarise("PACD discoveries, organic-adjusted", el, pacd, ex_adj, D, valid_adj, rng, ex_adj["SEI"][ref])
        rows.append({**(out[0] if isinstance(out, tuple) else out), "beta_dilution": beta})
        for name, m in groups.items():
            out = summarise(name, el, m, ex, D, valid_h, rng, ex["SEI"][ref])
            if isinstance(out, tuple):
                row, h = out
                if name == "PACD discoveries":
                    h_pool[el] = h
            else:
                row = out
            rows.append(row)
        # P3 positive control for the test itself: reference sites by land use
        for g in ["natural", "pasture", "urban", "cropland"]:
            m = (lu == g) & ~pacd & np.isfinite(ex["SEI"])
            till_rows.append(dict(element=el, landuse=g, n=int(m.sum()), median_SEI_x=10 ** np.median(ex["SEI"][m]),
                                  rho_SEI_D=spearmanr(ex["SEI"][m & a_top0], D[m & a_top0]).statistic))
        for i in np.nonzero(pacd)[0]:
            site_rows.append(dict(element=el, SiteID=df.index[i], state=df["StateID"].iat[i], landuse=lu[i],
                                  top5=10 ** L["Top5"][el][i], A=10 ** L["A"][el][i], C=10 ** L["C"][el][i],
                                  A_depth_top=df["A_depth_top"].iat[i], A_depth_bot=D[i],
                                  TEI_x=10 ** ex["TEI"][i], SEI_x=10 ** ex["SEI"][i], AEI_x=10 ** ex["AEI"][i],
                                  h_cm=D[i] * (10 ** ex["AEI"][i] - 1) / (10 ** ex["TEI"][i] - 1) if valid_h[i] else np.nan))

    # selection placebo on immobile elements: top-K by Top5-over-C excess (leave-one-out index)
    for el in IMMOBILE:
        ex = excess(el)
        valid_h = a_top0 & (D > 5) & (ex["TEI"] >= np.log10(2)) & np.isfinite(ex["AEI"])
        top = np.zeros(len(df), bool)
        top[np.argsort(-np.nan_to_num(ex["TEI"], nan=-np.inf))[:K_PLACEBO]] = True
        out = summarise(f"Placebo: top-{K_PLACEBO} Top5/C excess", el, top, ex, D, valid_h, rng, ex["SEI"][nat])
        row, h = out if isinstance(out, tuple) else (out, np.array([]))
        rows.append(row)
        h_pool[f"placebo_{el}"] = h

    res = pd.DataFrame(rows)
    res.to_csv(RES / "s18_depth_groups.csv", index=False)
    sites = pd.DataFrame(site_rows)
    sites.to_csv(RES / "s18_depth_discovery_sites.csv", index=False)
    till = pd.DataFrame(till_rows)
    till.to_csv(RES / "s18_depth_reference_by_landuse.csv", index=False)

    # P2 dose-response, pooled over element-discovery pairs (SEI already centred per element)
    for name, s in [("PACD, untilled", sites[sites.landuse.isin(UNTILLED)]), ("PACD, cropland", sites[sites.landuse == "cropland"])]:
        s = s[(s.A_depth_top == 0) & np.isfinite(np.log10(s.SEI_x))]
        if len(s) >= 5:
            r = spearmanr(np.log10(s.SEI_x), s.A_depth_bot)
            dose_rows.append(dict(group=name, n=len(s), rho=r.statistic, p=r.pvalue))
    pd.DataFrame(dose_rows).to_csv(RES / "s18_depth_doseresponse.csv", index=False)

    pd.set_option("display.width", 250)
    cols = ["n", "n_valid", "median_TEI_x", "median_AEI_x", "median_SEI_x", "SEI_ci_lo", "SEI_ci_hi", "p_SEI_gt_ref",
            "n_h", "median_D", "median_h", "h_q25", "h_q75", "share_h_lt_2p5", "share_h_gt_0p8D"]
    show = res[res.n > 0]
    for el in ["Pb", "Sb", "Zn", "Cd"]:
        print(f"\n== {el}\n" + show[show.element == el].set_index("group")[cols].round(3).to_string())
    print("\n== placebo (immobile elements)\n" + show[show.group.str.startswith("Placebo")].set_index("element")[cols].round(3).to_string())
    hp = np.concatenate([h_pool[e] for e in ["Pb", "Sb", "Zn", "Cd"] if e in h_pool])
    hpl = np.concatenate([h_pool[k] for k in h_pool if k.startswith("placebo_")])
    print(f"\n== pooled h (cm): PACD Pb/Sb/Zn/Cd n={len(hp)} median {np.median(hp):.1f} IQR {np.percentile(hp, 25):.1f}-{np.percentile(hp, 75):.1f}; "
          f"placebo n={len(hpl)} median {np.median(hpl):.1f} IQR {np.percentile(hpl, 25):.1f}-{np.percentile(hpl, 75):.1f}")
    print("\n== P2 dose-response\n" + pd.DataFrame(dose_rows).round(3).to_string(index=False))
    print("\n== P3 reference sites by land use (median SEI ratio; Spearman SEI vs A depth)\n"
          + till.pivot_table(index="element", columns="landuse", values=["median_SEI_x", "rho_SEI_D"]).round(3).to_string())
    print("\n== PACD discoveries by land use (h, cm)\n"
          + sites.groupby(["element", "landuse"]).agg(n=("SiteID", "size"), SEI_x=("SEI_x", "median"),
                                                      AEI_x=("AEI_x", "median"), h=("h_cm", "median"),
                                                      D=("A_depth_bot", "median")).round(2).to_string())
