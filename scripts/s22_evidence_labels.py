"""Evidence labels for Label-Trim on the national natural reference.

Rules fixed in docs/contaminated_reference_comparison.md before any national result was computed:
  section 9 (MRDS, plan B)  E1 worked deposit of matching commodity <= 2 km; E2 worked M/L deposit <= 5 km;
                            E3 plant <= 10 km.
  section 10 (EPA, extended) E4 smelting-related SEMS site <= d4; E5 NPL site with the element as a soil/sediment
                            contaminant of concern <= d5; E6 TRI facility with air releases of the element <= d6.
                            (d4, d5, d6) = (10, 2, 5) km, stepped down per score to (5, 1, 2) and (2, 0.5, 1) km
                            when the base rate exceeds F_CAP; MRDS rules only if every step exceeds it.
For each score, the m = 50 natural sites with the highest score are candidates; a candidate meeting any rule is
labelled contaminated, everything else is treated as clean (Bashari et al. 2025 default).

evidence_labels(d, rule="ext"|"mrds") -> (labels, audit, base): labels[score] is a boolean mask over the rows of
d, audit lists every candidate with its evidence, base gives the share of all natural sites meeting the rule
(a source-proximity rule-match prevalence, not an identified false-label probability) and the distances used.
Run directly to write tables/s22_* for both rule sets.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from s06_injection_v2 import SIGS, ELS
from s17_mrds_validation import COMMOD, R_EARTH

M_CAND = 50
F_CAP = 0.05  # cap on source-proximity rule-match share among natural sites
RULES = [("E1", ["Producer", "Past Producer"], None, 2.0),
         ("E2", ["Producer", "Past Producer"], ["M", "L"], 5.0),
         ("E3", ["Plant"], None, 10.0)]
EPA_RULES = [("E4", "SMELT"), ("E5", "NPL"), ("E6", "TRI")]
STEPS = [(10.0, 2.0, 5.0), (5.0, 1.0, 2.0), (2.0, 0.5, 1.0)]
MEMBERS = {**{el: [el] for el in ELS}, **{f"sig_{k}": v for k, v in SIGS.items()}, "any_max": ELS}


def _nearest(lat, lon, rad):
    if len(lat) == 0:
        return np.full(len(rad), np.inf), np.full(len(rad), -1)
    t = BallTree(np.radians(np.c_[lat, lon]), metric="haversine")
    dd, ii = t.query(rad, k=1)
    return dd[:, 0] * R_EARTH, ii[:, 0]


def evidence_mrds(rad, mrds, score):
    rx = "|".join(sorted({COMMOD[e] for e in MEMBERS[score]}))
    mc = mrds[mrds["commod"].str.contains(rx, na=False)]
    hit = np.zeros(len(rad), bool)
    desc = np.full(len(rad), "", dtype=object)
    for tag, stat, size, R in RULES:
        dep = mc[mc["dev_stat"].isin(stat)]
        if size:
            dep = dep[dep["prod_size"].isin(size)]
        dist, idx = _nearest(dep["latitude"].to_numpy(), dep["longitude"].to_numpy(), rad)
        new = (dist <= R) & ~hit
        for i in np.nonzero(new)[0]:
            row = dep.iloc[idx[i]]
            desc[i] = f"{tag}: {row['site_name']} ({row['state']}; {row['dev_stat']}; {row['commod1']}) {dist[i]:.1f} km"
        hit |= dist <= R
    return hit, desc


def epa_distances(rad, epa, score):
    """Distance to the nearest EPA source of each type with a matching element, and its description."""
    e = epa[epa["element"].isin(MEMBERS[score])].drop_duplicates(["source", "site_id"])
    out = {}
    for tag, src in EPA_RULES:
        s = e[e["source"] == src]
        dist, idx = _nearest(s["latitude"].to_numpy(), s["longitude"].to_numpy(), rad)
        out[tag] = (dist, s, idx)
    return out


def evidence(rad, mrds, epa, score, radii):
    hit, desc = evidence_mrds(rad, mrds, score)
    if radii is None:
        return hit, desc
    for (tag, (dist, s, idx)), R in zip(epa_distances(rad, epa, score).items(), radii):
        new = (dist <= R) & ~hit
        for i in np.nonzero(new)[0]:
            row = s.iloc[idx[i]]
            desc[i] = f"{tag}: {row['name']} ({row['state']}; {row['source']}; {row['detail']}) {dist[i]:.1f} km"
        hit = hit | (dist <= R)
    return hit, desc


def evidence_labels(d, rule="ext", mrds=None, epa=None):
    mrds = pd.read_parquet(ROOT / "data_derived" / "mrds_conus.parquet") if mrds is None else mrds
    if rule == "ext" and epa is None:
        epa = pd.read_parquet(ROOT / "data_derived" / "evidence_sources.parquet")
    nat = (d["landuse"] == "natural").to_numpy()
    nat_rows = np.nonzero(nat)[0]
    rad = np.radians(d.loc[nat, ["Latitude", "Longitude"]].to_numpy())
    scores = [c[len("score_"):] for c in d.columns if c.startswith("score_")]
    labels, audit, base = {}, [], []
    for sc in scores:
        radii, step = None, "MRDS only"
        if rule == "ext":
            for k, r in enumerate(STEPS):
                if evidence(rad, mrds, epa, sc, r)[0].mean() <= F_CAP:
                    radii, step = r, k
                    break
        hit, desc = evidence(rad, mrds, epa, sc, radii)
        base.append(dict(score=sc, base_rate=hit.mean(), step=step,
                         d4=radii[0] if radii else np.nan, d5=radii[1] if radii else np.nan,
                         d6=radii[2] if radii else np.nan))
        s = d.loc[nat, f"score_{sc}"].to_numpy()
        cand = np.argsort(s)[-M_CAND:]
        lab = np.zeros(len(d), bool)
        lab[nat_rows[cand[hit[cand]]]] = True
        labels[sc] = lab
        for rank, j in enumerate(cand[::-1], 1):
            audit.append(dict(score=sc, rank=rank, site=d.index[nat_rows[j]], state=d["StateID"].iloc[nat_rows[j]],
                              score_value=s[j], labelled=bool(hit[j]), evidence=desc[j]))
    return labels, pd.DataFrame(audit), pd.DataFrame(base).set_index("score")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    d = pd.read_csv(RES / "s14_site_pvalues_discoveries.csv", index_col=0)
    summ = []
    for rule in ["mrds", "ext"]:
        labels, audit, base = evidence_labels(d, rule)
        audit.insert(0, "rule", rule)
        audit.to_csv(RES / f"s22_evidence_candidates{'' if rule == 'ext' else '_mrds'}.csv", index=False)
        base["n_labelled"] = pd.Series({k: int(v.sum()) for k, v in labels.items()})
        base["expected_rule_matches"] = (base["base_rate"] * M_CAND).round(2)
        base["base_rate_pct"] = (base["base_rate"] * 100).round(2)
        summ.append(base.drop(columns="base_rate").assign(rule=rule))
        print(f"\n== rule {rule}\n" + base.drop(columns="base_rate").to_string())
        print("\nLabelled candidates\n" + audit[audit["labelled"]].drop(columns="rule").to_string(index=False))
    pd.concat(summ).to_csv(RES / "s22_evidence_summary.csv")
