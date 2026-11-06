"""National PACD variants recomputed from the s14 scores (no refitting).

Variants: plain cross-conformal | self-cleaned with the v2 rule (BH 0.01, iterated, chosen in s20) |
self-cleaned with the v1 rule (BH 0.10, one pass; used up to s19) | evidence-labelled Label-Trim
(s22 extended MRDS + EPA rule) | v2 + evidence labels (union of removals) | v2 + MRDS-only labels (SI
sensitivity; the plan-B rule of comparison doc section 9). MAIN names the variant copied to the
`disc_main_*` / `p_main_*` columns that s17, s18 and the figures read. The EVT tail extension is
dropped (s15/s19: P(any discovery) 0.30-0.35 under a clean reference).
Natural sites are tested leave-fold-out (same spatial folds as s14); the other land uses are tested
against the full natural reference, BH within each land-use family at q = 0.10.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from pacd import conf_p, bh, clean_reference, CLEAN_V1, CLEAN_V2
from s09_national_discovery import landuse
from s22_evidence_labels import evidence_labels

Q = 0.10
FAMS = ["urban", "cropland", "pasture", "orchard"]
VARIANTS = {"plain": (None, None), "cleaned": (CLEAN_V2, None), "cleaned_v1": (CLEAN_V1, None),
            "labelled": (None, "ext"), "cleaned_labelled": (CLEAN_V2, "ext"),
            "cleaned_labelled_mrds": (CLEAN_V2, "mrds")}
MAIN = "cleaned_labelled"  # v2 + extended evidence labels (contaminated_reference_comparison.md sections 9-10)


def keep_mask(ref, rule, lab=None):
    keep = np.ones(len(ref), bool) if rule is None else clean_reference(ref, use_evt=True, **rule)
    return keep if lab is None else keep & ~lab


if __name__ == "__main__":
    d = pd.read_csv(RES / "s14_site_pvalues_discoveries.csv", index_col=0)
    df = load()
    df = df[df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()].copy()
    df["landuse"] = landuse(df)
    df = df[df["landuse"] != "other"]
    assert (df.index.astype(str) == d.index.astype(str)).all()
    nat_idx = np.nonzero((df["landuse"] == "natural").to_numpy())[0]
    g = spatial_groups(df, 150)
    cal_sets = [nat_idx[te] for _, te in GroupKFold(5).split(nat_idx, groups=g[nat_idx])]
    scores = [c[len("score_"):] for c in d.columns if c.startswith("score_")]
    labels = {rule: evidence_labels(d, rule)[0] for rule in ["ext", "mrds"]}
    rows, removed = [], []
    for name in scores:
        s = d[f"score_{name}"].to_numpy()
        for variant, (rule, lab_rule) in VARIANTS.items():
            lab_all = labels[lab_rule][name] if lab_rule else None
            p_all = np.full(len(d), np.nan)
            for k, c in enumerate(cal_sets):
                ri = np.concatenate([cc for j, cc in enumerate(cal_sets) if j != k])
                ref = s[ri]
                p_all[c] = conf_p(ref[keep_mask(ref, rule, lab_all[ri] if lab_rule else None)], s[c])
            ref = s[nat_idx]
            keep = keep_mask(ref, rule, lab_all[nat_idx] if lab_rule else None)
            removed.append(dict(score=name, variant=variant, n_ref=len(ref), n_removed=int((~keep).sum()),
                                min_removed_score=float(ref[~keep].min()) if (~keep).any() else np.nan))
            for fam in FAMS:
                idx = np.nonzero((d["landuse"] == fam).to_numpy())[0]
                p_all[idx] = conf_p(ref[keep], s[idx])
            disc = np.zeros(len(d), bool)
            for fam in ["natural"] + FAMS:
                idx = np.nonzero((d["landuse"] == fam).to_numpy())[0]
                disc[idx] = bh(p_all[idx], Q)
                rows.append(dict(score=name, variant=variant, family=fam, n=len(idx), n_disc=int(disc[idx].sum()),
                                 rate=disc[idx].mean()))
            d[f"p_{variant}_{name}"] = p_all
            d[f"disc_{variant}_{name}"] = disc
            if variant == MAIN:
                d[f"p_main_{name}"] = p_all
                d[f"disc_main_{name}"] = disc
    out = pd.DataFrame(rows)
    out.to_csv(RES / "s16_national_variants.csv", index=False)
    rem = pd.DataFrame(removed)
    rem.to_csv(RES / "s16_reference_removed.csv", index=False)
    d.to_csv(RES / "s16_site_discoveries_variants.csv")
    pd.set_option("display.width", 220)
    print(out.pivot_table(index="score", columns=["variant", "family"], values="n_disc").astype(int).to_string())
    print("\nTotal discoveries\n" + out.groupby(["score", "variant"]).n_disc.sum().unstack()[list(VARIANTS)].to_string())
    print("\nReference points removed (full natural reference)\n"
          + rem.pivot(index="score", columns="variant", values="n_removed")[list(VARIANTS)].to_string())
