"""Plan B: Label-Trim with imperfect, evidence-based labels.

Bashari et al. (2025) assume correct labels for the m highest reference scores. In the national data
the labels come from independent evidence (distance to matching MRDS sources), so they are noisy:
  - a contaminated reference point is labelled with sensitivity s (only source-attributable
    contamination is recognisable);
  - a clean point is labelled with false-positive rate f; under the null the evidence is independent of
    the score, so f is the share of natural sites that meet the evidence rule (0.1-1.2 % for the rule
    fixed in docs/contaminated_reference_comparison.md section 9).
Same design, cached scores and injections as s20 (identical rng sequence), so results are paired with
s20. Label draws use a separate rng. Methods: Label-Trim with noisy labels (LT) and its union with the
v2 self-cleaning rule (v2 + LT).
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from pacd import conf_p, bh, clean_reference, CLEAN_V2
from s15_contaminated_reference import ELS, SIGS, ev
from s19_trim_comparison import M_LABEL
from s20_clean_variants import cached_scores, summary, N_REP

REGIMES = [(0.0, 10)] + [(r, kr) for r in (0.01, 0.03, 0.05, 0.08) for kr in (3, 10)]
SENS = (0.3, 0.6)
FPR = (0.01, 0.03, 0.05)


def run(rep, nat, X, g):
    ca, te, Z0 = cached_scores(rep, nat, X, g)
    rng = np.random.default_rng(2000 + rep)  # same sequence as s20
    rl = np.random.default_rng(5000 + rep)   # label noise
    ca_idx, te_idx = np.nonzero(ca)[0], np.nonzero(te)[0]
    targets = [(el, [el]) for el in ELS] + [(f"sig:{k}", v) for k, v in SIGS.items()]
    rows = []
    for r, k_ref in REGIMES:
        for name, els in targets:
            ref_bad = rng.choice(ca_idx, int(round(r * len(ca_idx))), replace=False)
            bad = np.isin(ca_idx, ref_bad)
            base = np.mean([(Z0[el][0] + np.isin(np.arange(len(ca)), ref_bad) * np.log10(k_ref)) / Z0[el][1]
                            for el in els], axis=0)
            cal = base[ca]
            top = np.argsort(cal)[-M_LABEL:]
            u = rl.uniform(size=M_LABEL)
            v2 = clean_reference(cal, use_evt=True, **CLEAN_V2)
            keep = {"Oracle": ~bad, "Standard": np.ones(len(cal), bool), "v2": v2}
            for s in (1.0,) + SENS:
                for f in ((0.0,) if s == 1.0 else FPR):
                    lab = np.where(bad[top], u < s, u < f)
                    k_lt = np.ones(len(cal), bool)
                    k_lt[top[lab]] = False
                    tag = "LT exact" if s == 1.0 else f"LT s={s:.1f} f={f:.2f}"
                    keep[tag] = k_lt
                    keep["v2 + " + tag] = k_lt & v2
            for k in [1, 3, 5, 10]:
                inj = rng.choice(te_idx, int(.10 * len(te_idx)), replace=False) if k > 1 else np.array([], int)
                Z = base.copy()
                Z[inj] += np.mean([np.log10(k) / Z0[el][1][inj] for el in els], axis=0)
                truth = np.isin(te_idx, inj)
                tst = Z[te]
                for m, kp in keep.items():
                    p = conf_p(cal[kp], tst)
                    rows.append(dict(rep=rep, r=r, k_ref=k_ref, target=name, k=k, method=m,
                                     clean_removed=int((~kp & ~bad).sum()), bad_removed=int((~kp & bad).sum()),
                                     typeI=(p[~truth] <= 0.01).mean(), **ev(bh(p), truth)))
    return rows


if __name__ == "__main__":
    df = load()
    nat = df[df["LandCover1"].isin(NATURAL) & df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    X = dual_features(nat)
    g = spatial_groups(nat, 120)
    out = Parallel(n_jobs=6)(delayed(run)(i, nat, X, g) for i in range(N_REP))
    res = pd.DataFrame([row for rows in out for row in rows])
    res.to_csv(RES / "s21_label_noise_raw.csv", index=False)
    pd.set_option("display.width", 300)
    s = summary(res)
    alt = res[res["k"] > 1]
    t = alt[alt["r"] == 0].groupby(["rep", "method"])["FDR"].mean().unstack()
    exc = (t.sub(t["Oracle"], axis=0)).agg(["mean", "sem"]).T.round(3)
    print("== all 30 reps\n" + s.round(3).to_string())
    print("\n== r = 0 FDR excess over Oracle (single + signature, paired over reps)\n" + exc.to_string())
