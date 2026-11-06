"""Plan A: tighten PACD self-cleaning to remove the clean-side cost found in s19.

Variants of pacd.clean_reference are compared on the s19 design (dual anchor, spatial splits,
reference contamination r in {0, 1, 3, 5, 8} %, strength k_ref in {3, 10}). Reps 0-9 select the
variant; reps 10-19 were inspected while adding the iterative variants, so reps 20-29 (fresh
splits) give the final confirmation. Acceptance: with an uncontaminated reference the BH
FDR stays <= q = 0.10, and power at r = 1-3 % is not markedly lower than the current rule.
Anchor scores are cached per rep in data_derived/cache_s19/ so variants never refit models.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
sys.path.insert(0, str(Path(__file__).parent))
from common import *
from pacd import conf_p, bh, clean_reference
from s15_contaminated_reference import ELS, SIGS, ev
from s19_trim_comparison import fit_scores, M_LABEL

N_REP = 30
CACHE = ROOT / "data_derived" / "cache_s19"
REGIMES = [(0.0, 10)] + [(r, kr) for r in (0.01, 0.03, 0.05, 0.08) for kr in (3, 10)]
VARIANTS = {
    "BH 0.10, 85-97 (current)": dict(q=0.10, rule="bh", band=(0.85, 0.97)),
    "BH 0.10 + backfill": dict(q=0.10, rule="bh", band=(0.85, 0.97), backfill=True),
    "BH 0.01, 85-97": dict(q=0.01, rule="bh", band=(0.85, 0.97)),
    "Bonf 0.10, 85-97": dict(q=0.10, rule="bonferroni", band=(0.85, 0.97)),
    "BH 0.01, 80-92": dict(q=0.01, rule="bh", band=(0.80, 0.92)),
    "Bonf 0.10, 80-92": dict(q=0.10, rule="bonferroni", band=(0.80, 0.92)),
    "Bonf 0.10, 75-90": dict(q=0.10, rule="bonferroni", band=(0.75, 0.90)),
    "BH 0.01, 85-97, iter": dict(q=0.01, rule="bh", band=(0.85, 0.97), iterate=True),
    "BH 0.02, 85-97, iter": dict(q=0.02, rule="bh", band=(0.85, 0.97), iterate=True),
    "BH 0.01, 80-92, iter": dict(q=0.01, rule="bh", band=(0.80, 0.92), iterate=True),
}


def cached_scores(rep, nat, X, g):
    f = CACHE / f"rep{rep:02d}.npz"
    if f.exists():
        z = np.load(f)
        return z["ca"], z["te"], {el: (z[f"res_{el}"], z[f"s_{el}"]) for el in ELS}
    ca, te, Z0 = fit_scores(rep, nat, X, g)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f, ca=ca, te=te, **{f"res_{el}": Z0[el][0] for el in ELS},
                        **{f"s_{el}": Z0[el][1] for el in ELS})
    return ca, te, Z0


def run(rep, nat, X, g):
    ca, te, Z0 = cached_scores(rep, nat, X, g)
    rng = np.random.default_rng(2000 + rep)
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
            order = np.argsort(cal)
            keep = {"Oracle": ~bad, "Standard": np.ones(len(cal), bool)}
            kl = np.ones(len(cal), bool)
            top = order[-M_LABEL:]
            kl[top[bad[top]]] = False
            keep["Label-Trim"] = kl
            for vname, kw in VARIANTS.items():
                # Candidate rules without an explicit iteration option are single-pass.
                # Keep this explicit when the shared function's main-method default changes.
                keep[vname] = clean_reference(cal, use_evt=True, **{"iterate": False, **kw})
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


def summary(res):
    res = res.assign(kind=np.where(res["target"].str.startswith("sig"), "signature", "single"))
    alt = res[res["k"] > 1]
    out = {}
    r0 = alt[alt["r"] == 0]
    out["r0 FDR single"] = r0[r0["kind"] == "single"].groupby("method")["FDR"].mean()
    out["r0 FDR signature"] = r0[r0["kind"] == "signature"].groupby("method")["FDR"].mean()
    out["r0 FDR worst target"] = r0.groupby(["method", "target"])["FDR"].mean().groupby("method").max()
    n0 = res[(res["k"] == 1) & (res["r"] == 0)]
    out["r0 P(any) null"] = n0.assign(a=n0["n_disc"] > 0).groupby("method")["a"].mean()
    out["r0 clean removed"] = n0.groupby("method")["clean_removed"].mean()
    out["FDR worst regime"] = alt.groupby(["method", "r", "k_ref", "kind"])["FDR"].mean().groupby("method").max()
    for r in (0.01, 0.03, 0.05, 0.08):
        for kr in (3, 10):
            t = alt[(alt["r"] == r) & (alt["k_ref"] == kr)]
            out[f"power r={r:.0%} k_ref={kr}"] = t.groupby("method")["power"].mean()
    return pd.DataFrame(out).T


if __name__ == "__main__":
    df = load()
    nat = df[df["LandCover1"].isin(NATURAL) & df["C_Quartz"].notna() & df["C_Pb"].notna() & df["Top5_Pb"].notna()]
    X = dual_features(nat)
    g = spatial_groups(nat, 120)
    out = Parallel(n_jobs=6)(delayed(run)(i, nat, X, g) for i in range(N_REP))
    res = pd.DataFrame([row for rows in out for row in rows])
    res.to_csv(RES / "s20_clean_variants_raw.csv", index=False)
    pd.set_option("display.width", 260)
    cols = ["Oracle", "Standard", "Label-Trim"] + list(VARIANTS)
    for lab, reps in [("SELECTION reps 0-9", range(10)), ("CHECK reps 10-19 (inspected)", range(10, 20)),
                      ("FRESH CONFIRMATION reps 20-29", range(20, 30))]:
        s = summary(res[res["rep"].isin(reps)])
        print(f"\n== {lab}\n" + s[cols].round(3).to_string())
