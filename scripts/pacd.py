"""PACD core: conformal p-values with EVT tail extension and self-cleaning of a contaminated
reference set.

conf_p(cal, test)                      one-sided split-conformal p-value
evt_p(cal, test)                       identical to conf_p for test <= max(cal); beyond the largest
                                       calibration score the p-value is extrapolated with a robust
                                       (right-truncated) generalised Pareto tail and capped at
                                       1/(n+1), so it can only refine the conformal floor.
loo_p(cal)                             leave-one-out conformal p-values of calibration points
clean_reference(cal, ...)              BH discoveries among calibration points are removed
                                       (defaults = CLEAN_V2: BH 0.01, iterated; CLEAN_V1 = s14-s19 rule)
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import genpareto

LO_Q, HI_Q = 0.85, 0.97  # robust tail band: shape/scale estimated from these quantiles only


def conf_p(cal, test):
    cal = np.sort(np.asarray(cal))
    return (1 + len(cal) - np.searchsorted(cal, test, side="left")) / (len(cal) + 1)


def _gpd(cal, tail):
    u = np.quantile(cal, 1 - tail)
    exc = cal[cal > u] - u
    xi, _, sigma = genpareto.fit(exc, floc=0)
    return u, xi, sigma, len(exc) / len(cal)


def robust_tail(cal, lo_q=LO_Q, hi_q=HI_Q):
    """Right-truncated GPD fit to exceedances of the lo_q quantile that lie below hi_q.
    Robustness to contamination is evaluated empirically; this fit supplies no finite-sample guarantee."""
    cal = np.asarray(cal)
    u, v = np.quantile(cal, [lo_q, hi_q])
    e = cal[(cal > u) & (cal <= v)] - u
    t = v - u

    def nll(th):
        xi, ls = th
        sig = np.exp(ls)
        Ft = genpareto.cdf(t, xi, scale=sig)
        if Ft <= 1e-12:
            return 1e12
        return -(genpareto.logpdf(e, xi, scale=sig).sum() - len(e) * np.log(Ft))

    r = minimize(nll, x0=[0.1, np.log(e.std() + 1e-6)], method="Nelder-Mead",
                 bounds=None, options=dict(maxiter=2000, xatol=1e-6, fatol=1e-8))
    xi, sig = float(np.clip(r.x[0], 0.0, 1.0)), float(np.exp(r.x[1]))  # xi >= 0: never lighter than exponential
    return u, xi, sig, 1 - lo_q


def tail_sf(s, fit):
    u, xi, sig, frac = fit
    return frac * genpareto.sf(np.asarray(s) - u, xi, scale=sig)


def evt_p(cal, test, fit=None):
    """Conformal p inside the calibration range; robust-GPD extrapolation (capped at the
    conformal floor) beyond the largest calibration score."""
    cal = np.asarray(cal)
    test = np.asarray(test)
    p = conf_p(cal, test)
    hi = test > cal.max()
    if hi.any():
        fit = fit or robust_tail(cal)
        p[hi] = np.minimum(tail_sf(test[hi], fit), 1.0 / (len(cal) + 1))
    return np.clip(p, 1e-12, 1)


def bh(p, q=0.10):
    p = np.asarray(p)
    m = len(p)
    o = np.argsort(p)
    ok = np.nonzero(p[o] <= q * np.arange(1, m + 1) / m)[0]
    rej = np.zeros(m, bool)
    if len(ok):
        rej[o[: ok.max() + 1]] = True
    return rej


def loo_p(cal):
    cal = np.asarray(cal)
    # Include every tied score in the upper tail. The self-count supplies the +1
    # in a leave-one-out rank p-value, so the numerator is the total count >= cal_i.
    return (len(cal) - np.searchsorted(np.sort(cal), cal, side="left")) / len(cal)


# v1 (s14-s19): BH 0.10, one pass. s20 showed a clean-side FDR cost (+0.036 single elements over the
# clean-reference oracle, 30 reps). v2 (main since s20): BH 0.01, iterated to a fixed point; selected on
# reps 0-9 by the pre-set rule (r = 0 FDR <= 0.10, then best power at 1-3 % contamination).
CLEAN_V1 = dict(q=0.10, rule="bh", band=(LO_Q, HI_Q), iterate=False)
CLEAN_V2 = dict(q=0.01, rule="bh", band=(LO_Q, HI_Q), iterate=True)


def clean_reference(cal, q=0.01, use_evt=True, rule="bh", band=(LO_Q, HI_Q), backfill=False, iterate=True):
    """Self-cleaning of a contaminated reference: every calibration point above the band top gets
    a p-value from the robust tail fitted on the band (masking-resistant); the rest get
    leave-one-out conformal p-values. Points rejected by BH at q (rule="bh") or by Bonferroni at
    family-wise level q (rule="bonferroni") are removed. With backfill, ceil(q * R) of the R
    removed points with the lowest scores are restored (FDR budget). With iterate, the rule is
    re-applied to the retained points (tail refitted) until nothing more is removed, which
    unmasks contamination that sat inside the fitting band. Defaults = CLEAN_V2; pass
    **CLEAN_V1 for the s14-s19 rule. Returns a keep-mask."""
    cal = np.asarray(cal)
    if iterate:
        keep = np.ones(len(cal), bool)
        for _ in range(10):
            k2 = clean_reference(cal[keep], q, use_evt, rule, band, backfill, iterate=False)
            if k2.all():
                break
            idx = np.nonzero(keep)[0]
            keep[idx[~k2]] = False
        return keep
    p = loo_p(cal)
    if use_evt:
        fit = robust_tail(cal, *band)
        hi = cal > np.quantile(cal, band[1])
        p[hi] = np.minimum(p[hi], tail_sf(cal[hi], fit))
    rej = bh(p, q) if rule == "bh" else p <= q / len(p)
    if backfill and rej.any():
        idx = np.nonzero(rej)[0]
        rej[idx[np.argsort(cal[idx])][: int(np.ceil(q * len(idx)))]] = False
    return ~rej
