"""
Analysis of the PTRS Poisson variance-inflation bug, with an inflation-free
reimplementation.

numpy draws Poisson (lam>=10) with PTRS: propose from a fat algebraic hat, trim
to the Poisson shape with ONE acceptance test whose right-hand side is the
log-pmf
        R(k) = -lam + k*log(lam) - logGamma(k+1).
At large lam this is a difference of two ~ k*log k ~ 1e20 terms, each with
float64 spacing ulp ~ 1e5, so the O(1) information that shapes the tails
(dR/dk ~ -0.5 per standard deviation near the mode) is annihilated by
catastrophic cancellation. The hat's heavy tail is then not trimmed and the
sampled variance inflates to ~1.5-1.75 * lam (the mean stays correct).

FIX (this file, `logpmf_stable`): evaluate the SAME R(k) in mode-relative form.
With delta = k - lam and Stirling for log(k!),
    R(k) = (k-lam) - k*log(k/lam) - 0.5*log(2*pi*k) - 1/(12k) + ...
         = [delta - k*log1p(delta/lam)] - 0.5*log(2*pi*k) - ...
The bracket equals -delta^2/(2 lam) + O(delta^3/lam^2); computed as a difference
of two ~delta quantities it carries only ulp(delta) ~ 1e-7 absolute error, so the
-0.5/std signal is preserved. No logGamma of a huge argument is ever formed.

The rejection loop is IDENTICAL for the naive and stable variants; only the
log-pmf evaluator is swapped, isolating the fix. `logpmf_hi` (mpmath) is an
exact oracle used to score both.
"""
import math
import numpy as np
from scipy.special import gammaln

try:
    import mpmath as mp
    mp.mp.dps = 50
    HAVE_MP = True
except Exception:
    HAVE_MP = False


# --------------------------------------------------------------------------- #
# PTRS with a pluggable log-pmf.  Vectorized rejection: the algorithm (hat,
# squeeze, fast-reject, accept test) is exactly numpy's; only `logpmf` changes.
# --------------------------------------------------------------------------- #
def _consts(lam):
    slam = math.sqrt(lam); loglam = math.log(lam)
    b = 0.931 + 2.53 * slam
    a = -0.059 + 0.02483 * b
    invalpha = 1.1239 + 1.1328 / (b - 3.4)
    vr = 0.9277 - 3.6224 / (b - 2)
    return slam, loglam, a, b, invalpha, vr


def ptrs(lam, N, rng, logpmf):
    slam, loglam, a, b, invalpha, vr = _consts(lam)
    out = np.empty(N); filled = 0
    while filled < N:
        M = int((N - filled) * 1.6) + 64
        U = rng.random(M) - 0.5
        V = rng.random(M)
        us = 0.5 - np.abs(U)
        with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
            k = np.floor((2 * a / us + b) * U + lam + 0.43)
            sq = (us >= 0.07) & (V <= vr)                       # fast squeeze accept
            rej = (k < 0) | ((us < 0.013) & (V > us))           # fast reject
            rest = (~sq) & (~rej)
            lhs = np.log(V) + math.log(invalpha) - np.log(a / (us * us) + b)
            rhs = logpmf(k, lam, loglam)                        # <-- the only difference
            acc = sq | (rest & (lhs <= rhs))
        ak = k[acc]
        take = min(ak.size, N - filled)
        out[filled:filled + take] = ak[:take]
        filled += take
    return out


# --------------------------------------------------------------------------- #
# three evaluators of R(k) = -lam + k*log(lam) - logGamma(k+1)
# --------------------------------------------------------------------------- #
def logpmf_naive(k, lam, loglam):
    # numpy's exact form: forms logGamma(~1e18) ~ 1e20 then subtracts -> cancels.
    return -lam + k * loglam - gammaln(k + 1.0)


def logpmf_stable(k, lam, loglam):
    # inflation-free: mode-relative, no logGamma of a huge argument.
    delta = k - lam
    core = delta - k * np.log1p(delta / lam)          # = -delta^2/(2 lam) + O(.)
    return core - 0.5 * np.log(2.0 * math.pi * k) - 1.0 / (12.0 * k)


def logpmf_hi(k, lam):
    L = mp.mpf(lam)
    return float(-L + mp.mpf(k) * mp.log(L) - mp.loggamma(mp.mpf(k) + 1))


# --------------------------------------------------------------------------- #
# analyses
# --------------------------------------------------------------------------- #
def show_rhs_cancellation(lam):
    print(f"\n[1] acceptance-test RHS at lam={lam:.0e}: the -0.5/std shape signal")
    slam = math.sqrt(lam); loglam = math.log(lam)
    if not HAVE_MP:
        print("    (mpmath unavailable; skipping exact oracle)"); return
    for d in (0.0, slam, 3 * slam):
        k = float(round(lam + d))
        n = float(logpmf_naive(np.array([k]), lam, loglam)[0])
        s = float(logpmf_stable(np.array([k]), lam, loglam)[0])
        h = logpmf_hi(k, lam)
        print(f"    k=lam+{d/slam:.0f}std: naive={n:14.4f}  stable={s:12.6f}  exact={h:12.6f}"
              f"   err(naive)={abs(n-h):.2e}  err(stable)={abs(s-h):.2e}")
    # the discriminating drop that shapes the tail
    k0 = float(round(lam)); k1 = float(round(lam + slam))
    dn = float(logpmf_naive(np.array([k1]), lam, loglam)[0] - logpmf_naive(np.array([k0]), lam, loglam)[0])
    ds = float(logpmf_stable(np.array([k1]), lam, loglam)[0] - logpmf_stable(np.array([k0]), lam, loglam)[0])
    dh = logpmf_hi(k1, lam) - logpmf_hi(k0, lam)
    print(f"    R(lam+std)-R(lam): exact={dh:+.4f} (~ -0.5)   naive={dn:+.4f}   stable={ds:+.4f}")


def variance_sweep():
    print("\n[2] var/lam: numpy vs naive-reimpl vs stable-reimpl  (target 1.000)")
    print(f"    {'lam':>8} | {'numpy':>8} {'naive':>8} {'stable':>8}")
    for lam in (1e12, 1e15, 1e16, 1e17, 1e18, 9e18):
        def vr(x):
            d = np.asarray(x, float) - lam
            return d.var() / lam
        rng = np.random.default_rng(0)
        xn = np.random.default_rng(0).poisson(lam, 1_000_000)
        va = vr(ptrs(lam, 300_000, np.random.default_rng(1), logpmf_naive))
        vs = vr(ptrs(lam, 300_000, np.random.default_rng(1), logpmf_stable))
        print(f"    {lam:8.0e} | {vr(xn):8.3f} {va:8.3f} {vs:8.3f}"
              + ("   <- inflated" if vr(xn) > 1.05 else ""))


def correctness_small_lam():
    print("\n[3] stable matches truth where numpy is already correct (small lam GOF)")
    from scipy import stats
    for lam in (30.0, 300.0, 3000.0):
        x = ptrs(lam, 2_000_000, np.random.default_rng(2), logpmf_stable).astype(int)
        kmax = int(x.max())
        obs = np.bincount(x, minlength=kmax + 1).astype(float)
        k = np.arange(kmax + 1)
        pmf = stats.poisson.pmf(k, lam)
        keep = pmf * len(x) >= 25
        o = obs[keep]; e = pmf[keep] * len(x); e = e * o.sum() / e.sum()
        chi2 = (((o - e) ** 2) / e).sum(); dof = keep.sum() - 1
        pv = stats.chi2.sf(chi2, dof)
        d = x - lam
        print(f"    lam={lam:7.0f}: stable var/lam={d.var()/lam:.3f}  GOF p-value={pv:.3f} "
              f"{'OK' if pv > 0.01 else 'MISFIT'}")


if __name__ == "__main__":
    print("numpy", np.__version__, "| mpmath oracle:", HAVE_MP)
    show_rhs_cancellation(9e18)
    variance_sweep()
    correctness_small_lam()
