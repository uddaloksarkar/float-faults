"""
Floating-point / range bugs in numpy's random samplers, organized BY SAMPLER.

Each distribution has ONE function; inside it, every check is tagged with the
error CLASS it demonstrates. Arguments are exact float64 dyadics (2**-k, or
integers / 9e18 = 8789062500000000*1024) wherever the bug is a cancellation or
underflow effect, so the loss provably happens inside numpy's C backend, not at
the Python->C decimal boundary. The DISTORT / NONTERM cases use representative
large parameters (lam, a) rather than dyadics, since those defects are about
magnitude, not exact representation.

ERROR CLASSES
  ROUND    catastrophic cancellation inside C (1-p -> 1, log(1-p) -> 0, ...)
  UFLOW    an intermediate underflows to exactly 0, then propagates (/0->inf, 0/0->nan)
  LATTICE  the int64 output cannot represent the support near large magnitude
  RANGE    a derived rate/count exceeds int64 and is saturated instead of guarded
  INTOVF   signed-int64 overflow in scalar validation (integer, not float)
  DISTORT  a rejection acceptance-test log-pmf loses its shape to cancellation
           (mean stays correct, variance is wrong)
  NONTERM  unbounded rejection / inf-nan loop: the sampler hangs, never returns
  LATENT   a real code-level defect that is unreachable or benign under normal use

APIs: legacy = RandomState (MT19937 + frozen old paths)
      gen    = Generator   (PCG64 + newer paths)
Recurring pattern: the frozen legacy path lacks a guard or a stable reformulation
(log1p/expm1, mode-relative log-pmf, Umin, result<1) that the Generator path has.
"""
import math
import subprocess
import sys
import time

import numpy as np

I64MAX = np.iinfo(np.int64).max
I64MIN = np.iinfo(np.int64).min
rs = lambda s=1: np.random.RandomState(s)
gn = lambda s=1: np.random.default_rng(s)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _stat(v):
    v = np.asarray(v)
    return (f"min={v.min()} max={v.max()} mean={v.astype(float).mean():.4g} "
            f"nuniq={np.unique(v).size}")


def _var_ratio(x, mean, var):
    """Stable var/mean check: mean is known and |x-mean| ~ std << mean, so the
    subtraction keeps full float64 precision even at mean ~ 1e18."""
    d = np.asarray(x, float) - mean
    return d.var() / var, d.mean() / var**0.5


def _timed(snippet, t=5.0):
    """Run snippet in a subprocess with a timeout so a hang can't stall us."""
    start = time.time()
    try:
        r = subprocess.run([sys.executable, "-c", snippet],
                           capture_output=True, text=True, timeout=t)
        return f"completed in {time.time()-start:5.2f}s -> {r.stdout.strip()}"
    except subprocess.TimeoutExpired:
        return f"HANG (no result in {t:.0f}s)"


# --------------------------------------------------------------------------- #
# binomial
# --------------------------------------------------------------------------- #
def binomial():
    print("\n### binomial ###")

    # [ROUND] n=2**62, p=2**-58 (E[X]=16). C forms q = 1.0 - p; fl(1-2**-58)=1 so
    # qn = exp(n*log(q)) = exp(0) = 1 and the inversion always returns 0.
    # legacy: all versions. gen: SAME collapse through numpy<=2.3.3, fixed to
    # exp(n*log1p(-p)) in 2.4.0 -- so the verdict is version-dependent; derive it.
    n, p = 2**62, 2.0**-58
    print(f"  [ROUND] n=2**62 p=2**-58 (E[X]=16; Generator fixed in 2.4.0):")
    for name, r in (("legacy", rs()), ("gen   ", gn())):
        x = r.binomial(n, p, 10000)
        verdict = "COLLAPSED to 0" if bool((x == 0).all()) else f"ok (mean~{x.mean():.1f})"
        print(f"            {name}: {_stat(x)}  <- {verdict}")

    # [ROUND] the collapse is the k=58 member of an infinite exact family:
    # exact p=2**-k triggers it for all k>=54 (boundary fl(1-p)=1); k=53 is fine.
    print(f"  [ROUND] exact family p=2**-k, n=16*2**k (legacy):")
    for k in (53, 54, 58):
        pp, nn = 2.0**-k, 16 * (2**k)
        x = rs().binomial(nn, pp, 20000)
        print(f"            k={k}: fl(1-p)==1?{1.0-pp==1.0}  all_zero={bool((x==0).all())}")

    # [clean] BTPE (large n*p(1-p)) evaluates its log-pmf in MODE-RELATIVE ratio
    # form, never forming log(n!), so it does NOT get the PTRS cancellation:
    # var/npq stays ~1.0 (only ~1.08 at the 2**62 int64 ceiling).
    vr, _ = _var_ratio(gn().binomial(2**58, 0.5, 1_000_000), 2**58 * 0.5, 2**58 * 0.25)
    print(f"  [clean] BTPE n=2**58 p=0.5: var/npq={vr:.3f} (mode-relative log-pmf, no PTRS bug)")


# --------------------------------------------------------------------------- #
# geometric
# --------------------------------------------------------------------------- #
def geometric():
    print("\n### geometric ###")
    # [ROUND] legacy uses ceil(log1p(-U)/log(1-p)); fl(1-2**-k)=1 for k>=54 makes
    # log(1-p)=0 -> quotient -inf -> (int64)ceil(-inf) = INT64_MIN. Support is >=1.
    # gen uses log1p(-p) in the denominator, so it is correct (large but finite).
    print("  [ROUND] p=2**-k (support >= 1); legacy collapses to INT64_MIN for k>=54:")
    for k in (53, 54, 58):
        p = 2.0**-k
        L = rs().geometric(p, 50); G = gn().geometric(p, 50)
        print(f"            k={k}: legacy all==INT64_MIN?{bool((L==I64MIN).all())} "
              f"(min={L.min()})   gen min={G.min()} (finite, ~1/p)")


# --------------------------------------------------------------------------- #
# negative_binomial
# --------------------------------------------------------------------------- #
def negative_binomial():
    print("\n### negative_binomial ###")

    # [RANGE] n=1, p=2**-62: internal Gamma*Poisson rate ~4.6e18 exceeds the
    # int64-safe Poisson range. legacy saturates to INT64_MAX (and wraps to
    # INT64_MIN); Generator has an explicit guard and raises ValueError.
    n, p = 1, 2.0**-62
    L = rs().negative_binomial(n, p, 10000)
    print(f"  [RANGE] n=1 p=2**-62: legacy {_stat(L)}  "
          f"==I64MAX:{int((L==I64MAX).sum())} ==I64MIN:{int((L==I64MIN).sum())}")
    try:
        gn().negative_binomial(n, p, 10000); print("            gen: (no error)")
    except Exception as e:
        print(f"            gen: raises {type(e).__name__}: {e}")

    # [DISTORT] negative_binomial = Poisson(Gamma(...)). With a TIGHT gamma
    # (n >> mean) it ~ Poisson(mean) and inherits the PTRS variance inflation.
    print("  [DISTORT] inherits PTRS via internal large-rate Poisson (var/theory, target 1.0):")
    for mt in (1e12, 1e15, 1e17):
        nn = 1e6 * mt; pp = nn / (nn + mt)
        mean, var = nn * (1 - pp) / pp, nn * (1 - pp) / pp**2
        vr, _ = _var_ratio(gn().negative_binomial(nn, pp, 1_000_000), mean, var)
        print(f"            mean={mt:.0e} (tight gamma): var/theory={vr:.3f}"
              + ("  <- INFLATED" if vr > 1.05 else ""))


# --------------------------------------------------------------------------- #
# poisson
# --------------------------------------------------------------------------- #
def poisson():
    print("\n### poisson ###")

    # [DISTORT] the ROOT. PTRS (lam>=10) trims a fat hat with the acceptance test
    #   -lam + k*log(lam) - lgamma(k+1),
    # a difference of two ~k*log k ~ 1e20 terms at float64 ulp ~1e5, so the O(1)
    # shape signal (-0.5/std) is lost -> hat survives -> var ~ 1.5-1.75 * lam.
    # Both APIs, all versions, no upper-lam guard; mean stays correct.
    print("  [DISTORT] PTRS var/lam (target 1.0), both APIs:")
    for lam in (1e12, 1e15, 1e17, 9e18):
        vals = []
        for r in (rs(), gn()):
            x = r.poisson(lam, 1_000_000)
            vr, _ = _var_ratio(x, float(lam), lam)
            vals.append(vr)
        print(f"            lam={lam:.0e}: legacy var/lam={vals[0]:.3f}  gen var/lam={vals[1]:.3f}"
              + ("  <- INFLATED" if max(vals) > 1.05 else ""))

    # [LATTICE] for lam>=2**53 the candidate (a double near lam) can't represent
    # consecutive integers, so outputs land on a grid of spacing ulp(lam). This
    # is COSMETIC: quantization << std, and it starts two decades above where the
    # DISTORT inflation already dominates.
    print("  [LATTICE] output grid spacing = ulp(lam) for lam>=2**53 (cosmetic):")
    for lam in (1e15, 1e17, 9e18):
        x = gn().poisson(lam, 500000)
        grid = int(np.diff(np.unique(x)).min())
        print(f"            lam={lam:.0e}: grid spacing={grid}  (ulp={int(np.spacing(np.float64(lam)))})")


# --------------------------------------------------------------------------- #
# hypergeometric
# --------------------------------------------------------------------------- #
def hypergeometric():
    print("\n### hypergeometric ###")
    # [INTOVF] integer bug, not float: legacy validation does
    #   if lngood + lnbad < lnsample: raise
    # in int64; ngood+nbad = 2**62 + 2**62 = 2**63 wraps negative, so a valid urn
    # is wrongly rejected. Generator rejects too, but via its documented <1e9
    # magnitude limit (a different, correct-for-precision reason).
    ng, nb, ns = 2**62, 2**62, 1
    print("  [INTOVF] ngood=nbad=2**62, nsample=1 (a valid urn):")
    for name, r in (("legacy", rs()), ("gen   ", gn())):
        try:
            r.hypergeometric(ng, nb, ns, 20); print(f"            {name}: accepted")
        except Exception as e:
            print(f"            {name}: raises {type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# dirichlet
# --------------------------------------------------------------------------- #
def dirichlet():
    print("\n### dirichlet ###")
    # [UFLOW] legacy draws the component gammas and divides by their sum; for
    # small alpha the gammas underflow and the sum is exactly 0.0 -> 0/0 -> NaN
    # rows. Onset is the common sparse-prior regime. Generator is immune.
    print("  [UFLOW] symmetric alpha=2**-k; legacy NaN rows, Generator immune:")
    for k in (7, 8, 10, 12, 14):
        a = 2.0**-k
        L = rs(7).dirichlet([a, a, a], size=50000)
        G = gn(7).dirichlet([a, a, a], size=50000)
        print(f"            k={k:2d} a=2**-{k}={a:.2e}: legacy NaN-rows={int(np.isnan(L).any(1).sum()):>6d}/50000"
              f"  gen NaN-rows={int(np.isnan(G).any(1).sum())}/50000")


# --------------------------------------------------------------------------- #
# standard_t
# --------------------------------------------------------------------------- #
def standard_t():
    print("\n### standard_t ###")
    # [UFLOW] standard_t = Z / sqrt(chisquare(df)/df), and chisquare(df) =
    # 2*Gamma(df/2) underflows to exactly 0 for tiny df -> division by 0 -> +/-inf.
    # Both APIs. df<1 Student-t is a legitimate heavy-tail model.
    print("  [UFLOW] df=2**-k -> +/-inf via chisquare underflow (both APIs):")
    for k in (4, 6, 8, 10):
        df = 2.0**-k
        li = int(np.isinf(rs().standard_t(df, 50000)).sum())
        gi = int(np.isinf(gn().standard_t(df, 50000)).sum())
        print(f"            k={k:2d} df=2**-{k}={df:.2e}: legacy inf={li:>6d}/50000  gen inf={gi}/50000")


# --------------------------------------------------------------------------- #
# standard_gamma / beta  (the gamma family)
# --------------------------------------------------------------------------- #
def gamma_beta():
    print("\n### standard_gamma / beta ###")
    # [UFLOW] open support (0,inf) collapses onto exact 0 for tiny shape (both
    # APIs); this 0 then detonates any downstream log()/reciprocal (it is what
    # feeds the standard_t inf above).
    print("  [UFLOW] shape=2**-k -> exact 0 outside open support (both APIs):")
    for k in (8, 10, 12, 16):
        a = 2.0**-k
        gz = int((gn().standard_gamma(a, 50000) == 0).sum())
        b = gn().beta(a, a, 50000)
        bz = int(((b == 0) | (b == 1)).sum())
        print(f"            k={k:2d} a=2**-{k}={a:.2e}: gamma exact-0={gz:>6d}/50000  beta in{{0,1}}={bz}/50000")


# --------------------------------------------------------------------------- #
# noncentral_chisquare
# --------------------------------------------------------------------------- #
def noncentral_chisquare():
    print("\n### noncentral_chisquare ###")
    # [DISTORT] for df<=1 it is chisquare(df + 2*Poisson(nonc/2)); the internal
    # large-rate Poisson inherits the PTRS variance inflation.
    print("  [DISTORT] df=0.5 large nonc -> chisq(df+2*Poisson(nonc/2)) inherits PTRS:")
    for nonc in (1e6, 1e16, 2e18):
        mean, var = 0.5 + nonc, 2 * (0.5 + 2 * nonc)
        vr, _ = _var_ratio(gn().noncentral_chisquare(0.5, nonc, 1_000_000), mean, var)
        print(f"            nonc={nonc:.0e}: var/theory={vr:.3f}" + ("  <- INFLATED" if vr > 1.05 else ""))


# --------------------------------------------------------------------------- #
# zipf
# --------------------------------------------------------------------------- #
def zipf():
    print("\n### zipf ###")
    # [NONTERM] legacy has no guards. For a->1 the fraction of proposals with
    # X<=INT64_MAX ~ (a-1)*log(2**63) -> 0, so it rejects (near-)forever. For
    # a>=1025, b=pow(2,a-1)=inf and T=(1+1/X)**am1=inf make the accept test
    # nan<=nan -> false -> infinite loop. Generator guards both (Umin, a>=1025->1).
    # Demonstrated in a subprocess with a timeout so this harness can't hang.
    print("  [NONTERM] legacy hangs (Generator guards):")
    snip = lambda api, a, n: f"import numpy as np;x={api}.zipf({a},{n});print('ok, max=%d'%x.max())"
    print("            a=1.0000001:  legacy:", _timed(snip("np.random.RandomState(0)", 1.0000001, 20000)))
    print("                          gen   :", _timed(snip("np.random.default_rng(0)", 1.0000001, 20000)))
    print("            a=2000     :  legacy:", _timed(snip("np.random.RandomState(0)", 2000.0, 1000)))
    print("                          gen   :", _timed(snip("np.random.default_rng(0)", 2000.0, 1000)))

    # [LATENT] both APIs: the reject test is `X > (double)RAND_INT_MAX`, but
    # (double)INT64_MAX rounds UP to 2**63, so a candidate X==2**63 (pow can round
    # up to it) passes and (int64)2**63 = INT64_MIN. Unreachable in practice:
    # next_double's 53-bit resolution keeps X ~6e11 below 2**63.
    print("  [LATENT] X==2**63 -> INT64_MIN off-by-one, unreachable under 53-bit U01:")
    TWO63 = 2.0**63; a = 1.1; am1 = a - 1.0; Umin = TWO63**(-am1)
    with np.errstate(invalid='ignore'):
        bad = np.float64(TWO63).astype(np.int64)
    U01g = 1.0 - np.logspace(-18, -6, 1_000_000)             # sub-ulp grid (not producible)
    U01r = gn().random(50_000_000)                           # real 53-bit next_double
    with np.errstate(over='ignore'):
        Xg = np.floor((U01g * Umin + (1 - U01g))**(-1.0 / am1))
        Xr = np.floor((U01r * Umin + (1 - U01r))**(-1.0 / am1))
    print(f"            (int64)(2**63)={bad} == INT64_MIN? {bad == I64MIN}")
    print(f"            sub-ulp U grid : #(X==2**63)={int((Xg==TWO63).sum())}  -> reachable in principle")
    print(f"            real 53-bit U01: #(X==2**63)={int((Xr==TWO63).sum())}/50e6  "
          f"max X={np.nanmax(Xr):.6e} (2**63={TWO63:.6e}) -> never")


# --------------------------------------------------------------------------- #
# logseries
# --------------------------------------------------------------------------- #
def logseries():
    print("\n### logseries ###")
    # [LATENT] legacy uses r=log(1.0-p) and q=1.0-exp(r*U) (cancellation-prone);
    # gen uses log1p(-p) and -expm1(r*U). For exact p<=2**-54, fl(1-p)=1 so legacy
    # r=0 -- the SAME ROUND collapse as geometric. But logseries ~ point mass at 1
    # there (P(X>=2) ~ p/2), so legacy returning all-1s is correct to ~15 digits;
    # unlike geometric (-> INT64_MIN), the collapse lands on the correct value 1.
    # (Otherwise logseries is robust: GOF passes, result<1/V==0 guards prevent
    #  out-of-support values, and the inversion cannot overflow int64.)
    print("  [LATENT] legacy log(1-p) collapse for p<=2**-54, benign (point mass at 1):")
    for k in (40, 53, 54, 58):
        p = 2.0**-k
        L = rs().logseries(p, 100000); G = gn().logseries(p, 100000)
        pge2 = 1.0 + p / math.log1p(-p)
        print(f"            p=2**-{k}: fl(1-p)==1?{1.0-p==1.0}  legacy all==1?{bool((L==1).all())}"
              f"  gen all==1?{bool((G==1).all())}  theory P(X>=2)={pge2:.1e}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    print("numpy", np.__version__)
    binomial()
    geometric()
    negative_binomial()
    poisson()
    hypergeometric()
    dirichlet()
    standard_t()
    gamma_beta()
    noncentral_chisquare()
    zipf()
    logseries()
