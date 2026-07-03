"""
Rounding / underflow bugs in random samplers beyond the five in the seed script.

Each probe prints the observed anomaly against the analytic expectation. The
common mechanisms are:
  (M1) open-support (0, inf) continuous draws underflowing to *exactly* 0 for
       tiny shape/rate, and that 0 then propagating through a division;
  (M2) 0/0 when a normalizing sum of underflowed gammas is exactly 0 (NaN);
  (M3) k = n - failures collapsing back to n when failures << ulp(n);
  (M4) inversion exponents overflowing to +inf;
  (M5) stdlib raising OverflowError where numpy silently returns inf.

APIs: RandomState = legacy (MT19937 + old code paths); default_rng = Generator
(PCG64 + newer paths); random = CPython stdlib (MT19937).
"""
import math
import random
import numpy as np

N = 200_000
rs = lambda s=1: np.random.RandomState(s)
gn = lambda s=1: np.random.default_rng(s)


def stat(v):
    v = np.asarray(v, dtype=float)
    fin = v[np.isfinite(v)]
    return dict(
        min=float(v.min()), max=float(v.max()),
        mean=float(fin.mean()) if fin.size else float("nan"),
        nan=int(np.isnan(v).sum()), inf=int(np.isinf(v).sum()),
        zero=int((v == 0).sum()), nuniq=int(np.unique(v).size),
    )


def report(tag, v, expect):
    s = stat(v)
    print(f"  {tag:34s} nan={s['nan']:>7d} inf={s['inf']:>7d} zero={s['zero']:>7d} "
          f"nuniq={s['nuniq']:>7d} mean={s['mean']:.4g}   [expect: {expect}]")


# ---------------------------------------------------------------------------
# BUG A. Legacy RandomState.dirichlet -> all-NaN for small alpha (M2).
#   Generator.dirichlet is immune. Onset is in the *common* sparse-prior regime.
# ---------------------------------------------------------------------------
def bug_dirichlet_nan():
    print("\n[A] dirichlet, symmetric alpha (rows must be nonneg and sum to 1)")
    for a in [1e-2, 1e-3, 1e-4, 1e-6]:
        L = rs().dirichlet([a, a, a], size=N)
        G = gn().dirichlet([a, a, a], size=N)
        lnan = int(np.isnan(L).any(1).sum())
        gnan = int(np.isnan(G).any(1).sum())
        print(f"  alpha={a:<7g} legacy NaN-rows={lnan:>7d}/{N}   "
              f"gen NaN-rows={gnan:>7d}/{N}")


# ---------------------------------------------------------------------------
# BUG B. standard_t with small df -> +/-inf (both APIs) (M1).
#   t = Z / sqrt(chisq(df)/df); chisq(df)=2*gamma(df/2) underflows to 0.
# ---------------------------------------------------------------------------
def bug_student_t_inf():
    print("\n[B] standard_t df=1e-3 (a.s. finite; heavy but proper)")
    report("legacy chisquare(1e-3)", rs().chisquare(1e-3, N), "no exact zeros")
    report("legacy standard_t(1e-3)", rs().standard_t(1e-3, N), "all finite")
    report("gen    standard_t(1e-3)", gn().standard_t(1e-3, N), "all finite")


# ---------------------------------------------------------------------------
# BUG C. Open-support continuous draws collapse to exact 0 (both APIs) (M1).
#   gamma/beta/chisquare with tiny shape return 0.0, outside (0, inf);
#   any downstream log()/reciprocal then diverges.
# ---------------------------------------------------------------------------
def bug_open_support_zero():
    print("\n[C] standard_gamma / beta tiny shape (support is open; 0 is invalid)")
    report("gen gamma(1e-10)   mean~=1e-10", gn().standard_gamma(1e-10, N), "few/no exact 0")
    report("gen beta(1e-10,1e-10) mean=0.5", gn().beta(1e-10, 1e-10, N), "no mass pile at 0/1 only")


# ---------------------------------------------------------------------------
# BUG D. binomial with p -> 1 collapses to the constant n, in BOTH APIs (M3).
#   Mirror of the seed script's p->0 legacy-only collapse. Here n*(1-p)=16
#   failures are lost: n - failures rounds back to n at ulp(2**62)=1024.
# ---------------------------------------------------------------------------
def bug_binomial_p_near_one():
    n, p = 2**62, 1 - 2**-58
    print(f"\n[D] binomial n=2**62 p=1-2**-58 (E[X]=n-16, Var=16)")
    report("legacy binomial", rs().binomial(n, p, N), f"~{n-16} with spread")
    report("gen    binomial", gn().binomial(n, p, N), f"~{n-16} with spread")


# ---------------------------------------------------------------------------
# BUG E. CPython stdlib random (a *different* generator) analogues.
#   E1 gammavariate(tiny)  -> exact 0 (M1)
#   E2 betavariate(tiny)   -> silent collapse to 0, mean 0 not 0.5 (guarded 0/0)
#   E3 paretovariate(tiny) -> uncaught OverflowError where numpy returns inf (M5)
# ---------------------------------------------------------------------------
def bug_stdlib():
    print("\n[E] CPython stdlib random")
    random.seed(1)
    g = [random.gammavariate(1e-10, 1.0) for _ in range(50_000)]
    print(f"  E1 gammavariate(1e-10,1): frac_zero={sum(x==0 for x in g)/len(g):.3f} "
          f"(support (0,inf); mean should be 1e-10)")
    b = [random.betavariate(1e-8, 1e-8) for _ in range(50_000)]
    print(f"  E2 betavariate(1e-8,1e-8): mean={sum(b)/len(b):.4f} "
          f"(true 0.5), frac_zero={sum(x==0 for x in b)/len(b):.3f}")
    try:
        random.paretovariate(1e-10)
        print("  E3 paretovariate(1e-10): returned (no error)")
    except OverflowError as e:
        print(f"  E3 paretovariate(1e-10): OverflowError: {e}  "
              f"(numpy silently returns +inf here)")


if __name__ == "__main__":
    print("numpy", np.__version__)
    bug_dirichlet_nan()
    bug_student_t_inf()
    bug_open_support_zero()
    bug_binomial_p_near_one()
    bug_stdlib()
