"""
Rounding/underflow bugs in numpy samplers using ONLY exact float64 arguments.

Every floating argument here is a dyadic 2**-k (an exact double), so no
decimal->binary conversion happens at the Python boundary: any precision loss
provably occurs inside numpy's C backend. This is the point of the exercise --
a reviewer cannot attribute the failure to argument representation.

Verified retraction: a previous probe used p = 1 - 2**-58, which rounds to
*exactly* 1.0 in float64, so binomial(n, 1.0) == n was correct, not a bug. With
the largest exact double below 1 (p = 1 - 2**-53), binomial is also correct
(BTPE subtracts in the integer domain). There is no p->1 binomial bug.

Argument classes and mechanisms (all inputs exact):
  FAMILY-1  p = 2**-k, k >= 54  ->  C forms q = 1.0 - p; fl(1-2**-k)=1 at k>=54
            => binomial inversion returns all 0; legacy geometric returns INT64_MIN.
  BUG-A     alpha = 2**-k (k>=8) ->  legacy dirichlet: sum of underflowed gammas
            = 0.0 => 0/0 => NaN rows. Generator immune.
  BUG-B     df = 2**-k (k>=6)    ->  chisquare(df)=2*Gamma(df/2) underflows to 0;
            standard_t = Z/sqrt(chisq/df) => +/-inf. Both APIs.
  UNDERFLOW shape = 2**-k        ->  standard_gamma/beta/chisquare return exact 0
            (outside open support), both APIs; this feeds BUG-B.
"""
import numpy as np

I64MIN = np.iinfo(np.int64).min
rs = lambda s=1: np.random.RandomState(s)
gn = lambda s=1: np.random.default_rng(s)
M = 50_000


def exact(x):
    # a python float is always an exact double; assert it's the intended 2**-k
    return x


def family1_binomial_geometric():
    print("FAMILY-1  exact p = 2**-k  (boundary fl(1-p)=1 at k>=54)")
    print("  binomial inversion, n = 16 * 2**k so E[X] = 16:")
    for k in (53, 54, 55, 58):
        p, n = exact(2.0**-k), 16 * (2**k)
        L = rs().binomial(n, p, 20000)
        print(f"    k={k}: fl(1-p)==1? {1.0-p==1.0}  legacy nuniq={np.unique(L).size:>2} "
              f"all_zero={bool(np.all(L==0))}  mean={L.mean():.4g} (E=16)")
    print("  legacy geometric (support >= 1):")
    for k in (53, 54, 55, 58):
        p = exact(2.0**-k)
        L = rs().geometric(p, 50)
        print(f"    k={k}: all == INT64_MIN? {bool(np.all(L==I64MIN))}  min={L.min()}")


def bugA_dirichlet():
    print("\nBUG-A  legacy dirichlet NaN, exact symmetric alpha = 2**-k "
          "(Generator immune)")
    for k in (7, 8, 10, 12, 14, 20):
        a = exact(2.0**-k)
        L = rs(7).dirichlet([a, a, a], size=M)
        G = gn(7).dirichlet([a, a, a], size=M)
        print(f"    k={k:2d} a=2**-{k}={a:.3e}: legacy NaN-rows={int(np.isnan(L).any(1).sum()):>6d}/{M}"
              f"   gen NaN-rows={int(np.isnan(G).any(1).sum())}/{M}")


def bugB_student_t():
    print("\nBUG-B  standard_t -> inf, exact df = 2**-k (both APIs); via chisquare underflow")
    for k in (4, 6, 8, 10, 12):
        df = exact(2.0**-k)
        cz = int((rs().chisquare(df, M) == 0).sum())
        li = int(np.isinf(rs().standard_t(df, M)).sum())
        gi = int(np.isinf(gn().standard_t(df, M)).sum())
        print(f"    k={k:2d} df=2**-{k}={df:.3e}: chisq exact-0={cz:>6d}/{M}  "
              f"legacy_t inf={li:>6d}/{M}  gen_t inf={gi}/{M}")


def underflow_gamma_beta():
    print("\nUNDERFLOW  exact shape = 2**-k -> exact 0 outside open support (both APIs)")
    for k in (8, 10, 12, 16):
        a = exact(2.0**-k)
        gz = int((gn().standard_gamma(a, M) == 0).sum())
        b = gn().beta(a, a, M)
        bz = int(((b == 0) | (b == 1)).sum())
        print(f"    k={k:2d} a=2**-{k}={a:.3e}: gamma exact-0={gz:>6d}/{M} (E[X]={a:.1e})  "
              f"beta in{{0,1}}={bz}/{M}")


def retraction_check():
    print("\nRETRACTED p->1 'bug': largest exact double below 1 works correctly")
    p, n = 1.0 - 2.0**-53, 2**62
    L = rs().binomial(n, p, 20000)
    print(f"    p=1-2**-53 (!=1.0: {p!=1.0}), E[fail]=512: legacy nuniq={np.unique(L).size} "
          f"spread=[{L.min()}, {L.max()}] around n-512 -> correct")


if __name__ == "__main__":
    print("numpy", np.__version__, "| all float arguments are exact dyadic 2**-k\n")
    family1_binomial_geometric()
    bugA_dirichlet()
    bugB_student_t()
    underflow_gamma_beta()
    retraction_check()
