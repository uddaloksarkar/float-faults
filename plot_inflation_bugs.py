"""
Plot the DISTORT (variance-inflation) bugs from all_rounding_bugs.py.

poisson, negative_binomial, and noncentral_chisquare each run an internal
PTRS-style rejection sampler whose acceptance test cancels catastrophically at
large magnitude (see the [DISTORT] comments in all_rounding_bugs.py and the
root-cause writeup in ptrs_analysis.py). The mean stays correct; the sampled
variance drifts above the theoretical variance. This sweeps each sampler's
magnitude parameter on a log grid and plots var(x - mean) / theoretical_var,
which should sit at 1.0 everywhere.

Usage: python3 plot_inflation_bugs.py [--n SAMPLES] [--out FILE.png]
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

from all_rounding_bugs import rs, gn, _var_ratio

# poisson runs both APIs (both are affected, LATENT-free of any lam guard);
# negative_binomial / noncentral_chisquare only exercise Generator, matching
# the DISTORT sweeps in all_rounding_bugs.py.
SERIES_COLOR = {"gen": "#2a78d6", "legacy": "#1baf7a"}


def sweep_poisson(n, lo=2, hi=18.95, points=18):
    lams = np.logspace(lo, hi, points)
    legacy, gen = [], []
    for lam in lams:
        lam = float(lam)
        vr_l, _ = _var_ratio(rs().poisson(lam, n), lam, lam)
        vr_g, _ = _var_ratio(gn().poisson(lam, n), lam, lam)
        legacy.append(vr_l)
        gen.append(vr_g)
    return lams, {"legacy": legacy, "gen": gen}


def sweep_negative_binomial(n, lo=6, hi=17, points=12):
    means = np.logspace(lo, hi, points)
    gen = []
    for mt in means:
        mt = float(mt)
        nn = 1e6 * mt
        pp = nn / (nn + mt)
        mean = nn * (1 - pp) / pp
        var = nn * (1 - pp) / pp**2
        vr, _ = _var_ratio(gn().negative_binomial(nn, pp, n), mean, var)
        gen.append(vr)
    return means, {"gen": gen}


def sweep_noncentral_chisquare(n, lo=4, hi=18.3, points=12, df=0.5):
    noncs = np.logspace(lo, hi, points)
    gen = []
    for nonc in noncs:
        nonc = float(nonc)
        mean = df + nonc
        var = 2 * (df + 2 * nonc)
        vr, _ = _var_ratio(gn().noncentral_chisquare(df, nonc, n), mean, var)
        gen.append(vr)
    return noncs, {"gen": gen}


def plot(ax, x, series, title, subtitle):
    for name, y in series.items():
        ax.plot(x, y, marker="o", markersize=4, linewidth=1.8,
                color=SERIES_COLOR[name], label=name)
    ax.axhline(1.0, color="#898781", linewidth=1.2, linestyle="--", zorder=0)
    ax.text(x[-1], 1.0, "  correct", va="center", ha="left",
            color="#898781", fontsize=8)
    ax.set_xscale("log")
    ax.set_title(title, fontsize=11, fontweight="bold", loc="left")
    ax.text(0.0, 1.06, subtitle, transform=ax.transAxes, fontsize=8.5,
            color="#52514e", va="bottom")
    ax.set_ylabel("empirical var / theoretical var")
    ax.grid(True, which="major", axis="both", linewidth=0.5, color="#e1e0d9")
    ax.set_axisbelow(True)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, loc="upper left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300_000,
                     help="samples per sweep point (default: 300000)")
    ap.add_argument("--out", default="inflation_bugs.png",
                     help="output image path (default: inflation_bugs.png)")
    ap.add_argument("--show", action="store_true",
                     help="also open an interactive window")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    x, series = sweep_poisson(args.n)
    plot(axes[0], x, series, "poisson",
         "PTRS test, affected APIs: both legacy & Generator")

    x, series = sweep_negative_binomial(args.n)
    plot(axes[1], x, series, "negative_binomial",
         "Poisson(Gamma(n,·)). Inherits PTRS")

    x, series = sweep_noncentral_chisquare(args.n)
    plot(axes[2], x, series, "noncentral_chisquare (df=0.5)",
         "chisq(df+2·Poisson(nonc/2)). Inherits PTRS")

    fig.suptitle("Variance inflation in NumPy's rejection samplers",
                  fontsize=13, fontweight="bold")
    fig.savefig(args.out, dpi=150)
    print(f"saved {args.out}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
