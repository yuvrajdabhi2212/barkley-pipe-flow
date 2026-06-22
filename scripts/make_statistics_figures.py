"""Generate the Phase-2 survival-statistics figures (reduced ensembles).

Produces, into ``figures/``:

* ``fig6_survival.png``         -- decay survival curves P(n) (memoryless, Fig. 12)
* ``fig7_tau_crossing.png``     -- tau_decay(R) and tau_split(R) crossing (Fig. 5a)
* ``fig8_turbulence_fraction.png`` -- order parameter F_t(R) and its onset
* ``statistics_sweep.csv``      -- the underlying tau(R) / F_t(R) data

Ensembles are intentionally small (Colab-friendly), so R_x ~ 2040 and
R_c ~ 2046 are reproduction targets within tolerance, not high-precision values.

Run from the repo root:  ``python scripts/make_statistics_figures.py``
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from barkley_pipe.discrete import DiscreteParams, initial_puff, simulate_discrete
from barkley_pipe.statistics import (
    fit_tau,
    sample_lifetimes,
    survival_function,
    tau_of_R,
    turbulence_fraction,
)

FIGDIR = "figures"
R_C = 2046.0  # nominal sustained-turbulence onset (Barkley)


def figure_survival() -> None:
    """Decay survival curves: P(n) ~ exp(-n/tau) is straight on a log axis."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for R, color in [(1800, "#1b9e77"), (1850, "#d95f02"), (1900, "#7570b3")]:
        life = sample_lifetimes(R, "decay", n_realizations=400, n_sites=200,
                                max_steps=9000, width=40, seed=1)
        n, surv = survival_function(life)
        tau = fit_tau(life)
        ax.semilogy(n, surv, ".", color=color, ms=4,
                    label=fr"$R={R}$  ($\tau\approx{tau:.0f}$, N={life.size})")
        nn = np.linspace(0, n.max(), 100)
        ax.semilogy(nn, np.exp(-nn / tau), "-", color=color, lw=1)
    ax.set_xlabel("time $n$ (steps)")
    ax.set_ylabel("survival probability $P(n)$")
    ax.set_ylim(1e-3, 1.2)
    ax.set_title("Puff-decay survival is memoryless: $P(n)\\sim e^{-n/\\tau(R)}$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig6_survival.png", dpi=130)
    plt.close(fig)
    print("saved fig6_survival.png")


def figure_tau_crossing() -> dict:
    """tau_decay(R) and tau_split(R); their crossing locates R_x ~ 2040."""
    Rd = np.array([1900, 1950, 2000, 2020, 2040, 2060], dtype=float)
    Rs = np.array([2020, 2040, 2060, 2080, 2120, 2160, 2200], dtype=float)
    _, td = tau_of_R(Rd, n_realizations=140, event="decay",
                     n_sites=200, max_steps=11000, width=40)
    _, ts = tau_of_R(Rs, n_realizations=140, event="splitting",
                     n_sites=400, max_steps=9000, width=30, check_every=5)

    # crossing: interpolate log(tau) on the overlapping range and find sign change
    grid = np.linspace(2010, 2070, 601)
    ld = np.interp(grid, Rd, np.log(td))
    ls = np.interp(grid, Rs, np.log(ts))
    diff = ld - ls
    sign = np.where(np.diff(np.sign(diff)))[0]
    r_x = float(grid[sign[0]]) if sign.size else float("nan")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogy(Rd, td, "o-", color="#0072b2", label=r"$\tau_\mathrm{decay}$")
    ax.semilogy(Rs, ts, "s-", color="#c1272d", label=r"$\tau_\mathrm{split}$")
    if np.isfinite(r_x):
        ax.axvline(r_x, color="0.4", ls="--",
                   label=fr"crossing $R_\times\approx{r_x:.0f}$")
    ax.set_xlabel("$R$ (discrete Reynolds proxy)")
    ax.set_ylabel(r"characteristic time $\tau$ (steps)")
    ax.set_title(r"Decay vs splitting lifetimes cross near $R_\times\approx2040$")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig7_tau_crossing.png", dpi=130)
    plt.close(fig)
    print(f"saved fig7_tau_crossing.png  (R_x ~ {r_x:.0f})")
    return {"Rd": Rd, "td": td, "Rs": Rs, "ts": ts, "r_x": r_x}


def figure_turbulence_fraction() -> dict:
    """Order parameter F_t(R): onset near R_c with a continuous transition."""
    Rs = np.array([2000, 2040, 2080, 2120, 2160, 2200, 2300, 2500], dtype=float)
    ft = np.empty(Rs.size)
    for i, R in enumerate(Rs):
        q0, u0 = initial_puff(1000, width=500, q_level=1.0, seed=0)
        qst = simulate_discrete(q0, u0, DiscreteParams(R=R), n_steps=5000, store_every=20)
        ft[i] = turbulence_fraction(qst, tail=0.4)

    above = Rs > R_C
    beta = np.nan
    if above.sum() >= 2 and np.all(ft[above] > 0):
        beta = float(np.polyfit(np.log(Rs[above] - R_C), np.log(ft[above]), 1)[0])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(Rs, ft, "o-", color="#c1272d")
    ax.axvline(R_C, color="C2", ls="--", label=fr"$R_c\approx{R_C:.0f}$")
    ax.set_xlabel("$R$ (discrete Reynolds proxy)")
    ax.set_ylabel(r"turbulent fraction $F_t$")
    ax.set_title(rf"Onset of sustained turbulence (effective exponent $\approx{beta:.2f}$)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig8_turbulence_fraction.png", dpi=130)
    plt.close(fig)
    print(f"saved fig8_turbulence_fraction.png  (effective exponent ~ {beta:.2f})")
    return {"Rs": Rs, "ft": ft, "beta": beta}


def main() -> None:
    figure_survival()
    tau = figure_tau_crossing()
    ft = figure_turbulence_fraction()

    with open(f"{FIGDIR}/statistics_sweep.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["kind", "R", "value"])
        for R, t in zip(tau["Rd"], tau["td"]):
            w.writerow(["tau_decay", R, t])
        for R, t in zip(tau["Rs"], tau["ts"]):
            w.writerow(["tau_split", R, t])
        for R, f in zip(ft["Rs"], ft["ft"]):
            w.writerow(["F_t", R, f])
    print("saved statistics_sweep.csv")
    print(f"SUMMARY: R_x ~ {tau['r_x']:.0f} (target ~2040), "
          f"F_t effective exponent ~ {ft['beta']:.2f} (DP ~ 0.28)")


if __name__ == "__main__":
    main()
