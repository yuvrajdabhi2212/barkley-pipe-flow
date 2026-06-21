"""Generate all Phase-1 validation figures for the Barkley continuous model.

Reproduces, into ``figures/``:

* ``fig1d_equilibrium_puff.png``  -- equilibrium puff profile, r=0.7  (Fig. 1d)
* ``fig1e_intermediate.png``      -- near-critical weak slug, r=0.85  (Fig. 1e)
* ``fig1f_expanding_slug.png``    -- expanding slug profile, r=1.0    (Fig. 1f)
* ``fig2_phase_plane.png``        -- nullclines + puff/slug trajectories (Fig. 2)
* ``fig3_spacetime.png``          -- x-t diagrams of the three regimes
* ``fig4_rc_transition.png``      -- front-expansion rate vs r (the r_c transition)
* ``rc_sweep.csv``                -- the sweep data behind fig4

Run from the repo root:  ``python scripts/make_figures.py``
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from barkley_pipe.continuous import ContinuousParams, PeriodicGrid, puff_seed, simulate
from barkley_pipe.diagnostics import front_kinematics, turbulent_mass
from barkley_pipe.nullclines import critical_r, fixed_points
from barkley_pipe.plotting import (
    plot_phase_plane,
    plot_profile,
    plot_spacetime,
    recenter_snapshot,
)

FIGDIR = "figures"


def _run(r, length, n, t_end, snaps=151):
    grid = PeriodicGrid(n=n, length=length)
    y0 = puff_seed(grid, center=length / 2, width=5.0, q_amplitude=1.0, u_dip=0.6)
    return simulate(grid, ContinuousParams(r=r), y0, (0.0, t_end), n_snapshots=snaps)


def main() -> None:
    # ------------------------------------------------------------------ runs
    print("running regime simulations ...")
    puff = _run(0.70, 400.0, 1600, 300.0)
    inter = _run(0.85, 600.0, 2000, 300.0)
    slug = _run(1.00, 800.0, 2400, 250.0, snaps=126)

    # --------------------------------------------------- Fig 1 d/e/f profiles
    for res, tag, fname, title in [
        (puff, "1d", "fig1d_equilibrium_puff", r"Fig. 1(d): equilibrium puff, $r=0.7$"),
        (inter, "1e", "fig1e_intermediate", r"Fig. 1(e): near-critical weak slug, $r=0.85$"),
        (slug, "1f", "fig1f_expanding_slug", r"Fig. 1(f): expanding slug, $r=1.0$"),
    ]:
        ax = plot_profile(res, index=-1, recenter=True)
        c = 0.5 * res.grid.length
        span = 60 if tag == "1d" else (160 if tag == "1e" else 220)
        ax.set_xlim(c - span, c + span)
        ax.set_title("Reproduction of Barkley " + title)
        ax.figure.tight_layout()
        ax.figure.savefig(f"{FIGDIR}/{fname}.png", dpi=130)
        plt.close(ax.figure)

    # ----------------------------------------------------- Fig 2 phase plane
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, res, r, lab in [(axes[0], puff, 0.7, "puff"), (axes[1], slug, 1.0, "slug")]:
        qf, uf = recenter_snapshot(res.q[-1], res.u[-1], res.grid.length)
        plot_phase_plane(r, trajectory=(qf, uf), q_max=2.0, ax=ax, label=lab)
        fp = fixed_points(r)
        ax.scatter(fp[:, 0], fp[:, 1], c="k", s=55, zorder=5,
                   label="fixed points")
        ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle("Reproduction of Barkley Fig. 2: nullclines + solution trajectories")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig2_phase_plane.png", dpi=130)
    plt.close(fig)

    # -------------------------------------------------- Fig 3 space-time x-t
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    for ax, res, r, name in [(axes[0], puff, 0.7, "puff"),
                             (axes[1], inter, 0.85, "weak slug"),
                             (axes[2], slug, 1.0, "slug")]:
        drift = front_kinematics(res.q, res.x, res.t, res.grid.length)["drift"]
        plot_spacetime(res, comoving_speed=drift, ax=ax)
        ax.set_title(rf"{name}, $r={r:g}$ (co-moving)")
    fig.suptitle("Space-time diagrams of $q(x,t)$ in co-moving frames")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig3_spacetime.png", dpi=130)
    plt.close(fig)

    # ---------------------------------------------- Fig 4 r_c transition sweep
    print("running r_c sweep ...")
    r_values = [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.833,
                0.85, 0.88, 0.92, 0.96, 1.00, 1.05, 1.10]
    rows = []
    for r in r_values:
        res = _run(r, 500.0, 1250, 220.0, snaps=111)
        kin = front_kinematics(res.q, res.x, res.t, res.grid.length)
        mass = [turbulent_mass(res.q[k], res.grid.dx) for k in range(len(res.t))]
        rows.append({"r": r, "expansion_rate": kin["expansion_rate"],
                     "drift": kin["drift"], "c_leading": kin["c_leading"],
                     "c_trailing": kin["c_trailing"], "decayed": kin["decayed"],
                     "mass_growth": mass[-1] / mass[len(mass) // 3]})
        print(f"  r={r:5.3f}  expansion={kin['expansion_rate']:+.4f}  "
              f"drift={kin['drift']:+.3f}  decayed={kin['decayed']}")

    with open(f"{FIGDIR}/rc_sweep.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    rc = critical_r()
    rr = np.array([row["r"] for row in rows])
    exp = np.array([row["expansion_rate"] for row in rows])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axhline(0, color="0.7", lw=1)
    ax.axvline(rc, color="C2", ls="--", lw=1.5, label=rf"$r_c={rc:.3f}$")
    ax.plot(rr, exp, "o-", color="#c1272d", lw=2, label="expansion rate")
    ax.fill_between(rr, 0, exp, where=exp > 1e-3, alpha=0.15, color="#c1272d")
    ax.set_xlabel(r"$r$ (Reynolds-number proxy)")
    ax.set_ylabel(r"front expansion rate  $(c_\mathrm{lead}-c_\mathrm{trail})/2$")
    ax.set_title("The $r_c$ transition: equilibrium puffs (left) -> expanding slugs (right)")
    ax.annotate("equilibrium puffs\n(expansion $\\approx 0$)", (0.66, 0.02),
                fontsize=9, color="0.3")
    ax.annotate("expanding slugs", (0.95, 0.45), fontsize=9, color="0.3")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig4_rc_transition.png", dpi=130)
    plt.close(fig)

    print("\nslug plateau (q,u):",
          tuple(np.round(fixed_points(1.0)[-1], 3)),
          " == analytic turbulent fixed point")
    print("saved all figures to", FIGDIR)


if __name__ == "__main__":
    main()
