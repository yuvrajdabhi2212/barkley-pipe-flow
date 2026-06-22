"""Original value-add: a quantitative continuous-vs-discrete comparison.

Barkley's two reduced models describe the *same* laminar->turbulent transition in
different parameterizations. This figure puts them side by side:

* **Continuous model** -- front *expansion rate* ``(c_lead - c_trail)/2`` vs the
  proxy ``r`` (loaded from ``figures/rc_sweep.csv``); it switches on at the
  excitable->bistable point ``r_c = eps2/(eps1+eps2) = 0.833``.
* **Discrete model** -- mean turbulent-region *spreading rate* (sites/step) vs
  the proxy ``R``; it switches on at the sustained-turbulence onset
  ``R_c ~ 2046``.

Both observables measure "does turbulence spread?", and both cross zero at their
respective critical point -- the structural correspondence between the two
models. Writes ``figures/fig9_continuous_vs_discrete.png``.

Run from the repo root:  ``python scripts/make_comparison_figure.py``
"""

from __future__ import annotations

import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from barkley_pipe.discrete import DiscreteParams, step
from barkley_pipe.nullclines import critical_r

FIGDIR = "figures"
R_C = 2046.0


def _load_continuous():
    r, exp = [], []
    with open(f"{FIGDIR}/rc_sweep.csv") as fh:
        for row in csv.DictReader(fh):
            r.append(float(row["r"]))
            exp.append(float(row["expansion_rate"]))
    return np.array(r), np.array(exp)


def discrete_spreading_rate(R, n=600, m=100, steps=1500, width=40, seed=0):
    """Mean growth rate (sites/step) of the turbulent-region extent at proxy R."""
    rng = np.random.default_rng(seed)
    q = np.zeros((m, n))
    u = np.ones((m, n))
    s = (n - width) // 2
    q[:, s : s + width] = 1.0 + 0.1 * rng.standard_normal((m, width))
    p = DiscreteParams(R=R)
    cols = np.arange(n)[None, :]
    extents = np.empty(steps)
    for t in range(steps):
        q, u = step(q, u, p)
        mask = q > 0.5
        any_turb = mask.any(axis=1)
        right = np.where(mask, cols, -1).max(axis=1)
        left = np.where(mask, cols, n).min(axis=1)
        extents[t] = np.where(any_turb, right - left + 1, 0).mean()
    t = np.arange(steps)
    win = (t > 50) & (extents < 0.7 * n)
    if win.sum() < 5:
        win = t > 50
    return float(np.polyfit(t[win], extents[win], 1)[0])


def main() -> None:
    r, cont_exp = _load_continuous()
    rc = critical_r()

    R = np.array([1950, 2000, 2046, 2100, 2160, 2250, 2400], dtype=float)
    disc_rate = np.array([discrete_spreading_rate(float(x), seed=i)
                          for i, x in enumerate(R)])
    onset = float(np.interp(0.0, disc_rate, R)) if np.any(disc_rate > 0) else float("nan")

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 5))

    a.axhline(0, color="0.8", lw=1)
    a.axvline(rc, color="C2", ls="--", label=fr"$r_c={rc:.3f}$")
    a.plot(r, cont_exp, "o-", color="#0072b2")
    a.set_xlabel(r"$r$ (continuous proxy)")
    a.set_ylabel(r"front expansion rate")
    a.set_title("Continuous model: puffs $\\to$ slugs at $r_c$")
    a.legend(frameon=False)

    b.axhline(0, color="0.8", lw=1)
    b.axvline(R_C, color="C2", ls="--", label=fr"$R_c\approx{R_C:.0f}$")
    b.plot(R, disc_rate, "s-", color="#c1272d")
    b.set_xlabel(r"$R$ (discrete proxy)")
    b.set_ylabel(r"turbulent spreading rate (sites/step)")
    b.set_title("Discrete model: decay $\\to$ spreading at $R_c$")
    b.legend(frameon=False)

    fig.suptitle("Same laminar$\\to$turbulent transition, two reduced models")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig9_continuous_vs_discrete.png", dpi=130)
    plt.close(fig)

    print("continuous onset r_c =", round(rc, 3))
    print("discrete spreading rates:", dict(zip(R.astype(int), np.round(disc_rate, 4))))
    print(f"discrete spreading onset (rate=0 crossing) ~ R={onset:.0f}  (target R_c~2046)")
    print("saved figures/fig9_continuous_vs_discrete.png")


if __name__ == "__main__":
    main()
