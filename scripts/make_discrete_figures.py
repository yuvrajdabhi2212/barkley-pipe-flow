"""Generate the discrete-model (Phase 2) validation figure.

Reproduces Barkley Fig. 4 — space-time diagrams of the discrete coupled-map
lattice in the three regimes selected by the Reynolds proxy ``R``:

* decaying puff      ``R = 1900``  (below onset; transient chaos -> relaminarizes)
* puff splitting     ``R = 2200``  (proliferation into multiple puffs)
* expanding slug     ``R = 3000``  (turbulence fills the domain)

Writes ``figures/fig5_discrete_spacetime.png``.

Run from the repo root:  ``python scripts/make_discrete_figures.py``
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from barkley_pipe.discrete import DiscreteParams, initial_puff, simulate_discrete
from barkley_pipe.plotting import plot_discrete_spacetime

FIGDIR = "figures"


def _run(r, n, width, n_steps, store_every, seed=0):
    q0, u0 = initial_puff(n, width=width, q_level=1.0, seed=seed)
    qst = simulate_discrete(q0, u0, DiscreteParams(R=r), n_steps, store_every=store_every)
    return qst, store_every


def main() -> None:
    decay, se_d = _run(1900, n=400, width=40, n_steps=2400, store_every=10)
    split, se_s = _run(2200, n=700, width=30, n_steps=7000, store_every=28)
    slug, se_g = _run(3000, n=400, width=30, n_steps=1600, store_every=6)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    plot_discrete_spacetime(decay, store_every=se_d, ax=axes[0],
                            title=r"(a) decaying puff, $R=1900$")
    plot_discrete_spacetime(split, store_every=se_s, ax=axes[1],
                            title=r"(b) puff splitting, $R=2200$")
    plot_discrete_spacetime(slug, store_every=se_g, ax=axes[2],
                            title=r"(c) expanding slug, $R=3000$")
    fig.suptitle("Reproduction of Barkley Fig. 4: discrete coupled-map-lattice "
                 "space-time diagrams")
    fig.tight_layout()
    fig.savefig(f"{FIGDIR}/fig5_discrete_spacetime.png", dpi=140)
    print(f"saved {FIGDIR}/fig5_discrete_spacetime.png")

    for name, qst in [("decay", decay), ("split", split), ("slug", slug)]:
        print(f"  {name}: final turbulent fraction = {(qst[-1] > 0.5).mean():.3f}")


if __name__ == "__main__":
    main()
