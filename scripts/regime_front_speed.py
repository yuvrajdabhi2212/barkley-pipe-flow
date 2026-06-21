"""Measure front speeds of a triggered structure for one value of ``r``.

Runs the continuous Barkley model from a localised puff seed, tracks the
leading/trailing turbulent fronts, and reports their speeds together with the
expansion rate ``(c_lead - c_trail)/2`` and drift ``(c_lead + c_trail)/2``.
The expansion rate is ~0 in the excitable regime (equilibrium puff, constant
width) and positive in the bistable regime (expanding slug), so sweeping ``r``
locates the ``r_c`` transition.

Usage
-----
    python scripts/regime_front_speed.py R [L N T]

Prints a single JSON line of diagnostics (consumed by the regime-sweep
workflow and by notebook ``02``).
"""

from __future__ import annotations

import json
import sys

import numpy as np

from barkley_pipe.continuous import (
    ContinuousParams,
    PeriodicGrid,
    puff_seed,
    simulate,
)
from barkley_pipe.diagnostics import front_kinematics, turbulent_mass


def measure(r: float, length: float, n: int, t_end: float) -> dict:
    """Simulate at proxy ``r`` and return front-speed diagnostics as a dict."""
    grid = PeriodicGrid(n=n, length=length)
    y0 = puff_seed(grid, center=length / 2, width=5.0, q_amplitude=1.0, u_dip=0.6)
    res = simulate(grid, ContinuousParams(r=r), y0, (0.0, t_end), n_snapshots=121)

    kin = front_kinematics(res.q, res.x, res.t, length, threshold=0.5, t_min=40.0)

    q_final = res.q[-1]
    u_final = res.u[-1]
    core = q_final > 0.5
    mass = np.array([turbulent_mass(res.q[k], grid.dx) for k in range(len(res.t))])

    return {
        "r": float(r),
        "decayed": bool(kin["decayed"]),
        "c_leading": float(kin["c_leading"]),
        "c_trailing": float(kin["c_trailing"]),
        "expansion_rate": float(kin["expansion_rate"]),
        "drift_speed": float(kin["drift"]),
        "final_width": float(kin["final_width"]),
        "peak_q": float(q_final.max()),
        "plateau_q": float(np.median(q_final[core])) if core.any() else 0.0,
        "plateau_u": float(np.median(u_final[core])) if core.any() else 1.0,
        "mass_initial": float(mass[len(mass) // 3]),
        "mass_final": float(mass[-1]),
        "success": bool(res.success),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("usage: python scripts/regime_front_speed.py R [L N T]", file=sys.stderr)
        return 2
    r = float(argv[1])
    length = float(argv[2]) if len(argv) > 2 else 600.0
    n = int(argv[3]) if len(argv) > 3 else 2400
    t_end = float(argv[4]) if len(argv) > 4 else 300.0
    print(json.dumps(measure(r, length, n, t_end)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
