"""Barkley (2011) reduced-order model of transitional pipe flow.

A small, tested, CPU-only reproduction of the reduced model in

    Barkley, D. (2011). Simplifying the complexity of pipe flow.
    Phys. Rev. E 84, 016309. arXiv:1101.4125.

Phase 1 (implemented) solves the two coupled continuous PDEs by the method of
lines and reproduces the excitable/critical/bistable regimes, the nullcline
phase plane and space-time diagrams. Phase 2 (roadmap/stubs) covers the
discrete coupled-map-lattice model and its survival statistics.

Subpackages/modules
-------------------
operators
    Periodic finite-difference stencils (upwind, central, Laplacian).
nullclines
    Local dynamics, analytic nullclines, fixed points, regime classification.
"""

from __future__ import annotations

from . import (
    continuous,
    diagnostics,
    discrete,
    nullclines,
    operators,
    plotting,
    reference_curves,
    reference_data,
    statistics,
)
from .continuous import (
    ContinuousParams,
    ContinuousResult,
    PeriodicGrid,
    laminar_state,
    puff_seed,
    simulate,
)
from .nullclines import (
    DELTA,
    EPS1,
    EPS2,
    critical_r,
    fixed_points,
    local_dynamics,
    q_nullcline,
    regime,
    u_nullcline,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "operators",
    "nullclines",
    "continuous",
    "diagnostics",
    "plotting",
    "discrete",
    "statistics",
    "reference_data",
    "reference_curves",
    # re-exported convenience API
    "EPS1",
    "EPS2",
    "DELTA",
    "local_dynamics",
    "q_nullcline",
    "u_nullcline",
    "fixed_points",
    "critical_r",
    "regime",
    "PeriodicGrid",
    "ContinuousParams",
    "ContinuousResult",
    "laminar_state",
    "puff_seed",
    "simulate",
]
