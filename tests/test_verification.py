"""Code verification by the Method of Manufactured Solutions (MMS).

Verification answers *"are we solving the equations right?"* — and its reference
is mathematics we construct ourselves, so (unlike validation against external
data) it cannot be fabricated or hallucinated.

MMS recipe: choose a smooth, periodic exact solution ``(q*, u*)``, substitute it
into the continuous model to obtain an analytic forcing ``S(x, t)``, then solve
the *forced* equations numerically from the manufactured initial condition. The
numerical solution must converge to ``(q*, u*)`` at the formal order of the
spatial discretization — **first order** for the upwind advection scheme,
**second order** for the central scheme. A wrong observed order exposes a bug in
the finite-difference operators or in the RHS assembly.

The temporal error is driven far below the spatial error with tight ``solve_ivp``
tolerances, so the measured convergence rate reflects the spatial scheme.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from barkley_pipe.continuous import ContinuousParams, PeriodicGrid, continuous_rhs
from barkley_pipe.operators import get_advection_operator

# --- Manufactured solution: smooth and periodic on [0, L) --------------------
_L = 2.0 * np.pi
_K = 2.0 * np.pi / _L  # one full wave across the domain
_OMEGA = 0.7
_AQ, _BQ = 1.2, 0.4  # q* in [0.8, 1.6]  (> 0)
_AU, _BU = 0.6, 0.25  # u* in [0.35, 0.85]  (in [0, 1])


def _theta(x, t):
    return _K * x - _OMEGA * t


def _q_star(x, t):
    return _AQ + _BQ * np.sin(_theta(x, t))


def _u_star(x, t):
    return _AU + _BU * np.cos(_theta(x, t))


def _q_t(x, t):
    return -_BQ * _OMEGA * np.cos(_theta(x, t))


def _q_x(x, t):
    return _BQ * _K * np.cos(_theta(x, t))


def _q_xx(x, t):
    return -_BQ * _K**2 * np.sin(_theta(x, t))


def _u_t(x, t):
    return _BU * _OMEGA * np.sin(_theta(x, t))


def _u_x(x, t):
    return -_BU * _K * np.sin(_theta(x, t))


def _reaction_q(q, u, p):
    return q * (u + p.r - 1.0 - (p.r + p.delta) * (q - 1.0) ** 2)


def _reaction_u(q, u, p):
    return p.eps1 * (1.0 - u) - p.eps2 * u * q


def _forcing(x, t, p):
    """Analytic source making (q*, u*) the exact solution of the forced model."""
    s_q = _q_t(x, t) - (
        -p.U * _q_x(x, t) + _q_xx(x, t)
        + _reaction_q(_q_star(x, t), _u_star(x, t), p)
    )
    s_u = _u_t(x, t) - (
        -(p.U + 1.0) * _u_x(x, t)
        + _reaction_u(_q_star(x, t), _u_star(x, t), p)
    )
    return s_q, s_u


def _run_mms(n: int, scheme: str, t_end: float = 0.3) -> float:
    """Solve the forced model on an n-point grid; return max error vs (q*, u*)."""
    grid = PeriodicGrid(n=n, length=_L)
    params = ContinuousParams(r=0.7, U=1.0, advection=scheme)
    advect_op = get_advection_operator(scheme)
    x = grid.x

    def rhs(t, y):
        base = continuous_rhs(t, y, grid, params, advect_op)
        s_q, s_u = _forcing(x, t, params)
        return base + np.concatenate([s_q, s_u])

    y0 = np.concatenate([_q_star(x, 0.0), _u_star(x, 0.0)])
    sol = solve_ivp(rhs, (0.0, t_end), y0, method="RK45",
                    t_eval=[t_end], rtol=1e-10, atol=1e-12)
    q_num, u_num = sol.y[:n, -1], sol.y[n:, -1]
    return max(np.max(np.abs(q_num - _q_star(x, t_end))),
               np.max(np.abs(u_num - _u_star(x, t_end))))


def _measured_order(scheme: str, ns=(24, 48, 96)):
    errors = [_run_mms(n, scheme) for n in ns]
    spacings = [_L / n for n in ns]
    order = float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])
    return order, errors


def test_mms_upwind_is_first_order() -> None:
    """Upwind advection -> the coupled solver is first-order accurate in space."""
    order, errors = _measured_order("upwind")
    assert all(e < 1.0 for e in errors)  # sane magnitude, monotone-ish
    assert 0.85 <= order <= 1.25, f"expected ~1, got {order:.3f} (errors {errors})"


def test_mms_central_is_second_order() -> None:
    """Central advection -> the coupled solver is second-order accurate in space."""
    order, errors = _measured_order("central")
    assert 1.80 <= order <= 2.20, f"expected ~2, got {order:.3f} (errors {errors})"
