"""Local (spatially homogeneous) dynamics, nullclines and fixed points.

The Barkley (2011) reduced model couples a turbulence-intensity field
``q >= 0`` and a centreline-velocity proxy ``u`` in ``[0, 1]`` (``u = 1`` is
laminar).  Dropping the spatial terms (``q_x``, ``q_xx``, ``u_x``) leaves the
**local reaction dynamics**

.. math::

    \\dot q &= f(q, u) = q\\,[\\,u + r - 1 - (r + \\delta)(q - 1)^2\\,] \\\\
    \\dot u &= g(q, u) = \\varepsilon_1 (1 - u) - \\varepsilon_2 u q ,

with the Reynolds-number proxy ``r`` and fixed parameters
:math:`\\varepsilon_1 = 0.04`, :math:`\\varepsilon_2 = 0.2`,
:math:`\\delta = 0.1`.

Nullclines (curves where one time-derivative vanishes):

* **q-nullcline**, :math:`f = 0`: the trivial branch ``q = 0`` plus the
  non-trivial parabola

  .. math:: u = 1 - r + (r + \\delta)(q - 1)^2 .

  This parabola meets the laminar axis ``q = 0`` at ``u = 1 + delta`` for every
  ``r``; its lower branch (``q < 1``) is the repelling finite-amplitude
  threshold that makes the laminar state *excitable* rather than merely stable.

* **u-nullcline**, :math:`g = 0`:

  .. math:: u = \\frac{\\varepsilon_1}{\\varepsilon_1 + \\varepsilon_2 q} .

The **laminar fixed point** ``(q, u) = (0, 1)`` is a root of the local dynamics
for every ``r`` and is always linearly stable but *excitable*.  Non-trivial
(turbulent) fixed points are intersections of the two non-trivial branches;
substituting one into the other yields a cubic in ``q`` solved in closed form
by :func:`fixed_points`.

**Regime boundary.**  Setting the two non-trivial nullclines equal at the core
turbulent value ``q = 1`` gives ``1 - r = \\varepsilon_1/(\\varepsilon_1 +
\\varepsilon_2)``, i.e. the critical proxy

.. math:: r_c = \\frac{\\varepsilon_2}{\\varepsilon_1 + \\varepsilon_2}
          \\approx 0.833 .

For ``r < r_c`` the system is *excitable* (equilibrium puffs); for ``r > r_c``
it is *bistable* (expanding slugs).

References
----------
Barkley, D. (2011). Simplifying the complexity of pipe flow.
*Phys. Rev. E* **84**, 016309. arXiv:1101.4125.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EPS1",
    "EPS2",
    "DELTA",
    "local_dynamics",
    "q_nullcline",
    "u_nullcline",
    "jacobian",
    "fixed_points",
    "critical_r",
    "regime",
    "Regime",
]

#: Default model parameters (Barkley 2011).
EPS1: float = 0.04  #: u-relaxation rate towards laminar (slow; source of stiffness).
EPS2: float = 0.2  #: rate at which turbulence (q) suppresses the velocity proxy u.
DELTA: float = 0.1  #: shape parameter of the q reaction term.

#: Regime labels returned by :func:`regime`.
Regime = Literal["excitable", "critical", "bistable"]


def local_dynamics(
    q: ArrayLike,
    u: ArrayLike,
    r: float,
    eps1: float = EPS1,
    eps2: float = EPS2,
    delta: float = DELTA,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Right-hand side of the spatially homogeneous reaction dynamics.

    Computes :math:`\\dot q = f(q, u)` and :math:`\\dot u = g(q, u)` with the
    spatial terms removed. Fully vectorised over ``q`` and ``u``.

    Parameters
    ----------
    q, u : array_like
        Turbulence intensity and velocity proxy. Broadcast against each other.
    r : float
        Reynolds-number proxy.
    eps1, eps2, delta : float, optional
        Model parameters; default to the module constants.

    Returns
    -------
    dq, du : numpy.ndarray
        Time derivatives ``(f(q, u), g(q, u))``, broadcast to a common shape.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    u_arr = np.asarray(u, dtype=np.float64)
    dq = q_arr * (u_arr + r - 1.0 - (r + delta) * (q_arr - 1.0) ** 2)
    du = eps1 * (1.0 - u_arr) - eps2 * u_arr * q_arr
    return dq, du


def q_nullcline(
    q: ArrayLike, r: float, delta: float = DELTA
) -> NDArray[np.float64]:
    """Velocity proxy ``u`` on the non-trivial q-nullcline ``f(q, u) = 0``.

    Returns ``u = 1 - r + (r + delta) (q - 1)**2``. (The trivial branch
    ``q = 0`` is handled separately by callers and by :func:`fixed_points`.)

    Parameters
    ----------
    q : array_like
        Turbulence-intensity values at which to evaluate the nullcline.
    r : float
        Reynolds-number proxy.
    delta : float, optional
        Shape parameter; default :data:`DELTA`.

    Returns
    -------
    numpy.ndarray
        ``u`` values on the non-trivial q-nullcline.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    return 1.0 - r + (r + delta) * (q_arr - 1.0) ** 2


def u_nullcline(
    q: ArrayLike, eps1: float = EPS1, eps2: float = EPS2
) -> NDArray[np.float64]:
    """Velocity proxy ``u`` on the u-nullcline ``g(q, u) = 0``.

    Returns ``u = eps1 / (eps1 + eps2 * q)``, the value of ``u`` for which the
    relaxation towards laminar balances turbulent suppression at intensity ``q``.

    Parameters
    ----------
    q : array_like
        Turbulence-intensity values (``q >= 0``).
    eps1, eps2 : float, optional
        Model parameters; default to the module constants.

    Returns
    -------
    numpy.ndarray
        ``u`` values on the u-nullcline.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    return eps1 / (eps1 + eps2 * q_arr)


def jacobian(
    q: float,
    u: float,
    r: float,
    eps1: float = EPS1,
    eps2: float = EPS2,
    delta: float = DELTA,
) -> NDArray[np.float64]:
    """Jacobian of the local dynamics at a state ``(q, u)``.

    The 2x2 matrix of partial derivatives

    .. math::

        J = \\begin{pmatrix}
            \\partial f/\\partial q & \\partial f/\\partial u \\\\
            \\partial g/\\partial q & \\partial g/\\partial u
        \\end{pmatrix},

    used to classify the linear stability of fixed points (e.g. for the phase
    portrait of Fig. 2).

    Parameters
    ----------
    q, u : float
        State at which to evaluate the Jacobian.
    r : float
        Reynolds-number proxy.
    eps1, eps2, delta : float, optional
        Model parameters; default to the module constants.

    Returns
    -------
    numpy.ndarray
        The ``(2, 2)`` Jacobian ``J``.
    """
    a = r + delta
    df_dq = (u + r - 1.0 - a * (q - 1.0) ** 2) + q * (-2.0 * a * (q - 1.0))
    df_du = q
    dg_dq = -eps2 * u
    dg_du = -eps1 - eps2 * q
    return np.array([[df_dq, df_du], [dg_dq, dg_du]], dtype=np.float64)


def fixed_points(
    r: float,
    eps1: float = EPS1,
    eps2: float = EPS2,
    delta: float = DELTA,
) -> NDArray[np.float64]:
    """All fixed points of the local dynamics for a given ``r``.

    Always includes the laminar fixed point ``(0, 1)``. Non-trivial (turbulent)
    fixed points solve ``q_nullcline(q) == u_nullcline(q)``, which after
    clearing the denominator becomes the cubic

    .. math::

        (\\varepsilon_2 a)\\,q^3
        + a(\\varepsilon_1 - 2\\varepsilon_2)\\,q^2
        + [\\,(a + b)\\varepsilon_2 - 2 a \\varepsilon_1\\,]\\,q
        + \\delta\\,\\varepsilon_1 = 0 ,

    with ``a = r + delta`` and ``b = 1 - r``. Only real roots with ``q > 0`` are
    kept and paired with ``u = u_nullcline(q)``.

    Parameters
    ----------
    r : float
        Reynolds-number proxy.
    eps1, eps2, delta : float, optional
        Model parameters; default to the module constants.

    Returns
    -------
    numpy.ndarray
        Array of shape ``(k, 2)`` of fixed points ``[q, u]``, sorted by ``q``.
        Row 0 is always the laminar point ``[0, 1]``.

    Notes
    -----
    The constant term ``delta * eps1`` is strictly positive, so ``q = 0`` is
    never a root of the cubic; the laminar point is added explicitly.
    """
    a = r + delta
    b = 1.0 - r
    # Cubic coefficients (highest power first); see this function's docstring.
    c3 = eps2 * a
    c2 = a * (eps1 - 2.0 * eps2)
    c1 = -2.0 * a * eps1 + (a + b) * eps2
    c0 = delta * eps1

    roots = np.roots([c3, c2, c1, c0])
    tol = 1e-9
    turbulent = [
        (root.real, float(u_nullcline(root.real, eps1, eps2)))
        for root in roots
        if abs(root.imag) < tol and root.real > tol
    ]

    points = [(0.0, 1.0)] + turbulent
    arr = np.array(points, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    return arr[order]


def critical_r(eps1: float = EPS1, eps2: float = EPS2) -> float:
    """Critical Reynolds-number proxy separating the two regimes.

    Returns ``r_c = eps2 / (eps1 + eps2)`` (``~0.833`` for the default
    parameters), the value at which the turbulent fixed point sits exactly at
    the core intensity ``q = 1``.

    Parameters
    ----------
    eps1, eps2 : float, optional
        Model parameters; default to the module constants.

    Returns
    -------
    float
        The critical proxy ``r_c``.
    """
    return eps2 / (eps1 + eps2)


def regime(
    r: float, eps1: float = EPS1, eps2: float = EPS2, tol: float = 1e-9
) -> Regime:
    """Classify the dynamical regime for a Reynolds-number proxy ``r``.

    Parameters
    ----------
    r : float
        Reynolds-number proxy.
    eps1, eps2 : float, optional
        Model parameters; default to the module constants.
    tol : float, optional
        Half-width of the band around ``r_c`` classified as ``'critical'``.

    Returns
    -------
    {'excitable', 'critical', 'bistable'}
        ``'excitable'`` for ``r < r_c`` (equilibrium puffs), ``'bistable'`` for
        ``r > r_c`` (expanding slugs), ``'critical'`` within ``tol`` of ``r_c``.
    """
    rc = critical_r(eps1, eps2)
    if abs(r - rc) <= tol:
        return "critical"
    return "excitable" if r < rc else "bistable"
