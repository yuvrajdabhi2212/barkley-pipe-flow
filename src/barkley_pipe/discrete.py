"""Discrete coupled-map-lattice Barkley model (Phase 2, Milestone 1).

Implements the *discrete* Barkley (2011) model (his Eqs. 3-6), a coupled-map
lattice that — unlike the continuous model — supports stochastic puff **decay**
and **splitting**, the origin of the memoryless lifetime statistics and the
directed-percolation onset of sustained turbulence.

Model
-----
With spatial index ``i`` and time index ``n`` (periodic lattice):

.. math::

    q_i^{n+1} &= F\\!\\left(q_i^{n} + d\\,(q_{i-1}^{n} - 2 q_i^{n} + q_{i+1}^{n}),
                 \\; u_i^{n}\\right), \\\\
    u_i^{n+1} &= u_i^{n} + \\varepsilon_1 (1 - u_i^{n})
                 - \\varepsilon_2\\, u_i^{n} q_i^{n}
                 - c\\,(u_i^{n} - u_{i-1}^{n}),

where ``F = f^k`` is the ``k``-fold iterate of a piecewise-linear **tent map**
``f`` whose shape depends on ``u`` through the threshold

.. math:: \\alpha(u, R) = 2000\\,(1 - 0.8\\,u)\\,R^{-1}.

With breakpoints :math:`Q_1 = \\alpha/(2-\\gamma)` and
:math:`Q_2 = (4 + \\beta - \\alpha - \\gamma Q_1)/(2 + \\beta)`,

.. math::

    f(q) = \\begin{cases}
        \\gamma q                          & q < Q_1 \\\\
        2q - \\alpha                        & Q_1 \\le q < 1 \\\\
        4 + \\beta - \\alpha - (2+\\beta)q   & 1 \\le q < Q_2 \\\\
        \\gamma Q_1                         & Q_2 \\le q .
    \\end{cases}

``f`` is continuous (the breakpoints are *defined* to make it so) with a
contracting laminar branch (slope ``gamma < 1`` toward the fixed point
``q = 0``) and an expanding/folding chaotic region; ``beta > 0`` opens the
escape window that lets a turbulent site spontaneously relaminarize, giving
transient chaos. Parameters: ``eps1=0.04``, ``eps2=0.2``, ``k=2``, ``beta=0.4``,
``gamma=0.95``, ``c=0.45``, ``d=0.15`` (``d <= 0.5`` for stability). ``R`` is the
discrete Reynolds-number proxy (``R ~ 2000`` ~ ``Re``; distinct from the
continuous model's ``r = O(1)``).

References
----------
Barkley, D. (2011). Simplifying the complexity of pipe flow.
*Phys. Rev. E* **84**, 016309. arXiv:1101.4125.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "EPS1",
    "EPS2",
    "K",
    "BETA",
    "GAMMA",
    "C",
    "D",
    "DiscreteParams",
    "threshold_alpha",
    "tent_map",
    "tent_map_iterated",
    "initial_puff",
    "step",
    "simulate_discrete",
]

#: Default parameters of the discrete model (Barkley 2011, Appendix).
EPS1: float = 0.04  #: u-relaxation rate towards laminar.
EPS2: float = 0.2  #: rate at which turbulence suppresses u.
K: int = 2  #: number of tent-map iterates composing F = f^k.
BETA: float = 0.4  #: escape-window width; > 0 enables spontaneous decay.
GAMMA: float = 0.95  #: contraction slope of the tent map's laminar branch.
C: float = 0.45  #: u-advection (upwind) coefficient, ~ 1/dx.
D: float = 0.15  #: diffusive coupling strength (must be <= 0.5 for stability).


@dataclass(frozen=True)
class DiscreteParams:
    """Parameters of the discrete coupled-map-lattice model.

    Parameters
    ----------
    R : float
        Discrete Reynolds-number proxy (control parameter; ``R ~ 2000`` ~ ``Re``).
    eps1, eps2, beta, gamma, c, d : float, optional
        Model parameters; default to the Barkley (2011) values.
    k : int, optional
        Number of tent-map iterates in ``F = f^k``. Default ``2``.
    """

    R: float
    eps1: float = EPS1
    eps2: float = EPS2
    beta: float = BETA
    gamma: float = GAMMA
    c: float = C
    d: float = D
    k: int = K


def threshold_alpha(u: ArrayLike, R: float) -> NDArray[np.float64]:
    """Tent-map threshold ``alpha(u, R) = 2000 (1 - 0.8 u) / R``.

    Parameters
    ----------
    u : array_like
        Velocity-proxy field.
    R : float
        Discrete Reynolds-number proxy.

    Returns
    -------
    numpy.ndarray
        Threshold ``alpha`` per lattice site.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    return 2000.0 * (1.0 - 0.8 * u_arr) / R


def tent_map(
    q: ArrayLike, alpha: ArrayLike, beta: float = BETA, gamma: float = GAMMA
) -> NDArray[np.float64]:
    """Single piecewise-linear tent map ``f(q)`` (see module docstring).

    Parameters
    ----------
    q : array_like
        Input intensity values.
    alpha : array_like
        Per-site threshold from :func:`threshold_alpha` (broadcasts with ``q``).
    beta, gamma : float, optional
        Tent-map shape parameters.

    Returns
    -------
    numpy.ndarray
        ``f(q)`` evaluated piecewise with breakpoints ``Q1``, ``Q2``.
    """
    q_arr, a_arr = np.broadcast_arrays(
        np.asarray(q, dtype=np.float64), np.asarray(alpha, dtype=np.float64)
    )
    q1 = a_arr / (2.0 - gamma)
    q2 = (4.0 + beta - a_arr - gamma * q1) / (2.0 + beta)
    conditions = [q_arr < q1, q_arr < 1.0, q_arr < q2, np.ones(q_arr.shape, bool)]
    choices = [
        gamma * q_arr,
        2.0 * q_arr - a_arr,
        4.0 + beta - a_arr - (2.0 + beta) * q_arr,
        gamma * q1,
    ]
    return np.select(conditions, choices)


def tent_map_iterated(
    q: ArrayLike,
    alpha: ArrayLike,
    k: int = K,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> NDArray[np.float64]:
    """The map ``F = f^k``: ``k`` successive applications of :func:`tent_map`.

    The threshold ``alpha`` (which depends on the slowly varying ``u``) is held
    fixed across the ``k`` iterates, as in the model.

    Parameters
    ----------
    q, alpha : array_like
        Intensity and per-site threshold.
    k : int, optional
        Number of iterates. Default ``2``.
    beta, gamma : float, optional
        Tent-map shape parameters.

    Returns
    -------
    numpy.ndarray
        ``f^k(q)``.
    """
    out = np.asarray(q, dtype=np.float64)
    for _ in range(int(k)):
        out = tent_map(out, alpha, beta, gamma)
    return out


def initial_puff(
    n: int, width: int = 20, q_level: float = 1.0, seed: Optional[int] = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build an initial localised turbulent puff on a lattice of ``n`` sites.

    A band of ``width`` sites at the centre is seeded with turbulent intensity
    around ``q_level`` (with small random fluctuations to break symmetry and seed
    the chaotic dynamics); the rest is laminar (``q = 0``, ``u = 1``).

    Parameters
    ----------
    n : int
        Number of lattice sites.
    width : int, optional
        Width of the turbulent seed in sites. Default ``20``.
    q_level : float, optional
        Mean initial turbulent intensity in the seed. Default ``1.0``.
    seed : int or None, optional
        Seed for the random initial fluctuations (reproducibility).

    Returns
    -------
    tuple of numpy.ndarray
        Initial ``(q, u)`` lattice fields, each shape ``(n,)``.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros(n, dtype=np.float64)
    u = np.ones(n, dtype=np.float64)
    start = (n - width) // 2
    q[start : start + width] = q_level * (1.0 + 0.1 * rng.standard_normal(width))
    return q, u


def step(
    q: NDArray[np.float64], u: NDArray[np.float64], params: DiscreteParams
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance the coupled-map lattice by one time step (Eqs. 3-4).

    Both updates are explicit functions of the current state ``(q, u)``: the
    ``q``-update applies discrete diffusion then the iterated tent map; the
    ``u``-update relaxes towards laminar, is suppressed by turbulence, and is
    upwind-advected.

    Parameters
    ----------
    q, u : numpy.ndarray
        Current lattice fields. The lattice is the **last** axis, so a 1-D
        ``(n,)`` state or a stacked ensemble ``(m, n)`` of ``m`` independent
        realizations are both accepted (rolls use ``axis=-1``).
    params : DiscreteParams
        Model parameters.

    Returns
    -------
    tuple of numpy.ndarray
        Updated ``(q, u)`` fields, same shape as the inputs.
    """
    laplacian = np.roll(q, 1, axis=-1) - 2.0 * q + np.roll(q, -1, axis=-1)
    q_arg = q + params.d * laplacian
    alpha = threshold_alpha(u, params.R)
    q_new = tent_map_iterated(q_arg, alpha, params.k, params.beta, params.gamma)
    u_new = (
        u
        + params.eps1 * (1.0 - u)
        - params.eps2 * u * q
        - params.c * (u - np.roll(u, 1, axis=-1))
    )
    return q_new, u_new


def simulate_discrete(
    q0: NDArray[np.float64],
    u0: NDArray[np.float64],
    params: DiscreteParams,
    n_steps: int,
    store_every: int = 1,
) -> NDArray[np.float64]:
    """Iterate the lattice for ``n_steps`` and return the ``q`` space-time array.

    Parameters
    ----------
    q0, u0 : numpy.ndarray
        Initial lattice fields, shape ``(n,)``.
    params : DiscreteParams
        Model parameters.
    n_steps : int
        Number of time steps to advance.
    store_every : int, optional
        Store a snapshot every this many steps (the initial state is always
        stored). Default ``1``.

    Returns
    -------
    numpy.ndarray
        Space-time array of ``q``, shape ``(n_stored, n)`` (row 0 is the initial
        state); suitable for Fig. 4 space-time diagrams.
    """
    q = np.array(q0, dtype=np.float64)
    u = np.array(u0, dtype=np.float64)
    stored = [q.copy()]
    for i in range(1, int(n_steps) + 1):
        q, u = step(q, u, params)
        if i % store_every == 0:
            stored.append(q.copy())
    return np.asarray(stored)
