"""Discrete coupled-map-lattice Barkley model (Phase 2 — DOCUMENTED STUBS).

This module will implement the *discrete* Barkley (2011) model (his Eqs. 3-6),
a coupled-map lattice that, unlike the continuous model, supports the stochastic
puff *decay* and *splitting* responsible for the memoryless lifetime statistics
and the directed-percolation onset of sustained turbulence.

Model
-----
With spatial index ``i`` and time index ``n``:

.. math::

    q_i^{n+1} &= F\\!\\left(q_i^{n} + d\\,(q_{i-1}^{n} - 2 q_i^{n} + q_{i+1}^{n}),
                 \\; u_i^{n}\\right), \\\\
    u_i^{n+1} &= u_i^{n} + \\varepsilon_1 (1 - u_i^{n})
                 - \\varepsilon_2\\, u_i^{n} q_i^{n}
                 - c\\,(u_i^{n} - u_{i-1}^{n}),

where the map ``F = f^k`` is the ``k``-fold iterate of a piecewise-linear
**tent map** ``f`` whose shape depends on ``u`` through the threshold

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

Parameters: ``eps1=0.04``, ``eps2=0.2``, ``k=2``, ``beta=0.4``, ``gamma=0.95``,
``c=0.45``, ``d=0.15``. ``beta > 0`` opens the escape window that enables
spontaneous decay (transient chaos); ``d <= 0.5`` is required for numerical
stability of the explicit diffusion. ``R`` is the discrete Reynolds-number
proxy, scaled so ``R ~ 2000`` corresponds to ``Re ~ 2000`` (distinct from the
continuous model's ``r = O(1)``).

Status
------
**Phase 2 — not yet implemented.** Every public function below raises
``NotImplementedError`` and documents the intended behaviour. See ``ROADMAP.md``
for the implementation and validation plan (reproduce Barkley Fig. 4 space-time
diagrams for decaying puff ``R=1900``, splitting ``R=2200``, slug ``R=3000``).

References
----------
Barkley, D. (2011). Simplifying the complexity of pipe flow.
*Phys. Rev. E* **84**, 016309. arXiv:1101.4125.
"""

from __future__ import annotations

from dataclasses import dataclass

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

_STUB = "Phase 2 not yet implemented — see ROADMAP.md"


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

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


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

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


def tent_map_iterated(
    q: ArrayLike,
    alpha: ArrayLike,
    k: int = K,
    beta: float = BETA,
    gamma: float = GAMMA,
) -> NDArray[np.float64]:
    """The map ``F = f^k``: ``k`` successive applications of :func:`tent_map`.

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

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


def initial_puff(
    n: int, width: int = 20, q_level: float = 1.0, seed: int | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Build an initial localised turbulent puff on a lattice of ``n`` sites.

    Parameters
    ----------
    n : int
        Number of lattice sites.
    width : int, optional
        Width of the turbulent seed in sites. Default ``20``.
    q_level : float, optional
        Initial turbulent intensity in the seed. Default ``1.0``.
    seed : int or None, optional
        Seed for the random initial fluctuations (reproducibility).

    Returns
    -------
    tuple of numpy.ndarray
        Initial ``(q, u)`` lattice fields, each shape ``(n,)``.

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


def step(
    q: NDArray[np.float64], u: NDArray[np.float64], params: DiscreteParams
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Advance the coupled-map lattice by one time step (Eqs. 3-4).

    Parameters
    ----------
    q, u : numpy.ndarray
        Current lattice fields, shape ``(n,)`` (periodic).
    params : DiscreteParams
        Model parameters.

    Returns
    -------
    tuple of numpy.ndarray
        Updated ``(q, u)`` fields.

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


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
        Store a snapshot every this many steps. Default ``1``.

    Returns
    -------
    numpy.ndarray
        Space-time array of ``q``, shape ``(n_stored, n)`` (for Fig. 4 diagrams).

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)
