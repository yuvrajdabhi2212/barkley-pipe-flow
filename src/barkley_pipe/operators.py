"""Finite-difference spatial operators for the Barkley pipe-flow model.

This module provides the one-dimensional finite-difference stencils used by
:mod:`barkley_pipe.continuous` to assemble the method-of-lines right-hand side
of the Barkley (2011) reduced model

.. math::

    q_t + U q_x &= q\\,[u + r - 1 - (r + \\delta)(q - 1)^2] + q_{xx} \\\\
    u_t + U u_x &= \\varepsilon_1 (1 - u) - \\varepsilon_2 u q - u_x .

All operators assume a **uniform grid** with **periodic boundary conditions**,
implemented with :func:`numpy.roll` so that wrap-around is exact and the
stencils are fully vectorised (no Python-level loops).  Two families are
provided.

First-derivative *advection* operators (for the ``U q_x`` and ``u_x`` terms):

* :func:`first_derivative_upwind` -- first-order one-sided differencing whose
  stencil follows the sign of the advection speed.  Upwinding suppresses the
  dispersive two-cell (``2 dx``) oscillations that a centred scheme produces at
  the sharp turbulent fronts of puffs and slugs, at the cost of an
  :math:`O(\\Delta x)` numerical-diffusion term.
* :func:`first_derivative_central` -- second-order centred differencing,
  provided so the advection scheme is swappable and the two can be compared.

Second-derivative *diffusion* operator (for the ``q_xx`` term):

* :func:`laplacian` -- second-order centred three-point stencil.

A small registry, :func:`get_advection_operator`, returns either first-derivative
operator behind a common ``op(field, dx, speed)`` signature so callers can swap
schemes by name.

Convergence on a smooth periodic field (uniform grid):

================================  =====  ===============================
operator                          order  leading truncation error
================================  =====  ===============================
``first_derivative_upwind``         1     :math:`-\\tfrac{\\Delta x}{2} f''`
``first_derivative_central``        2     :math:`-\\tfrac{\\Delta x^2}{6} f'''`
``laplacian``                       2     :math:`-\\tfrac{\\Delta x^2}{12} f''''`
================================  =====  ===============================

References
----------
Barkley, D. (2011). Simplifying the complexity of pipe flow.
*Phys. Rev. E* **84**, 016309. arXiv:1101.4125.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "first_derivative_upwind",
    "first_derivative_central",
    "laplacian",
    "get_advection_operator",
    "AdvectionOperator",
]

#: Type of an advection operator with a uniform ``op(field, dx, speed)`` call
#: signature, as returned by :func:`get_advection_operator`.
AdvectionOperator = Callable[[NDArray[np.float64], float, float], NDArray[np.float64]]


def first_derivative_upwind(
    field: NDArray[np.float64], dx: float, speed: float = 1.0
) -> NDArray[np.float64]:
    """First-order upwind approximation of ``d(field)/dx`` on a periodic grid.

    The one-sided stencil is selected from the *upwind* side, i.e. the side from
    which information propagates for an advection term ``speed * d(field)/dx``:

    * ``speed >= 0`` (transport in ``+x``): backward difference
      ``(field[i] - field[i-1]) / dx``;
    * ``speed < 0`` (transport in ``-x``): forward difference
      ``(field[i+1] - field[i]) / dx``.

    Upwinding is monotone and damps the spurious ``2 dx`` oscillations a centred
    scheme generates at the sharp turbulent fronts of puffs/slugs, introducing
    instead an :math:`O(\\Delta x)` numerical diffusion (leading error
    :math:`-\\tfrac{\\Delta x}{2}\\,\\mathrm{sgn}(\\text{speed})\\,f''`).

    Parameters
    ----------
    field : numpy.ndarray
        Field samples on a uniform, periodic grid, shape ``(N,)``.
    dx : float
        Grid spacing. Must be positive.
    speed : float, optional
        Advection speed; only its sign selects the stencil. Default ``1.0``
        (flow left-to-right). ``speed == 0`` falls back to the backward stencil.

    Returns
    -------
    numpy.ndarray
        Approximation of ``d(field)/dx``, same shape as ``field``.

    Notes
    -----
    Periodicity is handled by :func:`numpy.roll`; index ``i-1`` of ``field[0]``
    wraps to ``field[-1]`` and ``i+1`` of ``field[-1]`` wraps to ``field[0]``.
    """
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx!r}")
    if speed >= 0.0:
        # backward difference: f[i] - f[i-1]
        return (field - np.roll(field, 1)) / dx
    # forward difference: f[i+1] - f[i]
    return (np.roll(field, -1) - field) / dx


def first_derivative_central(
    field: NDArray[np.float64], dx: float
) -> NDArray[np.float64]:
    """Second-order centred approximation of ``d(field)/dx`` on a periodic grid.

    Uses the symmetric three-point stencil
    ``(field[i+1] - field[i-1]) / (2 dx)``. Second-order accurate but
    *dispersive*: it can generate two-cell oscillations near steep fronts, which
    is why :func:`first_derivative_upwind` is the default for advection in this
    model. Provided for scheme comparison and for smooth diagnostics.

    Parameters
    ----------
    field : numpy.ndarray
        Field samples on a uniform, periodic grid, shape ``(N,)``.
    dx : float
        Grid spacing. Must be positive.

    Returns
    -------
    numpy.ndarray
        Approximation of ``d(field)/dx``, same shape as ``field``.
    """
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx!r}")
    return (np.roll(field, -1) - np.roll(field, 1)) / (2.0 * dx)


def laplacian(field: NDArray[np.float64], dx: float) -> NDArray[np.float64]:
    """Second-order centred approximation of ``d^2(field)/dx^2`` (periodic).

    Uses the three-point stencil
    ``(field[i-1] - 2 field[i] + field[i+1]) / dx**2``. In one dimension this is
    the Laplacian and supplies the diffusion term ``q_xx`` of the model.

    Parameters
    ----------
    field : numpy.ndarray
        Field samples on a uniform, periodic grid, shape ``(N,)``.
    dx : float
        Grid spacing. Must be positive.

    Returns
    -------
    numpy.ndarray
        Approximation of ``d^2(field)/dx^2``, same shape as ``field``.
    """
    if dx <= 0.0:
        raise ValueError(f"dx must be positive, got {dx!r}")
    return (np.roll(field, -1) - 2.0 * field + np.roll(field, 1)) / (dx * dx)


def get_advection_operator(scheme: str) -> AdvectionOperator:
    """Return an advection operator by name with a uniform call signature.

    The returned callable has signature ``op(field, dx, speed) -> ndarray`` so
    that :mod:`barkley_pipe.continuous` can switch between schemes without
    knowing their individual signatures. For the centred scheme the ``speed``
    argument is accepted but ignored (the stencil is symmetric).

    Parameters
    ----------
    scheme : {'upwind', 'central'}
        Name of the finite-difference advection scheme.

    Returns
    -------
    AdvectionOperator
        Callable ``op(field, dx, speed)`` approximating ``d(field)/dx``.

    Raises
    ------
    ValueError
        If ``scheme`` is not a known scheme name.
    """
    if scheme == "upwind":
        return first_derivative_upwind
    if scheme == "central":
        return lambda field, dx, speed=0.0: first_derivative_central(field, dx)
    raise ValueError(
        f"unknown advection scheme {scheme!r}; choose 'upwind' or 'central'"
    )
