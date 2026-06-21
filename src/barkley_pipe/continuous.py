"""Continuous two-PDE Barkley model solved by the method of lines (MOL).

This module assembles and integrates the Barkley (2011) reduced model

.. math::

    q_t + U q_x &= q\\,[\\,u + r - 1 - (r + \\delta)(q - 1)^2\\,] + q_{xx}, \\\\
    u_t + U u_x &= \\varepsilon_1 (1 - u) - \\varepsilon_2\\,u\\,q - u_x ,

for the turbulence-intensity field ``q(x, t) >= 0`` and velocity proxy
``u(x, t)`` on a periodic domain.

**Method of lines.**  Space is discretised on a uniform periodic grid (see
:class:`PeriodicGrid`); the spatial operators from :mod:`barkley_pipe.operators`
turn the PDEs into a system of ``2N`` coupled ODEs ``dy/dt = F(t, y)`` with the
state ``y = [q_0..q_{N-1}, u_0..u_{N-1}]``, which is then advanced in time by
:func:`scipy.integrate.solve_ivp`.

**Advection speeds.**  Rearranging the ``u`` equation,

.. math:: u_t = \\varepsilon_1(1-u) - \\varepsilon_2 u q - (U + 1)\\,u_x ,

so ``q`` is advected at speed ``U`` and ``u`` at speed ``U + 1``.  The fixed
*relative* speed of ``1`` between the two fields is the physical asymmetry that
gives puffs their sharp upstream front and long downstream recovery tail; ``U``
itself only sets the reference frame and may be chosen freely (e.g. ``U = 0`` is
the frame co-moving with the ``q`` field).  The first-derivative terms use the
swappable scheme from :func:`~barkley_pipe.operators.get_advection_operator`
(default first-order upwind); the diffusion term ``q_xx`` uses the centred
Laplacian.

**Stability / cost (explicit integrators).**  ``solve_ivp`` with ``RK45`` is
*adaptive*, so it shrinks the step automatically for stability; the classical
explicit limits below set the expected step size and hence the cost:

* advection (CFL): ``|speed| * dt / dx <= 1`` with ``speed = max(|U|, |U+1|)``;
* diffusion: ``dt <= dx**2 / 2``.

On a fine grid the diffusion limit usually dominates (``dt ~ dx**2``).

**Stiffness.**  Because ``eps1 = 0.04`` makes the ``u``-relaxation slow relative
to the fast diffusive/reaction scales, the system is mildly stiff.  ``RK45`` is
the robust default; if it takes very many steps, diverges, or fails, switch
``method`` to an implicit integrator (``'Radau'`` or ``'BDF'``, with ``'LSODA'``
a good automatic switcher) via the :func:`simulate` argument.

References
----------
Barkley, D. (2011). Simplifying the complexity of pipe flow.
*Phys. Rev. E* **84**, 016309. arXiv:1101.4125.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from .nullclines import DELTA, EPS1, EPS2
from .operators import AdvectionOperator, get_advection_operator, laplacian

__all__ = [
    "PeriodicGrid",
    "ContinuousParams",
    "ContinuousResult",
    "laminar_state",
    "puff_seed",
    "continuous_rhs",
    "simulate",
]


@dataclass(frozen=True)
class PeriodicGrid:
    """A uniform, periodic 1-D grid on ``[0, length)``.

    Parameters
    ----------
    n : int
        Number of grid points (``>= 4``).
    length : float
        Physical domain length ``L`` in model units (``> 0``).

    Attributes
    ----------
    dx : float
        Grid spacing ``length / n``.
    x : numpy.ndarray
        Node coordinates ``[0, dx, 2 dx, ..., (n-1) dx]`` (cached).
    """

    n: int
    length: float
    dx: float = field(init=False)
    x: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n < 4:
            raise ValueError(f"n must be >= 4, got {self.n}")
        if self.length <= 0.0:
            raise ValueError(f"length must be positive, got {self.length}")
        dx = self.length / self.n
        object.__setattr__(self, "dx", dx)
        object.__setattr__(self, "x", np.arange(self.n) * dx)


@dataclass(frozen=True)
class ContinuousParams:
    """Parameters of the continuous Barkley model.

    Parameters
    ----------
    r : float
        Reynolds-number proxy (the control parameter).
    eps1, eps2, delta : float, optional
        Model parameters; default to the Barkley (2011) values
        ``0.04``, ``0.2``, ``0.1``.
    U : float, optional
        Bulk advection speed of the ``q`` field (``u`` is advected at ``U + 1``).
        Sets the reference frame; default ``0.0``.
    advection : {'upwind', 'central'}, optional
        Finite-difference scheme for the first-derivative (advection) terms.
        Default ``'upwind'`` (monotone; avoids oscillations at sharp fronts).
    """

    r: float
    eps1: float = EPS1
    eps2: float = EPS2
    delta: float = DELTA
    U: float = 0.0
    advection: str = "upwind"


@dataclass
class ContinuousResult:
    """Container for the output of :func:`simulate`.

    Attributes
    ----------
    t : numpy.ndarray
        Snapshot times, shape ``(M,)``.
    x : numpy.ndarray
        Grid coordinates, shape ``(N,)``.
    q, u : numpy.ndarray
        Field snapshots, shape ``(M, N)`` (row ``k`` is the field at ``t[k]``).
    params : ContinuousParams
        Parameters used for the run.
    grid : PeriodicGrid
        Grid used for the run.
    success : bool
        Whether the integrator reported success.
    message : str
        Solver status message.
    nfev : int
        Number of right-hand-side evaluations.
    """

    t: NDArray[np.float64]
    x: NDArray[np.float64]
    q: NDArray[np.float64]
    u: NDArray[np.float64]
    params: ContinuousParams
    grid: PeriodicGrid
    success: bool
    message: str
    nfev: int


def laminar_state(grid: PeriodicGrid) -> NDArray[np.float64]:
    """Return the laminar state ``(q, u) = (0, 1)`` as a packed state vector.

    Parameters
    ----------
    grid : PeriodicGrid
        Grid defining the number of points.

    Returns
    -------
    numpy.ndarray
        Packed state ``[q (=0), u (=1)]`` of length ``2 * grid.n``.
    """
    return np.concatenate([np.zeros(grid.n), np.ones(grid.n)])


def puff_seed(
    grid: PeriodicGrid,
    center: Optional[float] = None,
    width: float = 5.0,
    q_amplitude: float = 1.0,
    u_dip: float = 0.6,
) -> NDArray[np.float64]:
    """Construct a localised Gaussian turbulent seed on a laminar background.

    Sets a Gaussian bump in ``q`` and a matching dip in ``u`` around ``center``;
    used to trigger a puff/slug from the (otherwise stable) laminar state.

    Parameters
    ----------
    grid : PeriodicGrid
        Grid on which to build the state.
    center : float, optional
        Centre of the bump in model units. Default: middle of the domain.
    width : float, optional
        Gaussian standard deviation (model units). Default ``5.0``.
    q_amplitude : float, optional
        Peak turbulence intensity of the seed. Default ``1.0``. Must be large
        enough to exceed the excitability threshold to trigger a sustained puff.
    u_dip : float, optional
        Peak reduction of ``u`` at the seed centre (``u = 1 - u_dip`` at peak).
        Default ``0.6``. Kept ``< 1`` so ``u`` stays non-negative.

    Returns
    -------
    numpy.ndarray
        Packed state ``[q, u]`` of length ``2 * grid.n``.
    """
    if center is None:
        center = 0.5 * grid.length
    bump = np.exp(-0.5 * ((grid.x - center) / width) ** 2)
    q = q_amplitude * bump
    u = 1.0 - u_dip * bump
    return np.concatenate([q, u])


def continuous_rhs(
    t: float,
    y: NDArray[np.float64],
    grid: PeriodicGrid,
    params: ContinuousParams,
    advect_op: Optional[AdvectionOperator] = None,
) -> NDArray[np.float64]:
    """Method-of-lines right-hand side ``dy/dt`` of the continuous model.

    Parameters
    ----------
    t : float
        Time (unused; the system is autonomous, but ``solve_ivp`` passes it).
    y : numpy.ndarray
        Packed state ``[q, u]`` of length ``2 * grid.n``.
    grid : PeriodicGrid
        Spatial grid.
    params : ContinuousParams
        Model parameters.
    advect_op : AdvectionOperator, optional
        Advection operator with signature ``op(field, dx, speed)``. If ``None``
        it is resolved from ``params.advection`` (convenient for direct calls;
        :func:`simulate` resolves it once and passes it in to avoid per-step
        dictionary lookups).

    Returns
    -------
    numpy.ndarray
        Packed time derivative ``[dq/dt, du/dt]`` of length ``2 * grid.n``.
    """
    if advect_op is None:
        advect_op = get_advection_operator(params.advection)

    n = grid.n
    dx = grid.dx
    q = y[:n]
    u = y[n:]

    q_x = advect_op(q, dx, params.U)
    u_x = advect_op(u, dx, params.U + 1.0)
    q_xx = laplacian(q, dx)

    reaction_q = q * (u + params.r - 1.0 - (params.r + params.delta) * (q - 1.0) ** 2)
    reaction_u = params.eps1 * (1.0 - u) - params.eps2 * u * q

    dq_dt = -params.U * q_x + q_xx + reaction_q
    du_dt = -(params.U + 1.0) * u_x + reaction_u
    return np.concatenate([dq_dt, du_dt])


def simulate(
    grid: PeriodicGrid,
    params: ContinuousParams,
    y0: NDArray[np.float64],
    t_span: tuple[float, float],
    *,
    n_snapshots: int = 200,
    method: str = "RK45",
    rtol: float = 1e-6,
    atol: float = 1e-8,
    max_step: Optional[float] = None,
) -> ContinuousResult:
    """Integrate the continuous model in time via the method of lines.

    Parameters
    ----------
    grid : PeriodicGrid
        Spatial grid.
    params : ContinuousParams
        Model parameters.
    y0 : numpy.ndarray
        Initial packed state ``[q, u]`` of length ``2 * grid.n`` (e.g. from
        :func:`puff_seed`).
    t_span : tuple of float
        ``(t0, t1)`` integration interval.
    n_snapshots : int, optional
        Number of equally spaced snapshots stored (via ``t_eval``) for plotting
        space-time diagrams. Default ``200``.
    method : str, optional
        ``solve_ivp`` integrator. Default ``'RK45'``. Use ``'Radau'``/``'BDF'``/
        ``'LSODA'`` for stiff regimes (see module docstring).
    rtol, atol : float, optional
        Relative/absolute tolerances passed to ``solve_ivp``.
    max_step : float, optional
        Maximum internal step. Default ``None`` (unbounded; the integrator
        chooses). Set this to bound the step on very long runs if needed.

    Returns
    -------
    ContinuousResult
        Snapshots and metadata. Inspect ``.success``/``.message`` to confirm the
        integration completed.

    Notes
    -----
    The advection operator is resolved once here and passed into
    :func:`continuous_rhs` to avoid a per-step lookup.
    """
    advect_op = get_advection_operator(params.advection)
    t_eval = np.linspace(t_span[0], t_span[1], n_snapshots)
    sol = solve_ivp(
        continuous_rhs,
        t_span,
        y0,
        method=method,
        t_eval=t_eval,
        args=(grid, params, advect_op),
        rtol=rtol,
        atol=atol,
        max_step=np.inf if max_step is None else max_step,
    )
    n = grid.n
    return ContinuousResult(
        t=sol.t,
        x=grid.x,
        q=sol.y[:n].T,
        u=sol.y[n:].T,
        params=params,
        grid=grid,
        success=bool(sol.success),
        message=str(sol.message),
        nfev=int(sol.nfev),
    )
