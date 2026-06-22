"""Matplotlib helpers for visualising the continuous model.

Thin, dependency-light wrappers used by the notebooks to reproduce the paper's
figures:

* :func:`plot_profile` -- ``q(x)`` and ``u(x)`` profiles at one snapshot
  (Barkley Fig. 1);
* :func:`plot_phase_plane` -- the ``q``-``u`` phase plane with analytic
  nullclines and an overlaid trajectory (Barkley Fig. 2);
* :func:`plot_spacetime` -- an ``x``-``t`` space-time diagram of ``q``.

``matplotlib`` is imported lazily inside each function so that importing
:mod:`barkley_pipe` does not require a display backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
from numpy.typing import NDArray

from .diagnostics import puff_center
from .nullclines import q_nullcline, u_nullcline

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.axes import Axes

    from .continuous import ContinuousResult

__all__ = ["plot_profile", "plot_phase_plane", "plot_spacetime",
           "plot_discrete_spacetime", "recenter_snapshot"]

# Consistent colours for the two fields across all figures.
_Q_COLOR = "#c1272d"  # turbulence intensity q
_U_COLOR = "#0072b2"  # velocity proxy u


def recenter_snapshot(
    q: NDArray[np.float64],
    u: NDArray[np.float64],
    length: float,
    target: Optional[float] = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Roll a periodic snapshot so the puff centroid sits at ``target``.

    Parameters
    ----------
    q, u : numpy.ndarray
        Field snapshots, shape ``(N,)``.
    length : float
        Domain length.
    target : float, optional
        Desired centroid position. Default: domain centre ``length / 2``.

    Returns
    -------
    tuple of numpy.ndarray
        The rolled ``(q, u)`` arrays.
    """
    if target is None:
        target = 0.5 * length
    center = puff_center(q, length)
    if not np.isfinite(center):
        return q, u
    dx = length / q.size
    shift = int(round((target - center) / dx))
    return np.roll(q, shift), np.roll(u, shift)


def plot_profile(
    result: "ContinuousResult",
    index: int = -1,
    *,
    recenter: bool = True,
    ax: Optional["Axes"] = None,
) -> "Axes":
    """Plot ``q(x)`` and ``u(x)`` at one snapshot (reproduces Barkley Fig. 1).

    Parameters
    ----------
    result : ContinuousResult
        Output of :func:`barkley_pipe.continuous.simulate`.
    index : int, optional
        Snapshot index to plot. Default ``-1`` (final time).
    recenter : bool, optional
        If ``True`` (default), roll the periodic snapshot so the puff is centred
        in the domain for a clean, frame-independent profile.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created if omitted.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    q = result.q[index]
    u = result.u[index]
    if recenter:
        q, u = recenter_snapshot(q, u, result.grid.length)

    ax.plot(result.x, q, color=_Q_COLOR, lw=2, label=r"$q$ (turbulence intensity)")
    ax.plot(result.x, u, color=_U_COLOR, lw=2, label=r"$u$ (velocity proxy)")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$q,\ u$")
    ax.set_title(
        rf"$r = {result.params.r:g}$,  $t = {result.t[index]:.0f}$  "
        rf"({result.params.advection} advection)"
    )
    ax.legend(loc="best", frameon=False)
    ax.margins(x=0)
    return ax


def plot_phase_plane(
    r: float,
    *,
    trajectory: Optional[tuple[NDArray[np.float64], NDArray[np.float64]]] = None,
    q_max: float = 2.0,
    ax: Optional["Axes"] = None,
    label: Optional[str] = None,
) -> "Axes":
    """Plot the ``q``-``u`` phase plane with analytic nullclines (Barkley Fig. 2).

    Draws the trivial (``q = 0``) and non-trivial q-nullclines and the
    u-nullcline for the given ``r``, optionally overlaying a state-space
    trajectory ``(q(x), u(x))`` (e.g. a puff or slug profile).

    Parameters
    ----------
    r : float
        Reynolds-number proxy.
    trajectory : tuple of numpy.ndarray, optional
        ``(q, u)`` samples to overlay as a curve in the plane.
    q_max : float, optional
        Upper limit of the ``q`` axis. Default ``2.0``.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created if omitted.
    label : str, optional
        Legend label for the overlaid trajectory.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    q_grid = np.linspace(0.0, q_max, 400)
    ax.plot(q_grid, q_nullcline(q_grid, r), color=_Q_COLOR, lw=1.5,
            label=r"$q$-nullcline ($\dot q=0$)")
    ax.axvline(0.0, color=_Q_COLOR, lw=1.5, ls=":")
    ax.plot(q_grid, u_nullcline(q_grid), color=_U_COLOR, lw=1.5,
            label=r"$u$-nullcline ($\dot u=0$)")

    if trajectory is not None:
        q_traj, u_traj = trajectory
        ax.plot(q_traj, u_traj, color="0.2", lw=2, label=label or "trajectory")

    ax.set_xlim(0.0, q_max)
    ax.set_ylim(0.0, 1.0 + 0.2)
    ax.set_xlabel(r"$q$")
    ax.set_ylabel(r"$u$")
    ax.set_title(rf"Phase plane, $r = {r:g}$")
    ax.legend(loc="best", frameon=False, fontsize=9)
    return ax


def plot_spacetime(
    result: "ContinuousResult",
    *,
    comoving_speed: Optional[float] = None,
    ax: Optional["Axes"] = None,
    cmap: str = "inferno",
) -> "Axes":
    """Plot an ``x``-``t`` space-time diagram of ``q`` (reaction front history).

    Parameters
    ----------
    result : ContinuousResult
        Output of :func:`barkley_pipe.continuous.simulate`.
    comoving_speed : float, optional
        If given, shift each row by ``-comoving_speed * t`` (periodic roll) to
        view the dynamics in a frame moving at this speed, so a steadily
        travelling structure appears vertical.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created if omitted.
    cmap : str, optional
        Matplotlib colormap name. Default ``'inferno'``.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the image.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    field = result.q
    if comoving_speed is not None:
        dx = result.grid.dx
        field = np.empty_like(field)
        for k, t in enumerate(result.t):
            shift = int(round(-comoving_speed * t / dx))
            field[k] = np.roll(result.q[k], shift)

    im = ax.imshow(
        field,
        origin="lower",
        aspect="auto",
        extent=(float(result.x[0]), float(result.x[-1]),
                float(result.t[0]), float(result.t[-1])),
        cmap=cmap,
    )
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$t$")
    frame = "" if comoving_speed is None else rf" (frame speed $={comoving_speed:g}$)"
    ax.set_title(rf"Space-time $q(x,t)$, $r={result.params.r:g}$" + frame)
    plt.colorbar(im, ax=ax, label=r"$q$")
    return ax


def plot_discrete_spacetime(
    q_spacetime: NDArray[np.float64],
    *,
    store_every: int = 1,
    ax: Optional["Axes"] = None,
    cmap: str = "inferno",
    title: Optional[str] = None,
) -> "Axes":
    """Plot an ``i``-``n`` space-time diagram of the discrete-model ``q`` lattice.

    Companion to :func:`plot_spacetime` for the discrete coupled-map lattice
    (:func:`barkley_pipe.discrete.simulate_discrete`), which returns a raw
    ``(n_stored, n_sites)`` array rather than a :class:`ContinuousResult`.
    Reproduces the style of Barkley Fig. 4.

    Parameters
    ----------
    q_spacetime : numpy.ndarray
        Space-time array of ``q``, shape ``(n_stored, n_sites)`` (row 0 earliest).
    store_every : int, optional
        Steps between stored rows, used only to label the time axis. Default ``1``.
    ax : matplotlib.axes.Axes, optional
        Axis to draw on. A new figure/axis is created if omitted.
    cmap : str, optional
        Matplotlib colormap name. Default ``'inferno'``.
    title : str, optional
        Axis title.

    Returns
    -------
    matplotlib.axes.Axes
        The axis containing the image.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))

    arr = np.asarray(q_spacetime)
    n_rows, n_sites = arr.shape
    im = ax.imshow(
        arr,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=(0.0, float(n_sites), 0.0, float(n_rows * store_every)),
    )
    ax.set_xlabel(r"lattice site $i$")
    ax.set_ylabel(r"time step $n$")
    if title is not None:
        ax.set_title(title)
    plt.colorbar(im, ax=ax, label=r"$q$")
    return ax
