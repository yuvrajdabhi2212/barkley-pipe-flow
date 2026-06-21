"""Scalar diagnostics for the spatially extended Barkley model.

Reduce a field snapshot ``q(x)`` (and time series of snapshots) to the few
scalars used to characterise the regimes:

* :func:`turbulent_energy`, :func:`turbulent_mass` -- integrated measures of how
  much turbulence is present (used to detect whether a puff is sustained,
  decaying, or growing);
* :func:`puff_center` -- a periodic-aware centroid for locating/recentring a
  localised structure;
* :func:`leading_edge`, :func:`trailing_edge` -- front positions from a
  threshold crossing;
* :func:`front_speed` -- the slope of a front trajectory ``position(t)``, i.e.
  the propagation speed used to distinguish an equilibrium puff (symmetric
  fronts) from an expanding slug (leading front faster than trailing).

All integrals use the rectangle rule on the uniform grid, which is spectrally
accurate for the smooth, periodic fields here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "turbulent_energy",
    "turbulent_mass",
    "puff_center",
    "leading_edge",
    "trailing_edge",
    "track_fronts",
    "front_speed",
    "front_kinematics",
]


def turbulent_energy(q: ArrayLike, dx: float) -> float:
    """Integrated turbulent energy ``E = ∫ q^2 dx`` on the periodic grid.

    Parameters
    ----------
    q : array_like
        Turbulence-intensity field on a uniform grid, shape ``(N,)``.
    dx : float
        Grid spacing.

    Returns
    -------
    float
        ``sum(q**2) * dx``.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    return float(np.sum(q_arr * q_arr) * dx)


def turbulent_mass(q: ArrayLike, dx: float) -> float:
    """Integrated turbulent mass ``M = ∫ q dx`` on the periodic grid.

    Parameters
    ----------
    q : array_like
        Turbulence-intensity field on a uniform grid, shape ``(N,)``.
    dx : float
        Grid spacing.

    Returns
    -------
    float
        ``sum(q) * dx``.
    """
    return float(np.sum(np.asarray(q, dtype=np.float64)) * dx)


def puff_center(q: ArrayLike, length: float) -> float:
    """Periodic-aware centroid (circular mean) of the intensity field.

    Computes the intensity-weighted mean position treating the domain as a
    circle, so a structure straddling the periodic seam is located correctly.
    Returns the centroid in model units in ``[0, length)``.

    Parameters
    ----------
    q : array_like
        Non-negative intensity field, shape ``(N,)``.
    length : float
        Domain length ``L``.

    Returns
    -------
    float
        Centroid position in ``[0, length)``; ``nan`` if the field is all zero.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    n = q_arr.size
    total = q_arr.sum()
    if total <= 0.0:
        return float("nan")
    theta = 2.0 * np.pi * np.arange(n) / n
    s = float(np.sum(q_arr * np.sin(theta)))
    c = float(np.sum(q_arr * np.cos(theta)))
    angle = np.arctan2(s, c) % (2.0 * np.pi)
    return float(angle * length / (2.0 * np.pi))


def leading_edge(
    q: ArrayLike, x: ArrayLike, threshold: float = 0.5
) -> float:
    """Position of the right-most point where ``q`` exceeds ``threshold``.

    Parameters
    ----------
    q : array_like
        Intensity field, shape ``(N,)``.
    x : array_like
        Grid coordinates, shape ``(N,)``.
    threshold : float, optional
        Intensity level defining the turbulent–laminar interface. Default ``0.5``.

    Returns
    -------
    float
        ``x`` at the right-most super-threshold point; ``nan`` if none.

    Notes
    -----
    Intended for a single localised structure kept away from the periodic
    boundary; it does not attempt to disambiguate multiple wrapped structures.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    above = np.flatnonzero(q_arr > threshold)
    if above.size == 0:
        return float("nan")
    return float(x_arr[above[-1]])


def trailing_edge(
    q: ArrayLike, x: ArrayLike, threshold: float = 0.5
) -> float:
    """Position of the left-most point where ``q`` exceeds ``threshold``.

    See :func:`leading_edge`; this returns the left-most super-threshold node.

    Parameters
    ----------
    q : array_like
        Intensity field, shape ``(N,)``.
    x : array_like
        Grid coordinates, shape ``(N,)``.
    threshold : float, optional
        Intensity level defining the interface. Default ``0.5``.

    Returns
    -------
    float
        ``x`` at the left-most super-threshold point; ``nan`` if none.
    """
    q_arr = np.asarray(q, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    above = np.flatnonzero(q_arr > threshold)
    if above.size == 0:
        return float("nan")
    return float(x_arr[above[0]])


def track_fronts(
    q_snapshots: ArrayLike, x: ArrayLike, threshold: float = 0.5
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Track the leading/trailing edges and width across a stack of snapshots.

    Parameters
    ----------
    q_snapshots : array_like
        Field snapshots, shape ``(M, N)`` (row ``k`` is ``q`` at time ``k``).
    x : array_like
        Grid coordinates, shape ``(N,)``.
    threshold : float, optional
        Intensity level defining the turbulent–laminar interface. Default ``0.5``.

    Returns
    -------
    leading, trailing, width : numpy.ndarray
        Per-snapshot right-edge, left-edge, and ``leading - trailing`` width,
        each shape ``(M,)``; entries are ``nan`` where no super-threshold point
        exists.

    Notes
    -----
    Like :func:`leading_edge`/:func:`trailing_edge`, this assumes a single
    contiguous structure kept away from the periodic seam over the measurement
    window (use a long enough domain so an expanding slug does not wrap).
    """
    q_arr = np.asarray(q_snapshots, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    m = q_arr.shape[0]
    leading = np.empty(m)
    trailing = np.empty(m)
    for k in range(m):
        leading[k] = leading_edge(q_arr[k], x_arr, threshold)
        trailing[k] = trailing_edge(q_arr[k], x_arr, threshold)
    return leading, trailing, leading - trailing


def front_speed(times: ArrayLike, positions: ArrayLike) -> float:
    """Propagation speed as the least-squares slope of ``position(t)``.

    Parameters
    ----------
    times : array_like
        Snapshot times, shape ``(M,)``.
    positions : array_like
        Front positions at those times, shape ``(M,)``. ``nan`` entries (e.g.
        before the structure forms) are ignored.

    Returns
    -------
    float
        ``d(position)/dt`` from a linear fit; ``nan`` if fewer than two finite
        samples are available.
    """
    t_arr = np.asarray(times, dtype=np.float64)
    p_arr = np.asarray(positions, dtype=np.float64)
    mask = np.isfinite(t_arr) & np.isfinite(p_arr)
    if mask.sum() < 2:
        return float("nan")
    slope = np.polyfit(t_arr[mask], p_arr[mask], 1)[0]
    return float(slope)


def front_kinematics(
    q_snapshots: ArrayLike,
    x: ArrayLike,
    t: ArrayLike,
    length: float,
    threshold: float = 0.5,
    t_min: float = 40.0,
) -> dict:
    """Wrap-robust drift / expansion / front speeds of a localised structure.

    Raw leading/trailing edge tracking breaks when the structure crosses the
    periodic seam (the position trajectory jumps by ``+/- length`` and a linear
    fit returns nonsense). This routine avoids that by working *relative to the
    intensity centroid*:

    1. the centroid trajectory ``c(t)`` is found with the wrap-safe
       :func:`puff_center`, unwrapped (period ``length``), and fitted for the
       **drift** speed;
    2. each snapshot is rolled to put the centroid at mid-domain, where the
       super-threshold left/right **offsets** from the centre are wrap-free; the
       offset slopes give the half-width growth.

    The leading/trailing front speeds are ``drift + offset_speed`` and the
    **expansion rate** is half the width-growth rate (``~0`` for an equilibrium
    puff, ``>0`` for an expanding slug).

    Parameters
    ----------
    q_snapshots : array_like
        Field snapshots, shape ``(M, N)``.
    x : array_like
        Grid coordinates, shape ``(N,)``.
    t : array_like
        Snapshot times, shape ``(M,)``.
    length : float
        Domain length ``L``.
    threshold : float, optional
        Turbulent–laminar interface level. Default ``0.5``.
    t_min : float, optional
        Ignore snapshots before this time (structure still forming).
        Default ``40.0``.

    Returns
    -------
    dict
        Keys: ``decayed`` (bool), ``drift``, ``expansion_rate``, ``c_leading``,
        ``c_trailing``, ``final_width``, ``n_window`` (number of fitted
        snapshots). Speeds are ``nan``/zero and ``decayed`` is ``True`` if no
        usable window exists.
    """
    q_arr = np.asarray(q_snapshots, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    t_arr = np.asarray(t, dtype=np.float64)
    m, n = q_arr.shape
    dx = length / n
    half = 0.5 * length

    centers = np.full(m, np.nan)
    right = np.full(m, np.nan)
    left = np.full(m, np.nan)
    for k in range(m):
        c = puff_center(q_arr[k], length)
        if not np.isfinite(c):
            continue
        centers[k] = c
        shift = int(round((half - c) / dx))
        qc = np.roll(q_arr[k], shift)
        above = np.flatnonzero(qc > threshold)
        if above.size == 0:
            continue
        right[k] = x_arr[above[-1]] - half
        left[k] = x_arr[above[0]] - half

    width = right - left
    mask = (t_arr > t_min) & np.isfinite(width) & (width < 0.75 * length)
    decayed = bool(mask.sum() < 2)
    if decayed:
        return {
            "decayed": True,
            "drift": 0.0,
            "expansion_rate": 0.0,
            "c_leading": 0.0,
            "c_trailing": 0.0,
            "final_width": 0.0,
            "n_window": int(mask.sum()),
        }

    tw = t_arr[mask]
    centroid_unwrapped = np.unwrap(centers[mask], period=length)
    drift = float(np.polyfit(tw, centroid_unwrapped, 1)[0])
    right_speed = float(np.polyfit(tw, right[mask], 1)[0])
    left_speed = float(np.polyfit(tw, left[mask], 1)[0])
    return {
        "decayed": False,
        "drift": drift,
        "expansion_rate": 0.5 * (right_speed - left_speed),
        "c_leading": drift + right_speed,
        "c_trailing": drift + left_speed,
        "final_width": float(width[mask][-1]),
        "n_window": int(mask.sum()),
    }
