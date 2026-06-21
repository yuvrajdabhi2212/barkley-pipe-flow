"""Survival statistics of the discrete model (Phase 2 — DOCUMENTED STUBS).

This module will turn ensembles of discrete-model trajectories
(:mod:`barkley_pipe.discrete`) into the quantitative statistics that make pipe
transition a *phase transition*:

* **Puff lifetimes.** Detect decay (turbulent energy drops below threshold) and
  splitting (a second turbulent patch appears, separated by a laminar gap) and
  record the time of each event over many randomized initial puffs.
* **Survival function.** The survival probability is memoryless,
  :math:`P(n) \\sim \\exp(-n/\\tau(R))`. The characteristic time ``tau(R)`` is
  estimated by maximum likelihood — for an exponential the MLE is simply the
  sample mean, obtained robustly via ``scipy.stats.expon.fit(data, floc=0)``.
  Reproduces Barkley Fig. 12 and the decay/split lifetime crossing near
  ``R_x ~ 2040`` (his Fig. 5a inset).
* **Turbulence fraction.** The order parameter ``F_t(R)`` (fraction of the
  domain that is turbulent in steady state) locates the sustained-turbulence
  onset ``R_c ~ 2046.2`` and is expected to scale as
  :math:`F_t \\sim (R - R_c)^{\\beta_{DP}}` with the (1+1)D directed-percolation
  exponent :math:`\\beta_{DP} = 0.276486(8)` (Barkley's quoted ``0.28``).

Status
------
**Phase 2 — not yet implemented.** Every public function raises
``NotImplementedError`` and documents the intended behaviour. See ``ROADMAP.md``.
On free Colab the published ensembles (~4000 realizations on grids up to
``1.2e5`` sites for ``8e6`` steps) are infeasible, so the plan is *reduced*
ensembles with documented scaling rather than matching the published precision.

References
----------
Barkley, D. (2011). Simplifying the complexity of pipe flow.
*Phys. Rev. E* **84**, 016309. arXiv:1101.4125.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = [
    "detect_decay",
    "detect_splitting",
    "measure_lifetimes",
    "fit_tau",
    "survival_function",
    "turbulence_fraction",
    "tau_of_R",
]

_STUB = "Phase 2 not yet implemented — see ROADMAP.md"


def detect_decay(
    q_spacetime: ArrayLike, energy_threshold: float = 0.1
) -> int | None:
    """Return the time index at which a puff decays to laminar, or ``None``.

    Decay is the first time the total turbulent energy falls below
    ``energy_threshold`` and stays there.

    Parameters
    ----------
    q_spacetime : array_like
        Space-time array of ``q``, shape ``(n_time, n_sites)``.
    energy_threshold : float, optional
        Energy below which the flow is considered relaminarized. Default ``0.1``.

    Returns
    -------
    int or None
        Time index of decay, or ``None`` if the puff survives the run.

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


def detect_splitting(
    q_spacetime: ArrayLike, gap_sites: int = 80, threshold: float = 0.5
) -> int | None:
    """Return the time index at which a puff splits into two, or ``None``.

    Splitting is detected when two distinct super-threshold turbulent patches
    are separated by a laminar gap wider than ``gap_sites``.

    Parameters
    ----------
    q_spacetime : array_like
        Space-time array of ``q``, shape ``(n_time, n_sites)``.
    gap_sites : int, optional
        Minimum laminar gap (in sites) separating two patches. Default ``80``.
    threshold : float, optional
        Turbulent–laminar threshold. Default ``0.5``.

    Returns
    -------
    int or None
        Time index of the split, or ``None`` if no split occurs.

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


def measure_lifetimes(
    ensemble: ArrayLike, event: str = "decay"
) -> NDArray[np.float64]:
    """Collect decay (or splitting) times across an ensemble of trajectories.

    Parameters
    ----------
    ensemble : array_like
        Sequence of space-time arrays, one per realization.
    event : {'decay', 'splitting'}, optional
        Which event to time. Default ``'decay'``.

    Returns
    -------
    numpy.ndarray
        Observed lifetimes (censored runs excluded), shape ``(n_events,)``.

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


def fit_tau(lifetimes: ArrayLike) -> float:
    """Maximum-likelihood characteristic time of an exponential survival law.

    For ``P(n) ~ exp(-n/tau)`` the MLE is the sample mean; computed via
    ``scipy.stats.expon.fit(lifetimes, floc=0)`` (location fixed at 0).

    Parameters
    ----------
    lifetimes : array_like
        Observed lifetimes.

    Returns
    -------
    float
        Estimated ``tau``.

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


def survival_function(
    lifetimes: ArrayLike, n_grid: ArrayLike | None = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Empirical survival function ``P(n) = Pr(lifetime > n)``.

    Parameters
    ----------
    lifetimes : array_like
        Observed lifetimes.
    n_grid : array_like or None, optional
        Times at which to evaluate ``P``. Defaults to the sorted lifetimes.

    Returns
    -------
    tuple of numpy.ndarray
        ``(n, P(n))`` for plotting Barkley Fig. 12 (log-linear survival curves).

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


def turbulence_fraction(q_spacetime: ArrayLike, threshold: float = 0.5) -> float:
    """Steady-state turbulent fraction ``F_t`` (order parameter).

    Fraction of lattice sites that are turbulent (``q > threshold``), averaged
    over the late-time portion of the run.

    Parameters
    ----------
    q_spacetime : array_like
        Space-time array of ``q``, shape ``(n_time, n_sites)``.
    threshold : float, optional
        Turbulent–laminar threshold. Default ``0.5``.

    Returns
    -------
    float
        Turbulent fraction in ``[0, 1]``.

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)


def tau_of_R(
    r_values: ArrayLike, n_realizations: int = 200, event: str = "decay"
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sweep ``R`` and return ``tau(R)`` for decay or splitting (Fig. 5a).

    Parameters
    ----------
    r_values : array_like
        Discrete Reynolds proxies ``R`` to sweep.
    n_realizations : int, optional
        Ensemble size per ``R``. Default ``200`` (reduced for Colab).
    event : {'decay', 'splitting'}, optional
        Which lifetime to characterize. Default ``'decay'``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(R, tau(R))``; the decay and splitting curves cross near ``R_x ~ 2040``.

    Raises
    ------
    NotImplementedError
        Phase 2 stub.
    """
    raise NotImplementedError(_STUB)
