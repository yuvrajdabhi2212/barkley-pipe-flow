"""Survival statistics of the discrete model (Phase 2, Milestones 2-4).

Turns ensembles of discrete-model trajectories (:mod:`barkley_pipe.discrete`)
into the quantitative statistics that make pipe transition a *phase transition*:

* **Lifetimes.** :func:`detect_decay` / :func:`detect_splitting` time the
  relaminarization and splitting events; :func:`measure_lifetimes` collects them
  over an ensemble of stored trajectories.
* **Survival function.** The survival probability is memoryless,
  :math:`P(n) \\sim \\exp(-n/\\tau(R))`. :func:`fit_tau` gives the maximum-
  likelihood :math:`\\tau` (for an exponential, the sample mean) and
  :func:`survival_function` the empirical curve (Barkley Fig. 12).
* **tau(R).** :func:`tau_of_R` sweeps ``R`` for decay and splitting using a fast
  vectorised ensemble and a **right-censored** exponential MLE
  (:math:`\\tau = \\text{total exposure}/\\text{events}`), since near onset many
  puffs outlive the run. The decay and splitting curves cross near
  ``R_x ~ 2040`` (Barkley Fig. 5a).
* **Turbulence fraction.** :func:`turbulence_fraction` is the order parameter
  ``F_t``; its onset locates ``R_c ~ 2046`` and is expected to scale as
  :math:`F_t \\sim (R - R_c)^{\\beta_{DP}}` with the (1+1)D directed-percolation
  exponent :math:`\\beta_{DP} \\approx 0.2765`.

On free Colab the published ensembles (~4000 realizations on ``1e5``-site
lattices) are infeasible, so the helpers use *reduced* ensembles; the exact
``R_x`` / ``R_c`` are reproduction targets within tolerance, not high-precision
constants.

References
----------
Barkley, D. (2011). Simplifying the complexity of pipe flow.
*Phys. Rev. E* **84**, 016309. arXiv:1101.4125.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from .discrete import DiscreteParams, step

__all__ = [
    "detect_decay",
    "detect_splitting",
    "turbulence_fraction",
    "measure_lifetimes",
    "sample_lifetimes",
    "fit_tau",
    "survival_function",
    "tau_of_R",
]


# --------------------------------------------------------------------------- #
# Event detection on a single stored trajectory
# --------------------------------------------------------------------------- #
def detect_decay(q_spacetime: ArrayLike, threshold: float = 0.5) -> Optional[int]:
    """Return the row index at which a puff relaminarizes, or ``None``.

    The laminar state ``q = 0`` is absorbing, so decay is the first snapshot with
    no turbulent site (``max_i q_i < threshold``).

    Parameters
    ----------
    q_spacetime : array_like
        Space-time array of ``q``, shape ``(n_time, n_sites)``.
    threshold : float, optional
        ``q`` level above which a site counts as turbulent. Default ``0.5``.

    Returns
    -------
    int or None
        Row index of decay, or ``None`` if the puff survives the run.
    """
    q = np.asarray(q_spacetime, dtype=np.float64)
    laminar = q.max(axis=1) < threshold
    idx = np.flatnonzero(laminar)
    return int(idx[0]) if idx.size else None


def _count_big_gaps(mask: NDArray[np.bool_], gap_sites: int) -> int:
    """Number of laminar gaps of length >= ``gap_sites`` on a periodic ring.

    This equals the number of distinct turbulent clusters separated by laminar
    stretches wider than ``gap_sites`` — small internal sub-threshold dips within
    one chaotic puff (gaps shorter than ``gap_sites``) are ignored — so a value
    ``>= 2`` signals a genuine split into separate puffs.
    """
    if not mask.any() or mask.all():
        return 0
    off = int(np.flatnonzero(mask)[0])  # rotate to start at a turbulent site (no wrap)
    laminar = (~np.roll(mask, -off)).astype(np.int8)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], laminar, [0]))))
    run_lengths = edges[1::2] - edges[::2]
    return int((run_lengths >= gap_sites).sum())


def _count_big_gaps_vec(mask: NDArray[np.bool_], gap_sites: int) -> NDArray[np.int64]:
    """Vectorised :func:`_count_big_gaps` over a stacked ensemble ``(m, n)``."""
    m, n = mask.shape
    lam = (~mask).astype(np.int32)
    ext = np.concatenate([lam, lam[:, :gap_sites]], axis=1)  # wrap for periodicity
    csum = np.concatenate([np.zeros((m, 1), np.int32), np.cumsum(ext, axis=1)], axis=1)
    winsum = csum[:, gap_sites : n + gap_sites] - csum[:, 0:n]  # sum over each L-window
    all_laminar = winsum >= gap_sites  # (m, n): a full laminar L-window starts at i
    # number of contiguous True-blocks per row (periodic) == number of big gaps
    return (all_laminar & ~np.roll(all_laminar, 1, axis=-1)).sum(axis=1)


def detect_splitting(
    q_spacetime: ArrayLike, gap_sites: int = 80, threshold: float = 0.5
) -> Optional[int]:
    """Return the row index at which a puff first splits in two, or ``None``.

    A split is the first snapshot with at least two turbulent clusters separated
    by a laminar gap wider than ``gap_sites`` (a genuine second puff, not
    transient internal fragmentation of one).

    Parameters
    ----------
    q_spacetime : array_like
        Space-time array of ``q``, shape ``(n_time, n_sites)``.
    gap_sites : int, optional
        Minimum laminar gap (sites) separating two clusters. Default ``80``.
    threshold : float, optional
        Turbulent–laminar threshold. Default ``0.5``.

    Returns
    -------
    int or None
        Row index of the split, or ``None`` if none occurs.
    """
    q = np.asarray(q_spacetime, dtype=np.float64)
    for t in range(q.shape[0]):
        if _count_big_gaps(q[t] > threshold, gap_sites) >= 2:
            return t
    return None


def turbulence_fraction(
    q_spacetime: ArrayLike, threshold: float = 0.5, tail: float = 0.5
) -> float:
    """Order parameter ``F_t``: mean turbulent fraction over the late-time tail.

    Parameters
    ----------
    q_spacetime : array_like
        Space-time array of ``q``, shape ``(n_time, n_sites)``.
    threshold : float, optional
        Turbulent–laminar threshold. Default ``0.5``.
    tail : float, optional
        Fraction of the (final) run averaged over, to skip transients.
        Default ``0.5`` (the second half).

    Returns
    -------
    float
        Turbulent fraction in ``[0, 1]``.
    """
    q = np.asarray(q_spacetime, dtype=np.float64)
    start = int((1.0 - tail) * q.shape[0])
    return float((q[start:] > threshold).mean())


# --------------------------------------------------------------------------- #
# Lifetime statistics
# --------------------------------------------------------------------------- #
def measure_lifetimes(
    ensemble: Iterable[ArrayLike], event: str = "decay", **detect_kwargs
) -> NDArray[np.float64]:
    """Collect decay (or splitting) times across stored trajectories.

    Parameters
    ----------
    ensemble : iterable of array_like
        Sequence of space-time arrays, one per realization.
    event : {'decay', 'splitting'}, optional
        Which event to time. Default ``'decay'``.
    **detect_kwargs
        Passed to :func:`detect_decay` / :func:`detect_splitting`.

    Returns
    -------
    numpy.ndarray
        Observed lifetimes (censored runs, where the event did not occur, are
        excluded), shape ``(n_events,)``.
    """
    detect = detect_decay if event == "decay" else detect_splitting
    times = [detect(traj, **detect_kwargs) for traj in ensemble]
    return np.array([t for t in times if t is not None], dtype=np.float64)


def fit_tau(lifetimes: ArrayLike) -> float:
    """Maximum-likelihood ``tau`` of an exponential survival law ``exp(-n/tau)``.

    For an exponential the MLE is the sample mean; computed via
    ``scipy.stats.expon.fit(lifetimes, floc=0)`` (location fixed at 0). This is
    the estimator for a fully observed (uncensored) sample; for censored
    ensembles use :func:`tau_of_R`, which applies the right-censored MLE.

    Parameters
    ----------
    lifetimes : array_like
        Observed lifetimes (all events observed).

    Returns
    -------
    float
        Estimated ``tau`` (``nan`` if no data).
    """
    data = np.asarray(lifetimes, dtype=np.float64)
    if data.size == 0:
        return float("nan")
    _, scale = stats.expon.fit(data, floc=0.0)
    return float(scale)


def survival_function(
    lifetimes: ArrayLike, n_grid: Optional[ArrayLike] = None
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Empirical survival function ``P(n) = Pr(lifetime > n)``.

    Parameters
    ----------
    lifetimes : array_like
        Observed lifetimes.
    n_grid : array_like or None, optional
        Times at which to evaluate ``P``. Defaults to the sorted unique
        lifetimes (starting at 0).

    Returns
    -------
    tuple of numpy.ndarray
        ``(n, P(n))`` for the log-linear survival plot (Barkley Fig. 12).
    """
    data = np.sort(np.asarray(lifetimes, dtype=np.float64))
    if data.size == 0:
        return np.array([0.0]), np.array([1.0])
    if n_grid is None:
        n_grid = np.concatenate(([0.0], np.unique(data)))
    n_grid = np.asarray(n_grid, dtype=np.float64)
    surv = np.array([(data > n).mean() for n in n_grid])
    return n_grid, surv


# --------------------------------------------------------------------------- #
# tau(R) via a fast vectorised ensemble + right-censored MLE
# --------------------------------------------------------------------------- #
def _ensemble_event_stats(
    r: float,
    event: str,
    n: int,
    m: int,
    max_steps: int,
    width: int,
    seed: int,
    threshold: float = 0.5,
    split_gap: int = 80,
    check_every: int = 1,
) -> tuple[NDArray[np.float64], float, int]:
    """Run ``m`` vectorised realizations; return event times, exposure, #events.

    Implements competing risks: a realization is *at risk* until it experiences
    the target ``event``, decays first (censored, for splitting), or the run ends
    (censored). Returns the observed event times, the total time-at-risk
    (exposure) summed over all realizations, and the number of events.
    """
    rng = np.random.default_rng(seed)
    q = np.zeros((m, n), dtype=np.float64)
    u = np.ones((m, n), dtype=np.float64)
    s = (n - width) // 2
    q[:, s : s + width] = 1.0 * (1.0 + 0.1 * rng.standard_normal((m, width)))
    params = DiscreteParams(R=r)

    risk_end = np.full(m, max_steps, dtype=np.float64)  # censored at run end by default
    had_event = np.zeros(m, dtype=bool)
    at_risk = np.ones(m, dtype=bool)

    for t in range(1, max_steps + 1):
        q, u = step(q, u, params)
        laminar = q.max(axis=1) < threshold
        if event == "decay":
            newly = at_risk & laminar
            risk_end[newly] = t
            had_event[newly] = True
            at_risk &= ~newly
        else:  # splitting (competing risk: relaminarization censors at decay time)
            newly_dead = at_risk & laminar
            risk_end[newly_dead] = t
            at_risk &= ~newly_dead
            if t % check_every == 0:
                split = _count_big_gaps_vec(q > threshold, split_gap) >= 2
                newly = at_risk & split
                risk_end[newly] = t
                had_event[newly] = True
                at_risk &= ~newly

        if not at_risk.any():
            break

    event_times = risk_end[had_event]
    exposure = float(risk_end.sum())
    return event_times, exposure, int(had_event.sum())


def sample_lifetimes(
    r: float,
    event: str = "decay",
    n_realizations: int = 300,
    *,
    n_sites: int = 240,
    max_steps: int = 20000,
    width: int = 40,
    seed: int = 0,
    check_every: int = 1,
) -> NDArray[np.float64]:
    """Sample observed event lifetimes from a fast vectorised ensemble.

    Convenience wrapper that runs ``n_realizations`` puffs at a single ``R`` and
    returns the times of the observed events (decay or splitting). Censored runs
    (the event did not occur within ``max_steps``) are excluded — choose
    ``max_steps`` large enough that censoring is negligible for survival plots.

    Parameters
    ----------
    r : float
        Discrete Reynolds proxy.
    event : {'decay', 'splitting'}, optional
        Which event to time. Default ``'decay'``.
    n_realizations : int, optional
        Ensemble size. Default ``300``.
    n_sites, max_steps, width, seed, check_every : optional
        Lattice size, run length, seed width, RNG seed, and split-check stride.

    Returns
    -------
    numpy.ndarray
        Observed lifetimes, shape ``(n_events,)``.
    """
    times, _, _ = _ensemble_event_stats(
        float(r), event, n_sites, n_realizations, max_steps, width, seed,
        check_every=check_every,
    )
    return times


def tau_of_R(
    r_values: ArrayLike,
    n_realizations: int = 200,
    event: str = "decay",
    *,
    n_sites: int = 240,
    max_steps: int = 30000,
    width: int = 40,
    seed: int = 0,
    check_every: int = 1,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Characteristic lifetime ``tau(R)`` for decay or splitting (Barkley Fig. 5a).

    For each ``R`` a vectorised ensemble of ``n_realizations`` puffs is evolved
    and ``tau`` is the right-censored exponential MLE (total exposure / number of
    events). The decay and splitting curves cross near ``R_x ~ 2040``.

    Parameters
    ----------
    r_values : array_like
        Discrete Reynolds proxies ``R`` to sweep.
    n_realizations : int, optional
        Ensemble size per ``R``. Default ``200`` (reduced for Colab).
    event : {'decay', 'splitting'}, optional
        Which lifetime to characterize. Default ``'decay'``.
    n_sites, max_steps, width, seed : optional
        Lattice size, run length, seed width, and RNG seed. ``n_sites`` should be
        comfortably larger than ``split_gap`` (80) for splitting.

    Returns
    -------
    tuple of numpy.ndarray
        ``(R, tau(R))``. ``tau`` is ``nan`` for an ``R`` with no observed events.
    """
    r_arr = np.asarray(r_values, dtype=np.float64)
    taus = np.empty(r_arr.size, dtype=np.float64)
    for i, r in enumerate(r_arr):
        _, exposure, n_events = _ensemble_event_stats(
            float(r), event, n_sites, n_realizations, max_steps, width, seed + i,
            check_every=check_every,
        )
        taus[i] = exposure / n_events if n_events > 0 else float("nan")
    return r_arr, taus
