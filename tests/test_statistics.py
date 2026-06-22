"""Tests for :mod:`barkley_pipe.statistics` (Phase 2, Milestones 2-4)."""

from __future__ import annotations

import numpy as np
import pytest

from barkley_pipe.statistics import (
    detect_decay,
    detect_splitting,
    fit_tau,
    measure_lifetimes,
    survival_function,
    tau_of_R,
    turbulence_fraction,
)


def _single_patch(t, n, lo, hi):
    q = np.zeros((t, n))
    q[:, lo:hi] = 1.0
    return q


# --------------------------------------------------------------------------- #
# Event detection
# --------------------------------------------------------------------------- #
def test_detect_decay_returns_first_laminar_row() -> None:
    q = np.zeros((10, 50))
    q[:4, 20:30] = 1.0  # turbulent for the first 4 rows, then laminar
    assert detect_decay(q) == 4


def test_detect_decay_none_if_never_laminar() -> None:
    assert detect_decay(_single_patch(6, 50, 20, 30)) is None


def test_detect_splitting_finds_two_separated_patches() -> None:
    n, t = 300, 10
    q = np.zeros((t, n))
    q[:5, 140:160] = 1.0  # one puff
    q[5:, 50:70] = 1.0  # split into two well-separated puffs
    q[5:, 200:220] = 1.0
    assert detect_splitting(q, gap_sites=80) == 5


def test_detect_splitting_none_for_single_patch() -> None:
    assert detect_splitting(_single_patch(8, 300, 140, 160), gap_sites=80) is None


def test_turbulence_fraction() -> None:
    q = np.ones((10, 100))
    assert turbulence_fraction(q) == pytest.approx(1.0)
    half = np.zeros((10, 100))
    half[:, :50] = 1.0
    assert turbulence_fraction(half) == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# Lifetime statistics
# --------------------------------------------------------------------------- #
def test_fit_tau_is_sample_mean() -> None:
    data = [10.0, 20.0, 30.0, 40.0]
    assert fit_tau(data) == pytest.approx(25.0)
    assert np.isnan(fit_tau([]))


def test_survival_function_is_monotone_from_one() -> None:
    n, p = survival_function([5.0, 10.0, 15.0, 20.0])
    assert p[0] == pytest.approx(1.0)  # all lifetimes exceed 0
    assert np.all(np.diff(p) <= 1e-12)  # non-increasing
    assert p[-1] == pytest.approx(0.0)


def test_measure_lifetimes_excludes_censored() -> None:
    decaying = np.zeros((10, 50))
    decaying[:3, 20:30] = 1.0  # decays at row 3
    surviving = _single_patch(10, 50, 20, 30)  # never decays
    lifetimes = measure_lifetimes([decaying, surviving], event="decay")
    np.testing.assert_array_equal(lifetimes, [3.0])


# --------------------------------------------------------------------------- #
# tau(R): lifetime grows with R (small ensemble for speed)
# --------------------------------------------------------------------------- #
def test_tau_of_R_increases_with_R() -> None:
    r, tau = tau_of_R([1850.0, 1950.0], n_realizations=40, event="decay",
                      n_sites=120, max_steps=5000, width=30)
    assert np.all(np.isfinite(tau))
    assert tau[1] > tau[0]  # puffs live longer closer to onset
