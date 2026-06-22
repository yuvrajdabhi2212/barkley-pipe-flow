"""Tests for :mod:`barkley_pipe.discrete` (Phase 2, Milestone 1).

Cover the tent map (branch values, continuity, laminar fixed point), the
threshold, the lattice step (laminar is a fixed point), and the regime control
by ``R`` (a puff decays well below onset; a slug fills the domain well above it).
Stochastic runs use fixed RNG seeds so they are deterministic in CI.
"""

from __future__ import annotations

import numpy as np
import pytest

from barkley_pipe.discrete import (
    BETA,
    D,
    GAMMA,
    K,
    DiscreteParams,
    initial_puff,
    simulate_discrete,
    step,
    tent_map,
    tent_map_iterated,
    threshold_alpha,
)


def _frac(q, thr=0.5):
    return float((q > thr).mean())


# --------------------------------------------------------------------------- #
# Parameters
# --------------------------------------------------------------------------- #
def test_constants_and_params() -> None:
    assert BETA > 0  # escape window -> spontaneous decay
    assert D <= 0.5  # diffusion stability bound
    assert K == 2
    p = DiscreteParams(R=2200.0)
    assert p.R == 2200.0
    assert (p.gamma, p.k) == (GAMMA, K)


# --------------------------------------------------------------------------- #
# Threshold and tent map
# --------------------------------------------------------------------------- #
def test_threshold_alpha_formula() -> None:
    assert threshold_alpha(0.0, 2000.0) == pytest.approx(1.0)
    assert threshold_alpha(1.0, 2000.0) == pytest.approx(0.2)
    np.testing.assert_allclose(
        threshold_alpha(np.array([0.0, 0.5]), 1000.0),
        np.array([2.0, 2000.0 * 0.6 / 1000.0]),
    )


def test_tent_map_laminar_fixed_point() -> None:
    assert float(tent_map(0.0, 0.6)) == 0.0


@pytest.mark.parametrize("alpha", [0.3, 0.6, 0.9])
def test_tent_map_is_continuous_at_breakpoints(alpha: float) -> None:
    q1 = alpha / (2.0 - GAMMA)
    q2 = (4.0 + BETA - alpha - GAMMA * q1) / (2.0 + BETA)
    eps = 1e-7
    for bp in (q1, 1.0, q2):
        left = float(tent_map(bp - eps, alpha))
        right = float(tent_map(bp + eps, alpha))
        assert abs(left - right) < 1e-5


def test_tent_map_branch_values() -> None:
    alpha = 0.6
    q1 = alpha / (2.0 - GAMMA)
    q2 = (4.0 + BETA - alpha - GAMMA * q1) / (2.0 + BETA)
    # laminar (contracting) branch
    q = 0.5 * q1
    assert tent_map(q, alpha) == pytest.approx(GAMMA * q)
    # expanding branch
    q = 0.5 * (q1 + 1.0)
    assert tent_map(q, alpha) == pytest.approx(2.0 * q - alpha)
    # folding branch
    q = 0.5 * (1.0 + q2)
    assert tent_map(q, alpha) == pytest.approx(4.0 + BETA - alpha - (2.0 + BETA) * q)
    # reinjection branch (constant)
    assert tent_map(q2 + 1.0, alpha) == pytest.approx(GAMMA * q1)


def test_tent_map_iterated_composition() -> None:
    q = np.array([0.2, 0.7, 1.2, 2.0])
    alpha = np.full(4, 0.6)
    np.testing.assert_allclose(tent_map_iterated(q, alpha, k=1), tent_map(q, alpha))
    np.testing.assert_allclose(
        tent_map_iterated(q, alpha, k=2), tent_map(tent_map(q, alpha), alpha)
    )
    np.testing.assert_allclose(tent_map_iterated(q, alpha, k=0), q)


# --------------------------------------------------------------------------- #
# Initial condition and step
# --------------------------------------------------------------------------- #
def test_initial_puff_structure_and_reproducibility() -> None:
    q, u = initial_puff(100, width=20, q_level=1.0, seed=7)
    assert q.shape == (100,) and u.shape == (100,)
    np.testing.assert_array_equal(u, 1.0)  # laminar background
    assert (q[:30] == 0).all() and (q[-30:] == 0).all()  # localized
    assert q.max() > 0.5
    q2, _ = initial_puff(100, width=20, q_level=1.0, seed=7)
    np.testing.assert_array_equal(q, q2)  # deterministic for a given seed


def test_laminar_is_fixed_point_of_the_lattice() -> None:
    n = 64
    q = np.zeros(n)
    u = np.ones(n)
    q1, u1 = step(q, u, DiscreteParams(R=2200.0))
    np.testing.assert_allclose(q1, 0.0, atol=1e-12)
    np.testing.assert_allclose(u1, 1.0, atol=1e-12)


def test_simulate_shapes_and_finiteness() -> None:
    q0, u0 = initial_puff(120, width=20, seed=0)
    qst = simulate_discrete(q0, u0, DiscreteParams(R=2200.0), n_steps=200, store_every=10)
    assert qst.shape == (21, 120)  # initial row + 200/10
    assert np.isfinite(qst).all()


# --------------------------------------------------------------------------- #
# Regime control by R
# --------------------------------------------------------------------------- #
def test_puff_decays_below_onset() -> None:
    """A puff well below the sustained-turbulence onset relaminarizes."""
    q0, u0 = initial_puff(400, width=40, q_level=1.0, seed=0)
    qst = simulate_discrete(q0, u0, DiscreteParams(R=1900.0), n_steps=3000, store_every=50)
    assert _frac(qst[-1]) < 0.05


def test_slug_fills_domain_above_onset() -> None:
    """Well above onset, turbulence spreads to fill the domain (slug)."""
    q0, u0 = initial_puff(300, width=30, q_level=1.0, seed=0)
    qst = simulate_discrete(q0, u0, DiscreteParams(R=3000.0), n_steps=1500, store_every=50)
    assert _frac(qst[-1]) > 0.8
