"""Tests for :mod:`barkley_pipe.continuous`.

These exercise the method-of-lines solver against qualitative facts the paper
guarantees: the laminar state is an exact, stable steady state; the excitable
regime (``r < r_c``) has a finite-amplitude threshold (small perturbations
decay, large ones ignite a puff); a triggered puff settles to a localised
equilibrium of constant size; and the integration stays finite.

Grids are kept small/short so the suite runs quickly in CI; the convergence to
the published profile is demonstrated separately in the notebooks.
"""

from __future__ import annotations

import numpy as np
import pytest

from barkley_pipe.continuous import (
    ContinuousParams,
    PeriodicGrid,
    continuous_rhs,
    laminar_state,
    puff_seed,
    simulate,
)
from barkley_pipe.diagnostics import front_kinematics, turbulent_energy
from barkley_pipe.nullclines import regime


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #
def test_grid_geometry() -> None:
    grid = PeriodicGrid(n=100, length=50.0)
    assert grid.dx == pytest.approx(0.5)
    assert grid.x[0] == 0.0
    assert grid.x[-1] == pytest.approx(50.0 - 0.5)
    assert grid.x.shape == (100,)


@pytest.mark.parametrize("kwargs", [{"n": 3, "length": 1.0}, {"n": 10, "length": 0.0}])
def test_grid_rejects_bad_parameters(kwargs) -> None:
    with pytest.raises(ValueError):
        PeriodicGrid(**kwargs)


# --------------------------------------------------------------------------- #
# Laminar state
# --------------------------------------------------------------------------- #
def test_laminar_is_exact_fixed_point_of_discretization() -> None:
    """The discretised RHS is exactly zero at the laminar state (q=0, u=1)."""
    grid = PeriodicGrid(n=256, length=80.0)
    rhs = continuous_rhs(0.0, laminar_state(grid), grid, ContinuousParams(r=0.7))
    assert np.max(np.abs(rhs)) == 0.0


@pytest.mark.parametrize("r", [0.7, 1.0])
def test_laminar_state_remains_laminar(r: float) -> None:
    """Started laminar, the solution stays laminar over time."""
    grid = PeriodicGrid(n=128, length=64.0)
    res = simulate(grid, ContinuousParams(r=r), laminar_state(grid), (0.0, 40.0),
                   n_snapshots=2)
    assert res.success
    np.testing.assert_allclose(res.q[-1], 0.0, atol=1e-10)
    np.testing.assert_allclose(res.u[-1], 1.0, atol=1e-10)


# --------------------------------------------------------------------------- #
# Excitability threshold (r < r_c)
# --------------------------------------------------------------------------- #
def test_subthreshold_perturbation_decays() -> None:
    """A small perturbation at r < r_c decays back to laminar (excitability)."""
    grid = PeriodicGrid(n=256, length=80.0)
    y0 = puff_seed(grid, width=3.0, q_amplitude=0.3, u_dip=0.3)
    e0 = turbulent_energy(y0[: grid.n], grid.dx)
    res = simulate(grid, ContinuousParams(r=0.7), y0, (0.0, 80.0), n_snapshots=2)
    e1 = turbulent_energy(res.q[-1], grid.dx)
    assert e1 < 1e-3 * e0


def test_suprathreshold_perturbation_ignites_a_puff() -> None:
    """A large enough perturbation at r < r_c ignites a sustained puff."""
    grid = PeriodicGrid(n=256, length=80.0)
    y0 = puff_seed(grid, width=5.0, q_amplitude=1.0, u_dip=0.6)
    res = simulate(grid, ContinuousParams(r=0.7), y0, (0.0, 80.0), n_snapshots=2)
    assert turbulent_energy(res.q[-1], grid.dx) > 1.0


# --------------------------------------------------------------------------- #
# Equilibrium puff
# --------------------------------------------------------------------------- #
def test_equilibrium_puff_reaches_constant_size() -> None:
    """At r < r_c a triggered puff settles to constant turbulent energy."""
    grid = PeriodicGrid(n=250, length=100.0)
    y0 = puff_seed(grid, width=5.0, q_amplitude=1.0, u_dip=0.6)
    res = simulate(grid, ContinuousParams(r=0.7), y0, (0.0, 150.0), n_snapshots=60)
    energy = np.array([turbulent_energy(res.q[k], grid.dx) for k in range(len(res.t))])
    tail = energy[len(energy) * 2 // 3 :]
    # plateau: relative fluctuation of the energy in the last third is tiny
    assert tail.std() / tail.mean() < 0.02


def test_equilibrium_puff_is_localized() -> None:
    """The puff is localised: turbulent in its core, laminar elsewhere."""
    grid = PeriodicGrid(n=250, length=100.0)
    y0 = puff_seed(grid, width=5.0, q_amplitude=1.0, u_dip=0.6)
    res = simulate(grid, ContinuousParams(r=0.7), y0, (0.0, 120.0), n_snapshots=2)
    q_final = res.q[-1]
    assert q_final.max() > 0.5  # has a turbulent core
    assert q_final.min() < 0.05  # and laminar regions


# --------------------------------------------------------------------------- #
# Robustness
# --------------------------------------------------------------------------- #
def test_solution_stays_finite_and_bounded() -> None:
    grid = PeriodicGrid(n=256, length=80.0)
    y0 = puff_seed(grid, width=5.0, q_amplitude=1.0, u_dip=0.6)
    res = simulate(grid, ContinuousParams(r=1.0), y0, (0.0, 60.0), n_snapshots=10)
    assert np.isfinite(res.q).all() and np.isfinite(res.u).all()
    assert res.q.max() < 5.0  # bounded by the cubic reaction term
    assert res.q.min() > -0.05  # upwind keeps q essentially non-negative


@pytest.mark.parametrize("r,should_expand", [(0.7, False), (1.0, True)])
def test_regime_classification_matches_front_dynamics(r: float, should_expand: bool) -> None:
    """regime(r) agrees with measured dynamics: puffs stay put, slugs expand."""
    grid = PeriodicGrid(n=1000, length=400.0)
    y0 = puff_seed(grid, width=5.0, q_amplitude=1.0, u_dip=0.6)
    res = simulate(grid, ContinuousParams(r=r), y0, (0.0, 140.0), n_snapshots=71)
    kin = front_kinematics(res.q, res.x, res.t, grid.length)
    if should_expand:
        assert regime(r) == "bistable"
        assert kin["expansion_rate"] > 0.3
        assert kin["c_leading"] > kin["c_trailing"]
    else:
        assert regime(r) == "excitable"
        assert abs(kin["expansion_rate"]) < 0.02


@pytest.mark.parametrize("scheme", ["upwind", "central"])
def test_both_advection_schemes_run(scheme: str) -> None:
    grid = PeriodicGrid(n=200, length=80.0)
    y0 = puff_seed(grid, width=5.0, q_amplitude=1.0, u_dip=0.6)
    res = simulate(grid, ContinuousParams(r=0.7, advection=scheme), y0,
                   (0.0, 40.0), n_snapshots=2)
    assert res.success and np.isfinite(res.q).all()


def test_result_shapes() -> None:
    grid = PeriodicGrid(n=64, length=32.0)
    res = simulate(grid, ContinuousParams(r=0.7), laminar_state(grid),
                   (0.0, 5.0), n_snapshots=11)
    assert res.t.shape == (11,)
    assert res.x.shape == (64,)
    assert res.q.shape == (11, 64)
    assert res.u.shape == (11, 64)
