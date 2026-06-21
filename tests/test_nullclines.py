"""Tests for :mod:`barkley_pipe.nullclines`.

These check the analytic backbone of the model: the critical proxy ``r_c``, the
always-present and always-stable laminar fixed point, consistency between the
nullclines and the computed fixed points, and the regime classifier.
"""

from __future__ import annotations

import numpy as np
import pytest

from barkley_pipe.nullclines import (
    DELTA,
    EPS1,
    EPS2,
    critical_r,
    fixed_points,
    jacobian,
    local_dynamics,
    q_nullcline,
    regime,
    u_nullcline,
)


# --------------------------------------------------------------------------- #
# Critical proxy r_c
# --------------------------------------------------------------------------- #
def test_critical_r_default_value() -> None:
    """r_c = eps2 / (eps1 + eps2) ~ 0.833 for the default parameters."""
    assert critical_r() == pytest.approx(EPS2 / (EPS1 + EPS2))
    assert critical_r() == pytest.approx(0.2 / 0.24)
    assert critical_r() == pytest.approx(0.8333333, abs=1e-6)


def test_critical_r_depends_on_parameters() -> None:
    assert critical_r(eps1=0.1, eps2=0.3) == pytest.approx(0.75)


# --------------------------------------------------------------------------- #
# Laminar fixed point
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("r", [0.0, 0.3, 0.7, 0.8333, 1.0, 1.5])
def test_laminar_is_a_fixed_point_for_all_r(r: float) -> None:
    """(q, u) = (0, 1) is a root of the local dynamics for every r."""
    dq, du = local_dynamics(0.0, 1.0, r)
    assert dq == pytest.approx(0.0, abs=1e-14)
    assert du == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("r", [0.3, 0.7, 1.0, 1.5])
def test_laminar_point_present_in_fixed_points(r: float) -> None:
    pts = fixed_points(r)
    assert np.any(np.all(np.isclose(pts, [0.0, 1.0]), axis=1))


@pytest.mark.parametrize("r", [0.3, 0.7, 1.0, 1.5])
def test_laminar_point_is_linearly_stable(r: float) -> None:
    """Eigenvalues of J at (0, 1) are {-delta, -eps1}: always stable."""
    eigvals = np.linalg.eigvals(jacobian(0.0, 1.0, r))
    assert np.all(eigvals.real < 0.0)


# --------------------------------------------------------------------------- #
# Nullclines <-> fixed points consistency
# --------------------------------------------------------------------------- #
def test_q_nullcline_passes_through_one_minus_r() -> None:
    """The non-trivial q-nullcline has u = 1 - r at the core value q = 1."""
    for r in (0.5, 0.7, 1.0):
        assert q_nullcline(1.0, r) == pytest.approx(1.0 - r)


def test_q_nullcline_meets_laminar_axis_at_one_plus_delta() -> None:
    """The q-nullcline crosses the q = 0 axis at u = 1 + delta for every r."""
    for r in (0.0, 0.5, 0.7, 1.0, 1.5):
        assert q_nullcline(0.0, r) == pytest.approx(1.0 + DELTA)


def test_u_nullcline_endpoints() -> None:
    """u-nullcline equals 1 at q = 0 and decays towards 0 as q grows."""
    assert u_nullcline(0.0) == pytest.approx(1.0)
    assert u_nullcline(1e6) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("r", [0.5, 0.7, 0.9, 1.0, 1.2])
def test_computed_fixed_points_are_roots_of_the_dynamics(r: float) -> None:
    """Every returned fixed point satisfies dq = du = 0."""
    for q, u in fixed_points(r):
        dq, du = local_dynamics(q, u, r)
        assert dq == pytest.approx(0.0, abs=1e-9)
        assert du == pytest.approx(0.0, abs=1e-9)


def test_turbulent_fixed_point_sits_at_q_equals_one_at_r_critical() -> None:
    """At r = r_c the turbulent fixed point is exactly at q = 1."""
    pts = fixed_points(critical_r())
    q_values = pts[:, 0]
    assert np.any(np.isclose(q_values, 1.0, atol=1e-6))


# --------------------------------------------------------------------------- #
# Regime classification
# --------------------------------------------------------------------------- #
def test_regime_classifies_across_rc() -> None:
    rc = critical_r()
    assert regime(0.7) == "excitable"
    assert regime(1.0) == "bistable"
    assert regime(rc) == "critical"
    assert regime(rc - 0.05) == "excitable"
    assert regime(rc + 0.05) == "bistable"
