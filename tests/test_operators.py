"""Tests for :mod:`barkley_pipe.operators`.

The finite-difference stencils are verified by (a) their formal order of
accuracy under grid refinement on a smooth periodic field, (b) exact periodic
wrap-around at the domain ends, and (c) basic sanity (constant fields, input
validation, upwind direction switching).
"""

from __future__ import annotations

import numpy as np
import pytest

from barkley_pipe.operators import (
    first_derivative_central,
    first_derivative_upwind,
    get_advection_operator,
    laplacian,
)

# Grid resolutions used for the convergence (order-of-accuracy) studies.
REFINEMENTS = [50, 100, 200, 400, 800]


def _periodic_grid(n: int) -> tuple[np.ndarray, float]:
    """Return a uniform periodic grid on ``[0, 2*pi)`` and its spacing."""
    x = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    dx = 2.0 * np.pi / n
    return x, dx


def _measured_order(errors: list[float], spacings: list[float]) -> float:
    """Slope of log(error) vs log(dx): the empirical order of accuracy."""
    return float(np.polyfit(np.log(spacings), np.log(errors), 1)[0])


# --------------------------------------------------------------------------- #
# Order of accuracy
# --------------------------------------------------------------------------- #
def test_central_first_derivative_is_second_order() -> None:
    """Centred first derivative of sin(x) converges at order ~2."""
    errors, spacings = [], []
    for n in REFINEMENTS:
        x, dx = _periodic_grid(n)
        approx = first_derivative_central(np.sin(x), dx)
        errors.append(float(np.max(np.abs(approx - np.cos(x)))))
        spacings.append(dx)
    assert _measured_order(errors, spacings) == pytest.approx(2.0, abs=0.1)


def test_upwind_first_derivative_is_first_order() -> None:
    """Upwind first derivative of sin(x) converges at order ~1."""
    errors, spacings = [], []
    for n in REFINEMENTS:
        x, dx = _periodic_grid(n)
        approx = first_derivative_upwind(np.sin(x), dx, speed=1.0)
        errors.append(float(np.max(np.abs(approx - np.cos(x)))))
        spacings.append(dx)
    assert _measured_order(errors, spacings) == pytest.approx(1.0, abs=0.1)


def test_laplacian_is_second_order() -> None:
    """Centred second derivative of sin(x) (== -sin(x)) converges at order ~2."""
    errors, spacings = [], []
    for n in REFINEMENTS:
        x, dx = _periodic_grid(n)
        approx = laplacian(np.sin(x), dx)
        errors.append(float(np.max(np.abs(approx - (-np.sin(x))))))
        spacings.append(dx)
    assert _measured_order(errors, spacings) == pytest.approx(2.0, abs=0.1)


# --------------------------------------------------------------------------- #
# Periodic wrap-around
# --------------------------------------------------------------------------- #
def test_central_periodic_wraparound() -> None:
    """Endpoints of the centred stencil use the periodic neighbours."""
    rng = np.random.default_rng(0)
    f = rng.standard_normal(16)
    dx = 0.3
    approx = first_derivative_central(f, dx)
    assert approx[0] == pytest.approx((f[1] - f[-1]) / (2.0 * dx))
    assert approx[-1] == pytest.approx((f[0] - f[-2]) / (2.0 * dx))


def test_laplacian_periodic_wraparound() -> None:
    """Endpoints of the Laplacian stencil use the periodic neighbours."""
    rng = np.random.default_rng(1)
    f = rng.standard_normal(16)
    dx = 0.5
    approx = laplacian(f, dx)
    assert approx[0] == pytest.approx((f[-1] - 2.0 * f[0] + f[1]) / dx**2)
    assert approx[-1] == pytest.approx((f[-2] - 2.0 * f[-1] + f[0]) / dx**2)


def test_upwind_periodic_wraparound() -> None:
    """Upwind stencil wraps at the correct end for each flow direction."""
    rng = np.random.default_rng(2)
    f = rng.standard_normal(16)
    dx = 0.25
    back = first_derivative_upwind(f, dx, speed=1.0)
    fwd = first_derivative_upwind(f, dx, speed=-1.0)
    # backward difference wraps at index 0
    assert back[0] == pytest.approx((f[0] - f[-1]) / dx)
    # forward difference wraps at the last index
    assert fwd[-1] == pytest.approx((f[0] - f[-1]) / dx)


# --------------------------------------------------------------------------- #
# Upwind direction switching
# --------------------------------------------------------------------------- #
def test_upwind_direction_follows_speed_sign() -> None:
    """Positive speed -> backward difference; negative speed -> forward."""
    rng = np.random.default_rng(3)
    f = rng.standard_normal(32)
    dx = 0.1
    np.testing.assert_allclose(
        first_derivative_upwind(f, dx, speed=2.5), (f - np.roll(f, 1)) / dx
    )
    np.testing.assert_allclose(
        first_derivative_upwind(f, dx, speed=-2.5), (np.roll(f, -1) - f) / dx
    )


# --------------------------------------------------------------------------- #
# Sanity: constants, validation, registry
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("operator", [first_derivative_central, laplacian])
def test_constant_field_has_zero_derivative(operator) -> None:
    """Derivatives of a constant field vanish (catches wrap-sign errors)."""
    f = np.full(20, 3.14)
    np.testing.assert_allclose(operator(f, 0.2), 0.0, atol=1e-13)


def test_upwind_constant_field_has_zero_derivative() -> None:
    f = np.full(20, 3.14)
    np.testing.assert_allclose(
        first_derivative_upwind(f, 0.2, speed=1.0), 0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        first_derivative_upwind(f, 0.2, speed=-1.0), 0.0, atol=1e-13
    )


@pytest.mark.parametrize("dx", [0.0, -1.0])
def test_operators_reject_nonpositive_dx(dx) -> None:
    f = np.ones(10)
    with pytest.raises(ValueError):
        first_derivative_central(f, dx)
    with pytest.raises(ValueError):
        first_derivative_upwind(f, dx)
    with pytest.raises(ValueError):
        laplacian(f, dx)


def test_advection_registry_returns_matching_operators() -> None:
    rng = np.random.default_rng(4)
    f = rng.standard_normal(24)
    dx = 0.4
    upwind = get_advection_operator("upwind")
    central = get_advection_operator("central")
    np.testing.assert_allclose(
        upwind(f, dx, 1.0), first_derivative_upwind(f, dx, 1.0)
    )
    # central ignores the speed argument (symmetric stencil)
    np.testing.assert_allclose(
        central(f, dx, 1.0), first_derivative_central(f, dx)
    )
    np.testing.assert_allclose(
        central(f, dx, -1.0), first_derivative_central(f, dx)
    )


def test_advection_registry_rejects_unknown_scheme() -> None:
    with pytest.raises(ValueError):
        get_advection_operator("quick")
