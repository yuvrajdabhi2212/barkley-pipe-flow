"""Tests for :mod:`barkley_pipe.diagnostics`."""

from __future__ import annotations

import numpy as np
import pytest

from barkley_pipe.diagnostics import (
    front_kinematics,
    front_speed,
    leading_edge,
    puff_center,
    trailing_edge,
    turbulent_energy,
    turbulent_mass,
)


def _box_snapshots(x, t, centers, half_widths, length):
    """Build (M, N) box-shaped q snapshots (periodic-aware) for testing."""
    q = np.zeros((len(t), x.size))
    for k in range(len(t)):
        d = np.abs(x - centers[k])
        d = np.minimum(d, length - d)
        q[k] = np.where(d < half_widths[k], 1.0, 0.0)
    return q


def _circular_distance(a: float, b: float, length: float) -> float:
    d = abs(a - b) % length
    return min(d, length - d)


def test_energy_and_mass_on_constant_field() -> None:
    n, dx = 50, 0.2
    q = np.full(n, 2.0)
    assert turbulent_mass(q, dx) == pytest.approx(2.0 * n * dx)
    assert turbulent_energy(q, dx) == pytest.approx(4.0 * n * dx)


def test_puff_center_locates_centered_bump() -> None:
    n, length = 100, 100.0
    x = np.arange(n) * (length / n)
    q = np.exp(-0.5 * ((x - 30.0) / 4.0) ** 2)
    assert puff_center(q, length) == pytest.approx(30.0, abs=0.5)


def test_puff_center_handles_periodic_wrap() -> None:
    """A bump straddling the x=0 seam is still located at ~0 (circular mean)."""
    n, length = 100, 100.0
    x = np.arange(n) * (length / n)
    dist = np.minimum(np.abs(x - 0.0), length - np.abs(x - 0.0))
    q = np.exp(-0.5 * (dist / 4.0) ** 2)
    center = puff_center(q, length)
    assert _circular_distance(center, 0.0, length) < 0.5


def test_puff_center_zero_field_is_nan() -> None:
    assert np.isnan(puff_center(np.zeros(20), 10.0))


def test_edges_bracket_superthreshold_region() -> None:
    x = np.linspace(0.0, 10.0, 101)
    q = np.where((x >= 3.0) & (x <= 6.0), 1.0, 0.0)
    assert leading_edge(q, x, 0.5) == pytest.approx(6.0, abs=0.1)
    assert trailing_edge(q, x, 0.5) == pytest.approx(3.0, abs=0.1)


def test_edges_nan_when_all_laminar() -> None:
    x = np.linspace(0.0, 10.0, 50)
    q = np.zeros_like(x)
    assert np.isnan(leading_edge(q, x))
    assert np.isnan(trailing_edge(q, x))


def test_front_speed_recovers_linear_slope() -> None:
    t = np.linspace(0.0, 10.0, 20)
    positions = 2.5 * t + 7.0
    assert front_speed(t, positions) == pytest.approx(2.5)


def test_front_speed_ignores_nan_and_needs_two_points() -> None:
    t = np.array([0.0, 1.0, 2.0, 3.0])
    positions = np.array([np.nan, 1.0, 2.0, 3.0])  # first sample missing
    assert front_speed(t, positions) == pytest.approx(1.0)
    assert np.isnan(front_speed([0.0], [1.0]))


def test_front_kinematics_measures_expansion_of_growing_box() -> None:
    """A stationary box whose half-width grows at 0.2 has expansion_rate 0.2."""
    length, n = 100.0, 500
    x = np.arange(n) * (length / n)
    t = np.linspace(0.0, 100.0, 51)
    centers = np.full_like(t, 50.0)
    half_widths = 5.0 + 0.2 * t
    q = _box_snapshots(x, t, centers, half_widths, length)
    kin = front_kinematics(q, x, t, length, t_min=0.0)
    assert not kin["decayed"]
    assert kin["expansion_rate"] == pytest.approx(0.2, abs=0.02)
    assert abs(kin["drift"]) < 0.02


def test_front_kinematics_is_robust_to_periodic_wrap() -> None:
    """A constant-width box drifting and wrapping has expansion 0 and drift=v.

    This is the regression test for the wrap artifact that corrupted naive
    leading/trailing-edge speed measurement.
    """
    length, n = 100.0, 500
    x = np.arange(n) * (length / n)
    t = np.linspace(0.0, 100.0, 51)
    v = 2.0
    centers = (50.0 + v * t) % length  # crosses the seam several times
    half_widths = np.full_like(t, 8.0)
    q = _box_snapshots(x, t, centers, half_widths, length)
    kin = front_kinematics(q, x, t, length, t_min=0.0)
    assert abs(kin["expansion_rate"]) < 0.05
    assert kin["drift"] == pytest.approx(v, abs=0.1)
