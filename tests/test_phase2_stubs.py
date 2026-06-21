"""Phase 2 boundary tests.

Phase 2 (discrete coupled-map-lattice model and its survival statistics) is
scaffolded as documented stubs. These tests assert that the modules import
cleanly (so the package as a whole is importable and CI stays green) and that
the public Phase 2 entry points raise ``NotImplementedError`` until they are
implemented per ``ROADMAP.md``. Replace each ``pytest.raises`` with a real
behavioural test as the corresponding function is implemented.
"""

from __future__ import annotations

import numpy as np
import pytest

from barkley_pipe import discrete, statistics


def test_discrete_module_constants_present() -> None:
    # Parameters are real scaffolding (not "logic") and should be defined.
    assert discrete.BETA > 0  # enables spontaneous decay
    assert discrete.D <= 0.5  # diffusion stability bound
    assert discrete.K == 2
    params = discrete.DiscreteParams(R=2200.0)
    assert params.R == 2200.0
    assert params.gamma == discrete.GAMMA


@pytest.mark.parametrize(
    "call",
    [
        lambda: discrete.threshold_alpha(np.ones(4), 2200.0),
        lambda: discrete.tent_map(np.ones(4), np.ones(4)),
        lambda: discrete.tent_map_iterated(np.ones(4), np.ones(4)),
        lambda: discrete.initial_puff(64),
        lambda: discrete.step(np.ones(4), np.ones(4), discrete.DiscreteParams(R=2200.0)),
        lambda: discrete.simulate_discrete(np.ones(4), np.ones(4),
                                           discrete.DiscreteParams(R=2200.0), 10),
    ],
)
def test_discrete_stubs_raise(call) -> None:
    with pytest.raises(NotImplementedError):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: statistics.detect_decay(np.ones((3, 4))),
        lambda: statistics.detect_splitting(np.ones((3, 4))),
        lambda: statistics.measure_lifetimes([np.ones((3, 4))]),
        lambda: statistics.fit_tau([1.0, 2.0, 3.0]),
        lambda: statistics.survival_function([1.0, 2.0, 3.0]),
        lambda: statistics.turbulence_fraction(np.ones((3, 4))),
        lambda: statistics.tau_of_R([1900.0, 2000.0]),
    ],
)
def test_statistics_stubs_raise(call) -> None:
    with pytest.raises(NotImplementedError):
        call()
