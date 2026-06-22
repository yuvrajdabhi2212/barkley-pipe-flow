"""Phase 2 boundary tests for the not-yet-implemented statistics module.

The discrete model (:mod:`barkley_pipe.discrete`) is implemented and tested in
``test_discrete.py``. The survival-statistics layer
(:mod:`barkley_pipe.statistics`) is still scaffolded as documented stubs; this
file asserts the module imports cleanly and its public entry points raise
``NotImplementedError`` until implemented per ``ROADMAP.md``. Replace each
``pytest.raises`` with a real behavioural test as the function is implemented.
"""

from __future__ import annotations

import numpy as np
import pytest

from barkley_pipe import statistics


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
