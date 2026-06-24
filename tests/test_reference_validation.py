"""Validation of model outputs against the citation-tagged reference registry.

These tests are the executable half of the anti-hallucination workflow (see
``VALIDATION.md``). They enforce three things:

1. *Analytic* reference values still equal what the model actually produces
   (so the registry can't silently drift from the code).
2. *Every external* reference value carries a real, complete citation
   (so an unsourced "NASA-ish" number can never enter the registry unnoticed).
3. *Internal reproduction targets* are consistent with the external anchor they
   claim to reproduce (so "we matched the literature" is a checkable assertion,
   not a vibe).

If any of these fail, the build fails -- which is the point: a hallucinated or
unsourced number should break CI, not quietly appear on a figure.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from barkley_pipe import nullclines, reference_data as rd


# --------------------------------------------------------------------------- #
# 1. Analytic values are re-derived and must match the registry               #
# --------------------------------------------------------------------------- #

def test_critical_r_matches_registry():
    derived = nullclines.critical_r()
    ref = rd.get("r_c")
    assert abs(derived - ref.value) <= ref.tolerance
    # And it is the exact rational 5/6, not an eyeballed 0.833.
    assert abs(derived - 5.0 / 6.0) < 1e-12


def test_turbulent_node_matches_registry():
    pts = nullclines.fixed_points(1.0)
    q, u = pts[-1]  # largest-q root = turbulent node
    assert abs(q - rd.get("turbulent_node_q_r1").value) <= rd.get("turbulent_node_q_r1").tolerance
    assert abs(u - rd.get("turbulent_node_u_r1").value) <= rd.get("turbulent_node_u_r1").tolerance
    # Sanity vs the human-readable README claim (1.34, 0.13).
    assert q == pytest.approx(1.34, abs=0.02)
    assert u == pytest.approx(0.13, abs=0.02)


def test_laminar_eigenvalues_match_registry():
    # Jacobian at the laminar point (0, 1) must have eigenvalues (-delta, -eps1).
    J = nullclines.jacobian(0.0, 1.0, r=0.7)
    eigs = sorted(np.linalg.eigvals(J).real)
    expected = sorted([rd.get("laminar_eig_1").value, rd.get("laminar_eig_2").value])
    assert eigs == pytest.approx(expected, abs=1e-9)


def test_analytic_entries_are_recomputable_not_just_literals():
    # Re-run the registry's own derivation path and confirm it reproduces the
    # stored analytic numbers. This guarantees they were derived, not pasted.
    assert rd._derive_critical_r() == pytest.approx(rd.get("r_c").value, abs=1e-12)
    dq, du = rd._derive_turbulent_node(1.0)
    assert dq == pytest.approx(rd.get("turbulent_node_q_r1").value, abs=1e-9)
    assert du == pytest.approx(rd.get("turbulent_node_u_r1").value, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Every external reference value is properly sourced                        #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", rd.external_keys())
def test_external_value_has_complete_citation(key):
    ref = rd.get(key)
    assert ref.citation is not None, f"{key} is external but has no citation"
    c = ref.citation
    assert c.authors and c.year and c.title and c.venue, f"{key}: incomplete citation"
    # Must be locatable: a DOI or a URL.
    assert c.doi or c.url, f"{key}: citation has neither DOI nor URL"
    # Must not silently claim more verification than was done.
    assert ref.verification in rd.VERIFICATION
    assert ref.verification != "unverified", f"{key}: unverified value must not ship"
    # External numbers should declare an uncertainty/tolerance, not pretend to be exact.
    assert ref.tolerance > 0 or ref.uncertainty, f"{key}: external value lacks an uncertainty"


def test_no_unverified_values_anywhere():
    offenders = [k for k, v in rd.REGISTRY.items() if v.verification == "unverified"]
    assert not offenders, f"Unverified values present (do not plot these): {offenders}"


def test_provenance_and_verification_vocab_are_respected():
    for key, v in rd.REGISTRY.items():
        assert v.provenance in rd.PROVENANCE, f"{key}: bad provenance"
        assert v.verification in rd.VERIFICATION, f"{key}: bad verification"


# --------------------------------------------------------------------------- #
# 3. Internal reproduction targets are consistent with their external anchor   #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("key", rd.internal_target_keys())
def test_internal_target_is_anchored_or_explicitly_not(key):
    ref = rd.get(key)
    # An internal target must EITHER name an external anchor it reproduces,
    # OR carry a note explaining why it has none (different quantity, etc.).
    has_anchor = ref.anchor_key is not None
    explains_absence = bool(ref.verification_note)
    assert has_anchor or explains_absence, (
        f"{key}: internal target with no anchor and no explanation"
    )
    if has_anchor:
        anchor = rd.get(ref.anchor_key)
        assert anchor.provenance in rd.EXTERNAL_PROVENANCE, (
            f"{key}: anchor {ref.anchor_key} is not an external reference"
        )


def test_reproduced_crossover_falls_within_experimental_band():
    target = rd.get("R_x_reproduced")
    anchor = rd.get(target.anchor_key)  # Re = 2040 +/- 10 (Avila 2011)
    band = anchor.tolerance
    assert abs(target.value - anchor.value) <= band, (
        f"Reproduced crossing R_x={target.value} is outside the experimental "
        f"band {anchor.value} +/- {band} (Avila et al. 2011)"
    )


def test_check_helper_reports_pass_and_fail():
    ok, msg = rd.check("r_c", 5.0 / 6.0)
    assert ok and "PASS" in msg
    bad, msg2 = rd.check("r_c", 0.9)
    assert not bad and "FAIL" in msg2


# --------------------------------------------------------------------------- #
# JSON export round-trips (keeps data/reference_values.json honest)           #
# --------------------------------------------------------------------------- #

def test_json_export_round_trips(tmp_path):
    import json

    out = rd.export_json(tmp_path / "ref.json")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["values"]) == set(rd.REGISTRY)
    # Spot-check that the anchor relationship survives serialization.
    assert payload["values"]["R_x_reproduced"]["anchor_key"] == "Re_crossover_avila2011"
