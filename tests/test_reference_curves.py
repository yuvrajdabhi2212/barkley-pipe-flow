"""Tests for :mod:`barkley_pipe.reference_curves` (curve validation)."""

from __future__ import annotations

import json

import numpy as np
import pytest

from barkley_pipe.nullclines import u_nullcline
from barkley_pipe.reference_curves import (
    ReferenceCurve,
    compare_curve,
    get_curve,
    load_digitized_curve,
    stamp_curve_checksum,
)


# --------------------------------------------------------------------------- #
# compare_curve + the derived (analytic) example
# --------------------------------------------------------------------------- #
def test_analytic_curve_matches_the_model() -> None:
    curve = get_curve("u_nullcline_analytic")
    # candidate sampled on the reference grid -> exact agreement
    result = compare_curve(curve, curve.x, u_nullcline(curve.x))
    assert result["passed"]
    assert result["max_rel_error"] < 1e-12


def test_compare_curve_flags_a_wrong_candidate() -> None:
    curve = get_curve("u_nullcline_analytic")
    result = compare_curve(curve, curve.x, u_nullcline(curve.x) * 1.2)
    assert not result["passed"]
    assert result["max_rel_error"] == pytest.approx(0.2, abs=0.02)
    assert "FAIL" in result["message"]


def test_compare_curve_reports_insufficient_overlap() -> None:
    curve = get_curve("u_nullcline_analytic")  # defined on q in [0, 2]
    result = compare_curve(curve, np.array([5.0, 6.0, 7.0]), np.array([1.0, 1.0, 1.0]))
    assert not result["passed"]
    assert result["n_overlap"] < 2


# --------------------------------------------------------------------------- #
# ReferenceCurve input validation
# --------------------------------------------------------------------------- #
def test_reference_curve_requires_increasing_x() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        ReferenceCurve(key="k", description="d", x=[2.0, 1.0, 0.0], y=[1.0, 2.0, 3.0])


def test_external_curve_requires_a_citation() -> None:
    with pytest.raises(ValueError, match="external curve needs a citation"):
        ReferenceCurve(key="k", description="d", x=[0.0, 1.0], y=[1.0, 2.0],
                       provenance="dns", verification="primary_verified")


# --------------------------------------------------------------------------- #
# Digitized-curve loader: enforce provenance + integrity
# --------------------------------------------------------------------------- #
def _write_curve(directory, name, source, csv="R,tau\n1900,600\n2000,3000\n2040,60000\n"):
    (directory / f"{name}.csv").write_text(csv, encoding="utf-8")
    (directory / f"{name}.json").write_text(
        json.dumps({"description": "t", "source": source, "provenance": "dns",
                    "verification": "primary_verified", "rel_tolerance": 0.3}),
        encoding="utf-8",
    )


def test_missing_csv_raises_helpful_error(tmp_path) -> None:
    (tmp_path / "c.json").write_text(json.dumps({"source": {"doi": "10.1/x"}}), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="digitize"):
        load_digitized_curve("c", tmp_path)


def test_loader_rejects_unsourced_curve(tmp_path) -> None:
    _write_curve(tmp_path, "c", source={})  # no doi/url
    with pytest.raises(ValueError, match="no citable source"):
        load_digitized_curve("c", tmp_path)


def test_loader_enforces_checksum(tmp_path) -> None:
    _write_curve(tmp_path, "c", source={"doi": "10.1/x", "year": 2011})
    with pytest.raises(ValueError, match="no recorded sha256"):
        load_digitized_curve("c", tmp_path)
    stamp_curve_checksum("c", tmp_path)
    curve = load_digitized_curve("c", tmp_path)
    assert curve.citation.doi == "10.1/x"
    assert curve.x.shape == (3,)
    # tamper -> refused
    (tmp_path / "c.csv").write_text("R,tau\n1900,999\n2000,3000\n2040,60000\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_digitized_curve("c", tmp_path)


# --------------------------------------------------------------------------- #
# Honesty guard: the shipped template carries NO fabricated data
# --------------------------------------------------------------------------- #
def test_shipped_template_ships_no_data() -> None:
    # the repo ships a sidecar (citation) but no CSV; loading must refuse, not invent
    with pytest.raises(FileNotFoundError):
        load_digitized_curve("barkley2011_fig5a_tau_decay")
