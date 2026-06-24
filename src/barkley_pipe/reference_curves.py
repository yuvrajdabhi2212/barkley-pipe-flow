"""Validate reproduced *curves* against citation-tagged reference curves.

Companion to :mod:`barkley_pipe.reference_data` (which handles scalar values).
A *trend* — e.g. the decay/splitting ``tau(R)`` crossing of Barkley Fig. 5a — is
validated by overlaying the reproduced curve on a reference curve and quantifying
the agreement (max / RMS relative error over the overlap), exactly as
``VALIDATION.md`` §5 prescribes for "trends/curves rather than single numbers".

The same anti-hallucination rule applies as for scalars: a reference curve must
be either

* **derived** — computed in closed form from the model (cannot be hallucinated,
  only mis-derived, which a test catches), or
* **digitized** — extracted from a *cited* published figure (e.g. with
  WebPlotDigitizer) and loaded **only if** it carries a verifiable citation and
  still matches its recorded checksum.

:func:`load_digitized_curve` refuses unsourced or silently-edited curve data, so
— just like the scalar registry — a fabricated curve cannot reach a comparison.
Digitized reference curves live under ``data/reference_curves/`` as a ``<name>.csv``
plus a ``<name>.json`` provenance sidecar; **this repo ships none pre-filled** (a
digitized curve must come from you reading a real figure, not from an AI). See
``data/reference_curves/README.md`` for the one-time procedure.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .reference_data import Citation, EXTERNAL_PROVENANCE, PROVENANCE, VERIFICATION

__all__ = [
    "ReferenceCurve",
    "compare_curve",
    "curve_dir",
    "load_digitized_curve",
    "stamp_curve_checksum",
    "get_curve",
    "curve_keys",
    "CURVES",
]

_PathLike = Union[str, Path]


def curve_dir() -> Path:
    """Default directory for digitized reference curves, ``<repo>/data/reference_curves``."""
    return Path(__file__).resolve().parents[2] / "data" / "reference_curves"


def _content_hash(path: _PathLike) -> str:
    """SHA-256 of a file's newline-normalized UTF-8 text (EOL-independent)."""
    return hashlib.sha256(
        Path(path).read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class ReferenceCurve:
    """A reference trend with full provenance (the curve analogue of ``ReferenceValue``).

    Attributes
    ----------
    key, description : str
        Identifier and plain-language meaning.
    x, y : numpy.ndarray
        Reference samples (``x`` strictly increasing).
    x_label, y_label, units : str
        Axis labels / units, for honest plotting.
    provenance : str
        One of :data:`barkley_pipe.reference_data.PROVENANCE`.
    verification : str
        One of :data:`barkley_pipe.reference_data.VERIFICATION`.
    rel_tolerance : float
        Max allowed relative error of a candidate over the overlap (acceptance bar).
    citation : Citation, optional
        Required for external provenance; ``None`` allowed for analytic curves.
    uncertainty, verification_note : str
        Reported uncertainty / exactly what was checked.
    """

    key: str
    description: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    x_label: str = ""
    y_label: str = ""
    units: str = ""
    provenance: str = "analytic"
    verification: str = "derived"
    rel_tolerance: float = 0.10
    citation: Optional[Citation] = None
    uncertainty: str = ""
    verification_note: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=np.float64))
        object.__setattr__(self, "y", np.asarray(self.y, dtype=np.float64))
        if self.x.shape != self.y.shape or self.x.ndim != 1:
            raise ValueError(f"{self.key}: x and y must be 1-D and the same length")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError(f"{self.key}: x must be strictly increasing")
        if self.provenance not in PROVENANCE:
            raise ValueError(f"{self.key}: unknown provenance {self.provenance!r}")
        if self.verification not in VERIFICATION:
            raise ValueError(f"{self.key}: unknown verification {self.verification!r}")
        if self.provenance in EXTERNAL_PROVENANCE and self.citation is None:
            raise ValueError(f"{self.key}: external curve needs a citation")


def compare_curve(
    curve: Union[ReferenceCurve, str],
    candidate_x: ArrayLike,
    candidate_y: ArrayLike,
) -> dict:
    """Quantify how well a reproduced curve matches a reference curve.

    The candidate is interpolated onto the reference ``x`` over the overlapping
    range, and the relative error ``|cand - ref| / |ref|`` is summarised. Returns
    a result dict (not a bare verdict) so you report the residual, per
    ``VALIDATION.md`` §5.

    Parameters
    ----------
    curve : ReferenceCurve or str
        The reference curve, or a key in :data:`CURVES`.
    candidate_x, candidate_y : array_like
        The reproduced curve to test.

    Returns
    -------
    dict
        ``{passed, max_rel_error, rms_rel_error, n_overlap, message}``.
    """
    ref = get_curve(curve) if isinstance(curve, str) else curve
    cx = np.asarray(candidate_x, dtype=np.float64)
    cy = np.asarray(candidate_y, dtype=np.float64)
    order = np.argsort(cx)
    cx, cy = cx[order], cy[order]

    lo, hi = max(ref.x.min(), cx.min()), min(ref.x.max(), cx.max())
    mask = (ref.x >= lo) & (ref.x <= hi)
    n = int(mask.sum())
    if n < 2:
        return {"passed": False, "max_rel_error": np.nan, "rms_rel_error": np.nan,
                "n_overlap": n, "message": f"{ref.key}: insufficient x-overlap"}

    cand_on_ref = np.interp(ref.x[mask], cx, cy)
    denom = np.where(np.abs(ref.y[mask]) > 1e-12, np.abs(ref.y[mask]), np.nan)
    rel = np.abs(cand_on_ref - ref.y[mask]) / denom
    max_rel = float(np.nanmax(rel))
    rms_rel = float(np.sqrt(np.nanmean(rel**2)))
    passed = bool(max_rel <= ref.rel_tolerance)
    src = ref.citation.short() if ref.citation else f"{ref.provenance}/{ref.verification}"
    return {
        "passed": passed,
        "max_rel_error": max_rel,
        "rms_rel_error": rms_rel,
        "n_overlap": n,
        "message": (
            f"[{'PASS' if passed else 'FAIL'}] {ref.key}: max rel err={max_rel:.3g}, "
            f"rms={rms_rel:.3g} over {n} pts (tol={ref.rel_tolerance:g}) -- vs {src}"
        ),
    }


def stamp_curve_checksum(name: str, directory: Optional[_PathLike] = None) -> str:
    """Record a digitized curve's content hash into its sidecar; return the hash.

    Run this once after exporting ``<name>.csv`` from WebPlotDigitizer and filling
    in the ``<name>.json`` source block.
    """
    directory = Path(directory) if directory is not None else curve_dir()
    sidecar = directory / f"{name}.json"
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    digest = _content_hash(directory / f"{name}.csv")
    meta["sha256"] = digest
    sidecar.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return digest


def load_digitized_curve(name: str, directory: Optional[_PathLike] = None) -> ReferenceCurve:
    """Load a digitized reference curve, enforcing citation + content integrity.

    Reads ``<name>.csv`` (two numeric columns ``x,y``) and ``<name>.json`` (a
    provenance sidecar). Raises if the sidecar names no citable source or the
    file's content hash no longer matches the recorded one.

    Parameters
    ----------
    name : str
        Curve name (without extension).
    directory : path-like, optional
        Directory holding the curve. Defaults to :func:`curve_dir`.

    Raises
    ------
    FileNotFoundError
        If the ``.csv`` data file (still to be digitized) or sidecar is missing.
    ValueError
        If the sidecar lacks a citable source / checksum, or the hash mismatches.
    """
    directory = Path(directory) if directory is not None else curve_dir()
    csv_path = directory / f"{name}.csv"
    sidecar = directory / f"{name}.json"
    if not sidecar.exists():
        raise FileNotFoundError(f"no provenance sidecar for curve {name!r}: {sidecar}")
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    if not csv_path.exists():
        raise FileNotFoundError(
            f"curve {name!r}: no data file {csv_path.name} yet -- digitize the cited "
            "figure (see data/reference_curves/README.md) and stamp it before use"
        )
    source = meta.get("source") or {}
    if not (source.get("doi") or source.get("url")):
        raise ValueError(f"{name!r}: curve sidecar names no citable source (doi/url)")
    recorded = meta.get("sha256")
    if not recorded:
        raise ValueError(f"{name!r}: curve sidecar has no recorded sha256")
    if _content_hash(csv_path) != recorded:
        raise ValueError(
            f"{name!r}: curve data changed since it was sourced/stamped (hash mismatch)"
        )

    rows = [ln for ln in csv_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    data = np.array([[float(v) for v in ln.split(",")] for ln in rows[1:]], dtype=np.float64)
    citation = Citation(
        authors=source.get("authors", ""), year=int(source.get("year", 0) or 0),
        title=source.get("title", ""), venue=source.get("venue", ""),
        doi=source.get("doi", ""), url=source.get("url", ""),
        note=source.get("obtained_by", ""),
    )
    return ReferenceCurve(
        key=name, description=meta.get("description", ""),
        x=data[:, 0], y=data[:, 1],
        x_label=meta.get("x_label", ""), y_label=meta.get("y_label", ""),
        units=meta.get("units", ""),
        provenance=meta.get("provenance", "experimental"),
        verification=meta.get("verification", "citation_verified"),
        rel_tolerance=float(meta.get("rel_tolerance", 0.15)),
        citation=citation, uncertainty=meta.get("uncertainty", ""),
        verification_note=meta.get("verification_note", ""),
    )


# --------------------------------------------------------------------------- #
# Derived (analytic) reference curves — recomputed from the model, never hardcoded
# --------------------------------------------------------------------------- #
def _derive_u_nullcline() -> ReferenceCurve:
    """The u-nullcline u = eps1/(eps1 + eps2 q): an exact curve from the model.

    Used to demonstrate (and unit-test) the curve-comparison machinery end to end
    with genuinely non-fabricated data. The *external* validation slot — a
    digitized Barkley Fig. 5a — is left for the user to fill (see module docstring).
    """
    from .nullclines import u_nullcline

    q = np.linspace(0.0, 2.0, 41)
    return ReferenceCurve(
        key="u_nullcline_analytic",
        description="u-nullcline u = eps1/(eps1+eps2 q), exact consequence of the model.",
        x=q, y=u_nullcline(q),
        x_label="q", y_label="u", units="dimensionless",
        provenance="analytic", verification="derived", rel_tolerance=1e-9,
        verification_note="Recomputed from nullclines.u_nullcline; demonstrates the "
        "curve-comparison machinery with derived (un-hallucinatable) data.",
    )


CURVES: dict[str, ReferenceCurve] = {c.key: c for c in (_derive_u_nullcline(),)}


def curve_keys() -> list[str]:
    """Names of the registered reference curves."""
    return sorted(CURVES)


def get_curve(key: str) -> ReferenceCurve:
    """Return a registered reference curve, or raise with the known keys."""
    try:
        return CURVES[key]
    except KeyError as exc:
        raise KeyError(f"No reference curve {key!r}. Known: {', '.join(curve_keys())}") from exc
