"""Validate the reproduced tau(R) crossing against a digitized Barkley Fig. 5a.

This is the *external validation* of a trend (not just scalars): it compares this
project's reproduced ``tau_decay(R)`` / ``tau_split(R)`` curves against curves
digitized from the real Fig. 5a. It refuses to compare against a reference curve
that has no citation or whose file changed since it was stamped.

Until you digitize the figure (see ``data/reference_curves/README.md``), the
script tells you exactly what to do instead of inventing data.

Run from the repo root:  ``python scripts/validate_against_figure.py``
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from barkley_pipe.reference_curves import compare_curve, load_digitized_curve

SWEEP = Path("figures/statistics_sweep.csv")
BRANCHES = {
    "tau_decay": "barkley2011_fig5a_tau_decay",
    "tau_split": "barkley2011_fig5a_tau_split",
}


def _load_reproduced(kind: str) -> tuple[np.ndarray, np.ndarray]:
    """Read this project's reproduced (R, tau) for one branch from the sweep CSV."""
    if not SWEEP.exists():
        raise SystemExit(
            f"{SWEEP} not found — run `python scripts/make_statistics_figures.py` first."
        )
    R, tau = [], []
    with SWEEP.open() as fh:
        for row in csv.DictReader(fh):
            if row["kind"] == kind:
                R.append(float(row["R"]))
                tau.append(float(row["value"]))
    return np.array(R), np.array(tau)


def main() -> None:
    print("Validating reproduced tau(R) against digitized Barkley Fig. 5a\n")
    any_compared = False
    for kind, curve_name in BRANCHES.items():
        R, tau = _load_reproduced(kind)
        try:
            ref = load_digitized_curve(curve_name)
        except FileNotFoundError as exc:
            print(f"[{kind}] no digitized reference yet -- {exc}")
            continue
        except ValueError as exc:
            print(f"[{kind}] reference rejected (unsourced/altered): {exc}")
            continue
        result = compare_curve(ref, R, tau)
        print(f"[{kind}] {result['message']}")
        any_compared = True

    if not any_compared:
        print(
            "\nNo digitized reference curves found. To do the real external "
            "validation:\n"
            "  1. Digitize Barkley (2011) Fig. 5a with WebPlotDigitizer.\n"
            "  2. Save the points as data/reference_curves/"
            "barkley2011_fig5a_tau_{decay,split}.csv (+ filled .json sidecars).\n"
            "  3. stamp_curve_checksum(...) each, then re-run this script.\n"
            "See data/reference_curves/README.md. (No data is invented in the "
            "meantime — that is the point.)"
        )


if __name__ == "__main__":
    main()
