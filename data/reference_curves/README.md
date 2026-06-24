# Reference curves — validating a *trend* against a digitized figure

This folder holds **reference curves** (x, y trends) used to validate reproduced
curves quantitatively — the curve analogue of the scalar registry in
[`reference_data.py`](../../src/barkley_pipe/reference_data.py). It implements
[`VALIDATION.md`](../../VALIDATION.md) §5's prescription for trends:

> overlay your curve on the digitized source curve with its citation, and
> quantify agreement (RMS error over the overlap) rather than asserting the
> shapes "look the same".

**No curve data is shipped pre-filled.** A reference curve must be either
*derived* from the model (see `reference_curves._derive_*`) or *digitized by you*
from a cited figure — never numbers an AI produced from memory. Each digitized
curve is a pair:

```
<name>.csv     # two columns: x,y  (header row + numeric rows)
<name>.json    # provenance sidecar (citation + checksum)
```

## One-time procedure: validate against Barkley Fig. 5a

The headline comparison is your reproduced `τ(R)` crossing
([`figures/fig7_tau_crossing.png`](../../figures/fig7_tau_crossing.png)) against
the real Fig. 5a from the paper.

1. **Get the figure.** Open Barkley (2011), Phys. Rev. E **84**, 016309 — the
   open-access version is [arXiv:1101.4125](https://arxiv.org/abs/1101.4125).
   Look at Fig. 5a (the decay and splitting lifetime curves).
2. **Digitize it.** In [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/):
   load the figure image, calibrate the axes (note the `τ` axis is logarithmic),
   and extract points along the **decay** branch and the **splitting** branch.
   Export each as CSV (`x,y` = `R, τ`).
3. **Drop them in** as `barkley2011_fig5a_tau_decay.csv` and
   `barkley2011_fig5a_tau_split.csv` here, each next to a `.json` sidecar
   (copy the template `barkley2011_fig5a_tau_decay.json`, fill `accessed`, and set
   `obtained_by` to record that *you* digitized it and roughly how many points).
4. **Stamp the checksum** so the file can't change silently:
   ```python
   from barkley_pipe.reference_curves import stamp_curve_checksum
   stamp_curve_checksum("barkley2011_fig5a_tau_decay")
   stamp_curve_checksum("barkley2011_fig5a_tau_split")
   ```
5. **Run the comparison:**
   ```bash
   python scripts/validate_against_figure.py
   ```
   It loads your reproduced `τ(R)` from `figures/statistics_sweep.csv`, loads the
   digitized reference curves (refusing them if unsourced or altered), and reports
   the max/RMS relative error over the overlap — a real number, not "looks close".

## Honesty notes

- Digitized points carry **real extraction error** (you are reading a log axis by
  eye). Set `rel_tolerance` in the sidecar to a value that reflects that
  (~15–25%) plus this project's reduced-ensemble error — *before* you look at the
  result, so the bar isn't moved to fit the outcome.
- Set `verification` honestly: `primary_verified` only once you have actually read
  the curve off the paper's own figure (not a summary of it).
- The shipped `*.json` is a **template with no `*.csv`**; `load_digitized_curve`
  raises a clear "digitize it first" error until you provide genuine data — by
  design, so an empty/unsourced curve cannot quietly enter a comparison.
