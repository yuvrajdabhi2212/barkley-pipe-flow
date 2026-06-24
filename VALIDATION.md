# VALIDATION — sourcing real data and keeping AI hallucination out

This document is the method behind the numbers. Its single job is to make every
quantitative claim in this project **either derivable or traceable** — and to
make the absence of a source a *build failure*, not a footnote.

The threat it defends against: an AI assistant (or a tired human) producing a
number that *looks* right — "NASA measured the critical Reynolds number at
2,300", "the directed-percolation exponent is about 0.27" — that was never
actually read off a primary source. These plausible-but-fabricated values are
the validation equivalent of a beautiful plot of nonsense. The fix is not "trust
the AI less"; it's to build a system where an unsourced number physically cannot
reach a figure.

---

## 1. The one principle

> **A number may appear in a result only if it is (a) *derived* from the model
> in code, or (b) *tagged with a verifiable citation* to a primary source.
> Everything else is a placeholder and is treated as broken until promoted.**

This splits every constant in the project into two clean classes, and each class
has its own defence:

- **Derived / analytic.** If a value follows in closed form from the model
  (`r_c`, the turbulent fixed point, the laminar eigenvalues), you *recompute*
  it from the code. A derived number can't be hallucinated — only mis-derived —
  and a unit test catches mis-derivation. **Never paste an analytic value as a
  literal when you can compute it.**
- **External.** If a value comes from the outside world (an experiment, a DNS, a
  universality-class exponent), it must carry a source you can open: authors,
  year, venue, and a DOI or URL. No citation → it doesn't exist yet.

A third bucket matters for a *reproduction* study like this one:

- **Internal target.** Your own simulation outputs (e.g. the measured
  decay/splitting crossing `R_x ≈ 2038`). These are **not reference data** —
  they're the thing being validated. They must always be compared *against* an
  external anchor, never quoted as if they were authoritative.

The confusion of these three categories is where most "validation" quietly goes
wrong. Comparing your output to your own earlier output and calling it
"validated against the literature" is the most common self-deception. Keeping the
buckets separate (in code, not just in your head) is half the battle.

---

## 2. How this is enforced in the repo

Two files turn the principle above into something a machine checks:

| File | Role |
|------|------|
| [`src/barkley_pipe/reference_data.py`](src/barkley_pipe/reference_data.py) | The **registry**: one object per validation target, each carrying its value, units, provenance, tolerance, citation, and a *verification status* recording how far it was actually checked. Analytic entries are **recomputed from the model**; external entries hold a full `Citation`. |
| [`tests/test_reference_validation.py`](tests/test_reference_validation.py) | The **enforcement**: 13 tests that fail the build if an analytic value drifts from the code, if an external value lacks a complete citation, if any value is marked `unverified`, or if an internal target falls outside the experimental band it claims to reproduce. |

A machine-readable copy lives at
[`data/reference_values.json`](data/reference_values.json) (regenerate with
`python -m barkley_pipe.reference_data`) so plotting code, notebooks, or
non-Python tooling can read the same single source of truth.

The day-to-day workflow:

```python
from barkley_pipe import reference_data as rd

# Measured something? Check it against a sourced target before you trust it.
passed, msg = rd.check("Re_crossover_avila2011", my_measured_crossing)
print(msg)
# [PASS] Re_crossover_avila2011: candidate=2038 vs reference=2040
#        (tol=10, |Δ|=2) -- experimental/citation_verified, source: Avila et al. (2011)
```

If you find yourself about to type a reference number directly into a plotting
script — stop, and put it in the registry with its source instead. The registry
is the *only* place external numbers are allowed to live.

---

## 3. The verification ladder (be honest about how far you checked)

A citation existing is **not** the same as the *number* being confirmed. This is
the subtle trap: an AI can hand you a perfectly real paper (right authors, real
DOI) attached to a number that *isn't in it*. So the registry records a
`verification` status, and you should treat the rungs differently:

| Status | Meaning | Trust |
|--------|---------|-------|
| `derived` | Recomputed in this repo from the model | Highest — reproducible on demand |
| `primary_verified` | The *number* was read in the primary source's own text/figures | High |
| `secondary_corroborated` | Confirmed via a reputable review or independent source | Good |
| `citation_verified` | The *paper* is confirmed real, but the exact number sits behind a paywall/figure you didn't open | **Provisional — flag it** |
| `unverified` | Placeholder. **Must never appear on a published plot** (a test enforces this) | None |

Worked example from this project. The headline external number — the
decay/splitting crossover **`Re = 2040 ± 10`** (Avila et al. 2011, *Science*) —
is logged as `citation_verified`, not `primary_verified`. Why the honesty? The
bibliographic record was confirmed directly against PubMed (PMID 21737736) and
the ISTA Research Explorer (Science **333**(6039):192–196, DOI
`10.1126/science.1203223`), and the abstract confirms the "distinct critical
point" claim — but the exact figure `2040 ± 10` lives in the paper's paywalled
body, corroborated by the Avila–Barkley–Hof (2023) review rather than read
first-hand. That distinction is logged in the entry's `verification_note`. **Tell
the reader exactly what you confirmed and what you didn't** — that single habit
is what separates a trustworthy reproduction from a confident-sounding one.

---

## 4. Where to get authoritative data (for this problem)

For transitional pipe flow specifically, the genuinely citable sources are a
short, knowable list. You do not need a sprawling dataset hunt; you need the
*right* handful and their DOIs.

**Primary model & theory**
- **Barkley (2011)**, *Simplifying the complexity of pipe flow*, Phys. Rev. E
  **84**, 016309 — the model itself; everything analytic here is *its* algebra.
  [arXiv:1101.4125](https://arxiv.org/abs/1101.4125) (open access — read the real
  equations, don't paraphrase them from memory).
- **Barkley (2016)**, *Theoretical perspective on the route to turbulence in a
  pipe*, J. Fluid Mech. **803**, P1.

**External validation anchors**
- **Avila, Moxey, de Lózar, Avila, Barkley, Hof (2011)**, *The onset of
  turbulence in pipe flow*, Science **333**, 192–196 — the `Re = 2040 ± 10`
  crossover. [PMID 21737736](https://pubmed.ncbi.nlm.nih.gov/21737736/).
- **Avila, Barkley, Hof (2023)**, *Transition to turbulence in pipe flow*, Annu.
  Rev. Fluid Mech. **55**, 575–602 — a review that corroborates the numbers and
  is often open enough to read directly.
- Directed-percolation exponents (`β = 0.276486(8)`, (1+1)D): the DP literature
  (Jensen 1999; Hinrichsen's review; [arXiv:cond-mat/0205490](https://arxiv.org/abs/cond-mat/0205490)).

**Raw experimental / DNS data, when you need curves not constants**
- Author / group data repositories (e.g. the Avila group's pipe-flow codes
  **nsPipe / nsCouette**) and the journals' supplementary-material archives.
- General-purpose archives that mint a DOI for a dataset: **Zenodo**,
  **figshare**, **Dryad**, institutional repositories. A dataset with a DOI and a
  README is citable; a number someone typed into a chat window is not.

Rules of thumb that keep this honest:

1. **Prefer the open-access version of the same paper** (arXiv, author's site)
   so you can read the actual figure instead of trusting a summary of it.
2. **Cite the dataset, not the screenshot.** If you digitize points off a
   published plot (e.g. with WebPlotDigitizer), record *which figure of which
   paper*, and treat the digitized values as having real extraction error.
3. **A peer-reviewed number with a DOI beats a blog beats an AI summary** — in
   that order, every time.

---

## 5. How to actually use the data to validate

Validation is a *comparison with a stated tolerance*, never an "it looks close".
For each claim:

1. **Pick the right comparison — like for like.** This is where reproductions
   most often cheat themselves. In this project `R_x` (the decay/splitting
   *crossover*) and `R_c` (the sustained-turbulence *onset*) are **different
   physical quantities**: `R_x ≈ 2038` is validated against Avila's `2040 ± 10`,
   but `R_c ≈ 2046` is *not* forced into that band — it's a different point on
   the curve. The registry encodes this by giving `R_x_reproduced` an
   `anchor_key` and deliberately giving `R_c_onset_reproduced` none, with a note
   saying why. Forcing two different quantities to match is a classic way to
   manufacture false agreement.
2. **State a tolerance up front,** from the source's own uncertainty (`± 10`) or
   the known finite-size/ensemble error of your method — *before* you see your
   result, so the bar isn't moved to fit the outcome.
3. **Assert it in a test,** so the comparison reruns on every change and a
   regression breaks the build (see `test_reproduced_crossover_falls_within_experimental_band`).
4. **Match the parameterization.** This model uses two distinct Reynolds proxies
   — continuous `r = O(1)` (`r_c ≈ 0.833`) and discrete `R` (`R ≈ 2000 ↔ Re`).
   Comparing an `r` to an `Re` is a units error dressed up as a result.
5. **Report the residual, not a verdict.** "Reproduced `R_x = 2038` vs
   experimental `2040 ± 10` (|Δ| = 2, inside band)" is a result. "Validated ✓" is
   marketing.

For trends/curves rather than single numbers, the same logic applies pointwise:
overlay your curve on the *digitized source curve with its citation in the
caption*, and quantify agreement (RMS error over the overlap, slope on the
expected scaling axis) rather than asserting the shapes "look the same".

---

## 6. Detecting and avoiding AI hallucination (a checklist)

When an AI assistant — including this one — hands you a reference value, run it
through this gate before it touches your project. Most hallucinated numbers fail
at least one rung.

**Refuse the number until it has a source.** The correct prompt is *"give me the
citation and DOI; if you can't, say so"* — not *"what's the critical Reynolds
number?"*. A model that can't produce a checkable source for a specific figure is
guessing, and a good one will tell you it's guessing.

**Open the source yourself.** Click the DOI. Confirm the paper exists, the
authors/venue/year match, and — for anything you'll lean on — that the *number is
actually in it*. AIs hallucinate the binding between a real paper and a number
far more often than they invent a fake paper outright.

**Watch for the tells of a fabricated value:**
- Suspiciously round or "classic-textbook" numbers offered for a quantity the
  literature actually reports with scatter (e.g. a flat "2300" for pipe
  transition, when the sustained-turbulence onset is `~2040` and the transitional
  range is broad and setup-dependent).
- A citation with everything *except* a working DOI/URL, or page numbers that
  don't resolve.
- Confidence that *increases* when you ask for the source (a real source makes a
  claim more specific and hedged; a fabricated one tends to get smoother).
- The same number attributed to different sources on re-asking, or shifting by a
  few percent between answers.

**Cross-check across independent sources.** A value that appears, identical, in
the primary paper *and* an independent review (here: Avila 2011 and the Avila
2023 review) is trustworthy. A value that appears only in the AI's answer is not.

**Prefer derivation over citation whenever the math allows.** This project's
strongest numbers — `r_c = 5/6`, the turbulent node `(1.343, 0.130)`, the
eigenvalues `(-0.1, -0.04)` — are bulletproof precisely because they are
*recomputed from the model* in `reference_data.py` and re-checked by tests. There
is nothing to hallucinate. Push as many of your constants as possible into this
category.

**Let the test suite be the gatekeeper.** The human reviewer (and the AI) can be
fooled; `pytest` checking a number against a sourced registry entry cannot be
sweet-talked. If a value can't pass `tests/test_reference_validation.py`, it
doesn't ship — regardless of how plausible it sounds.

**Separate "draft with AI" from "verify without it".** It's fine to let an AI
*draft* the registry, *find candidate* sources, or *write* the tests. It is not
fine to let it be the final authority on a number. The drafting and the
verification must use independent channels — the whole point of the
primary-source click is that it doesn't route through the model.

---

## 7. Adding a new reference value (the routine)

1. Decide its bucket: **analytic**, **external**, or **internal_target**.
2. If analytic — write a `_derive_*` function that computes it from the model and
   reference it; don't hardcode the literal.
3. If external — open the primary source, copy the value *and* its uncertainty,
   and fill in a complete `Citation` (DOI or URL mandatory). Set `verification`
   honestly (`primary_verified` only if you read the number in the source).
4. If internal_target — set `anchor_key` to the external entry it reproduces (or
   leave it `None` and explain in `verification_note` why it has no anchor).
5. Run `pytest tests/test_reference_validation.py`. If it's external and
   unsourced, the build will reject it — as intended.
6. Regenerate the JSON: `python -m barkley_pipe.reference_data`.

---

*Verification pass logged 2026-06-23: `r_c`, the turbulent node, and the laminar
eigenvalues re-derived from the model and unit-checked; `β_DP = 0.276486(8)`
corroborated against the directed-percolation literature; the Avila et al. (2011)
citation confirmed against PubMed and ISTA, with the `2040 ± 10` value recorded
as `citation_verified` pending first-hand reading of the paper body.*
