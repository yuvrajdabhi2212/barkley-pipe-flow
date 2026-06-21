# ROADMAP — Phase 2: the discrete model and survival statistics

Phase 1 (the continuous two-PDE model) is implemented and validated against
Barkley's Figs. 1–2 (see [README](README.md)). Phase 2 reproduces the
*discrete* coupled-map-lattice model (his Eqs. 3–6), which — unlike the
continuous model — supports stochastic puff **decay** and **splitting** and
hence the memoryless lifetime statistics and the directed-percolation onset of
sustained turbulence.

The module skeletons exist as documented stubs raising `NotImplementedError`:
[`discrete.py`](src/barkley_pipe/discrete.py) and
[`statistics.py`](src/barkley_pipe/statistics.py).

---

## Why a separate discrete model?

The continuous model is excitable/bistable but **deterministic**: a puff either
sits at constant size or expands. It cannot relaminarize or split on its own.
Barkley's discrete model replaces the smooth `q`-reaction with a piecewise-linear
**tent map** `F = f^k` whose escape window (controlled by `β > 0`) produces
**transient chaos** — so a turbulent site can spontaneously collapse. That single
ingredient yields the full statistical phenomenology of pipe transition.

> **Reynolds proxies are different.** The continuous model uses `r = O(1)`
> (`r_c ≈ 0.833`); the discrete model uses `R`, scaled so `R ≈ 2000` ↔ `Re`
> (onset `R_c ≈ 2046.2`). Keep them distinct in code (`r` vs `R`).

---

## Milestone 1 — Discrete model core (`discrete.py`)

Implement Eqs. 3–6 on a periodic coupled-map lattice:

```
q_i^{n+1} = F( q_i^n + d (q_{i-1}^n - 2 q_i^n + q_{i+1}^n), u_i^n )
u_i^{n+1} = u_i^n + ε1 (1 - u_i^n) - ε2 u_i^n q_i^n - c (u_i^n - u_{i-1}^n)
α(u,R)    = 2000 (1 - 0.8 u) / R
F = f^k,  k = 2
```

Tent map `f`, breakpoints `Q1 = α/(2-γ)`, `Q2 = (4+β-α-γ Q1)/(2+β)`:

```
f(q) = γ q                       if q < Q1
f(q) = 2q - α                    if Q1 ≤ q < 1
f(q) = 4 + β - α - (2+β) q       if 1 ≤ q < Q2
f(q) = γ Q1                      if Q2 ≤ q
```

Parameters: `ε1=0.04, ε2=0.2, k=2, β=0.4, γ=0.95, c=0.45, d=0.15` (`d ≤ 0.5`
for stability; note `c ≈ 1/Δx` from upwind advection, `d ≈ 1/Δx²` from
diffusion — `d=0.15` is *tuned* to place onset at the pipe-flow critical
Reynolds number rather than the "natural" `c² = 0.2025`).

**Tasks**
- [ ] `tent_map`, `tent_map_iterated` (vectorised, per-site `α`); unit-test the
      four branches and continuity at `Q1`, `1`, `Q2`.
- [ ] `step`, `simulate_discrete` (periodic lattice, returns `q` space-time).
- [ ] `initial_puff` with reproducible RNG seeding.

**Validation — reproduce Barkley Fig. 4** space-time diagrams:
decaying puff `R=1900`, puff splitting `R=2200`, slug from an edge state
`R=3000`.

---

## Milestone 2 — Event detection + ensembles (`statistics.py`)

- [ ] `detect_decay` (turbulent energy below threshold) and `detect_splitting`
      (two patches separated by a laminar gap `≳ 80` sites).
- [ ] `measure_lifetimes` over Monte-Carlo ensembles of randomized initial puffs.

**Compute note (Colab).** Barkley's published statistics use ~4000 realizations
on grids up to `1.2×10⁵` sites for up to `8×10⁶` steps near criticality. That is
infeasible on free Colab CPU. Plan **reduced ensembles** (e.g. a few hundred
realizations on `~10³`-site lattices) and **document the scaling**, rather than
matching the published precision. Log any truncation explicitly.

---

## Milestone 3 — Survival statistics (`statistics.py`)

- [ ] `fit_tau`: MLE for the memoryless law `P(n) ~ exp(-n/τ(R))`. The exponential
      MLE is the sample mean; use `scipy.stats.expon.fit(data, floc=0)`.
- [ ] `survival_function` + log-linear plot — **reproduce Fig. 12** (decay
      `R=1800–2000`, splitting `R=2060–2180`).
- [ ] `tau_of_R` for decay and splitting; show the **lifetime crossing near
      `R_× ≈ 2040`** (Fig. 5a inset). This must match the experimental/DNS
      crossover `Re = 2040 ± 10` (Avila et al. 2011, *Science*) — the single
      most important external validation number.

---

## Milestone 4 — Turbulence fraction & directed percolation

- [ ] `turbulence_fraction` order parameter `F_t(R)`.
- [ ] Locate onset `R_c ≈ 2046.2`; test the scaling `F_t ~ (R - R_c)^{β_DP}`.
      The canonical (1+1)D directed-percolation exponent is `β_DP = 0.276486(8)`
      (Barkley's quoted `0.28` is this rounded). **A fitted exponent in the
      `0.26–0.29` band counts as success** given finite-size effects.

---

## Milestone 5 — Original value-add

- [ ] A `τ(R)` parameter sweep (beyond the figures reproduced above).
- [ ] A quantitative **continuous-vs-discrete comparison** (e.g. front/expansion
      behaviour of the continuous model vs the discrete model's mean dynamics).

---

## Acceptance criteria

1. Fig. 4 space-time diagrams qualitatively match (decay / split / slug).
2. Survival curves are straight on a log-linear plot (memoryless) and `τ(R)`
   increases steeply (super-exponentially) with `R`.
3. The decay/splitting `τ(R)` curves cross near `R_× ≈ 2040`.
4. `F_t(R)` onset near `R_c ≈ 2046` with a DP-like exponent in `[0.26, 0.29]`.

---

## References for Phase 2

- Barkley, D. (2011). *Phys. Rev. E* **84**, 016309 — the model and all target
  figures (Figs. 4, 5, 12).
- Avila, K. et al. (2011). *Science* **333**, 192 — decay/splitting crossover at
  `Re = 2040 ± 10`.
- Hof, B. et al. (2006). *Nature* **443**, 59; Avila, Willis & Hof (2010),
  *JFM* **646**, 127 — super-exponential puff-decay lifetimes.
- Lemoult, G. et al. (2016). *Nature Physics* **12**, 254; Sipos & Goldenfeld
  (2011). *Phys. Rev. E* **84**, 035304 — directed-percolation universality.

**Citation corrections (flagged from the source reading list `1.pdf`):**
the Morón, Vela-Martín & Avila (2025), *JFM* **1022**:A48, *"Probabilistic
thresholds of turbulence decay in transitional shear flows"* concerns decay
**predictability**, not puff splitting. For puff-**splitting** / edge-state
mechanisms cite instead **Svirsky, Grafke & Frishman (2025**, arXiv:2505.05075,
*"Self-Replication of Turbulent Puffs"*) and **Frishman & Grafke (2022**,
arXiv:2205.05578, *"Mechanism for turbulence proliferation in subcritical
flows"*).
