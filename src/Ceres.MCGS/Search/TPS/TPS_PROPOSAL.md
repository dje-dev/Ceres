# Tempered Posterior Search (TPS)
### A unification and radical simplification of CBGPUCT
*(draft proposal 2026-07-13; dje concept, elaborated by Claude; no code exists yet)*

---

## 1. Technical description

### Notation
At a node with parent visit count `N`, children `i = 1..k`:
- `p_i` — network policy prior (normalized over legal moves)
- `n_i` — child visit count (per-edge), `Q_i` — child value estimate from search
- `q^FPU_i` — the per-child policy-imputed FPU value (existing PolicyImputedRPO machinery, unchanged)
- `s_i` — child leaf-value volatility (existing `RunningStdDevShort`, debiased)
- `σ̂_i = s_i / sqrt(n_i + 1)` — estimated standard error of `Q_i` ("child noise")
- `σ̄` — the node's noise scale: visit-weighted median of `σ̂_i` over visited children
  (fallback when volatility tracking is off: `σ̄ = s₀/sqrt(n_med+1)` with global default s₀)

### STEP 1 — Robust Q (one formula replaces the three cases)
Inverse-variance shrinkage of each child's search Q toward its FPU-imputed value:

    q̃_i = w_i·Q_i + (1 − w_i)·q^FPU_i,      w_i = σ₀² / (σ₀² + σ̂_i²)

where σ₀ is the *prior confidence scale* — how much noise we tolerate before distrusting
search Q. Properties: `n_i = 0` → σ̂ = ∞ → w = 0 → pure FPU imputation (exact match to the
sketch's case 2); large `n_i` → w → 1 → raw child Q (case 1, but smooth — no arbitrary
N>50 cutoff needed); in between, prior influence decays like σ̂_i ∝ 1/sqrt(n_i) (case 3,
exactly the suggested rate), automatically modulated by the child's *measured* volatility —
the FUTURE ENHANCEMENT (uncertainty-head / visit-volatility awareness) is thus built in from
day one rather than bolted on, at zero extra machinery. An uncertainty head, when available,
simply replaces or blends into `s_i`.

Note this *subsumes the existing support-shrinkage* (Select/BackupSupportShrinkageK): the
consensus-prior shrinkage and the FPU-imputation shrinkage collapse into one operation with
one target (q^FPU, which is itself consensus-anchored via the anchor machinery).

### STEP 2 — Posterior policy (suggested name for "effective policy")
The anchored-softmax the sketch asks for is exponential tilting of the prior — the closed-form
solution of entropy-regularized policy improvement
(`argmax_π ⟨π, q̃⟩ − τ·KL(π ‖ p^β/Z)`):

    π̃_i(τ) ∝ p_i^{β(N)} · exp( q̃_i / τ )

- **Prior anchor decay**: `β(N) = 1 / sqrt(1 + N/N₀)` — full policy anchoring at small N,
  fading like 1/sqrt(N) (the suggested rate) as evidence accumulates; at N ≫ N₀ the
  posterior is driven almost purely by robust Q. (Equivalently: a log-prior bonus
  `β(N)·τ·ln p_i` added to q̃_i — the PUCT-like "policy bonus that evidence retires".)
- **Temperature is noise-referenced (self-calibrating)**:

      τ_select = k_s · σ̄        τ_backup = k_b · σ̄ · sqrt(2·ln k_visited)

  Dimensionless multiples of the node's *measured* value-noise scale. This is the direct
  lesson of the 2026-07-13 instrumentation runs: the small net and HOP_SP_C4 differ ~35% in
  shallow-node noise but not in deep-node noise/signal ratio — a fixed λ schedule cannot fit
  both, a noise multiple fits both automatically. The `sqrt(2·ln k)` factor in backup is the
  analytic winner's-curse scale (max of k noisy estimates is optimistically biased by
  ~σ·sqrt(2 ln k)); making it explicit keeps k_b ≈ O(1) and net-independent.
- As a node settles (σ̄ → 0), both temperatures → 0 and π̃ → argmax(q̃): select becomes
  exploitative and backup becomes minimax *exactly as fast as the noise actually shrinks* —
  the schedule is measured, not tuned. This replaces the entire λ apparatus
  (C/exp/denomBase/logBase/logFactor/NCap × 2 phases) and, we hypothesize, the
  SelectDeficitHorizon: the horizon was needed because pi_bar was a noise-amplified,
  policy-dragged target at large N; π̃ is anchored to fading policy, built on shrunk Q, and
  cooled by measured noise, so its batch-to-batch movement is intrinsically small.

### SELECT
Deficit allocation of the batch budget `B` against the posterior policy at select temperature:

    deficit_i = max(0, π̃_i(τ_select)·(N + B) − (n_i + inflight_i))

filled by the existing largest-deficit iterator. (No pi_bar floor needed in principle:
p^β keeps every legal move's π̃ strictly positive at moderate N, and at large N exploration
of near-tied moves is carried by τ; if a floor is retained it should be ε·p_i — prior-shaped
— rather than uniform.)

### BACKUP
    V = Σ_i π̃_i(τ_backup) · q̃_i        (blended with the node's own NN value as one visit,
                                          as today)

A dot product. No solver.

### Shared computation + the PathVisit cache (with the one required fix)
Select at node P already computes q̃ and both tilts (two softmaxes over the same q̃ — the
backup tilt is a cheap second exponentiation, or reuse via
`π̃(τ_b) ∝ π̃(τ_s)^{τ_s/τ_b}·p^{β(1−τ_s/τ_b)}`... in practice: just compute both). Store in
the transient `MCGSPathVisit` for each descended edge:

    S  = Σ_j π̃_j(τ_b)·q̃_j     (the backup dot product, select-time)
    (π̃_c, q̃_c_old)            (the descended child's backup weight and robust Q)

**Fatal flaw found & fixed:** naively backing up the cached `S` would propagate *stale*
values — the descended child's Q changes because of the very evaluations the visit produced,
and with a pure cache that new information would never reach the parent. The fix is O(1) and
preserves the no-solver property: at backup, recompute only the descended child's robust Q
(closed form, cheap) and update incrementally:

    V = S + Σ_{c ∈ descended} π̃_c · (q̃_c_new − q̃_c_old)

The π̃ weights are one batch stale (they refresh at the node's next select), which is benign —
weights move slowly by construction; the *values* are fresh, which is what matters for
information propagation. Graph-mode full recomputations (BottomUpQRecalculator etc.) simply
evaluate the closed form directly — no solver anywhere in the system.

### What falls away
The iterative RPO solver (both phases), both λ schedules and all their constants, both
support-shrinkage knobs + decay exponents, SelectDeficitHorizon (hypothesized), the pi_bar
floor (or reduced to ε·p_i), CBGPUCT_PUCTAboveN (select is CBGPUCT/TPS at every node by
construction), crossN + absorb machinery, flow/prior-blend experiments, breadth bonus,
schedule/denominator enums, QAnchor enum. FPU imputation machinery is *retained* (STEP 1
depends on it) as is the RunningStdDevShort tracking (promoted to load-bearing).

## 2. Tunable coefficients

| # | symbol | role | expected scale | prior evidence |
|---|--------|------|----------------|----------------|
| 1 | `k_s` | select temperature, × node noise σ̄ | O(1–5) | maps to selC=0.5 sweet spot via σ̄≈0.02–0.03 at mid-N |
| 2 | `k_b` | backup temperature, × σ̄·sqrt(2 ln k) | O(0.5–2) | winner's-curse-referenced; bkC=0.75 analogue |
| 3 | `N₀` | prior-anchor half-life (parent N at which policy influence has decayed ~30%) | O(30–300) | plays the role of denomBase=30 / PUCTAboveN=50 |
| (4) | `σ₀` | robust-Q prior-confidence scale | O(σ̄) — candidate for elimination by tying to σ̄, leaving 3 | replaces shrinkageK=1/3 |

All dimensionless or a single N-scale; 1–2 fixed analytic constants (sqrt(2 ln k), the σ̂
definition) deliberately not exposed as knobs.

## 3. Abstract

**Tempered Posterior Search: policy-anchored, noise-tempered value posteriors for
Monte-Carlo graph search.** Modern PUCT-family engines interleave two heuristics — an
exploration formula for descending the tree and an averaging rule for backing values up —
each governed by hand-tuned schedules in absolute visit counts that transfer poorly across
networks and search depths. We propose a single object that serves both roles: at every node,
a *posterior policy* formed by exponentially tilting the network's policy prior by a
*robust value vector* (child values shrunk toward policy-imputed priors in proportion to
their estimated standard error), with the prior's anchor decaying as evidence accumulates and
with temperatures expressed as dimensionless multiples of the node's *measured* value noise.
Selection allocates visits toward this posterior's deficits; backup is its expectation over
the same robust values at a winner's-curse-referenced temperature. Both computations share
one closed-form evaluation — no iterative solver — and the backup reuses selection-time
quantities via an O(1) incremental update carried in the search path, making the scheme
cheaper than the regularized-policy-optimization search it replaces.

The design is motivated by an empirical campaign on a CBGPUCT (RPO-based) implementation in
which every observed failure traced to constants denominated in absolute N standing proxy for
noise-denominated quantities: selection targets that fought accumulated evidence at large N,
value aggregation that decayed to winner's-curse-prone maxima on schedule rather than on
merit, and sharp, network-specific optima that reversed sign between a 256×10 and a 512×15
network at deep search. Direct instrumentation showed the two networks' noise profiles
differing precisely where fixed constants assume they do not (shallow nodes, ~35%) and
agreeing where schedules assume they differ (deep nodes). TPS replaces roughly fifteen live
coefficients with three dimensionless ones, each self-scaling in depth, branching factor, and
network calibration — the regime where tuning is cheap becoming, by construction, predictive
of the regime where failure is expensive.

## 4. RunningStdDevShort as the load-bearing noise oracle (speculation, per request)

The instrumentation runs validated σ̂ = s/sqrt(n+1) as smooth, ~1/sqrt(N)-decaying, and
net-discriminating exactly where it needs to be. To weave it in elegantly:
1. **It becomes always-on** (currently opt-in): ~2 bytes/node, one update per backup — the
   price of deleting a dozen constants.
2. **One quantity, three consumers**: σ̂_i drives STEP-1 shrinkage weights; the visit-weighted
   median σ̄ drives both temperatures; nothing else consumes it — no scattered
   volatility-scaling side-channels (delete both UncertaintyScalingFactor knobs and their
   misnamed semantics).
3. **Guardrails for a 2-byte estimator**: clamp σ̂ to [σ_min, σ_max] (e.g. [0.005, 0.5]) so
   quantization/early-sample artifacts cannot zero a temperature or blow up a shrinkage;
   at n_i < ~4 use the parent's σ̄ instead of the child's own (borrowed strength).
4. **Natural upgrade path**: when a net exposes an uncertainty head, it enters as the n=0
   prior for s_i (Bayes-updated by observed volatility thereafter) — no structural change.
5. **A diagnostic dividend**: since all temperatures are multiples of σ̄, logging σ̄
   percentiles per game is a one-line health check that tuning transfers across nets — the
   check we lacked when selC=0.5 silently failed on HOP.

## 5. Risks / open questions (none fatal)
- Deficit allocation retains the O(Δ·N) re-pricing property on genuine q̃ rank changes at
  large N. Believed benign here (π̃ is stable by construction; genuine discoveries *should*
  reallocate), but if empirics disagree the horizon rescale can be re-introduced as a
  fourth constant. This is the single most likely place the "horizon falls away" hope fails.
- τ → 0 at fully settled nodes makes exp(q̃/τ) numerically delicate: compute in log-space
  with max-subtraction (standard), and floor τ at τ_min = k·σ_min via the σ̂ clamp.
- The one-batch-stale π̃ weights in backup are an approximation; if measurable, refreshing
  π̃ at merge points for large deltas is a cheap escalation.
- Backup dot product includes FPU-imputed (unvisited) children, as ComputeVBar does today
  via its P ≥ 5% rule; with π̃ weights this needs no threshold, but the imputed tail's mass
  should be monitored early in implementation.
