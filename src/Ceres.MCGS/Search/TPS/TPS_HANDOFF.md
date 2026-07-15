# TPS implementation session — handoff brief
*(Base commits: Ceres engine `07c9ce6c`, Ceres.MCGS `af4e76d9`, both on main, NOT pushed. 2026-07-13.)*

## Mission
Implement **Tempered Posterior Search** per `TPS_PROPOSAL.md` (same directory) as a SIBLING
mode alongside CBGPUCT (do not replace it): suggested `ParamsSelect.CBGPUCT_Mode` additions or
a parallel flag, so three-way A/B is possible from day one. Background and pitfalls:
`CBGPUCT_CAMPAIGN_REPORT.md`. Sequencing: backup side first (`ComputeVBar` replacement), then
select, then the PathVisit incremental-backup cache as a pure optimization WITH an
equivalence check against the uncached path.

## Baselines to beat (pentanomial 95%, vs pure-default vanilla `ParamsSelect`)
| baseline | net | npm | Elo |
|---|---|---|---|
| CBGPUCT champion (selC=0.5, selExp=0.6, crossN=0, selShrinkK=1, Horizon=100, gate-free, backup 0.75/0.45/K3) | ~T1_DISTILL_256_10_FP16_TRT | 12000 | +43.8 ± 14.9 |
| same | small net | 15000 | +17.5 ± 14.9 |
| same (and 3 variants) | HOP_SP_C4 | 15000 | **−28…−40 (open failure; the acid test)** |
| BackupOnly (0.75/0.45/K3) | small net | 7000 | +32.5 ± 14.9 |

TPS "works" = ≥ champion on small net @12k AND positive on HOP @15k.

## Testing regime (exact recipe)
- Entry: `MCGSTest.cs` calls `new RRTournamentTestDriver().TestSimple()` (already wired).
- Run: from `src/Ceres.MCGS.TestSuite`:
  `TUNE_...=... dotnet run -c Release --no-build > log 2>&1` (build first, VERIFY 0 errors —
  `dotnet run --no-build` silently uses stale binaries after a failed build).
- Nets: primary `~T1_DISTILL_256_10_FP16_TRT` (default); verification
  `TUNE_NET="HOP_SP_C4_512_15_16H_F2_EG_NLA_D_5bn_fp16_ema_4599988224|bf16=true"`.
- Stop rules (driver-implemented): `TUNE_ERRBAR` (pentanomial 95% half-width, 0=off),
  `TUNE_FUTILITY` (stop when Elo+margin < floor; use ~ +10 below current champion to cut
  losers early; NaN=off), `TUNE_ACCEPT` (stop when Elo−margin > floor), `TUNE_MAXPAIRS`,
  `TUNE_MINPAIRS` (default 25). Openings are SEQUENTIAL (paired, same set every run) —
  cross-run comparisons are low-variance; do not switch to randomized for A/B tuning.
- Calibration (6 workers on 0,1,2): error constant k ≈ 230 Elo·sqrt(pairs) on small net
  (±15 ≈ 235 pairs). Pace: small net ≈ 8.6 s/pair @12k, ≈ 10.6 @15k (≈ 35–45 min to ±15);
  HOP ≈ 2.2 min/pair @15k (75-pair cap ≈ 2.5 h, ±≈27) — use HOP runs as capped sign checks.
- Console `BLUNDER:` lines appear only at npm > 12,500 (minilog gate npm×80 > 1M) — they are
  diagnostics, not errors; Test-vs-Ref blunder counts are a useful asymmetry signal.
- Instrumentation: `CERES_CBGPUCT_STATS=file.csv` samples per-node uncertainty stats
  (~1/1024 calls) from the CBGPUCT select path; `analyze`-style binning script pattern in the
  campaign scratchpad; consider adding an equivalent TPS-side sampler.
- Watch logs for `Internal error: search ended` (in-flight leak indicator — must stay 0) and
  `Exception`.

## Key engine code entry points
- `Search/PUCT/CBGPUCTScoreCalc.cs` — ScoreCalc (select), ComputeVBar (backup),
  ComputeLambdaN* wrappers, ApplyRPOImputedFPU anchor helpers, stats sampler.
- `Search/Params/ParamsSelect.cs` — all CBGPUCT_* knobs (post-cleanup surface).
- `Search/Strategies/MCGSStrategyPUCT.cs:BackupToNode`, `QRecomputeHelper`,
  `BottomUpQRecalculator`, `SelectiveQPropagator` — the four backup sites (all funnel
  through ComputeVBar).
- `Search/Paths/MCGSPathVisit.cs` — the transient per-edge structure for the TPS cache.
- `Utils/RunningStdDevShort.cs`, `GNode.LeafValueVolatilityDebiased` — the noise oracle;
  enable via `ParamsSearch.TrackLeafValueVolatility` (behavior-neutral, tracking only).
- FPU imputation (q^FPU per child): `PUCTSelector.ApplyRPOImputedFPU` /
  `CBGPUCTScoreCalc.ComputeImputationAnchor`.

## Coefficient starting points (from proposal §2)
k_s ≈ 2–3 (match λ_select(50)≈0.07 at σ̄≈0.025), k_b ≈ 1, N₀ ≈ 50–100, σ₀ ≈ 0.03.
Validate σ̂ clamps [0.005, 0.5] before trusting temperatures.

## End-of-project deliverable (dje request)
If TPS works: delete the excess CBGPUCT machinery, then produce a patch RELATIVE TO THE
CURRENT MAIN TIPS (`07c9ce6c` engine / `af4e76d9` harness):
`git diff 07c9ce6c..HEAD > tps_engine.patch` (and analogously for the harness repo), or
`git format-patch 07c9ce6c..HEAD` for commit-preserving mailbox patches. Work on a branch or
directly on main — either way DO NOT PUSH; the SHAs above are the agreed patch base.
