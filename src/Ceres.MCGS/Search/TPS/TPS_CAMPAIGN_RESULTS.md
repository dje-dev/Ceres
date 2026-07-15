# TPS Implementation Campaign — Results & Conclusions (2026-07-13)

Implementation of `TPS_PROPOSAL.md` per `TPS_HANDOFF.md`, one session. All Elo pentanomial
95%, sequential paired UHO openings, vs pure-default vanilla `ParamsSelect`, small net
`~T1_DISTILL_256_10_FP16_TRT` unless noted. Success criteria were: ≥ champion (+43.8) on
small net @12k AND positive on HOP @15k. **Not met** — but the backup half of the thesis
is validated, and the select half produced sharp, quantified negative results.

## What was built (all committed locally, NOT pushed; bases 07c9ce6c / af4e76d9)

`TPSScoreCalc.cs` sibling to CBGPUCT (modes `TPSBackupOnly / TPSSelectOnly /
TPSSelectAndBackup / CBGPUCTSelectTPSBackup`), sharing one kernel: per-child robust Q
(inverse-variance shrinkage toward consensus-anchored policy-imputed FPU,
w = σ₀²/(σ₀²+σ̂²), σ̂ = s/√(n+1) from the 2-byte volatility tracker), node noise scale σ̄
(support-weighted median), closed-form posteriors. Sampler: `CERES_TPS_STATS` (incl.
per-node vTPS-vs-vVanilla bias columns). Harness: `TUNE_TPS_KS/KB/HORIZON/SIGMA0`.
Engine commits 0217a549, 8d90135b, 02a2b5f9, 4b89262d; harness 120c7560, d34ab815, 65aaa475.

## Results

| config | @2k | @7k | @12k | HOP @15k |
|---|---|---|---|---|
| **TPS BackupOnly (tuned σ₀=0.10, k_b=0.25, β=1)** | **+24.7 ± 14.9** | +4.5 ± 14.8 | — | **−9.3 ± 27.2, blunders 8:15 IN FAVOR** |
| TPS BackupOnly (proposal defaults) | — | −102 ± 38 | — | — |
| TPS S&B, exp-tilt select (k_s 2.5→20) | −107 → −290 | −127 | — | — |
| TPS S&B, reverse-KL noise-λ select (k_s opt=2.5) | −27.7 ± 14.8 | −40.8 ± 18.4 | — | — |
| Hybrid: champion CBGPUCT select + TPS backup | +27.3 ± 14.8 | — | −6.6 ± 17.3 | **−70.4 ± 27.8** |
| *Reference: CBGPUCT BackupOnly* | *+43.5* | *+32.5* | — | *−33.4, blunders 2-3× against* |
| *Reference: CBGPUCT champion S&B* | — | *+49.9* | *+43.8* | *−30…−40* |

## Conclusions

1. **The core thesis is validated for BACKUP.** Noise-referenced aggregation transfers to
   the regime where every N-denominated constant failed: on HOP @15k TPS backup is −9 ± 27
   with the blunder asymmetry *reversed* (8:15 in TPS's favor vs CBGPUCT's 25:9 against).
   It is the only configuration family tested to date that survives the large-net/deep
   acid test. On the small net it is real but weaker than CBGPUCT V_bar (+24.7 vs +43.5 @2k).
2. **Two proposal constants were materially wrong, with clean quantifications.**
   σ₀=0.03 ⇒ K_eff = (s/σ₀)² ≈ 11–25, i.e. 4–8× over-shrinkage vs the validated K=3
   (fix: σ₀=0.10, interior optimum). The winner's-curse temperature k_b≈1 was ~4× too warm
   (fix: k_b=0.25, interior optimum). And β(N) prior-anchor decay is harmful in backup —
   the backup posterior must stay policy-PROPORTIONAL (β=1) at all N (15–45 Elo).
3. **Exponential tilting is the wrong family for SELECT — in principle, not just in tuning.**
   No temperature works (cold → one-hot greedy, measured π̃ entropy p50 = 0.07 at N ≥ 500;
   warm → policy-flat flooding; −107…−290 across k_s 2.5–20). Cumulative visit targets need
   reverse-KL's polynomial tails (the Grill/PUCT correspondence), exactly as the campaign's
   G-phase hinted.
4. **Noise-referenced λ is also wrong for SELECT — the N-profile, not the scale.** The
   champion's λ(N) is nearly flat (0.034–0.065 over N=5→8000; the log-growth term cancels
   the schedule decay) while k_s·σ̄ falls ~15×: too warm at small N, too cold at large N,
   and no scalar k_s fixes a shape mismatch (best −27.7 @2k). Hypothesis for future work:
   select temperature should reference an ~N-invariant, net-calibrated scale — the child
   VALUE DISPERSION (the stakes) — rather than estimation noise (which vanishes with visits).
   Estimation noise is the right currency for how much to *trust* values (backup); the value
   spread is the right currency for how much to *explore* (select).
5. **The hybrid (champion select + TPS backup) fails by destructive interaction** — worse on
   HOP (−70) than either half alone (−9 backup-only / −30 CBGPUCT S&B), and select adds ~0
   at 2k. Champion select's tuning is evidently coupled to the CBGPUCT-backup Q field:
   deficit-targeting select chases TPS's measurably sharper/optimistic Q (bias +0.02–0.04
   vs visit-weighted, all N bands) and self-reinforces, where vanilla PUCT's 1/(1+n) decay
   damps that loop. Untested rescue: retune selC/horizon against the TPS Q field.
5b. **Ablation (dje follow-up): the empirical variance IS the transfer ingredient.**
   Freezing s to a global constant chosen to preserve the tuned config's *average*
   shrinkage (`CERES_TPS_FREEZE_S=0.12`, K_eff ≈ 1.4 — removing only the adaptivity)
   collapses HOP @15k from −9.3 ± 27.2 (blunders 8:15 in favor) to **−34.9 ± 24.1
   (blunders 23:13 against)** — precisely CBGPUCT's failure level and blunder signature.
   The functional form alone does not transfer. Mechanism: HOP's measured leaf noise is
   ~2× lower than the small net's at matched N (σ̄ p50 0.069→0.029 at N<10), so live
   measurement correctly shrinks less / stays colder there; any fixed constant
   over-regularizes. Caveat: 75 pairs (Δ≈25 Elo within overlapping bars alone), but
   Elo level + blunder flip + signature match are three concordant signals.

6. **Open problems, ranked:** (a) backup small-net depth decay (+24.7 @2k → +4.5 @7k) —
   prime suspect is the σ̂ clamp floor turning τ_backup into hard-max at large N (levers:
   clamp value, per-child uncertainty penalty in the tilt, top-contender-referenced τ);
   (b) dispersion-referenced select λ (one kernel change, untested); (c) hybrid retune.

## Consolidation (2026-07-14, engine commit fd1abd89)

Per dje direction, the tree was consolidated to the validated core: **all CBGPUCT
machinery and both TPS select designs deleted** (−3,590 lines engine-side; everything
recoverable from git history, champion baselines at commits `4021df69..4b89262d`).
Surviving surface: `ParamsSelect.TPS_Mode` (`None` = standard Ceres PUCT, byte-identical;
`BackupOnly` = the tuned TPS backup; `SelectOnly`/`SelectAndBackup` = placeholders that
throw pending the new SELECT algorithm) plus exactly two knobs
(`TPS_BackupTemperatureK=0.25`, `TPS_ShrinkageSigma0=0.10`). FPU logic unchanged
(helpers rehomed to `RPOImputation`; `FPU_QAnchorType` renamed off the CBGPUCT prefix).
Measured volatility always on for BackupOnly (`CERES_TPS_FREEZE_S` kept as ablation gate);
console diagnostics retained (FPU dump, per-node TPS backup dump + windowed
V-vs-vanilla bucket stats, `DEBUG_TPS` deviation trace). Driver slimmed to
`TUNE_TEST_MODE/TUNE_REF_MODE` (TPSModeType names) + `TUNE_TPS_KB`/`TUNE_TPS_SIGMA0`.

**Post-consolidation regression:** BackupOnly vs None @2k = **+26.4 ± 14.8 (LOS 100.0%,
257 pairs, converged, relTime +8.4%)** — statistically identical to the pre-refactor
+24.7 ± 14.9; select-mode attempts throw cleanly; vanilla smoke clean; zero
in-flight-leak warnings.

## Operational notes
- Mechanical health: zero in-flight leaks and zero exceptions across every TPS run
  (select allocator, hole fixup, and all five backup sites).
- TPS backup time overhead ~+8–12% (no solver); reverse-KL TPS select ~+3–9%.
