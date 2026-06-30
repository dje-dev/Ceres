#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directives

using System;

using Ceres.Chess;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Managers.Limits;

/// <summary>
/// Manager of time which estimates the optimal amount of time
/// to spend on the next move (default Ceres version)
/// </summary>
[Serializable]
public class ManagerGameLimitCeresMCGS : IManagerGameLimit
{
  public readonly float Aggressiveness;

  /// <summary>
  /// If set, every ComputeMoveAllocation builds a diagnostic string (attached to the outputs)
  /// describing how the allocation was derived, so it can be recorded in the game move-log.
  /// Independent of MCGSParamsFixed.DUMP_LIMIT_CALC, which separately governs console dumping.
  /// </summary>
  public bool CaptureDiagnostics { get; set; }

  const int EARLY_SMOOTHING_MAX_ADJUSTS = 3;
  const float EARLY_SMOOTHING_BOOST = 1.3f;

  // Courtesy speedup: in clearly-won positions, deliberately spend less time
  // (play faster) when the clock has ample time. This is a courtesy for
  // tournaments that require playing on to mate. Thresholds are expressed in Q
  // (not centipawns) so behavior is invariant to future changes in the cp<->Q
  // display mapping. The Q values are the inverse of the current
  // EncodedEvalLogistic.LogisticToCentipawn mapping (cp = 90*tan(1.5637541897*Q)):
  //   +400cp -> Q = 0.8630,   +700cp -> Q = 0.9227.
  // (Hard-coded rather than computed via EncodedEvalLogistic.CentipawnToLogistic
  //  at runtime so these thresholds stay fixed even if that mapping is later retuned.)
  const float COURTESY_Q_400CP = 0.8630f;
  const float COURTESY_Q_700CP = 0.9227f;
  const float COURTESY_MULT_400CP = 0.80f;
  const float COURTESY_MULT_700CP = 0.60f;
  const float COURTESY_MIN_CLOCK_SECONDS = 180f; // 3 minutes, excluding increment


  /// <summary>
  /// Returns target minimum fraction of initial move nodes in early game.
  /// Expectations are lower for games with smaller searches 
  /// more suboptimal moves and therefore less graph reuse 
  /// and therefore also smaller accumulated graph sizes will be expected.
  /// </summary>
  /// <param name="baselineN"></param>
  /// <returns></returns>
  static float EarlySmoothingBaselineFracForBaselineN(int baselineN)
    => baselineN switch
    {
      < 50_000 => 0, // disabled
      < 150_000 => 0.20f,
      < 500_000 => 0.25f,
      < 2_000_000 => 0.30f,
      < 5_000_000 => 0.35f,
      _ => 0.40f
    };


  /// <summary>
  /// Returns the number of moves in the early game smoothing window, 
  /// as a function of the baselineN.
  /// </summary>
  /// <param name="baselineN"></param>
  /// <returns></returns>
  static int EarlySmoothingWindowMoves(int baselineN)
    => baselineN switch
    {
      < 150_000 => 12,
      < 500_000 => 15,
      < 2_000_000 => 20,
      < 5_000_000 => 24,
      _ => 25
    };

  // Per-game state for GameLimitEarlySmoothing. Reset on IsFirstMoveOfGame == true,
  // so the manager can be safely reused across multiple games in self-play tournaments.
  int earlySmoothingBaselineN = -1;     // FinalN of the first-move-bonus search (once recorded)
  bool earlySmoothingRecordPending = false;  // armed on move 1, consumed on move 2
  int earlySmoothingMovesRemaining = 0;      // counts down from EARLY_SMOOTHING_WINDOW_MOVES
  int earlySmoothingAdjustsApplied = 0;      // capped at EARLY_SMOOTHING_MAX_ADJUSTS

  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="aggressiveness"></param>
  public ManagerGameLimitCeresMCGS(float aggressiveness = 1.0f)
  {
    Aggressiveness = aggressiveness;
  }


  /// <summary>
  /// Mutable bag of intermediate values captured during a single ComputeMoveAllocation
  /// call when MCGSParamsFixed.DUMP_LIMIT_CALC is enabled, used to emit a comprehensive
  /// per-move diagnostic line. Fields left at their float.NaN sentinel were not reached
  /// on the return path actually taken (e.g. the per-move factor breakdown is skipped on
  /// the last-move / panic / near-exhaustion shortcuts).
  /// </summary>
  sealed class LimitCalcTrace
  {
    // FractionOfBasePerMoveToUse breakdown.
    public float GraphReuseShrinkage = float.NaN;
    public float FactorWinningness   = float.NaN;
    public bool  CourtesyApplied;
    public float FactorFirstMove     = float.NaN;
    public float StartDivisor        = float.NaN;
    public float BaseDivisor         = float.NaN;
    public float BaseMultiplier      = float.NaN;
    public bool  EarlyGameExtension;
    public float FractionOfBase      = float.NaN;

    // ComputeMoveAllocation breakdown.
    public float BaseValueToUse          = float.NaN;
    public float FractionOfIncrementToUse = float.NaN;
    public float TotalValuePreSmoothing  = float.NaN;
    public bool  EarlySmoothingBoostApplied;
    public string Path = "normal";
    public float ExtensionFraction = float.NaN;
    public bool  ClampedByMaxFraction;
  }


  /// <summary>
  /// Determines what fraction of the base move should
  /// be consumed for this move.
  /// </summary>
  /// <param name="inputs"></param>
  /// <returns></returns>
  float FractionOfBasePerMoveToUse(ManagerGameLimitInputs inputs, bool earlyGameExtension, LimitCalcTrace trace = null)
  {
    int priorRootN = inputs.PriorMoveStats != null && inputs.PriorMoveStats.Count > 1 ? inputs.PriorMoveStats[^2].FinalN : 0;
    int thisRootN = inputs.RootN;
    float graphReuseShrinkageMultiplier = 1.0f;
    if (inputs.QuickMoveEnabled && priorRootN != 0 && thisRootN != 0)
    {
      float fracNodesRetainedSinceLastMove = (float)thisRootN / priorRootN;
      graphReuseShrinkageMultiplier = MapFractionGraphReusedToShrinkageMultiplier(fracNodesRetainedSinceLastMove);
      //Console.WriteLine(treeReuseShrinkageMultiplier + "  " + fracNodesRetainedSinceLastMove);
    }

    // TODO: Someday implement this idea to prevent too many in a row.
    //       For this, need to make this ManagerGameLimitCeresMCGS to be held at
    //       GameEngineCeresMCGSInProcess (like the NNEvaluatorSet) is 
    //       and keep one for a whole game (this is a better design).
#if NOT
// FIELD      float runningAverageTreeReuseShrinkage = 1.0f;
    runningAverageTreeReuseShrinkage = runningAverageTreeReuseShrinkage * 0.65f 
                                     + treeReuseShrinkageMultiplier * 0.35f;
    if (runningAverageTreeReuseShrinkage < 0.5f)
    {
      treeReuseShrinkageMultiplier = MathF.Max(0.5f, treeReuseShrinkageMultiplier);
      Console.WriteLine("ADJUST to " + treeReuseShrinkageMultiplier + " because of running " + runningAverageTreeReuseShrinkage);
    }
#endif

    const bool VERBOSE = false;
    if (VERBOSE)
    {
      Console.WriteLine($"{graphReuseShrinkageMultiplier,5:F2}" + " " + priorRootN + " -> " + thisRootN + "  (+" + (thisRootN - priorRootN) + ")");
    }

    // Courtesy speedup gate: only for time-based play, and only with at least
    // 3 minutes on the clock (excluding increment) so this can never cause time
    // trouble. RemainingFixedSelf already includes any increment (per UCI), so
    // subtract it to get the clock time "not counting increment".
    float remainingExcludingIncrement = inputs.RemainingFixedSelf - inputs.IncrementSelf;
    bool courtesyEligible = inputs.TargetLimitType == SearchLimitType.SecondsPerMove
                         && remainingExcludingIncrement >= COURTESY_MIN_CLOCK_SECONDS;

    // When we are behind then it's worth taking a gamble and using more time
    // but when we are ahead, take a little less time to be sure we don't err in time pressure.
    float factorWinningness;
    bool courtesyApplied = false;
    if (courtesyEligible && inputs.RootQ >= COURTESY_Q_700CP)
    {
      factorWinningness = COURTESY_MULT_700CP;  // absolutely won (~+700cp): play faster
      courtesyApplied = true;
    }
    else if (courtesyEligible && inputs.RootQ >= COURTESY_Q_400CP)
    {
      factorWinningness = COURTESY_MULT_400CP;  // clearly won (~+400cp): play a little faster
      courtesyApplied = true;
    }
    else
    {
      factorWinningness = inputs.RootQ switch
      {
        // 0.55 is about 125, 0.75 is about 190
        < -0.75f => 1.10f,
        < -0.50f => 1.05f,
        > 0.75f => 0.90f,
        > 0.50f => 0.95f,
        _ => 1.0f
      };
    }


    // Spend 2.5x time first move of game (definitely no graph reuse available)
    float factorFirstMove = inputs.IsFirstMoveOfGame ? 2.5f : 1.0f;

    // Make a divisor which is between about 10 and 12.
    // For low absolute time (seconds) we expect less graph reuse
    // and also higher possibility of very suboptimal moves if time allowed to fall too low.
    // Therefore we are more conservative with time usage (higher divisor).
    // The actual time usage is highly sensitive to this value.
    float startDivisor = (inputs.TargetLimitType, inputs.RemainingFixedSelf) switch
    {
      (not SearchLimitType.SecondsPerMove, _) => 10f,
      (_, <= 60) => 12f,
      (_, >= 180) => 10f,
      (_, var t) => 12f - (t - 60f) / 60f, // linear ramp between 60s and 180s
    };

    // Now make also an increasing function of the piece count.
    // Note that this is a relatively small number because
    //  - some moves will not do any search at all (due to instamoves), and
    //  - many moves will not actually run the full search duration (due to smart pruning)
    //  - thinking time is deliberately somewhat frontloaded because 
    //    its value as a deferred asset must be discounted by the possibility
    //    that it might never be gainfully used (if a loss comes first).
    float baseDivisor = startDivisor + MathF.Pow(inputs.StartPos.PieceCount, 0.5f);

    if (inputs.IncrementSelf > 0)
    {
      const float MULTIPLIER = 200f;
      float fractionIncrementOfRemaining = inputs.IncrementSelf / inputs.RemainingFixedSelf;
      float adj = fractionIncrementOfRemaining * MULTIPLIER;
      adj = MathF.Min(8, adj);
      baseDivisor -= adj;
    }

    // This is a key scaling factor controlling aggressiveness.
    // Small changes can induce significant differences because they compound over time.
    // Values of 0.75 or even higher may perform well for short games and/or weak nets
    // because games are often decided early on missed tactics. 
    // But for longer games (e.g. 3 to 5 minutes) somewhat lower values seem better.
    // Extensive tests at (300 + 5) suggested perhaps 0.70 optimal against LC0, 0.67 against Stockfish.

    // In observing LTC games vs Stockfish, Ceres seemed to be extremely conservative in clock use.
    // Therefore an adjustment was made to use more time in the early game (first 40% of time used).
    // However even in this early phases, extensive testing (versus SF) showed that
    // large increases in time spent are Elo negative; only modest increases are slightly helpful (5 to 10 Elo).
    const float BASE_MULTIPLIER_EARLY = 0.72f; // use more time early in game
    const float BASE_MULTIPLIER_NOT_EARLY = 0.70f;

    float adjustedBaseMultiplier = earlyGameExtension ? BASE_MULTIPLIER_EARLY : BASE_MULTIPLIER_NOT_EARLY;
    float ret = Aggressiveness
              * graphReuseShrinkageMultiplier
              * adjustedBaseMultiplier
              * (1.0f / baseDivisor)
              * factorWinningness
              * factorFirstMove;

    if (trace != null)
    {
      trace.GraphReuseShrinkage = graphReuseShrinkageMultiplier;
      trace.FactorWinningness   = factorWinningness;
      trace.CourtesyApplied     = courtesyApplied;
      trace.FactorFirstMove     = factorFirstMove;
      trace.StartDivisor        = startDivisor;
      trace.BaseDivisor         = baseDivisor;
      trace.BaseMultiplier      = adjustedBaseMultiplier;
      trace.EarlyGameExtension  = earlyGameExtension;
      trace.FractionOfBase      = ret;
    }

    return ret;
  }

  static bool Panic(ManagerGameLimitInputs inputs)
  {
    return inputs.TargetLimitType == SearchLimitType.NodesForAllMoves
                                   ? inputs.RemainingFixedSelf + inputs.IncrementSelf < 50
                                   : (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 0.25;
  }

  static bool NearExhaustion(ManagerGameLimitInputs inputs)
  {
    return inputs.TargetLimitType == SearchLimitType.NodesForAllMoves
                                   ? (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 200
                                   : (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 1;
  }


  // Amount of potential dynamic search extensions is gated by degree of time pressure.
  const float EXTENSION_FRACTION_PANIC = 0.0f;
  const float EXTENSION_FRACTION_NEAR_EXHAUSTION = 0.2f;
  const float EXTENSION_FRACTION_NORMAL = 0.6f;

  // Hard ceiling on a single move's allocation as a fraction of remaining fixed time, applied as a
  // final anti-flag safety. This is the dominant lever on the low-clock equilibrium in Fischer play:
  // the clock parks near (increment / this fraction). At 0.50 a 1.5s increment parks near ~3s.
  const float MAX_FRACTION_REMAINING_PER_MOVE = 0.50f;


  /// <summary>
  /// Returns the shrinkage multiplier to be applied to limits allocation
  /// based on the fraction of the graph which is being retained from prior position.
  /// If for example the full graph size was retained (input = 1.0) 
  /// allocation would be cut to = 45%.
  /// </summary>
  /// <param name="fracReused"></param>
  /// <returns></returns>
  public static float MapFractionGraphReusedToShrinkageMultiplier(float fracReused) =>
    fracReused switch
    {
      // Reuse is less than 70%, no shrinkage of allocation.
      < 0.7f => 1.0f,

      // Linear slope to 0.45 shrinkage (when full reuse).
      <= 1.0f => -1.5f * fracReused + 1.95f,

      // Shrink further (to as much as 0.2) if more than full reuse (transposed to well explored node).
      < 2.0f => 0.45f - 0.25f * (fracReused - 1.0f),

      // Allocate 0.20 at minimum.
      _ => 0.20f
    };


  /// <summary>
  /// Maintains GameLimitEarlySmoothing state. Called at the top of ComputeMoveAllocation
  /// when the flag is enabled. Resets on the first move of a game and records the baseline
  /// root N on the move immediately following the first-move-bonus move.
  /// </summary>
  void UpdateEarlySmoothingState(ManagerGameLimitInputs inputs)
  {
    if (inputs.IsFirstMoveOfGame)
    {
      earlySmoothingBaselineN = -1;
      earlySmoothingRecordPending = true;
      earlySmoothingMovesRemaining = 0;
      earlySmoothingAdjustsApplied = 0;
      return;
    }

    if (earlySmoothingRecordPending
        && inputs.PriorMoveStats != null
        && inputs.PriorMoveStats.Count >= 2)
    {
      // PriorMoveStats[^2] is the engine's prior move (the first-move-bonus search).
      int recordedN = inputs.PriorMoveStats[^2].FinalN;

      earlySmoothingBaselineN = recordedN;
      earlySmoothingMovesRemaining = EarlySmoothingWindowMoves(earlySmoothingBaselineN);
      earlySmoothingRecordPending = false;
    }
  }


  /// Determines how much time or nodes resource to
  /// allocate to the the current move in a game subject to
  /// a limit on total number of time or nodes over
  /// some number of moves (or possibly all moves).
  public ManagerGameLimitOutputs ComputeMoveAllocation(ManagerGameLimitInputs inputs, bool applyEarlySmoothing = true)
  {
    // applyEarlySmoothing is false when this is a recomputation for a move whose per-game
    // early-smoothing state was already advanced by an earlier call (e.g. a cold-start
    // reallocation after a reused graph was abandoned). In that case neither update nor apply
    // the early-smoothing state here, to avoid double-counting the window/adjustments.
    if (applyEarlySmoothing && inputs.SearchParams.GameLimitEarlySmoothing)
    {
      UpdateEarlySmoothingState(inputs);
    }

    // Only incur the cost of capturing intermediates / formatting the diagnostic string when it
    // will actually be consumed: console dumping (DUMP_LIMIT_CALC) or move-log capture.
    LimitCalcTrace trace = (MCGSParamsFixed.DUMP_LIMIT_CALC || CaptureDiagnostics) ? new LimitCalcTrace() : null;

    // Check if the early game extension mode should be enabled.
    bool earlyGameExtensionMode = false;
    if (inputs.TargetLimitType == SearchLimitType.SecondsPerMove // TODO: Extend this to NodesPerMove?
     && inputs.PriorMoveStats != null
     && inputs.PriorMoveStats.Count >= 2 // i.e. not first move
        )
    {
      float totalGameTimeAtLeast = inputs.RemainingFixedSelf + inputs.PriorMoveStats[^2].ClockSecondsAlreadyConsumed;
      float fractionUsed = inputs.PriorMoveStats[^2].ClockSecondsAlreadyConsumed / totalGameTimeAtLeast;

      const float EARLY_GAME_FRAC_TIME = 0.4f; // Extension only enabled in first 40% of time used
      if (fractionUsed < EARLY_GAME_FRAC_TIME)
      {
        earlyGameExtensionMode = true;
      }
    }

    ManagerGameLimitOutputs Return(float value, float extensionFraction, string path = "normal")
    {
      float finalValue = SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType) ? Math.Max(1, value) : value;
      var outputs = new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType,
                                                     finalValue,
                                                     fractionExtensibleIfNeeded: extensionFraction,
                                                     maxTreeNodes: inputs.MaxTreeNodesSelf,
                                                     maxTreeVisits: inputs.MaxTreeVisitsSelf));
      if (trace != null)
      {
        trace.Path = path;
        trace.ExtensionFraction = extensionFraction;

        // Build the diagnostic string once. Attach it to the outputs so the move-log can record it,
        // and (only when explicitly enabled) also echo it to the console.
        string diag = BuildLimitCalcString(inputs, trace, finalValue);
        outputs.DiagnosticText = diag;

        if (MCGSParamsFixed.DUMP_LIMIT_CALC)
        {
          ConsoleColor savedColor = Console.ForegroundColor;
          Console.ForegroundColor = ConsoleColor.Cyan;
          Console.WriteLine("\r\n" + diag);
          Console.ForegroundColor = savedColor;
        }
      }
      return outputs;
    }

    // If this is the last move to go, use almost all available time.
    // TODO: but can a player carry forward time on a clock? Then this doesn't make sense.
    if (inputs.MaxMovesToGo.HasValue && inputs.MaxMovesToGo < 2)
    {
      return Return(inputs.RemainingFixedSelf * 0.98f, 0, "last-move");
    }

    float incrementMeaningfulThreshold = SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType) ? 1 : 0.01f;
    bool hasMeaningfulIncrement = inputs.IncrementSelf > incrementMeaningfulThreshold;

    if (Panic(inputs))
    {
      float multiplier = hasMeaningfulIncrement ? 0.50f : 0.01f;
      return Return(inputs.RemainingFixedSelf * multiplier, EXTENSION_FRACTION_PANIC, "panic");
    }
    else if (NearExhaustion(inputs))
    {
      float multiplier = hasMeaningfulIncrement ? 0.70f : 0.03f;
      return Return(inputs.RemainingFixedSelf * multiplier, EXTENSION_FRACTION_NEAR_EXHAUSTION, "near-exhaustion");
    }

    // Note that per the UCI specification, the RemainingFixedSelf already includes any increment.
    float remainingExcludingIncrement = inputs.RemainingFixedSelf - inputs.IncrementSelf;
    float baseValueToUse = remainingExcludingIncrement * FractionOfBasePerMoveToUse(inputs, earlyGameExtensionMode, trace);
    if (trace != null)
    {
      trace.BaseValueToUse = baseValueToUse;
    }

    float fractionOfIncrementToUse = 0;
    if (inputs.IncrementSelf > 0)
    {
      float numIncrementsAvailableTime = remainingExcludingIncrement / inputs.IncrementSelf;
      bool isLowTimeIncrement = inputs.TargetLimitType == SearchLimitType.SecondsPerMove && inputs.IncrementSelf < 0.5f;

      // Possibly use a lower fraction of the increment if little left in reserve.
      // For normal increments we now spend nearly the full increment even when the base-time
      // reserve is small, so in low-clock Fischer play the equilibrium clock parks lower (~2s with
      // a 1.5s increment) rather than ~4s. Tiny increments (isLowTimeIncrement) stay conservative.
      fractionOfIncrementToUse = numIncrementsAvailableTime switch
      {
        < 0.0f => 0.05f, // possibly already in technical forfeit!
        < 1.0f => isLowTimeIncrement ? 0.30f : 0.97f,
        < 2.0f => isLowTimeIncrement ? 0.75f : 0.98f,
        < 3.0f => isLowTimeIncrement ? 0.90f : 0.98f,
        _ => 0.98f,
      };
    }

    // Try to use almost all of the increment plus part of base time remaining.
    float totalValueUse = baseValueToUse + inputs.IncrementSelf * fractionOfIncrementToUse;
    if (trace != null)
    {
      trace.FractionOfIncrementToUse = fractionOfIncrementToUse;
      trace.TotalValuePreSmoothing = totalValueUse;
    }

    // Prevent totalValueFromUse being zero or negative.
    if (SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType))
    {
      totalValueUse = Math.Max(1, totalValueUse);
    }
    else if (SearchLimit.TypeIsTimeLimit(inputs.TargetLimitType))
    {
      const float MIN_TIME = 0.05f; // 50 milliseconds minimum
      totalValueUse = Math.Max(MIN_TIME, totalValueUse);
    }

    // Optionally boost allocation under the GameLimitEarlySmoothing feature.
    if (applyEarlySmoothing && inputs.SearchParams.GameLimitEarlySmoothing && earlySmoothingMovesRemaining > 0)
    {
      earlySmoothingMovesRemaining--;  // consume one slot from the 20-move window

      float earlySmothingBaselineFrac = EarlySmoothingBaselineFracForBaselineN(earlySmoothingBaselineN);
      bool canStillAdjust = earlySmoothingAdjustsApplied < EARLY_SMOOTHING_MAX_ADJUSTS;
      bool baselineValid = earlySmoothingBaselineN > 0;
      bool isEarlyUnsmooth = baselineValid
                          && inputs.RootN < earlySmothingBaselineFrac * earlySmoothingBaselineN;

      if (canStillAdjust && isEarlyUnsmooth)
      {
        earlySmoothingAdjustsApplied++;
        totalValueUse *= EARLY_SMOOTHING_BOOST;
        if (trace != null)
        {
          trace.EarlySmoothingBoostApplied = true;
        }

        // DUMP_EARLY_SMOOTHING_BOOST gates only the diagnostic console output, not the boost itself.
        if (MCGSParamsFixed.DUMP_EARLY_SMOOTHING_BOOST)
        {
          ConsoleColor savedColor = Console.ForegroundColor;
          Console.ForegroundColor = ConsoleColor.Yellow;
          int earlySmoothingMoves = EarlySmoothingWindowMoves(earlySmoothingBaselineN);
          Console.WriteLine($"\r\n[GameLimitEarlySmoothing] boost x{EARLY_SMOOTHING_BOOST:F2}  "
                          + $"RootN={inputs.RootN} < {earlySmothingBaselineFrac:F2}*baseline({earlySmoothingBaselineN})"
                          + $"={earlySmothingBaselineFrac * earlySmoothingBaselineN:F0}  "
                          + $"adj {earlySmoothingAdjustsApplied}/{EARLY_SMOOTHING_MAX_ADJUSTS}  "
                          + $"window {earlySmoothingMoves - earlySmoothingMovesRemaining}/{earlySmoothingMoves}");
          Console.ForegroundColor = savedColor;
        }
      }
    }

    // But never spend more than MAX_FRACTION_REMAINING_PER_MOVE of fixed time remaining
    // (final anti-flag safety; see the constant's note on the resulting low-clock equilibrium).
    float maxFractionCap = inputs.RemainingFixedSelf * MAX_FRACTION_REMAINING_PER_MOVE;
    if (trace != null)
    {
      trace.ClampedByMaxFraction = totalValueUse > maxFractionCap;
    }
    return Return(MathF.Min(totalValueUse, maxFractionCap), EXTENSION_FRACTION_NORMAL);
  }


  /// <summary>
  /// Builds one comprehensive multi-line diagnostic block describing how the allocation for this
  /// move was derived (inputs, intermediate factors, return path, final value). Called only when
  /// diagnostics are being captured (move-log) or dumped (console); never on the hot path otherwise.
  /// </summary>
  string BuildLimitCalcString(ManagerGameLimitInputs inputs, LimitCalcTrace trace, float finalValue)
  {
    bool isNodes = SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType);
    string unit = isNodes ? "nodes" : "sec";

    // Render a float that may be NaN (i.e. not reached on the path taken) as "n/a".
    static string F(float v, string fmt = "F3") => float.IsNaN(v) ? "n/a" : v.ToString(fmt);

    System.Text.StringBuilder sb = new();

    sb.AppendLine($"[LimitCalc] path={trace.Path}  type={inputs.TargetLimitType}  "
                + $"=> ALLOC {F(finalValue)} {unit}  extFrac={F(trace.ExtensionFraction, "F2")}"
                + (trace.ClampedByMaxFraction ? "  [clamped@35%fixed]" : ""));

    sb.AppendLine($"           in: rootN={inputs.RootN} rootQ={F(inputs.RootQ, "F4")} "
                + $"remFixed={F(inputs.RemainingFixedSelf)} incr={F(inputs.IncrementSelf)} "
                + $"pieces={inputs.StartPos.PieceCount} firstMove={inputs.IsFirstMoveOfGame} "
                + $"movesToGo={(inputs.MaxMovesToGo.HasValue ? inputs.MaxMovesToGo.Value.ToString() : "-")} "
                + $"quickMove={inputs.QuickMoveEnabled}");

    sb.AppendLine($"           frac: aggr={Aggressiveness:F3} reuseShrink={F(trace.GraphReuseShrinkage)} "
                + $"baseMult={F(trace.BaseMultiplier)}(early={trace.EarlyGameExtension}) "
                + $"startDiv={F(trace.StartDivisor, "F2")} baseDiv={F(trace.BaseDivisor, "F2")} "
                + $"winFactor={F(trace.FactorWinningness, "F2")}(courtesy={trace.CourtesyApplied}) "
                + $"firstMoveFactor={F(trace.FactorFirstMove, "F2")} => fracOfBase={F(trace.FractionOfBase, "F4")}");

    sb.Append($"           build: baseVal={F(trace.BaseValueToUse)} "
                + $"incrFrac={F(trace.FractionOfIncrementToUse, "F2")} "
                + $"preSmoothTotal={F(trace.TotalValuePreSmoothing)} "
                + $"smoothBoost={trace.EarlySmoothingBoostApplied} "
                + $"[smoothState baselineN={earlySmoothingBaselineN} "
                + $"windowLeft={earlySmoothingMovesRemaining} adj={earlySmoothingAdjustsApplied}]");

    return sb.ToString();
  }
}
