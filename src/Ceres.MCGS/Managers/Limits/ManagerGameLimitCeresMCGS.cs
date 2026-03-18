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
  /// Constructor.
  /// </summary>
  /// <param name="aggressiveness"></param>
  public ManagerGameLimitCeresMCGS(float aggressiveness = 1.0f)
  {
    Aggressiveness = aggressiveness;
  }


  /// <summary>
  /// Determines what fraction of the base move should
  /// be consumed for this move.
  /// </summary>
  /// <param name="inputs"></param>
  /// <returns></returns>
  float FractionOfBasePerMoveToUse(ManagerGameLimitInputs inputs, bool earlyGameExtension)
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

    // When we are behind then it's worth taking a gamble and using more time
    // but when we are ahead, take a little less time to be sure we don't err in time pressure.
    float factorWinningness = inputs.RootQ switch
    {
      // 0.55 is about 125, 0.75 is about 190
      < -0.75f => 1.10f,
      < -0.50f => 1.05f,
      > 0.75f => 0.90f,
      > 0.50f => 0.95f,
      _ => 1.0f
    };


    // Spend 2.5x time first move of game (definitely no graph reuse available)
    float factorFirstMove = inputs.IsFirstMoveOfGame ? 2.5f : 1.0f;

    // Make a divisor which is between about 13 and 18
    // and a increasing function of the piece count.
    // Note that this is a relatively small number because
    //  - some moves will not do any search at all (due to instamoves), and
    //  - many moves will not actually run the full search duration (due to smart pruning)
    //  - thinking time is deliberately somewhat frontloaded because 
    //    its value as a deferred asset must be discounted by the possibility
    //    that it might never be gainfully used (if a loss comes first).
    float baseDivisor = 10f + MathF.Pow(inputs.StartPos.PieceCount, 0.5f);

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


  /// Determines how much time or nodes resource to
  /// allocate to the the current move in a game subject to
  /// a limit on total number of time or nodes over 
  /// some number of moves (or possibly all moves).
  public ManagerGameLimitOutputs ComputeMoveAllocation(ManagerGameLimitInputs inputs)
  {
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

    ManagerGameLimitOutputs Return(float value, float extensionFraction)
      => new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType,
                                                     SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType) ? Math.Max(1, value) : value,
                                                     fractionExtensibleIfNeeded: extensionFraction,
                                                     maxTreeNodes: inputs.MaxTreeNodesSelf,
                                                     maxTreeVisits: inputs.MaxTreeVisitsSelf));

    // If this is the last move to go, use almost all available time.
    // TODO: but can a player carry forward time on a clock? Then this doesn't make sense.
    if (inputs.MaxMovesToGo.HasValue && inputs.MaxMovesToGo < 2)
    {
      return Return(inputs.RemainingFixedSelf * 0.98f, 0);
    }

    float incrementMeaningfulThreshold = SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType) ? 1 : 0.01f;
    bool hasMeaningfulIncrement = inputs.IncrementSelf > incrementMeaningfulThreshold;

    if (Panic(inputs))
    {
      float multiplier = hasMeaningfulIncrement ? 0.50f : 0.01f;
      return Return(inputs.RemainingFixedSelf * multiplier, EXTENSION_FRACTION_PANIC);
    }
    else if (NearExhaustion(inputs))
    {
      float multiplier = hasMeaningfulIncrement ? 0.70f : 0.03f;
      return Return(inputs.RemainingFixedSelf * multiplier, EXTENSION_FRACTION_NEAR_EXHAUSTION);
    }

    // Note that per the UCI specification, the RemainingFixedSelf already includes any increment.
    float remainingExcludingIncrement = inputs.RemainingFixedSelf - inputs.IncrementSelf;
    float baseValueToUse = remainingExcludingIncrement * FractionOfBasePerMoveToUse(inputs, earlyGameExtensionMode);

    float fractionOfIncrementToUse = 0;
    if (inputs.IncrementSelf > 0)
    {
      float numIncrementsAvailableTime = remainingExcludingIncrement / inputs.IncrementSelf;

      // Possibly use a lower fraction of the increment if little left in reserve.
      fractionOfIncrementToUse = numIncrementsAvailableTime switch
      {
        < 0.0f => 0.05f, // possibly already in technical forfeit!
        < 1.0f => 0.50f, 
        < 2.0f => 0.90f, 
        < 3.0f => 0.96f, // if at least 2 increments are available we don't need to hold much back
        _ => 0.99f,
      };
    }

    // Try to use almost all of the increment plus part of base time remaining.
    float totalValueUse = baseValueToUse + inputs.IncrementSelf * fractionOfIncrementToUse;

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

    // But never spend more than 35% of fixed time remaining
    // (since the increment is not earned until after the move is made).
    return Return(MathF.Min(totalValueUse, inputs.RemainingFixedSelf * 0.35f), EXTENSION_FRACTION_NORMAL);
  }
}
