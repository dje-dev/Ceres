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
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;


#endregion

namespace Ceres.MCTS.Managers.Limits
{
  /// <summary>
  /// Manager of time which estimates the optimal amount of time
  /// to spend on the next move (default Ceres version)
  /// </summary>
  [Serializable]
  public class ManagerGameLimitCeres : IManagerGameLimit
  {
    public readonly float Aggressiveness;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="aggressiveness"></param>
    public ManagerGameLimitCeres(float aggressiveness = 1.0f)
    {
      Aggressiveness = aggressiveness;
    }

    /// <summary>
    /// Method called periodically to determine if an instamove would be
    /// acceptable given the current state.
    /// </summary>
    /// <param name="search"></param>
    /// <param name="inputs"></param>
    /// <returns></returns>
    public LimitsManagerInstamoveDecision CheckInstamove(MCTSearch search, MCTSNode newRoot, ManagerGameLimitInputs inputs)
    {
      if (inputs != null)
      {
        // Veto any possible instamove if the tree from the new root would be 
        // small relative to the average over a trailing few moves, 
        // to make sure any move decision is based sufficiently large tree.
        // Never veto if only very little time remains because
        // instamoves may be particularly helpful to avoid fixed overhead.
        const float LOW_TIME_THRESHOLD_SECS = 3;
        const int FINAL_N_LOOKBACK_MOVES = 5;
        const float MIN_RELATIVE_TREE_SIZE = 0.6f;
        bool isLowTime = SearchLimit.TypeIsTimeLimit(inputs.TargetLimitType) && inputs.RemainingFixedSelf < LOW_TIME_THRESHOLD_SECS;
        float THRESHOLD_VETO_INSTAMOVE = isLowTime ? (0.5f * MIN_RELATIVE_TREE_SIZE) : MIN_RELATIVE_TREE_SIZE;

        if (search.Manager.PriorMoveStats.Count >= FINAL_N_LOOKBACK_MOVES * 2)
        {
          float avgPriorFinalN = inputs.TrailingAvgFinalNodes(FINAL_N_LOOKBACK_MOVES, newRoot.SideToMove);

          if (newRoot.N < avgPriorFinalN * THRESHOLD_VETO_INSTAMOVE)
          {
            return LimitsManagerInstamoveDecision.DoNotInstamove;
          }
        }
      }

      return LimitsManagerInstamoveDecision.NoDecision;
    }


    /// <summary>
    /// Determines what fraction of the base move should
    /// be consumed for this move.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    float FractionOfBasePerMoveToUse(ManagerGameLimitInputs inputs)
    {
      // When we are behind then it's worth taking a gamble and using more time
      // but when we are ahead, take a little less time to be sure we don't err in time pressure.
      float factorWinningness = inputs.RootQ switch
      {
        < -0.65f => 1.08f,
        < -0.55f => 1.04f,
        > 0.65f => 0.90f,
        > 0.55f => 0.95f,
        _ => 1.0f
      };


      // Spend 50% more on first move of game (definitely no tree reuse, etc.)
      float factorFirstMove = inputs.IsFirstMoveOfGame ? 1.50f : 1.0f;

      // Make a divisor which is between about 11 and 17
      // and a incresing function of the piece count.
      // Note that this is a relatively small number because
      //  - some moves will not do any search at all (due to instamoves), and
      //  - many moves will not actually run the full search duration (due to smart pruning)
      //  - thinking time is deliberately somewhat frontloaded because 
      //    its value as a deferred asset must be discounted by the possibility
      //    that it might never be gainfully used (if a loss comes first).
      float baseDivisor = 9 + MathF.Pow(inputs.StartPos.PieceCount, 0.5f);

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
      const float BASE_MULTIPLIER = 0.68f;

      float ret = Aggressiveness * BASE_MULTIPLIER * (1.0f / baseDivisor) * factorWinningness * factorFirstMove;

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

    /// Determines how much time or nodes resource to
    /// allocate to the the current move in a game subject to
    /// a limit on total numbrer of time or nodes over 
    /// some number of moves (or possibly all moves).
    public ManagerGameLimitOutputs ComputeMoveAllocation(ManagerGameLimitInputs inputs)
    {
      ManagerGameLimitOutputs Return(float value, float extensionFraction)
        => new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType, value,
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

      float baseTimeToUse = inputs.RemainingFixedSelf * FractionOfBasePerMoveToUse(inputs);

      float fractionOfIncrementToUse = 0;
      if (inputs.IncrementSelf > 0)
      {
        float numIncrementsAvailableTime = inputs.RemainingFixedSelf / inputs.IncrementSelf;
        fractionOfIncrementToUse = numIncrementsAvailableTime > 3 ? 0.95f : 0.80f;
      }

      // Try to use almost all of the increment plus part of base time remaining.
      float totalTimeToUse = baseTimeToUse + inputs.IncrementSelf * fractionOfIncrementToUse;

      // But never spend more than 35% of fixed time remaining
      // (since the increment is not earned until after the move is made).
      return Return(MathF.Min(totalTimeToUse, inputs.RemainingFixedSelf * 0.35f), EXTENSION_FRACTION_NORMAL);
    }

  }
}
