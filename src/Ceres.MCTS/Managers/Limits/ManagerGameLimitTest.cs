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
using System.Collections.Generic;
using System.Linq;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;


#endregion

namespace Ceres.MCTS.Managers.Limits
{
  /// <summary>
  /// Manager of time which estimates the optimal amount of time
  /// to spend on the next move (testbed version).
  /// </summary>
  [Serializable]
  public class ManagerGameLimitTest : IManagerGameLimit
  {
    public readonly float Aggressiveness;
    public float InstaMoveLimit { get; set; } = 0.3f;
    public float TrendMultiplier { get; set; } = 10f;
    public float TrendMultiplierLimit { get; set; } = 0.25f;
    public float QValueMultiplier { get; set; } = 0.12f;
    public float AlphaZeroFraction { get; set; } = 1f / 16;
    public float MaxAttentionLevel { get; set; } = 1.6f;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="aggressiveness"></param>
    public ManagerGameLimitTest(float aggressiveness = 1.0f)
    {
      Aggressiveness = aggressiveness;
    }

    public LimitsManagerInstamoveDecision CheckInstamove(MCTSearch search, MCTSNode node, ManagerGameLimitInputs inputs)
    {
      if (inputs.PriorMoveStats.Count == 0)
      {
        return LimitsManagerInstamoveDecision.NoDecision;
      }

      //don't instamove when Q-value is above InstamoveLimit
      if (Math.Abs(inputs.RootQ) > InstaMoveLimit)
      {
        return LimitsManagerInstamoveDecision.DoNotInstamove;
      }

      //No need to instamove if we have 20% or more time
      GameMoveStat oppLastMove = inputs.PriorMoveStats[^1];
      GameMoveStat firstMove = inputs.PriorMoveStats[0];
      bool isAMoveBehind = firstMove.Side == oppLastMove.Side;
      (float restTimeOpp, float usedLastMoveOpp) = (oppLastMove.SearchLimit.Value, oppLastMove.TimeElapsed);

      //Check if we have more time than the opponent
      float estMoveTime = inputs.RemainingFixedSelf * AlphaZeroFraction + inputs.IncrementSelf;
      float remainingTime = inputs.RemainingFixedSelf + inputs.IncrementSelf;
      float diffTime = isAMoveBehind ?
        (remainingTime - estMoveTime) / restTimeOpp :
        remainingTime / restTimeOpp;

      //do not instamove when we have much more time
      if (diffTime > 1.2f)
      {
        return LimitsManagerInstamoveDecision.DoNotInstamove;
      }

      return LimitsManagerInstamoveDecision.NoDecision;
    }

    public bool CheckStopSearch(MCTSearch search, ManagerGameLimitInputs inputs)
    {
      return false;
    }

    public ManagerGameLimitOutputs ComputeMoveAllocation(ManagerGameLimitInputs inputs)
    {
      ManagerGameLimitOutputs Return(float value, float extensionFraction)
      {
        return new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType, value,
                                                       fractionExtensibleIfNeeded: extensionFraction,
                                                       maxTreeNodes: inputs.MaxTreeNodesSelf,
                                                       maxTreeVisits: inputs.MaxTreeVisitsSelf));
      }

      //games without meaningful increment reduces the fixed time allocation to 1/18 and enable instamoves
      if (inputs.IncrementSelf < 0.1f)
      {
        AlphaZeroFraction = 1f / 18f;
        InstaMoveLimit = 0.9f;
      }
      //allocate normal time usage
      float totalTimeToUse = inputs.IncrementSelf + (inputs.RemainingFixedSelf * AlphaZeroFraction);
      float remainingTime = inputs.RemainingFixedSelf + inputs.IncrementSelf;

      //never use more than 20% of fixed time in one move
      float upperTimeCut = inputs.RemainingFixedSelf * 0.2f + inputs.IncrementSelf * 0.95f;

      //add some extra time for the first move
      if (inputs.IsFirstMoveOfGame)
      {
        float extraTime = remainingTime > 45f ? 1.45f : 1 + (0.01f * remainingTime);
        return Return(totalTimeToUse * extraTime, 0.6f);
      }

      //from legacy time control
      float incrementMeaningfulThreshold = SearchLimit.TypeIsNodesLimit(inputs.TargetLimitType) ? 1 : 0.01f;
      bool hasMeaningfulIncrement = inputs.IncrementSelf > incrementMeaningfulThreshold;

      bool Panic(ManagerGameLimitInputs inputs) =>
        inputs.TargetLimitType == SearchLimitType.NodesForAllMoves
          ? inputs.RemainingFixedSelf + inputs.IncrementSelf < 50
          : (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 0.25;

      bool NearExhaustion(ManagerGameLimitInputs inputs) =>
        inputs.TargetLimitType == SearchLimitType.NodesForAllMoves
          ? (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 200
          : (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 1;

      if (Panic(inputs))
      {
        float multiplier = hasMeaningfulIncrement ? 0.50f : 0.01f;
        return Return(inputs.RemainingFixedSelf * multiplier, 0.0f);
      }

      else if (NearExhaustion(inputs))
      {
        float multiplier = hasMeaningfulIncrement ? 0.70f : 0.03f;
        return Return(inputs.RemainingFixedSelf * multiplier, 0.2f);
      }

      //look at time usage for opponent - especially useful for AB engines
      GameMoveStat oppLastMove = inputs.PriorMoveStats[^1];
      GameMoveStat firstMove = inputs.PriorMoveStats[0];
      bool isAMoveBehind = firstMove.Side == oppLastMove.Side;
      (float restTimeOpp, float usedLastMoveOpp) = (oppLastMove.SearchLimit.Value, oppLastMove.TimeElapsed);

      //do we have more or less time than opponent
      float diffTime = isAMoveBehind ?
        (remainingTime - totalTimeToUse) / restTimeOpp :
        remainingTime / restTimeOpp;

      //against AB engines we look for a long think and increase our own time usage when it occurs
      //check if opponent used more than 20% of his time on the last move
      float fractionLastMove = usedLastMoveOpp / restTimeOpp;
      if (restTimeOpp > 10f && fractionLastMove >= 0.2f)
      {
        float allocatedTime = Math.Min(totalTimeToUse * 2 * diffTime, upperTimeCut);
        //Console.WriteLine($"Opponent did think > 20%: {fractionLastMove:f2} Diff: {diffTime:f2} Ceres rem: {remainingTime:f2} - Time given: {allocatedTime:f2} ");
        return Return(allocatedTime, 0.6f);
      }

      float qValue = Math.Abs(inputs.RootQ);
      var attentionLevel = CalcExtraTime(qValue, inputs.PriorMoveStats, diffTime) * Aggressiveness;

      //add or reduce time based on relative time usage - these could be improved on
      if (diffTime < 0.9f && attentionLevel < 1.05f)
      {
        attentionLevel *= 0.9f;
      }
      else if (diffTime > 1.05f && attentionLevel > 1.05f)
      {
        attentionLevel *= 1.05f;
      }
      else if (diffTime > 1.2f)
      {
        attentionLevel *= 1.1f;
      }

      //add extra time based on calculation but never more than MaxAttentionLevel
      totalTimeToUse *= Math.Min(MaxAttentionLevel, attentionLevel);
      return Return(Math.Min(totalTimeToUse, upperTimeCut), 0.6f);
    }

    private float CalcExtraTime(float qValue, List<GameMoveStat> priorMoves, float diffTime)
    {
      float trendScore = FindTrend(priorMoves, qValue);
      float adjustedQ = 1 + (qValue * QValueMultiplier) + trendScore;
      //Console.WriteLine($"Time on move {priorMoves.Count} is {diffTime:f2} Q-value: {qValue:f2} Attention: {adjustedQ:f2} Trend: {trendScore:f2}");
      return adjustedQ;
    }

    private float FindTrend(List<GameMoveStat> priorMoves, float qValue)
    {
      //game just started
      if (priorMoves.Count < 4)
      {
        return 0.05f;
      }

      //do not bother with further calculation when q-value is too low or too high
      if (qValue < 0.15f || qValue > 0.7)
      {
        return 0.0f;
      }

      var nextLastQ = priorMoves[^4].ScoreQ;
      var lastQ = priorMoves[^2].ScoreQ;
      float trend = Math.Abs(lastQ) - Math.Abs(nextLastQ);
      float delta = Math.Abs(trend);
      float trendScore = Math.Min(TrendMultiplierLimit, delta * TrendMultiplier);
      return trendScore;
    }
  }
}
