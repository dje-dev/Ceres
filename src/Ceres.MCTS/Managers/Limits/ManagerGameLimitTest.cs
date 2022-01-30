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
using System.Linq;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.MCTS.Iteration;


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

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="aggressiveness"></param>
    public ManagerGameLimitTest(float aggressiveness = 1.0f)
    {
      Aggressiveness = aggressiveness;
    }

    public LimitsManagerInstamoveDecision CheckInstamove(MCTSearch search, ManagerGameLimitInputs inputs)
    {
      return LimitsManagerInstamoveDecision.NoDecision;
    }

    public bool CheckStopSearch(MCTSearch search, ManagerGameLimitInputs inputs)
    {
      //var s = search;
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

      //no meaningful increment
      if (inputs.IncrementSelf < 0.01)
      {
        (float time, float ext) = CalcNoIncrementTimeUse(inputs);
        return Return(time, ext);
      }

      else
      {
        (float time, float ext) = CalcWithIncrementTimeUse(inputs);
        return Return(time, ext);
      }
    }

    //todo - work on no increment time control
    private (float, float) CalcNoIncrementTimeUse(ManagerGameLimitInputs inputs)
    {
      //fixed time is increased when enableInstamove = true
      float alphaZeroFractionFixed = inputs.SearchParams.EnableInstamoves ? 1f / 15f : 1f / 22f;
      float scheduledTime = inputs.IncrementSelf + (inputs.RemainingFixedSelf * alphaZeroFractionFixed);
      float totalTimeToUse = scheduledTime;
      float remainingTime = inputs.RemainingFixedSelf + inputs.IncrementSelf;

      //never use more than 20% of time in one move
      float upperTimeCut = remainingTime * 0.1f;

      //add some extra time for the first move
      if (inputs.IsFirstMoveOfGame)
      {
        return (totalTimeToUse * 1.5f, 0.6f);
      }

      //just boost alpha-zero time control when time is very low
      if (remainingTime < 2.0f)
      {
        return (totalTimeToUse * 1.3f, 0.0f);
      }

      //look at time usage for opponent - especially useful for AB engines
      GameMoveStat oppLastMove = inputs.PriorMoveStats[^1];
      (float restTimeOpp, float usedLastMoveOpp) = (oppLastMove.SearchLimit.Value, oppLastMove.TimeElapsed);

      //do we have more or less time than opponent
      float diffTime = remainingTime / restTimeOpp;

      //check if opponent used a lot of time on his last move
      float fractionLastMove = usedLastMoveOpp / restTimeOpp;
      if (fractionLastMove >= 0.125f)
      {
        var time = Math.Min(totalTimeToUse * 2 * diffTime, upperTimeCut);
        //Console.WriteLine($"Opponent did think > 15%: {fractionLastMove:f2} Diff until now: {diffTime:f2} Ceres rem: {remainingTime:f2} - Time given: {time:f2} ");    
        return (time, 0.6f);
      }

      //var moves = inputs.PriorMoveStats.Count;
      //Console.WriteLine($"Time advantage fraction for Ceres on move {moves}: {diffTime:f2} Ceres: {remainingTime:f2} Opp: {rest:f2} ");    

      float qAbs = Math.Abs(inputs.RootQ);

      (float qFactor, float extension) = qAbs switch
      {
        > 0.32f => (0.2f, 0.4f),
        > 0.22f => (0.15f, 0.4f),
        > 0.10f => (0.1f, 0.2f),
        _ => (0.0f, 0.0f)
      };

      totalTimeToUse *= (1 + qFactor);

      //do not stay too far from opponents time control - correction based on previous focus
      if (diffTime < 0.7f)
      {
        if (qFactor < 0.4f)
        {
          totalTimeToUse *= diffTime;
          //Console.WriteLine($"Less time now:  {diffTime:f2} Ceres rem: {remainingTime:f2} Time given: {totalTimeToUse:f2}");
        }
      }

      if (diffTime > 1.35f)
      {
        totalTimeToUse *= diffTime;
        //Console.WriteLine($"More time now:  {diffTime:f2} Ceres rem: {remainingTime:f2} Time given: {totalTimeToUse:f2}");       
      }

      return (Math.Min(totalTimeToUse, upperTimeCut), extension);
    }




    private (float, float) CalcWithIncrementTimeUse(ManagerGameLimitInputs inputs)
    {
      //fixed time is increased when enableInstamove = true
      float alphaZeroFractionFixed = inputs.SearchParams.EnableInstamoves ? 1f / 20f : 1f / 25f;
      float scheduledTime = inputs.IncrementSelf + (inputs.RemainingFixedSelf * alphaZeroFractionFixed);
      float totalTimeToUse = scheduledTime;
      float remainingTime = inputs.RemainingFixedSelf + inputs.IncrementSelf;
      float fixedAllot = alphaZeroFractionFixed * inputs.RemainingFixedSelf;
      float fixedRatio = inputs.RemainingFixedSelf / inputs.IncrementSelf;

      //never use more than 20% of fixed time in one move
      float upperTimeCut = inputs.RemainingFixedSelf * 0.2f + inputs.RemainingFixedSelf * 0.95f;

      //add some extra time for the first move
      if (inputs.IsFirstMoveOfGame)
      {
        float extraTime = remainingTime > 45f ? 1.7f : 1 + (0.015f * remainingTime);
        return (totalTimeToUse * extraTime, 0.6f);
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
        return (inputs.RemainingFixedSelf * multiplier, 0.0f);
      }

      else if (NearExhaustion(inputs))
      {
        float multiplier = hasMeaningfulIncrement ? 0.70f : 0.03f;
        return (inputs.RemainingFixedSelf * multiplier, 0.2f);
      }

      //look at time usage for opponent - especially useful for AB engines
      GameMoveStat oppLastMove = inputs.PriorMoveStats[^1];
      bool isWhite = oppLastMove.Side == SideType.Black;
      (float restTimeOpp, float usedLastMoveOpp) = (oppLastMove.SearchLimit.Value, oppLastMove.TimeElapsed);

      //do we have more or less time than opponent
      float diffTime = isWhite ?
        remainingTime / restTimeOpp :
        (remainingTime - totalTimeToUse) / restTimeOpp;

      //check if opponent used a lot of time on his last move
      float fractionLastMove = usedLastMoveOpp / restTimeOpp;
      if (fractionLastMove >= 0.125f)
      {
        float time = Math.Min(totalTimeToUse * 2 * diffTime, upperTimeCut);
        //Console.WriteLine($"Opponent did think > 15%: {fractionLastMove:f2} Diff until now: {diffTime:f2} Ceres rem: {remainingTime:f2} - Time given: {time:f2} ");    
        return (time, 0.6f);
      }

      //calculate how difficult the position has been (on average) until now.
      int moves = inputs.PriorMoveStats.Count(e => e.Side != oppLastMove.Side);
      float qDifficultLevel = 0.33f;
      int numberOfHighQmoves = inputs.PriorMoveStats.Count(e => Math.Abs(e.ScoreQ) > qDifficultLevel && e.Side != oppLastMove.Side);
      float fractionDifficultMoves = numberOfHighQmoves / (float)moves;
      float q = Math.Abs(inputs.RootQ);

      //tracks number of moves with high difficult level and reduces the extra time based on this fraction
      //if a high fraction of all moves are above qdifficult level we need to adjust to the situation and give less time alotted to every move.
      float difficultLevel = 0.5f * fractionDifficultMoves;
      //a function based on q-value that tells the TM how much more time should be used
      float attentionLevel = q < 0.6 ? 1.6f * q : 0.6f;

      float extension = q < 0.6 ? 0.6f : 1.0f; //1 or above indicates no instamoves - not sure how important it is yet...

      //Important to reduce attention level based on how often you had to use extra time until now (difficult level) and never allow the calculation to be negative
      float extraTimeFactor = Math.Max(0f, attentionLevel - difficultLevel) * Aggressiveness;

      //Console.WriteLine($"Time advantage on move {moves} is {diffTime:f2} (Ceres: {remainingTime:f2} Opp: {restTimeOpp:f2}) Extra time factor: {extraTimeFactor:f2} " +
      //                  $"Fract difficult moves: {fractionDifficultMoves:f2} Qvalue: {inputs.RootQ:f2} Att: {attentionLevel:f2} Difficult: {difficultLevel:f2}");

      //adjust the time calculation with extraTimeFactor and relative time usage (diffTime)      
      totalTimeToUse = moves < 5 ? totalTimeToUse * (1 + attentionLevel) : totalTimeToUse * (1 + extraTimeFactor) * diffTime;
      //totalTimeToUse = totalTimeToUse * (1 + qFactor - factor) * diffTime;

      return (Math.Min(totalTimeToUse, upperTimeCut), (difficultLevel > 0.7f ? 0.2f : extension));
    }
  }
}
