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

namespace Ceres.MCTS.Managers.Limits
{
  /// <summary>
  /// Manager of time which estimates the optimal amount of time
  /// to spend on the next move (default Ceres version)
  /// </summary>
  [Serializable]
  public class ManagerGameLimitCeres: IManagerGameLimit
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
    /// Determines what fraction of the base move should
    /// be consumed for this move.
    /// </summary>
    /// <param name="inputs"></param>
    /// <returns></returns>
    float FractionOfBasePerMoveToUse(ManagerGameLimitInputs inputs)
    {
      float factorLargeIncrement = 1.0f;
      if (inputs.IncrementSelf > 0)
      {
        // With increasingly significant increment,
        // frontload the use of base time more aggressively
        // because we don't have to worry about 
        // "running out of time."
        const float MAX_LARGE_INCREMENT_MULTIPLIER = 2.5f;
        float estBasePerMove = inputs.RemainingFixedSelf / 30;
        float incrementRelativeToBasePerMove = inputs.IncrementSelf / estBasePerMove;
        if (incrementRelativeToBasePerMove > 0.2)
        {
          factorLargeIncrement = 1.0f + 0.5f * ((incrementRelativeToBasePerMove - 0.2f) / 0.3f);
          factorLargeIncrement = MathF.Min(MAX_LARGE_INCREMENT_MULTIPLIER, factorLargeIncrement);
        }
      }

      // When we are behind then it's worth taking a gamble and using more time
      // but when we are ahead, take a little less time to be sure we don't err in time pressure.
      float factorWinningness = inputs.RootQ switch
      {
        < -0.25f => 1.15f,
        < -0.15f => 1.05f,
        > 0.25f => 0.90f,
        > 0.15f => 0.95f,
        _ => 1.0f
      };

      // Spend 30% more on first move of game (definitely no tree reuse, etc.)
      float factorFirstMove = inputs.IsFirstMoveOfGame ? 1.3f : 1.0f;

      // Make a divisor which is between about 12 and 17
      // and a incresing function of the piece count.
      // Note that this is a relatively small number because
      //  - some moves will not do any search at all (due to instamoves), and
      //  - many moves will not actually run the full search duration (due to smart pruning)
      float baseDivisor = 10 + MathF.Pow(inputs.StartPos.PieceCount, 0.5f);

      float ret = Aggressiveness * (1.0f / baseDivisor) * factorLargeIncrement * factorWinningness * factorFirstMove;

      return ret;
    }

    static bool Panic(ManagerGameLimitInputs inputs)
    {
      if (inputs.TargetLimitType == SearchLimitType.NodesForAllMoves)
        return inputs.RemainingFixedSelf + inputs.IncrementSelf < 50;
      else
        return (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 0.25;
    }

    static bool NearExhaustion(ManagerGameLimitInputs inputs)
    {
      if (inputs.TargetLimitType == SearchLimitType.NodesForAllMoves)
        return (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 200;
      else
        return (inputs.RemainingFixedSelf + inputs.IncrementSelf) < 1;
    }


    /// Determines how much time or nodes resource to
    /// allocate to the the current move in a game subject to
    /// a limit on total numbrer of time or nodes over 
    /// some number of moves (or possibly all moves).
    public ManagerGameLimitOutputs ComputeMoveAllocation(ManagerGameLimitInputs inputs)
    {
      ManagerGameLimitOutputs Return(float value) => new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType, value));

      // If this is the last move to go, use almost all available time.
      // TODO: but can a player carry forward time on a clock? Then this doesn't make sense.
      if (inputs.MaxMovesToGo.HasValue && inputs.MaxMovesToGo < 2)
        return Return(inputs.RemainingFixedSelf * 0.98f);

      bool isNodes = inputs.TargetLimitType == SearchLimitType.NodesForAllMoves;
      float incrementMeaningfulThreshold = isNodes ? 1 : 0.01f;
      bool hasMeaningfulIncrement = inputs.IncrementSelf > incrementMeaningfulThreshold;

      if (Panic(inputs))
      {
        float multiplier = hasMeaningfulIncrement ? 0.50f : 0.01f;
        return Return(inputs.RemainingFixedSelf * multiplier);
      }
      else if (NearExhaustion(inputs))
      {
        float multiplier = hasMeaningfulIncrement ? 0.70f : 0.03f;
        return Return(inputs.RemainingFixedSelf * multiplier);

      }

      float baseTimeToUse = inputs.RemainingFixedSelf * FractionOfBasePerMoveToUse(inputs);

      if (inputs.RemainingFixedSelf > inputs.IncrementSelf)
      {
        // Since no danger of running out of time, use almost all of the increment now.
        return Return(baseTimeToUse + inputs.IncrementSelf * 0.95f);
      }
      else
      {
        // Can't spend the increment because we haven't been awarded it yet
        // for this move, and not enough left on clock. 
        // Just use most of fixed time, counting on increment to rebulid spare time in the future.
        return Return(MathF.Min(baseTimeToUse, inputs.RemainingFixedSelf * 0.9f));
      }
    }

  }
}