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
  /// to spend on the next move using a very simplistic algorithm
  /// (fixed traction of remaining search allowance).
  /// </summary>
  [Serializable]
  public class ManagerGameLimitSimple : IManagerGameLimit
  {
    public readonly float Divisor;

    public float FRACTION_PER_MOVE => 1.0f / Divisor;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="divisor"></param>
    public ManagerGameLimitSimple(float divisor) => Divisor = divisor;


    /// Determines how much time or nodes resource to
    /// allocate to the the current move in a game subject to
    /// a limit on total numbrer of time or nodes over 
    /// some number of moves (or possibly all moves).
    public ManagerGameLimitOutputs ComputeMoveAllocation(ManagerGameLimitInputs inputs)
    {
      ManagerGameLimitOutputs Return(float value) => new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType, value,
                                                                                                 maxTreeNodes: inputs.MaxTreeNodesSelf,
                                                                                                 maxTreeVisits: inputs.MaxTreeVisitsSelf));

      if (inputs.MaxMovesToGo.HasValue && inputs.MaxMovesToGo < 2)
      {
        return new ManagerGameLimitOutputs(new SearchLimit(inputs.TargetLimitType,
                                                           inputs.RemainingFixedSelf * 0.99f,
                                                           maxTreeNodes: inputs.MaxTreeNodesSelf,
                                                           maxTreeVisits: inputs.MaxTreeVisitsSelf));
      }

      float baseTimeToUse = inputs.RemainingFixedSelf * FRACTION_PER_MOVE;

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