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


#endregion

using System;

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Defines noise used in choosing root best moves via
  /// softmax sampling moves with a given temperature and minimum proximity to best move).
  /// </summary>
  [Serializable]
  public class SearchNoiseBestMoveSamplingDef
  { 
    /// <summary>
    /// Move number within game at which noise is turned off.
    /// </summary>
    public int MoveSamplingNumMovesApply = 0; // 30

    /// <summary>
    /// Minimum fraction of N of best move that a move must attain to be eligible for selection.
    /// 
    /// A value of 0.5 to 0.65 may be reasonable when using primary move pruning
    /// </summary>
    public float MoveSamplingConsiderMovesWithinFraction = 0; // 

    /// <summary>
    /// Softmax temperature to be applied to eligible moves.
    /// 
    /// Typical values are approximately 5
    /// </summary>
    public float MoveSamplingConsideredMovesTemperature = float.NaN; // 0.10

    /// <summary>
    /// Maximum number of times to allow moves to be overriden from greedy best move within a single game.
    /// </summary>
    public int MoveSamplingMaxMoveModificationsPerGame = int.MaxValue;   

    /// <summary>
    /// 
    /// </summary>
    /// <param name="withinFration"></param>
    /// <param name="temperature"></param>
    /// <param name="numMovesApply"></param>
    /// <param name="maxMoveModificationsPerGame"></param>
    /// <returns></returns>
    public static SearchNoiseBestMoveSamplingDef MoveSamplingDiversity(float withinFration, float temperature, 
                                                                       int numMovesApply = int.MaxValue, 
                                                                       int maxMoveModificationsPerGame = int.MaxValue)
    {
      return new SearchNoiseBestMoveSamplingDef()
      {
        MoveSamplingConsiderMovesWithinFraction = withinFration,
        MoveSamplingConsideredMovesTemperature = temperature,
        MoveSamplingNumMovesApply = numMovesApply,
        MoveSamplingMaxMoveModificationsPerGame = maxMoveModificationsPerGame
      };
    }

  }
}

