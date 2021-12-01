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

#endregion

namespace Ceres.MCTS.Params
{
  /// <summary>
  /// Parameters relating to the secondary neural network evaluator (if in use).
  /// </summary>
  [Serializable]
  public record ParamsSearchSecondaryEvaluator
  {
    /// <summary>
    /// Interval of tree growth in nodes between which 
    /// secondary evaluation is run over tree
    /// (as an absolute tree size).
    /// </summary>
    public int UpdateFrequencyMinNodesAbsolute = int.MaxValue;

    /// <summary>
    /// Interval of tree growth in nodes between which 
    /// secondary evaluation is run over tree
    /// (as a fraction of current tree size).
    /// </summary>
    public float UpdateFrequencyMinNodesRelative = float.MaxValue;

    /// <summary>
    /// Number of initial nodes in the tree which are forced to use the secondary evaluator.
    /// </summary>
    public int InitialTreeNodesForceSecondary = 0;

    /// <summary>
    /// Minimum number of nodes a node must have to be eligible for
    /// secondary evaluation (expressed as a fraction of tree size).
    /// </summary>
    public float UpdateMinNFraction = float.MaxValue;

    /// <summary>
    /// Fraction at which value head output of secondary evaluator is blended in.
    /// </summary>
    public float UpdateValueFraction = 1;

    /// <summary>
    /// Fraction at which policy head output of secondary evaluator is blended in.
    /// </summary>
    public float UpdatePolicyFraction = 1;

    /// <summary>
    /// Returns the minimum batch size to allow to accumulate
    /// before sending secondary nodes to evaluator.
    /// </summary>
    /// <param name="currentRootN"></param>
    /// <returns></returns>
    public int MinBatchSize(int currentRootN)
    {
      bool selectMany = UpdateMinNFraction < 0.002f;

      if (currentRootN < 5_000)
      {
        return selectMany ? 30 : 20;
      }
      else if (currentRootN < 50_000)
      {
        return selectMany ? 80 : 35;
      }
      else 
      {
        return selectMany ? 140 : 50;
      }
    }
  }
}
