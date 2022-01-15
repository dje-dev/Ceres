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
using System.Runtime.InteropServices;
using Ceres.Base.DataType.Trees;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  /// <summary>
  /// Structure that tracks the rolling uncertainty (mean absolute deviation)
  /// of backed up evaluations to a node.
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Sequential, Pack = 1, Size = 4)]
  public struct MCTSNodeUncertaintyAccumulator
  {
    /// <summary>
    /// The first few visits are ignored because the average evaluation
    /// against which the visits are compared is still very  noise at small sample sizes.
    /// </summary>
    public const int MIN_N_UPDATE = 10;

    /// <summary>
    /// Uncertainty estimates are only considered available/valid
    /// after node has reached enough visits and therefore 
    /// enough samples have been seen.
    /// </summary>
    public const int MIN_N_ESTIMATE = 15;

    /// <summary>
    /// Uncertainty estimates are computed with some maximum 
    /// notional sample size, for two reasons:
    ///   - provides an exponential moving window behavior, and
    ///   - prevents "saturation" 
    /// </summary>
    const int NUM_TRAILING_VALUES = 100_000;

    /// <summary>
    /// The current MAD estimate.
    /// </summary>
    float currentMADEstimate;

    /// <summary>
    /// Clears value of uncertainty estimate.
    /// </summary>
    public void Clear() => currentMADEstimate = 0;

    /// <summary>
    /// Returns current value 
    /// </summary>
    public float Uncertainty => currentMADEstimate;

    /// <summary>
    /// Returns the uncertainty value to be used assuming
    /// a speciifed node visit count.
    /// </summary>
    /// <param name="nodeN"></param>
    /// <param name="fillInValueIfNone"></param>
    /// <returns></returns>
    public float UncertaintyAtN(int nodeN, float fillInValueIfNone = float.NaN)
      => GetUncertainty(nodeN, currentMADEstimate, fillInValueIfNone);

   
    /// <summary>
    /// Updates the uncertainty estimate based on visitCount additional visits
    /// having visitDeviation deviation from the mean value.
    /// </summary>
    /// <param name="visitDeviation"></param>
    /// <param name="visitCount"></param>
    /// <param name="nodeN"></param>
    public void UpdateUncertainty(ref MCTSNodeStruct nodeRef, float visitDeviation, float visitCount, int nodeN)
    {
      // Only update in the range 
      if (nodeN < MIN_N_UPDATE)
      {
        // Ignore
      }
      else if (nodeN == MIN_N_UPDATE)
      {
        // Traverse tree to get prior values
        CalcInitialMADEstimate(ref nodeRef);
      }
      else // (nodeN > MIN_N_UPDATE)
      {
        if (currentMADEstimate == 0)
        {
          CalcInitialMADEstimate(ref nodeRef);
        }
        int numPriorEsts = nodeN > NUM_TRAILING_VALUES ? NUM_TRAILING_VALUES : nodeN;
        double numerator = (double)currentMADEstimate * numPriorEsts
                         + (double)MathF.Abs(visitDeviation) * visitCount;
        double denominator = (numPriorEsts + visitCount);
        currentMADEstimate = (float)(numerator / denominator);
      }
    }

    #region Static helpers

    /// <summary>
    /// Traverses the subtree below this node to compute an initial MAD estimate.
    /// </summary>
    /// <param name="nodeRef"></param>
    private void CalcInitialMADEstimate(ref MCTSNodeStruct nodeRef)
    {
      float rootQ = (float)nodeRef.Q;
      int depthRoot = nodeRef.DepthInTree;
      bool rootWhite = nodeRef.IsWhite;

      int n = 0;
      float acc = 0;
      nodeRef.Traverse(nodeRef.Context.Store,
                   (ref MCTSNodeStruct innerNode) =>
                   {
                     if (!float.IsNaN(innerNode.V))
                     {
                       float mult = rootWhite == innerNode.IsWhite ? 1 : -1;
                       acc += MathF.Abs((mult * innerNode.V) - rootQ);
                       n++;
                     }
                     return true;
                   }, TreeTraversalType.BreadthFirst);
      currentMADEstimate = acc / n;
    }


    /// <summary>
    /// Static helper that returns the uncertainty value to be used
    /// given a specified number of samples and current estimate.
    /// </summary>
    /// <param name="nodeN"></param>
    /// <param name="madEstimate"></param>
    /// <param name="fillInValueIfNone"></param>
    /// <returns></returns>
    public static float GetUncertainty(int nodeN, float madEstimate, float fillInValueIfNone = float.NaN)
    {
      return nodeN >= MIN_N_ESTIMATE ? madEstimate : fillInValueIfNone;
    }

    #endregion
  }

}