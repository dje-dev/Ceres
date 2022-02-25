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
using System.Threading.Tasks;
using Ceres.Base.Math;
using Ceres.Base.Math.Random;
using Ceres.Chess;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Static calculation methods for Thompson sampling descendents from nodes
  /// according to a speciifed temperature and descent termination rule.
  /// </summary>
  public static class MCTSNodeResampling
  {
    #region Internal data structures

    const int MAX_SAMPLES = 1024;

    [ThreadStatic]
    static float[] empiricalDistribScratch;

    #endregion


    /// <summary>
    /// Returns the average and standard devation over a set of numSamples
    /// samples taken from leaf descendents of node where the leaf is at least minN
    /// using Thompson sampling over a specified resampling temperature.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="numSamples"></param>
    /// <param name="minN"></param>
    /// <param name="resamplingTemperature"></param>
    /// <returns></returns>
    /// <exception cref="ArgumentException"></exception>
    public static (float avg, float sd) GetResampledStats(MCTSNode node, int numSamples, int minN, float resamplingTemperature)
    {
      if (numSamples > MAX_SAMPLES)
      {
        throw new ArgumentException(nameof(numSamples) + " exceeds max supported samples of " + MAX_SAMPLES);
      }

      node.Annotate();
      int thisDepth = node.Depth;

      // Extract as many samples as requested.
      float sum = 0;
      float sumSq = 0;
      object lockObj = new();


      const int MAX_THREADS = 4;
      Parallel.For(0, numSamples, new ParallelOptions() { MaxDegreeOfParallelism = MAX_THREADS }, i =>
      {
        // Make sure all the ThreadStatic variables are initialized.
        if (empiricalDistribScratch == null)
        {
          empiricalDistribScratch = new float[64];
        }

        (int depth, float sample) = DescendGetSample(node, resamplingTemperature, minN);

        // Save sample (adjusted to be from perspective of this node)
        bool isSamePerspective = depth % 2 == thisDepth % 2;
        float sampleToUse = isSamePerspective ? sample : -sample;
        lock (lockObj)
        {
          sum += sampleToUse;
          sumSq += sum * sum;
        }
      });

      // Compute summary statistics.
      float avg = sum / numSamples;
      float sd = MathF.Sqrt(sumSq - avg * avg);

      return (avg, sd);
    }


    static (int depth, float sample) DescendGetSample(MCTSNode node, float temperature, int minN)
    {
      return DoDescendGetSample(node, temperature, minN, node.Depth);
    }


    static (int depth, float sample) DoDescendGetSample(MCTSNode node, float temperature, int minN, int depth)
    {
      if (node.Terminal.IsTerminal() || node.NumChildrenExpanded == 0)
      {
        return (depth, (float)node.Q);
      }

      // Extract the emprical policy distribution
      GetEmpiricalPolicyDistribution(node, node.NumChildrenExpanded, empiricalDistribScratch);

      // Use Thompson sampling to pick one of the children according to empirical distribution.
      int drawIndex = ThompsonSampling.Draw(empiricalDistribScratch, node.NumChildrenExpanded, temperature);

      // Get this child node.
      MCTSNode childNode = node.ChildAtIndex(drawIndex);

      // Stop descending if max depth reached or there are no more descendents.
      if (childNode.N < minN)
      {
        // Do not descend further, child has too few visits.
        return (depth, (float)node.Q);
      }

      // Continue descent with chosen child.
      return DoDescendGetSample(childNode, temperature, minN, depth + 1);
    }


    private static void GetEmpiricalPolicyDistribution(MCTSNode node, int numChildrenToCheck, Span<float> empiricalDistrib)
    {
      float count = 0;
      for (int i = 0; i < numChildrenToCheck; i++)
      {
        ref readonly MCTSNodeStruct child = ref node.StructRef.ChildAtIndexRef(i);
        empiricalDistrib[i] = child.N;
        count += child.N;
      }
      for (int i = 0; i < numChildrenToCheck; i++)
      {
        empiricalDistrib[i] /= count;
      }
    }

  }
}
