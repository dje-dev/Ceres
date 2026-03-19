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
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Graphs;

/// <summary>
/// Partial class containing computed metrics and statistics for Graph.
/// </summary>
public unsafe partial class Graph
{
  /// <summary>
  /// Calculates the average Shannon entropy of visit distributions across sampled nodes in the graph.
  /// For each node, entropy is computed over the distribution of visits to children,
  /// where the distribution includes all policy moves (including unexpanded moves which count as 0 visits).
  /// </summary>
  /// <param name="maxSamples">Maximum number of nodes to sample (default 2000)</param>
  /// <returns>Average entropy across sampled nodes, or 0 if no valid samples</returns>
  public float CalcAvgVisitEntropy(int maxSamples = 2000)
  {
    int numUsedNodes = NodesStore.NumUsedNodes;

    // Need at least one valid node (starting from index 1)
    if (numUsedNodes < 1)
    {
      return 0;
    }

    Random random = new();
    double totalEntropy = 0;
    int validSamples = 0;

    int samplesToTake = Math.Min(maxSamples, numUsedNodes);

    for (int i = 0; i < samplesToTake; i++)
    {
      // Pick a random node index in range [1, numUsedNodes]
      // Index 0 is reserved (null node), so start from 1
      int nodeIndex = random.Next(1, numUsedNodes + 1);

      GNode node = this[nodeIndex];

      // Skip nodes that are old generation (orphaned)
      if (node.NodeRef.IsOldGeneration)
      {
        continue;
      }

      double entropy = CalcNodeVisitEntropy(node);
      if (!double.IsNaN(entropy))
      {
        totalEntropy += entropy;
        validSamples++;
      }
    }

    return validSamples > 0 ? (float)(totalEntropy / validSamples) : 0;
  }


  /// <summary>
  /// Calculates the Shannon entropy of the visit distribution for a single node.
  /// The distribution includes all policy moves, with unexpanded moves counting as 0 visits.
  /// </summary>
  /// <param name="node">The node to calculate entropy for</param>
  /// <returns>Entropy value, or NaN if the node has no visits to children</returns>
  private static double CalcNodeVisitEntropy(GNode node)
  {
    int numPolicyMoves = node.NumPolicyMoves;
    if (numPolicyMoves == 0)
    {
      return double.NaN;
    }

    // Total visits to children is N - 1 (excluding the root visit to this node itself)
    int totalVisits = node.N - 1;
    if (totalVisits <= 0)
    {
      return double.NaN;
    }

    double entropy = 0;
    int numExpandedEdges = node.NumEdgesExpanded;

    // Process expanded edges (which have actual visit counts)
    for (int i = 0; i < numExpandedEdges; i++)
    {
      int edgeN = node.ChildEdgeAtIndex(i).N;
      if (edgeN > 0)
      {
        double p = (double)edgeN / totalVisits;
        entropy -= p * Math.Log(p);
      }
      // If edgeN == 0, the contribution is 0 (0 * log(0) is defined as 0)
    }

    // Unexpanded edges have 0 visits, contributing nothing to entropy
    // (since 0 * log(0) = 0 by convention)

    return entropy;
  }
}
