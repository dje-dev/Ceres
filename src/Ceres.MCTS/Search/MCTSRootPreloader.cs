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

using Ceres.MCTS.MTCSNodes;
using System.Collections.Generic;
using Ceres.Base.DataTypes;
using System;
using System.Diagnostics;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Search
{
  /// <summary>
  /// Manages preloading (populating cache for possible future use) 
  /// set of positions near the root.
  /// 
  /// Positions are preloaded up to a specified depth, with a specified:
  ///   - number of positions visited from each child (those having highest prior probability),
  ///   - and optional minimum probability
  /// </summary>
  public class MCTSRootPreloader
  {
    /// <summary>
    /// Only nodes having prior probability above this threshold are eligible for preloading.
    /// </summary>
    public const float PRELOAD_MIN_P = 0.10f;

    /// <summary>
    /// We only preload when the batch would otherwise be smaller than this many nodes.
    /// </summary>
    public const int PRELOAD_THRESHOLD_BATCH_SIZE = 24;


    /// <summary>
    /// Total number of nodes preloaded across all sessions.
    /// </summary>
    public static long TotalCumulativeRootPreloadNodes;


    /// <summary>
    /// Publicly exposed method that launches a round of root preloading
    /// based on search paratemrs and updates associated statistics.
    /// </summary>
    /// <param name="root"></param>
    /// <param name="selectorID"></param>
    /// <param name="maxNodes"></param>
    /// <param name="pThreshold"></param>
    /// <returns></returns>
    public List<MCTSNode> GetRootPreloadNodes(MCTSNode root, int selectorID, int maxNodes, float pThreshold = float.NaN)
    {
      ParamsSearch parmsSearch = root.Context.ParamsSearch;

      // Nothing to do if this feature is not active
      if (parmsSearch.Execution.RootPreloadDepth == 0) return null;

      // Try to gather some nodes
      List<MCTSNode> nodes = GatherRootPreloadNodes(selectorID, root, maxNodes, 
                                                    parmsSearch.Execution.RootPreloadDepth, 
                                                    parmsSearch.Execution.RootPreloadWidth, pThreshold);

      int numNodesSelectedThisAttempt = nodes is null ? 0 : nodes.Count;

      TotalCumulativeRootPreloadNodes += numNodesSelectedThisAttempt;

      return nodes;
    }


    /// <summary>
    /// Coordinates the root preloading.
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="root"></param>
    /// <param name="maxNodes"></param>
    /// <param name="maxDepth"></param>
    /// <param name="width"></param>
    /// <param name="pThreshold"></param>
    /// <returns></returns>
    List<MCTSNode> GatherRootPreloadNodes(int selectorID, MCTSNode root, int maxNodes, int maxDepth, int width, float pThreshold)
    {
      if (root.NumPolicyMoves == 0) return null;

      List<MCTSNode> nodes = new List<MCTSNode>();

      GatherRootPreloadNodes(selectorID, root, maxNodes, maxDepth, width, nodes, pThreshold);

      return nodes;
    }


    /// <summary>
    /// Implements the gathering of nodes up to speecified maximum depth and width.
    /// </summary>
    /// <param name="selectorID"></param>
    /// <param name="node"></param>
    /// <param name="maxNodes"></param>
    /// <param name="maxDepth"></param>
    /// <param name="width"></param>
    /// <param name="nodes"></param>
    /// <param name="pThreshold"></param>
    void GatherRootPreloadNodes(int selectorID, MCTSNode node, int maxNodes, int maxDepth, int width, List<MCTSNode> nodes, float pThreshold)
    {
      if (nodes.Count >= maxNodes) return;
      if (node.IsTranspositionLinked) return;

      // For each top child
      for (int i = 0; i < Math.Min(node.NumPolicyMoves, width); i++)
      {
        (MCTSNode childNode, EncodedMove move, FP16 p) = node.ChildAtIndexInfo(i);


        if (float.IsNaN(pThreshold) || p >= pThreshold)
        {
          // Add this node if not already extant
          if (childNode == null)
          {
            // Create this child (as cache only) and update in flight
            childNode = node.CreateChild(i);
            childNode.ActionType = MCTSNode.NodeActionType.CacheOnly;
            if (selectorID == 0)
              childNode.Ref.BackupIncrementInFlight(1, 0);
            else
              childNode.Ref.BackupIncrementInFlight(0, 1);

            nodes.Add(childNode);
            if (nodes.Count == maxNodes) return;
          }
          else
          {
            node.Annotate();

            if (node.Depth < maxDepth - 1 && childNode.N > 0)
            {
              // Recursively descend
              Debug.Assert(childNode != null);
              GatherRootPreloadNodes(selectorID, childNode, maxNodes, maxDepth, width, nodes, pThreshold);
            }
          }
        }
      }

    }
  }


}
