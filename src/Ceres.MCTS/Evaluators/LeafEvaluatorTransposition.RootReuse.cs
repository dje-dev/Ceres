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
using System.Diagnostics;

using Ceres.MCTS.Environment;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.Evaluators
{

  /*
    The state variables involved in tracking transposition linked nodes:

    At MCTSNodeStruct there are 2 durable state variables possibly set by LeafEvaluatorTransposition:
       TranspositionRootIndex - the index of the node which is our tranposition root to which the node is still attached
       NumVisitsPendingTranspositionRootExtraction - the number of times future visits will be sourced from the transposition root instead of this node

    At MCTSNode has 3 new state variables which hold values computed during(parallel) 
    batch gathering(in LeafSelectorMulti) and then used by during backup(in MCTSApply):
       PendingTranspositionV
       PendingTranspositionM
       PendingTranspositionD

    The sequence of processing during search is:
      1. (preprocessing in batch gathering): 
         LeafEvaluatorTransposition will recognize transpositions set the two state variables at MCTSNodeStruct for them
      2. (batch gathering) 
          LeafSelectorMulti will notice transposition linked nodes and:
            - they are treated as leafs if NumVisitsPendingTranspositionRootExtraction indicates sufficiently many 
              pending values can come from the node as requested as targets visits for the node.
              If so, the pending values stored in MCTSNode are potentially refreshed.
            - otherwise the CopyUnexpandedChildrenFromOtherNode is called to permanently delink the node from its transposition root
      3. (backup) 
         MCTSApply treats nodes which are still transposition linked in a special way.
         The 3 PendingTransposition values in MCTSNode are use to provide values for backup in the tree.
         Additionally NumVisitsPendingTranspositionRootExtraction is decremented by the number of visits to the node.
  */


  /// <summary>
  /// 
  /// </summary>
  public partial class LeafEvaluatorTransposition
  {
    /// <summary>
    /// Set or update the pending transposition values in a Node
    /// if they are missing or stale.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="transpositionRootNode"></param>
    public static void EnsurePendingTranspositionValuesSet(MCTSNode node, bool possiblyRefresh)
    {
      // TODO: make this a constant set in MCTSParamsFixed or MCTSSearch
      const int TRANSPOSITION_VALUE_REFRESH_INTERVAL = 1;

      // Possibly the value cached in PendingTranspositionV to be used for the pending
      // transposition values is not present (because the prior MCTSNode was lost from cache).
      // If so recalculate and set it here.
      bool cachedTranspositionValuesMissing = float.IsNaN(node.PendingTranspositionV);

      //if (cachedTranspositionValuesMissing) MCTSEventSource.TestMetric1++;

      // Possibly periodically refresh the value used for transposition backup
      // because the root node may have had more visits (and therefore be more accurate)
      // since last time calculated.
      bool timeToRefresh = possiblyRefresh && node.NumVisitsPendingTranspositionRootExtraction % TRANSPOSITION_VALUE_REFRESH_INTERVAL == 0;

      if (cachedTranspositionValuesMissing || timeToRefresh)
      {
        int transpositionNodeIndex = node.TranspositionRootIndex;
        if (transpositionNodeIndex == 0)
        {
          if (!node.Tree.TranspositionRoots.TryGetValue(node.Ref.ZobristHash, out transpositionNodeIndex))
          {
            throw new Exception("Internal error, transposition root lost");
          }
        }

        ref readonly MCTSNodeStruct transpositionNode = ref node.Context.Tree.Store.Nodes.nodes[transpositionNodeIndex];
        SetPendingTransitionValues(node, in transpositionNode, ForceUseV(node, in transpositionNode));
        Debug.Assert(!float.IsNaN(node.PendingTranspositionV));
      }
    }


    /// <summary>
    /// Set the PendingTransposition values for this node based on speciifed transposition root.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="transpositionRootNode"></param>
    static void SetPendingTransitionValues(MCTSNode node,
                                           in MCTSNodeStruct transpositionRootNode,
                                           bool forceUseV)
    {
      if (forceUseV)
      {
        node.PendingTranspositionV = transpositionRootNode.V;
        node.PendingTranspositionM = transpositionRootNode.MPosition;
        node.PendingTranspositionD = transpositionRootNode.DrawP;
      }
      else
      {
        node.PendingTranspositionV = (float)transpositionRootNode.Q;
        node.PendingTranspositionM = transpositionRootNode.MAvg;
        node.PendingTranspositionD = transpositionRootNode.DAvg;
      }
    }


    static bool ForceUseV(MCTSNode node, in MCTSNodeStruct transpositionRootNode)
    {
      if (!node.Context.ParamsSearch.TranspositionUseTransposedQ)
      {
        return true;
      }

      bool tooBig = transpositionRootNode.N > node.Context.ParamsSearch.MaxNTranspositionRootReuse;

      if (tooBig)
      {
        MCTSEventSource.TestMetric1++;
      }

      return tooBig;
    }


  }
}
