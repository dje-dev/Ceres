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
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Evaluators
{

  /*
    The state variables involved in tracking transposition linked nodes:

    At MCTSNodeStruct there are 2 durable state variables possibly set by LeafEvaluatorTransposition:
       TranspositionRootIndex - the index of the node which is our tranposition root to which the node is still attached
       NumVisitsPendingTranspositionRootExtraction - the number of times future visits will be sourced from the transposition root instead of this node

    At MCTSNode has 3 new state variables which hold values computed during(parallel) 
    batch gathering (in LeafSelectorMulti) and then used by during backup(in MCTSApply):
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
            - otherwise the MaterializeSubtreeFromTranspositionRoot is called to permanently delink the node from its transposition root
      3. (backup) 
         MCTSApply treats nodes which are still transposition linked in a special way.
         The 3 PendingTransposition values in MCTSNode are use to provide values for backup in the tree.
         Additionally NumVisitsPendingTranspositionRootExtraction is decremented by the number of visits to the node.
  */

  
  /// <summary>
  /// 
  /// </summary>
  public sealed partial class LeafEvaluatorTransposition
  {
    /// <summary>
    /// Upon first visit to node to be attached to transposition root, computes and sets in the node structure:
    ///   - TranspositionRootIndex
    ///   - NumVisitsPendingTranspositionRootExtraction
    /// Also computes set the pending transposition fields in the Node object.  
    /// </summary>
    /// <param name="node"></param>
    /// <param name="transpositionRootNodeIndex"></param>
    /// <param name="transpositionRootNode"></param>
    void SetTranspositionRootReuseFields(MCTSNode node, MCTSNodeStructIndex transpositionRootNodeIndex, in MCTSNodeStruct transpositionRootNode)
    {
      Debug.Assert(transpositionRootNodeIndex.Index != 0);

      ref MCTSNodeStruct nodeRef = ref node.Ref;

      // We mark this as just extracted, but do not (yet) allocate and move over the children.
      nodeRef.NumPolicyMoves = transpositionRootNode.NumPolicyMoves;
      nodeRef.TranspositionRootIndex = transpositionRootNodeIndex.Index;

      ParamsSearch paramsSearch = node.Context.ParamsSearch;

      // Compute the number of times to apply
      int applyTarget = paramsSearch.MaxTranspositionRootUseCount;
      if (applyTarget <= 0 || applyTarget > 3)
      {
        throw new Exception("MaxTranspositionRootUseCount must 1, 2 or 3, not: " + applyTarget);
      }

      // Never reuse more than are available from the transposition root subtree.
      applyTarget = Math.Min(applyTarget, transpositionRootNode.NumUsableSubnodesForCloning);

      // Finally, the field holding the target has a fixed maximum representable size, ensure not more than that.
      applyTarget = Math.Min(applyTarget, MCTSNodeStruct.MAX_NUM_VISITS_PENDING_TRANSPOSITION_ROOT_EXTRACTION);

      Debug.Assert(applyTarget > 0);
      nodeRef.NumVisitsPendingTranspositionRootExtraction = applyTarget;

      EnsurePendingTranspositionValuesSet(node, false);
    }


    // TODO: make this a constant set in MCTSParamsFixed or MCTSSearch
    const int TRANSPOSITION_VALUE_REFRESH_INTERVAL = 1;



    /// <summary>
    /// Set or update the pending transposition values in a Node
    /// if they are missing or stale.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="transpositionRootNode"></param>
    public static void EnsurePendingTranspositionValuesSet(MCTSNode node, bool possiblyRefresh)
    {
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
        Debug.Assert(transpositionNodeIndex != 0);

        ref readonly MCTSNodeStruct transpositionNode = ref node.Context.Tree.Store.Nodes.nodes[transpositionNodeIndex];
        SetPendingTransitionValues(node, in transpositionNode);
        Debug.Assert(!float.IsNaN(node.PendingTranspositionV));
      }
    }


    /// <summary>
    /// Set the PendingTransposition values for this node based on speciifed transposition root.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="transpositionRootNode"></param>
    static void SetPendingTransitionValues(MCTSNode node,
                                           in MCTSNodeStruct transpositionRootNode)
    {
      Debug.Assert(node.N <= 2); // Max supported N in SetPendingTransitionValues is 3

      float FRAC_ROOT = node.Context.ParamsSearch.TranspositionRootBackupSubtreeFracs[node.N];

      float FRAC_POS = 1f - FRAC_ROOT;

      var visit0Ref = MCTSNodeStruct.SubnodeRefVisitedAtIndex(in transpositionRootNode, 0, out bool foundV0);

      // Helper method to set the PendingTransposition values from specified subnode.
      void SetNodePendingValues(in MCTSNodeStruct transpositionRootRef, float multiplier, in MCTSNodeStruct subnodeRef, bool subnodeRefIsValid)
      {
        Debug.Assert(subnodeRefIsValid);

        node.PendingTranspositionV = FRAC_POS * subnodeRef.V * multiplier
                                   + FRAC_ROOT * (float)transpositionRootRef.Q;

        node.PendingTranspositionM = FRAC_POS * subnodeRef.MPosition 
                                   + FRAC_ROOT * transpositionRootRef.MAvg;
        node.PendingTranspositionD = FRAC_POS * subnodeRef.DrawP 
                                   + FRAC_ROOT * transpositionRootRef.DAvg;
      }

      if (node.N == 0)
      {
        SetNodePendingValues(in transpositionRootNode, 1, in visit0Ref, foundV0);
      }
      else
      {
        var visit1Ref = MCTSNodeStruct.SubnodeRefVisitedAtIndex(in transpositionRootNode, 1, out bool foundV1);
        Debug.Assert(foundV1);

        if (node.N == 1)
        {
          SetNodePendingValues(in transpositionRootNode, -1, in visit1Ref, foundV1);
        }
        else if (node.N == 2)
        {
          var visit2Ref = MCTSNodeStruct.SubnodeRefVisitedAtIndex(in transpositionRootNode, 2, out bool foundV2);
          Debug.Assert(foundV2);
          float multiplier = visit2Ref.ParentIndex == transpositionRootNode.Index ? -1 : 1;

          SetNodePendingValues(in transpositionRootNode, multiplier, in visit2Ref, foundV2);
        }
        else
        {
          throw new Exception("Unexpected N in SetPendingTransitionValues");
        }
      }
    }

  }
}
