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
using Ceres.MCTS.MTCSNodes;
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

      // Compute the number of times to apply, first compute target based on fixed and fractional components.
      int applyTarget = paramsSearch.MaxTranspositionRootApplicationsFixed;
      applyTarget += (int)Math.Round(transpositionRootNode.N * paramsSearch.MaxTranspositionRootApplicationsFraction, 0);

      // But never allow reuse more than the number of visits to the root.
      applyTarget = Math.Min(applyTarget, transpositionRootNode.N);

      // Also apply final max value specified in parameters.
      applyTarget = Math.Min(applyTarget, paramsSearch.MaxTranspositionRootReuse);

      // Finally, the field holding the target has a fixed maximum representable size, ensure not more than that.
      applyTarget = Math.Min(applyTarget, MCTSNodeStruct.MAX_NUM_VISITS_PENDING_TRANSPOSITION_ROOT_EXTRACTION);

      //      if (applyTarget > 1 && applyTarget == transpositionRootNode.N)
      //      {
      //        applyTarget--;
      //      }

      // Repeatedly sampling the same leaf node value is not a reasonable strategy.
//      Debug.Assert(applyTarget <= 1 || paramsSearch.TranspositionUseTransposedQ);

      // If the root has far more visits than our fixed target
      // then only use it once (since it represents a value based on a very different subtree).
      if (ForceUseV(node, in transpositionRootNode))
      {
//        applyTarget = 1;
      }

      nodeRef.NumVisitsPendingTranspositionRootExtraction = applyTarget;

      EnsurePendingTranspositionValuesSet(node, false);
    }


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
        if (node.N > 2) throw new Exception("SetPendingTransitionValues not supporting N > 2");

        //        float wtQ = (0.5f / MathF.Pow(transpositionRootNode.N, 0.5f));
        float wtQ = 1f / MathF.Pow(transpositionRootNode.N, 0.75f);
        float wtV = 1.0f - wtQ;

        if (node.N == 2)
        {
          ref MCTSNodeStruct nullNode = ref node.Context.Root.Ref;
          //var visit0Ref = MCTSNodeStruct.SubnodeRefVisitedAtIndex(in node.Ref, 0, in nullNode);
          var visit1Ref = MCTSNodeStruct.SubnodeRefVisitedAtIndex(in node.Ref, 1, in nullNode);
          var visit2Ref = MCTSNodeStruct.SubnodeRefVisitedAtIndex(in node.Ref, 2, in nullNode);

          float v1 = visit1Ref.Index.Index == 0 ? node.PendingTranspositionV : visit1Ref.V;
          float v2 = visit2Ref.Index.Index == 0 ? node.PendingTranspositionV : visit2Ref.V;
          node.PendingTranspositionV = 0.5f * (v1 + v2);
          node.PendingTranspositionM = transpositionRootNode.MPosition;
          node.PendingTranspositionD = transpositionRootNode.DrawP;
        }
        else if (node.N == 1)
        {
          if (transpositionRootNode.NumChildrenExpanded > 0
           && transpositionRootNode.ChildAtIndexRef(0).Terminal != Chess.GameResult.NotInitialized
           && !transpositionRootNode.ChildAtIndexRef(0).IsTranspositionLinked)
          {
            //            Console.WriteLine(transpositionRootNode.V + " " + -transpositionRootNode.ChildAtIndexRef(0).V);
            float priorQ = (float)transpositionRootNode.Q;
            float priorV = (float)transpositionRootNode.V;
            transpositionRootNode = ref transpositionRootNode.ChildAtIndexRef(0);

wtQ = 0;
wtV = 1;
            node.PendingTranspositionV = wtV * (0.5f * -transpositionRootNode.V + 0.5f * priorV);
            //            node.PendingTranspositionV = wtQ * priorq + wtV * -transpositionRootNode.V;
            node.PendingTranspositionM = transpositionRootNode.MPosition;
            node.PendingTranspositionD = transpositionRootNode.DrawP;
            return;
          }
        }

wtQ = 1;
wtV = 0;

        node.PendingTranspositionV = wtV * transpositionRootNode.V + wtQ * (float)transpositionRootNode.Q;
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
      return !node.Context.ParamsSearch.TranspositionUseTransposedQ;
    }


  }
}
