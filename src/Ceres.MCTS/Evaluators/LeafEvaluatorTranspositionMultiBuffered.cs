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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;

using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Analysis;
using Ceres.MCTS.MTCSNodes.Struct;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Leaf evaluator which links to or extracts transposition nodes
  /// from another subtree. Only used for mode TranspositionMode.MultiNodeBuffered.
  /// </summary>
  public sealed partial class LeafEvaluatorTransposition : LeafEvaluatorBase
  {
    ConcurrentDictionary<int, MCTSNodeTranspositionVisitor> activeTranspositionVisitors 
      = new ConcurrentDictionary<int, MCTSNodeTranspositionVisitor>();

    private static LeafEvaluationResult ExtractTranspositionNodesFromSubtree(MCTSNode node, ref MCTSNodeStruct transpositionRootNode, 
                                                                             ref int numAlreadyLinked, MCTSNodeTranspositionVisitor linkedVisitor)
    {
      LeafEvaluationResult result = default;

      // Determine how many evaluations we should extract (based on number requested and number available)
      int numAvailable = linkedVisitor.TranspositionRootNWhenVisitsStarted - numAlreadyLinked;
      int numDesired = node.NInFlight + node.NInFlight2;
      if (numDesired > numAvailable && WARN_COUNT < 10)
      {
        Console.WriteLine(numDesired + " Warning: multiple nodes were requested from the transposition subtree, available " + numAvailable);
        WARN_COUNT++;
      }
      int numToFetch = Math.Min(numDesired, numAvailable);
      Debug.Assert(numToFetch > 0);

      // Extract each evaluation
      for (int i = 0; i < numToFetch; i++)
      {
        MCTSNodeStructIndex transpositionSubnodeIndex = linkedVisitor.Visitor.GetNext();
        Debug.Assert(!transpositionSubnodeIndex.IsNull);

        numAlreadyLinked++;

        // Prepare the result to return
        ref MCTSNodeStruct transpositionSubnode = ref node.Store.Nodes.nodes[transpositionSubnodeIndex.Index];
        LeafEvaluationResult thisResult = new LeafEvaluationResult(transpositionSubnode.Terminal, transpositionRootNode.WinP, 
                                                                   transpositionRootNode.LossP, transpositionRootNode.MPosition);

        // Update our result node to include this node
        result = AddResultToResults(result, numToFetch, i, thisResult);

        if (VERBOSE) Console.WriteLine($"ProcessAlreadyLinked {node.Index} yields {result.WinP} {result.LossP} via linked subnode root {transpositionRootNode.Index.Index} {transpositionRootNode} chose {transpositionSubnode.Index.Index}");

//        node.Ref.NumNodesTranspositionExtracted++;
      }

      return result;
    }

    private static LeafEvaluationResult AddResultToResults(LeafEvaluationResult result, int numToFetch, int i, LeafEvaluationResult thisResult)
    {
      throw new NotImplementedException();
#if NOT
      if (i == 0)
      {
        // First (and possibly only) result is place directly in the result field
        result = thisResult;
      }
      else
      {
        // Results after the first are put in the "ExtraResults" field
        if (result.ExtraResults == null) result.ExtraResults = new List<LeafEvaluationResult>(numToFetch - 1);
        result.ExtraResults.Add(thisResult);
      }

      return result;
#endif
    }

    const int THRESHOLD_STOP_LINKAGE = 100_000;

    private MCTSNodeStruct CheckNeedToMaterializeTranspositionSubtree(MCTSNode node, ref MCTSNodeStruct transpositionRootNode, int numAlreadyLinked, MCTSNodeTranspositionVisitor linkedVisitor)
    {
      if (numAlreadyLinked > THRESHOLD_STOP_LINKAGE || linkedVisitor.TranspositionRootNWhenVisitsStarted <= numAlreadyLinked)
      {
        if (VERBOSE)
        {
          if (true && node.N < 100)
            MCTSPosTreeNodeDumper.DumpAllNodes(node.Context, ref transpositionRootNode);
        }

        if (VERBOSE) Console.WriteLine("Cloning above node into " + node.StructRef);

        // We are almost exahusted in the trasposition subtree, 
        // so now we copy over the subtree so any subsequent visits to this node will continue descending
        int startNodes = node.Store.Nodes.NumUsedNodes;
        node.StructRef.CloneSubtree(node.Store, activeTranspositionVisitors, ref transpositionRootNode, numAlreadyLinked);
        if (VERBOSE) Console.WriteLine($"num allocated {node.Store.Nodes.NumUsedNodes - startNodes} " +
                        $"when cloning node of size {transpositionRootNode.N} with target {numAlreadyLinked} " +
                         $"total nodes now { node.Store.Nodes.NumUsedNodes}");

        // We have cloned and will use this tree directly in the future.
        // Delete the transposition visitor previously used
        if (numAlreadyLinked > 1)
          activeTranspositionVisitors.TryRemove(node.Index, out _);

        if (true && VERBOSE && node.N < 100)
        {
          Console.WriteLine("cloned node now looks like this:");
          MCTSPosTreeNodeDumper.DumpAllNodes(node.Context, ref node.StructRef);
          Console.WriteLine();
        }

        // Make sure we are now detached (this should have happened as part of the clone)
//        Debug.Assert(node.Ref.NumNodesTranspositionExtracted == 0);
        Debug.Assert(!node.IsTranspositionLinked);
      }

      return transpositionRootNode;
    }

    private MCTSNodeTranspositionVisitor GetOrCreateVisitorForNode(MCTSNode node, MCTSNodeStructIndex transpositionRootNodeIndex, ref MCTSNodeStruct transpositionRootNode)
    {
      // Lookup (or create new) MCTSNodeTranspositionVisitor for this node
      MCTSNodeTranspositionVisitor linkedVisitor;
      if (!activeTranspositionVisitors.TryGetValue(node.Index, out linkedVisitor))
      {
        linkedVisitor = new MCTSNodeTranspositionVisitor()
        {
          TranspositionRootNWhenVisitsStarted = transpositionRootNode.N,
          Visitor = new MCTSNodeIteratorInVisitOrder(node.Store, transpositionRootNodeIndex)
        };

        activeTranspositionVisitors[node.Index] = linkedVisitor;
        MCTSNodeStructIndex firstVisitedNodeIndex = linkedVisitor.Visitor.GetNext(); // we already applied first, consume it now (to skip it)
#if DEBUG
        if (node.Tree.Store.Nodes.nodes[firstVisitedNodeIndex.Index].V != node.V)
          Console.WriteLine("wrong " + node.Tree.Store.Nodes.nodes[firstVisitedNodeIndex.Index].V + " " + node.V);
#endif
      }

      return linkedVisitor;
    }

    internal LeafEvaluationResult ProcessAlreadyLinked(MCTSNode node, MCTSNodeStructIndex transpositionRootNodeIndex, ref MCTSNodeStruct transpositionRootNode)
    {
      throw new NotImplementedException();
      int numAlreadyLinked = 0;// ********* node.Ref.NumNodesTranspositionExtracted;
      Debug.Assert(numAlreadyLinked > 0);

      // Lookup (or create if never used before) a visitor which can traverse starting at the linked transposition subroot
      MCTSNodeTranspositionVisitor linkedVisitor = GetOrCreateVisitorForNode(node, transpositionRootNodeIndex, ref transpositionRootNode);

      LeafEvaluationResult result = ExtractTranspositionNodesFromSubtree(node, ref transpositionRootNode, ref numAlreadyLinked, linkedVisitor);

      // Possibly copy over the subtree and delink if it became too large
      transpositionRootNode = CheckNeedToMaterializeTranspositionSubtree(node, ref transpositionRootNode, numAlreadyLinked, linkedVisitor);

      return result;
    }


  }
}
