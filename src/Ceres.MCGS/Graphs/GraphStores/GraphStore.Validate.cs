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
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PathEvaluators;
using Ceres.MCGS.Search.Phases.Evaluation;

#endregion

namespace Ceres.MCGS.Graphs.GraphStores;

/// <summary>
/// A store of MCGS graph nodes (and associated parent/children information) 
/// representing the state of a Monte Carlo graph search.
/// </summary>

internal class GraphStoreValidator
{
  readonly GraphStore store;
  int numWarnings;
  readonly Graph graph;
  readonly bool quiescent;


  internal GraphStoreValidator(GraphStore store, Graph graph, bool quiescent = true)
  {
    this.store = store;
    this.graph = graph;
    this.quiescent = quiescent;
  }


  /// <summary>
  /// Asserts on a node (only if validation mode is quiescent).
  /// </summary>
  /// <param name="condition"></param>
  /// <param name="err"></param>
  /// <param name="nodeIndex"></param>
  /// <param name="warnOnly"></param>
  void AssertNodeQuiescent(bool condition, Func<string> err, int nodeIndex, bool warnOnly = false)
    => AssertNodeQuiescent(condition, err, nodeIndex, in store.NodesStore.Span[nodeIndex], warnOnly);
  

  void AssertNode(bool condition, Func<string> err, int nodeIndex, bool warnOnly = false)
  {
    if (!condition)
    {
      AssertNode(false, err(), nodeIndex, in store.NodesStore.Span[nodeIndex], warnOnly);
    }
  }

  void Assert(bool condition, string err)
  {
    if (!condition)
    {
      throw new Exception($"MCGSNodeStore::Validate failed on graph with RootN={graph.GraphRootNode.N}, NumNodes={graph.Store.NodesStore.NumUsedNodes}: {err} ");
    }
  }

 
  void AssertNodeQuiescent(bool condition, Func<string> err, int nodeIndex, in GNodeStruct node, bool warnOnly = false)
  {
    if (quiescent)
    {
      AssertNode(condition, err, nodeIndex, warnOnly);
    }
  }


  void AssertNode(bool condition, string err, int nodeIndex, in GNodeStruct node, bool warnOnly = false)
  {
    if (!condition)
    {
      string errStr = $"MCGSNodeStore::Validate failed on graph with RootN={graph.GraphRootNode.N}, NumNodes={graph.Store.NodesStore.NumUsedNodes}: {err} on node: #{nodeIndex} {node.Terminal} ";
      if (warnOnly)
      {
        if (numWarnings == 0)
        {
          Console.WriteLine(errStr);
          Console.WriteLine("NOTE: Suppressing subsequent warnings for validation of graph.");
        }
        numWarnings++;
      }
      {
        throw new Exception(errStr);
      }
    }
  }


  void ValidateEdgeHeaderBlockIndexOrDeferredNode(int i, in GNodeStruct nodeR, GNode node, bool isQuiescent)
  {
    if (node.NodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull)
    {
      if (node.IsEvaluated && node.NumPolicyMoves > 0)
      {
        AssertNode(false, () => $"edgeHeaderBlockIndexOrDeferredNode is null on an evaluated node", i);
      }
    }
    else
    {
      if (node.NodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNodeIndex)
      {
        GNode referredToNode = graph[node.NodeRef.edgeHeaderBlockIndexOrDeferredNode.NodeIndex];
        AssertNode(referredToNode.IsEvaluated, () => $"edgeHeaderBlockIndexOrDeferredNode referred to #{referredToNode.Index} but was not evaluated", i);
      }
      else
      {
        int edgeHeaderIndexItem = node.NodeRef.edgeHeaderBlockIndexOrDeferredNode.BlockIndexIntoEdgeHeaderStore;
        long numAllocatedEdgeHeaderBlocks = graph.EdgeHeadersStore.NumAllocatedItems / GEdgeHeadersStore.NUM_EDGE_HEADERS_PER_BLOCK;
        AssertNode(edgeHeaderIndexItem <= numAllocatedEdgeHeaderBlocks, () => $"edgeHeaderBlockIndexOrDeferredNode too large, had value {edgeHeaderIndexItem} but only {numAllocatedEdgeHeaderBlocks} allocated", i);
      }
    }
  }


  void ValidateNodeFields(int i, in GNodeStruct nodeR, GNode node, bool isQuiescent)
  {

    if (node.DrawKnownToExistAmongChildren)
    {
      bool foundDraw = false;
      foreach (GEdge edge in node.ChildEdgesExpanded)
      {
        if (edge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn)
        {
          foundDraw = true;
          break;
        }
      }

      AssertNode(foundDraw, () => $"Node #{i} had DrawKnownToExistAmongChildren but no children were TerminaEdgeDrawn", i, false);
    }

    // Note that we ignore locked nodes if they were orphaned
    // (created but never used, as can happen due to use of GetOrAdd in Graph.cs).
    if (isQuiescent)
    {
      AssertNode(!node.IsLocked, () => $"Node #{i} still locked in expected quiescent graph.", i, false);
    }

    if (node.N > 0)
    {
      const float MAX_Q = 1.2f;
      AssertNode(Math.Abs(node.V) <= MAX_Q, () => $"Node #{i} has non-terminal V={node.V}", i, true);
      AssertNode(Math.Abs(node.Q) <= MAX_Q, () => $"Node #{i} has non-terminal Q={node.Q}", i, true);
    }

    if (isQuiescent && nodeR.N == 0)
    {
      AssertNode(double.IsNaN(node.Q), () => $"Node #{i} has N=0 but Q is not NaN", i, true);
    }
    else if (nodeR.N == 1 && !graph.NodesWithOneVisitMayHaveDifferentQ)
    {
      // TODO: Currently if a node was visited for the first time simultaneously by multiple parents
      //       then it is evaluated multiple times by the NN and the values may not match exactly
      //       because they arrive at the same node but have history fed to the NN which differs.
      //       Thus we use much larger threshold for now.
      float THRESHOLD = node.NumParents > 1 ? 0.20f : 0.005f;
      AssertNode(Math.Abs(node.Q - nodeR.V) < THRESHOLD, () => $"Node #{i} has N=1 but Q={node.Q} != V={node.V}", i, true);  
    }

    AssertNode(node.NumPieces >= 2 && nodeR.NumPieces <= 32, $"NumPieces {node.NumPieces} not in [2,32]", i, in nodeR, true);
    AssertNode(node.NumRank2Pawns <= 16, $"NumRank2Pawns {node.NumRank2Pawns} > 16", i, in nodeR, true);
    AssertNode(node.HashStandalone != default, $"HashStandalone uninitialized", i, in nodeR, true);

    if (nodeR.Terminal == GameResult.Draw)
    {
      AssertNode(node.Q == 0, () => $"Node #{i} marked as Draw but Q={node.Q}", i, true);
      AssertNode(node.WinP == 0, () => $"Node #{i} marked as Draw but WinP={node.WinP}", i,  true);
      AssertNode(node.LossP == 0, () => $"Node #{i} marked as Draw but LossP={node.LossP}", i,  true);
      AssertNode(node.UncertaintyValue == 0, () => $"Node #{i} marked as Draw but UncertaintyValue={node.UncertaintyValue}", i,  true);
    }
    else if (nodeR.Terminal == GameResult.Checkmate)
    {
//      AssertNode(node.Q < -0.999, () => $"Node #{i} marked as Checkmate but Q={node.Q}", i, true);
      AssertNode(Math.Abs(node.WinP + node.LossP) >= 1, () => $"Node #{i} marked as but WinP={node.WinP} LossP={node.LossP}", i, true);
      AssertNode(node.UncertaintyValue == 0, () => $"Node #{i} marked as Checkmate but UncertaintyValue={node.UncertaintyValue}", i, true);
    }

    AssertNode(node.N == 0 || !double.IsNaN(node.Q), () => $"Node #{i} has N={node.N} but Q is NaN", i, true);  

    // Visited nodes should have values for WinP/LossP
    AssertNode(!(node.N > 0 && FP16.IsNaN(node.WinP + node.LossP)), () => "N > 0 but WinP/LossP NaN", i, false);
    AssertNodeQuiescent(node.N == 0 || Math.Abs(node.Q) < 1.50, () => $"Node Q Q was unreasonable {node.Q}", i);

    if (node.N > 1 && node.NumEdgesExpanded == 0 && !node.Terminal.IsTerminal())
    {
      AssertNode(node.Terminal.IsTerminal(), () => $"NumEdgesExpanded=0 and not Terminal, expected N=1 but found N={node.N}", i, false);
    }

    ValidateEdgeHeaderBlockIndexOrDeferredNode(i, in nodeR, node, isQuiescent);
  }


  /// <summary>
  /// Diagnostic method which traverses full tree and performs 
  /// a variety of integrity checks on internal consistency,
  /// throwing an Exception if any fails.
  /// </summary>
  /// <param name="expectCacheIndexZero"></param>
  public unsafe void Validate(Graph graph, 
                              bool isQuiescent = true, 
                              int? singleNodeIndex = null, 
                              bool fastMode = false)
  {
    numWarnings = 0;

    Assert(store.NodesStore.nodes[0].N == 0, "Null node");
    Assert(store.NodesStore.nodes[GraphStore.ROOT_NODE_INDEX].IsGraphRoot, "IsRoot");

    int countNodesNoExpandedChildren = 0;
    int countSearchRootNodesFound = 0;

    int startNodeIndex = singleNodeIndex ?? 1;
    int endNodeIndex = singleNodeIndex.HasValue ? singleNodeIndex.Value + 1 
                                                : store.NodesStore.nextFreeIndex;
    // Validate all nodes
    // Note that we proceed from last (most recently added) node toward earliest (root)
    // which may help identify nodes with problems most proximal to their origin
    // (rather than a root level inconsistency which is merely a consequence of an error near the leaf).
    Parallel.For(startNodeIndex, endNodeIndex,
                 new ParallelOptions { MaxDegreeOfParallelism = 16 },
    i =>
    {
      int idx = endNodeIndex - 1 - (i - startNodeIndex);
      Span<GNodeStruct> nodes = store.NodesStore.Span;
      ref readonly GNodeStruct nodeR = ref nodes[idx];
      GNode node = graph[new NodeIndex(idx)];

      if (idx == 0)
      {
        AssertNode(node.BlockIndexIntoEdgeHeaderStore == 0, () => $"Null node (index #0) was not in null state", idx);
        return;
      }

      if (node.IsOldGeneration)
      {
        return;
      }

      if (node.IsSearchRoot)
      {
        Interlocked.Increment(ref countSearchRootNodesFound);
      }

      if (node.NumEdgesExpanded < 2)
      {
        Interlocked.Increment(ref countNodesNoExpandedChildren);
      }

      // Validate fields on the node itself.
      ValidateNodeFields(idx, in nodeR, node, isQuiescent);

      // Validate MoveInfos.
      node = ValidateEdgeHeadersAndEdges(idx, nodeR, node);

      // Validate Chlidren.
      node = ValidateChildEdges(isQuiescent, idx, nodeR, node);

      // Validate parent has a child that points to this node.
      node = VerifyHasParentWithThisAsChild(graph.GraphEnabled, idx, nodeR, node);

      // Verify all expanded children point back to ourself
      bool haveSeenUnexpanded = false;
      int sumN = 1;

#if NOT
      int childIndex = 0;
      foreach (GMoveInfoStruct childMoveInfo in node.MoveInfos)
      {
          if (childMoveInfo.IsExpanded)
          {
            ref GPositionStruct childRef = ref childMoveInfo.ChildRef(this);
            sumN += childRef.N;

            // Any expanded nodes should appear before all unexpanded nodes
            AssertNode(!haveSeenUnexpanded, "expanded after unexpanded", i, in nodeR);

            if (!graphEnabled)
            {
              AssertNode(node[childIndex].ChildRef.N <= nodeR.N, "child N", i, in nodeR);
              //              Assert(node[childIndex].ChildNode.Index == node.Index, "bug");
              //              AssertNode(node[childIndex].ChildRef.ParentIndex == node.Index, $"ParentRef is {node[childIndex].ChildRef.ParentIndex}", i, in nodeR);
            }

            Assert(node[childIndex].ChildNode.VisitsFromNodes.Contains(new GPosition(graph, node.Index)), "ChildNode did not have a VisitFrom pointing back to this node");

            if (!node.IsRoot)
            {
              // Verify the child lists us as a parent.
              //AssertNode(VisitsFrom.ParentHasChild(node.Index, childMoveInfo.ChildNodeIndex), "Child did not list node as parent", i, in nodeR);
            }

            numExpanded++;
          }
          else
          {
            haveSeenUnexpanded = true;
          }
        childIndex++;
      }

      AssertNode(nodeR.NumPolicyMoves == childIndex, "NumPolicyMoves", i, in nodeR);
#endif
    });

    Assert(countSearchRootNodesFound <= 1, $"More than one node was marked as SearchRoot (count={countSearchRootNodesFound})");


    if (!fastMode)
    {
      // Validate we can traverse tree
      // (this indirectly tests if all move sequences are valid).
      // TODO: improve this; the logic of VisitSubtreeBreadthFirst into this class
      //       so we can better try/catch on any edges for which the child move was invalid
      int numNodesFound = 0;
      HashSet<string> foundFENs = graph.GraphEnabled ? new(store.NodesStore.NumUsedNodes) : null;
      NodeUtils.VisitSubtreeBreadthFirst(graph, store.HistoryHashes.PriorPositionsMG[^1], new NodeIndex(GraphStore.ROOT_NODE_INDEX),
                                         (GNode node, MGPosition mgPos) =>
                                         {
                                           //                                         Console.WriteLine("Tree visit " + node + " " + mgPos.ToPosition.FEN);
                                           numNodesFound++;
#if NOT
                                         if (graph.GraphEnabled)
                                         {
                                           if (foundFENs.Contains(mgPos.ToPosition.FEN))
                                           {
                                             // TODO: Remove this? It is not a valid test.
                                             //       Because we may have multiple paths to the same position
                                             //       (the hash used is PositionAndSequence) it is perfectly valid
                                             //       for the same FEN to appear multiple times.
                                             AssertNode(false, () => $"Duplicate FEN found in traversal: {mgPos.ToPosition.FEN}", node.NodeRef.Index.Index, true);
                                           }
                                           foundFENs.Add(mgPos.ToPosition.FEN);
                                         }
#endif
                                           return true;
                                         });
      Assert(numNodesFound > 0, "No nodes found in traversal!");
    }

    if (numWarnings > 0)
    {
      Console.WriteLine($"Number of graph validation warnings: {numWarnings}");
    }
  }


  private unsafe GNode ValidateChildEdges(bool isQuiescent, int i, GNodeStruct nodeR, GNode node)
  {
    Span<GEdgeHeaderStruct> headerStructs = node.EdgeHeadersSpan;

    int childIndexInParent = 0;
    int sumEdgeN = 0;
    double sumEdgeW = 0;
    foreach (GEdge childEdge in node.ChildEdgesExpanded)
    {
      sumEdgeN += childEdge.N;
      sumEdgeW += childEdge.N * childEdge.Q;

      GEdgeHeaderStruct edgeHeaderStruct = headerStructs[childIndexInParent];
      AssertNode(childEdge.IsExpanded, "Unexpanded edge returned by ChildEdgesExpanded", i, in nodeR);

      if (childEdge.ChildNodeHasDrawKnownToExist)
      {
        AssertNode(childEdge.ChildNode.DrawKnownToExistAmongChildren, () => $"Edge {childEdge} had ChildNodeHasDrawKnownToExist but the child node had false for DrawKnownToExistAmongChildren", i);
      }

      // Verify equality of fields copied over from GEdgeHeader into GEdge that are also retained in GEdgeHeader
      if (!childEdge.Type.IsTerminal())
      {
        int nExpectedFromChild = childEdge.N - childEdge.NDrawByRepetition;
        AssertNode(nExpectedFromChild <= childEdge.ChildNode.N, () => $"Edge N exceeded child N ({childEdge.N} vs {childEdge.ChildNode.N})", childEdge.ChildNode.Index.Index);

#if ACTION_ENABLED
        if (MCGSParamsFixed.GEDGE_HAS_ACTIONV)
        { 
          // Do not apply this test if terminal edge, since the action V will have been overwritten in the edge
          // but currently that overwriting is not replicated at the GEdgeHeaderStruct level
          AssertNode(MathUtils.EqualsOrBothNaN(childEdge.ActionV, edgeHeaderStruct.ActionV),
                                               () => $"Chlid/MoveInfo disagreement on ActionV of {childEdge.ActionV} vs {edgeHeaderStruct.ActionV} at move index {childIndexInParent}", i);
        // TODO: if/when we also copy over ActionU, then enable this    AssertNode(MathUtils.EqualsOrBothNaN(child.ActionU, moveInfo.ActionU), $"Chlid/MoveInfo disagreement on ActionU of {child.ActionU} vs {moveInfo.ActionU} at move index {visitToChildIndex}", i, in nodeR);
        }
#endif
      }

      if (!graph.GraphEnabled)
      {
        AssertNode(childEdge.ParentNode.NodeRef.N > childEdge.N, () => $"Parent's N {childEdge.ParentNode.NodeRef.N} expected greater than edge N of {childEdge.N} at child index {childIndexInParent}", i);
      }

      if (childEdge.Type == GEdgeStruct.EdgeType.Uninitialized)

      {
        AssertNode(false, $"Edge Type was uninitialized at child index {childIndexInParent}", i, in nodeR);
      }
      else if (childEdge.Type.IsTerminal())
      {
        // TODO: consider if this tolerance could/should be tightened
        AssertNodeQuiescent(Math.Abs(childEdge.Q) < EvaluatorSyzygy.BLESSED_WIN_LOSS_MAGNITUDE + 0.005 || Math.Abs(childEdge.Q) >= 0.99, () => $"Edge Type was terminal but Q was {childEdge.Q} at child index {childIndexInParent}", i);
        AssertNode(childEdge.UncertaintyV == 0, () => $"Edge Type was terminal but UncertaintyV was nonzero ({childEdge.UncertaintyV}) at child index {childIndexInParent}", i);
        AssertNode(childEdge.UncertaintyP == 0, () => $"Edge Type was terminal but UncertaintyP was nonzero ({childEdge.UncertaintyP}) at child index {childIndexInParent}", i);

        // This test disabled because it was decided in ConvertToTerminalDraw to leave the ChildNode 
        // which may point to the child node that created a cycle which triggered the conversion to draw edge.
        //AssertNode(childEdge.ChildNodeIndex.IsNull, () => $"Edge Type was terminal but ChildNode was not null at child index {childIndexInParent}", i);
      }
      else if (childEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        // TODO: This assumes complete synchrony between parent/child at all times,
        //       which may not always be possible since backups may not all-inclusively update the tree.
        GNode child = node.ChildEdgeAtIndex(childIndexInParent).ChildNode;

        if (!graph.GraphEnabled) // in graph mode each parent will have own Q value
        {
          const float THRESHOLD_NODE_EDGE_DELTA = 0.002f;
          AssertNode(Math.Abs(childEdge.Q - child.Q) < THRESHOLD_NODE_EDGE_DELTA, () => $"ChildEdge Q {childEdge.Q} did not agree with actual child Q {child.NodeRef.Q} at child index {childIndexInParent}", i);
        }

        // N.B. Is it possible that with graph enabled this test may fail despite even with correct code?
        //      Specifically, if it happened that multiple paths in a batch (coming from different parents)
        //      decide the child has sufficient N without realizing that the other edge
        //      exists and has in flight visits that are already being processed.
        int nExpectedFromChild = childEdge.N - childEdge.NDrawByRepetition;

        AssertNode(child.N >= nExpectedFromChild, () => $"Child's N {child.N} expected greater than or equal to edge N/NRP of {childEdge.N}/{childEdge.NDrawByRepetition} at child index {childIndexInParent}", i);

        if (!graph.GraphEnabled)// || childEdge.ChildNode.NumParents == 1)
        {
          AssertNode(childEdge.N == child.N, () => $"ChildEdge N {childEdge.N} did not agree with actual child N {child.NodeRef.N} at child index {childIndexInParent}", i);
        }
        else
        {
          if (childEdge.Q == 0)
          {
            // This looks like draw determined by repetition.
            // The edge many have been visited many times, but each time we did not take the value from the child
            // but instead just marked it as a draw directly. So the child N is not related to this edge N.
            // TODO: think about this more. Perhaps the edge needs a field for "was draw by repetition" so we can cleanly track this condition
            //       (what if W were coincidentally 0 for reasons unrelated to draw by repetition?).
          }
          else
          {
            if (!graph.GraphEnabled)
            {
              // When connecting to a child transposition node, child may not have exactly same number of visits as the edge (but not less)
              AssertNode(childEdge.N <= child.NodeRef.N, () => $"VisitTo N {childEdge.N} was greater than actual child N {child.NodeRef.N}", i);
            }
          }
        }
        //AssertNode(visitTo.P == node[visitToChildIndex].ChildRef.P, $"VisitTo P {visitTo.P} did not agree with actual child P {node[visitToChildIndex].ChildRef.P}", i, in nodeR);      
      }

      childIndexInParent++;
    }

    if (isQuiescent && node.N > 0)
    {
      AssertNode(node.N == sumEdgeN + 1, () => $"node with N= {node.N} had sum(edgeN)= {sumEdgeN}", i);

      // Verify node Q agrees with self and accumulated edges.
      int nToUseForSelf = (nodeR.Terminal.IsTerminal() ? nodeR.N : 1); ;
      int sumN = sumEdgeN + nToUseForSelf;
      double sumW = -sumEdgeW + nodeR.V * nToUseForSelf;

      double correctQ = sumW / sumN;  
      double errAbsQ = Math.Abs(nodeR.Q - correctQ);
      // RESTORE? AssertNode(false, () => $"node Q {nodeR.Q} disagreed with sum using (self + edges) Q values {nodeR.Q} {correctQ} (abs err={errAbsQ})", i);      
    }

    AssertNode(childIndexInParent == nodeR.NumEdgesExpanded, "NumChildrenExpanded inconsistent with number of populated entries in ChildVisits", i, in nodeR);
    //        AssertNode(numVisitedThisChild == nodeR.NumChildrenVisited, "NumChildrenVisited inconsistent with number of populated entries in ChildVisits", i, in nodeR);

    AssertNode(!(node.Terminal.IsTerminal() && node.NumEdgesExpanded > 0), "Terminal node should have no expanded children", i, in nodeR, true);

    // Verify all visits to children point have correct statistics
    int sumChildEdgeN = 0;
    for (int c = 0; c < node.NumEdgesExpanded; c++)
    {
      GEdge childEdge = node.ChildEdgeAtIndex(c);
      sumChildEdgeN += childEdge.N;

      if (!childEdge.Type.IsTerminal())
      {
        GNode childNode = childEdge.ChildNode;
//        if (!graph.GraphEnabled)
        {
          AssertNode(childNode.N >= childEdge.N - childEdge.NDrawByRepetition , () => $"child node N {childNode.N} was not large enough to support edge visits of N/ND={childEdge.N}/{childEdge.NDrawByRepetition}", i);
        }

        // Update the AssertNode calls with the new equality check
        if (false)
        {
          // TODO: Restore this!
          //       Currently disabled, may be bug where change to uncertainty
          //       due to discovered force win/draw/loss is not yet
          //       propagated also to the VisitTo structure.
          AssertNode(MathUtils.EqualsOrBothNaN((float)childNode.NodeRef.UncertaintyValue, (float)childEdge.UncertaintyV), "Non-matching UncertaintyValue for Node vs. VisitTo", i, in nodeR);
          AssertNode(MathUtils.EqualsOrBothNaN((float)childNode.NodeRef.UncertaintyPolicy, (float)childEdge.UncertaintyP), "Non-matching UncertaintyPolicy for Node vs. VisitTo", i, in nodeR);
          //AssertNode(MathUtils.EqualsOrBothNaN((float)childNode.NodeRef.ActionV, (float)visitTo.ActionV), "Non-matching ActionV for VisitToChild", i, in nodeR);
          //AssertNode(MathUtils.EqualsOrBothNaN((float)childNode.NodeRef.ActionU, (float)visitTo.ActionU), "Non-matching ActionU for VisitToChild", i, in nodeR);
        }

        if (childNode.NodeRef.N != 0)
        {
          AssertNode(!double.IsNaN(childEdge.Q), "Q is NaN in VisitTo despite N>0", i, in nodeR);
          //AssertNode(!float.IsNaN(childEdge.DSum), "DSum is NaN VisitTo despite N>0", i, in nodeR);
        }

        // Each node sees its own set of (potentially differing) visits and statistics to a child. So limited validation possible
        if (!graph.GraphEnabled)
        {
          AssertNode(childNode.NodeRef.N >= childEdge.N - 1, "Parent visits to child cannot exceed number of evaluations of child", i, in nodeR);
        }
      }

      if (isQuiescent)
      {
        AssertNode(childEdge.NumInFlight0 == 0, () => $"NumInFlight0 = {childEdge.NumInFlight0} != 0 for child edge {c}", i);
        AssertNode(childEdge.NumInFlight1 == 0, () => $"NumInFlight1 = {childEdge.NumInFlight1} != 0 for child edge {c}", i);
      }
    }

    // N on the node must be at least as large as sum of children visits + 1 (for itself)
    if (nodeR.Terminal.IsTerminal())
    {
      AssertNode(sumChildEdgeN == 0, () => $"Node was terminal but child edges exist with {sumChildEdgeN} total visits", i);
    }
    else if (nodeR.IsEvaluated && isQuiescent)
    {
      AssertNode(nodeR.N == 1 + sumChildEdgeN, () => $"N {nodeR.N} does not match sum of child edges plus one of {sumChildEdgeN + 1}", i);
    }
    return node;
  }


  private unsafe GNode VerifyHasParentWithThisAsChild(bool graphEnabled, int i, GNodeStruct nodeR, GNode node)
  {
    bool foundParentPointsToNode = false;
    if (!node.IsGraphRoot && !graphEnabled)
    {
      foreach (GNode parentNode in node.Parents)
      {
        if (!foundParentPointsToNode)
        {
          foreach (GEdge child in parentNode.ChildEdgesExpanded)
          {
            if (child.ChildNodeIndex == node.Index)
            {
              foundParentPointsToNode = true;
              break;
            }
          }
        }
      }
      AssertNode(foundParentPointsToNode, "No parent had a VisitTo to this node", i, in nodeR);
    }

    return node;
  }


  private unsafe GNode ValidateEdgeHeadersAndEdges(int i, GNodeStruct nodeR, GNode node)
  {
    // Verify correct number of GEdgeHeaderStruct.
    AssertNode(node.EdgeHeadersSpan.Length == node.NumPolicyMoves, "Length of EdgeHeadersSpan differs from number of policy moves", i, in nodeR);

    // Check EdgeHeadersSpan structures.
    float lastP = float.MaxValue;
    bool haveSeenUnexpanded = false;
    int headerIndex = 0;
    foreach (GEdgeHeaderStruct gEdgeHeaderStruct in node.EdgeHeadersSpan)
    {
      // Verify MoveInfo is initialized.
      if (gEdgeHeaderStruct.IsUnintialized)
      {
        AssertNode(false, "GEdgeHeaderStruct was not initialized", i, in nodeR);
      }

      if (gEdgeHeaderStruct.IsExpanded)
      {
        if (haveSeenUnexpanded)
        {
          AssertNode(!haveSeenUnexpanded, "Expanded child appears before unexpanded node", i, in nodeR);
        }

        Span<GEdgeStruct> edgesSpan = store.EdgesStore.SpanAtBlockIndex(gEdgeHeaderStruct.EdgeStoreBlockIndex);
        int indexInBlock = headerIndex % GEdgeStore.NUM_EDGES_PER_BLOCK;
        GEdge childEdge = node.ChildEdgeAtIndex(headerIndex);
        AssertNode(childEdge.ParentNode.Index == node.Index, $"ParentNode of child specified in edge does not match", i, in nodeR);

        // P and Move are not available (overwritten by edge block index)
        // But ActionV is still preserved
        if (childEdge.Type.IsTerminal())
        {
          if (childEdge.Type == GEdgeStruct.EdgeType.TerminalEdgeDrawn)
          {
            // TODO: consider if these tolerances could/should be tightened
            AssertNodeQuiescent(Math.Abs(childEdge.Q) < EvaluatorSyzygy.BLESSED_WIN_LOSS_MAGNITUDE + 0.005, () => $"Terminal edge's Q was {childEdge.Q} but Type is TerminalEdgeDrawn", i);
          }
          else if (childEdge.Type == GEdgeStruct.EdgeType.TerminalEdgeDecisive)
          {
            // TODO: consider if these tolerances could/should be tightened
            AssertNodeQuiescent(Math.Abs(childEdge.Q) >= 1, () => $"Terminal edge's Q was {childEdge.Q} but Type is TerminalDecisive", i);
            AssertNodeQuiescent(Math.Abs(childEdge.Q) < 1.50, () => $"Terminal edge's Q was unreasonable {childEdge.Q} but Type is TerminalDecisive", i);
          }
          else
          {
            throw new Exception("Unknown terminal edge type");  
          }
        }
        else
        {
          // Do not apply this test if terminal edge, since the action V will have been overwritten in the edge
          // but currently that overwriting is not replicated at the GEdgeHeaderStruct level
#if ACTION_ENABLED
          if (MCGSParamsFixed.GEDGE_HAS_ACTIONV)
          {
            AssertNode(MathUtils.EqualsOrBothNaN(childEdge.ActionV, gEdgeHeaderStruct.ActionV), $"ActionV of child specified in edge does not match", i, in nodeR);
          }
#endif
          AssertNodeQuiescent(Math.Abs(childEdge.Q) < 1.50, () => $"Child edge's Q was unreasonable {childEdge.Q}", i);
          
        }
        // if we someday also copy ActionU: AssertNode(MathUtils.EqualsOrBothNaN(childEdge.ActionU, gEdgeHeaderStruct.ActionU), $"ActionU of child specified in edge does not match", i, in nodeR);

        // If we appeared not at the first offset in the allocated block,
        // verify all prior block entries are initialized
        for (int priorBlockIndex = 0; priorBlockIndex < indexInBlock; priorBlockIndex++)
        {
          if (edgesSpan[priorBlockIndex].Type == GEdgeStruct.EdgeType.ChildEdge)
          {
            AssertNode(edgesSpan[priorBlockIndex].ChildNodeIndex != default, "Unpopulated block entry appeared before populated block entry", i, in nodeR);
          }
        }

        // If we are last child and appeared not at the last offset in the allocated block,
        // verify all subsequent block entries are uninitialized
        bool thisChildLast = headerIndex == node.NumEdgesExpanded - 1;
        if (thisChildLast)
        {
          for (int emptyBlockIndex = indexInBlock + 1; emptyBlockIndex < GEdgeStore.NUM_EDGES_PER_BLOCK; emptyBlockIndex++)
          {
            AssertNode(edgesSpan[emptyBlockIndex].ChildNodeIndex == default, "More populated block entries found than expected", i, in nodeR);          
          }
        }
      }
      else
      {
        haveSeenUnexpanded = true;

        // Verify policy non-negative.
        AssertNode(gEdgeHeaderStruct.P >= 0, () => $"MoveInfo policy was negative: {gEdgeHeaderStruct.P}", i);

        // Verify unexpanded edges appear in non-ascending order by prior probability (unless action head enabled).
#if !ACTION_ENABLED
        AssertNode(gEdgeHeaderStruct.P <= lastP, () => $"Unexpanded children were not in non-ascending order by prior probability: {gEdgeHeaderStruct.P} vs. {lastP}", i);
#endif
        lastP = gEdgeHeaderStruct.P;
      }

      headerIndex++;
    }

    return node;
  }
}
