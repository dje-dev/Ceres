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
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GraphStores;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Miscellaneous helper methods relating to MCTSNodeStructs.
/// </summary>
public static class NodeUtils
{
  /// <summary>
  /// Modify values so that they sum to 1.0.
  /// </summary>
  /// <param name="values"></param>
  static void Normalize(Span<float> values)
  {
    float sum = 0;
    for (int i = 0; i < values.Length; i++)
    {
      sum += values[i];
    }
    for (int i = 0; i < values.Length; i++)
    {
      values[i] /= sum;
    }
  }


  /// <summary>
  /// Extracts a policy vector reflecting the empirical distribution of visits to the node.
  /// </summary>
  /// <param name="nodeRef"></param>
  /// <param name="policy"></param>
  public static void ExtractPolicyVectorFromVisitDistribution(in GNodeStruct nodeRef, ref CompressedPolicyVector policy)
  {
    Span<float> probabilities = stackalloc float[nodeRef.NumPolicyMoves];
    Span<int> indices = stackalloc int[nodeRef.NumPolicyMoves];

    for (int i = 0; i < nodeRef.NumPolicyMoves; i++)
    {
      if (i < nodeRef.NumEdgesExpanded)
      {
        ref readonly GNodeStruct childNode = ref nodeRef.ChildAtIndexRef(i);
        float rawProb = (float)childNode.N / nodeRef.N;
        probabilities[i] = Math.Max(CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE, rawProb);
        throw new NotImplementedException("Next line needs remediation");
        //indices[i] = (ushort)nodeRef.ChildAtIndexRef(i).PriorMove.IndexNeuralNet;
      }
      else
      {
        probabilities[i] = CompressedPolicyVector.DEFAULT_MIN_PROBABILITY_LEGAL_MOVE;
        throw new Exception("remediate next line");
        //indices[i] = (ushort)nodeRef.MoveInfos[i].Move.IndexNeuralNet;
      }
    }

    // Normalize probabilities and encode.
    Normalize(probabilities);

    SideType side = SideType.White; // TODO: This is not correct, not known. Somehow determine this.
    CompressedPolicyVector.Initialize(ref policy, side, indices, probabilities, false);
  }


  /// <summary>
  /// Extracts the policy vector output by network when a specified node was evaluated (without temperature).
  /// </summary>
  /// <param name="softmaxValue"></param>
  /// <param name="nodeRef"></param>
  /// <param name="policy"></param>
  [SkipLocalsInit]
  public static void ExtractPolicyVector(float softmaxValue, in GNodeStruct nodeRef, ref CompressedPolicyVector policy)
  {
    // Note: no benefit to sizing these spans smaller when possible, since no initialization cost (due to SkipLocalsInits).
    Span<ushort> indicies = stackalloc ushort[CompressedPolicyVector.NUM_MOVE_SLOTS];
    Span<float> probsBeforeNormalization = stackalloc float[CompressedPolicyVector.NUM_MOVE_SLOTS];
    Span<ushort> probabilities = stackalloc ushort[CompressedPolicyVector.NUM_MOVE_SLOTS];

    throw new NotImplementedException();
#if NOT
    GraphStore store = default;// nodeRef.Context.Store;
    Span<GEdgeHeaderStruct> children = store.EdgeHeadersStore.SpanAtIndex(nodeRef.ChildInfo.ChildInfoStartIndex(store), nodeRef.NumPolicyMoves);
    int numPolicyMoves = nodeRef.NumPolicyMoves;

    // Extract the probabilities, invert soft max, track sum.
    float accProbabilities = 0;
    for (int i = 0; i < numPolicyMoves; i++)
    {
      GEdgeHeaderStruct child = children[i];
      throw new NotImplementedException();
      if (true)//child.IsExpanded)
      {
//          ref readonly GPositionStruct childRef = ref child.ChildRef(store);
//          throw new NotImplementedException();
//          float prob = float.NaN;// MathF.Pow(childRef.P, softmaxValue);
//          probsBeforeNormalization[i] = prob;
//          accProbabilities += prob;
      }
      else
      {
        float prob = MathF.Pow(child.P, softmaxValue);
        probsBeforeNormalization[i] = prob;
        accProbabilities += prob;
      }
    }

    // Build spans of indices and probabilities (normalized and then encoded).
    float probScaling = 1.0f / accProbabilities;
    for (int i = 0; i < numPolicyMoves; i++)
    {
      GEdgeHeaderStruct child = children[i];
      throw new NotImplementedException();
      if (true)//(child.IsExpanded)
      {
        throw new NotImplementedException("Need to follow Child structs instead of MoveInfos");
//          ref readonly GPositionStruct childRef = ref child.ChildRef(store);
//          throw new NotImplementedException("next line needs remediation");
//          indicies[i] = default;// (ushort)childRef.PriorMove.IndexNeuralNet;
//          probabilities[i] = CompressedPolicyVector.EncodedProbability(probsBeforeNormalization[i] * probScaling);
      }
      else
      {
        indicies[i] = (ushort)child.Move.IndexNeuralNet;
        float prob = MathF.Pow(child.P, softmaxValue);
        probabilities[i] = CompressedPolicyVector.EncodedProbability(probsBeforeNormalization[i] * probScaling);
      }
    }


    if (nodeRef.NumPolicyMoves < CompressedPolicyVector.NUM_MOVE_SLOTS)
    {
      indicies[nodeRef.NumPolicyMoves] = CompressedPolicyVector.SPECIAL_VALUE_SENTINEL_TERMINATOR;
    }

    SideType side = SideType.White; // TODO: This is not correct, not known. Somehow determine this.
    CompressedPolicyVector.Initialize(ref policy, side, indicies, probabilities);
#endif
  }



  public static void VisitSubtreeBreadthFirst(Graph graph,
                                              MGPosition mgPos,
                                              NodeIndex nodeIndex,
                                              Func<GNode, MGPosition, bool> action)
  {
    MGPosition initialPos = mgPos;

    Queue<(NodeIndex node, MGPosition position)> queue = new();
    HashSet<NodeIndex> visited = [];

    queue.Enqueue((nodeIndex, initialPos));

    while (queue.Count > 0)
    {
      (NodeIndex currentIndex, MGPosition currentPos) = queue.Dequeue();

      if (!visited.Add(currentIndex))
      {
        // Already visited
        continue;
      }

      GNode node = graph[currentIndex];
      bool shouldContinue = action(node, currentPos);

      if (!shouldContinue)
      {
        continue;
      }

      foreach (GEdge childEdge in node.ChildEdgesExpanded)
      {
        MGPosition nextPos = currentPos;
        MGMove mgMove = MGMoveConverter.ToMGMove(in nextPos, childEdge.Move);
        nextPos.MakeMove(mgMove);

        if (childEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
        {
          queue.Enqueue((childEdge.ChildNode.Index, nextPos));
        }
      }
    }
  }


  // TODO: If EnableGraph is false then more efficient versions of:
  //         VisitSubtreeDepthFirst
  //         VisitSubtreeDepthFirst
  //       could be more written/used.
  public static void VisitSubtreeDepthFirst(Graph graph,
                                        MGPosition mgPos,
                                        NodeIndex nodeIndex,
                                        Func<GNode, MGPosition, bool> action)
  {
    MGPosition initialPos = mgPos;

    Stack<(NodeIndex node, MGPosition position)> stack = new();
    HashSet<NodeIndex> visited = [];

    stack.Push((nodeIndex, initialPos));

    while (stack.Count > 0)
    {
      (NodeIndex currentIndex, MGPosition currentPos) = stack.Pop();

      if (!visited.Add(currentIndex))
      {
        // Already visited
        continue;
      }

      GNode node = graph[currentIndex];
      bool shouldContinue = action(node, currentPos);

      if (!shouldContinue)
      {
        continue;
      }

      // To mimic recursive DFS order, you  push children in reverse order
      // so first child to be processed first.
      for (int i = node.NumEdgesExpanded - 1; i >= 0; i--)
      {
        GEdge childEdge = node.ChildEdgeAtIndex(i);

        MGPosition nextPos = currentPos;
        MGMove mgMove = MGMoveConverter.ToMGMove(in nextPos, childEdge.Move);
        nextPos.MakeMove(mgMove);

        stack.Push((childEdge.ChildNode.Index, nextPos));
      }
    }
  }

  /// <summary>
  /// 
  /// NOTE: Preliminary testing suggests this is correct implementation,
  ///       however the action must be multi-thread safe.
  /// </summary>
  /// <param name="store"></param>
  /// <param name="nodeIndex"></param>
  /// <param name="action"></param>
  /// <param name="parallelThresholdNumNodes"></param>
  public static void VisitSubtreeParallel(GraphStore store,
                                          NodeIndex nodeIndex,
                                          Action<NodeIndex> action,
                                          int parallelThresholdNumNodes)
  {
    throw new NotImplementedException();
#if NOT
    ref GPositionStruct nodeRef = ref store.Nodes.nodes[nodeIndex.Index];

    action(nodeIndex);

    ref readonly GPositionStruct node = ref store.Nodes.nodes[nodeIndex.Index];
    Span<GMoveInfoStruct> span = store.MoveInfos.SpanAtIndex(node.ChildInfo.ChildInfoStartIndex(store), node.NumPolicyMoves);

    // Only spawn another thread if we have a child which is
    //   - not too big (otherwise we should do it inline to avoid recursively repeated forks)
    //   - not too small (otherwise not worth the effort of spawning)
    int thresholdParallelMinThisLevel = nodeRef.N / 3;
    int thresholdParallelMaxThisLevel = nodeRef.N * 2 / 3;
    foreach (GMoveInfoStruct child in span)
    {
      if (child.IsExpanded)
      {
        ref GPositionStruct childRef = ref store.Nodes.nodes[child.ChildNodeIndex.Index];

        if (childRef.N < parallelThresholdNumNodes)
        {
          // Too small, fall back to non-parallel version
          VisitSubtreeBreadthFirst(store, child.ChildNodeIndex, action);
        }
        else if (childRef.N >= thresholdParallelMinThisLevel
              && childRef.N < thresholdParallelMaxThisLevel)
        {
          ThreadPool.UnsafeQueueUserWorkItem((obj) => VisitSubtreeParallel(store, child.ChildNodeIndex, action, parallelThresholdNumNodes), false);
        }
        else
        {
          VisitSubtreeParallel(store, child.ChildNodeIndex, action, parallelThresholdNumNodes);
        }
      }

    }
#endif
  }
}
