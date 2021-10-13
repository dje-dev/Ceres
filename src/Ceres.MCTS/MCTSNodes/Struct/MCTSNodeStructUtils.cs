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
using System.Runtime.CompilerServices;
using System.Threading;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.EncodedPositions;
using Ceres.MCTS.MTCSNodes.Storage;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  /// <summary>
  /// Miscellaneous helper methods relating to MCTSNodeStructs.
  /// </summary>
  public static class MCTSNodeStructUtils
  {
    [SkipLocalsInit]
    public static void ExtractPolicyVector(float softmaxValue, in MCTSNodeStruct nodeRef, ref CompressedPolicyVector policy)
    {
      Span<ushort> indicies = stackalloc ushort[CompressedPolicyVector.NUM_MOVE_SLOTS];
      Span<float> probsBeforeNormalization = stackalloc float[CompressedPolicyVector.NUM_MOVE_SLOTS];
      Span<ushort> probabilities = stackalloc ushort[CompressedPolicyVector.NUM_MOVE_SLOTS];

      Span<MCTSNodeStructChild> children = MCTSNodeStoreContext.Store.Children.SpanForNode(in nodeRef);
      int numPolicyMoves = nodeRef.NumPolicyMoves;
 
      // Extract the probabilities, invert soft max, track sum.
      float accProbabilities = 0;
      for (int i = 0; i < numPolicyMoves; i++)
      {
        MCTSNodeStructChild child = children[i];
        if (child.IsExpanded)
        {
          ref readonly MCTSNodeStruct childRef = ref child.ChildRef;
          float prob = MathF.Pow(childRef.P, softmaxValue);
          probsBeforeNormalization[i] = prob;
          accProbabilities += prob;
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
        MCTSNodeStructChild child = children[i];
        if (child.IsExpanded)
        {
          ref readonly MCTSNodeStruct childRef = ref child.ChildRef;
          indicies[i] = (ushort)childRef.PriorMove.IndexNeuralNet;
          probabilities[i] = CompressedPolicyVector.EncodedProbability(probsBeforeNormalization[i] * probScaling);
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

      CompressedPolicyVector.Initialize(ref policy, indicies, probabilities);
    }


    public static void Dump(ref MCTSNodeStruct node, bool childDetail)
    {
      Console.WriteLine($"{node.Index,7} {node.PriorMove} {node.V,6:F2} {node.W,9:F2} Parent={node.ParentIndex} ChildStartIndex={node.ChildStartIndex}");

      if (childDetail)
      {
        int childIndex = 0;
        foreach (MCTSNodeStructChild child in node.Children)
        {
          Console.Write($"          [{node.ChildStartIndex + childIndex++,8}] ");
          if (child.IsExpanded)
          {
            Console.WriteLine($"{child.ChildIndex} --> {child.ChildRef.ToString()}");
          }
          else
          {
            Console.WriteLine($"{child.Move} {child.P} ");
          }
        }
      }
      Console.WriteLine();
    }

    public static void VisitNodesSequentially(ref MCTSNodeStruct root, Action<MCTSNodeStructIndex> action)
    {
      MCTSNodeStore store = MCTSNodeStoreContext.Store;

      for (int i = 1; i < store.Nodes.NumTotalNodes; i++)
      {
        action(new MCTSNodeStructIndex(i));
      }
    }

    public static void VisitSubtreeBreadthFirst(MCTSNodeStore store, MCTSNodeStructIndex nodeIndex, Action<MCTSNodeStructIndex> action)
    {
      action(nodeIndex);

      foreach (MCTSNodeStructChild child in store.Children.SpanForNode(in store.Nodes.nodes[nodeIndex.Index]))
      {
        if (child.IsExpanded)
          VisitSubtreeBreadthFirst(store, child.ChildIndex, action);
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
    public static void VisitSubtreeParallel(MCTSNodeStore store,
                                            MCTSNodeStructIndex nodeIndex,
                                            Action<MCTSNodeStructIndex> action,
                                            int parallelThresholdNumNodes)
    {
      ref MCTSNodeStruct nodeRef = ref store.Nodes.nodes[nodeIndex.Index];

      action(nodeIndex);

      Span<MCTSNodeStructChild> span = store.Children.SpanForNode(in store.Nodes.nodes[nodeIndex.Index]);

      // Only spawn another thread if we have a chlid which is
      //   - not too big (otherwise we should do it inline to avoid recursively repeated forks)
      //   - not too small (otherwise not worth the effort of spawning)
      int thresholdParallelMinThisLevel = nodeRef.N / 3;
      int thresholdParallelMaxThisLevel = (nodeRef.N * 2) / 3;
      foreach (MCTSNodeStructChild child in span)
      {
        if (child.IsExpanded)
        {
          ref MCTSNodeStruct childRef = ref store.Nodes.nodes[child.ChildIndex.Index];

          if (childRef.N < parallelThresholdNumNodes)
          {
            // Too small, fall back to non-parallel version
            VisitSubtreeBreadthFirst(store, child.ChildIndex, action);
          }
          else if (childRef.N >= thresholdParallelMinThisLevel
                && childRef.N < thresholdParallelMaxThisLevel)
          {
            ThreadPool.UnsafeQueueUserWorkItem((obj) => VisitSubtreeParallel(store, child.ChildIndex, action, parallelThresholdNumNodes), false);
          }
          else
          {
            VisitSubtreeParallel(store, child.ChildIndex, action, parallelThresholdNumNodes);
          }
        }

      }
    }

    public static BitArray BitArrayNodesInSubtree(MCTSNodeStore store, ref MCTSNodeStruct newRoot,
                                                  bool setOldGeneration, out uint numNodes,
                                                  int numExtraPaddingNodesAtEnd, bool clearCacheItems)
    {
      return BitArrayNodesInSubtree(store, ref newRoot, setOldGeneration, out numNodes, null, numExtraPaddingNodesAtEnd, clearCacheItems);
    }


    /// <summary>
    /// Scans all nodes in store and constructs a BitArray capturing
    /// all the nodes which belongs to the subtree rooted at a specified newRoot node.
    /// Additionally the CacheIndex is reset to 0 for all nodes not already old generation.
    /// 
    /// Optionally any nodes not belonging are marked as old generation.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="newRoot"></param>
    /// <param name="setOldGeneration"></param>
    /// <param name="numNodes"></param>
    /// <param name="nodesNewlyBecomingOldGeneration"></param>
    /// <returns></returns>
    public unsafe static BitArray BitArrayNodesInSubtree(MCTSNodeStore store, ref MCTSNodeStruct newRoot,
                                                         bool setOldGeneration, out uint numNodes,
                                                         BitArray nodesNewlyBecomingOldGeneration,
                                                         int numExtraPaddingNodesAtEnd, bool clearCacheItems)
    {
      BitArray includedNodes = new BitArray(store.Nodes.NumTotalNodes + numExtraPaddingNodesAtEnd);

      // Start by including the new root node.
      int newRootIndex = newRoot.Index.Index;
      includedNodes.Set(newRootIndex, true);

      // We can use a highly efficient sequential scan, which is is possible only because the
      // tree has the special property that children of nodes always appear after their parent.
      uint countNumNodes = 0;
      for (int i = 1; i < store.Nodes.nextFreeIndex; i++)
      {
        ref MCTSNodeStruct nodeRef = ref store.Nodes.nodes[i];

        if (!nodeRef.IsOldGeneration)
        {
          if (clearCacheItems)
          {
            nodeRef.CachedInfoPtr = null;
          }

          if (includedNodes.Get(nodeRef.ParentIndex.Index) || i == newRootIndex)
          {
            includedNodes.Set(i, true);
            countNumNodes++;
          }
          else if (setOldGeneration)
          {
            if (nodesNewlyBecomingOldGeneration != null)
            {
              nodesNewlyBecomingOldGeneration[i] = true;
            }
            nodeRef.IsOldGeneration = true;
            store.Nodes.NumOldGeneration++;
          }
        }
      }

      numNodes = countNumNodes;

      return includedNodes;
    }

  }
}
