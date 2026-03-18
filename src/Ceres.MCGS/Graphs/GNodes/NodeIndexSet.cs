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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using Ceres.Base.Misc;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Utils;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Contains a fixed-size set of NodeIndex elements with a maximum capacity of MAX_ELEMENTS,
/// with support for computing statistics from a Graph base on the nodes in the set.
/// </summary>
public record struct NodeIndexSet
{
  /// <summary>
  /// Maximum number in set.
  /// When used as standalone has alias set, extensive tests showed that:
  ///   - a significant number of nodes overflowed (had more than 4 positions with same standalone hash)
  ///   - in middlegame, many are only 1 or 2 slots, but for endgames it is common to exceed 8 or even 16.
  ///   - expanding to support up to 16 elements yielded only +3 Elo (+/-9) and more memory, very slightly slower
  /// </summary>
  public const int MAX_ELEMENTS = 8;

  [InlineArray(MAX_ELEMENTS)]
  internal struct SlotsInlineArray
  {
    internal NodeIndex element0;
  }

  private SlotsInlineArray slots;

  /// <summary>
  /// Returns number of occupied slots in the set.
  /// </summary>
  public int Count
  {
    get
    {
      for (int i = 0; i < MAX_ELEMENTS; i++)
      {
        if (slots[i].IsNull)
        {
          return i;
        }
      }
      return MAX_ELEMENTS;
    }
  }

  /// <summary>
  /// Returns a span over all the slots in the set (both occupied and unoccupied).
  /// </summary>
  public Span<NodeIndex> Slots => MemoryMarshal.CreateSpan(ref slots.element0, MAX_ELEMENTS);


  /// <summary>
  /// Returns if the set consists of a single element with specified value.
  /// </summary>
  /// <param name="index"></param>
  /// <returns></returns>
  public readonly bool IsSingleton(NodeIndex index) => slots[0] == index && slots[1].IsNull;


  /// <summary>
  /// Returns the GNode in the set having the largest N.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="nodeIndexToIgnore"></param>
  /// <returns></returns>
  public readonly GNode MaxEligibleNNode(Graph graph, NodeIndex nodeIndexToIgnore)
  {
    int maxN = int.MinValue;
    GNode maxNode = default;
    for (int i = 0; i < MAX_ELEMENTS; i++)
    {
      NodeIndex thisNodeIndex = slots[i];
      if (thisNodeIndex.IsNull)
      {
        break;
      }

      if (thisNodeIndex != nodeIndexToIgnore)
      {
        GNode node = graph[thisNodeIndex];

        if (node.N > 0 
          && node.N > maxN
          && IsEligibleForPseudoTranspositionContribution(node)
          )
        {
          maxN = node.N;
          maxNode = node;
        }
      }
    }

    return maxNode;
  }


  /// <summary>
  /// Returns the NodeIndex at the specified index in the set.  
  /// </summary>
  /// <param name="index"></param>
  /// <returns></returns>
  public NodeIndex this[int index]
  {
    readonly get
    {
      NodeIndex nodeIndex = slots[index];
      Debug.Assert(!nodeIndex.IsNull);
      return nodeIndex;
    }
    private set => slots[index] = value;
  }


  /// <summary>
  /// Returns if the specified NodeIndex exists in the set.
  /// </summary>
  /// <param name="nodeIndex"></param>
  /// <returns></returns>
  public readonly bool Exists(NodeIndex nodeIndex)
  {
    for (int i = 0; i < MAX_ELEMENTS; i++)
    {
      NodeIndex thisNodeIndex = slots[i];
      if (thisNodeIndex.IsNull)
      {
        return false;
      }
      if (thisNodeIndex == nodeIndex)
      {
        return true;
      }
    }
    return false; 
  }


  /// <summary>
  /// Adds a new NodeIndex to the set.
  /// </summary>
  /// <param name="element"></param>
  /// <exception cref="InvalidOperationException"></exception>
  public void Add(NodeIndex element, bool ignoreIfFull = false)
  {
    Debug.Assert(!element.IsNull);

    // Find first null slot
    for (int i = 0; i < MAX_ELEMENTS; i++)
    {
      if (slots[i].IsNull)
      {
        Debug.Assert(!Exists(element));
        slots[i] = element;
        return;
      }
    }

    // All slots full
    if (!ignoreIfFull)
    {
      throw new InvalidOperationException("Maximum capacity reached.");
    }
  }

  /// <summary>
  /// If draw-by-repetition or move 50 rule/repetition dynamics may be in play, 
  /// do not use pseudotransposition information which might have the same context.
  /// </summary>
  /// <param name="node"></param>
  /// <returns></returns>
  internal static bool IsEligibleForPseudoTranspositionContribution(GNode node) =>
        node.N > 0
     && !node.DrawKnownToExistAmongChildren
     && node.NodeRef.Move50Category == Move50CategoryEnum.LessThan75
     && !node.HasRepetitions;


  public static GNode MaxNSiblingNode(Graph graph, PosHash64WithMove50AndReps hash)
  {
    if (!graph.transpositionsPosStandalone.TryGetValue(hash, out GNodeIndexSetIndex transpositionsSet))
    {
      return default;
    }
    else if (transpositionsSet.IsDirectNodeIndex)
    {
      GNode node = graph[transpositionsSet.DirectNodeIndex];
      return IsEligibleForPseudoTranspositionContribution(node) ? node : default;
    }
    else
    {
      NodeIndexSet siblings = graph.NodeIndexSetStore.sets[transpositionsSet.NodeSetIndex];
      GNode maxNode = siblings.MaxEligibleNNode(graph, default);
      return maxNode;
    }  
  }


  /// <summary>
  /// Returns the average Q value weighted by sqrt(N) of the nodes in the set.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="hash"></param>
  /// <returns></returns>
  public static (double avgQ, double avgD) AvgSqrtNWeightedStats(Graph graph, PosHash64WithMove50AndReps hash)
  {

    if (!graph.transpositionsPosStandalone.TryGetValue(hash, out GNodeIndexSetIndex transpositionsSet))
    {
      return (double.NaN, double.NaN);
    }
    else if (transpositionsSet.IsDirectNodeIndex)
    {
      GNode node = graph[transpositionsSet.DirectNodeIndex];
      return node.IsEvaluated && node.N > 0 ? (node.Q, node.D) : (double.NaN, double.NaN);
    }
    else
    {
      NodeIndexSet siblings = graph.NodeIndexSetStore.sets[transpositionsSet.NodeSetIndex];

      // TODO: consider adding a prefetch option like the Stats method below
      double sumSqrtN = 0;
      double sumW = 0;
      double sumD = 0;
      for (int i = 0; i < MAX_ELEMENTS; i++)
      {
        if (siblings.slots[i].IsNull)
        {
          break;
        }

        GNode node = graph[siblings.slots[i]];
        if (node.IsEvaluated && node.N > 0)
        {
          double sqrtN = Math.Sqrt(node.N);
          sumSqrtN += sqrtN;
          sumW += sqrtN * node.Q;
          sumD += sqrtN * node.D;
        }
      }

      return sumSqrtN > 0 ? (sumW / sumSqrtN, sumD / sumSqrtN) : (double.NaN, double.NaN);
    }
  }


  /// <summary>
  /// Returns total count of visits and weighted average Q values.
  /// The returned statistics count only visits in excess already seen by targetNode 
  /// and exclude node with the specified index if provided.
  /// </summary>
  /// <param name="graph"></param>
  /// <returns></returns>
  public readonly (float sumExcessN, double avgQ) Stats(GNode targetNode, 
                                                        int targetNodeNAfterPendingVisits,
                                                        Prefetcher.CacheLevel perfetchCacheLevel = Prefetcher.CacheLevel.None)
  {
    NodeIndex indexToIgnore = targetNode.Index;
    Graph graph = targetNode.Graph;

    // Start prefetch GNodeStruct before starting stating processing (if more than one).
    if (!slots[1].IsNull)
    {
      for (int i = 0; i < MAX_ELEMENTS; i++)
      {
        NodeIndex nodeIndex = slots[i];
        if (nodeIndex.IsNull)
        {
          break;
        }
        else if (nodeIndex != indexToIgnore)
        {
          GNode node = graph[nodeIndex];
          unsafe
          {
            void* nodePtr = Unsafe.AsPointer(ref node.NodeRef);
            Prefetcher.PrefetchLevel1(nodePtr);
          }
        }
      }
    }

    // Experimental feature.
    if (MCGSParamsFixed.USE_PSEUDOTRANSPOSITION_MAX_N_NODE_ONLY && !slots[1].IsNull)
    {
      GNode maxNNode = MaxEligibleNNode(targetNode.Graph, indexToIgnore);
      if (!maxNNode.IsNull)
      {
        Debug.Assert(MCGSParamsFixed.SIBLING_POWER_SHRINK_SIBLING_N == 1);

        int excessN = maxNNode.N - targetNodeNAfterPendingVisits;
        return excessN <= 0 ? default : (excessN, maxNNode.Q);        
      }
    }

    double sumExcessW = 0.0;
    float sumExcessN = 0;

    for (int i = 0; i < MAX_ELEMENTS; i++)
    {
      NodeIndex nodeIndex = slots[i];
      if (nodeIndex.IsNull)
      {
        break;
      }

      if (nodeIndex != indexToIgnore)
      {
        Debug.Assert(!nodeIndex.IsNull);

        GNode siblingNode = graph[nodeIndex];

        (float nContrib, double qContrib) = Graph.TranspositionContribution(targetNode, 
                                                                            targetNodeNAfterPendingVisits, 
                                                                            siblingNode);

        sumExcessN += nContrib;
        sumExcessW += nContrib * qContrib;
      }
    }

    double avgQ = sumExcessN > 0 ? sumExcessW / sumExcessN : 0.0;

    return (sumExcessN, avgQ);
  }


  /// <summary>
  /// Returns a string representation of the NodeIndexSet
  /// (including information from underlying graph).
  /// </summary>
  /// <returns></returns>
  public readonly string DumpString(GNode targetNode)
  {
    StringBuilder builder = new();
    builder.Append('[');

    int count = 0;
    for (int i = 0; i < MAX_ELEMENTS; i++)
    {
      if (slots[i].IsNull)
      {
        break;
      }

      if (i > 0)
      {
        builder.Append(", ");
      }

      GNode node = targetNode.Graph[slots[i]];
      builder.Append($"#{slots[i]}, N: {node.N}, Q: {node.Q:F2}");  
      builder.Append(slots[i]);
      count++;
    }

    (float n, double avgQ) = Stats(targetNode, targetNode.N);
    builder.Append($", Total N: {n}, Avg Q: {avgQ:F2}");

    builder.Append(']');
    return builder.ToString();
  }


  /// <summary>
  /// Returns a string representation of the NodeIndexSet.
  /// </summary>
  /// <returns></returns>
  public override readonly string ToString()
  {
    StringBuilder builder = new ();
    builder.Append('[');

    bool first = true;
    for (int i = 0; i < MAX_ELEMENTS; i++)
    {
      if (slots[i].IsNull)
      {
        break;
      }

      if (!first)
      {
        builder.Append(", ");
      }
      first = false;

      builder.Append(slots[i]);
    }

    builder.Append(']');
    return builder.ToString();
  }
}
