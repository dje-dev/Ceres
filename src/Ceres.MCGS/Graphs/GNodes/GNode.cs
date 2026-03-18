#define VERY_UNSAFE
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
using System.Diagnostics;
using System.Runtime.CompilerServices;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs.Enumerators;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GParents;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Search.Phases;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

public readonly unsafe partial struct GNode : IComparable<GNode>, IEquatable<GNode>
{
  public readonly Graph Graph;

  /// <summary>
  /// Pointer to the actual graph node data 
  /// (within the fixed contiguous virtual memory block holding all graph nodes).
  /// </summary>
  internal readonly GNodeStruct* nodePtr;

  public GraphStore GraphStore
  {
    [DebuggerStepThrough] get => Graph.Store;
  }


#if NOT
  public IEnumerable<GEdge> ParentEdges
  {
    get
    {
      Span<int> parents = stackalloc int[MAX_PARENTS_PER_NODE];
      GraphStore.ParentsStore.GetParentsNodeIndices(Index, parents);

      for (int i = 0; i < parents.Length; i++)
      {
        if (parents[i] == -1)
        {
          yield break;
        }
        else
        {
          // TODO: in the common case of only a single parent,
          //       we could replace this loop with a stored value
          //       ChildIndexInParent (only valid if not a transposition node)
          GNode parent = Graph[new NodeIndex(parents[i])];
          foreach (GEdge edge in parent.ChildEdges)
          {
            if (edge.ChildIndex == Index)
            {
              yield return edge;
            }
          }
        }
      }
    }
  }
#endif

  // TODO: do some analysis about largest value possible (or likely) in play
  const int MAX_PARENTS_PER_NODE = 50;


  public ParentEdgesEnumerable ParentEdges => new(Graph, Index);


  /// <summary>
  /// Returns the NodeIndex of this node's BFS-tree parent (position 0 in parent list).
  /// This parent lies on a guaranteed cycle-free path to the graph root.
  /// Invalid to call on the graph root (which has no parent).
  /// </summary>
  /// <remarks>
  /// The tree-parent invariant ensures position 0 in each node's parent list is always a 
  /// BFS-tree parent. During normal search, the creating parent is naturally position 0.
  /// GraphRewriter Phase 5 uses BFS discovery to preserve this invariant when rebuilding parents.
  /// </remarks>
  public NodeIndex TreeParentNodeIndex
  {
    get
    {
      GParentsHeader header = NodeRef.ParentsHeader;
      Debug.Assert(!header.IsEmpty, "TreeParentNodeIndex called on node with no parents");
      return header.IsDirectEntry
        ? header.AsDirectParentNodeIndex
        : GraphStore.ParentsStore.DetailSegments.SegmentRef(header.AsSegmentLinkIndex).Entries[0].AsDirectParentNodeIndex;
    }
  }


  /// <summary>
  /// Returns the edge from this node's BFS-tree parent (position 0 in parent list).
  /// This parent lies on a guaranteed cycle-free path to the graph root.
  /// Invalid to call on the graph root (which has no parent).
  /// </summary>
  /// <remarks>
  /// The tree-parent invariant ensures position 0 in each node's parent list is always a 
  /// BFS-tree parent. During normal search, the creating parent is naturally position 0.
  /// GraphRewriter Phase 5 uses BFS discovery to preserve this invariant when rebuilding parents.
  /// </remarks>
  public GEdge TreeParentEdge
  {
    get
    {
      Debug.Assert(!IsGraphRoot, "TreeParentEdge called on graph root node");
      GParentsHeader header = NodeRef.ParentsHeader;
      Debug.Assert(!header.IsEmpty, "TreeParentEdge called on node with no parents");

      NodeIndex parentIdx = header.IsDirectEntry
        ? header.AsDirectParentNodeIndex
        : GraphStore.ParentsStore.DetailSegments.SegmentRef(header.AsSegmentLinkIndex).Entries[0].AsDirectParentNodeIndex;

      GNode parent = Graph[parentIdx];
      int childSlot = parent.IndexOfChildInChildEdges(Index);
      Debug.Assert(childSlot != -1, "Tree parent does not have child edge to this node");
      return parent.ChildEdgeAtIndex(childSlot);
    }
  }


  /// <summary>
  /// Returns the depth of this node from the search root, following tree-parent edges.
  /// </summary>
  /// <remarks>
  /// Follows the tree-parent invariant to guarantee O(depth) traversal with no cycles.
  /// </remarks>
  public int DepthFromSearchRoot()
  {
    int depth = 0;
    GNode node = this;
    while (!node.IsSearchRoot)
    {
      node = Graph[node.TreeParentNodeIndex];
      depth++;
      Debug.Assert(depth < 512, "DepthFromSearchRoot: depth exceeded maximum — tree-parent invariant may be broken");
    }
    return depth;
  }


  /// <summary>
  /// Determines the position this node represents by following tree-parent edges to the graph root.
  /// </summary>
  /// <returns>The MGPosition represented by this node.</returns>
  /// <remarks>
  /// Uses the tree-parent invariant to guarantee O(depth) traversal with no cycles.
  /// The first-created parent (position 0) is always on a BFS-tree path to the root,
  /// eliminating the need for cycle detection.
  /// Note: Uses IsGraphRoot (not IsSearchRoot) because PriorPositionsMG[^1] contains
  /// the graph root position, which doesn't change when the search root moves.
  /// </remarks>
  public MGPosition CalcPosition()
  {
    const int MAX_DEPTH = 512;
    Span<EncodedMove> moves = stackalloc EncodedMove[MAX_DEPTH];
    int depth = 0;

    GNode node = this;
    while (!node.IsGraphRoot)
    {
      Debug.Assert(depth < MAX_DEPTH, "CalcPosition: depth exceeded maximum — tree-parent invariant may be broken");
      GEdge treeEdge = node.TreeParentEdge;
      moves[depth++] = treeEdge.Move;
      node = treeEdge.ParentNode;
    }

    // Apply moves starting from graph root position.
    MGPosition pos = Graph.Store.HistoryHashes.PriorPositionsMG[^1];
    for (int i = depth - 1; i >= 0; i--)
    {
      MGMove mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(moves[i], pos);
      pos.MakeMove(mgMove);
    }
    return pos;
  }


  /// <summary>
  /// Returns the GEdge corresponding to a specified child node (throws Exception if not found).
  /// </summary>
  /// <param name="move"></param>
  /// <returns></returns>
  public GEdge EdgeForNode(GNode node)
  {
    foreach (GEdge edge in ChildEdgesExpanded)
    {
      if (edge.ChildNode == node)
      {
        return edge;
      }
    }

    throw new Exception("Specified node not found as a child in EdgeForNode " + this + " " + node);
  }


  /// <summary>
  /// Returns the GEdge corresponding to a specified move (or default if not found).
  /// </summary>
  /// <param name="move"></param>
  /// <returns></returns>
  public GEdge EdgeForMove(MGMove move)
  {
    EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(move);
    foreach (GEdge edge in ChildEdgesExpanded)
    {
      if (edge.Move == encodedMove)
      {
        return edge;
      }
    }
    return default;
  }


  public bool TryGetSingleParentEdge(out GEdge parentChildEdge)
  {
    // Check if this is a single parent.
    GParentsHeader parentsHeader = Graph.NodesBufferOS[Index.Index].ParentsHeader;

    if (parentsHeader.IsEmpty ||
       !parentsHeader.IsDirectEntry)
    {
      parentChildEdge = default;
      return false;
    }

    GNode parentNode = Graph[parentsHeader.AsDirectParentNodeIndex];
    int indexInParent = parentNode.IndexOfChildInChildEdges(Index);
    if (indexInParent == -1)
    {
      throw new Exception("ParentEdgesEnumerator: Parent not found in child's edges");
    }

    // Create an GEdge between this node and its single parent.
    parentChildEdge = parentNode.ChildEdgeAtIndex(indexInParent);

    return true;
  }


  /// <summary>
  /// Returns the index of the child edge corresponding to the specified child node, or -1 if not found.
  /// We could use 1 byte in GNodeStruct to make this more efficient, but this method is not on the hot path for search.
  /// </summary>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  public int IndexOfChildInChildEdges(NodeIndex childIndex)
  {
    int i = 0;
    foreach (GEdge childEdge in ChildEdgesExpanded)
    {
      if (childEdge.ChildNodeIndex == childIndex)
      {
        return i;
      }
      i++;
    }

    return -1;
  }


  public GEdgeEnumerable ChildEdgesExpanded => new(this);

  public GEdgeWithIndexEnumerable ChildEdgesExpandedWithIndex => new(this);


  /// <summary>
  /// Low-level (but fast) method to return a ref to the GEdgeStruct 
  /// when the block and child index are known.
  /// </summary>
  /// <param name="edgeStoreBlockIndex"></param>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  internal ref GEdgeStruct EdgeStructAtIndexRef(int edgeStoreBlockIndex, int childIndex)
  {
    int offsetInBlock = childIndex % GEdgeStore.NUM_EDGES_PER_BLOCK;

#if VERY_UNSAFE
    ref GEdgeStructBlocked refBlock = ref Graph.EdgesStore.edgeStoreMemoryBuffer[edgeStoreBlockIndex];
    return ref Unsafe.Add(ref Unsafe.As<GEdgeStructBlocked, GEdgeStruct>(ref refBlock), offsetInBlock);
#else
    Span<GEdgeStruct> edgeSpan = Graph.EdgesStore.SpanAtBlockIndex(edgeStoreBlockIndex);
    return ref edgeSpan[offsetInBlock];
#endif
  }


  /// <summary>
  /// Returns the GEdgeHeaderStruct at the specified child index.
  /// </summary>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  public GEdgeHeaderStruct ChildEdgeHeaderAtIndex(int childIndex) => Graph.ChildEdgeHeaderAtIndex(BlockIndexIntoEdgeHeaderStore, childIndex);



  /// <summary>
  /// Returns the GEdge at the specified child index.
  /// </summary>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  public readonly GEdge ChildEdgeAtIndex(int childIndex)
  {
    if (childIndex < NumEdgesExpanded)
    {
      Debug.Assert(childIndex < NumEdgesExpanded);

      GEdgeHeaderStruct header = Graph.ChildEdgeHeaderAtIndex(BlockIndexIntoEdgeHeaderStore, childIndex);
      Span<GEdgeStruct> edgeSpan = Graph.EdgesStore.SpanAtBlockIndex(header.EdgeStoreBlockIndex);

      int offsetInBlock = childIndex % GEdgeStore.NUM_EDGES_PER_BLOCK;
      return new GEdge(edgeRef: ref edgeSpan[offsetInBlock], 
                       parent : this, 
                       child: new GNode(Graph, edgeSpan[offsetInBlock].ChildNodeIndex));
    }
    else
    {
      throw new Exception("Edge does not exist for specified child.");
    }
  }


#if NOT
  public ChildGEdgeEnumerable ChildEdges
  {
    get
    {
      ref GNodeStruct nodeRef = ref NodeRef;
      Span<GEdgeStruct> childSpan = GraphStore.EdgesStore.SpanAtBlockIndex(nodeRef.blockIndexIntoEdgeStore, nodeRef.NumPolicyMoves);

      return new ChildGEdgeEnumerable(ref nodeRef, childSpan, this, Graph);
    }
  }
#endif




  public readonly NodeIndex Index
  {
    [DebuggerStepThrough]
    get
    {
      return new NodeIndex((int)GraphStore.NodesStore.IndexOfNodeAtAddress(nodePtr));
    }
  }

  public readonly ref GNodeStruct NodeRef
  {
    [DebuggerStepThrough]
    get
    {
      return ref Unsafe.AsRef<GNodeStruct>(nodePtr); // slower: ref GraphStore.Nodes.nodes[Index.Index];
    }
  }

  public readonly bool IsLeaf => NodeRef.NumPolicyMoves == 0;

  /// <summary>
  /// Returns true if this NodeX represents an uninitialized or null node.
  /// </summary>
  public readonly bool IsNull => nodePtr == null || NodeRef.Terminal == Chess.GameResult.NotInitialized;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="index"></param>
  [DebuggerStepThrough]
  public GNode(Graph graph, NodeIndex index)
  {
    Graph = graph;
    nodePtr = graph.NodePtr(index);
  }

  /// <summary>
  /// Constructor from a pointer.
  /// </summary>
  /// <param name="graph"></param>
  /// <param name="nodePtr"></param>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  [DebuggerStepThrough]
  internal GNode(Graph graph, GNodeStruct* nodePtr)
  {
    Graph = graph;
    this.nodePtr = nodePtr;
  }

  public readonly ParentNodesEnumerable Parents => new(Graph, Index);

  public bool NumParentsMoreThanOne => !IsGraphRoot && !Graph.NodesBufferOS[Index.Index].ParentsHeader.IsDirectEntry;
  
  public readonly int NumParents
  {
    get
    {
      if (IsGraphRoot)
      {
        return 0;
      }

      // TODO: make this use an accessor
      GParentsHeader parentHeaderPointer = Graph.NodesBufferOS[Index.Index].ParentsHeader;

      if (parentHeaderPointer.IsDirectEntry) // Single entry inline.
      {
        return 1;
      }
      else if (parentHeaderPointer.IsEmpty)
      {
        Debug.Assert(false);
        return 0;
      }
      else
      {
        int parentCount = 0;
        foreach (GNode parent in Parents)
        {
          parentCount++;
        }
        return parentCount;
      }
    }
  }
#if NOT
  internal Span<MoveInfoStruct> ChildrenInfoSpanForInitialization
  {
    get 
    { 
      ref NodeStruct nodeRef = ref NodeRef;
      Debug.Assert(nodeRef.NumChildrenExpanded == 0);

      // Since none expanded yet we know that the block index refers to ChildInfo not VisitToChild.
      return GraphStore.MoveInfos.SpanAtBlockIndex(nodeRef.nodeChildrenOrVisitstoChildrenStartBlockIndex, nodeRef.NumPolicyMoves);
    }
  }
#endif

  /// <summary>
  /// Allocates space for specified number of children and returns Span over those uninitialized GEdgeHeaderStruct.
  /// </summary>
  /// <param name="numEdgeHeaders"></param>
  /// <returns></returns>
  internal GNodeEdgeHeaders AllocatedEdgeHeaders(int numEdgeHeaders)
  {
    Debug.Assert(numEdgeHeaders >= 0 && numEdgeHeaders < 255);
    Debug.Assert(NodeRef.edgeHeaderBlockIndexOrDeferredNode.IsNull);
    Debug.Assert(NodeRef.NumPolicyMoves == 0); // not expected to be already allocated
    Debug.Assert(NodeRef.LockRef.IsLocked);

    if (numEdgeHeaders == 0)
    {
      return new GNodeEdgeHeaders(this, []);
    }
    else
    {
      NodeRef.NumPolicyMoves = (byte)numEdgeHeaders;
      BlockIndexIntoEdgeHeaderStore = (int)GraphStore.EdgeHeadersStore.AllocateEntriesStartBlock(numEdgeHeaders);

      Span<GEdgeHeaderStruct> span = GraphStore.EdgeHeadersStore.SpanAtBlockIndex(BlockIndexIntoEdgeHeaderStore, (byte)numEdgeHeaders);
      return new GNodeEdgeHeaders(this, span);
    }
  }


  internal readonly GEdge ChildFromGPosition(GNode child)
  {
    foreach (GEdge visit in ChildEdgesExpanded)
    {
      if (visit.ChildNodeIndex == child.Index)
      {
        return visit;
      }
    } 
    throw new InvalidOperationException("Child not found in VisitsTo");
  }



  /// <summary>
  /// Returns array of all expanded GEdges, sorted by specified function.
  /// </summary>
  /// <param name="sortValueFunc"></param>
  /// <returns></returns>
  public readonly GEdge[] EdgesSorted(Func<GEdge, double> sortValueFunc)
  {
    GEdge[] edges = new GEdge[NumEdgesExpanded];
    for (int i = 0; i < NumEdgesExpanded; i++)
    {
      edges[i] = ChildEdgeAtIndex(i);
    }

    Array.Sort(edges, (v1, v2) => sortValueFunc(v1).CompareTo(sortValueFunc(v2)));
    return edges;
  }


  /// <summary>
  /// Returns GEdge having maximum value according to the specified function.
  /// </summary>
  /// <param name="sortValueFunc"></param>
  /// <returns></returns>
  public readonly GEdge EdgeWithMaxValue(Func<GEdge, double> valueFunc)
  {
    double maxValue = double.MinValue;
    int maxIndex = -1;  
    for (int i = 0; i < NumEdgesExpanded; i++)
    {
      double value = valueFunc(ChildEdgeAtIndex(i));
      if (value > maxValue)
      {
        maxValue = value;
        maxIndex = i;
      }
    }

    return ChildEdgeAtIndex(maxIndex);
  }


  public Span<GEdgeHeaderStruct> EdgeHeadersSpan
  {
    get
    {
      if (NodeRef.NumPolicyMoves == 0)
      {
        return [];
      }
      else
      {
        return GraphStore.EdgeHeadersStore.SpanAtBlockIndex(BlockIndexIntoEdgeHeaderStore, NodeRef.NumPolicyMoves);
      }
    }
  }

  /// <summary>
  /// Returns if this node is the root of the graph.
  /// </summary>
  public readonly bool IsGraphRoot => nodePtr == Graph.NodesRootNodePtr; // alternately:  nodePtr->IsGraphRoot;


  /// <summary>
  /// Returns if this node is the root of the currently active search.
  /// </summary>
  public readonly bool IsSearchRoot => nodePtr->IsSearchRoot;


  /// <summary>
  /// Extracts the policy vector from the node (inhereiting whatever temperature may have been applied to it).
  /// NOTE: See MCTSNodeStructUtils.ExtractPolicyVector if a version is needed that backs out any temperature that was applied.
  /// </summary>
  /// <param name="policy"></param>
  [SkipLocalsInit]
  public void ExtractPolicyVector(ref CompressedPolicyVector policy)
  {
    // Note: no benefit to sizing these spans smaller when possible, since no initialization cost (due to SkipLocalsInits).
    Span<ushort> indicies = stackalloc ushort[CompressedPolicyVector.NUM_MOVE_SLOTS];
    Span<ushort> probabilities = stackalloc ushort[CompressedPolicyVector.NUM_MOVE_SLOTS];

    // Extract the probabilities, invert soft max, track sum.
    for (int i = 0; i < NumPolicyMoves; i++)
    {
      GEdge child = ChildEdgeAtIndex(i);
      probabilities[i] = CompressedPolicyVector.EncodedProbability(child.P);
      indicies[i] = (ushort)child.Move.IndexNeuralNet;
    }

    if (NumPolicyMoves < CompressedPolicyVector.NUM_MOVE_SLOTS)
    {
      indicies[NumPolicyMoves] = CompressedPolicyVector.SPECIAL_VALUE_SENTINEL_TERMINATOR;
    }

    CompressedPolicyVector.Initialize(ref policy, IsWhite ? SideType.White : SideType.Black, indicies, probabilities);
  }


  /// <summary>
  /// Returns string summary.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    if (nodePtr == null)
    {
      return $"<GNode - Null>";
    }
    else
    {
      string parentsInfo = "";
      if (IsGraphRoot)
      {
        parentsInfo = "100%  ";
      }
      int numParents = 0;
      foreach (GEdge v in ParentEdges)
      {
        parentsInfo += $"{100 * v.P,5:F2}%";
        numParents++;
      }
      if (numParents > 1)
      {
        parentsInfo = "(multiple)"; // multiple parents, each of which will have different policy. TODO: consider showing average
      }
      
      string multiparentIndicator = numParents > 1 ? "*" : " ";
      string searchRootFlag = IsSearchRoot ? " (SROOT)" : " ";

      double siblingQ = double.NaN;
      return $"{multiparentIndicator}<GNode #{Index.Index} {searchRootFlag} N={NodeRef.N:N0} " 
           + $"P= {parentsInfo} V={NodeRef.V,6:F3} ({(NodeRef.Terminal == GameResult.Unknown ? "Unk" : NodeRef.Terminal.ToString())}) " 
           + $"Q={NodeRef.Q,6:F3} D={D,6:F3} "
//           + $"Q={NodeRef.Q,6:F3} SD={NodeRef.StdDevEstimator.RunningStdDev,6:F3} D={D,6:F3}  "
           + (float.IsNaN(NodeRef.UncertaintyValue)  ? "" : $"UV={NodeRef.UncertaintyValue,4:F3} ") 
           + (float.IsNaN(NodeRef.UncertaintyPolicy) ? "" : $"UP={NodeRef.UncertaintyPolicy,4:F3} ")
           + $"H={NodeRef.HashStandalone.Hash % 10000} Sib={100*NodeRef.SiblingsQFrac}%/{NodeRef.SiblingsQ,4:F2} "
           + $"E={NumEdgesExpanded} "
           + $"{CalcPosition().ToPosition.FEN}>";
    }
  }


  /// <summary>
  /// Returns a string summary of the parent(s) of this node.
  /// </summary>
  public readonly string ParentStr
  {
    get
    {
      if (IsGraphRoot)
      {
        return "(none)";
      }
      else
      {
        // TODO: make this more elegant, see code above using ParentEdges
        Span<int> parentIndices = stackalloc int[MAX_PARENTS_PER_NODE];
        GraphStore.ParentsStore.GetParentsNodeIndices(Index, parentIndices);

        int numChildNodes = parentIndices.Length;
        string extraParentDesc = numChildNodes > 1 ? $"and {numChildNodes - 1} more " : "";
        return $"{parentIndices[0]}{extraParentDesc}";
      }
    }
  }

  public readonly void DumpRaw()
  {
    Console.WriteLine();
    Console.WriteLine($"--------------- GNode dump {Index} from {Graph} parent {ParentStr} --------------- ");
    NodeRef.DumpRawFields();
    Console.WriteLine();
    Console.WriteLine("Policy length " + NumPolicyMoves);
    foreach ((GEdge Edge, int _) in ChildEdgesExpandedWithIndex)
    {
      Console.WriteLine("  Move= " + Edge.MoveMG + " P=" + Edge.P + "  expanded:" + Edge.IsExpanded);
    }
    for (int i=NumEdgesExpanded;i<NumPolicyMoves; i++)
    {
      GEdgeHeaderStruct header = Graph.ChildEdgeHeaderAtIndex(BlockIndexIntoEdgeHeaderStore, i);
      Console.WriteLine("  Move= " + header.Move + " P=" + header.P + "  not expanded");
    }
    Console.WriteLine();
  }



  #region Overrides (object)

  public bool Equals(GNode node) => node.nodePtr == nodePtr;
  public override int GetHashCode() => Index.Index.GetHashCode();

  public static bool operator ==(GNode lhs, GNode rhs) => lhs.nodePtr == rhs.nodePtr;
  public static bool operator !=(GNode lhs, GNode rhs) => lhs.nodePtr != rhs.nodePtr;

  public int CompareTo(GNode other) => Index.Index.CompareTo(other.Index.Index);

  public override bool Equals(object obj) => obj is GNode node && Equals(node);
  public static bool operator <(GNode left, GNode right) => left.CompareTo(right) < 0;

  public static bool operator <=(GNode left, GNode right) => left.CompareTo(right) <= 0;

  public static bool operator >(GNode left, GNode right) => left.CompareTo(right) > 0;

  public static bool operator >=(GNode left, GNode right) => left.CompareTo(right) >= 0;

  #endregion
}
