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
using System.Threading;
using Ceres.Base.Environment;
using Ceres.Base.OperatingSystem;
using Ceres.Chess.Positions;
using Ceres.MCGS.Environment;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GParents;
using Ceres.MCGS.Search.Params;

using Microsoft.Extensions.Logging;

#endregion

namespace Ceres.MCGS.Graphs.GraphStores;

public static class GraphStoreConfig
{
  // NOTE: These constants are shadowed in MCGSParamsFixed.

  /// <summary>
  /// If enabled the largest possible search graph is about 2 billion nodes
  /// rather than the default of 1 billion nodes.
  /// 
  /// This should be enabled only if needed and obviously 
  /// only if the system has the required 250GB to 500GB of memory
  /// (because it slightly increases memory usage per node).
  /// 
  /// NOTE: This is not currently supported in the MCGS engine.
  ///       In the MCGS engine it modified NUM_CHILDREN_PER_BLOCK to increase available slots.
  /// </summary>
  public const bool ENABLE_MAX_SEARCH_GRAPH = false;

  /// <summary>
  /// In incremental storage mode memory is reserved at initialization
  /// but only actually committed incrementally as the search tree grows.
  /// </summary>
  public const bool STORAGE_USE_INCREMENTAL_ALLOC = true;
}

/// <summary>
/// A store of MCGS graph nodes (and associated parent/children information) 
/// representing the state of a Monte Carlo graph search.
/// </summary>
public partial class GraphStore : IDisposable
{
  /// <summary>
  /// The root node is guaranteed to occupy the second slot.
  /// </summary>
  public const int ROOT_NODE_INDEX = 1;

  /// <summary>
  /// Flag indicating that the GraphRewriter is currently active.
  /// When true, normal lock assertions should be bypassed since the graph
  /// is quiescent and under exclusive control of the rewriter.
  /// </summary>
  internal bool IsRewriting;

  /// <summary>
  /// Approximate average total bytes consumed by a node, child pointers, 
  /// and associated data structures (such as possible entry transposition roots table).
  /// </summary>
  const ulong APPROX_BYTES_PER_NODE = 16
                                    + GNodeStruct.MCGSNodeStructSizeBytes
                                    + 4 * TYPICAL_AVG_CHILDREN_PER_NODE;

  /// <summary>
  /// Maximum number of nodes for which which the store could accomodate.
  /// This is bounded by physical RAM and also data structure limitations.
  /// </summary>
  public static int MAX_NODES
  {
    get
    {
      long maxByMemory = HardwareManager.MemorySize / (long)APPROX_BYTES_PER_NODE;
      maxByMemory = Math.Min(int.MaxValue, maxByMemory);
      return Math.Min((int)maxByMemory, GEdgeHeadersStore.MAX_NODES);
    }
  }


  /// <summary>
  /// The maximum number of nodes which this store was configured to contain.
  /// </summary>
  public readonly int MaxNodes;

  public bool IsDisposed => NodesStore == null;


  #region Sub-store Objects

  /// <summary>
  /// Underlying storage for tree nodes.
  /// </summary>
  public GNodeStore NodesStore;

  /// <summary>
  /// Underlying storage for children information related to each node.
  /// </summary>
  public GEdgeHeadersStore EdgeHeadersStore;

  /// <summary>
  /// Underlying storage for visit edge information related to each node.
  /// </summary>
  public GEdgeStore EdgesStore;

  /// <summary>
  /// Table containing information about all parents of nodes in the store. 
  /// </summary>
  public GParentsStore ParentsStore;

  /// <summary>
  /// Store for NodeIndexSet structures.
  /// </summary>
  public GNodeIndexSetStore NodeIndexSetStore;


  // TODO: make this into a proper sub-object, if we decide to keep this feature.
  public Half[][] AllStateVectors;


  MemoryBufferOS<GNodeStruct> nodes;

  #endregion

  public const long TYPICAL_AVG_CHILDREN_PER_NODE = 40; // typical value


  // Note that due to use of blocks in allocation, the impact of fragmentation 
  // should be accounted for in the multipliers below.
  public const long  MAX_AVG_LEGAL_MOVES_PER_NODE = 80; // conservative value unlikely to be exceeded

  // Most nodes are leaf nodes (or single extensions that lead to leafs)
  // with only a single child expanded (which occupies 2 slots due to blocking).
  public const long  MAX_AVG_EXPANDED_MOVES_PER_NODE = 10;

  // Note that nodes with only one parent (a typical case) require no parent slots
  // On the other hand, allocating 1 parent uses 2 slots
  public const float MAX_EXTRA_PARENTS_PER_NODE = 3f; 


  /// <summary>
  /// History of positions preceeding the graph (ending in the root move of the graph).
  /// </summary>
  public PositionWithHistory PositionHistory => NodesStore.PositionHistory;

  /// <summary>
  /// Set of hashes associated with the PositionHistory.
  /// </summary>
  public PositionWithHistoryHashes HistoryHashes;


  /// <summary>
  /// Returns reference to the root node.
  /// </summary>
  public ref GNodeStruct RootNode => ref NodesStore.nodes[GraphStore.ROOT_NODE_INDEX];

  /// <summary>
  /// The fraction of the nodes in use, relative to the maximum configured size of the store.
  /// </summary>
  public float FractionInUse => NodesStore.NumTotalNodes / (float)NodesStore.MaxNodes;

  /// <summary>
  /// If the graph feature (allowing shared transposition nodes) 
  /// is enabled (otherwise runs in tree mode).
  /// </summary>
  public readonly bool GraphEnabled;

  /// <summary>
  /// If positions with identical board state are considered equivalent and coalesced into a single node.
  /// </summary>
  public readonly bool UsesPositionEquivalenceMode;

  /// <summary>
  /// If the neural network used to generate position evaluations
  /// supports the "state" feature that carries information forward from move to move.
  /// </summary>
  public readonly bool HasState;

  /// <summary>
  /// If the neural network used to generate position evaluations 
  /// includdes an "action head" that outputs value estimates for all legal moves.
  /// </summary>
  public readonly bool HasAction;


  /// <summary>
  /// Optional logging object.
  /// </summary>
  internal readonly ILogger<GraphStore> Logger;

  public static int TotalNumDisposed;
  public static int TotalNumAllocated;

  public readonly int InstanceID;


  [Conditional("DEBUG")]
  internal void DebugLogInfo(string message, params object[] args) => Logger?.LogInformation($"#{InstanceID} " + message, args);


  /// <summary>
  /// Sets the prior moves and initializes related state (positions and hashes).
  /// </summary>
  /// <param name="priorPositions"></param>
  void SetPriorMoves(PositionWithHistory priorPositions)
  {
    Debug.Assert(priorPositions == null || priorPositions.Count > 0);

    priorPositions ??= PositionWithHistory.StartPosition;

    // Initialize collection of hashes associated with positions in history.
    PosHash96MultisetRunning initialRunningHash = default;
    HistoryHashes = new(priorPositions, initialRunningHash);

    NodesStore.SetPriorMoves(priorPositions);
  }


  /// <summary>
  /// Sets the prior moves during graph rewrite (compaction).
  /// Updates both position history and history hashes.
  /// </summary>
  internal void SetPriorMovesForRewrite(PositionWithHistory newPriorMoves) => SetPriorMoves(newPriorMoves);



  /// <summary>
  /// Constructor to create a store of specified maximum size.
  /// </summary>
  /// <param name="maxNodes"></param>
  /// <param name="hasAction"></param>
  /// <param name="hasState"></param>
  /// <param name="graphEnabled"></param>
  /// <param name="usesPositionEquivalenceMode"></param>
  /// <param name="tryEnableLargePages"></param>
  /// <param name="priorMoves"></param>
  public GraphStore(int maxNodes,
                    bool hasAction,
                    bool hasState,
                    bool graphEnabled,
                    bool usesPositionEquivalenceMode,
                    bool tryEnableLargePages,
                    PositionWithHistory priorMoves)
  {
    if (MCGSParamsFixed.LOGGING_ENABLED)
    {
      Logger = MCGSEnvironment.CreateLogger<GraphStore>();
    }

    InstanceID = Interlocked.Increment(ref TotalNumAllocated);
    DebugLogInfo($"Creating MCTSNodeStore with max {maxNodes} nodes, graphEnabled={graphEnabled}, hasState={hasState}, hasAction={hasAction} #priorMoves={priorMoves.Count}");

    GraphEnabled = graphEnabled;
    UsesPositionEquivalenceMode = usesPositionEquivalenceMode;
    MaxNodes = maxNodes;
    HasState = hasState;
    HasAction = hasAction;

    NodesStore = new GNodeStore(this, maxNodes, hasState, priorMoves,
                                MCGSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC,
                                MCGSParamsFixed.TryEnableLargePages, false);
    nodes = NodesStore.MemoryBufferOSStore;

    // Set prior moves and initialize related state
    SetPriorMoves(priorMoves);

    long reservedEdgeHeaders = 10000 +  maxNodes * (long)MAX_AVG_LEGAL_MOVES_PER_NODE;
    EdgeHeadersStore = new GEdgeHeadersStore(this, reservedEdgeHeaders, tryEnableLargePages);

    long reservedVisitEdges = 5000 + maxNodes * MAX_AVG_EXPANDED_MOVES_PER_NODE;
    if (GEdgeHeadersStore.NUM_EDGE_HEADERS_PER_BLOCK > 2)
    {
      // Adjust higher due to greater fragmentation
      reservedVisitEdges *= 2;
    }
    EdgesStore = new GEdgeStore(NodesStore.ParentStore, reservedVisitEdges, tryEnableLargePages);

    long maxParents = (long)(1000L + MAX_EXTRA_PARENTS_PER_NODE * maxNodes);
    int parentsMultiplier = usesPositionEquivalenceMode ? 8 : 1; // In coalesced mode many more nodes will converge on same node
    ParentsStore = new GParentsStore(this, maxParents, usesPositionEquivalenceMode, tryEnableLargePages);

    // Create the NodeIndexSet store with a reasonable size estimate
    long reservedNodeIndexSets = Math.Max(10000, maxNodes);
    NodeIndexSetStore = new GNodeIndexSetStore(this, (int)reservedNodeIndexSets,
                                              MCGSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC,
                                              MCGSParamsFixed.TryEnableLargePages, false);

    // Save a copy of the prior moves
    NodesStore.PositionHistory = new PositionWithHistory(priorMoves);

    // TODO: make this conditional on having state, resizable, perhaps do not have this at all (in Annotation only)
    if (HasState)
    {
      AllStateVectors = new Half[MaxNodes][];
    }

    CeresEnvironment.LogInfo("NodeStore", "Init", $"MCGSNodeStore created with max {maxNodes} nodes, max {reservedEdgeHeaders} children");

    GNodeStruct.ValidateMCGSNodeStruct();
    GEdgeStruct.ValidateEdgeStruct();
  }

  #region IDisposable Support


  /// <summary>
  /// Disposes of the store, releasing all nodes and children.
  /// </summary>
  /// <param name="disposing"></param>
  protected virtual void Dispose(bool disposing)
  {
    if (!disposedValue)
    {
      if (disposing)
      {
        Interlocked.Increment(ref TotalNumDisposed);

        // Release nodes and children.
        NodesStore.Deallocate();
        NodesStore = null;

        ParentsStore.Deallocate();
        ParentsStore = null;

        EdgeHeadersStore.Deallocate();
        EdgeHeadersStore = null;

        EdgesStore.Deallocate();
        EdgesStore = null;

        NodeIndexSetStore?.Deallocate();
        NodeIndexSetStore = null;
      }


      disposedValue = true;
    }
  }

  static bool finalizerHaveWarned;



  /// <summary>
  /// Destructor to release store.
  /// </summary>
  ~GraphStore()
  {
    // TODO: cleanup
    if (!finalizerHaveWarned)
    {
      finalizerHaveWarned = true;
    }
    Dispose(false);
  }

  private bool disposedValue = false; // To detect redundant calls
  readonly Lock disposingLock = new();


  /// <summary>
  /// Release the store.
  /// </summary>
  public void Dispose()
  {
    lock (disposingLock)
    {
      Dispose(true);
      GC.SuppressFinalize(this);
    }
  }

  /// <summary>
  /// Resizes underlying memory block to commit only up to currently used space.
  /// </summary>
  /// <param name="numNodes"></param>
  /// <exception cref="Exception"></exception>
  public void ResizeToCurrent()
  {
    NodesStore.ResizeToCurrent();
    EdgeHeadersStore.ResizeToCurrent();
    EdgesStore.ResizeToCurrent();
    NodeIndexSetStore.ResizeToCurrent();
  }

  #endregion


  #region Accessors

  /// <summary>
  /// Returns reference to node at specified index within store.
  /// </summary>
  /// <param name="index"></param>
  /// <returns></returns>
  public ref readonly GNodeStruct this[NodeIndex index] => ref nodes[index.Index];


  /// <summary>
  /// Returns reference to node at specified index within store.
  /// </summary>
  /// <param name="index"></param>
  /// <returns></returns>
  public ref readonly GNodeStruct this[int index] => ref nodes[index];


  /// <summary>
  /// Returns reference to child node at specified index within children.
  /// </summary>
  /// <param name="childIndex"></param>
  /// <returns></returns>
  public ref readonly GNodeStruct this[in GNodeStruct nodeRef, int childIndex]
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    get
    {
      throw new NotImplementedException("Someday restore");
      Debug.Assert(childIndex < nodeRef.NumEdgesExpanded);
      //return ref MoveInfos[nodeRef.ChildAtIndex(childIndex).ChildNodeIndex.Index];
    }
  }

  #endregion

  #region Validation
  public unsafe void Validate(Graph graph,  // TODO: why is full qualification required?
                        bool nodesInFlightExpectedZero = true,
                        bool dumpIfFails = false,
                        bool fastMode = false)
  {
    GraphStoreValidator validator = new(this, graph, nodesInFlightExpectedZero);

    if (dumpIfFails)
    {
      try 
      { 
        validator.Validate(graph, nodesInFlightExpectedZero, fastMode:fastMode); 
      }
      catch (Exception e)
      {
        graph.DumpNodesStructure();
        graph.Dump(true);

        Console.WriteLine();
        Console.WriteLine(e.Message);
        Console.WriteLine(e.StackTrace);
        System.Environment.Exit(3);
      } 
    }
    else
    {
      validator.Validate(graph, nodesInFlightExpectedZero, fastMode: fastMode);
    }
  }

  #endregion


  #region Diagnostic output

  /// <summary>
  /// Dumps summary of memory usage to console.
  /// </summary>
  public void DumpUsageSummary()
  {
    Console.WriteLine();
    Console.WriteLine("GRAPH STORE USAGE SUMMARY");
    NodesStore.MemoryBufferOSStore.DumpMemoryUseSummary();
    EdgeHeadersStore.MemoryBufferOSStore.DumpMemoryUseSummary();
    EdgesStore.MemoryBufferOSStore.DumpMemoryUseSummary();
    ParentsStore.MemoryBufferOSStore.DumpMemoryUseSummary();
    NodeIndexSetStore.MemoryBufferOSStore.DumpMemoryUseSummary();

    Console.WriteLine();
    Console.WriteLine(NodesStore.MemoryBufferOSStore.StoreInfoString());
    Console.WriteLine(EdgeHeadersStore.MemoryBufferOSStore.StoreInfoString());
    Console.WriteLine(EdgesStore.MemoryBufferOSStore.StoreInfoString());
    Console.WriteLine(ParentsStore.DetailSegments.MemoryBufferOSStore.StoreInfoString());
    Console.WriteLine(NodeIndexSetStore.MemoryBufferOSStore.StoreInfoString());
  }


  /// <summary>
  /// Returns string summary.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return NodesStore == null
        ? "<MCGSNodeStore DISPOSED>"
        : $"<MCGSNodeStore [#{InstanceID}] Nodes Occupied {NodesStore.NumUsedNodes,-15:N0}"
         + $" Chldren Allocated {EdgeHeadersStore.NumAllocatedItems,-15:N0}"
         + $" VisitsEdges Allocated {EdgesStore.NumAllocatedEdges,-15:N0}";
  }

  #endregion
}
