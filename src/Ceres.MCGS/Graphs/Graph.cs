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
using System.Runtime.CompilerServices;
using System.Threading;
using Ceres.Base.DataTypes;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GParents;
using Ceres.MCGS.Graphs.GraphStores;

using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PathEvaluators;
using Ceres.MCGS.Search.Phases.Evaluation;
using Ceres.MCGS.Search.PUCT;
using Ceres.MCGS.Utils;

#endregion

namespace Ceres.MCGS.Graphs;

/// <summary>
/// Structure used to track information about the nodes lying
/// between the graph root and a search root node.
/// </summary>
/// <param name="ChildNode"></param>
/// <param name="ChildPosMG"></param>
/// <param name="ChildHashStandalone64"></param>
/// <param name="ChildHashStandalone96"></param>
/// <param name="MoveToChild"></param>
/// <param name="MoveToChildIrreversible"></param>
public readonly record struct GraphRootToSearchRootNodeInfo(GNode ChildNode, in MGPosition ChildPosMG,
                                                            PosHash64 ChildHashStandalone64,
                                                            PosHash96 ChildHashStandalone96,
                                                            MGMove MoveToChild, bool MoveToChildIrreversible)
{
  /// <summary>
  /// Returns string representation.
  /// </summary>
  /// <returns></returns>
  public readonly override string ToString()
  {
    return $"<GraphRootToSearchRootNodeInfo Child=#{ChildNode.Index.Index} " +
           $"Pos={ChildPosMG.ToPosition.FEN} " +
           $"H64={ChildHashStandalone64.Hash % 10_000} " +
           $"H96={ChildHashStandalone96.Low % 10_000}/{ChildHashStandalone96.High % 10_000} " +
           $"Move={MoveToChild} " +
           $"Irrev={MoveToChildIrreversible}>";
  }
}


/// <summary>
/// Managers a rooted directed acyclic graph (DAG) consistig of GNodes (for chess positions)
/// and associated information (e.g. parent/child linkages).
/// </summary>
public unsafe class Graph : IDisposable
{
  public readonly GraphStore Store;

  /// <summary>
  /// Cached pointer to first (unused reserved) node.
  /// </summary>
  internal readonly GNodeStruct* NodesBasePtr;

  /// <summary>
  /// Boolean flag used for ad-hoc diagnostics/testing.
  /// </summary>
  public readonly bool TestFlag;

  /// <summary>
  /// Cached pointer to root node (for efficient IsRoot determination).
  /// </summary>
  internal GNodeStruct* NodesRootNodePtr { get; set; }

  public bool IsDisposed => Store == null;

  public bool GraphEnabled => Store.GraphEnabled;

  /// <summary>
  /// When true and PositionEquivalence mode is active, eliminates the 96-bit
  /// position+sequence dictionary and uses the standalone 64-bit dictionary
  /// as the primary dedup mechanism in LookupOrCreateAndAcquire.
  /// When false, preserves original dual-dictionary behavior.
  /// </summary>
  internal const bool SINGLE_DICTIONARY_POSITION_MODE = true;

  /// <summary>
  /// Dictionary of already extant positions,
  /// mapping their hash to corresponding node index.
  /// </summary>
  internal IConcurrentDictionary<PosHash96MultisetFinalized, int> transpositionPositionAndSequence;

  /// <summary>
  /// Dictionary mapping hash to NodeIndexSetIndex (reference to set of nodes with same standalone hash).
  /// </summary>
  public IConcurrentDictionary<PosHash64WithMove50AndReps, GNodeIndexSetIndex> transpositionsPosStandalone;


  /// <summary>
  /// Count of number of positions sent to the neural network for evaluation.
  /// </summary>
  public int NNPositionEvaluationsCount;

  /// <summary>
  /// Count of number of batches set to the neural network for evaluation.
  /// </summary>
  public int NNBatchesCount;


  /// <summary>
  /// Size of the largest batch of positions sent to the neural network for evaluation.
  /// </summary>
  public int NNBatchSizeMax;


  /// <summary>
  /// Returns the number of transpositions of various types being tracked in the graph.
  /// </summary>
  public (int CountTranspositionAndSequence, int CountTranspositionStandlone) TranspositionCounts
    => (transpositionPositionAndSequence == null ? 0 : transpositionPositionAndSequence.Count,
        transpositionsPosStandalone.Count);


  public int NumLinksToExistingNodes;

  public bool HasState => Store.HasState;

  public bool HasAction => Store.HasAction;

  /// <summary>
  /// If the search algorithm may set the Q of a node with N=1
  /// to possibly differ from the V of the node (e.g. when sibling feature enabled).
  /// This is used in validation logic.
  /// </summary>
  public bool NodesWithOneVisitMayHaveDifferentQ { get; private set; } = false;

  public float RatioVisitsToNodes => (float)Store.RootNode.N / (float)Store.NodesStore.NumUsedNodes;


  #region Local copies of store references (cache here for performance)

  /// <summary>
  // Reference to raw GEdgeHeaderStruct array cached here for fast access.
  /// </summary>
  readonly MemoryBufferOS<GEdgeHeaderStruct> edgeHeaderBufferOS;

  /// <summary>
  /// Underlying storage for tree nodes.
  /// </summary>
  public GNodeStore NodesStore;

  /// <summary>
  /// Underlying storage for nodes.
  /// </summary>
  public MemoryBufferOS<GNodeStruct> NodesBufferOS;

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

  #endregion

  public GEdgeHeaderStruct ChildEdgeHeaderAtIndex(int nodeBlockIndexIntoEdgeHeaderStore, int childIndex)
  {
    // TODO: simplify!
    long offsetEdgeHeader = GEdgeHeadersStore.NUM_EDGE_HEADERS_PER_BLOCK * nodeBlockIndexIntoEdgeHeaderStore + childIndex;
    return edgeHeaderBufferOS[offsetEdgeHeader];
  }


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="maxNodes"></param>
  /// <param name="hasAction"></param>
  /// <param name="hasState"></param>
  /// <param name="graphEnabled"></param>
  /// <param name="coalescedMode"></param>
  /// <param name="tryEnableLargePages"></param>
  /// <param name="priorHistory"></param>
  /// <param name="testFlag"></param>
  public Graph(int maxNodes,
               bool hasAction, bool hasState,
               bool graphEnabled,
               bool coalescedMode,
               bool tryEnableLargePages,
               bool nodesWithOneVisitMayHaveDifferentQ,
               PositionWithHistory priorHistory,
               bool testFlag)
  {
    // Create underlying store.
    Store = new GraphStore(maxNodes, hasAction, hasState, graphEnabled, coalescedMode, tryEnableLargePages, priorHistory);

    NodesBasePtr = (GNodeStruct*)Unsafe.AsPointer(ref Store.NodesStore.nodes[0]);
    NodesRootNodePtr = (GNodeStruct*)Unsafe.AsPointer(ref Store.NodesStore.nodes[GraphStore.ROOT_NODE_INDEX]);
    GraphRootNode = this[GraphStore.ROOT_NODE_INDEX];
    NodesWithOneVisitMayHaveDifferentQ = nodesWithOneVisitMayHaveDifferentQ;
    TestFlag = testFlag;

    InitializeNullNode();
    SetSearchRootNode(new NodeIndex(GraphStore.ROOT_NODE_INDEX));
    Initialize(maxNodes);

    // Make local copies of references to key data structures (for performance).
    NodesStore = Store.NodesStore;
    NodesBufferOS = Store.NodesStore.nodes;
    EdgeHeadersStore = Store.EdgeHeadersStore;
    EdgesStore = Store.EdgesStore;
    ParentsStore = Store.ParentsStore;
    edgeHeaderBufferOS = Store.EdgeHeadersStore.Entries;
    NodeIndexSetStore = Store.NodeIndexSetStore;
  }


  /// <summary>
  /// Initializes the null node.
  /// </summary>
  void InitializeNullNode()
  {
    // In Debug mode we initialize fields to most illegal/extreme values.
    // This makes it more likely that any bugs that end up referring to null node are surfaced.
    // In Release mode we leave defaults, so any bug still remaining would possibly do less damage.
#if DEBUG
    ref GNodeStruct rootNode = ref this[0].NodeRef;

    rootNode.WinP = FP16.NaN;
    rootNode.LossP = FP16.NaN;
    rootNode.UncertaintyValue = FP16.NaN;
    rootNode.UncertaintyPolicy = FP16.NaN;
    rootNode.MRaw = byte.MaxValue;
    rootNode.SetQNaN();
    rootNode.D = 0;// double.NaN;
    rootNode.NumEdgesExpanded = byte.MaxValue;
    rootNode.NumPolicyMoves = byte.MaxValue;
    // rootNode.blockIndexIntoEdgeHeaderStore = int.MinValue;    

    // Note that miscFields are left as default.
#endif
    this[0].SetLockIllegalValue();
  }


  /// <summary>
  /// Registers that a batch of positions (of specified size)
  /// has been sent to the neural network for evaluation.
  /// </summary>
  /// <param name="batchSize"></param>
  public void RegisterNNBatch(int batchSize)
  {
    // Update statistics.
    Interlocked.Increment(ref NNBatchesCount);
    Interlocked.Add(ref NNPositionEvaluationsCount, batchSize);
    NNBatchSizeMax = Math.Max(batchSize, NNBatchSizeMax);
  }


  private void Initialize(int maxNodes)
  {
    // Try to size Dictionary based on maxNodes but don't allow to be too large.
    const int MAX_DICTIONARY_SIZE_HINT = 2_000_000; 
    int dictionarySizeHint = Math.Min(maxNodes, MAX_DICTIONARY_SIZE_HINT);
    const int DICTIONARY_CONCURRENCY = 16;

    bool skipPosSeqDict = SINGLE_DICTIONARY_POSITION_MODE && Store.UsesPositionEquivalenceMode;

    if (MCGSParamsFixed.USE_LEGACY_CONCURRENT_DICTIONARY)
    {
      transpositionPositionAndSequence = (GraphEnabled && !skipPosSeqDict)
        ? new ConcurrentDictionaryAdapter<PosHash96MultisetFinalized, int>(DICTIONARY_CONCURRENCY, dictionarySizeHint)
        : null;
      transpositionsPosStandalone = new ConcurrentDictionaryAdapter<PosHash64WithMove50AndReps, GNodeIndexSetIndex>(DICTIONARY_CONCURRENCY, dictionarySizeHint);
    }
    else
    {
      transpositionPositionAndSequence = (GraphEnabled && !skipPosSeqDict)
        ? new ConcurrentDictionaryExtendible<PosHash96MultisetFinalized, int>(DICTIONARY_CONCURRENCY, dictionarySizeHint)
        : null;
      transpositionsPosStandalone = new ConcurrentDictionaryExtendible<PosHash64WithMove50AndReps, GNodeIndexSetIndex>(DICTIONARY_CONCURRENCY, dictionarySizeHint);
    }

    // Ensure NodeIndexSetStore is cached locally before we use it
    NodeIndexSetStore = Store.NodeIndexSetStore;
    
    // Initialize the root node with basic information.
    MGPosition initialPos = Store.NodesStore.PositionHistory.FinalPosMG;
    PosHash64 hash64 = MGPositionHashing.Hash64(in initialPos);

    PosHash96MultisetFinalized rootHash96Finalized = Store.HistoryHashes.PriorPositionsHashesRunningFinalized[^1];

    MGMoveList rootMoves = new MGMoveList();
    MGMoveGen.GenerateMoves(in initialPos, rootMoves);
    InitializeNodeForPos(GraphRootNode, true, in initialPos, rootMoves, hash64, rootHash96Finalized, out _);
  }


  internal NodeIndex priorSearchRoot = default;

  /// <summary>
  /// Sets the search root for the graph.
  /// </summary>
  /// <param name="searchRoot"></param>
  public void SetSearchRootNode(NodeIndex searchRoot)
  {
    // Transfer ownership of any search root designation
    // from possible prior search root to new search root.
    if (!priorSearchRoot.IsNull)
    {
      Debug.Assert(this[priorSearchRoot].IsSearchRoot);
      this[priorSearchRoot.Index].NodeRef.IsSearchRoot = false;
    }

    this[searchRoot.Index].NodeRef.IsSearchRoot = true;
    priorSearchRoot = searchRoot;
  }


  /// <summary>
  /// Returns a pointer to the node structure associated with the specified node index.
  /// </summary>
  internal GNodeStruct* NodePtr(NodeIndex index) => NodesBasePtr + (nuint)(uint)index.Index;


  /// <summary>
  /// Returns the root node of the graph.
  /// </summary>
  public GNode GraphRootNode;


  /// <summary>
  /// Returns Node at specified index.  
  /// </summary>
  /// <param name="i"></param>
  /// <returns></returns>
  public GNode this[int i] => new(this, new NodeIndex(i));

  /// <summary>
  /// Returns Node at specified index.  
  /// </summary>
  /// <param name="index"></param>
  /// <returns></returns>
  public GNode this[NodeIndex index] => this[index.Index];


  /// <summary>
  /// Attempts to lookup a node by its position+sequence hash,
  /// returning either the existing node or attempting to create and return a new one if not found.
  /// It is possible the insertion into the dictionary fails because another thread succeeded first,
  /// in which case wasCollision is set to true and the returned node should be discarded.
  /// </summary>
  /// <param name="hash"></param>
  /// <param name="okToCreate"></param>
  /// <returns></returns>
  (GNode node, bool wasCreated, bool wasCollision) LookupOrCreateAndAcquire(
    PosHash96MultisetFinalized hash, PosHash64WithMove50AndReps standaloneKey, bool okToCreate)
  {
    int index;
    GNode node;
    if (!GraphEnabled)
    {
      Debug.Assert(okToCreate);

      // In MCTS mode we always create a new node.
      NodeIndex newNodeIndex = Store.NodesStore.AllocateNext();
      node = new GNode(this, newNodeIndex);
      node.AcquireLock(); // TODO: someday eliminate (also matching Release)
      return (node, true, false);
    }

    // In single-dictionary position mode, dedup via the standalone 64-bit dict
    // instead of the 96-bit position+sequence dict.
    if (SINGLE_DICTIONARY_POSITION_MODE && Store.UsesPositionEquivalenceMode)
    {
      if (transpositionsPosStandalone.TryGetValue(standaloneKey, out GNodeIndexSetIndex setIndex)
          && !setIndex.IsNull)
      {
        int existingIdx = setIndex.IsDirectNodeIndex
          ? setIndex.DirectNodeIndex
          : NodeIndexSetStore.sets[setIndex.NodeSetIndex][0].Index;
        node = this[existingIdx];
        node.AcquireLock();
        return (node, false, false);
      }

      if (!okToCreate)
      {
        return (default, false, false);
      }

      NodeIndex candIdx = Store.NodesStore.AllocateNext();
      GNode cand = this[candIdx];
      cand.AcquireLock();
      cand.NodeRef.WinP = FP16.NaN;

      GNodeIndexSetIndex directIdx = GNodeIndexSetIndex.FromDirectNodeIndex(candIdx.Index);
      if (transpositionsPosStandalone.TryAdd(standaloneKey, directIdx))
      {
        return (cand, true, false);
      }
      else
      {
        // Lost race: another thread inserted first.
        cand.ReleaseLock();
        cand.NodeRef.IsOldGeneration = true;

        if (transpositionsPosStandalone.TryGetValue(standaloneKey, out setIndex) && !setIndex.IsNull)
        {
          int existingIdx = setIndex.IsDirectNodeIndex
            ? setIndex.DirectNodeIndex
            : NodeIndexSetStore.sets[setIndex.NodeSetIndex][0].Index;
          GNode collisionNode = this[existingIdx];
          collisionNode.AcquireLock();
          return (collisionNode, false, true);
        }
        else
        {
          throw new NotImplementedException("Internal error: requery after standalone dict add collision failed");
        }
      }
    }

    // Original path: dedup via 96-bit position+sequence dict.
    if (transpositionPositionAndSequence.TryGetValue(hash, out index))
    {
      node = this[index];
      node.AcquireLock();
      return (node, false, false);
    }

    if (!okToCreate)
    {
      return (default, false, false);
    }

    // Allocate a new node and immediately lock it.
    NodeIndex candIdx2 = Store.NodesStore.AllocateNext();
    GNode cand2 = this[candIdx2];
    cand2.AcquireLock();

    // Since IsEvaluated method looks for for NaN on WinP
    // it is essential to immediately set WinP before this node becomes visible.
    // TODO: consider having a separate flag for IsEvaluated?
    cand2.NodeRef.WinP = FP16.NaN;

    if (transpositionPositionAndSequence.TryAdd(hash, candIdx2.Index))
    {
      // Successfully added to dictionary as the node representing this position+sequence.
      return (cand2, true, false);
    }
    else
    {
      // Lost the race: someone else inserted first.
      // Release lock on this (now orphaned) node.
      cand2.ReleaseLock();

      cand2.NodeRef.IsOldGeneration = true; // mark as orphaned

      // Requery to find the winning node.
      // Mark this return as a collision that should result in an aborted visit.
      if (transpositionPositionAndSequence.TryGetValue(hash, out index))
      {
        GNode collisionNode = this[index];
        collisionNode.AcquireLock();
        return (collisionNode, false, true);
      }
      else
      {
        throw new NotImplementedException("Internal error: requery after dictionary add collision failed");
      }
    }
  }



  public void Validate(bool nodesInFlightExpectedZero = true, bool dumpIfFails = false, bool fastMode = false)
  {
    Store.Validate(this, nodesInFlightExpectedZero, dumpIfFails, fastMode);
  }



  internal static readonly LiveStats fiftyMoveCounter50 = MCGSParamsFixed.LOG_LIVE_STATS ? new LiveStats("50 move rule, over 50 ply", 1) : null;
  internal static readonly LiveStats fiftyMoveCounter90 = MCGSParamsFixed.LOG_LIVE_STATS ? new LiveStats("50 move rule, over 90 ply", 1) : null;



  /// <summary>
  /// Returns the node matching a specified standalone hash, or null node if not found.
  /// </summary>
  /// <param name="hash64WithMoveAndReps"></param>
  /// <returns></returns>
  public GNode TryLookupNode(PosHash64WithMove50AndReps hash64WithMoveAndReps)
  {
    bool found = transpositionsPosStandalone.TryGetValue(hash64WithMoveAndReps, out GNodeIndexSetIndex setIndex);
    if (found && !setIndex.IsNull)
    {
      if (setIndex.IsDirectNodeIndex)
      {
        // Direct node index case
        return this[setIndex.DirectNodeIndex];
      }
      else
      {
        // Set of node indices case
        NodeIndexSet siblingSet = NodeIndexSetStore.sets[setIndex.NodeSetIndex];
        return this[siblingSet[0]];
      }
    }
    else if (ReuseGraphProvider is not null)
    {
      Graph otherGraph = ReuseGraphProvider();
      if (otherGraph != null)
      {
        GNode lookupNode = otherGraph.TryLookupNode(hash64WithMoveAndReps);
        if (!lookupNode.IsNull && lookupNode.IsEvaluated)
        {
          return lookupNode;
        }
      }
    }

    return default;
  }


  /// <summary>
  /// Optional Func that can return another Graph
  /// (from another engine currently in memory)
  /// that can be a source of neural network initializations
  /// for nodes (via standalone transposition hash lookup table).
  /// </summary>
  public Func<Graph> ReuseGraphProvider;


  /// <summary>
  /// Reusable native memory buffers for GraphRewriter.
  /// Created lazily on first rewrite, disposed with the Graph.
  /// </summary>
  internal GraphRewriterScratchBuffers RewriterScratchBuffers;

  private bool disposedValue;

  void InitializeNodeForPos(GNode node,
                            bool isRoot,
                            in MGPosition mgPos,
                            MGMoveList moves,
                            PosHash64 standaloneHash,
                            PosHash96MultisetFinalized posAndSequenceHash,
                            out GNode standaloneTranspositionNode)
  {
    ref GNodeStruct nodeRef = ref node.NodeRef;
    Debug.Assert(node.IsSearchRoot || node.IsLocked);

    fiftyMoveCounter90?.Add(1, mgPos.Rule50Count >= 90 ? 1 : 0);
    fiftyMoveCounter50?.Add(1, mgPos.Rule50Count >= 50 ? 1 : 0);

    PosHash64WithMove50AndReps hash64WithMoveAndReps = MGPositionHashing.Hash64WithMove50AndRepsAdded(standaloneHash, mgPos.RepetitionCount, mgPos.Move50Category);
    bool got = transpositionsPosStandalone.TryGetValue(hash64WithMoveAndReps, out GNodeIndexSetIndex setIndex);

    standaloneTranspositionNode = default;
    if (got && !setIndex.IsNull)
    {
      if (setIndex.IsDirectNodeIndex)
      {
        // Direct node index stored in the dictionary
        standaloneTranspositionNode = this[setIndex.DirectNodeIndex];
      }
      else
      {
        // Get the NodeIndexSet from the store
        NodeIndexSet siblingSet = NodeIndexSetStore.sets[setIndex.NodeSetIndex];
        standaloneTranspositionNode = this[siblingSet[0]]; // TODO: choice of first is arbitrary, can we do better?
      }
    }

    // In single-dictionary mode we may find ourselves (pre-registered but unevaluated),
    // so also check ReuseGraphProvider when the found sibling is not usable.
    if (SINGLE_DICTIONARY_POSITION_MODE && Store.UsesPositionEquivalenceMode)
    {
      if ((standaloneTranspositionNode.IsNull || !standaloneTranspositionNode.IsEvaluated)
          && ReuseGraphProvider is not null)
      {
        Graph otherGraph = ReuseGraphProvider();
        if (otherGraph != null)
        {
          GNode lookupNode = otherGraph.TryLookupNode(hash64WithMoveAndReps);
          if (!lookupNode.IsNull && lookupNode.IsEvaluated)
          {
            standaloneTranspositionNode = lookupNode;
          }
        }
      }
    }
    else if (standaloneTranspositionNode.IsNull && ReuseGraphProvider is not null)
    {
      // Original path: only check ReuseGraphProvider when no sibling was found.
      Graph otherGraph = ReuseGraphProvider();
      if (otherGraph != null)
      {
        GNode lookupNode = otherGraph.TryLookupNode(hash64WithMoveAndReps);
        if (!lookupNode.IsNull && lookupNode.IsEvaluated)
        {
          standaloneTranspositionNode = lookupNode;
        }
      }
    }

    if (IsStandaloneHastableEligible(in mgPos))
    {
      if (!got || setIndex.IsNull)
      {
        // First node for this hash - store it directly in the dictionary
        GNodeIndexSetIndex directIndex = GNodeIndexSetIndex.FromDirectNodeIndex(node.Index.Index);
        transpositionsPosStandalone[hash64WithMoveAndReps] = directIndex;
      }
      else
      {
        // We already have an entry for this hash
        if (setIndex.IsDirectNodeIndex)
        {
          // There's currently only one node stored directly - need to convert to a NodeIndexSet.
          int existingNodeIndex = setIndex.DirectNodeIndex;

          // Skip if we found ourselves (already pre-registered by LookupOrCreateAndAcquire
          // in single-dictionary mode).
          if (existingNodeIndex != node.Index.Index)
          {
            // Create a new NodeIndexSet in the store
            int newSetIndex = NodeIndexSetStore.AllocateNext();
            NodeIndexSet siblingSet = new();

            // Add both the existing node and the new node
            // TODO: make a more efficient method to set slots 0 and 1 directly (update Count to 2).
            siblingSet.Add(new NodeIndex(existingNodeIndex), true);
            siblingSet.Add(node.Index, true);

            // Store the NodeIndexSet in the store
            NodeIndexSetStore.sets[newSetIndex] = siblingSet;

            // Update the dictionary with the reference to the NodeIndexSet
            transpositionsPosStandalone[hash64WithMoveAndReps] = GNodeIndexSetIndex.FromNodeSetIndex(newSetIndex);
          }
        }
        else
        {
          // Get the existing NodeIndexSet from the store
          int nodeSetIndex = setIndex.NodeSetIndex;
          NodeIndexSet siblingSet = NodeIndexSetStore.sets[nodeSetIndex];

          // Add the current node to the set
          // Note: This is not fully thread-safe as another thread could be updating this same NodeIndexSet
          // However, we accept the possibility of occasionally "lost" updates as an acceptable trade-off
          // for performance, since a "lost" update only means we might miss a potential transposition opportunity
          siblingSet.Add(node.Index, true);

          // Update the NodeIndexSet in the store
          NodeIndexSetStore.sets[nodeSetIndex] = siblingSet;
        }
      }
    }

    // Update hash table (null when single-dictionary mode skips 96-bit dict allocation).
    if (transpositionPositionAndSequence != null)
    {
      transpositionPositionAndSequence[posAndSequenceHash] = node.Index.Index;
    }

    nodeRef.HashStandalone = standaloneHash;
    nodeRef.NumPieces = (byte)mgPos.PieceCount;
    nodeRef.NumRank2Pawns = (byte)mgPos.NumPawnsRank2;
    nodeRef.IsGraphRoot = isRoot;
    nodeRef.IsWhite = mgPos.SideToMove == SideType.White;

    GameResult terminalStatus = mgPos.CalcTerminalStatus(moves); // TODO: potentially in most situations this is already known, do not recompute
    nodeRef.Terminal = terminalStatus;
    switch (terminalStatus)
    {
      case GameResult.Checkmate:
        //nodeRef.SetProvenLossAndPropagateToParent(node.GraphStore, 1, 0); // GFIX restore this logic!
        nodeRef.WinP = 0;
        nodeRef.LossP = 1;
        nodeRef.UncertaintyValue = 0;
        nodeRef.UncertaintyPolicy = 0;
        break;

      case GameResult.Draw:
        nodeRef.WinP = 0;
        nodeRef.LossP = 0;
        nodeRef.UncertaintyValue = 0;
        nodeRef.UncertaintyPolicy = 0;
        break;

      default:
        nodeRef.Terminal = GameResult.Unknown;
        nodeRef.WinP = FP16.NaN;
        nodeRef.LossP = FP16.NaN;
        break;
    }
  }


  public GEdge AddNewTerminalEdge(GNode parentNode, int indexOfChildInParent, double edgeV, double edgeD, int numVisits, bool propagateAsDraw)
  {
    Debug.Assert(numVisits > 0);
    Debug.Assert(Math.Abs(edgeV) <= EvaluatorSyzygy.BLESSED_WIN_LOSS_MAGNITUDE + 0.01 || Math.Abs(edgeV) >= 1.0);
    Debug.Assert(!propagateAsDraw || (edgeD == 1 && edgeV == 0));

    if (propagateAsDraw)
    {
      parentNode.SetDrawKnownToExistAtNode();
    }
    else
    {
      if (edgeV <= -1)
      {
        // TODO: consider if any of this is needed  parentNode.SetProvenLossAndPropagateToParent
      }
    }

    ref GEdgeStruct thisEdgeRef = ref InitializeNewEdge(parentNode, indexOfChildInParent, out GEdgeHeaderStruct thisEdgeHeader);

    // Setting the type (including information if drawn or lost) is
    // essential to do here during select phase (before backup) 
    // because another node in this same batch might revisit this edge
    // and will need to know what value to use for that path termination.
    thisEdgeRef.Type = Math.Abs(edgeV) <= (EvaluatorSyzygy.BLESSED_WIN_LOSS_MAGNITUDE + 0.01f) ? GEdgeStruct.EdgeType.TerminalEdgeDrawn : GEdgeStruct.EdgeType.TerminalEdgeDecisive;
    thisEdgeRef.P = thisEdgeHeader.P;
    thisEdgeRef.Move = thisEdgeHeader.Move;

    //      thisEdgeRef.W = numVisits * edgeV;
    //      thisEdgeRef.DSum = edgeV == 0 ? numVisits : 0;
    thisEdgeRef.QChild = edgeV;

    //      thisEdgeRef.N = numVisits;
    //      thisEdgeRef.W = numVisits * edgeV;
    thisEdgeRef.UncertaintyV = 0;
    thisEdgeRef.UncertaintyP = 0;

    // Other fields such as uncertainty and child index
    // can remain at their default values (zero).

    return new GEdge(ref thisEdgeRef, parent: parentNode, child: default);
  }


  static bool IsStandaloneHastableEligible(in MGPosition mgPos)
  {
    // Exclude two cases:
    //   - if repetition flag is set, then the context (move ordering) would be required to establish equivalence
    //   - if the Move50 counter is approaching the max value, the neural network evals
    //     begin to change rapidly (head toward draw as it approaches 100)
    return (mgPos.Rule50Count <= 90 && mgPos.RepetitionCount == 0);
  }


  /// <summary>
  /// Adds an edge to a new or existing child node, initializing the child node if necessary.
  /// In graph mode the returned GEdge may point to a child node already extant.
  /// </summary>
  /// <remarks>This method handles both the creation of a new child node and linking to an existing node if
  /// a transposition match is found. If a transposition match is found but the confirmation function returns false,
  /// the match is discarded.</remarks>
  /// <param name="parentNode">The parent node to which the edge will be added.</param>
  /// <param name="indexOfChildInParent">The index of the child node within the parent node's edge list. Must match the number of edges expanded in the
  /// parent node.</param>
  /// <param name="mgPos">The position data used to initialize the child node.</param>
  /// <param name="standaloneHash">A hash representing the position on a standalone basis.</param>
  /// <param name="lookupHash">A cumulative hash representing the sequence, order-insensitive except for the last element.</param>
  /// <param name="moves">A reference to the list of moves used to initialize the child node.</param>
  /// <returns>The edge connecting the parent node to the new or existing child node.</returns>
  public (GEdge childNode, bool wasCollision) AddEdgeToNewOrExistingNode(GNode parentNode,
                                                                         int indexOfChildInParent,
                                                                         in MGPosition mgPos,
                                                                         PosHash64 standaloneHash,
                                                                         PosHash96MultisetFinalized lookupHash,
                                                                         MGMoveList moves,
                                                                         out bool wasCreated,
                                                                         out GNode standaloneTranspositionNode,
                                                                         bool okToCreate)
  {
    Debug.Assert(parentNode.IsLocked);

    if (!parentNode.IsGraphRoot)
    {
      Debug.Assert(indexOfChildInParent == parentNode.NumEdgesExpanded); // assumed expanded in order
      Debug.Assert(!parentNode.ChildEdgeHeaderAtIndex(indexOfChildInParent).IsUnintialized); // expected that MoveInfo will be set before creation
    }

    // Use position-only dedup key (RepetitionCount=0, Move50Category=default) to match
    // the 96-bit hash behavior in PositionEquivalence mode. This ensures draw-by-repetition
    // positions (RepetitionCount=1) find the existing node instead of creating phantom nodes
    // with Q=NaN that break PUCT selection.
    PosHash64WithMove50AndReps standaloneKey = (SINGLE_DICTIONARY_POSITION_MODE && Store.UsesPositionEquivalenceMode)
      ? MGPositionHashing.Hash64WithMove50AndRepsAdded(standaloneHash, 0, default)
      : default;

    (GNode childNode, wasCreated, bool wasCollision) = LookupOrCreateAndAcquire(lookupHash, standaloneKey, okToCreate);
    Debug.Assert(childNode.IsNull || childNode.IsLocked);

    if (!wasCreated)
    {
      standaloneTranspositionNode = default;
      if (!okToCreate && childNode.IsNull)
      {
        return default;
      }
      else
      {
        NumLinksToExistingNodes++;
      }
    }
    else if (!wasCollision)
    {
      // Initialize the basic required fields.
      InitializeNodeForPos(childNode, false, in mgPos, moves, standaloneHash, lookupHash, out standaloneTranspositionNode);
    }
    else
    {
      standaloneTranspositionNode = default;
    }

    // Modify child entry to refer to this new child
    // N.B: It is essential to only swap out the child index here
    //      after the new child node has been fully initialized (above)
    //      because another thread might see and then follow the child reference.
    // no longer need parent[indexOfChildInParent].MoveInfoRef.SetExpandedChildIndex(childNode.Index);

    // Create a parent edge to point back to parent.
    // For newly created nodes, this establishes the tree-parent invariant:
    // the creating parent becomes position 0 in the parent list, guaranteeing
    // a cycle-free path to the root via TreeParentEdge/TreeParentNodeIndex.
    Store.ParentsStore.CreateParentEdge(parentNode.Index, childNode.Index);

#if DEBUG
    // Verify tree-parent invariant for newly created nodes:
    // the creating parent must be position 0 (single direct entry).
    if (wasCreated && !wasCollision)
    {
      Debug.Assert(childNode.NodeRef.ParentsHeader.IsDirectEntry,
        "Newly created node should have single direct parent entry");
      Debug.Assert(childNode.NodeRef.ParentsHeader.AsDirectParentNodeIndex == parentNode.Index,
        "Newly created node's first parent should be the creating parent");
    }
#endif

    GEdge ret = CreateEdge(parentNode, childNode, indexOfChildInParent);

    childNode.ReleaseLock();

    return (ret, wasCollision);
  }



  /// <summary>
  /// 
  /// </summary>
  /// <param name="parentNode"></param>
  /// <param name="childNode"></param>
  /// <param name="indexOfChildInParent"></param>
  /// <returns></returns>
  internal GEdge CreateEdge(GNode parentNode, GNode childNode, int indexOfChildInParent)
  {
    Debug.Assert(parentNode.IsLocked);

    ref GEdgeStruct thisEdgeRef = ref InitializeNewEdge(parentNode, indexOfChildInParent, out GEdgeHeaderStruct thisEdgeHeader);

    thisEdgeRef.Type = GEdgeStruct.EdgeType.ChildEdge;
    thisEdgeRef.P = thisEdgeHeader.P;
    thisEdgeRef.Move = thisEdgeHeader.Move;
#if ACTION_ENABLED
      thisEdgeRef.ActionV = thisEdgeHeader.ActionV;
      thisEdgeRef.ActionU = thisEdgeHeader.ActionU;
#endif
    thisEdgeRef.ChildNodeIndex = childNode.Index;

    if (childNode.IsEvaluated)
    {
      thisEdgeRef.SetUncertaintyValues(childNode.UncertaintyValue, childNode.UncertaintyPolicy);
    }

    return new GEdge(ref thisEdgeRef, parentNode, child: childNode);
  }


  static bool haveWarned = false;

  private ref GEdgeStruct InitializeNewEdge(GNode parentNode, int indexOfChildInParent, out GEdgeHeaderStruct thisEdgeHeader)
  {
    if (!haveWarned && indexOfChildInParent != parentNode.NumEdgesExpanded)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Warning: indexOfchildInParent = {indexOfChildInParent} but NumEdgesExpanded = {parentNode.NumEdgesExpanded} for {parentNode}");
      haveWarned = true;
    }
    Debug.Assert(indexOfChildInParent < parentNode.NumPolicyMoves);

    // Update number of expanded edges at the parent.
    parentNode.NodeRef.NumEdgesExpanded++;

    Span<GEdgeHeaderStruct> edgeHeaderStructsSpan = parentNode.EdgeHeadersSpan;

    int edgesStartBlockIndex;
    int edgesIndexInBlock = indexOfChildInParent % GEdgeStore.NUM_EDGES_PER_BLOCK;
    bool isFirstEdgeInBlock = edgesIndexInBlock == 0;

    if (isFirstEdgeInBlock)
    {
      // This will be the first edge in a new block. Allocate it.
      edgesStartBlockIndex = (int)Store.EdgesStore.AllocatedNewBlock();
    }
    else
    {
      // This edges shares same block as edge at prior index in parent.
      edgesStartBlockIndex = edgeHeaderStructsSpan[indexOfChildInParent - 1].EdgeStoreBlockIndex;
    }

    // Save a copy of the edge header before it is partly overwritten below.
    thisEdgeHeader = edgeHeaderStructsSpan[indexOfChildInParent];

    // Rewrite the GEdgeHeaderStruct to point to this block
    edgeHeaderStructsSpan[indexOfChildInParent].SetAsExpandedToEdgeBlock(edgesStartBlockIndex);

#if VERY_UNSAFE
    ref GEdgeStructBlocked refBlock = ref Store.EdgesStore.edgeStoreMemoryBuffer[edgesStartBlockIndex];
    return ref Unsafe.Add(ref Unsafe.As<GEdgeStructBlocked, GEdgeStruct>(ref refBlock), edgesIndexInBlock);
#else
    Span<GEdgeStruct> edgeSpanThisBlock = Store.EdgesStore.SpanAtBlockIndex(edgesStartBlockIndex);
    return ref edgeSpanThisBlock[edgesIndexInBlock];
#endif
  }


  /// <summary>
  /// Creates a new node and initializes its state (neural network evals) 
  /// by copying from an existing materialized node.
  /// </summary>
  /// <param name="copyFromNode"></param>
  /// <param name="initialN"></param>
  /// <returns></returns>
  internal NodeIndex CreateAndCopyNodeValues(GNode copyFromNode, int initialN)
  {
    // Assumed not locked because we acquire lock below in CopyNodeValues.
    Debug.Assert(!copyFromNode.IsLocked);

    // Create new node.
    NodeIndex newNodeIndex = Store.NodesStore.AllocateNext();
    GNode newNode = new(this, newNodeIndex);

    CopyNodeValues(initialN, copyFromNode, newNode, true);

    return newNodeIndex;
  }


  internal void CopyNodeValues(int numVisitsAccepted, GNode copyFromNode, GNode copyToNode, bool copyPolicy)
  {
    Debug.Assert(!Store.HasState); // Copying of state is not yet implemented
    Debug.Assert(!Store.HasAction); // Copying of action is not yet implemented
    Debug.Assert(copyFromNode.IsEvaluated);
    Debug.Assert(copyFromNode != copyToNode);

    // First make efficient shallow copy of node fields.
    ref readonly GNodeStruct copyFromRef = ref copyFromNode.NodeRef;
    ref GNodeStruct copyToRef = ref copyToNode.NodeRef;

    copyToRef.UncertaintyValue = copyFromRef.UncertaintyValue;
    copyToRef.UncertaintyPolicy = copyFromRef.UncertaintyPolicy;
    copyToRef.MRaw = copyFromRef.MRaw;
    copyToRef.HashStandalone = copyFromRef.HashStandalone;

    // Now adjust node fields that need to change
    copyToRef.NumPieces = copyFromNode.NodeRef.NumPieces;
    copyToRef.NumRank2Pawns = copyFromNode.NodeRef.NumRank2Pawns;
    copyToRef.IsWhite = copyFromNode.NodeRef.IsWhite;
    copyToRef.Terminal = copyFromNode.NodeRef.Terminal;
    copyToRef.IsOldGeneration = copyFromNode.NodeRef.IsOldGeneration;
    copyToRef.Move50Category = copyFromNode.NodeRef.Move50Category;
    copyToRef.HasRepetitions = copyFromNode.NodeRef.HasRepetitions;

    copyToRef.N = numVisitsAccepted;
    copyToRef.WinP = copyFromRef.WinP;
    copyToRef.LossP = copyFromRef.LossP;

    MCGSParamsFixed.AssertNotNaN(copyToNode.NodeRef.Q);

    if (copyPolicy)
    {
      using (new NodeLockBlock(copyFromNode))
      {
        using (new NodeLockBlock(copyToNode))
        {
          AllocateAndCopyPolicyValues(copyFromNode, copyToNode);
        }
      }
    }
    else
    {
      // Stuff the index of the node from which the policy will eventually be copied.
      Debug.Assert(copyFromNode.Graph == copyToNode.Graph); // cross-graph copy not supported
      copyToNode.NodeRef.edgeHeaderBlockIndexOrDeferredNode = new EdgeHeaderBlockIndexOrNodeIndex(copyFromNode.Index);
    }
  }


  internal static void AllocateAndCopyPolicyValues(GNode copyFrom, GNode copyTo)
  {
    if (copyFrom.NumPolicyMoves == 0)
    {
      return;
    }

    // Copy over moves and prior probabilities from the copyFrom node.
    GNodeEdgeHeaders allNewHeaders = copyTo.AllocatedEdgeHeaders(copyFrom.NumPolicyMoves);
    Span<GEdgeHeaderStruct> copyToHeaders = allNewHeaders.HeaderStructsSpan;

    Debug.Assert(copyTo.NumPolicyMoves > 0);

    // Copy the expanded edges one at a time
    // (since the source node headers are now just pointers we need to follow to get header info).
    foreach ((GEdge edge, int childIndex) in copyFrom.ChildEdgesExpandedWithIndex)
    {
#if ACTION_ENABLED
        copyToHeaders[childIndex].SetUnexpandedValues(edge.Move, edge.P, edge.ActionV, edge.ActionU);
#else
      copyToHeaders[childIndex].SetUnexpandedValues(edge.Move, edge.P, FP16.NaN, FP16.NaN);
#endif
    }

    // Copy unexpanded edges as just a sequence of records.
    int numExpanded = copyFrom.NumEdgesExpanded;
    Span<GEdgeHeaderStruct> copyFromHeaders = copyFrom.EdgeHeadersSpan[numExpanded..];
    copyToHeaders = copyToHeaders.Slice(numExpanded);
    copyFromHeaders.CopyTo(copyToHeaders);

    // When the source has expanded edges (numExpanded > 0), their P values may not be in
    // descending order — e.g. after selective rewrite Phase 4b compacts surviving expanded
    // edges by visit count (N), not policy (P). When continued search later re-expands all
    // reverted edges, the resulting fully-expanded node has two separately P-sorted segments.
    // Since all target headers are unexpanded, sort them by P descending to restore the invariant.
    if (numExpanded > 0)
    {
      Span<GEdgeHeaderStruct> allHeaders = allNewHeaders.HeaderStructsSpan;
      int nPol = copyFrom.NumPolicyMoves;
      for (int i = 1; i < nPol; i++)
      {
        GEdgeHeaderStruct key = allHeaders[i];
        float keyP = (float)key.RawP;
        int j = i - 1;
        while (j >= 0 && (float)allHeaders[j].RawP < keyP)
        {
          allHeaders[j + 1] = allHeaders[j];
          j--;
        }
        allHeaders[j + 1] = key;
      }

    }
  }


  /// <summary>
  /// Gathers information about children (visited and not) of a node.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="iteratorID"></param>
  /// <param name="maxIndex"></param>
  /// <param name="dualCollisionFraction"></param>
  /// <param name="stats"></param>
  /// <param name="refreshStaleEdges"></param>
  internal void GatherChildInfoViaChildren(GNode node,
                                           int iteratorID,
                                           int maxIndex,
                                           float dualCollisionFraction,
                                           GatheredChildStats stats,
                                           bool refreshStaleEdges)
  {
    Debug.Assert(node.IsSearchRoot || node.IsLocked);
    Debug.Assert(!node.BlockIndexIntoEdgeHeaderStoreIsDeferred);

    stats.ResetSummaryFields();

    Span<GEdgeHeaderStruct> childEdgeHeaders = node.EdgeHeadersSpan;


    Span<double> n = stats.N.Span;
    Span<double> nInFlightAdjusted = stats.NInFlightAdjusted.Span;
    Span<double> p = stats.P.Span;
    Span<double> w = stats.W.Span;
    Span<double> uv = stats.UV.Span;
#if ACTION_ENABLED
      Span<double> a = stats.A.Span;
#endif

    int numEdgesExpanded = node.NumEdgesExpanded;
    bool isIteratorIDZero = iteratorID == 0;

    for (int i = 0; i <= maxIndex; i++)
    {
      // Possibly start prefetching edge data future child blocks
      const int NUM_BLOCKS_PREFETCH_AHEAD = 1;
      int prefetchI = i + NUM_BLOCKS_PREFETCH_AHEAD * GEdgeStore.NUM_EDGES_PER_BLOCK;
      if (MCGSParamsFixed.PrefetchCacheLevel != Prefetcher.CacheLevel.None
        && prefetchI % GEdgeStore.NUM_EDGES_PER_BLOCK == 0
       && prefetchI < numEdgesExpanded)
      {
        void* nodePtr = Unsafe.AsPointer(ref node.EdgeStructAtIndexRef(childEdgeHeaders[prefetchI].EdgeStoreBlockIndex, 0));
        Prefetcher.PrefetchLevel1(nodePtr);
      }


      short sumVisitedThisChild;

      if (i < numEdgesExpanded)
      {
        // Use low-level accessor for efficiency
        int thisEdgeBlockIndex = childEdgeHeaders[i].EdgeStoreBlockIndex;
        ref GEdgeStruct refEdge = ref node.EdgeStructAtIndexRef(thisEdgeBlockIndex, i);

        // Refresh the edge Q if it was marked stale and requested.
        if (refreshStaleEdges && refEdge.IsStale)
        {
          GNode childNode = new GNode(node.Graph, refEdge.ChildNodeIndex);
          refEdge.QChild = childNode.Q;

          refEdge.IsStale = false;
        }

        p[i] = refEdge.P;
        n[i] = refEdge.N;
#if ACTION_ENABLED
          a[i] = (double)refEdge.ActionV;
#endif
        // Extract value uncertainty with fill-in if missing
        const double DEFAULT_UNCERTAINTY = 0.10f;
        uv[i] = float.IsNaN(refEdge.UncertaintyV) ? DEFAULT_UNCERTAINTY : (double)refEdge.UncertaintyV;

        w[i] = refEdge.N == 0 ? 0 : (refEdge.Q * refEdge.N);

        if (isIteratorIDZero)
        {
          Debug.Assert(refEdge.NumInFlight0 >= 0);
          nInFlightAdjusted[i] = refEdge.NumInFlight0 + dualCollisionFraction * refEdge.NumInFlight1;
        }
        else
        {
          Debug.Assert(refEdge.NumInFlight1 >= 0);
          nInFlightAdjusted[i] = refEdge.NumInFlight1 + dualCollisionFraction * refEdge.NumInFlight0;
        }
        sumVisitedThisChild = (short)(refEdge.NumInFlight0 + refEdge.NumInFlight1);
      }
      else
      {
        p[i] = (childEdgeHeaders[i].P);
        n[i] = 0;
#if ACTION_ENABLED
        a[i] = 0;
#endif
        uv[i] = 0;
        w[i] = 0;
        nInFlightAdjusted[i] = 0;
        sumVisitedThisChild = 0;
      }

      stats.SumNumInFlightAll += sumVisitedThisChild;

      double nI = n[i]; // Now n[i] is already an int, no need for conversion
      if (nI + sumVisitedThisChild > 0)
      {
        stats.SumPVisited += p[i];
      }

      if (nI > 0)
      {
        stats.SumNVisited += nI;
        stats.SumWVisited += w[i];
      }
    }
  }


  public IEnumerable<GNode> AllNodes
  {
    get
    {
      // Start at 1 to skip the null node.
      for (int i = 1; i < Store.NodesStore.NumTotalNodes; i++)
      {
        yield return new GNode(this, new NodeIndex(i));
      }
    }
  }


  public override string ToString()
  {
    return Store.IsDisposed ? "<Graph (Disposed)>"
                            : $"<Graph UsedNodes={Store.NodesStore.NumUsedNodes}>";
  }


  /// <summary>
  /// Dumps all nodes that are currently locked.
  /// </summary>
  public void DumpAllLockedNodes()
  {
    for (int i = 1; i <= Store.NodesStore.NumUsedNodes; i++)
    {
      if (this[i].IsLocked)
      {
        Console.WriteLine($"#{i,12:N0} locked: {this[i]}");
      }
    }
  }

  /// <summary>
  /// Dumps all standalone transpositions.
  /// </summary>
  public void DumpTranspositionsStandalone()
  {
    Console.WriteLine();
    Console.WriteLine($"Dump of standalone transpositions ({transpositionsPosStandalone.Count})");
    foreach (KeyValuePair<PosHash64WithMove50AndReps, GNodeIndexSetIndex> kvp in transpositionsPosStandalone)
    {
      if (!kvp.Value.IsNull)
      {
        if (kvp.Value.IsDirectNodeIndex)
        {
          // Direct node index case (single sibling)
          NodeIndex nodeIndex = new NodeIndex(kvp.Value.DirectNodeIndex);
          GNode node = this[nodeIndex];
          Console.WriteLine($"  Hash {kvp.Key,20} has 1 sibling (direct):");
          Console.WriteLine($"   {0,2:N0} {nodeIndex,-10:N0}  N={node.NodeRef.N,-10:N0}  V={node.NodeRef.V,6:F2} Q={node.NodeRef.Q,6:F2} "
                          + $"{node.NodeRef.Move50Category} {(node.NodeRef.HasRepetitions ? "R" : " ")}  {node.CalcPosition().ToPosition.FEN}");
        }
        else
        {
          // Set of node indices case
          NodeIndexSet siblingSet = NodeIndexSetStore.sets[kvp.Value.NodeSetIndex];
          Console.WriteLine($"  Hash {kvp.Key,20} has {siblingSet.Count} siblings: {siblingSet}");
          for (int i = 0; i < siblingSet.Count; i++)
          {
            NodeIndex nodeIndex = siblingSet[i];
            GNode node = this[nodeIndex];
            Console.WriteLine($"   {i,2:N0} {nodeIndex,-10:N0}  N={node.NodeRef.N,-10:N0}  V={node.NodeRef.V,6:F2} Q={node.NodeRef.Q,6:F2} "
                            + $"{node.NodeRef.Move50Category} {(node.NodeRef.HasRepetitions ? "R" : " ")}  {node.CalcPosition().ToPosition.FEN}");
          }
        }
      }
    }
  }


  /// <summary>
  /// Dumps the structure of nodes starting from the search root.
  /// </summary>
  /// <param name="maxDepth"></param>
  /// <param name="minEdgeN"></param>
  /// <param name="filterNodeIndex"></param>
  /// <param name="dumpUnvisited"></param>
  public void DumpNodesStructure(int maxDepth = int.MaxValue,
                                 int minEdgeN = 0,
                                 NodeIndex filterNodeIndex = default,
                                 bool dumpUnvisited = true)
  {
    HashSet<int> nodesAlreadySeen = new(this.Store.NodesStore.NumTotalNodes);

    Console.WriteLine();
    Console.WriteLine("=======================================================================================");
    Console.WriteLine($"Nodes structure for graph {this} {(GraphEnabled ? "+" : "-")}GraphEnabled {Store.NodesStore.PositionHistory}");
    DoDumpNodesStructure(this[priorSearchRoot], 1, dumpUnvisited, filterNodeIndex, false, default, nodesAlreadySeen, maxDepth, minEdgeN);
    Console.WriteLine();
  }


  void DoDumpNodesStructure(GNode node, int depthLevel, bool dumpUnvisited,
                            NodeIndex filterNodeIndex, bool haveSeenFilterNode,
                            GEdge edgeLeadingToThisNode, HashSet<int> nodesAlreadySeen,
                            int maxDepth,
                            int minEdgeN = 0,
                            bool suppressRepeatedSubgraphs = true)
  {
    if (filterNodeIndex.IsNull || node.Index == filterNodeIndex)
    {
      haveSeenFilterNode = true;
    }

    if (depthLevel > maxDepth)
    {
      return;
    }

    bool shouldWrite = haveSeenFilterNode || filterNodeIndex.IsNull;

    if (shouldWrite)
    {
      Console.Write(string.Empty.PadLeft(2 * depthLevel));
      if (suppressRepeatedSubgraphs && nodesAlreadySeen.Contains(node.Index.Index))
      {
        Console.WriteLine(" <ALREADY DUMPED " + node + " " + edgeLeadingToThisNode + ">");
        return;
      }

      nodesAlreadySeen.Add(node.Index.Index);

      Console.Write(node.ToString() + $" ({node.EdgeHeadersSpan.Length}m)");
      if (!edgeLeadingToThisNode.IsNull)
      {
        Console.Write($" from: {edgeLeadingToThisNode}");
      }
      Console.WriteLine();
    }

    foreach (GEdge childEdge in node.ChildEdgesExpanded)
    {
      if ((node.NumEdgesExpanded > 0 || dumpUnvisited) && childEdge.N >= minEdgeN)
      {
        if (childEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
        {
          GNode childNode = new(this, childEdge.ChildNodeIndex);
          DoDumpNodesStructure(childNode, depthLevel + 1, dumpUnvisited,
                               filterNodeIndex, haveSeenFilterNode,
                               childEdge, nodesAlreadySeen, maxDepth);
        }
        else // terminal
        {
          if (shouldWrite)
          {
            Console.WriteLine(string.Empty.PadLeft(2 * (depthLevel + 1)) + $" [{childEdge.Type} {childEdge.MoveMG} N={childEdge.N} P={100 * childEdge.P,6:F2}% Q={childEdge.Q} "
                                               + $"InFl={childEdge.NumInFlight0}/{childEdge.NumInFlight1}]");
          }
        }
      }
    }

    if (depthLevel == 1)
    {
      Console.WriteLine();
    }

  }


  /// <summary>
  /// Diagnostic method to dump contents of store to Console
  /// (optionally with full child detail).
  /// </summary>
  /// <param name="childDetail"></param>
  /// <param name="annotater">optionally a method called for each node which can further annotate</param>
  /// <param name="singleNodeIndex"></param>
  public void Dump(bool childDetail, Func<NodeIndex, string> annotater = null, int? singleNodeIndex = null)
  {
    if (IsDisposed)
    {
      Console.WriteLine("Graph is disposed - cannot Dump.");
    }

    Console.WriteLine();
    Console.WriteLine(ToString());
    Console.WriteLine("Prior moves " + Store.NodesStore.PositionHistory);
    Console.WriteLine();

    int startNodeIndex = singleNodeIndex ?? 1; // start at 1 to skip the null node
    int endNodeIndex = singleNodeIndex ?? Store.NodesStore.NumUsedNodes;
    for (int i = startNodeIndex; i <= endNodeIndex; i++)
    {
      ref GNodeStruct nodeRef = ref Store.NodesStore.nodes[i];
      GNode node = new(this, new NodeIndex(i));
      bool isWhite = nodeRef.IsWhite;
      string sideChar = isWhite ? "w" : "b";

      string annotation = annotater?.Invoke(new NodeIndex(i));
#if NOT
        Console.WriteLine($"{i,7} N={nodeRef.N} {sideChar}:{moveCorrectPerspective} V={nodeRef.V,6:F3} {nodeRef.Terminal} W={nodeRef.W,10:F3}"
                        + $" U={nodeRef.UncertaintyVPosition,6:F3} A={nodeRef.ActionV,6:F3} Parent={nodeRef.ParentIndex.Index} " +
                          $"InFlights={nodeRef.NInFlight}/{nodeRef.NInFlight1}" +
                          $"ChildStartIndex={nodeRef.ChildInfo.ChildInfoStartIndex(Store)} NumPolicyMoves={nodeRef.NumPolicyMoves} Cached?={nodeRef.IsCached} " + annotation);
#endif
      node.DumpRaw();

      if (nodeRef.IsOldGeneration)
      {
        Console.WriteLine("          OLD GENERATION");
      }
      else if (childDetail)
      {
        Console.WriteLine();
        Console.WriteLine();
        Ceres.Base.Misc.ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"------------------------------------------------- Parents -------------------------------------------------");
        foreach (GNode visitFromNode in node.Parents)
        {
          Console.WriteLine("  " + visitFromNode);
        }

        // GEdgeHeaders
        Console.WriteLine();
        Ceres.Base.Misc.ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"------------------------------------ GEdgeHeaderStruct [startBlockIndex={node.BlockIndexIntoEdgeHeaderStore}] ------------------------------------");
        int maxExpandedIndex = node.NumEdgesExpanded - 1;
        int childIndex = 0;
        foreach (GEdgeHeaderStruct child in node.EdgeHeadersSpan)
        {
          if (!child.IsExpanded)
          {
#if ACTION_ENABLED
              Console.WriteLine($"    {(isWhite ? child.Move.Flipped : child.Move)}  P={child.P,5:F3}  AV={child.ActionV,5:F3}  AU={child.ActionU,5:F3}");
#else
            Console.WriteLine($"    {(isWhite ? child.Move.Flipped : child.Move)}  P={child.P,5:F3}");
#endif
          }
          else
          {
            GEdge childEdge = node.ChildEdgeAtIndex(childIndex);
#if ACTION_ENABLED
              Console.WriteLine($"  --> {childEdge.ChildNodeIndex.Index}  {(isWhite ? childEdge.Move.Flipped : childEdge.Move)}  P={childEdge.P,5:F3}  AV={childEdge.ActionV,5:F3}  AU={childEdge.ActionU,5:F3}");
#else
            Console.WriteLine($"  --> {childEdge.ChildNodeIndex.Index}  {(isWhite ? childEdge.Move.Flipped : childEdge.Move)}  P={childEdge.P,5:F3}");
#endif
            // TODO: dump full properties on edge
          }

          if (childIndex > maxExpandedIndex + 2)
          {
            Console.WriteLine($"    ... followed by {nodeRef.NumPolicyMoves - childIndex - 1} additional unexpanded children ...");
            break;
          }

          childIndex++;
        }

        // GEdges
        Console.WriteLine();
        ConsoleUtils.WriteLineColored(ConsoleColor.Blue, $"------------------------------------  GEdgeStruct  ------------------------------------");
        int countVisitsTo = 0;
        foreach (GEdge childEdge in node.ChildEdgesExpanded)
        {
          Console.WriteLine($"[#{countVisitsTo}] " + childEdge);

          countVisitsTo++;
        }
        if (countVisitsTo == 0)
        {
          Console.WriteLine("  (none)");
        }

      }
      Console.WriteLine();
    }
  }


  /// <summary>
  /// TODO: temporary workaround for crashes
  /// </summary>
  /// <param name="pwh"></param>
  /// <returns></returns>
  public List<GraphRootToSearchRootNodeInfo> GuardedFindPathAlongPositionWithHistory(PositionWithHistory pwh)
  {
    try
    {
      return FindPathAlongPositionWithHistory(pwh);
    }
    catch (Exception ex)
    {
      Console.WriteLine($"Error in GuardedFindPathAlongPositionWithHistory: {ex.Message} {ex.StackTrace}");
      return default;
    }
  }


  /// <summary>
  /// Attempts to find sequence of GNodes descending from root corresponds to this PositionWithHistory.
  /// First element is not this, but rather first child.
  /// </summary>
  /// <returns></returns>
  public List<GraphRootToSearchRootNodeInfo> FindPathAlongPositionWithHistory(PositionWithHistory pwh)
  {
    Position[] positionsToMatch = pwh.Positions;
    Position[] positionsHistory = Store.PositionHistory.Positions;

    if (positionsToMatch.Length < positionsHistory.Length)
    {
      return default;
    }

    // Verify agreement in prehistory
    for (int i = 0; i < positionsHistory.Length; i++)
    {
      if (positionsToMatch[i] != positionsHistory[i])
      {
        // Mismatch in history
        return default;
      }
    }

    List<MGMove> movesToMatch = pwh.Moves;
    List<GraphRootToSearchRootNodeInfo> retInfo = [];

    MGPosition curPosNEW = positionsHistory[^1].ToMGPosition;
    MGPosition posChild = curPosNEW;
    GNode curNode = GraphRootNode;

    for (int i = positionsHistory.Length; i < positionsToMatch.Length; i++)
    {
      MGMove thisMove = movesToMatch[i - 1];

      MGPosition posParent = posChild;
      posChild.MakeMove(thisMove);

      GEdge thisEdge = curNode.EdgeForMove(thisMove);
      if (thisEdge.IsNull || thisEdge.Type != GEdgeStruct.EdgeType.ChildEdge)
      {
        return default;
      }
      curNode = thisEdge.ChildNode;

      if (curNode.IsNull)
      {
        return default;
      }

      PosHash64 childHash64 = MGPositionHashing.Hash64(in posChild);
      PosHash96 childHash96 = MGPositionHashing.Hash96(in posChild);

      bool isIrreversible = posParent.IsIrreversibleMove(thisMove, posChild);

      retInfo.Add(new GraphRootToSearchRootNodeInfo(curNode, in posChild, childHash64, childHash96, thisMove, isIrreversible));
    }

    return retInfo;
  }


  /// <summary>
  /// Attempts to retrieve the set of transposition nodes for the specified position hash.
  /// </summary>
  /// <param name="hash"></param>
  /// <param name="set"></param>
  /// <returns></returns>
  public bool TryGetTranspositionSet(PosHash64WithMove50AndReps hash, out NodeIndexSet set)
  {
    if (transpositionsPosStandalone.TryGetValue(hash, out GNodeIndexSetIndex setIndex))
    {
      Debug.Assert(!setIndex.IsDirectNodeIndex); // see comment below
      set = NodeIndexSetStore.sets[setIndex.NodeSetIndex];
      return true;
    }
    else
    {
      set = default;
      return false;
    }
  }


  /// <summary>
  /// Retrieves statistics from transposition nodes for the specified target node.
  /// </summary>
  /// <param name="targetNode">The target node to compute stats for</param>
  /// <param name="targetNodeNAfterPendingVisits">Number of visits including pending ones</param>
  /// <param name="hash">The position hash to look up</param>
  /// <returns>Tuple containing excess visit count and average Q value</returns>
  public (float sumExcessN, double avgQ) GetTranspositionStats(GNode targetNode,
                                                               int targetNodeNAfterPendingVisits,
                                                               PosHash64WithMove50AndReps hash)
  {
    // Try to find transposition nodes
    if (transpositionsPosStandalone.TryGetValue(hash, out GNodeIndexSetIndex setIndex))
    {
      if (setIndex.IsDirectNodeIndex)
      {
        // Direct node index case - just ourself, no transpositions available.
        return (0, targetNode.Q);
      }
      else
      {
        // Multiple transposition nodes in a set
        return NodeIndexSetStore.sets[setIndex.NodeSetIndex].Stats(targetNode, targetNodeNAfterPendingVisits);
      }
    }

    // No transposition nodes found or only the node itself
    return (0, 0);
  }



  /// <summary>
  /// Computes the contribution of a transposition node to the target node's statistics.
  /// This shared logic is used both for single nodes and within NodeIndexSet.Stats for multiple nodes.
  /// </summary>
  /// <param name="targetNode">The target node being evaluated</param>
  /// <param name="targetNodeNAfterPendingVisits">Visit count of target node including pending visits</param>
  /// <param name="transpositionNode">The transposition node to evaluate</param>
  /// <returns>Tuple of (excessN contribution, Q value to use)</returns>
  internal static (float excessNContribution, double qValue)
    TranspositionContribution(GNode targetNode,
                              int targetNodeNAfterPendingVisits,
                              GNode transpositionNode)
  {
    if (targetNode.Index == transpositionNode.Index)
    {
      // It's the same node - no information to gain
      return (0, 0);
    }

    int siblingN = transpositionNode.N;

    // Only include nodes with more visits than our current node
    if (!transpositionNode.IsEvaluated
     || siblingN <= targetNodeNAfterPendingVisits
     || !NodeIndexSet.IsEligibleForPseudoTranspositionContribution(transpositionNode))
    {
      return (0, 0);
    }

    // Make the contribution an increasing function of excess visits
    float excessN = siblingN - targetNodeNAfterPendingVisits;

    if (MCGSParamsFixed.SIBLING_POWER_SHRINK_SIBLING_N == 1)
    {
      return (excessN, transpositionNode.Q);
    }
    else
    {
      return (MathF.Pow(excessN, MCGSParamsFixed.SIBLING_POWER_SHRINK_SIBLING_N), transpositionNode.Q);
    }
  }



  protected virtual void Dispose(bool disposing)
  {
    if (!disposedValue)
    {
      if (disposing)
      {
        Store.Dispose();
        RewriterScratchBuffers?.Dispose();
        RewriterScratchBuffers = null;
      }

      // Set large fields to null
      transpositionPositionAndSequence = null;
      transpositionsPosStandalone = null;

      disposedValue = true;
    }
  }

  ~Graph() => Dispose(disposing: false);

  public void Dispose()
  {
    Dispose(disposing: true);
    GC.SuppressFinalize(this);
  }
}
