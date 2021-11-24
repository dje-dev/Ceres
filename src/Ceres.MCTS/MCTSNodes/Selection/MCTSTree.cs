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
using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Boards;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.NodeCache;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.LeafExpansion
{
  /// <summary>
  /// Encapsulates data structures needed to do MCTS search on a tree,
  /// including reference to underlying node store and associated data structures
  /// such as the dictionary of transposition roots.
  /// </summary>
  public class MCTSTree
  {
    const bool VERBOSE = false;

    PositionWithHistory PriorMoves => Store.Nodes.PriorMoves;

    MCTSIterator Context;
    public IMCTSNodeCache NodeCache;


    /// <summary>
    /// Underlying store of nodes.
    /// </summary>
    public readonly MCTSNodeStore Store;

    /// <summary>
    /// Optionally a set of evaluators that will be invoked upon each annotation.
    /// </summary>
    public List<LeafEvaluatorBase> ImmediateEvaluators { get; set; }

    /// <summary>
    /// Maintains a dictionary mapping position hash code 
    /// to index of correspondonding node in tree (if transposition table enabled).
    /// </summary>
    public TranspositionRootsDict TranspositionRoots;


    /// <summary>
    /// Optionally an externally provided position cache
    /// may be provided to obviate neural network evaluation.
    /// </summary>
    public PositionEvalCache PositionCache;

    MemoryBufferOS<MCTSNodeStruct> nodes;

    public readonly int EstimatedNumNodes;

    /// <summary>
    /// Set encoded board positions corresponding to positions prior to root position in history.
    /// </summary>
    public List<EncodedPositionBoard> EncodedPriorPositions;



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="context"></param>
    /// <param name="positionCache"></param>
    public MCTSTree(MCTSNodeStore store, MCTSIterator context,
                    int estimatedNumNodes,
                    PositionEvalCache positionCache, IMCTSNodeCache reuseNodeCache)
    {
      if (ParamsSearch.DrawByRepetitionLookbackPlies > MAX_LENGTH_POS_HISTORY)
      {
        throw new Exception($"DrawByRepetitionLookbackPlies exceeds maximum length of {MAX_LENGTH_POS_HISTORY}");
      }

      Store = store;
      Context = context;
      PositionCache = positionCache;
      EstimatedNumNodes = estimatedNumNodes;

      nodes = store.Nodes.nodes;

      NodeCache = reuseNodeCache ?? MakeNodeCache(reuseNodeCache);

      // Populate EncodedPriorPositions with encoded boards
      // corresponding to possible prior moves (before the root of this search)
      EncodedPriorPositions = new List<EncodedPositionBoard>();
      Position[] priorPositions = new Position[9];

      // Get prior positions (last position has highest index)
      priorPositions = PositionHistoryGatherer.DoGetHistoryPositions(PriorMoves, priorPositions, 0, 8, false).ToArray();
      for (int i = priorPositions.Length - 1; i >= 0; i--)
      {
        EncodedPositionBoard thisBoard = EncodedPositionBoard.GetBoard(in priorPositions[i], priorPositions[i].MiscInfo.SideToMove, false);
        EncodedPriorPositions.Add(thisBoard);
      }
    }

    /// <summary>
    /// Constructs and returns an IMCTSNodeCache of appropriate type and size.
    /// </summary>
    /// <param name="reuseNodeCache"></param>
    /// <returns></returns>
    IMCTSNodeCache MakeNodeCache(IMCTSNodeCache reuseNodeCache)
    {
      int configuredCacheSize = Context.ParamsSearch.Execution.NodeAnnotationCacheSize;

#if NOT
      TODO: restore, find some way of knowing max size
      // If all nodes fit within configured cache size, use a single cache.
      if (MaxNodesBound < configuredCacheSize)
      {
        return new MCTSNodeCacheArrayPurgeable(this, MaxNodesBound + 2000);
      }
#endif

#if NOT
      // Cache must be large enough to hold the full "working set" of nodes during leaf selection
      // (all selected leafs and their antecedents plus anciallary nodes such as associated transposition roots).
      const int ANNOTATION_MIN_CACHE_SIZE = 50_000;
      if (configuredCacheSize < ANNOTATION_MIN_CACHE_SIZE && configuredCacheSize < MaxNodesBound)
      {
        throw new Exception($"NODE_ANNOTATION_CACHE_SIZE is below minimum size of {ANNOTATION_MIN_CACHE_SIZE}");
      }
#endif
#if NOT
      TODO: possibly restore this code, after making sure MCTSNodeCacheArrayFixed is up-to-date
            Alternately maybe not necessary, MCTSNodeCacheArrayPurgeable may already be sufficiently or more efficient.
      if (maxNodesBound <= annotationCacheSize && !context.ParamsSearch.TreeReuseEnabled)
      {
        // We know with certainty the maximum size, and it will fit inside the cache
        // without purging needed - so just use a simple fixed size cache
        cache = new MCTSNodeCacheArrayFixed(this, maxNodesBound);
      }
      else
      {
#endif
      int numCachedNodes = configuredCacheSize;
      int? maxTreeNodes = CeresUserSettingsManager.Settings.MaxTreeNodes;
      if (maxTreeNodes.HasValue && maxTreeNodes.Value < numCachedNodes)
      {
        numCachedNodes = maxTreeNodes.Value;
      }
      return  new MCTSNodeCacheArrayPurgeableSet(Store, numCachedNodes,  EstimatedNumNodes);
    }

    public void PossiblyPruneCache() => NodeCache.PossiblyPruneCache(MCTSManager.ThreadSearchContext.Tree.Store);


    /// <summary>
    /// Returns root of node tree.
    /// </summary>
    public MCTSNode Root => GetNode(Store.RootIndex);


#if NOT
    /// <summary>
    /// Returns the MCTSNode stored in the cache 
    /// corresponding to specified MCTSNodeStruct
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeRef"></param>
    /// <returns></returns>
    public unsafe MCTSNode GetNode(in MCTSNodeStruct nodeRef)
    {
      MCTSNode node = cache.Lookup(in nodeRef);
      int nodeIndex = node == null ? (int)Store.Nodes.NodeOffsetFromFirst(in nodeRef) 
                                   : node.Index;
      return DoGetNode(node, new MCTSNodeStructIndex(nodeIndex), null);
    }
#endif

    static readonly MCTSNode nullNode = new MCTSNode(null, null, default, false);

    /// <summary>
    /// Attempts to return the MCTSNode associated with an annotation in the cache, 
    /// or null if not found
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public unsafe MCTSNode GetNode(MCTSNodeStructIndex nodeIndex)
    {
      if (nodeIndex.Index == 0)
      {
        return nullNode;
      }

      ref readonly MCTSNodeStruct nodeRef = ref nodes[nodeIndex.Index];
      bool wasAdded = false;
      if (nodeRef.CachedInfoPtr == null)
      {
        NodeCache.Add(nodeIndex);
        Debug.Assert(nodeRef.CachedInfoPtr != null);
        wasAdded = true;
      }

      MCTSNode node = new MCTSNode(Context, nodes, nodeIndex, wasAdded);
      node.InfoRef.LastAccessedSequenceCounter = NodeCache.NextBatchSequenceNumber;

      Debug.Assert(node.Index == nodeIndex.Index);

      return node;
    }


    [ThreadStatic]
    static Position[] posScratchBuffer;

    const int MAX_LENGTH_POS_HISTORY = 255;
    static Position[] PosScratchBuffer
    {
      get
      {
        Position[] buffer = posScratchBuffer;
        if (buffer != null)
        {
          return buffer;
        }

        return posScratchBuffer = new Position[MAX_LENGTH_POS_HISTORY];
      }
    }


    /// <summary>
    /// Initializes all fields in the Annotation field (if not already initialized).
    /// Also runs the immediate evaluators to determine if the node can be immediately applied.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="parentAnnotation">optionally a precomputed parent annotation (otherwise computed)</param>
    /// <returns></returns>
    public unsafe void Annotate(MCTSNode node, bool forceAnnotate = false)
    {
      if (!forceAnnotate & node.IsAnnotated)
      {
        return;
      }

      Debug.Assert(!node.StructRef.IsOldGeneration);
      //      Debug.Assert(!node.Ref.IsTranspositionLinked);

      ref MCTSNodeAnnotation annotation = ref node.InfoRef.Annotation;

      // Get the position corresponding to this node
      MGPosition newPos;
      if (!node.IsRoot)
      {
        // Apply move for this node to the prior position
        node.Parent.Annotate();
        newPos = node.Parent.Annotation.PosMG;
        newPos.MakeMove(ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(node.PriorMove, in newPos, true));
      }
      else
      {
        newPos = PriorMoves.FinalPosMG;
      }

      MGMove priorMoveMG = default;
      if (!node.IsRoot) priorMoveMG = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(node.PriorMove, in node.Parent.Annotation.PosMG, true);

      bool isRoot = node.IsRoot;

      Position newPosAsPos = newPos.ToPosition;
      Debug.Assert(newPosAsPos.PieceExists(Pieces.WhiteKing) && newPosAsPos.PieceExists(Pieces.BlackKing));

      // Create history, with prepended move representing this move.
      // Load only a limited position history for efficiency,
      // unless near the root where draw by repetition especially critical to detect.
      // TODO: Upon tree reuse, do we need to revisit shallow nodes with this deeper draw detection?
      int drawLookbackPlies = node.Depth < 5 ? MAX_LENGTH_POS_HISTORY : ParamsSearch.DrawByRepetitionLookbackPlies;
      Span<Position> posHistory = GetPriorHistoryPositions(node.Parent, in newPosAsPos, PosScratchBuffer,
                                                           drawLookbackPlies, true);

      // Determine the set (possibly a subset) of positions over which to compute hash
      Span<Position> posHistoryForCaching = posHistory;
      int numCacheHashPositions = node.Context.EvaluatorDef.NumCacheHashPositions;
      if (posHistory.Length > numCacheHashPositions)
      {
        posHistoryForCaching = posHistory.Slice(posHistory.Length - numCacheHashPositions, numCacheHashPositions);
      }

      node.InfoRef.LastAccessedSequenceCounter = node.Tree.NodeCache.NextBatchSequenceNumber;
      annotation.PriorMoveMG = priorMoveMG;
      annotation.PosMG = newPos;

      const bool FAST = true;
      if (!FAST)
      {
        annotation.LC0BoardPosition = EncodedPositionBoard.GetBoard(in posHistory[^1], annotation.PosMG.SideToMove, posHistory[^1].MiscInfo.RepetitionCount > 0);
      }
      else
      {
        EncodedPositionBoard.SetBoard(ref annotation.LC0BoardPosition, in annotation.PosMG, annotation.PosMG.SideToMove, posHistory[^1].MiscInfo.RepetitionCount > 0);
        annotation.MiscInfo = EncodedPositionWithHistory.GetMiscFromPosition(posHistory[^1].MiscInfo, SideType.White);
      }

      // Consider the annotation complete at this point where we assign Pos
      // (initialization status is considered complete when number of pieces > 0).
      annotation.Pos = posHistory[^1]; // this will have had its repetition count set

      //node.Ref.HashCrosscheck = annotation.Pos.PiecesShortHash;
      bool alreadyEvaluated = !FP16.IsNaN(node.V);

      if (alreadyEvaluated)
      {
        // Reconstruct the eval result from value already stored
        node.EvalResult.Initialize(node.Terminal, node.WinP, node.LossP, node.MPosition);

        // Re-establish linkages to transposition root node, if any.
        if (TranspositionRoots.TryGetValue(node.StructRef.ZobristHash, out int transpositionNodeIndex))
        {
          if (transpositionNodeIndex != node.Index)
          {
            node.InfoRef.TranspositionRootNodeIndex = new MCTSNodeStructIndex(transpositionNodeIndex);
          }
        }
      }
      else
      {
        node.StructRef.ZobristHash = EncodedBoardZobrist.ZobristHash(posHistoryForCaching, node.Context.EvaluatorDef.HashMode);

        if (!alreadyEvaluated && ImmediateEvaluators != null)
        {
          foreach (LeafEvaluatorBase immediateEvaluator in ImmediateEvaluators)
          {
            LeafEvaluationResult result = immediateEvaluator.TryEvaluate(node);
            if (!result.IsNull)
            {
              node.EvalResult = result;
              break;
            }

          }
        }
      }

    }





    /// <summary>
    /// Returns a partial set of history positions from the antecedents of this node (possibly also including initial move sequence).
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    public Span<Position> HistoryPositionsForNode(MCTSNode node, int maxLookback = MAX_LENGTH_POS_HISTORY - 1)
    {
      maxLookback = Math.Min(maxLookback, MAX_LENGTH_POS_HISTORY - 1);
      return GetPriorHistoryPositions(node.Parent, in node.Annotation.Pos, PosScratchBuffer, maxLookback, true);
    }


    /// <summary>
    /// Diagnostic helper method to dump the history positions
    /// corresponding to this node to the Console.
    /// </summary>
    /// <param name="node"></param>
    public void DumpNodePositionHistory(MCTSNode node)
    {
      Console.WriteLine("\r\n" + node.ToString());
      int count = 0;
      foreach (Position pos in HistoryPositionsForNode(node, ParamsSearch.DrawByRepetitionLookbackPlies))
      {
        Console.WriteLine($"  {count++,4:F0} {pos.MiscInfo.RepetitionCount,4:F0} { pos.FEN }");
      }
    }


    /// <summary>
    /// Returns Span of prior positions form specified node.
    /// 
    /// Note that we expect the positions stored in each node already include the repetition counts (if any)
    /// thus this does not need to be recomputed here.
    /// </summary>
    /// <param name="posSpan"></param>
    /// <param name="maxPositions"></param>
    /// <returns></returns>
    Span<Position> GetPriorHistoryPositions(MCTSNode node, in Position prependPosition,
                                            Span<Position> posSpan, int maxPositions, bool setFinalPositionRepetitionCount)
    {
      // Gather history positions by ascending tree
      posSpan[0] = prependPosition;
      int count = 1;
      MCTSNode thisAnnotation = node;
      while (count < maxPositions && !thisAnnotation.IsNull)
      {
        posSpan[count++] = thisAnnotation.Annotation.Pos;
        thisAnnotation = thisAnnotation.Parent;
      }

      // Gather positions from this path (possibly also reaching back into the start sequence)
      return PositionHistoryGatherer.DoGetHistoryPositions(PriorMoves, posSpan, count, maxPositions, setFinalPositionRepetitionCount);
    }

    /// <summary>
    /// Resets cache of nodes, optionally also resetting the CacheIndex of each node.
    /// </summary>
    /// <param name="resetNodeCacheIndex"></param>
    public void ClearNodeCache(bool resetNodeCacheIndex) => NodeCache.ResetCache(resetNodeCacheIndex);
  }


  internal class NodeComparer : IEqualityComparer<MCTSNode>
  {
    public bool Equals(MCTSNode x, MCTSNode y) => x.Index == y.Index;
    public int GetHashCode(MCTSNode node) => node.Index.GetHashCode();
  }

}
