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

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Boards;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;

using Ceres.MCTS.Evaluators;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Annotation;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.NodeCache;

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

    /// <summary>
    /// Nodes in node cache are stamped with the sequence number
    /// of the last batch in which they were accessed.
    /// </summary>
    internal long BATCH_SEQUENCE_COUNTER = 0;


    PositionWithHistory PriorMoves => Store.Nodes.PriorMoves;

    MCTSIterator Context;
    IMCTSNodeCache cache;


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


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="store"></param>
    /// <param name="context"></param>
    /// <param name="maxNodesBound"></param>
    /// <param name="positionCache"></param>
    public MCTSTree(MCTSNodeStore store, MCTSIterator context,
                    int maxNodesBound, int estimatedNumNodes,
                    PositionEvalCache positionCache)
    {
      if (context.ParamsSearch.DrawByRepetitionLookbackPlies > MAX_LENGTH_POS_HISTORY)
      {
        throw new Exception($"DrawByRepetitionLookbackPlies exceeds maximum length of {MAX_LENGTH_POS_HISTORY}");
      }

      Store = store;
      Context = context;
      PositionCache = positionCache;

      const int ANNOTATION_MIN_CACHE_SIZE = 50_000;
      int annotationCacheSize = Math.Min(maxNodesBound, context.ParamsSearch.Execution.NodeAnnotationCacheSize);
      if (annotationCacheSize < ANNOTATION_MIN_CACHE_SIZE
       && annotationCacheSize < maxNodesBound)
      {
        throw new Exception($"NODE_ANNOTATION_CACHE_SIZE is below minimum size of {ANNOTATION_MIN_CACHE_SIZE}");
      }

      if (maxNodesBound <= annotationCacheSize && !context.ParamsSearch.TreeReuseEnabled)
      {
        // We know with certainty the maximum size, and it will fit inside the cache
        // without purging needed - so just use a simple fixed size cache
        cache = new MCTSNodeCacheArrayFixed(this, maxNodesBound);
      }
      else
      {
        cache = new MCTSNodeCacheArrayPurgeableSet(this, annotationCacheSize, estimatedNumNodes);
      }

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
    /// Set encoded board positions corresponding to positions prior to root position in history.
    /// </summary>
    public List<EncodedPositionBoard> EncodedPriorPositions;

    public void PossiblyPruneCache() => cache.PossiblyPruneCache(MCTSManager.ThreadSearchContext.Tree.Store);


    MCTSNode cachedRoot = null;

    /// <summary>
    /// Returns root of node tree.
    /// </summary>
    public MCTSNode Root => cachedRoot is null ? (cachedRoot = GetNode(Store.RootIndex)) : cachedRoot;



    /// <summary>
    /// Returns the MCTSNode stored in the cache 
    /// corresponding to specified MCTSNodeStruct
    /// or null if not currently cached.
    /// </summary>
    /// <param name="nodeRef"></param>
    /// <returns></returns>
    public MCTSNode GetNode(in MCTSNodeStruct nodeRef)
    {
      MCTSNode node = cache.Lookup(in nodeRef);
      int nodeIndex = node == null ? (int)Store.Nodes.NodeOffsetFromFirst(in nodeRef) 
                                   : node.Index;
      return DoGetNode(node, new MCTSNodeStructIndex(nodeIndex), null);
    }


    /// <summary>
    /// Attempts to return the MCTSNode associated with an annotation in the cache, 
    /// or null if not found
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    public MCTSNode GetNode(MCTSNodeStructIndex nodeIndex, MCTSNode parent = null, bool checkCache = true)
    {
      MCTSNode ret = checkCache ? cache.Lookup(nodeIndex) : null;
      return DoGetNode(ret, nodeIndex, parent, checkCache);
    }

    /// <summary>
    /// Attempts to return the MCTSNode associated with an annotation in the cache, 
    /// or null if not found
    /// </summary>
    /// <param name="nodeIndex"></param>
    /// <returns></returns>
    MCTSNode DoGetNode(MCTSNode node, MCTSNodeStructIndex nodeIndex, MCTSNode parent = null, bool checkCache = true)
    {
      if (node is null)
      {
        ref readonly MCTSNodeStruct nodeRef = ref Store.Nodes.nodes[nodeIndex.Index];
        if (parent is null && !nodeIndex.IsRoot)
        {
          Debug.Assert(nodeRef.ParentIndex.Index != nodeIndex.Index);

          parent = GetNode(nodeRef.ParentIndex, null);
        }

        node = new MCTSNode(Context, nodeIndex, parent);
        cache?.Add(node);
      }

      node.LastAccessedSequenceCounter = BATCH_SEQUENCE_COUNTER;

      Debug.Assert(node.Index == nodeIndex.Index);

      return node;
    }


    [ThreadStatic]
    static Position[] posScratchBuffer;

    const int MAX_LENGTH_POS_HISTORY = 100;
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
    public unsafe void Annotate(MCTSNode node)
    {
      if (node.IsAnnotated)
      {
        return;
      }

      Debug.Assert(!node.Ref.IsOldGeneration);
      //      Debug.Assert(!node.Ref.IsTranspositionLinked);

      ref MCTSNodeAnnotation annotation = ref node.Annotation;

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

      // Create history, with prepended move representing this move
      Span<Position> posHistory = GetPriorHistoryPositions(node.Parent, in newPosAsPos, PosScratchBuffer,
                                                           node.Context.ParamsSearch.DrawByRepetitionLookbackPlies, true);

      // Determine the set (possibly a subset) of positions over which to compute hash
      Span<Position> posHistoryForCaching = posHistory;
      int numCacheHashPositions = node.Context.EvaluatorDef.NumCacheHashPositions;
      if (posHistory.Length > numCacheHashPositions)
      {
        posHistoryForCaching = posHistory.Slice(posHistory.Length - numCacheHashPositions, numCacheHashPositions);
      }

      // Compute the actual hash
      ulong zobristHashForCaching = EncodedBoardZobrist.ZobristHash(posHistoryForCaching, node.Context.EvaluatorDef.HashMode);

      node.LastAccessedSequenceCounter = node.Tree.BATCH_SEQUENCE_COUNTER;
      annotation.PriorMoveMG = priorMoveMG;

      annotation.Pos = posHistory[^1]; // this will have had its repetition count set
      annotation.PosMG = newPos;

      node.Ref.ZobristHash = zobristHashForCaching;
      //node.Ref.HashCrosscheck = annotation.Pos.PiecesShortHash;

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

      bool alreadyEvaluated = !FP16.IsNaN(node.V);

      // Consider the annotation complete at this point
      // (code which sets the EvalResult below expects that).
      annotation.IsInitialized = true;

      if (alreadyEvaluated)
      {
        // Reconstruct the eval result from value already stored
        node.EvalResult.Initialize(node.Terminal, node.WinP, node.LossP, node.MPosition);
      }
      else
      {
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

      // Possibly scan sibling and record information for
      // possible use later in value backup.
      if (node.N == 0
       && node.Context.ParamsSearch.EnableUseSiblingEvaluations
       && !node.IsRoot)
      {
        MCTSNodeSiblingEval.TrySetPendingSiblingValue(node, numCacheHashPositions);
      }

    }





    /// <summary>
    /// Returns a partial set of history positions from the antecedents of this node (possibly also including initial move sequence).
    /// </summary>
    /// <param name="node"></param>
    /// <returns></returns>
    public Span<Position> HistoryPositionsForNode(MCTSNode node)
    {
      return GetPriorHistoryPositions(node.Parent, in node.Annotation.Pos, PosScratchBuffer,
                                      node.Context.ParamsSearch.DrawByRepetitionLookbackPlies, true);
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
      foreach (Position pos in HistoryPositionsForNode(node))
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
    Span<Position> GetPriorHistoryPositions(MCTSNode parent, in Position prependPosition,
                                            Span<Position> posSpan, int maxPositions, bool setFinalPositionRepetitionCount)
    {
      // Gather history positions by ascending tree
      posSpan[0] = prependPosition;
      int count = 1;
      MCTSNode thisAnnotation = parent;
      while (count < maxPositions && thisAnnotation != null)
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
    public void ClearNodeCache(bool resetNodeCacheIndex) => cache.ResetCache(resetNodeCacheIndex);
  }


  internal class NodeComparer : IEqualityComparer<MCTSNode>
  {
    public bool Equals(MCTSNode x, MCTSNode y) => x.Index == y.Index;
    public int GetHashCode(MCTSNode node) => node.Index.GetHashCode();
  }

}
