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

using Ceres.Base.Benchmarking;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;

using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Search.Coordination;

public partial class MCGSManager
{
  public const bool DEBUG_MODE = true;

  public static bool NodeMatchesPosition(GNode node, in Position pos, in MGPosition mgPos)
  {
    // Do quick checks to see if we can prove the node does not match the position.
    ref readonly GNodeStruct nodeRef = ref node.NodeRef;
    if (nodeRef.NumRank2Pawns != mgPos.NumPawnsRank2 || nodeRef.NumPieces != pos.PieceCount)
    {
      Console.WriteLine("disproved by numRank2Pawns or numPieces");
      return false;
    }

    // If annotation happens to be present, use it directly and easily to check for equality modulo repetition.
    //      ref NodeAnnotationX existingAnnotation = ref node.Annotation;
    //      return pos.EqualAsRepetition(in node.Annotation.Pos);

    // If this is not a leaf node, then we have have the legal moves already stored.
    // Compare these legal moves against the legal moves for the specified position.
    if (node.NodeRef.Terminal != GameResult.NotInitialized)
    {
      // Generate all legal moves from this position and verify the move count agrees.
      MGMoveList moves = new();
      MGMoveGen.GenerateMoves(mgPos, moves);

      if (nodeRef.NumPolicyMoves != moves.NumMovesUsed)
      {
        return false;
      }

      // Verify the legal moves from the position are all present as legal moves in the MoveInfos.
      Span<GEdgeHeaderStruct> moveInfos = node.EdgeHeadersSpan;
      List<EncodedMove> moveInfoMoves = [];
      for (int i = 0; i < moves.NumMovesUsed; i++)
      {
        MGMove graphMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(moveInfos[i].Move, in mgPos);
        bool exists = Array.Exists(moves.MovesArray, m => m == graphMove);

        if (!exists)
        {
          return false;
        }
      }
    }

    // GFIX: Only way to test for equality is to ascend the tree (until we see an already annotated node)
    //       to derive this position and then compare.
    //       This is expensive. Skip for now. Two options to eventually improve:
    //          -- use a longer hash code in NodeStruct, or
    //          -- reconstruct the position from the moves in the tree and then compare
    //      Console.WriteLine($"Unconfirmed link to {node} will be assumed valid");
    return true;
  }



  static int numCreated = 0;
  public static void TestRecursivePreloadTree()
  {
    const bool GRAPH_ENABLED = false;
    const bool HAS_ACTION = false;
    const bool HAS_STATE = false;

    Graph graph = new(1_000, HAS_ACTION, HAS_STATE, GRAPH_ENABLED, false, false, false, PositionWithHistory.StartPosition, false);
    GNode rootForRecursivePreload = graph.GraphRootNode;

    //      if (BUILD_TREE_MODE)
    {
      using (new TimingBlock("build/use"))
      {
        //1 20
        //2 400 --> 420 [421]
        //3 8902 --> 9322 [actual 8421] *** WRONG *******
        //4 197281 --> 206603 [actual 168421
        //5 4865609
        //6 119060324

        const bool PREFETCH = true;
        if (PREFETCH)
        {
          const int MAX_DEPTH = 1;// 5;// 10;// 10; // https://www.chessprogramming.net/perfect-perft/
          const int MAX_WIDTH = 999;// 10;// 15;
          string POS_FEN = Position.StartPosition.FEN;
          POS_FEN = "1r6/k1q2n2/np1r1p2/pRpPpP1p/P1P1B1pP/2P1B1P1/1RQ2K2/8 w - - 0 1"; // closed but black winning https://www.chess.com/computer-chess-championship#event=classical-cup-2-match-4&game=3
          Position pos = Position.FromFEN(POS_FEN);
          using (new TimingBlock($"build depth:{MAX_DEPTH} width:{MAX_WIDTH}"))
          {
            CreateAllUnvisitedChildrenRecursive(graph, rootForRecursivePreload, pos.ToMGPosition, in pos, 0, MAX_DEPTH, MAX_WIDTH);
          }
        }

        int totalTreeNodes = graph.Store.NodesStore.NumUsedNodes + graph.NumLinksToExistingNodes;
        Console.WriteLine("TOTAL NODES (tree): " + totalTreeNodes);
        Console.WriteLine("NODE COUNT        : " + graph.Store.NodesStore.NumUsedNodes);
        Console.WriteLine("LINK COUNT        : " + graph.NumLinksToExistingNodes);
        Console.WriteLine("USAGE RATIO       : " + (float)graph.Store.NodesStore.NumUsedNodes / totalTreeNodes);

        graph.Validate();

        if (DEBUG_MODE)
        {
          graph.Dump(false);
        }
      }

      return;
    }
  }



  internal static void CreateAllUnvisitedChildrenRecursive(Graph graph,
                                                           GNode node,
                                                           in MGPosition mgPos,
                                                           in Position pos,
                                                           int nodeDepth, int maxDepth, int maxChildrenPerNode)
  {
    throw new NotImplementedException();
  }


  // N.B. also requires flag in ParamsSearch to be overridden from false to true
  public static bool ENABLE_PREFETCH = true;


  public void RunBatchSelectAndBackup(PositionWithHistory pos, int targetNumVisits, bool debugMode)
  {
    if (ParamsSearch.PrefetchParams != null)
    {
      GraphPrefetcher prefetcher = new(this);

      prefetcher.DoPrefetch(ParamsSearch.PrefetchParams,
                            Math.Min(NNEvaluator0.MaxBatchSize, ParamsSearch.Execution.MaxBatchSize),
                            ParamsSearch.DebugDumpVerifyMode);

      // Possibly use the newly gathered value head info to
      // rearrange children based on attractiveness of V.
      if (ParamsSearch.PrefetchParams.PrefetchResortChildrenUsingV)
      {
        prefetcher.PossiblyResortChildrenUsingV();
      }
    }


    SearchLimit searchLimit = SearchLimit.NodesPerMove(targetNumVisits);
    //      Console.WriteLine(searchLimit.HardMaxNumFinalNodes());
    //      searchLimit.HardMaxNumFinalNodes = targetNumVisits + 1000;

    RunLoopUntilGraphSize(pos, SearchLimit.NodesPerMove(targetNumVisits));

    if (debugMode)
    {
      //coordinator.Graph.Dump(true);
      //graph.DumpNodesStructure();
      //Console.WriteLine($"Processed {targetNumVisits} visits, dumped graph above.");
    }

    if (debugMode)
    {
      Console.WriteLine("VALIDATING GRAPH...");

      Engine.Graph.Validate(true);

      //        int numWasted = numNodes - numVisited;
      //        Console.WriteLine($"Batches {SelectTerminatorNeuralNet.CountTotalBatches}, positions {SelectTerminatorNeuralNet.CountTotalPositions}");
      //        Console.WriteLine($"Num nodes {numNodes}, num visited {numVisited}, prefetched {totalPrefetched}, wasted {numWasted}");
    }
  }
}
