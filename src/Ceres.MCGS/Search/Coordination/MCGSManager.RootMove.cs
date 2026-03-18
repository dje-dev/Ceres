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
using System.Linq;
using Ceres.Base.DataTypes;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Managers.Limits;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.Coordination;

public partial class MCGSManager : IDisposable
{
  /// <summary>
  /// The optimal move selected by tablebase probe (if any).
  /// </summary>
  public MGMove TablebaseImmediateBestMove;

  /// <summary>
  /// If playing in TopV mode (best value move) then this is set.
  /// </summary>
  public MGMove TopVForcedMove;

  /// <summary>
  /// Probes the tablebases (if any) for the best move at root,
  /// subject to verification that the possible move will not 
  /// trigger a draw by repetition given the present move history.
  /// </summary>
  /// <param name="node"></param>
  /// <returns></returns>
  internal (WDLResult result, MGMove immediateMove) TryGetTablebaseImmediateMove(GNode node)
  {
    // Not possible to find if tablebase method is not installed (not available).
    if (CheckTablebaseBestNextMove == null)
    {
      return (WDLResult.Unknown, default);
    }

    Position pos = node.IsSearchRoot ? Engine.SearchRootPosMG.ToPosition : node.CalcPosition().ToPosition;
    MGMove immediateMove = CheckTablebaseBestNextMove(in pos, out WDLResult result, out List<(MGMove, short)> fullWinningMoveList, out bool winningMoveListOrderedByDTM);
    if (result == WDLResult.Win && !winningMoveListOrderedByDTM)
    {
      // Not safe to rely upon the tablebase probe because unable (e.g. because of missing DTZ files)
      // to indicate which moves bring us closer to checkmate.
      // Instead return failure in lookup thus engine will do normal search.
      // However filter search moves to only include the winning moves.
      if (fullWinningMoveList != null && fullWinningMoveList.Count > 0)
      {
        List<MGMove> moveList = new(fullWinningMoveList.Count);
        foreach ((MGMove, short) move in fullWinningMoveList)
        {
          moveList.Add(move.Item1);
        }
        TerminationManager.SearchMovesTablebaseRestricted = moveList;
      }

      return (WDLResult.Unknown, default);
    }
    else if (result == WDLResult.Draw && !winningMoveListOrderedByDTM)
    {
      // TODO: Improve this: in some situations, the best move coming back might not be actually best
      //       due to partial TB files (winningMoveListOrderedByDTM is false indicating search required).
      //       TODO: centralize this logic appearing in multiple places.
      return (WDLResult.Unknown, default);
    }

    if (result == WDLResult.Win)
    {
      // TODO: make a robust helper method like the one below and use it instead,
      //       possibly also deal with not arbitrarily choosing parent when ascending tree below
      //      Span<Position> historyPositions = node.Context.Tree.HistoryPositionsForNode(node);

      Debug.Assert(pos.ToMGPosition.IsLegalMove(immediateMove));
      List<Position> historyPositions = [.. Engine.Graph.Store.NodesStore.PositionHistory.Positions];

      // Build position history by ascending to root via tree-parent edges.
      // The tree-parent invariant guarantees a cycle-free path to the root.
      GNode thisNode = node;
      while (!thisNode.IsSearchRoot)
      {
        historyPositions.Add(thisNode.CalcPosition().ToPosition);
        thisNode = thisNode.Graph[thisNode.TreeParentNodeIndex];
      }

      // Try to avoid making a move which would allow opponent to claim draw.
      bool wouldBeDrawByRepetition = PositionRepetitionCalc.DrawByRepetitionWouldBeClaimable(in pos, immediateMove, historyPositions.ToArray());
      if (wouldBeDrawByRepetition)
      {
        if (fullWinningMoveList == null)
        {
          // Perhaps DTZ tablebase files not available.
          // Do not blindly play into the draw, return no tablbase hit
          // and allow engine to search normally as fallback.
          return (WDLResult.Unknown, default);
        }
        else
        {
          // Check other moves to see if any of them avoids falling into the draw by repetition trap.
          foreach ((MGMove, short) move in fullWinningMoveList)
          {
            if (!PositionRepetitionCalc.DrawByRepetitionWouldBeClaimable(in pos, move.Item1, [.. historyPositions]))
            {
              immediateMove = move.Item1;
              break;
            }
          }
        }
      }

    }

    return (result, immediateMove);
  }


  /// <summary>
  /// Returns if an immediate best move is available from the tablebase.
  /// </summary>
  /// <param name="node"></param>
  /// <returns></returns>
  bool ImmediateTablebaseBestMoveIsAvailable(GNode node)
  {
    if (CheckTablebaseBestNextMove != null)
    {
      (WDLResult result, MGMove immediateMove) = TryGetTablebaseImmediateMove(node);
      return result == WDLResult.Win
          || result == WDLResult.Draw
          || result == WDLResult.Loss && immediateMove != default;
    }
    else
    {
      return false;
    }
  }


  /// <summary>
  /// Consults the tablebase to determine if immediate
  /// best move can be directly determined.
  /// </summary>
  void TrySetImmediateBestMove(GNode node)
  {
    TablebaseImmediateBestMove = default;

    // If using tablebases, lookup to see if we have immediate win (choosing the shortest one)
    if (CheckTablebaseBestNextMove != null)
    {
      (WDLResult result, MGMove immediateMove) = TryGetTablebaseImmediateMove(node);
      TablebaseImmediateBestMove = immediateMove;

      // When the search root was previously explored (tree reuse with N > 1),
      // skip modifying the graph node. Setting N=1 and Terminal on a node with
      // parent edges from prior search would violate edge.N <= child.N invariants.
      // The tablebase move is still returned via TablebaseImmediateBestMove.
      // Note: RunLoop(1) runs before this, so a fresh root has N=1 after eval.
      GNode root = Engine.SearchRootNode;
      if (root.NodeRef.N > 1)
      {
        return;
      }

      if (result == WDLResult.Win)
      {
        SetRootAsWin();
      }
      else if (result == WDLResult.Draw)
      {
        // Set the evaluation of the position to be a draw
        // TODO: possibly use distance to end of game to set the distance more accurately than fixed at 1
        // TODO: do we have to adjust for possible contempt?
        const int DISTANCE_TO_END_OF_GAME = 1;

        root.NodeRef.Q = 0;
        root.NodeRef.SiblingsQFrac = 0;
        root.NodeRef.D = 1;
        root.NodeRef.N = 1;
        root.NodeRef.WinP = 0;
        root.NodeRef.LossP = 0;
        root.NodeRef.UncertaintyValue = 0;
        root.NodeRef.UncertaintyPolicy = 0;
        root.NodeRef.MRaw = DISTANCE_TO_END_OF_GAME;
        root.NodeRef.Terminal = GameResult.Draw;
        //        root.NodeRef.EvalResult = new LeafEvaluationResult(GameResult.Draw, 0, 0, 1, 0, 0, 0);
      }
      else if (result == WDLResult.Loss && TablebaseImmediateBestMove != default)
      {
        // Set the evaluation of the position to be a loss.
        // TODO: possibly use distance to mate to set the distance more accurately than fixed at 1
        const int DISTANCE_TO_MATE = 1;

        float lossP = ParamsSelect.LossPForProvenLoss(DISTANCE_TO_MATE, false);

        root.NodeRef.Q = -lossP;
        root.NodeRef.SiblingsQFrac = 0;
        root.NodeRef.D = 0;
        root.NodeRef.N = 1;
        root.NodeRef.WinP = 0;
        root.NodeRef.LossP = (FP16)lossP;
        root.NodeRef.MRaw = DISTANCE_TO_MATE;
        root.NodeRef.UncertaintyValue = 0;
        root.NodeRef.UncertaintyPolicy = 0;
        //        root.NodeRef.EvalResult = new LeafEvaluationResult(GameResult.Checkmate, 0, (FP16)lossP, DISTANCE_TO_MATE, -1, 0, 0);
        root.NodeRef.Terminal = GameResult.Checkmate;
      }
    }
  }


  private void SetRootAsWin()
  {
    // Set the evaluation of the position to be a win
    // TODO: possibly use distance to mate to set the distance more accurately than fixed at 1
    const int DISTANCE_TO_MATE = 1;

    float winP = ParamsSelect.WinPForProvenWin(DISTANCE_TO_MATE, false);

    GNode root = Engine.SearchRootNode;

    root.NodeRef.Q = winP;
    root.NodeRef.SiblingsQFrac = 0;
    root.NodeRef.D = 0;
    root.NodeRef.N = 1;
    root.NodeRef.WinP = (FP16)winP;
    root.NodeRef.LossP = 0;
    root.NodeRef.UncertaintyValue = 0;
    root.NodeRef.UncertaintyPolicy = 0;
    root.NodeRef.MRaw = DISTANCE_TO_MATE;
    //Context.Root.EvalResult = new LeafEvaluationResult(GameResult.Checkmate, (FP16)winP, 0, DISTANCE_TO_MATE, -1, 0, 0);
    root.NodeRef.Terminal = GameResult.Checkmate;
  }


  readonly static object dumpToConsoleLock = new();


  /// <summary>
  /// Returns the move from the given position (with history).
  /// </summary>
  /// <param name="nnEvaluator"></param>
  /// <param name="priorMoves"></param>
  /// <param name="moves"></param>
  /// <param name="evalResults"></param>
  /// <param name="fillInHistory"></param>
  /// <param name="dumpInfo"></param>
  /// <returns></returns>
  public static (MGMove, float, float) BestValueMove(NNEvaluator nnEvaluator,
                                                     PositionWithHistory priorMoves,
                                                     ref MGMoveList moves,
                                                     ref NNEvaluatorResult[] evalResults,
                                                     bool fillInHistory,
                                                     bool dumpInfo,
                                                     byte[] searchRootPlySinceLastMove = null)
  {
    // Compute move list if not already provided.
    if (moves == null)
    {
      moves = new MGMoveList();
      MGMoveGen.GenerateMoves(priorMoves.FinalPosMG, moves);
    }

    // Prepare a batch builder in which to enqueue the positions to be evaluated.
    EncodedPositionBatchBuilder batchBuilder = new(128, nnEvaluator.InputsRequired | NNEvaluator.InputTypes.Positions);

    // Prepare array of prior positions initialized from prior positions,
    // with extra last slot to be for new position after each move to be evaluated.
    Position[] positions = new Position[priorMoves.Count + 1];
    Array.Copy(priorMoves.Positions.ToArray(), positions, priorMoves.Positions.Length);

    // If the evaluator needs ply-since-last-move, prepare a buffer for child ply values.
    bool needsLastMovePlies = nnEvaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.LastMovePlies);
    byte[] childPlies = needsLastMovePlies ? new byte[64] : null;

    Span<short> repetitionCounts = stackalloc short[moves.NumMovesUsed];

    // Loop over all moves, find new resulting position, and add to batch.
    int indexOfDrawByRepetition = -1;
    for (int i = 0; i < moves.NumMovesUsed; i++)
    {
      MGPosition pos = priorMoves.FinalPosition.ToMGPosition;
      pos.MakeMove(moves.MovesArray[i]);
      positions[^1] = pos.ToPosition;

      // Need to calc repetition count for position of this move in the context of all prior moves
      int finalRepetitionCount = PositionRepetitionCalc.SetFinalPositionRepetitionCount(positions);
      repetitionCounts[i] = (short)finalRepetitionCount;
      if (finalRepetitionCount >= 2)
      {
        indexOfDrawByRepetition = i;
      }

      EncodedPositionWithHistory eph = new();
      eph.SetFromSequentialPositions(positions, fillInHistory);

      // Compute ply-since-last-move for this child position from cached root values.
      if (needsLastMovePlies)
      {
        PlySinceLastMoveArray.ApplyMove(searchRootPlySinceLastMove, childPlies, moves.MovesArray[i]);
      }

      batchBuilder.Add(in eph, false, lastMovePlies: childPlies);
    }

    // Build the batch and evaluate it.
    EncodedPositionBatchFlat thisBatch = batchBuilder.GetBatch();
    evalResults = nnEvaluator.EvaluateBatch(thisBatch);

    float bestVRaw = evalResults.Min(v => -v.V);

    NNEvaluatorResult parentResult = dumpInfo ? nnEvaluator.Evaluate(priorMoves.FinalPosition) : default;

    // Determine which move had position yielding best value evaluation.
    int moveIndex = 0;
    int bestMoveIndex = 0;
    float bestV = float.MinValue;
    float dOfBestV = 0;
    foreach (NNEvaluatorResult evalResult in evalResults)
    {
      float vOurPerspective = -evalResult.V;

      if (nnEvaluator.UseBestValueMoveUseRepetitionHeuristic)
      {
        if (repetitionCounts[moveIndex] == 1) // on the way to draw by repetition
        {
          if (bestVRaw >= 0) // probably winning
          {
            // Disfavor positions where repetition count is nonzero.
            vOurPerspective -= 0.01f;
          }
          else if (bestVRaw <= 0) // probably losing
          {
            // Favor positions where repetition count is nonzero.
            vOurPerspective += 0.01f;
          }
        }
      }

      if (vOurPerspective > bestV)
      {
        bestV = vOurPerspective;
        dOfBestV = evalResult.D;
        bestMoveIndex = moveIndex;
      }
      moveIndex++;
    }

    if (nnEvaluator.UseBestValueMoveUseRepetitionHeuristic && indexOfDrawByRepetition != -1 && bestV < 0)
    {
      // If we appear worse, choose the draw by repetition at hand.
      bestV = 0;
      bestMoveIndex = indexOfDrawByRepetition;
    }

    if (dumpInfo)
    {
      Console.WriteLine($"BestValueMove detail using {nnEvaluator} from {priorMoves}");
      for (int i = 0; i < moves.Count(); i++)
      {
        EncodedMove encodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(moves.MovesArray[i]);
        float policyPct = 0;
        if (parentResult.Policy.IndexOfMove(encodedMove) != -1)
        {
          policyPct = 100 * parentResult.Policy.PolicyInfoAtIndex(parentResult.Policy.IndexOfMove(encodedMove)).Probability;
        }
        bool isBest = i == bestMoveIndex;
        string warnStr = isBest && policyPct < 10 ? "?" : " ";
        Console.Write(isBest ? "** " : "  ");
        Console.WriteLine(warnStr + "  " + moves.MovesArray[i] + " " + -evalResults[i].V
                    + "   " + policyPct + "%");
      }
    }

    // Return the best move.
    MGMove bestMove = moves.MovesArray[bestMoveIndex];
    return (bestMove, bestV, dOfBestV);
  }



  /// <summary>
  /// Returns the best move found in the most recent search,
  /// along with associated information.
  /// </summary>
  /// <param name="bestChildEdge"></param>
  /// <param name="bestMoveNode"></param>
  /// <param name="bestMove"></param>
  /// <returns></returns>
  /// <exception cref="Exception"></exception>
  public BestMoveInfoMCGS GetBestMove(out GEdge bestChildEdge,
                                      out GNode bestMoveNode,
                                      out MGMove bestMove,
                                      bool isFinalBestMoveCalc)
  {
    Graph graph = Engine.Graph;
    GNode rootX = Engine.SearchRootNode;

    if (rootX.NodeRef.N == 0)
    {
      throw new Exception("Cannot get best move, no search performed yet.");
    }

    ManagerChooseBestMoveMCGS chooseMCGS = new(this, rootX, true, default, isFinalBestMoveCalc);
    BestMoveInfoMCGS bm = chooseMCGS.BestMoveCalc;
    bestChildEdge = bm.BestMoveEdge;
    bestMoveNode = bm.BestMoveEdge.ChildNode;
    bestMove = bm.BestMove;
    return bm;
  }
}
