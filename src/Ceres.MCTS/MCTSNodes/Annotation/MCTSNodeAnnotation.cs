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
using System.Runtime.CompilerServices;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Boards;
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.MCTS.MTCSNodes.Annotation
{
  /// <summary>
  /// Structure that is embedded inside an MCTSNode object
  /// encapsulating supplemental data relating to the node
  /// that is not permanently stored in the main node store 
  /// (due to memory requirements).
  /// 
  /// These data items are mostly computed at MCTSNode initialization
  /// and then resused for the lifetime of the MCTSNode to improve performance.
  /// </summary>
  public struct MCTSNodeAnnotation : IEquatable<MCTSNodeAnnotation>
  {
    /// <summary>
    /// If the annotation has been fully intialized.
    /// </summary>
    public bool IsInitialized => Pos.PieceCount > 0;

    /// <summary>
    /// The move made at the parent which resulted in this position.
    /// </summary>
    public MGMove PriorMoveMG;

    /// <summary>
    /// Current position (as Position)
    /// </summary>
    public Position Pos;

    /// <summary>
    /// Current position (as MGPosition)
    /// </summary>
    public MGPosition PosMG;

    #region Moves

    /// <summary>
    /// The set of legal moves from this position.
    /// </summary>
    public MGMoveList Moves
    {
      get
      {
        if (moves == null || moves.NumMovesUsed == -1)
        {
          // Get MGMoveList based on local temporary buffer.
          MGMoveList localMovesBuffer = movesBuffer;
          if (localMovesBuffer == null)
          {
            localMovesBuffer = movesBuffer = new MGMoveList(128);
          }
          else
          {
            localMovesBuffer.Clear();
          }

          MGMoveGen.GenerateMoves(in PosMG, localMovesBuffer);
          if (moves != null)
          {
            moves.Copy(localMovesBuffer);
          }
          else
          {
            moves = new MGMoveList(localMovesBuffer);
          }
        }

        return moves;
      }
    }

    /// <summary>
    /// Move list
    /// </summary>
    internal MGMoveList moves;

    /// <summary>
    /// For efficiency moves are generated in to a local thread static buffer
    /// and then copied into an exact sized final array of moves in retained MGMoveList
    /// </summary>
    [ThreadStatic] static MGMoveList movesBuffer;

    #endregion


    /// <summary>
    /// Initializes a specified EncodedPosition to reflect the a specified node's position.
    /// </summary>
    /// <param name="node"></param>
    /// <param name="boardsHistory"></param>
    public unsafe void CalcRawPosition(MCTSNode node, ref EncodedPositionWithHistory boardsHistory)
    {
      Span<EncodedPositionBoard> destBoards = new Span<EncodedPositionBoard>(Unsafe.AsPointer(ref boardsHistory), EncodedPositionBoards.NUM_MOVES_HISTORY);

      // Now we fill in the history boards, extracted from up to 4 distinct places:
      //   1. the current node's board (always)
      //   2. possibly sequence of boards by ascending tree toward root
      //   3. possibly sequence of boards coming from the prior history that
      //      this search was launched with (nodes before the root)
      //   4. possibly one or more fill-in boards, which can be
      //      either zero (if history fill-in feature turned off) 
      //      otherwise the last actual board in the history repeated
      destBoards[0] = LC0BoardPosition;

      MCTSNode priorNode = node.Parent;

      // Ascend in tree copying positions from ancestors
      int nextBoardIndex = 1;
      while (nextBoardIndex < EncodedPositionBoards.NUM_MOVES_HISTORY && !priorNode.IsNull)
      {
        if (nextBoardIndex % 2 == 1)
        {
          destBoards[nextBoardIndex++] = priorNode.Annotation.LC0BoardPosition.ReversedAndFlipped;
        }
        else
        {
          destBoards[nextBoardIndex++] = priorNode.Annotation.LC0BoardPosition;
        }

        priorNode = priorNode.Parent;
      }

      // Boards from prior history
      int priorPositionsIndex = 0;
      while (nextBoardIndex < EncodedPositionBoards.NUM_MOVES_HISTORY && priorPositionsIndex < node.Tree.EncodedPriorPositions.Count)
      {
        if (nextBoardIndex % 2 == 1)
        {
          destBoards[nextBoardIndex++] = node.Tree.EncodedPriorPositions[priorPositionsIndex].ReversedAndFlipped;
        }
        else
        {
          destBoards[nextBoardIndex++] = node.Tree.EncodedPriorPositions[priorPositionsIndex];
        }

        priorPositionsIndex++;
      }

      // Finally, set last boards either with repeated last position (if fill in) or zeros
      int indexBoardToRepeat = nextBoardIndex - 1;
      bool historyFillIn = node.Context.ParamsSearch.HistoryFillIn;
      while (nextBoardIndex < EncodedPositionBoards.NUM_MOVES_HISTORY)
      {
        if (historyFillIn)
        {
          destBoards[nextBoardIndex++] = destBoards[indexBoardToRepeat];
        }
        else
        {
          destBoards[nextBoardIndex++] = default;
        }
      }

      boardsHistory.SetMiscInfo(new EncodedTrainingPositionMiscInfo(MiscInfo, default));
    }


    #region Board information

    /// <summary>
    /// Board corresponding to this position.
    /// </summary>
    internal EncodedPositionBoard LC0BoardPosition;

    /// <summary>
    /// Misc info associated with this position.
    /// </summary>
    internal EncodedPositionMiscInfo MiscInfo;

    #endregion

    #region Overrides

    /// <summary>
    /// Returns a string description of the annotations.
    /// </summary>
    /// <returns></returns>
    public override string ToString() => $"<MCTSNodeAnnotation>";


    /// <summary>
    /// Returns hash code.
    /// </summary>
    /// <returns></returns>
    public override int GetHashCode() => throw new NotImplementedException();


    /// <summary>
    /// Tests for equality (not supported).
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public bool Equals(MCTSNodeAnnotation other) => throw new NotImplementedException();

    #endregion
  }

}
