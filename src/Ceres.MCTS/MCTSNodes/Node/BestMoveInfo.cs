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

using System.Diagnostics;
using System.Runtime.CompilerServices;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

[assembly: InternalsVisibleTo("Ceres.EngineMCTS.Test")] // TODO: move or remove me.

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Set of information realting to the selection of the 
  /// best top-level move to be chosen at the end of a search.
  /// </summary>
  public record BestMoveInfo
  {
    public enum BestMoveReason
    {
      /// <summary>
      /// There are no legal moves (terminal).
      /// </summary>
      NoLegalMoves,

      /// <summary>
      /// There is only one legal move.
      /// </summary>
      OneLegalMove,

      /// <summary>
      /// Search was size zero, use best policy move.
      /// </summary>
      ImmediateNoSearchPolicyMove,

      /// <summary>
      /// The best immediate move is given by the tablebase.
      /// </summary>
      TablebaseImmediateMove,

      /// <summary>
      /// The search determined the best move.
      /// </summary>
      SearchResult
     };


    /// <summary>
    /// The method used to determine the best move.
    /// </summary>
    public readonly BestMoveReason Reason;

    /// <summary>
    /// The node corresponding to the move that was chosen to be best.
    /// This can possibly be null (e.g in case of tablebase immediate move or forced move).
    /// </summary>
    public MCTSNode BestMoveNode { get; init; }

    /// <summary>
    /// Node with best Q score.
    /// </summary>
    public MCTSNode BestQNode { get; init; }

    /// <summary>
    /// Node with best N score.
    /// </summary>
    public MCTSNode BestNNode { get; init; }

    /// <summary>
    /// The number of visits beneath the best node.
    /// </summary>
    public readonly int N;

    /// <summary>
    /// The Q for the best move (from the perspective of the player to move).
    /// </summary>
    public readonly float QOfBest;

    /// <summary>
    /// The largest Q among all moves at the root (from the perspective of the player to move).
    public readonly float QMaximal;

    /// <summary>
    /// The largest N among all moves at the root.
    /// </summary>
    public readonly float BestN;

    /// <summary>
    /// The N of the move having second largest N (or same as BestN if none).
    /// </summary>
    public float BestNSecond;

    /// <summary>
    /// The optional moves left head bonus applied in selecting this move.
    /// </summary>
    public readonly float MLHBonusApplied;

    /// <summary>
    /// Best move in this position.
    /// </summary>
    public MGMove BestMove;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="bestMoveNode"></param>
    /// <param name="qMaximal"></param>
    /// <param name="bestN"></param>
    /// <param name="bestNSecond"></param>
    /// <param name="mlhBonusApplied"></param>
    /// <param name="bestNNode"></param>
    /// <param name="bestQNode"></param>
    public BestMoveInfo(MCTSNode bestMoveNode, float qMaximal, float bestN, float bestNSecond, float mlhBonusApplied,
                        MCTSNode bestNNode = default, MCTSNode bestQNode = default)
    {
      BestMoveNode = bestMoveNode;
      N = bestMoveNode.N;
      QOfBest = (float)-bestMoveNode.Q;
      QMaximal = qMaximal;
      BestN = bestN;
      BestNSecond = bestNSecond;
      MLHBonusApplied = mlhBonusApplied;
      BestMoveNode.Annotate();
      BestMove = bestMoveNode.Annotation.PriorMoveMG;
      BestNNode = bestNNode;
      BestQNode = bestQNode;
    }


    /// <summary>
    /// Constructor for case of immediate move based on policy (no search).
    /// </summary>
    /// <param name="reason"></param>
    /// <param name="bestMove"></param>
    /// <param name="qOfBest"></param>
    public BestMoveInfo(MCTSNode parentNode, BestMoveReason reason, float q)
    {
      Debug.Assert(reason == BestMoveReason.ImmediateNoSearchPolicyMove);

      Reason = BestMoveReason.ImmediateNoSearchPolicyMove;
      QOfBest = QMaximal = q;
      BestMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(parentNode.ChildAtIndexInfo(0).move, parentNode.Annotation.PosMG);
    }



    /// <summary>
    /// Constructor for case of immediate move determined without search.
    /// </summary>
    /// <param name="reason"></param>
    /// <param name="bestMove"></param>
    /// <param name="qOfBest"></param>
    public BestMoveInfo(BestMoveReason reason, MGMove bestMove, float qOfBest)
    {
      Debug.Assert(reason != BestMoveReason.SearchResult);

      Reason = reason;
      QOfBest = QMaximal = qOfBest;
      BestMove = bestMove;
    }


    /// <summary>
    /// The magnitude of degree to which the Q associated with the chosen move
    /// is worse (if any) than the node with the best Q.
    /// </summary>
    public float BestMoveQSuboptimality => QMaximal - QOfBest;


    /// <summary>
    /// Returns if the best move chosen was the move having largest N.
    /// </summary>
    public bool BestMoveWasTopN => N == BestN;


    /// <summary>
    /// Returns string summary of information.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string bestNStr = BestN == N ? "(same)" : $"{BestN:N0}";
      string bestQStr = QMaximal == QOfBest ? "(same)" : $"{QMaximal:F2}";
      string mlhStr = MLHBonusApplied == 0 ? "" : $" MLHBonus={MLHBonusApplied}";
      return $"<BestMoveInfo {BestMove.MoveStr(MGMoveNotationStyle.LC0Coordinate)} N={N} Q={QOfBest} BestN={bestNStr} BestQ={bestQStr} BestN2={BestNSecond,5:F1} {mlhStr}>";
    }
  }
}

