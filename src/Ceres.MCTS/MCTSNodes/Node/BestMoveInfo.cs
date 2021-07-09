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
    /// The number of visits beneath the best node.
    /// </summary>
    public readonly int N;

    /// <summary>
    /// The Q (average evaluation) corresponding to the chosen best move.
    /// </summary>
    public readonly float Q;

    /// <summary>
    /// The largest Q among all moves at the root.
    /// </summary>
    public readonly float BestQ;

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
    /// <param name="node"></param>
    /// <param name="bestQ"></param>
    /// <param name="bestN"></param>
    /// <param name="bestNSecond"></param>
    /// <param name="mlhBonusApplied"></param>
    public BestMoveInfo(MCTSNode node, float bestQ, float bestN, float bestNSecond, float mlhBonusApplied)
    {
      BestMoveNode = node;
      N = node.N;
      Q = (float)node.Q;
      BestQ = bestQ;
      BestN = bestN;
      BestNSecond = bestNSecond;
      MLHBonusApplied = mlhBonusApplied;
      BestMoveNode.Annotate();
      BestMove = node.Annotation.PriorMoveMG;
    }

    /// <summary>
    /// Constructor for case of immediate move determined without search.
    /// </summary>
    /// <param name="reason"></param>
    /// <param name="bestMove"></param>
    /// <param name="bestQ"></param>
    public BestMoveInfo(BestMoveReason reason, MGMove bestMove, float bestQ)
    {
      Debug.Assert(reason != BestMoveReason.SearchResult);

      Reason = reason;
      Q = bestQ;
      BestMove = bestMove;
    }


    /// <summary>
    /// Returns string summary of information.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string bestNStr = BestN == N ? "(same)" : $"{BestN:N0}";
      string bestQStr = BestQ == Q ? "(same)" : $"{BestQ:F2}";
      string mlhStr = MLHBonusApplied == 0 ? "" : $" MLHBonus={MLHBonusApplied}";
      return $"<BestMoveInfo {BestMove.MoveStr(MGMoveNotationStyle.LC0Coordinate)} N={N} Q={Q} BestN={bestNStr} BestQ={bestQStr} BestN2={BestNSecond,5:F1} {mlhStr}>";
    }
  }
}

