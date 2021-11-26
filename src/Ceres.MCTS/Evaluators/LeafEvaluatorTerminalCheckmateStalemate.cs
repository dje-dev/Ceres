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
using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.Environment;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Checks for terminal states coming from checkmate or stalemate
  /// (which require move generation).
  /// 
  /// </summary>
  public sealed class LeafEvaluatorTerminalCheckmateStalemate : LeafEvaluatorBase
  {
    /// <summary>
    /// Applies some tests to try to detect terminal positions, returning the TerminalStatus.
    /// </summary>
    /// <returns></returns>
    static internal GameResult TryCalcTerminalStatus(MCTSNode node)
    {
      if (node.Annotation.Moves.NumMovesUsed > 0)
      {
        return GameResult.Unknown;
      }
      else if (MGMoveGen.IsInCheck(in node.Annotation.PosMG, node.Annotation.PosMG.BlackToMove))
      {
        return GameResult.Checkmate;
      }
      else
      {
        return GameResult.Draw;
      }
    }


    protected override LeafEvaluationResult DoTryEvaluate(MCTSNode node)
    {
      // First check for immediate checkmate/stalemate
      GameResult terminalStatus = TryCalcTerminalStatus(node);

      if (terminalStatus != GameResult.Unknown)
      {
        if (terminalStatus == GameResult.Draw)
        {
          // Position is stalemate (draw)
          return new LeafEvaluationResult(GameResult.Draw, 0, 0, 0);
        }
        else if (terminalStatus == GameResult.Checkmate)
        {
          // Position is checkmate (lost)
          return new LeafEvaluationResult(GameResult.Checkmate, 0, (FP16)ParamsSelect.LossPForProvenLoss(node.Depth, true), node.Depth);
          
        }
        else
        {
          throw new Exception($"Internal error: unexpected terminal status {terminalStatus} encountered for node {node}");
        }
      }
      else
      {
        // Check final possibility of draw by insufficient material
        if (node.Annotation.Pos.CheckDrawBasedOnMaterial == Position.PositionDrawStatus.DrawByInsufficientMaterial)
        {
          // Position is draw by insufficient material
          LeafEvaluationResult evalResult = new LeafEvaluationResult(GameResult.Draw, 0, 0, 0);
          return evalResult;
        }
        else if (node.Annotation.Pos.CheckDrawCanBeClaimed == Position.PositionDrawStatus.DrawCanBeClaimed)
        {
          // We hit a position repeated 3 times
          // Unless we had asymmetric contempt (which is not currently supported)
          // either the opponent (who was about to make this move and could have claimed draw)
          // or we (who are now on move can now claim draw) will
          // consider themselves inferior and will claim the draw
          LeafEvaluationResult evalResult = new LeafEvaluationResult(GameResult.Draw, 0, 0, 0);
          return evalResult;
        }
        else if (node.Context.ParamsSearch.TwofoldDrawEnabled 
               && node.Depth > 2
               && node.Annotation.Pos.MiscInfo.RepetitionCount >= 1)
        {
          // Experimentally consider repetition a draw

          LeafEvaluationResult evalResult = new LeafEvaluationResult(GameResult.Draw, 0, 0, 0);
          return evalResult;

          //node.OverrideVToApplyFromTransposition = node.Context.ContemptManager.BaseContempt;
          //node.OverrideVToApplyFromTransposition = 0;
          return default;
        }
        else
        {
          return default;
        }
      }
    }

  }
}
