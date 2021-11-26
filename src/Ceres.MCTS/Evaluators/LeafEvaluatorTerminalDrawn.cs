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

using Ceres.Chess;
using Ceres.MCTS.MTCSNodes;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Checks for draw terminal states currently on the board.
  /// Note that these checks are inexpensive (not requiring move generation).
  /// 
  /// </summary>
  public sealed class LeafEvaluatorTerminalDrawn : LeafEvaluatorBase
  {
    protected override LeafEvaluationResult DoTryEvaluate(MCTSNode node)
    {
      // Check  possibility of draw by insufficient material.
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
