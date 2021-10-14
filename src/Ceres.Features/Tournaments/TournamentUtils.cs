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
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators.LC0DLL;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Set of miscellaneous static methods related to tournaments.
  /// </summary>
  internal static class TournamentUtils
  {
    /// <summary>
    /// Translates a game result into a string representation.
    /// </summary>
    /// <param name="result"></param>
    /// <param name="isWhite"></param>
    /// <returns></returns>
    internal static string ResultStr(TournamentGameResult result, bool isWhite)
    {
      return result switch
      {
        TournamentGameResult.None => "",
        TournamentGameResult.Win => "+" + (isWhite ? "w" : "b"),
        TournamentGameResult.Loss => "-" + (isWhite ? "w" : "b"),
        TournamentGameResult.Draw => "="
      };
    }



    static Lazy<ISyzygyEvaluatorEngine> tbEvaluator = new Lazy<ISyzygyEvaluatorEngine>(() =>
    SyzygyEvaluatorPool.GetSessionForPaths(CeresUserSettingsManager.Settings.TablebaseDirectory));

    public static TournamentGameResult TryGetGameResultIfTerminal(PositionWithHistory game, 
                                                                    bool playerIsWhite, bool useTablebasesForAdjudication,
                                                                    out TournamentGameResultReason reason)
    {
      reason = default;
      Position pos = game.FinalPosition;
      bool whiteToMove = pos.MiscInfo.SideToMove == SideType.White;
      bool weArePlayerToMove = whiteToMove == playerIsWhite;

      if (useTablebasesForAdjudication)
      {
        // Probe endgame tablebase
        tbEvaluator.Value.ProbeWDL(in pos, out LC0DLLSyzygyEvaluator.WDLScore score,
                                           out LC0DLLSyzygyEvaluator.ProbeState result);

        if (result == LC0DLLSyzygyEvaluator.ProbeState.Ok)
        {
          if (score == LC0DLLSyzygyEvaluator.WDLScore.WDLWin)
          {
            reason = TournamentGameResultReason.AdjudicateTB;
            return weArePlayerToMove ? TournamentGameResult.Win : TournamentGameResult.Loss;
          }
          else if (score == LC0DLLSyzygyEvaluator.WDLScore.WDLLoss)
          {
            reason = TournamentGameResultReason.AdjudicateTB;
            return weArePlayerToMove ? TournamentGameResult.Loss : TournamentGameResult.Win;
          }
          else if (score == LC0DLLSyzygyEvaluator.WDLScore.WDLDraw
                || score == LC0DLLSyzygyEvaluator.WDLScore.WDLCursedWin
                || score == LC0DLLSyzygyEvaluator.WDLScore.WDLBlessedLoss)
          {
            reason = TournamentGameResultReason.AdjudicateTB;
            return TournamentGameResult.Draw;
          }
        }
      }

      GameResult terminalStatus = pos.CalcTerminalStatus();
      if (terminalStatus == GameResult.Unknown)
      {
        return TournamentGameResult.None;
      }
      else if (terminalStatus == GameResult.Draw)
      {
        reason = TournamentGameResultReason.Stalemate;
        return TournamentGameResult.Draw;
      }
      else if (terminalStatus == GameResult.Checkmate)
      {
        reason = TournamentGameResultReason.Checkmate;
        return weArePlayerToMove ? TournamentGameResult.Loss : TournamentGameResult.Win;
      }

      throw new NotImplementedException();

    }

  }
}
