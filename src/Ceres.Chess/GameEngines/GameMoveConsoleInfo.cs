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
using System.Security.Permissions;
using System.Text;

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Captures information relating to a game move.
  /// </summary>
  public class GameMoveConsoleInfo
  {
    public string ResultStr;
    public int MoveNum;

    public SearchLimit SearchLimit;
    public float WhiteNodesAllMoves;
    public float WhiteTimeAllMoves;
    public string WhiteMoveStr;
    public string WhiteCheckMoveStr;
    public SearchLimit WhiteSearchLimitPre;
    public SearchLimit WhiteSearchLimitPost;
    public float WhiteMoveTimeUsed;
    public float WhiteScoreCentipawns;
    public float WhiteScoreQ;
    public int WhiteStartN;
    public int WhiteFinalN;
    public float WhiteMAvg;
    public bool WhiteShouldHaveForfeitedOnLimit;

    public float BlackNodesAllMoves;
    public float BlackTimeAllMoves;
    public string BlackCheckMoveStr;
    public string BlackMoveStr;
    public SearchLimit BlackSearchLimitPre;
    public SearchLimit BlackSearchLimitPost;
    public float BlackMoveTimeUsed;
    public float BlackScoreCentipawns;
    public float BlackScoreQ;
    public int BlackStartN;
    public int BlackFinalN;
    public float BlackMAvg;
    public bool BlackShouldHaveForfeitedOnLimit;

    public int WhiteNumNodesComputed => WhiteFinalN - WhiteStartN;
    public int BlackNumNodesComputed => BlackFinalN - BlackStartN;

    public string FEN;

    public void PutStr()
    {
      // Only show the move from the check engine if it differed from the selected move
      if (WhiteCheckMoveStr != null && WhiteMoveStr == WhiteCheckMoveStr)
      {
        WhiteCheckMoveStr = "";
      }

      if (BlackCheckMoveStr != null && BlackMoveStr == BlackCheckMoveStr)
      {
        BlackCheckMoveStr = "";
      }

      Console.SetCursorPosition(0, Console.CursorTop);

      ConsoleColor colorDefault = Console.ForegroundColor;

      //      Console.Write($"{ ResultStr,10}  );
      Console.Write($"{MoveNum,5}.  ");
      //      Console.Write($"{WhiteNodesAllMoves,12:N0} {BlackNodesAllMoves,12:N0} ");
      //      Console.Write($"{WhiteTimeAllMoves,6:F2} {BlackTimeAllMoves,6:F2} ");

      //Console.Write($"{WhiteSearchLimitPre,15} --> {WhiteSearchLimitPost,15} ");
      Console.Write($"{WhiteSearchLimitPre,15} ");
      Console.Write($"{WhiteMoveStr,6}  {WhiteCheckMoveStr,6} {WhiteMoveTimeUsed,5:F2} ");
      Console.Write($"{(float)WhiteFinalN,12:N0} ");
      if (WhiteScoreCentipawns > 20) Console.ForegroundColor = ConsoleColor.Green;
      if (WhiteScoreCentipawns < -20) Console.ForegroundColor = ConsoleColor.Red;
      Console.Write($"{WhiteScoreCentipawns,5:F0}   ");
      Console.ForegroundColor = colorDefault;

      //Console.Write($"{BlackSearchLimitPre,15} --> {BlackSearchLimitPost,15} ");
      Console.Write($"{BlackSearchLimitPre,15} ");
      Console.Write($"{BlackMoveStr,7}  {BlackCheckMoveStr,6} {BlackMoveTimeUsed,5:F2} ");
      Console.Write($"{(float)BlackFinalN,12:N0} ");
      if (BlackScoreCentipawns > 20) Console.ForegroundColor = ConsoleColor.Green;
      if (BlackScoreCentipawns < -20) Console.ForegroundColor = ConsoleColor.Red;
      Console.Write($"{BlackScoreCentipawns,5:F0}   ");
      Console.ForegroundColor = colorDefault;
      Console.Write($"  {FEN,-50} ");
    }


  }
}
