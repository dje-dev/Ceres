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
using Ceres.Chess.Positions;
using Ceres.Chess.Textual.PgnFileTools;

#endregion

namespace Ceres.Chess.Games.Utils
{
  // TODO: delete this,we have Game class instead

  /// <summary>
  /// Represents a game loaded from a PGN file.
  ///  
  /// NOTE: There are known imperfections with malformed PGNs.
  ///       For example if there is a PGN with mostly blank headers and no moves then
  ///       the parser will fail on the subsequent game.
  /// </summary>
  public record PGNGame
  {
    public enum GameResult { WhiteWins, Draw, BlackWins, Unknown };

    /// <summary>
    /// Year in which game was played.
    /// </summary>
    public readonly int Year;

    /// <summary>
    /// Name of player with white pieces.
    /// </summary>
    public readonly string WhitePlayer;

    /// <summary>
    /// Name of player with black pieces.
    /// </summary>
    public readonly string BlackPlayer;

    /// <summary>
    /// Result of game.
    /// </summary>
    public readonly GameResult Result;

    /// <summary>
    /// List of game moves.
    /// </summary>
    public readonly PositionWithHistory Moves;

    /// <summary>
    /// The staring position (as a FEN) for the game.
    /// </summary>
    public readonly string StartFEN;


    public float MoveTimeTotalWhite => MoveTimeTotal(true);

    public float MoveTimeTotalBlack => MoveTimeTotal(false);

    float MoveTimeTotal(bool oddMoves)
    {
      throw new NotImplementedException();
#if NOT
      double total = 0;
      for (int i = 0; i < Moves.Count; i++)
        if (i % 2 == 1 == oddMoves)
          total += Moves[i];
//      total += new PGNGameMove(Moves[i]).MoveTimeSeconds;
      return total;
#endif
    }


   
    /// <summary>
    /// Constructor from a GameInfo object.
    /// </summary>
    /// <param name="gi"></param>
    public PGNGame(GameInfo gi)
    {
      string dateStr = gi.Headers.ContainsKey("Date") ? gi.Headers["Date"] : null;
      if (dateStr != null) int.TryParse(dateStr.Substring(0, 4), out Year);

      WhitePlayer = gi.Headers.ContainsKey("White") ? gi.Headers["White"] : "?";
      BlackPlayer = gi.Headers.ContainsKey("Black") ? gi.Headers["Black"] : "?";
      StartFEN = gi.Headers.ContainsKey("FEN") ? gi.Headers["FEN"] : null;

      if (gi.Headers.ContainsKey("Result"))
      {
        string result = gi.Headers["Result"];
        if (result.Contains("1/2-1/2"))
          Result = GameResult.Draw;
        else if (result.Contains("1-0"))
          Result = GameResult.WhiteWins;
        else if (result.Contains("0-1"))
          Result = GameResult.BlackWins;
      }
      else
        Result = GameResult.Unknown;

      StartFEN = StartFEN ?? Position.StartPosition.FEN;
      Moves = new PositionWithHistory(Position.FromFEN(StartFEN));

      foreach (Textual.PgnFileTools.Move move in gi.Moves)
      {
        Move m1 = Move.FromSAN(Moves.FinalPosition, move.ToAlgebraicString());
        Moves.AppendMove(m1.ToString());
      }
    }
  }
}
