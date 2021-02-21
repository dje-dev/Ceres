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
using System.Globalization;
using System.Text;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;
using Ceres.Chess.Textual.PgnFileTools;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// 
  /// </summary>
  public record Game
  {
    public enum GameResult { Draw, WhiteWins, BlackWins, Unterminated }

    /// <summary>
    /// Text description of game.
    /// </summary>
    public string Description { get; init; }

    /// <summary>
    /// Name of player with white pieces.
    /// </summary>
    public string PlayerWhite { get; init; }

    /// <summary>
    /// Name of player with black pieces.
    /// </summary>
    public string PlayerBlack { get; init; }

    /// <summary>
    /// Date the game was played.
    /// </summary>
    public DateTime Date { get; init; }

    /// <summary>
    /// Result of the game.
    /// </summary>
    public GameResult Result { get; init; }


    /// <summary>
    /// Optional starting position for game.
    /// </summary>
    public Position InitialPosition { get; init; }

    /// <summary>
    /// Sequence of moves in the game.
    /// </summary>
    public List<Move> Moves { get; init; }


    /// <summary>
    /// Returns string consisting of sequence of 
    /// all moves made in game (in UCI format).
    /// </summary>
    public string MoveStr
    {
      get
      {
        StringBuilder str = new StringBuilder();
        int i = 0;
        foreach (Move move in Moves)
        {
          if (i > 0) str.Append(" ");
          str.Append(move.ToString());
          i++;
        }
        return str.ToString();
      }
    }


    /// <summary>
    /// Returns a List of Games from a PGN file with specified name.
    /// </summary>
    /// <param name="pgnFileName"></param>
    /// <returns></returns>
    public static IEnumerable<Game> FromPGN(string pgnFileName)
    {
      if (!System.IO.File.Exists(pgnFileName)) 
        throw new ArgumentException($"Requested pgn file does not exist {pgnFileName}");

      PgnStreamReader pgnReader = new PgnStreamReader();
      foreach (GameInfo game in pgnReader.Read(pgnFileName))
      {
        Game gameS = new Game(game);

        // Skip null games
        if (gameS.Moves.Count > 0)
        {
          yield return gameS;
        }
      }
    }


    /// <summary>
    /// Enumerates all positions in game as a sequence of PositionsWithHistory
    /// (each capturing the fully history of the game up to a point).
    /// </summary>
    public IEnumerable<PositionWithHistory> PositionsWithHistory
    {
      get
      {
        PositionWithHistory ret = new PositionWithHistory(InitialPosition);
        ret.NextMove = Moves.Count == 0 ? default : Moves[0];
        yield return ret;

        // TODO: improve efficiency
        PositionWithHistory running = new PositionWithHistory(InitialPosition);
        int moveIndex = 0;
        foreach (Move move in Moves)
        {
          running.AppendMove(MGMoveConverter.MGMoveFromPosAndMove(running.FinalPosition, move));
          ret = (PositionWithHistory)running.Clone();
          ret.NextMove = moveIndex < Moves.Count - 1 ? Moves[moveIndex + 1] : default;
          yield return ret;
          moveIndex++;
        }
      }
    }

    /// <summary>
    /// Private constructor from a GameInfo.
    /// 
    /// TODO: Capture more or all fields from header and moves.
    /// </summary>
    /// <param name="pgnGame"></param>
    private Game(GameInfo pgnGame)
    {
      Description = pgnGame.Comment;

      pgnGame.Headers.TryGetValue("FEN", out string startFEN);
      InitialPosition = startFEN == null ? Position.StartPosition : Position.FromFEN(startFEN);

      string dateStr = pgnGame.Headers.ContainsKey("Date") ? pgnGame.Headers["Date"] : null;
      if (dateStr != null)
      {
        DateTime date;
        if (DateTime.TryParseExact(dateStr, "yyyy.MM.dd", CultureInfo.InvariantCulture, DateTimeStyles.None, out date))
        {
          Date = date;
        }
        else
        {
          if (int.TryParse(dateStr.Substring(0, 4), out int year))
            Date = new DateTime(year, 7, 1); // Arbitrarily choose middle of year if only year given
        }
      }

      PlayerWhite = pgnGame.Headers.ContainsKey("White") ? pgnGame.Headers["White"] : "?";
      PlayerBlack = pgnGame.Headers.ContainsKey("Black") ? pgnGame.Headers["Black"] : "?";

      if (pgnGame.Headers.ContainsKey("Result"))
      {
        string result = pgnGame.Headers["Result"];
        if (result.Contains("1/2-1/2"))
          Result = GameResult.Draw;
        else if (result.Contains("1-0"))
          Result = GameResult.WhiteWins;
        else if (result.Contains("0-1"))
          Result = GameResult.BlackWins;
      }
      else
        Result = GameResult.Unterminated;
      
      Moves = new List<Move>(pgnGame.Moves.Count);
      Position pos = InitialPosition;
      foreach (Textual.PgnFileTools.Move movePGN in pgnGame.Moves)
      {
        if (movePGN.HasError)
        {
          // TODO: Log.Info?
          Console.WriteLine("GameInfo contains error: " + movePGN.Annotation);
          continue;
        }

        Move move = Move.FromSAN(pos, movePGN.ToAlgebraicString());
        Moves.Add(move);
        pos = pos.AfterMove(move);
      }
    }

    #region Enumeration

    /// <summary>
    /// Enumerates all the moves/positions pairs in the history.
    /// </summary>
    public IEnumerable<PositionWithMove> PositionsWithMoves
    {
      get
      {
        Position pos = InitialPosition;
        foreach (Move move in Moves)
        {
          yield return (pos, move);
          pos = pos.AfterMove(move);
        }
      }
    }


    /// <summary>
    /// Enumerates all the positions in the history.
    /// </summary>
    public IEnumerable<Position> Positions
    {
      get
      {
        yield return InitialPosition;
        Position pos = InitialPosition;
        foreach (Move move in Moves)
        {
          pos = pos.AfterMove(move);
          yield return pos;
        }
      }
    }

    #endregion


    /// <summary>
    /// Returns if the either of the game players's last name
    /// contains a specified string.
    /// </summary>
    /// <param name="lastName"></param>
    /// <returns></returns>
    public bool PlayerLastNameContains(string lastName) =>   
      lastName!= null && 
      (PlayerWhite.ToLower().Contains(lastName.ToLower()) 
    || PlayerBlack.ToLower().Contains(lastName.ToLower()));

  }

}
