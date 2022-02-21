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
using Ceres.Base.DataTypes;
using Ceres.Base.Math;
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
    /// Index of game within file (starting with 0).
    /// </summary>
    public readonly int GameIndex;

    /// <summary>
    /// Round of game within file.
    /// </summary>
    public readonly int Round;

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

    /// <summary>
    /// Players evaluation in centipawns after each move.
    /// </summary>
    public float MovePlayerEvalCP(int moveIndex) => moveTimeSeconds == null ? float.NaN : movePlayerEvalCP[moveIndex];

    /// <summary>
    /// Number of nodes in search tree at end of each move.
    /// </summary>
    public ulong MoveNodes(int moveIndex) => moveNodes == null ? 0: moveNodes[moveIndex];

    /// <summary>
    /// Time spent by player on move (in seconds).
    /// </summary>
    public float MoveTimeSeconds(int moveIndex) => moveTimeSeconds == null ? float.NaN : moveTimeSeconds[moveIndex];


    public float MoveTimeTotalWhite => MoveTimeTotal(true);

    public float MoveTimeTotalBlack => MoveTimeTotal(false);


    short[] movePlayerEvalCP;
    ulong[] moveNodes;
    FP16[] moveTimeSeconds;

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
    public PGNGame(GameInfo gi, int gameIndex)
    {
      string dateStr = gi.Headers.ContainsKey("Date") ? gi.Headers["Date"] : null;
      if (dateStr != null) int.TryParse(dateStr.Substring(0, 4), out Year);

      GameIndex = gameIndex;
      WhitePlayer = gi.Headers.ContainsKey("White") ? gi.Headers["White"] : "?";
      BlackPlayer = gi.Headers.ContainsKey("Black") ? gi.Headers["Black"] : "?";
      StartFEN = gi.Headers.ContainsKey("FEN") ? gi.Headers["FEN"] : null;
      if (gi.Headers.ContainsKey("Round"))
      {
        if (float.TryParse(gi.Headers["Round"], out float round))
        {
          Round = (int)Round;
        }
      }

      if (gi.Headers.ContainsKey("Result"))
      {
        string result = gi.Headers["Result"];
        if (result.Contains("1/2-1/2"))
        {
          Result = GameResult.Draw;
        }
        else if (result.Contains("1-0"))
        {
          Result = GameResult.WhiteWins;
        }
        else if (result.Contains("0-1"))
        {
          Result = GameResult.BlackWins;
        }
      }
      else
      {
        Result = GameResult.Unknown;
      }

      StartFEN = StartFEN ?? Position.StartPosition.FEN;
      Moves = new PositionWithHistory(Position.FromFEN(StartFEN));

      int numMoves = gi.Moves.Count;

      int moveIndex = 0;

      void AddMoveEval(short cp)
      {
        if (movePlayerEvalCP == null)
        {
          movePlayerEvalCP = new short[numMoves];
        }
        movePlayerEvalCP[moveIndex] = cp;
      }

      void AddMoveTime(float time)
      {
        if (moveTimeSeconds == null)
        {
          moveTimeSeconds = new FP16[numMoves];
        }
        moveTimeSeconds[moveIndex] = (FP16)time;
      }

      void AddMoveNodes(ulong nodes)
      {
        if (moveNodes == null)
        {
          moveNodes = new ulong[numMoves];
        }
        moveNodes[moveIndex] = nodes;
      }

      foreach (Textual.PgnFileTools.Move move in gi.Moves)
      {
        if (move.HasError)
        {
          throw new Exception($"Encountered error in game { gameIndex + 1}, ply {Moves.Moves.Count + 1} parsing PGN: {move.ErrorMessage}");
        }
        Move m1 = Move.FromSAN(Moves.FinalPosition, move.ToAlgebraicString());
        Moves.AppendMove(m1.ToString());

        if (move.Comment != null)
        {
          try
          {
            string comment = move.Comment;
            //Console.WriteLine("COMMENT: " + move.Comment.Replace("\r","").Replace("\n",""));

            if (comment.Contains('='))
            {
              // example: 15.Bg1 {t=113460 e=219 nps=108329844 n=12291104138 d=42 sd=70}
              comment = ParseOctagonStyleComment(numMoves, moveIndex, comment);
            }
            else
            {
              // example: 5. Be3 {0.12/5 0.03s} 
              string[] parts = comment.Split(" ");
              if (parts.Length == 2)
              {
                ParseCutechessStyleComment(numMoves, moveIndex, parts);
              }
            }

          }
          catch (Exception ex)
          {
          }
        }
        moveIndex++;
      }

      FillInMissingEvals(numMoves);

      string ParseOctagonStyleComment(int numMoves, int moveIndex, string comment)
      {
        comment = comment.Substring(comment.IndexOf("{") + 1);
        comment = comment.Replace("{", "");
        comment = comment.Replace("\r", " ");
        comment = comment.Replace("\n", " ");
        foreach (string part in comment.Split(' '))
        {
          if (part.StartsWith("t="))
          {
            if (float.TryParse(part.Substring(2), out float value))
            {
              AddMoveTime(value / 1000.0f);
            }
          }
          else if (part.StartsWith("e="))
          {
            if (int.TryParse(part.Substring(2), out int value))
            {
              AddMoveEval((short)StatUtils.Bounded(value, short.MinValue, short.MaxValue));
            }
          }
          else if (part.StartsWith("n="))
          {
            if (ulong.TryParse(part.Substring(2), out ulong value))
            {
              AddMoveNodes(value);
            }

          }
        }

        return comment;
      }

      void ParseCutechessStyleComment(int numMoves, int moveIndex, string[] parts)
      {
        if (parts[0].Contains('/'))
        {
          if (float.TryParse(parts[0].Split("/")[0], out float eval))
          {
            eval = StatUtils.Bounded(MathF.Round(eval * 100), short.MinValue, short.MaxValue);
            AddMoveEval((short)eval);
          }
        }
        if (parts[1].EndsWith('s'))
        {
          if (float.TryParse(parts[1].Substring(0, parts[1].Length - 1), out float timeSeconds))
          {
            AddMoveTime(timeSeconds);
          }
        }
      }
    }

    /// <summary>
    /// Some engines may omit a centipawn evaluation upon instamove.
    /// Try to fill-in these evals by carrying forward last eval by that player.
    /// </summary>
    /// <param name="numMoves"></param>
    private void FillInMissingEvals(int numMoves)
    {
      if (movePlayerEvalCP != null && MoveTimeSeconds != null)
      {
        for (int i = 2; i < numMoves; i++)
        {
          if (moveTimeSeconds[i] == 0 && movePlayerEvalCP[i] == 0)
          {
            movePlayerEvalCP[i] = movePlayerEvalCP[i - 2];
          }
        }
      }
    }


    /// <summary>
    /// Returns string summary representation.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<PGNGame {Round}({GameIndex}): {WhitePlayer} vs {BlackPlayer} {ResultToString(Result)}, {Moves.Moves.Count} moves>";
    }


#region Static helpers

    /// <summary>
    /// Converts a GameResult into a corresponding short string (e.g. "1-0").
    /// </summary>
    /// <param name="result"></param>
    /// <returns></returns>
    public static string ResultToString(GameResult result)
    {
      switch (result)
      {
        case GameResult.WhiteWins:
          return "1-0";
        case GameResult.BlackWins:
          return "0-1";
        case GameResult.Draw:
          return "=";
        default:
          return "?";
      }
    }

#endregion

  }
}
