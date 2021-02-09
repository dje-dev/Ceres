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
using System.Text;
using Ceres.Base.Misc;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;
using Ceres.Chess.Textual;

#endregion

namespace Ceres.Chess.Games
{
  /// <summary>
  /// Manages generation of Portable Game Notation (PGN) files
  /// containing the moves and associated metadata of one or more chess games.
  /// 
  /// See: https://en.wikipedia.org/wiki/Portable_Game_Notation
  ///      http://www.saremba.de/chessgml/standards/pgn/pgn-complete.htm
  /// </summary>
  public class PGNWriter
  {

    int numPlyWritten = 0;
    Position startingPos;

    StringBuilder header = new StringBuilder();
    StringBuilder body = new StringBuilder();

    public string GetText() => header.ToString() + body.ToString() + Environment.NewLine;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="eventName"></param>
    /// <param name="playerWhite"></param>
    /// <param name="playerBlack"></param>
    /// <param name="tagPairs"></param>
    public PGNWriter(string eventName, string playerWhite, string playerBlack, 
                     params (string name, string value)[] tagPairs)
    {
      playerWhite = playerWhite ?? "Unknown";
      playerBlack = playerBlack ?? "Unknown";

      if (eventName != null) WriteTag("Event", eventName);
      WriteTag("White", playerWhite);
      WriteTag("Black", playerBlack);
      WriteTag("Date", DateTime.Now.ToString(("yyyy.MM.dd")));

      if (tagPairs != null)
      {
        foreach (var tagPair in tagPairs)
          WriteTag(tagPair.name, tagPair.value);
      }

      startingPos = Position.StartPosition;
    }


    const string RESULT_DRAW = "1/2-1/2";
    const string RESULT_WHITE_WINS = "1-0";
    const string RESULT_BLACK_WINS = "0-1";
    static string Quoted(string str) => "\"" + str + "\"";

    public void WriteStartPosition(string startingFEN)
    {
      if (startingFEN != Position.StartPosition.FEN)
      {
        FENParseResult parsedFEN = FENParser.ParseFEN(startingFEN);
        startingPos = parsedFEN.AsPosition;
        WriteTag("FEN", startingFEN);
      }

    }


    void WriteTag(string tagName, string tagValue)
    {
      // Truncate tag value if very long (to work around Fritz problem with very long tag value)
      tagValue = StringUtils.TrimmedIfNeeded(tagValue, 115);
      WriteHeaderLine($"[{tagName} \"{tagValue}\"]");
    }

    void WriteResult(string resultStr) { WriteBodyLine(" " + resultStr); WriteTag("Result", resultStr); }

    public void WriteResultDraw() => WriteResult(RESULT_DRAW);
    public void WriteResultWhiteWins() => WriteResult(RESULT_WHITE_WINS);
    public void WriteResultBlackWins() => WriteResult(RESULT_BLACK_WINS);


    public void WriteMove(MGMove move, Position pos, float? moveTimeSeconds = null, 
                          int? depth = null, float? scoreCentipawns = null)
    {
      bool isFirstMoveWritten = numPlyWritten == 0;

      // Write only two ply (one full move) per line.
      if (numPlyWritten++ % 2 == 0) WriteBodyLine("");

      int fullMoveNum = (startingPos.MiscInfo.MoveNum + numPlyWritten) / 2;

      string comment = "";
      if (moveTimeSeconds.HasValue || scoreCentipawns.HasValue)
      {
        // Example:  {-0.04 3.8s}
        string scoreStr = scoreCentipawns.HasValue ? $"{scoreCentipawns / 100.0f:F2}" : "";
        string depthStr = depth.HasValue ? $"/{depth}" : "";
        string timeStr = moveTimeSeconds.HasValue ? $" {moveTimeSeconds.Value:F2}s" : "";
        comment = " {" + scoreStr + depthStr + timeStr + "}";

      }

      string algebraicStr = MGMoveToString.AlgebraicMoveString(move, pos);
      if (move.WhiteToMove)
      {
        WriteBodyString($" {fullMoveNum}.");
      }
      else if (isFirstMoveWritten)
      {
        // Black makes first move, special case needing to write
        // move number before black move (see Section 8.2.2.2 of  PGN specification).
        WriteBodyString($" {fullMoveNum}...");
      }

      WriteBodyString($" {algebraicStr}{comment}");
    }


    public void WriteMoveSequence(PositionWithHistory moves)
    {
      if (moves.InitialPosMG != MGPosition.FromPosition(Position.StartPosition))
        WriteTag("FEN", moves.InitialPosMG.ToPosition.FEN);

      Position[] positions = moves.GetPositions();
      for (int i = 0; i < moves.Moves.Count; i++)
        WriteMove(moves.Moves[i], positions[i]);

      Position finalPos = moves.FinalPosition;
      GameResult terminal = finalPos.CalcTerminalStatus();
      if (terminal == GameResult.Checkmate && finalPos.MiscInfo.SideToMove == SideType.Black)
        WriteResultWhiteWins();
      else if (terminal == GameResult.Checkmate && finalPos.MiscInfo.SideToMove == SideType.White)
        WriteResultBlackWins();
      else if (terminal == GameResult.Draw || finalPos.CheckDrawCanBeClaimed != Position.PositionDrawStatus.NotDraw)
        WriteResultDraw();
    }


    void WriteHeaderLine(string str) => header.AppendLine(str);
    void WriteBodyLine(string str) => body.AppendLine(str);
    void WriteBodyString(string str) => body.Append(str);



    static void Test()
    {
      // {-0.03/9 5.0s}
      const string EN_PASSANT_BUG = "r2qkb1r/3n1ppp/p2pbn2/1p2p3/4P1P1/1NN1B2P/PPP2P2/R2QKB1R w KQkq - 0 20";
      const string EN_PASSANT_BUG_MOVES = "a2a4 b5b4 c3d5 h7h5 g4g5 f6d5 e4d5 e6f5 f1d3 e5e4 d3e2 g7g6 b3d4 f8g7";
      PositionWithHistory moves = PositionWithHistory.FromFENAndMovesUCI(EN_PASSANT_BUG, EN_PASSANT_BUG_MOVES);
      moves = PositionWithHistory.FromFENAndMovesUCI(Position.StartPosition.FEN, "e2e4 e7e6 d2d4 d7d5 b1c3 f8b4 e4d5 e6d5 f1d3 b8c6 a2a3 b4e7 c3d5 e7b4 d1d2 b4d2 c1d2 g8f6 e1c1 f6d7 b2b3 c6e5");

      PGNWriter writer = new PGNWriter("Test 3min/move", "Stockfish 11", "Ceres");
      writer.WriteMoveSequence(moves);
      writer.WriteResultDraw();
      Console.WriteLine(writer.GetText());
    }

  }
}
