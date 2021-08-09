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


using Ceres.Chess.Positions;
using Ceres.Chess.Textual;
using System;
using System.Collections.Generic;

#endregion

namespace Ceres.Chess.Games.Utils
{
  /// <summary>
  /// Represents a single position entry in an EPD file.
  /// 
  /// For specification, see: http://jchecs.free.fr/pdf/EPDSpecification.pdf
  /// 
  /// Example: 1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - bm Qd1+; id "BK.01";
  /// </summary>
  public partial record EPDEntry
  {
    public enum MovesFormatEnum
    {
      SAN,
      UCI
    }

    /// <summary>
    /// Starting FEN.
    /// </summary>
    public readonly string FEN;

    /// <summary>
    /// Optional sequence of moves which followed starting FEN.
    /// </summary>
    public readonly string StartMoves;

    /// <summary>
    /// Optional set of benchmark (good) moves.
    /// </summary>
    public readonly string[] BMMoves; 

    /// <summary>
    /// Optional set of avoid (bad) moves.
    /// </summary>
    public readonly string[] AMMoves;

    /// <summary>
    /// Optional descriptio of position.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Notation used to express any associated moves.
    /// </summary>
    public readonly MovesFormatEnum MovesFormat = MovesFormatEnum.SAN;


    /// <summary>
    /// Returns as a FEN and also "moves" with list of moves (if any).
    /// </summary>
    public string FENAndMoves => FEN + (StartMoves == null ? "" : " moves " + StartMoves);


    /// <summary>
    /// Creates a new EPD entry identical to this, except FEN is changed as specified
    /// </summary>
    /// <param name="newFEN"></param>
    /// <returns></returns>
    public EPDEntry WithChangedFEN(string newFEN) => new EPDEntry(newFEN, BMMoves, ID, ScoredMoves);

    /// <summary>
    /// Internal constructor.
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="bmMoves"></param>
    /// <param name="id"></param>
    /// <param name="moves"></param>
    internal EPDEntry(string fen, string[] bmMoves, string id, List<EPDScoredMove> moves)
    {
      FEN = fen.Replace("  ", " ");
      BMMoves = bmMoves;
      ID = id;
      ScoredMoves = moves;
    }

    /// <summary>
    /// Internal constructor.
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="bmMove"></param>
    /// <param name="id"></param>
    /// <param name="moves"></param>
    internal EPDEntry(string fen, string bmMove, string id, List<EPDScoredMove> moves)
    {
      FEN = fen.Replace("  ", " ");
      BMMoves = new string[] { bmMove };
      ID = id;
      ScoredMoves = moves;
    }

    public Position Position => Position.FromFEN(FEN);

    public List<EPDScoredMove> ScoredMoves = new List<EPDScoredMove>();

    public int ValueOfMove(Move move, Position pos)
    {
      foreach (EPDScoredMove sm in ScoredMoves)
      {
        if (SANParser.FromSAN(sm.MoveStr, pos).Move == move)
        {
          return sm.Score;
        }
      }

      return 0;
    }


    /// <summary>
    /// Constructor which parses an EPD line, 
    /// extracting FEN and benchmark moves/commentary
    /// </summary>
    /// <param name="epdLine"></param>
    public EPDEntry(string epdLine, bool lichessPuzzleFormat = false)
    {
      //xxx,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,2032,75,91,297,crushing hangingPiece long middlegame,https://lichess.org/787zsVup/black#48
      if (lichessPuzzleFormat)
      {
        int indexFirstComma = epdLine.IndexOf(",");
        ID = epdLine.Substring(0, indexFirstComma);

        epdLine = epdLine.Substring(indexFirstComma + 1);
        string[] parts = epdLine.Split(",");
        string fen = parts[0];
        string[] moveParts = parts[1].Split(" ");
        string movePre = moveParts[0];
        string moveCorrect = moveParts[1];

        FEN = fen;
        StartMoves = movePre;
        BMMoves = new string[] { moveCorrect };
        List<EPDScoredMove> moves = new List<EPDScoredMove>(1);
        moves.Add(new EPDScoredMove(moveCorrect, 1));
        MovesFormat = MovesFormatEnum.UCI;
        return;
      }

      int posBMOrAM = epdLine.IndexOf(" bm "); // benchmark move
      bool hasBM = posBMOrAM != -1;
      if (!hasBM) posBMOrAM = epdLine.IndexOf(" am "); // avoid move

      if (posBMOrAM == -1)
      {
        GetFENAndStartMovesFromFENStr(epdLine, out FEN, out StartMoves);
      }
      else
      {
        // Typically the beginning of the correct move is delimited by a semicolon
        int posSemi = epdLine.IndexOf(";");

        // Absent a semicolon, we look for the final space as a separator
        if (posSemi == -1) posSemi = epdLine.Length - 1;
        
        string fenStr = epdLine.Substring(0, posBMOrAM);

        // Replace any "-" at end indicating unknown move counts with default move counts since some programs don't accept this (e.g. Leela)
        string[] parts = fenStr.Split(' ');
        if (parts.Length == 4 || (parts.Length == 5 && parts[4] == "-"))
        {
          fenStr = parts[0] + " " + parts[1] + " " + parts[2] + " " + parts[3] + " 0 1";
        }

        GetFENAndStartMovesFromFENStr(fenStr, out FEN, out StartMoves);

        for (int i = 0; i < epdLine.Length; i++)
        {
          if (epdLine[i] == '=')
          {
            string[] left = epdLine.Substring(0, i).Split(new char[] { '"', ',', ' ' });
            string[] right = epdLine.Substring(i + 1).Split(new char[] { '"', ',', ' ' });
            if (!char.IsNumber(right[0][0]))
            {
              continue; // catch case where this is promotion, not a score, e.g. =Q
            }

            ScoredMoves.Add(new EPDScoredMove(left[left.Length - 1], int.Parse(right[0])));
          }
        }

        //"Bxe5=10, f4=3, Nc4=2"; c8 "10 3 2"; c9 "b8e5 f5f4 b6c4";
        string bmOrAMMoves = epdLine.Substring(posBMOrAM + 4, posSemi - posBMOrAM - 3);

        string[] amOrBMMovesArray = bmOrAMMoves.Replace(',', ' ').Replace("; ",";").Replace("  ", " ").Replace(";", "").Split(' ', ',');// sometimes a sequence of benchmark moves are specified, with spaces or commas in between

        // Remove empty tokens
        List<string> amOrBMMovesArrayList = new List<string>();
        foreach (string move in amOrBMMovesArray)
        {
          if (move.Trim() != "")
          {
            amOrBMMovesArrayList.Add(move);
          }
        }

        amOrBMMovesArray = amOrBMMovesArrayList.ToArray();

        // Check moves for validity
        Position pos = Position.FromFEN(FEN);
        if (hasBM)
        {
          BMMoves = amOrBMMovesArray;
          CheckAllMovesValid(in pos, BMMoves);
        }
        else
        {
          AMMoves = amOrBMMovesArray;
          CheckAllMovesValid(in pos, AMMoves);
        }

        if (epdLine.Contains("id "))
        {
          //r1bqk1r1/1p1p1n2/p1n2pN1/2p1b2Q/2P1Pp2/1PN5/PB4PP/R4RK1 w q - - bm Rxf4; id "ERET 001 - Entlastung";
          string idStr = epdLine.Substring(epdLine.IndexOf("id ") + 3);
          ID = idStr.Replace("\"", "").Replace(";", "");
        }
      }

    }


    /// <summary>
    /// Diagnostic method that throws an Exception if any of the 
    /// specified moves are not valid (e.g. benhcmark moves).
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="moves"></param>
    void CheckAllMovesValid(in Position pos, string[] moves)
    {
      foreach (string move in moves)
      {
        if (!MoveValid(in pos, move, out string errStr))
        {
          Console.WriteLine(errStr);
          throw new Exception(errStr);
        }
      }
    }


    /// <summary>
    /// Returns if a specified move is valid from a specified position.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="moveStr"></param>
    /// <param name="errString"></param>
    /// <returns></returns>
    bool MoveValid(in Position position, string moveStr, out string errString)
    {
      try
      {
        Move epdMove = MovesFormat == MovesFormatEnum.SAN ? SANParser.FromSAN(moveStr, in position).Move
                                                          : Move.FromUCI(moveStr);
        errString = null;
        return true;
      }
      catch (Exception exc)
      {
        errString = "Invalid move specified in EPD " + moveStr + " from FEN " + position.FEN;
        return false;
      }
    }


    /// <summary>
    /// Returns starting position of the suite test.
    /// </summary>
    public PositionWithHistory PosWithHistory
    {
      get
      {
        if (MovesFormat == MovesFormatEnum.SAN)
        {
          return PositionWithHistory.FromFENAndMovesSAN(FEN, StartMoves);
        }
        else if (MovesFormat == MovesFormatEnum.UCI)
        {
          return PositionWithHistory.FromFENAndMovesUCI(FEN, StartMoves);
        }
        else
        {
          throw new Exception("Unsupported notation type " + MovesFormat);
        }
      }
    }

    /// <summary>
    /// Checks the AMMoves and BMMoves for a specified move
    /// and returns corresponding score if found.
    /// </summary>
    /// <param name="move"></param>
    /// <param name="valueOfBestMove"></param>
    /// <returns></returns>
    public int CorrectnessScore(Move move, int valueOfBestMove)
    {
      int value = 0;
      bool correct = false;

      if (AMMoves != null)
      {
        // avoid move
        Move avoidMoveEPD = MovesFormat == MovesFormatEnum.SAN ? SANParser.FromSAN(AMMoves[0], Position).Move
                                                               : Move.FromUCI(AMMoves[0]);
        correct = move != avoidMoveEPD;
        if (correct) value = valueOfBestMove;
      }
      else if (BMMoves != null)
      {
        try
        {
          // best move
          foreach (string bmMove in BMMoves)
          {
            if (bmMove != "")
            {
              Move bmMoveDecoded = MovesFormat == MovesFormatEnum.SAN ? SANParser.FromSAN(BMMoves[0], Position).Move
                                                                      : Move.FromUCI(BMMoves[0]);

              if (bmMoveDecoded == move)
              {
                correct = true;
              }
            }
          }

          if (correct)
          {
            value = valueOfBestMove; // Some entries only have single designated Bm move
          }
          else
          {
            value = ValueOfMove(move, Position);
          }
        }
        catch (Exception e)
        {
          Console.WriteLine("Warn: unparsable correct EPD move ");
        }
      }

      return value;
    }


    /// <summary>
    /// Static helper to return List of all EPD entries in a specified file.
    /// </summary>
    /// <param name="epdFN"></param>
    /// <param name="maxEntries"></param>
    /// <returns></returns>
    public static List<EPDEntry> EPDEntriesInEPDFile(string epdFN, int maxEntries = int.MaxValue, 
                                                     bool skipFirstColumn = false, Predicate<string> includeFilter = null)
    {
      string[] lines = System.IO.File.ReadAllLines(epdFN);
      List<EPDEntry> ret = new List<EPDEntry>(lines.Length);

      foreach (string line in lines)
      {
        if (ret.Count >= maxEntries)
        {
          break;
        }

        if (includeFilter != null && !includeFilter(line))
        {
          continue;
        }

        // Skip comment lines
        if (line.StartsWith("#"))
        {
          continue;
        }

        // Skip emtpy line
        if (line.Trim().Length == 0)
        {
          continue;
        }

        ret.Add(new EPDEntry(line, skipFirstColumn));
      }

      return ret;
    }

    /// <summary>
    /// Returns string FEN position and string with sequence 
    /// of moves which followed from the position.
    /// </summary>
    /// <param name="fenStr"></param>
    /// <param name="fen"></param>
    /// <param name="movesStr"></param>
    void GetFENAndStartMovesFromFENStr(string fenStr, out string fen, out string movesStr)
    {
      if (fenStr.Contains("moves"))
      {
        int movesPos = fenStr.IndexOf("moves ");
        fen = fenStr.Substring(0, movesPos);
        movesStr = fenStr.Substring(movesPos + 6);
      }
      else
      {
        fen = fenStr;
        movesStr = null;
      }
    }


    /// <summary>
    /// Returns string representation of EPD entry.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string ret = "<EPDEntry "  /*+ Pos.pretty(0)*/ + " " + ID + " "; // TODO: + (BMMoves != null ?  "bm  + BMMoves[0]
      foreach (EPDScoredMove sm in ScoredMoves)
      {
        ret += " " + sm.MoveStr + "=" + sm.Score;
      }

      return ret + ">";
    }
  }

}
