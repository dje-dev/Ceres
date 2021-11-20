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

using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.Chess.Positions
{
  /// <summary>
  /// Represents a position position possibly also with history
  /// (set of prior moves that preceeded to this position).
  /// 
  /// TODO: The "finalization" of setting the repetition count
  ///       is probably done inconsistenly below.
  ///       Perhaps make this done only upon explicit call.
  ///       Note that in search (Annotate) we do not want this class
  ///       to do the calculation of reptition counts since it is 
  ///       already done there efficiently.
  /// </summary>
  [Serializable]
  public class PositionWithHistory : ICloneable
  {
    #region Private data

    public readonly MGPosition InitialPosMG;
    public List<MGMove> Moves;

    bool haveFinalized = false;
    MGPosition finalPosMG;

    Position[] positions;

    /// <summary>
    /// Optionally the next actual move (made after the final position).
    /// </summary>
    public Move NextMove { set; get; }

    #endregion

    #region Constructor and update methods

    public PositionWithHistory(PositionWithHistory copy)
    {
      InitialPosMG = copy.InitialPosMG;
      Moves = new List<MGMove>(copy.Moves);
      haveFinalized = copy.haveFinalized;
      finalPosMG = copy.FinalPosMG;
      NextMove = copy.NextMove;

      positions = new Position[copy.positions.Length];
      Array.Copy(copy.positions, positions, positions.Length);
    }


    public PositionWithHistory(Position initialPos, List<MGMove> moves = null)
    {
      InitialPosMG = MGChessPositionConverter.MCChessPositionFromPosition(in initialPos);
      Moves = moves ?? new List<MGMove>();
    }


    public PositionWithHistory(MGPosition initialPosMG, List<MGMove> moves = null)
    {
      InitialPosMG = initialPosMG;
      Moves = moves ?? new List<MGMove>();
    }


    public void AppendMove(string moveStr)
    {
      MGPosition mgPos = MGPosition.FromPosition(FinalPosition);
      MGMove thisMove = MGMoveFromString.ParseMove(mgPos, moveStr);
      if (thisMove.IsNull) throw new Exception("Unexpected null move");

      // Verify move is legal from this position
      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(in mgPos, moves);
      if (Array.IndexOf(moves.MovesArray, thisMove) == -1)
        throw new Exception($"The move {moveStr} is not legal from position {FinalPosition.FEN}");

      Moves.Add(MGMoveFromString.ParseMove(mgPos, moveStr));
      if (haveFinalized) InitPositionsAndFinalPosMG();
    }


    public void AppendMove(MGMove move)
    {
      Moves.Add(move);
      if (haveFinalized) InitPositionsAndFinalPosMG();
    }

    /// <summary>
    /// Searches the position list for a given position, or -1 if not present.
    /// If the position exists more than once, the index of the last occurence is returned.
    /// </summary>
    /// <param name="findPosition"></param>
    /// <returns></returns>
    public int FindPositionLast(Position findPosition)
    {
      Position[] currentPositionsHistory = GetPositions();
      for (int i = currentPositionsHistory.Length - 1; i >= 0; i--)
      {
        Position thisPos = currentPositionsHistory[i];
        if (findPosition.Equals(thisPos))
          return i;
      }
      return -1;
    }


    /// <summary>
    /// Returns a PositionWithHistory from a specified starting FEN and sequence of move strings in SAN format.
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="sanMoves"></param>
    /// <returns></returns>
    public static PositionWithHistory FromFENAndMovesSAN(string fen, params string[] sanMoves)
    {
      Position pos = Position.FromFEN(fen);
      PositionWithHistory ret = new PositionWithHistory(pos);
      if (sanMoves != null)
      {
        foreach (string sanMoveString in sanMoves)
        {
          if (sanMoveString != null)
          {
            Move move = pos.MoveSAN(sanMoveString);
            ret.AppendMove(MGMoveConverter.MGMoveFromPosAndMove(in pos, move));
            pos = pos.AfterMove(move);
          }
        }
      }

      return ret;
    }


    /// <summary>
    /// Returns a PositionWithHistory corresponding to the starting position.
    /// </summary>
    public static PositionWithHistory StartPosition => new PositionWithHistory(Position.StartPosition);


    /// <summary>
    /// Returns a PositionWithHistory corresponding to specified starting position (and no history).
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public static PositionWithHistory FromPosition(in Position pos) => new PositionWithHistory(pos);


    /// <summary>
    /// Constructs a new MGMoveSequence given a starting position (as a FEN) 
    /// followed by an optional string containing a sequence of subsequent moves (in coordiante notation).
    /// </summary>
    /// <param name="fenAndMovesStr"></param>
    /// <returns></returns>
    public static PositionWithHistory FromFENAndMovesUCI(string fenAndMovesStr)
    {
      int movesIndex = fenAndMovesStr.IndexOf(" moves");
      if (movesIndex == -1) return FromFENAndMovesUCI(fenAndMovesStr, null);

      fenAndMovesStr = fenAndMovesStr.Replace("startpos", Position.StartPosition.FEN);

      string[] parts = fenAndMovesStr.Split(" moves");

      if (parts.Length == 1)
        return FromFENAndMovesUCI(parts[0], ""); // nothing after the moves token
      else
        return FromFENAndMovesUCI(parts[0], parts[1]);
    }

    /// <summary>
    /// Returns the FEN followed by "moves" and the move list (if any).
    /// </summary>
    public string FENAndMovesString
    {
      get
      {
        string ret = InitialPosition.FEN;
        if (Moves != null && Moves.Count > 0) ret += " moves " + MovesStr;
        return ret;
      }
    }


    /// <summary>
    /// Constructs a new MGMoveSequence given a starting position (as a FEN) 
    /// and an optional string containing a sequence of subsequent moves (in coordiante notation).
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="movesStr"></param>
    /// <returns></returns>
    public static PositionWithHistory FromFENAndMovesUCI(string fen, string movesStr)
    {
      fen = fen.Replace("startpos", Position.StartPosition.FEN);
      MGPosition mgPos = MGPosition.FromFEN(fen);

      PositionWithHistory ret = new PositionWithHistory(mgPos);
      if (movesStr != null && movesStr != "")
      {
        string[] parts = movesStr.Split(" ");

        for (int i = 0; i < parts.Length; i++)
        {
          string moveStr = parts[i];
          if (moveStr != "")
          {
            MGMove mgMove = MGMoveFromString.ParseMove(mgPos, moveStr);
            ret.Moves.Add(mgMove);

            mgPos.MakeMove(mgMove);
          }
        }
      }

      return ret;
    }

    #endregion

    public Position FinalPosition => GetPositions()[^1];
    public Position InitialPosition => GetPositions()[0];

    /// <summary>
    /// Returns array of all Positions which sequentially
    /// occurred in the PositionWitHistory..
    /// </summary>
    /// <returns></returns>
    public Position[] GetPositions()
    {
      CheckInit();
      return positions;
    }

#if NOT
    public Position[] GetPositionsWithEnPassantPreExtension()
    {
      CheckInit();

      // Nothing to do if first position is not en passant.
      if (positions[0].MiscInfo.EnPassantFileIndex == PositionMiscInfo.EnPassantFileIndexEnum.FileNone)
        return positions;

      // Create a new array with one extra slot.
      Position[] extendedPositions = new Position[positions.Length + 1];

      // The new first move is the pre-extension.
      extendedPositions[0] = positions[0].PosWithEnPassantUndone();

      // Copy over all the original positions.
      for (int i = 0; i < positions.Length; i++)
        extendedPositions[i + 1] = positions[i];

      return extendedPositions;
    }
#endif

    public MGPosition FinalPosMG
    {
      get
      {
        CheckInit();
        return finalPosMG;
      }
    }


    private void CheckInit()
    {
      if (!haveFinalized)
      {
        InitPositionsAndFinalPosMG();
        haveFinalized = true;
      }
    }

    private void InitPositionsAndFinalPosMG()
    {
      positions = new Position[Moves.Count + 1];

      int index = 0;
      positions[index++] = InitialPosMG.ToPosition;

      MGPosition mgPos = InitialPosMG;
      foreach (MGMove mgMove in Moves)
      {
        mgPos.MakeMove(mgMove);
        positions[index++] = mgPos.ToPosition;
      }

      PositionRepetitionCalc.SetRepetitionsCount(positions);
      finalPosMG = mgPos;
    }


    /// <summary>
    /// Returns if another MGMoveSequence is identical to this, except that
    /// this has exactly one extra move on the end
    /// </summary>
    /// <param name="otherMoves"></param>
    /// <returns></returns>
    public bool IsIdenticalToPriorToLastMove(PositionWithHistory otherMoves)
    {
      // Verify starting position same
      if (InitialPosMG != otherMoves.InitialPosMG) return false;

      // Verify we are not shorter
      if (Moves.Count < otherMoves.Moves.Count) return false;

      // Verify all moves in other are same
      for (int i = 0; i < otherMoves.Moves.Count; i++)
        if (Moves[i] != otherMoves.Moves[i])
          return false;

      return true;
    }

    /// <summary>
    /// Returns the number of times a specified position occurred
    /// in the move history (repetition equality).
    /// </summary>
    /// <param name="pos"></param>
    /// <returns></returns>
    public int NumOccurences(in Position pos)
    {
      int count = 0;
      foreach (Position priorPos in Positions)
      {
        if (priorPos.EqualAsRepetition(in pos))
        {
          count++;
        }
      }
      return count;
    }


    #region Enumeration

    /// <summary>
    /// Enumerates all the moves/positions pairs in the history.
    /// </summary>
    public IEnumerable<PositionWithMove> PositionsWithMoves
    {
      get
      {
        MGPosition pos = InitialPosMG;
        foreach (MGMove move in Moves)
        {
          yield return (pos.ToPosition, MGMoveConverter.ToMove(move));
          pos.MakeMove(move);          
        }
      }
    }


    /// <summary>
    /// Enumerates all positions in game (including their move histories).
    /// </summary>
    public IEnumerable<PositionWithHistory> PositionWithHistories
    {
      get
      {
        PositionWithHistory pwh = new(InitialPosMG);

        // Return first position.
        yield return new PositionWithHistory(pwh);

        // Return all subsequent positions.
        for (int i = 0; i < Moves.Count; i++)
        {
          // Append this move and return a clone.
          pwh.AppendMove(Moves[i]);
          yield return new PositionWithHistory(pwh); 
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
        yield return InitialPosMG.ToPosition;
        MGPosition pos = InitialPosMG;
        foreach (MGMove move in Moves)
        {
          pos.MakeMove(move);
          yield return pos.ToPosition;
        }
      }
    }

    #endregion


    #region Descriptive strings

    /// <summary>
    /// Returns space separated sequence of consecutive history moves (in coordinate style).
    /// </summary>
    public string MovesStr
    {
      get
      {
        string moveStr = "";
        foreach (MGMove move in Moves) moveStr += move.MoveStr(MGMoveNotationStyle.LC0Coordinate) + " ";
        return moveStr;
      }
    }


    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      if (Moves.Count == 0)
        return $"<PositionWithHistory {InitialPosMG.ToPosition.FEN} (no history)>";
      else
        return $"<PositionWithHistory {InitialPosMG.ToPosition.FEN} moves {MovesStr} --> final position {FinalPosition.FEN}>";
    }

    /// <summary>
    /// Returns a deep clone of the object.
    /// </summary>
    /// <returns></returns>
    public object Clone()
    {
      PositionWithHistory clone = new PositionWithHistory(InitialPosition);
      foreach (MGMove move in Moves)
        clone.AppendMove(move);
      return clone;
    }

    #endregion
  }

}
