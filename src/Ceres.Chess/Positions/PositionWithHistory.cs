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
using System.Linq;

using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.Chess.Positions
{
  /// <summary>
  /// Represents a position position possibly also with history
  /// (set of prior moves that preceded to this position).
  /// 
  /// TODO: The "finalization" of setting the repetition count
  ///       is probably done inconsistently below.
  ///       Perhaps make this done only upon explicit call.
  ///       Note that in search (Annotate) we do not want this class
  ///       to do the calculation of repetition counts since it is 
  ///       already done there efficiently.
  ///       
  /// TODO: More generally than the above, probably this class should be
  ///       split into two. Currently it combines the function of a builder 
  ///       and a final representation of a sequence of positions.
  /// </summary>
  [Serializable]
  public class PositionWithHistory : ICloneable, IEquatable<PositionWithHistory>
  {
    #region Private data

    public MGPosition InitialPosMG { get; private set; }

    List<MGMove> moves;

    public List<MGMove> Moves => moves;

    bool haveFinalized = false;
    MGPosition finalPosMG;

    Position[] positions;


    /// <summary>
    /// Optionally the next actual move (made after the final position).
    /// </summary>
    public Move NextMove { set; get; }

    /// <summary>
    /// Total number of positions in history.
    /// </summary>
    public int Count
    {
      get
      {
        if (!haveFinalized)
        {
          InitPositionsAndFinalPosMG();
        }
        return positions.Length;
      }
    }

    #endregion

    #region Constructor and update methods

    public PositionWithHistory(PositionWithHistory copy)
    {
      InitialPosMG = copy.InitialPosMG;
      if (copy.Moves != null)
      {
        moves = new List<MGMove>(copy.Moves);
      }
      haveFinalized = copy.haveFinalized;
      finalPosMG = copy.FinalPosMG;
      NextMove = copy.NextMove;

      positions = new Position[copy.positions.Length];
      Array.Copy(copy.positions, positions, positions.Length);
    }


    public PositionWithHistory(in Position initialPos, List<MGMove> moves = null)
    {
      InitialPosMG = MGChessPositionConverter.MGChessPositionFromPosition(in initialPos);
      this.moves = moves ?? new List<MGMove>();
      InitPositionsAndFinalPosMG();
    }

    public PositionWithHistory(IEnumerable<Position> positions, bool firstPositionMayBeMissingEnPassant, bool recalcRepetitions = false)
      => SetFromPositions(positions, firstPositionMayBeMissingEnPassant, recalcRepetitions);

    public PositionWithHistory(Span<Position> positions, bool firstPositionMayBeMissingEnPassant, bool recalcRepetitions = false)
      => SetFromPositions(positions, firstPositionMayBeMissingEnPassant, recalcRepetitions);


    public PositionWithHistory(in MGPosition initialPosMG, List<MGMove> moves = null)
    {
      InitialPosMG = initialPosMG;
      this.moves = moves ?? new List<MGMove>();
      InitPositionsAndFinalPosMG();
    }


    public void AppendMove(string moveStr)
    {
      MGPosition mgPos = MGPosition.FromPosition(FinalPosition);
      MGMove thisMove = MGMoveFromString.ParseMove(in mgPos, moveStr);
      if (thisMove.IsNull)
      {
        throw new Exception("Unexpected null move");
      }

      AppendMove(thisMove);
    }


    public void AppendPosition(MGPosition position, MGMove move)
    {
      Moves.Add(move);
      InitPositionsAndFinalPosMG();
    }


    public void AppendMove(MGMove move)
    {
      // Verify move is legal from this position.
      MGMoveList moves = new MGMoveList();
      MGMoveGen.GenerateMoves(FinalPosMG, moves);
      if (Array.IndexOf(moves.MovesArray, move) == -1)
      {
        throw new Exception($"The move {move} is not legal from position {FinalPosition.FEN}");
      }

      Moves.Add(move);

      InitPositionsAndFinalPosMG();
    }

    /// <summary>
    /// Forces the final position to a specified value.
    /// This may be needed in rare cases where this was derived from an EncodedBoardPosition/TPG.
    /// In that case, the state of the castling flags at the beginning of the sequence is not known for sure.
    /// Therefore reconstructing the final position from the moves may not be correct,
    /// (as might happen if an AppendMove call were made to thi sposition).
    /// This methods allows overriding the final position with the correct castling rights.
    /// 
    /// TODO: someday try to modify TGPRecord/EncodedPositionBoard to better reconstruct castling rights
    ///       by assuming all positions have castling rights as final position
    ///       but then adding back castling rights at point in prior sequence before castling moves were made (if any).
    /// </summary>
    /// <param name="mgPos"></param>
    public void ForceFinalPosMG(MGPosition mgPos)
    {
      finalPosMG = mgPos;
      positions[^1] = mgPos.ToPosition;
    }


    /// <summary>
    /// Searches the position list for a given position, or -1 if not present.
    /// If the position exists more than once, the index of the last occurrence is returned.
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
        {
          return i;
        }
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
      PositionWithHistory ret = new PositionWithHistory(in pos);
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
    public static PositionWithHistory FromPosition(in Position pos) => new PositionWithHistory(in pos);


    /// <summary>
    /// Constructs a new MGMoveSequence given a starting position (as a FEN) 
    /// followed by an optional string containing a sequence of subsequent moves (in coordinate notation).
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
      {
        return FromFENAndMovesUCI(parts[0], ""); // nothing after the moves token
      }
      else
      {
        return FromFENAndMovesUCI(parts[0], parts[1]);
      }
    }

    /// <summary>
    /// Returns the FEN followed by "moves" and the move list (if any).
    /// </summary>
    public string FENAndMovesString
    {
      get
      {
        string ret = InitialPosition.FEN;
        if (Moves != null && Moves.Count > 0)
        {
          ret += " moves " + MovesStr;
        }
        return ret;
      }
    }


    /// <summary>
    /// Constructs a new MGMoveSequence given a starting position (as a FEN) 
    /// and an optional string containing a sequence of subsequent moves (in coordinate notation).
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
            MGMove mgMove = MGMoveFromString.ParseMove(in mgPos, moveStr);
            ret.Moves.Add(mgMove);

            mgPos.MakeMove(mgMove);
          }
        }
        ret.InitPositionsAndFinalPosMG();
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

    private void SetFromPositions(IEnumerable<Position> fromPositions,
                                  bool firstPositionMayBeMissingEnPassant,
                                  bool recalcRepetitions = false)
    {
      SetFromPositions(fromPositions.ToArray().AsSpan(), firstPositionMayBeMissingEnPassant, recalcRepetitions);
    }


    public void SetFromPositions(Span<Position> positions,
                                  bool firstPositionMayBeMissingEnPassant,
                                  bool recalcRepetitions = false)
    {
      InitialPosMG = positions[0].ToMGPosition;
      finalPosMG = positions[^1].ToMGPosition;

      if (recalcRepetitions)
      {
        PositionRepetitionCalc.SetRepetitionsCount(positions);
      }

      haveFinalized = true;

      // Reconstruct the sequence of moves from the positions.
      moves = new(positions.Length - 1);
      for (int i = 0; i < positions.Length - 1; i++)
      {
        MGPosition posCurrent = positions[i].ToMGPosition;
        MGPosition posNext = positions[i + 1].ToMGPosition;

        if (i == 0 && firstPositionMayBeMissingEnPassant)
        {
          // To make sure we find a possible en passant move actually made,
          // change the position such that all files are marked as en passant eligible.
          posCurrent.SetAllEnPassantAllowed();
        }

        // Generate moves and iterate over all positions resulting from those moves to look for a match.
        MGMoveList movesList = new MGMoveList();
        MGMoveGen.GenerateMoves(posCurrent, movesList);

        bool foundContinuation = false;
        for (int j = 0; j < movesList.NumMovesUsed; j++)
        {
          MGPosition tryNextPos = posCurrent;
          tryNextPos.MakeMove(movesList.MovesArray[j]);

          if (tryNextPos.EqualPiecePositionsExcludingEnPassant(posNext))
          {
            Moves.Add(movesList.MovesArray[j]);
            foundContinuation = true;
            break;
          }
        }

        if (i > 0 || !firstPositionMayBeMissingEnPassant)
        {
          if (!foundContinuation
            && EqualityComparer<Position>.Default.Equals(positions[i], positions[i + 1])) // ok if repetition (fill in planes0
          {
            string errMsg = i + " Position sequence illegal, saw " + positions[i + 1].ToMGPosition.ToPosition.FEN
                          + " then " + positions[i].ToMGPosition.ToPosition.FEN
                          + ". This could possibly arise due to FRC position in training data and not be an error.";
            throw new Exception(errMsg);
          }
        }
      }

      // N.B. We do not call InitPositionsAndFinalPosMG because
      //      the positions passed in may not be exact (if coming from training data may lack castling info, etc)
      //      the the rebuilding of positions in this method may not be exact.
      //      Instead just directly use positions.
      this.positions = positions.ToArray();
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
      if (InitialPosMG != otherMoves.InitialPosMG)
      {
        return false;
      }

      // Verify we are not shorter
      if (Moves.Count < otherMoves.Moves.Count)
      {
        return false;
      }

      // Verify all moves in other are same
      for (int i = 0; i < otherMoves.Moves.Count; i++)
      {
        if (Moves[i] != otherMoves.Moves[i])
        {
          return false;
        }
      }

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
    public Position[] Positions
    {
      get
      {
        CheckInit();
        return positions;
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
        foreach (MGMove move in Moves)
        {
          moveStr += move.MoveStr(MGMoveNotationStyle.Coordinates) + " ";
        }
        return moveStr;
      }
    }


    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      if (Moves == null || Moves.Count == 0)
      {
        return $"<PositionWithHistory ({positions.Length}) {InitialPosMG.ToPosition.FEN} --> final position {FinalPosition.FEN} (no moves list)>";
      }
      else
      {
        return $"<PositionWithHistory ({positions.Length}) {InitialPosMG.ToPosition.FEN} moves {MovesStr} --> final position {FinalPosition.FEN}>";
      }
    }


    /// <summary>
    /// Returns a deep clone of the object.
    /// </summary>
    /// <returns></returns>
    public object Clone()
    {
      PositionWithHistory clone = new PositionWithHistory(InitialPosition);
      foreach (MGMove move in Moves)
      {
        clone.AppendMove(move);
      }

      return clone;
    }


    #region Equality

    /// <summary>
    /// Returns if two PositionWithHistory objects are equal.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns></returns>
    public static bool operator ==(PositionWithHistory a, PositionWithHistory b)
    {
      if (a is null)
      {
        return b is null;
      }
      else
      {
        return a.Equals(b);
      }
    }



    /// <summary>
    /// Returns if two PositionWithHistory objects are not equal.
    /// </summary>
    /// <param name="a"></param>
    /// <param name="b"></param>
    /// <returns></returns>
    public static bool operator !=(PositionWithHistory a, PositionWithHistory b) => !(a == b);


    /// <summary>
    /// Implements equality operator.
    /// </summary>
    /// <param name="other"></param>
    /// <returns></returns>
    public bool Equals(PositionWithHistory other)
    {
      // Equality of positions suffices.
      return other != null
          && positions.Length == other.positions.Length
          && positions.SequenceEqual(other.positions);
    }

    #endregion
  }

  #endregion
}

