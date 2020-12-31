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
using System.Runtime.InteropServices;
using Ceres.Chess.Textual;

#endregion

namespace Ceres.Chess
{
  /// <summary>
  /// Represents a move and the associated position from which it was made.
  /// </summary>
  [StructLayout(LayoutKind.Sequential, Pack = 1)]
  [Serializable]
  public readonly struct PositionWithMove
  {
    /// <summary>
    /// The starting position.
    /// </summary>
    public readonly Position Position;

    /// <summary>
    /// The move made.
    /// </summary>
    public readonly Move Move;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="move"></param>
    /// <param name="position"></param>
    public PositionWithMove(Position position, Move move)
    {
      Move = move;
      Position = position;
    }


    /// <summary>
    /// Deconstructor to tuple.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="move"></param>
    public void Deconstruct(out Position position, out Move move)
    {
      position = Position;
      move = Move;
    }


    /// <summary>
    /// Constructor from tuple.
    /// </summary>
    /// <param name="value"></param>
    public static implicit operator PositionWithMove((Position position, Move move) value)  => new (value.position, value.move);


    /// <summary>
    /// Returns the PositionWithMove corresponding to specified starting position FEN and move in SAN format.
    /// </summary>
    /// <param name="fen"></param>
    /// <param name="san"></param>
    /// <returns></returns>
    public static PositionWithMove FromFENAndSAN(string fen, string san)
    {
      Position position = Position.FromFEN(fen);
      Move move = Move.FromSAN(in position, san);
      return (position, move);
    }

    /// <summary>
    /// Returns SAN string corresponding to the position from this move.
    /// </summary>
    public string ToSAN => SANGenerator.ToSANMove(Position, Move);

    /// <summary>
    /// Returns string description.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<Move {Move.ToString() } from { Position }>";
    }
  }

}


