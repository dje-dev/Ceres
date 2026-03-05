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
using System.Runtime.CompilerServices;
using System.Text;

using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.Chess.Positions;

/// <summary>
/// Number of plies since the last move of the piece on the square.
/// 
/// The encoding uses values 1-255 where:
///   - 1 = square was involved in a move this ply (just moved)
///   - N = square was last involved in a move N plies ago
///   - 0 = invalid (never used in computed values; would incorrectly indicate "never moved")
/// </summary>
public record struct PlySinceLastMove
{
  /// <summary>
  /// Array of 64 bytes representing the ply count since last move for each square.
  /// </summary>
  public SquarePlySinceArray SquarePlySince;

  /// <summary>
  /// Inline array of 64 bytes for storing ply counts per square.
  /// </summary>
  [InlineArray(64)]
  public struct SquarePlySinceArray
  {
    private byte _element0;
  }


  /// <summary>
  /// Dumps a visual representation of the ply counts alongside the piece positions.
  /// </summary>
  /// <param name="posWithHistory">The position with history to display.</param>
  public readonly void Dump(PositionWithHistory posWithHistory)
  {
    Position pos = posWithHistory.FinalPosition;
    string fen = pos.FEN;

    Console.WriteLine(fen);
    Console.WriteLine();

    // Build both boards side by side
    StringBuilder sb = new StringBuilder();
    sb.AppendLine("  Ply Since Last Move          Pieces");
    sb.AppendLine("  -----------------            ------");

    for (int rank = 7; rank >= 0; rank--)
    {
      // Ply count board
      sb.Append($"{rank + 1} ");
      for (int file = 0; file < 8; file++)
      {
        int squareIndex = rank * 8 + file;
        byte plyCount = SquarePlySince[squareIndex];
        sb.Append($"{plyCount,3} ");
      }

      sb.Append("     ");

      // Piece board
      sb.Append($"{rank + 1} ");
      for (int file = 0; file < 8; file++)
      {
        Square square = Square.FromFileAndRank(file, rank);
        Piece piece = pos.PieceOnSquare(square);
        char pieceChar = GetPieceChar(piece);
        sb.Append($"  {pieceChar} ");
      }

      sb.AppendLine();
    }

    // File labels
    sb.Append("   ");
    for (int file = 0; file < 8; file++)
    {
      sb.Append($"  {(char)('a' + file)} ");
    }
    sb.Append("        ");
    for (int file = 0; file < 8; file++)
    {
      sb.Append($"  {(char)('a' + file)} ");
    }
    sb.AppendLine();

    Console.WriteLine(sb.ToString());
  }

  /// <summary>
  /// Dumps a summary of distinct ply values and their counts.
  /// </summary>
  public readonly string SummaryStr()
  {
    Dictionary<byte, int> valueCounts = new Dictionary<byte, int>();

    for (int i = 0; i < 64; i++)
    {
      byte value = SquarePlySince[i];
      if (valueCounts.TryGetValue(value, out int count))
      {
        valueCounts[value] = count + 1;
      }
      else
      {
        valueCounts[value] = 1;
      }
    }

    StringBuilder sb = new StringBuilder();
    sb.Append("<PlySinceLastMoveArray with ");

    bool first = true;
    foreach (KeyValuePair<byte, int> kvp in valueCounts)
    {
      if (!first)
      {
        sb.Append(", ");
      }
      sb.Append($"{kvp.Value} squares @{kvp.Key}");
      first = false;
    }
    sb.Append(">");

    return sb.ToString();
  }


  /// <summary>
  /// Gets the character representation of a piece.
  /// </summary>
  private static char GetPieceChar(Piece piece)
  {
    if (piece.Type == PieceType.None)
    {
      return '.';
    }

    char pieceChar = piece.Type switch
    {
      PieceType.Pawn => 'P',
      PieceType.Knight => 'N',
      PieceType.Bishop => 'B',
      PieceType.Rook => 'R',
      PieceType.Queen => 'Q',
      PieceType.King => 'K',
      _ => '?'
    };

    return piece.Side == SideType.White ? pieceChar : char.ToLower(pieceChar);
  }


  #region ApplyMove Methods

  /// <summary>
  /// Applies a move to update the ply-since-last-move values.
  /// All squares are incremented by 1 (capped at 255), then the from/to squares
  /// of the move are set to 1 (indicating they were just involved in this move).
  /// </summary>
  /// <param name="current">Current ply-since values (64 bytes, will not be modified)</param>
  /// <param name="target">Target buffer for updated values (64 bytes)</param>
  /// <param name="move">The move being applied</param>
  public static void ApplyMove(ReadOnlySpan<byte> current, Span<byte> target, in MGMove move)
  {
    // Increment all squares by 1 (one more ply has passed).
    // The XOR 56 converts between square index conventions.
    for (int s = 0; s < 64; s++)
    {
      target[s ^ 56] = (byte)Math.Min(255, current[s] + 1);
    }

    // Set to 1 (not 0) because the move just happened this ply.
    // A value of 0 would incorrectly indicate "never moved" per the encoding scheme.
    target[move.FromSquare.SquareIndexStartA1 ^ 56] = 1;
    target[move.ToSquare.SquareIndexStartA1 ^ 56] = 1;
  }


  /// <summary>
  /// Applies a move in-place using two ping-pong buffers, swapping the references.
  /// After the call, current contains the updated values.
  /// </summary>
  /// <param name="current">Reference to the current buffer span</param>
  /// <param name="temp">Reference to the temporary buffer span</param>
  /// <param name="move">The move being applied</param>
  public static void ApplyMoveWithSwap(ref Span<byte> current, ref Span<byte> temp, in MGMove move)
  {
    ApplyMove(current, temp, in move);
    Span<byte> swap = current;
    current = temp;
    temp = swap;
  }


  /// <summary>
  /// Applies a move in-place using two ping-pong arrays, swapping the references.
  /// After the call, current contains the updated values.
  /// </summary>
  /// <param name="current">Reference to the current array</param>
  /// <param name="temp">Reference to the temporary array</param>
  /// <param name="move">The move being applied</param>
  public static void ApplyMoveWithSwap(ref byte[] current, ref byte[] temp, in MGMove move)
  {
    ApplyMove(current, temp, in move);
    byte[] swap = current;
    current = temp;
    temp = swap;
  }


  /// <summary>
  /// Applies a move in-place using two ping-pong PlySinceLastMove structs, swapping the references.
  /// After the call, current contains the updated values.
  /// </summary>
  /// <param name="current">Reference to the current PlySinceLastMove</param>
  /// <param name="temp">Reference to the temporary PlySinceLastMove</param>
  /// <param name="move">The move being applied</param>
  public static void ApplyMoveWithSwap(ref PlySinceLastMove current, ref PlySinceLastMove temp, in MGMove move)
  {
    ApplyMove(current.SquarePlySince, temp.SquarePlySince, in move);

    // Swap
    PlySinceLastMove swap = current;
    current = temp;
    temp = swap;
  }

  #endregion


  /// <summary>
  /// Returns a string that represents the current object.
  /// </summary>
  /// <returns>A string that represents the current object.</returns>
  public override readonly string ToString() => SummaryStr();
}