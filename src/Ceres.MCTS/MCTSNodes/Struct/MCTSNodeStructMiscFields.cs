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

using System.Diagnostics;
using System.Runtime.CompilerServices;
using Ceres.Base.DataType;
using Ceres.Chess;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  /// <summary>
  /// Miscellaneous collection of infrequently accessed fields in structure,
  /// packed into an In32 for storage compactness.
  /// </summary>
  internal struct MCTSNodeStructMiscFields
  {
    private uint bits;

    const int BIT_LENGTH_TERMINAL = 2;
    const int BIT_LENGTH_DRAW_KNOWN_EXIST = 1;
    const int BIT_LENGTH_M_POSITION = 8;
    const int BIT_LENGTH_IS_OLD_GENERATION = 1;
    const int BIT_LENGTH_TRANSPOSITION_UNLINK_INPROGRESS = 1;
    const int BIT_LENGTH_UNUSED = 19;

    const int BIT_INDEX_TERMINAL = 0;
    const int BIT_INDEX_DRAW_KNOWN_EXIST = BIT_INDEX_TERMINAL + BIT_LENGTH_TERMINAL;
    const int BIT_INDEX_M_POSITION = BIT_INDEX_DRAW_KNOWN_EXIST + BIT_LENGTH_DRAW_KNOWN_EXIST;
    const int BIT_INDEX_IS_OLD_GENERATION = BIT_INDEX_M_POSITION + BIT_LENGTH_M_POSITION;
    const int BIT_INDEX_TRANSPOSITION_UNLINK_INPROGRESS = BIT_INDEX_IS_OLD_GENERATION + BIT_LENGTH_TRANSPOSITION_UNLINK_INPROGRESS;


    public void Clear() => bits = 0;


    /// <summary>
    /// Terminal status of the node.
    /// </summary>
    internal GameResult Terminal
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return (GameResult)BitUtils.ExtractRange(bits, BIT_INDEX_TERMINAL, BIT_LENGTH_TERMINAL);
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        Debug.Assert((int)value < 2 << BIT_LENGTH_TERMINAL);
        BitUtils.SetRange(ref bits, BIT_INDEX_TERMINAL, BIT_LENGTH_TERMINAL, (uint)value);
      }
    }


    /// <summary>
    /// If a draw has been shown to exist among the children.
    /// </summary>
    internal bool DrawKnownToExistAmongChildren
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return BitUtils.HasFlag(bits, BIT_INDEX_DRAW_KNOWN_EXIST);
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        BitUtils.SetFlag(ref bits, BIT_INDEX_DRAW_KNOWN_EXIST, value);
      }
    }


    /// <summary>
    /// Moves left head output for this position (rounded to a whole number).
    /// </summary>
    internal byte MPosition
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return (byte)BitUtils.ExtractRange(bits, BIT_INDEX_M_POSITION, BIT_LENGTH_M_POSITION);
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        Debug.Assert(value < 2 << BIT_LENGTH_M_POSITION);
        BitUtils.SetRange(ref bits, BIT_INDEX_M_POSITION, BIT_LENGTH_M_POSITION, (uint)value);
      }
    }


    /// <summary>
    /// If the node belonged to a prior seacrh tree but is 
    /// now unreachable due to a new root having been swapped into place.
    /// </summary>
    internal bool IsOldGeneration
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return BitUtils.ExtractRange(bits, BIT_INDEX_IS_OLD_GENERATION, BIT_LENGTH_IS_OLD_GENERATION) > 0;
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        BitUtils.SetRange(ref bits, BIT_INDEX_IS_OLD_GENERATION, BIT_LENGTH_IS_OLD_GENERATION, value ? (uint)1 : (uint)0);
      }
    }


    /// <summary>
    /// Internal synchronization variable indicating if the 
    /// node was transposition linked but is currently (very briefly) 
    /// in process of being unlinked.
    /// </summary>
    internal bool TranspositionUnlinkIsInProgress
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return BitUtils.ExtractRange(bits, BIT_INDEX_TRANSPOSITION_UNLINK_INPROGRESS, BIT_LENGTH_TRANSPOSITION_UNLINK_INPROGRESS) > 0;
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        BitUtils.SetRange(ref bits, BIT_INDEX_TRANSPOSITION_UNLINK_INPROGRESS, BIT_LENGTH_TRANSPOSITION_UNLINK_INPROGRESS, value ? (uint)1 : (uint)0);
      }
    }




    /// <summary>
    /// Returns string summary of record fields.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<MCTSNodeStructMiscFields Terminal: {Terminal}, DrawExist: {DrawKnownToExistAmongChildren}, M: {MPosition}, Gen#: {IsOldGeneration}";
    }

  }
}