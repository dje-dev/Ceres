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
    const int BIT_LENGTH_REUSE_GEN_NUM = 4;
    const int BIT_LENGTH_UNUSED = 17;

    const int BIT_INDEX_TERMINAL = 0;
    const int BIT_INDEX_DRAW_KNOWN_EXIST = BIT_INDEX_TERMINAL + BIT_LENGTH_TERMINAL;
    const int BIT_INDEX_M_POSITION = BIT_INDEX_DRAW_KNOWN_EXIST + BIT_LENGTH_DRAW_KNOWN_EXIST;
    const int BIT_INDEX_REUSE_GEN_NUM = BIT_INDEX_M_POSITION + BIT_LENGTH_M_POSITION;


    public void Clear() => bits = 0;


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


    internal byte ReuseGenerationNum
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        return (byte)BitUtils.ExtractRange(bits, BIT_INDEX_REUSE_GEN_NUM, BIT_LENGTH_REUSE_GEN_NUM);
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        Debug.Assert(value < 2 << BIT_LENGTH_REUSE_GEN_NUM);

        BitUtils.SetRange(ref bits, BIT_INDEX_REUSE_GEN_NUM, BIT_LENGTH_REUSE_GEN_NUM, (uint)value);
      }
    }


    /// <summary>
    /// Returns string summary of record fields.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<MCTSNodeStructMiscFields Terminal: {Terminal}, DrawExist: {DrawKnownToExistAmongChildren}, M: {MPosition}, Gen#: {ReuseGenerationNum}";
    }

  }
}