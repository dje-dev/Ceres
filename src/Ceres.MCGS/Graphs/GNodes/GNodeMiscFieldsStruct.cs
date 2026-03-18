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

using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Ceres.Base.DataType;
using Ceres.Chess;
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Miscellaneous collection of infrequently accessed fields in structure,
/// packed into an Int32 for storage compactness.
/// </summary>

[Serializable]
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 4)]
internal struct GNodeMiscFieldsStruct
{
  /// <summary>
  /// Internal backing field for bit-packed fields.
  /// </summary>
  private uint bits;

  // ************************************************************
  // N.B. Changes/additions to these fields may require update in
  //      initialization logic in GNode or Graph, for example:
  //        Graph.CopyChildValues
  // ************************************************************
  const int BIT_LENGTH_TERMINAL = 2;
  const int BIT_LENGTH_DRAW_KNOWN_EXIST = 1;
  const int BIT_LENGTH_CHECKMATE_KNOWN_EXIST = 1;
  const int BIT_LENGTH_IS_OLD_GENERATION = 1;
  const int BIT_LENGTH_MOVE50_CATEGORY = 2; 
  const int BIT_LENGTH_UNUSED_6_BITS = 6;
  const int BIT_LENGTH_HAS_REPETITIONS = 1;
  const int BIT_LENGTH_NUM_PIECES = 5;
  const int BIT_LENGTH_NUM_RANK2_PAWNS = 5;
  const int BIT_LENGTH_IS_SEARCH_ROOT = 1;
  const int BIT_LENGTH_IS_WHITE = 1;
  const int BIT_LENGTH_DIRTY = 1;
  const int BIT_LENGTH_UNUSED_BOOL1 = 1;
  const int BIT_LENGTH_IS_GRAPH_ROOT = 1;
  const int BIT_LENGTH_UNUSED = 3;

  const int BIT_INDEX_TERMINAL = 0;
  const int BIT_INDEX_DRAW_KNOWN_EXIST = BIT_INDEX_TERMINAL + BIT_LENGTH_TERMINAL;
  const int BIT_INDEX_CHECKMATE_KNOWN_EXIST = BIT_INDEX_DRAW_KNOWN_EXIST + BIT_LENGTH_DRAW_KNOWN_EXIST;
  const int BIT_INDEX_IS_OLD_GENERATION = BIT_INDEX_CHECKMATE_KNOWN_EXIST + BIT_LENGTH_CHECKMATE_KNOWN_EXIST;
  const int BIT_INDEX_MOVE50_CATEGORY = BIT_INDEX_IS_OLD_GENERATION + BIT_LENGTH_IS_OLD_GENERATION; // Combined UNUSED1 and UNUSED2
  const int BIT_INDEX_UNUSED_6_BITS = BIT_INDEX_MOVE50_CATEGORY + BIT_LENGTH_MOVE50_CATEGORY;
  const int BIT_INDEX_HAS_REPETITIONS = BIT_INDEX_UNUSED_6_BITS + BIT_LENGTH_UNUSED_6_BITS;

  const int BIT_INDEX_NUM_PIECES = BIT_INDEX_HAS_REPETITIONS + BIT_LENGTH_HAS_REPETITIONS;
  const int BIT_INDEX_NUM_RANK2_PAWNS = BIT_INDEX_NUM_PIECES + BIT_LENGTH_NUM_PIECES;

  const int BIT_INDEX_IS_SEARCH_ROOT = BIT_INDEX_NUM_RANK2_PAWNS + BIT_LENGTH_NUM_RANK2_PAWNS;
  const int BIT_INDEX_IS_WHITE = BIT_INDEX_IS_SEARCH_ROOT + BIT_LENGTH_IS_SEARCH_ROOT;
  const int BIT_INDEX_DIRTY = BIT_INDEX_IS_WHITE + BIT_LENGTH_IS_WHITE;
  const int BIT_INDEX_IS_UNUSED_BOOL1 = BIT_INDEX_DIRTY + BIT_LENGTH_DIRTY;
  const int BIT_INDEX_IS_GRAPH_ROOT = BIT_INDEX_IS_UNUSED_BOOL1 + BIT_LENGTH_UNUSED_BOOL1;

  internal void Clear() => bits = 0;


  /// <summary>
  /// Terminal status of the node.
  /// </summary>
  internal GameResult Terminal
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return (GameResult)BitUtils.ExtractRange(bits, BIT_INDEX_TERMINAL, BIT_LENGTH_TERMINAL);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      Debug.Assert((int)value < 2 << BIT_LENGTH_TERMINAL - 1);
      BitUtils.SetRange(ref bits, BIT_INDEX_TERMINAL, BIT_LENGTH_TERMINAL, (uint)value);
    }
  }


  /// <summary>
  /// If a checkmate has been shown to exist among the children.
  /// </summary>
  internal bool CheckmateKnownToExistAmongChildren
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return BitUtils.HasFlag(bits, BIT_INDEX_CHECKMATE_KNOWN_EXIST);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      BitUtils.SetFlag(ref bits, BIT_INDEX_CHECKMATE_KNOWN_EXIST, value);
    }
  }


  /// <summary>
  /// If a draw has been shown to exist among the children.
  /// </summary>
  internal bool DrawKnownToExistAmongChildren
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
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
  /// If the node belonged to a prior seacrh graph but is 
  /// now unreachable due to a new root having been swapped into place.
  /// </summary>
  internal bool IsOldGeneration
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return BitUtils.ExtractRange(bits, BIT_INDEX_IS_OLD_GENERATION, BIT_LENGTH_IS_OLD_GENERATION) > 0;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      BitUtils.SetRange(ref bits, BIT_INDEX_IS_OLD_GENERATION, BIT_LENGTH_IS_OLD_GENERATION, value ? 1 : (uint)0);
    }
  }


  /// <summary>
  /// Category based on the move 50 counter value.
  /// </summary>
  internal Move50CategoryEnum Move50Category
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return (Move50CategoryEnum)BitUtils.ExtractRange(bits, BIT_INDEX_MOVE50_CATEGORY, BIT_LENGTH_MOVE50_CATEGORY);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      Debug.Assert((int)value < (1 << BIT_LENGTH_MOVE50_CATEGORY));
      BitUtils.SetRange(ref bits, BIT_INDEX_MOVE50_CATEGORY, BIT_LENGTH_MOVE50_CATEGORY, (uint)value);
    }
  }


  /// <summary>
  /// Not currently used.
  /// 
  /// 
  /// </summary>
  internal byte Unused6Bits
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return (byte)BitUtils.ExtractRange(bits, BIT_INDEX_UNUSED_6_BITS, BIT_LENGTH_UNUSED_6_BITS);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      Debug.Assert(value < 2 << BIT_LENGTH_UNUSED_6_BITS - 1);
      BitUtils.SetRange(ref bits, BIT_INDEX_UNUSED_6_BITS, BIT_LENGTH_UNUSED_6_BITS, value);
    }
  }


  /// <summary>
  /// If the position has one more repetitions in the history.
  /// </summary>
  internal bool HasRepetitions
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return BitUtils.HasFlag(bits, BIT_INDEX_HAS_REPETITIONS);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      BitUtils.SetFlag(ref bits, BIT_INDEX_HAS_REPETITIONS, value);
    }
  }


  /// <summary>
  /// Number of pieces on board.
  /// The value actually stored is N-1 so that we fit in the range [0..31].
  /// </summary>
  internal byte NumPieces
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return (byte)(1 + BitUtils.ExtractRange(bits, BIT_INDEX_NUM_PIECES, BIT_LENGTH_NUM_PIECES));
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      Debug.Assert(value >= 2 && value <= 32);
      BitUtils.SetRange(ref bits, BIT_INDEX_NUM_PIECES, BIT_LENGTH_NUM_PIECES, (uint)value - 1);
    }
  }


  /// <summary>
  /// Number of pawns still on second rank.
  /// </summary>
  internal byte NumRank2Pawns
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return (byte)BitUtils.ExtractRange(bits, BIT_INDEX_NUM_RANK2_PAWNS, BIT_LENGTH_NUM_RANK2_PAWNS);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      Debug.Assert(value <= 16);
      Debug.Assert(value < 2 << BIT_LENGTH_NUM_RANK2_PAWNS - 1);
      BitUtils.SetRange(ref bits, BIT_INDEX_NUM_RANK2_PAWNS, BIT_LENGTH_NUM_RANK2_PAWNS, value);
    }
  }


  /// <summary>
  /// If the node is the root of the search currently active.
  /// </summary>
  internal bool IsSearchRoot
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return BitUtils.HasFlag(bits, BIT_INDEX_IS_SEARCH_ROOT);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      BitUtils.SetFlag(ref bits, BIT_INDEX_IS_SEARCH_ROOT, value);
    }
  }


  /// <summary>
  /// If the node corresponds to a position with white to play.
  /// </summary>
  internal bool IsWhite
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return BitUtils.HasFlag(bits, BIT_INDEX_IS_WHITE);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      BitUtils.SetFlag(ref bits, BIT_INDEX_IS_WHITE, value);
    }
  }


  /// <summary>
  /// If the node has a child that was updated in a way
  /// that may not yet be reflected in the node statistics.
  /// </summary>
  internal bool IsDirty
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return BitUtils.HasFlag(bits, BIT_INDEX_DIRTY);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      BitUtils.SetFlag(ref bits, BIT_INDEX_DIRTY, value);
    }
  }


  /// <summary>
  /// UnusedBool1.
  /// </summary>
  internal bool UnusedBool1
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return BitUtils.HasFlag(bits, BIT_INDEX_IS_UNUSED_BOOL1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      BitUtils.SetFlag(ref bits, BIT_INDEX_IS_UNUSED_BOOL1, value);
    }
  }


  /// <summary>
  /// Bit indicating if the node is the root of the graph.
  /// </summary>
  internal bool IsGraphRoot
  {
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    readonly get
    {
      return BitUtils.HasFlag(bits, BIT_INDEX_IS_GRAPH_ROOT);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    set
    {
      BitUtils.SetFlag(ref bits, BIT_INDEX_IS_GRAPH_ROOT, value);
    }
  }


  /// <summary>
  /// Returns string summary of record fields.
  /// </summary>
  /// <returns></returns>
  public readonly override string ToString()
  {
    return $"<GNodeMiscFieldsStruct Terminal: {Terminal}, DrawExist: {DrawKnownToExistAmongChildren}, Gen#: {IsOldGeneration}";
  }

}
