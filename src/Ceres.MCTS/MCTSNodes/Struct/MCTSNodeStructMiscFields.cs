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

    // N.B. Changes/additions to these fields may require upate in
    //      initialization logic in MCTSNode and also in MCTSNodeStructClone.TryCloneChild method.

    const int BIT_LENGTH_TERMINAL = 2;
    const int BIT_LENGTH_DRAW_KNOWN_EXIST = 1;
    const int BIT_LENGTH_CHECKMATE_KNOWN_EXIST = 1;
    const int BIT_LENGTH_IS_OLD_GENERATION = 1;
    const int BIT_LENGTH_TRANSPOSITION_UNLINK_INPROGRESS = 1;
    const int BIT_LENGTH_IS_TRANSPOSITION_ROOT = 1;
    const int BIT_LENGTH_INDEX_IN_PARENT = 6;
    const int BIT_LENGTH_HAS_REPETITIONS = 1;
    const int BIT_LENGTH_NUM_PIECES = 5;
    const int BIT_LENGTH_NUM_RANK2_PAWNS = 5;
    const int BIT_LENGTH_SECONDARY_NN = 1;
    const int BIT_LENGTH_TEST = 1;
    const int BIT_LENGTH_UNUSED = 6;

    const int BIT_INDEX_TERMINAL = 0;
    const int BIT_INDEX_DRAW_KNOWN_EXIST = BIT_INDEX_TERMINAL + BIT_LENGTH_TERMINAL;
    const int BIT_INDEX_CHECKMATE_KNOWN_EXIST = BIT_INDEX_DRAW_KNOWN_EXIST + BIT_LENGTH_DRAW_KNOWN_EXIST;
    const int BIT_INDEX_IS_OLD_GENERATION = BIT_INDEX_CHECKMATE_KNOWN_EXIST + BIT_LENGTH_CHECKMATE_KNOWN_EXIST;
    const int BIT_INDEX_TRANSPOSITION_UNLINK_INPROGRESS = BIT_INDEX_IS_OLD_GENERATION + BIT_LENGTH_IS_OLD_GENERATION;
    const int BIT_INDEX_IS_TRANSPOSITION_ROOT = BIT_INDEX_TRANSPOSITION_UNLINK_INPROGRESS + BIT_LENGTH_TRANSPOSITION_UNLINK_INPROGRESS;
    const int BIT_INDEX_INDEX_IN_PARENT = BIT_INDEX_IS_TRANSPOSITION_ROOT + BIT_LENGTH_IS_TRANSPOSITION_ROOT;
    const int BIT_INDEX_HAS_REPETITIONS = BIT_INDEX_INDEX_IN_PARENT + BIT_LENGTH_INDEX_IN_PARENT;

    const int BIT_INDEX_NUM_PIECES = BIT_INDEX_HAS_REPETITIONS + BIT_LENGTH_HAS_REPETITIONS;
    const int BIT_INDEX_NUM_RANK2_PAWNS = BIT_INDEX_NUM_PIECES + BIT_LENGTH_NUM_PIECES;

    const int BIT_INDEX_SECONDARY_NN = BIT_INDEX_NUM_RANK2_PAWNS + BIT_LENGTH_NUM_RANK2_PAWNS;
    const int BIT_INDEX_TEST = BIT_INDEX_SECONDARY_NN + BIT_LENGTH_SECONDARY_NN;

    public void Clear() => bits = 0;


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
        Debug.Assert((int)value < 2 << (BIT_LENGTH_TERMINAL - 1));
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
    /// If the node belonged to a prior seacrh tree but is 
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
        BitUtils.SetRange(ref bits, BIT_INDEX_IS_OLD_GENERATION, BIT_LENGTH_IS_OLD_GENERATION, value ? (uint)1 : (uint)0);
      }
    }


    /// <summary>
    /// If the node was successfully added to the transposition root dictionary
    /// as a transposition root.
    /// </summary>
    internal bool IsTranspositionRoot
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      readonly get
      {
        return BitUtils.ExtractRange(bits, BIT_INDEX_IS_TRANSPOSITION_ROOT, BIT_LENGTH_IS_TRANSPOSITION_ROOT) > 0;
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        BitUtils.SetRange(ref bits, BIT_INDEX_IS_TRANSPOSITION_ROOT, BIT_LENGTH_IS_TRANSPOSITION_ROOT, value ? (uint)1 : (uint)0);
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
      readonly get
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
    /// Index of the position in the child's policy array.
    /// </summary>
    internal byte IndexInParent
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      readonly get
      {
        return (byte)BitUtils.ExtractRange(bits, BIT_INDEX_INDEX_IN_PARENT, BIT_LENGTH_INDEX_IN_PARENT);
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        Debug.Assert(value < 2 << (BIT_LENGTH_INDEX_IN_PARENT - 1));
        BitUtils.SetRange(ref bits, BIT_INDEX_INDEX_IN_PARENT, BIT_LENGTH_INDEX_IN_PARENT, (uint)value);
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
        Debug.Assert(value >=2 && value <= 32);
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
        Debug.Assert(value < 2 << (BIT_LENGTH_NUM_RANK2_PAWNS - 1));
        BitUtils.SetRange(ref bits, BIT_INDEX_NUM_RANK2_PAWNS, BIT_LENGTH_NUM_RANK2_PAWNS, (uint)value);
      }
    }


    /// <summary>
    /// If the node was evaluated by the secondary (alternate) neural network.
    /// </summary>
    internal bool SecondaryNN
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      readonly get
      {
        return BitUtils.HasFlag(bits, BIT_INDEX_SECONDARY_NN);
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        BitUtils.SetFlag(ref bits, BIT_INDEX_SECONDARY_NN, value);
      }
    }


    /// <summary>
    /// Value of test flag (miscellaneous ad hoc tests).
    /// </summary>
    internal bool TestFlag
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      readonly get
      {
        return BitUtils.HasFlag(bits, BIT_INDEX_TEST);
      }

      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      set
      {
        BitUtils.SetFlag(ref bits, BIT_INDEX_TEST, value);
      }
    }



    /// <summary>
    /// Returns string summary of record fields.
    /// </summary>
    /// <returns></returns>
    public readonly override string ToString()
    {
      return $"<MCTSNodeStructMiscFields Terminal: {Terminal}, DrawExist: {DrawKnownToExistAmongChildren}, Gen#: {IsOldGeneration}";
    }

  }
}