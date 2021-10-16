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

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCTS.MTCSNodes.Storage;

#endregion

namespace Ceres.MCTS.MTCSNodes.Struct
{
  /// <summary>
  /// Represents a child node, and can be in one of two states:
  ///   - an unexpanded child (leaf), we store the output of the NN (move and P)
  ///   - an expanded child (inner node), we store the index of the child
  ///   
  /// Data layout:
  ///   - we exploit fact that P cannot be negative for any valid policy value. 
  ///     thus turning on the sign bit of P becomes our flag if the node is actually expanded as an index
  ///     
  /// </summary>
  [Serializable]
  [StructLayout(LayoutKind.Explicit, Size = 4, Pack = 1)]
  public struct MCTSNodeStructChild
  {
    [FieldOffset(0)] int childIndex;

    [FieldOffset(0)] internal ushort lc0PositionMoveRawValue;
    [FieldOffset(2)] internal FP16 p;

    public const int SIZE_BYTES = 4;

    public readonly bool IsNull => childIndex == 0;

    // Because the uppermost bit is used to distinguish 
    // between child index (if 1) or probability, we can only use positive values
    public const int MaxChildIndex = int.MaxValue;


    /// <summary>
    /// If the child is expanded (and therefore we are storing the child index)
    /// </summary>
    public readonly bool IsExpanded
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get => childIndex < 0;
    }


    /// <summary>
    /// The LZPositionMove (if the child is unexpanded)
    /// </summary>
    public readonly EncodedMove Move => IsExpanded ? ChildRef.PriorMove : new EncodedMove(lc0PositionMoveRawValue);

    public readonly int N => IsExpanded ? ChildRef.N : 0;
    public readonly double W => IsExpanded ? ChildRef.W : 0.0f;
    public readonly FP16 P => IsExpanded ? ChildRef.P : p;


    public readonly MCTSNodeStructIndex ChildIndex
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        Debug.Assert(IsExpanded);
        return new MCTSNodeStructIndex(-childIndex);
      }
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public readonly ref MCTSNodeStruct ChildRefFromStore(MCTSNodeStore store)
    {
      Debug.Assert(IsExpanded);
      return ref store.Nodes.nodes[ChildIndex.Index];
    }


    public readonly ref MCTSNodeStruct ChildRef
    {
      [MethodImpl(MethodImplOptions.AggressiveInlining)]
      get
      {
        Debug.Assert(IsExpanded);
        return ref MCTSNodeStoreContext.Nodes[ChildIndex.Index];
      }
    }


    public void SetExpandedChildIndex(MCTSNodeStructIndex childIndex)
    {
      Debug.Assert(childIndex.Index != 0);
      Debug.Assert(childIndex.Index <= MaxChildIndex);

      this.childIndex = -childIndex.Index;
    }


    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public void SetUnexpandedPolicyValues(EncodedMove move, FP16 p)
    {
      Debug.Assert(move != default);

      this.p = p;
      this.lc0PositionMoveRawValue = (ushort)move.RawValue;
    }


    public readonly override string ToString()
    {
      string detail;
      if (IsExpanded)
      {
        detail = "-->#" + ChildIndex;
      }
      else
      {
        detail = $"P={P*100.0f,6:F2},Move={Move.AlgebraicStr}";
      }

      return "<MCTSNodeStructChild " + detail + ">";
    }
  }
}
