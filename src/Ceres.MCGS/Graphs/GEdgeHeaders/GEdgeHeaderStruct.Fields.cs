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
using System.Runtime.InteropServices;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions.Basic;

#endregion

namespace Ceres.MCGS.Graphs.GEdgeHeaders;

/// <summary>
/// Structure to hold either:
///   - (for unexpanded nodes) the raw per-move policy/action outputs from the neural network.
///   - (for expanded nodes) a pointer to the child edge information (index in the child edge store).
///   
/// The leftmost bit of the first field  is used to indicate if this represents unexpanded or expanded.
/// </summary>
[Serializable]
[StructLayout(LayoutKind.Explicit, Pack = 1)]
public struct GEdgeHeaderStruct : IEquatable<GEdgeHeaderStruct>
{
  #region Fields if unexpanded

  // First two fields overlap with edgeBlockIndex, used only when edge not yet allocated
  [FieldOffset(0)] private FP16 p; // N.B. must appear at same offset as edgeBlockIndex
  [FieldOffset(2)] private ushort encodedMoveRawValue;

#if ACTION_ENABLED
  [FieldOffset(4)] private FP16 actionV;
  [FieldOffset(6)] private FP16 actionU;
#endif

  #endregion

  #region Fields if expanded

  [FieldOffset(0)] private int edgeBlockIndex; // Stored in negative form if in use (IsExpanded)

  #endregion


  /// <summary>
  /// Reads the raw FP16 at offset 0 (the P field) without checking IsExpanded.
  /// </summary>
  internal readonly FP16 RawP => p;

     
  // Because the uppermost bit is used to distinguish 
  // between child index (if 1) or probability, we can only use positive values
  public const int MaxChildIndex = int.MaxValue;

  /// <summary>
  /// If the child is expanded (and therefore we are storing the child index)
  /// </summary>
  public readonly bool IsExpanded => edgeBlockIndex < 0;

  public readonly bool IsUnintialized => edgeBlockIndex == 0;


  /// <summary>
  /// Neural network policy value.
  /// </summary>
  public readonly FP16 P
  {
    [DebuggerStepThrough]
    get
    {
      Debug.Assert(!IsExpanded);
      return p;
    }
  }

  /// <summary>
  /// Neural network move encoding.
  /// </summary>
  public readonly EncodedMove Move
  {
    [DebuggerStepThrough]
    get
    {
      Debug.Assert(!IsExpanded);
      return new EncodedMove(encodedMoveRawValue);
    }
  }

  /// <summary>
  /// Index of the block within the EdgeStore containing the edge body.
  /// </summary>
  public readonly int EdgeStoreBlockIndex
  {
    [DebuggerStepThrough]
    get
    {
      Debug.Assert(IsExpanded);
      return -edgeBlockIndex;
    }
  }

#if ACTION_ENABLED

  /// <summary>
  /// Action head value.
  /// </summary>
  public readonly FP16 ActionV
  {
    [DebuggerStepThrough]
    get
    {
      return actionV;
    }
  }

  /// <summary>
  /// Action head value uncertainty.
  /// </summary>
  public readonly FP16 ActionU
  {
    [DebuggerStepThrough]
    get
    {
      return actionU;
    }
  }

#endif


  /// <summary>
  /// Constructor for unexpanded MoveInfo having specified move and probability.
  /// </summary>
  /// <param name="move"></param>
  /// <param name="p"></param>
  /// <param name="actionV"></param>
  /// <param name="actionU"></param>
  public GEdgeHeaderStruct(EncodedMove move, FP16 p, FP16 actionV, FP16 actionU)
  {
    // When using the unexpanded fields, edgeBodyStructIndex is not used.
    // Note: Since this is an explicit layout, C# requires we initialize both sets of fields.
    edgeBlockIndex = 0;

    this.p = p;
    encodedMoveRawValue = move.RawValue;

#if ACTION_ENABLED
    this.actionV = actionV;
    this.actionU = actionU;
#endif
  }


  /// <summary>
  /// Sets the struct field values for an unexpanded edge header.
  /// </summary>
  /// <param name="move"></param>
  /// <param name="p"></param>
  /// <param name="actionV"></param>
  /// <param name="actionU"></param>
  internal void SetUnexpandedValues(EncodedMove move, FP16 p, FP16 actionV, FP16 actionU)
  {
    Debug.Assert(IsUnintialized);

    this.p = p;
    encodedMoveRawValue = move.RawValue;

#if ACTION_ENABLED
    this.actionV = actionV;
    this.actionU = actionU;
#endif
  }


  /// <summary>
  /// Converts the struct to represent an expanded child with specified edge block.
  /// </summary>
  /// <param name="edgeBlockIndex"></param>
   internal void SetAsExpandedToEdgeBlock(int edgeBlockIndex)
   {
     Debug.Assert(!IsUnintialized);
     Debug.Assert(!IsExpanded);
     this.edgeBlockIndex = -edgeBlockIndex;
   }


  /// <summary>
  /// Forces the edge block index to a new value without assertions.
  /// Used during graph compaction when edge blocks are relocated.
  /// </summary>
  internal void ForceSetEdgeBlockIndex(int edgeBlockIndex) => this.edgeBlockIndex = -edgeBlockIndex;


  /// <summary>
  /// Returns string representation of GEdgeHeaderStruct.
  /// </summary>
  /// <returns></returns>
  public readonly override string ToString()
  {
    string detail;

#if ACTION_ENABLED
    if (IsExpanded)
    {
      detail = $"Edge block={EdgeStoreBlockIndex}, AV={(float)ActionV,6:F2},AU={(float)ActionU,6:F2}";
    }
    else
    {
      detail = $"Move={Move.AlgebraicStr},P={P * 100.0f,6:F2},AV={(float)ActionV,6:F2},AU={(float)ActionU,6:F2}";
    }
#else
    if (IsExpanded)
    {
      detail = $"Edge block={EdgeStoreBlockIndex}";
    }
    else
    {
      detail = $"Move={Move.AlgebraicStr},P={P * 100.0f,6:F2}";
    }
#endif
    return "<GEdgeHeaderStruct " + detail + ">";
  }


  #region Struct overrides

  public bool Equals(GEdgeHeaderStruct other)
  {
#if ACTION_ENABLED
      return p.Equals(other.p) &&
             encodedMoveRawValue == other.encodedMoveRawValue &&
             actionV.Equals(other.actionV) &&
             actionU.Equals(other.actionU);      
#else
    return p.Equals(other.p) && encodedMoveRawValue == other.encodedMoveRawValue;
#endif
  }


  public override int GetHashCode() => edgeBlockIndex;

  public override bool Equals(object obj) => obj is GEdgeHeaderStruct other && Equals(other);

  public static bool operator ==(GEdgeHeaderStruct left, GEdgeHeaderStruct right) => left.Equals(right);

  public static bool operator !=(GEdgeHeaderStruct left, GEdgeHeaderStruct right) => !left.Equals(right);

  #endregion
}
