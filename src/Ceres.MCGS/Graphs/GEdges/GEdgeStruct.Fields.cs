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
using Ceres.Base.DataType;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Utils;

#endregion

namespace Ceres.MCGS.Graphs.GEdges;
public static class GEdgeStructExtensions
{
  public static bool IsTerminal(this GEdgeStruct.EdgeType edge) => edge >= GEdgeStruct.EdgeType.TerminalEdgeDrawn;
}


/// <summary>
/// Represents an edge from a parent to a child node, tracking structural information and visit statistics.
/// 
/// Edges may possibly be in an unmaterialized state, in which case the child node has not yet been visited or created.
/// 
/// ??? We need to store the index of the start block of Children in the VisitToChild array
/// (because the Node contains only has room for one index value, and this value will be:
///   - initially the block index of the first child,
///   - converted to block index of first VisitToChild once visited at least once
///     (because the Node value index will have been modified)
/// </summary>
[Serializable]
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct GEdgeStruct
{
  /// <summary>
  /// Type of edge (uninitialized, normal parent/child, or terminal).
  /// 
  /// N.B. The ordering has significance (see IsTerminal()).
  /// </summary>
  public enum EdgeType : byte
  {
    /// <summary>
    /// Not net in use.
    /// </summary>
    Uninitialized = 0,

    /// <summary>
    /// Points to another node in the tree as a child.
    /// </summary>
    ChildEdge = 1,

    /// <summary>
    /// Edge leading to terminal position (a loss).
    /// Corresponding terminal position does not appear in graph.
    /// </summary>
    TerminalEdgeDrawn = 2,

    /// <summary>
    /// Edge leading to decisive (won/lost) position.
    /// Corresponding terminal position does not appear in graph.
    /// </summary>
    TerminalEdgeDecisive = 3
  }

  /// <summary>
  /// Flags enum combining multiple boolean flags into a single byte.
  /// </summary>
  [Flags]
  private enum GEdgeFlags : byte
  {
    None = 0,
    IsStale = 1,
    ChildNodeHasDrawKnownToExist = 2
  }


  #region Constants

  #endregion

  #region Fields from parent's policy head

  /// <summary>
  /// Policy probability from policy head  as a fraction.
  /// </summary>
  public FP16 P;

  /// <summary>
  /// Packed struct storing move (14 bits) + type (2 bits).
  /// </summary>
  private Packed16As14_2 PackedMoveAndType;

  #endregion


  #region Search accumulation values

  /// <summary>
  /// The mostly recently refreshed copy of the Q of the child node 
  /// (from the perspective of the child).
  /// </summary>
  double qChild;

  /// <summary>
  /// The mostly recently refreshed copy of the Q of the child node 
  /// (from the perspective of the child).
  /// </summary>
  public double QChild
  {
    readonly get => qChild;
    set
    {
      Debug.Assert(!double.IsNaN(value) && !double.IsInfinity(value));
      qChild = value;
    }
  }

  /// <summary>
  /// Number of visits to this child.
  /// </summary>
  internal int n;

  /// <summary>
  /// Number of visits to this child.
  /// </summary>
  public int N
  {
    readonly get => n;
    set
    {
      Debug.Assert(value >= 0);
      n = value;
    }
  }

  /// <summary>
  /// Number of visits which were due to draw by repetition
  /// (these numbers are also included in N).
  /// </summary>
  internal int nDrawByRepetition;

  /// <summary>
  /// Number of visits which were due to draw by repetition
  /// (these numbers are also included in N).
  /// </summary>
  public int NDrawByRepetition
  {
    readonly get => nDrawByRepetition;
    set
    {
      Debug.Assert(value >= 0);
      nDrawByRepetition = value;
    }
  }


  /// <summary>
  /// Number of visits currently in flight from iterator 0.
  /// </summary>
  public ushort NumInFlight0;

  /// <summary>
  /// Number of visits currently in flight from iterator 1.
  /// </summary>
  public ushort NumInFlight1;

  #endregion


  #region Pointers to related data

  /// <summary>
  /// Index of child node to which this edge leads.
  /// </summary>
  public NodeIndex ChildNodeIndex;

  #endregion


  /// <summary>
  /// Pair of value and policy uncertainties, compressed.
  /// </summary>
  private FloatPairCompressedToByte Uncertainty;

  /// <summary>
  /// Combined flags for IsStale and ChildNodeHasDrawKnownToExist.
  /// </summary>
  private GEdgeFlags Flags;

  #region Accessors for packed fields

  /// <summary>
  /// Played move (lower 13 bits of <see cref="PackedMoveTypeBool"/>).
  /// </summary>
  public EncodedMove Move
  {
    readonly get => new(PackedMoveAndType.Value14BitsUShort);
    set => PackedMoveAndType.Value14BitsUShort = (ushort)value.IndexPacked;
  }

  public EdgeType Type
  {
    readonly get => (EdgeType)PackedMoveAndType.Value2BitsByte;
    set => PackedMoveAndType.Value2BitsByte = (byte)value;
  }

  /// <summary>
  /// Value uncertainty.
  /// </summary>
  public float UncertaintyV
  {
    readonly get => Uncertainty.V1;
    set => Uncertainty.V1 = value;
  }

  /// <summary>
  /// Policy uncertainty.
  /// </summary>
  public float UncertaintyP
  {
    readonly get => Uncertainty.V2;
    set => Uncertainty.V2 = value;
  }

  /// <summary>
  /// Sets both uncertainty values (value and policy) efficiently in one operation.
  /// Values are automatically clamped to the valid range.
  /// </summary>
  /// <param name="uncertaintyValue">Value uncertainty</param>
  /// <param name="uncertaintyPolicy">Policy uncertainty</param>
  public void SetUncertaintyValues(float uncertaintyValue, float uncertaintyPolicy)
  {
    Uncertainty = new FloatPairCompressedToByte(FloatPairCompressedToByte.Clamped(uncertaintyValue),
                                                FloatPairCompressedToByte.Clamped(uncertaintyPolicy));
  }


  public RunningStdDevShort StdDevEstimate;


  /// <summary>
  /// If the child node has DrawKnownToExist set to true.
  /// </summary>
  public bool ChildNodeHasDrawKnownToExist
  {
    readonly get => (Flags & GEdgeFlags.ChildNodeHasDrawKnownToExist) != 0;
    set
    {
      if (value)
      {
        Flags |= GEdgeFlags.ChildNodeHasDrawKnownToExist;
      }
      else
      {
        Flags &= ~GEdgeFlags.ChildNodeHasDrawKnownToExist;
      }
    }
  }


  /// <summary>
  /// If the child Q is known to have changed since 
  /// this edge Q was last updated and propagated to the parent Q.
  /// </summary>
  public bool IsStale
  {
    readonly get => (Flags & GEdgeFlags.IsStale) != 0;
    set
    {
      if (value)
      {
        Flags |= GEdgeFlags.IsStale;
      }
      else
      {
        Flags &= ~GEdgeFlags.IsStale;
      }
    }
  }

  #endregion


  #region Dummy fields (currently no space for actual data; values live in edge headers)

  /// <summary>
  /// Returns action head value estimate (no storage in edge struct; read from edge header instead).
  /// </summary>
  public FP16 ActionV
  {
    readonly get => FP16.NaN;
    set { }
  }


  /// <summary>
  /// Returns action head uncertainty estimate (no storage in edge struct; read from edge header instead).
  /// </summary>
  public FP16 ActionU
  {
    readonly get => FP16.NaN;
    set { }
  }

  #endregion


  /// <summary>
  /// Aggregate Q which combines visits to child and any possible draw by repetition visits).
  /// (the NDrawByRepetition count is used to dilute child Q 
  /// proportionally to the draw visits treating draws as 0).
  /// </summary>
  public readonly double Q => NDrawByRepetition == 0 ? QChild : QChild * ((double)(N - NDrawByRepetition) / N);


  /// <summary>
  /// Returns string representation.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"<GEdgeStruct [{(ChildNodeIndex.IsNull ? "(none)" : ChildNodeIndex.ToString())}]" +
      $" {Math.Round(100f * (float)P, 3)}%  Move={Move,6}  N={N:N0} ND={NDrawByRepetition:N0} QChild={QChild,6:F3}  UV={UncertaintyV,6:F3} UP={UncertaintyP,6:F3} NInFlight1={NumInFlight0,3:N0} NInFlight2={NumInFlight1,3:N0}>";
//    $" {Math.Round(100f * (float)P, 3)}%  Move={Move,6}  A={ActionV,6:F3} AU={ActionU,6:F3} N={N:N0} ND={NDrawByRepetition:N0} Q={Q,6:F3}  QChild={QChild,6:F3}  UV={UncertaintyV,6:F3} UP={UncertaintyP,6:F3} NInFlight1={NumInFlight0,3:N0} NInFlight2={NumInFlight1,3:N0}>";
  }



  /// <summary>
  /// Perform various integrity checks on structure layout (debug mode only).
  /// </summary>
  [Conditional("DEBUG")]
  internal static void ValidateEdgeStruct()
  {
    // Verify expected size
    long size = Marshal.SizeOf<GEdgeStruct>();
    if (size != 32)
    {
      throw new Exception("Internal error, wrong size GEdgeStruct " + size);
    }
  }

}
