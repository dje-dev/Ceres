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
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GParents;
using Ceres.MCGS.Utils;

#endregion

namespace Ceres.MCGS.Graphs.GNodes;

/// <summary>
/// Raw data fields appearing in MCGS node structure. 
/// 
/// Note that the structure size is exactly 64 bytes 
/// to optimize memory access efficiency (cache line alignment).
/// 
/// ************************************************************
/// N.B. Changes/additions to these fields may require update in
///      initialization logic in GNode or Graph, for example:
///        Graph.CopyChildValues
/// *************************************************************
/// </summary>
[Serializable]
[StructLayout(LayoutKind.Sequential, Pack = 1, Size = 64)]
public unsafe partial struct GNodeStruct
{
  #region Constants (scaling factors)

  /// <summary>
  /// Raw values from engine are scaled up by this amount to fit into byte (as a whole number).
  /// </summary>
  public const float UNCERTAINTY_SCALE = 100f;

  const byte MAX_M = 254; // 255 used to indicate not available

  #endregion


  #region Fields containing accumulated values during search

  /// <summary>
  /// Number of visits to self plus children.
  /// </summary>
  private int n;

  /// <summary>
  /// Number of visits to self plus children (see backing field documentation).
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
  /// All-inclusive win score for the node which is weighted average of:
  ///   - QPure (weighted average of children Q with the children N as weight, plus self-contribution from V), and
  ///   - sibling average Q (SiblingQ)
  /// where the weighting between the two is controlled by SiblingsQFrac.
  /// Note that the sibling fields are quantized to reduce structure size, but
  /// the QPure contribution remains in high precision and
  /// can always be recovered exactly from the other fields as needed.
  /// 
  /// Represented in high precision (double) to avoid "round to zero" update errors when W becomes large.
  /// </summary>
  private double q;

  /// <summary>
  /// All-inclusive win score for the node (see backing field documentation).
  /// </summary>
  public double Q
  {
    readonly get => q;
    set
    {
      Debug.Assert(!double.IsNaN(value) && !double.IsInfinity(value));
      q = value;
    }
  }

  /// <summary>
  /// Average of draw scores across self and all child visits.
  /// 
  /// NOTE: 
  ///   Children's D values are maintained via running average, refreshed on every on-path visit.
  ///   Heavily visited children are backed up nearly every batch — near-exact D.
  ///   Rarely visited children may have stale D but contribute proportionally tiny amounts to root D.
  ///   This is the same structural staleness that Q has for non-backed-up sibling edges (Q's delta-W
  ///   only refreshes the one edge being backed up; other edges retain stale edge.QChild values).
  ///   The error attenuates toward the root: high-N children are fresh, low-N children are negligible.
  /// </summary>
  public double D;

  /// <summary>
  /// Fortress probability metric stored as packed byte.
  /// Values 0-254 map to [0.0, 1.0], value 255 represents NaN.
  /// </summary>
  public byte FortressPByte;


  /// <summary>
  /// Number of children which have been expanded 
  /// (a corresponding node created in the tree).
  /// </summary>
  public byte NumEdgesExpanded;

  /// <summary>
  /// Set of packed miscellaneous fields.
  /// </summary>
  internal GNodeMiscFieldsStruct miscFields;

  #endregion


  #region Fields directly from neural network output or search initialization

  /// <summary>
  /// Win probability.
  /// </summary>
  public FP16 WinP;

  /// <summary>
  /// Loss probability.
  /// </summary>
  public FP16 LossP;

  /// <summary>
  /// Uncertainty associated with value score.
  /// </summary>
  public FP16 UncertaintyValue;

  /// <summary>
  /// Uncertainty associated with policy.
  /// </summary>
  public FP16 UncertaintyPolicy;

  /// <summary>
  /// Raw MLH (moves left) value (before any transformation).
  /// </summary>
  internal byte MRaw; // N.B. Get/set this thru MPosition

  /// <summary>
  /// Number of policy moves (children)
  /// Possibly this set of moves is incomplete due to either:
  ///   - implementation decision to "throw away" lowest probability moves to save storage, or
  ///   - error in policy evaluation which resulted in certain legal moves not being recognized
  /// </summary>
  public byte NumPolicyMoves;

  /// <summary>
  /// Hash value for the position (standalone, i.e. not inclusive of any history or Move50/repetition state).
  /// </summary>
  public PosHash64 HashStandalone;


  #endregion


  #region Fields with data structure pointers/indices

  /// <summary>
  /// Index of block in a move info store where GEdgeHeaderStruct sequence starts.
  /// (or zero if not yet initialized).
  /// </summary>
  internal EdgeHeaderBlockIndexOrNodeIndex edgeHeaderBlockIndexOrDeferredNode;



  /// <summary>
  /// Header pointer for parent information, representing either
  /// a direct pointer to the parent (if only a single one), 
  /// otherwise a pointer to the first block of entries with a linked list of parents.
  /// </summary>
  internal GParentsHeader ParentsHeader;

  #endregion

  #region Unused space for fields

  /// <summary>
  /// Percentage of Q to be contributed by the siblings Q.
  /// Encoded [0,1] into byte.
  /// </summary>
  byte siblingsQFrac;

  /// <summary>
  /// Average siblings Q.
  /// Encoded [-1.1,1.1] into ushort.
  /// 
  /// 32 bits is thought (hoped) to be sufficient precision because
  ///   - we can always back out exactly the same value that was applied to Q
  ///     because we store the (quantized) value that was used to compute it, 
  ///   - with large N, the pure Q part is computed and stored in high precision
  ///     so the impact of small (even single visit) updates can still be
  ///     capture and flow up the graph
  /// </summary>
  ushort siblingsQ;

  /// <summary>
  /// Fraction of Q to be contributed by the siblings Q.
  /// </summary>
  internal double SiblingsQFrac
  {
    readonly get => siblingsQFrac * (1.0 / 255.0);
    set => siblingsQFrac = (byte)Math.Round(value * 255.0);
  }

  /// <summary>
  /// Average siblings Q.
  /// </summary>
  internal double SiblingsQ
  {
    readonly get => (siblingsQ / 29752.0) - 1.1;
    set => siblingsQ = (ushort)Math.Round((Math.Clamp(value, -1.1, 1.1) + 1.1) * 29752.0);
  }


  /// <summary>
  /// Fortress probability metric: minimum (1 - P(NEVER)) over all pawn squares.
  /// Low values indicate a pawn unlikely to ever move, suggesting fortress-like structure.
  /// Encoded: 0-254 maps to [0.0, 1.0], 255 represents NaN.
  /// </summary>
  public float FortressP
  {
    readonly get => FortressPByte == 255 ? float.NaN : FortressPByte * (1.0f / 254.0f);
    set => FortressPByte = float.IsNaN(value) ? (byte)255 : (byte)Math.Round(Math.Clamp(value, 0f, 1f) * 254.0f);
  }


  //    public NodeIndex NodeHashSibling;
  public RunningStdDevShort StdDevEstimate;
  //public readonly short Unused1;
  //public readonly short Unused2;
  //public int NDrawByRepetition;
  public short UnusedShort;
  //public byte UnusedByte;

  /// <summary>
  /// Lock for multithreaded synchronization.
  /// </summary>
  private SpinLockByte LockField;

  /// <summary>
  /// A by-ref wrapper for the LockField (to prevent silent copies).
  /// </summary>
  public ref SpinLockByte LockRef => ref LockField;

  #endregion
}
