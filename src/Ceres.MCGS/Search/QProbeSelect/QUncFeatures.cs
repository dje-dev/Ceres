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
using System.Numerics;

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Search.QProbeSelect;


/// <summary>
/// The raw per-(child node, parent edge, parent) capture values consumed by the
/// Q-uncertainty feature row, with field types IDENTICAL to the TestSuite harvest
/// SnapshotRec so that float rounding behavior is reproduced exactly. The TestSuite
/// exporter (QProbeTrainData.FillFeaturesInto) delegates its formulas to
/// QUncFeatures.FillFeatures via this struct, making train/serve formula skew
/// structurally impossible; the remaining capture-cast surface is guarded by the
/// qprobe tt6 bit-exactness gate.
/// </summary>
public struct QUncRawSnapshot
{
  public uint N0;
  public float QPure;
  public float D;
  public float V;
  public float WinP;
  public float LossP;
  public float MLeft;
  public float UncertaintyV;
  public float UncertaintyP;
  public float LeafVolatilityDebiased;
  public float QTrendEWDebiased;
  public float RepDrawFraction;
  public byte NumPolicyMoves;
  public byte NumEdgesExpanded;
  public ushort NumParentsInGraph;
  public ushort DepthFromSearchRoot;

  public float EdgeP;
  public byte ChildIndexInParent;
  public uint EdgeN;
  public float EdgeQ;
  public float EdgeUncertaintyV;
  public float EdgeUncertaintyP;

  public uint ParentN;
  public float ParentQPure;
  public float CPUCTAtParent;
  public float BestSiblingQGap;
  public byte MilestoneBand;
  public bool DrawKnownBelow;
}


/// <summary>
/// Live extraction of the 38 Q-uncertainty model input features, mirroring the
/// harvest capture (TestSuite TrajectoryHarvester.TakeSnapshot) and the training
/// feature construction (TestSuite QProbeTrainData.FillFeaturesInto) bit-exactly.
/// Reads only STORED node/edge fields (quiescent post-batch semantics; no in-flight
/// state), matching how training data was captured.
/// </summary>
public static class QUncFeatures
{
  public const int NUM_FEATURES = 38;


  /// <summary>
  /// Depth of the node below the SEARCH root via the tree-parent chain, guarded for
  /// live play: under graph reuse / transpositions the chain can bypass the search
  /// root and reach the GRAPH root (where reading TreeParentNodeIndex would
  /// access-violate - this crashed the predecessor integration's first tournament
  /// game). On escape (or a runaway chain) returns fallbackDepth, the selection-path
  /// depth, which equals the tree depth in the common case. The harvester always ran
  /// with graph root == search root, so its unguarded walk was safe AND this guarded
  /// walk reproduces it exactly there (verified by tt6 with fallbackDepth 0).
  /// </summary>
  public static int SafeDepthFromSearchRoot(GNode node, int fallbackDepth)
  {
    int depth = 0;
    GNode current = node;
    while (!current.IsSearchRoot)
    {
      if (current.IsGraphRoot || depth >= 511)
      {
        return fallbackDepth;
      }
      current = node.Graph[current.TreeParentNodeIndex];
      depth++;
    }
    return depth;
  }


  /// <summary>
  /// Captures the raw values for one (child node, parent edge, parent) triple with
  /// exactly the harvest's cast semantics. Per-parent constants (parentQPure,
  /// cpuctAtParent, bestSiblingQGap) are passed in so callers compute them once per
  /// parent gather with the exact harvest expressions:
  ///   parentQPure    = (float)parent.ComputeQPure()
  ///   cpuctAtParent  = (float)paramsSelect.CalcCPUCT(parent.IsSearchRoot, parent.N)
  ///   bestSiblingQGap: see ComputeSiblingBests / SiblingGapForChild.
  /// fallbackDepth is the node's selection-path depth (see SafeDepthFromSearchRoot).
  /// </summary>
  public static void Capture(GNode node, GEdge edge, GNode parent, byte childIndexInParent,
                             float parentQPure, float cpuctAtParent, float bestSiblingQGap,
                             bool trackVolatility, int fallbackDepth, out QUncRawSnapshot raw)
  {
    raw = new QUncRawSnapshot
    {
      N0 = (uint)node.N,
      QPure = (float)node.ComputeQPure(),
      D = (float)node.D,
      V = node.V,
      WinP = (float)node.WinP,
      LossP = (float)node.LossP,
      MLeft = node.NodeRef.M,
      UncertaintyV = (float)node.UncertaintyValue,
      UncertaintyP = (float)node.UncertaintyPolicy,
      LeafVolatilityDebiased = trackVolatility ? (float)node.LeafValueVolatilityDebiased : float.NaN,
      QTrendEWDebiased = trackVolatility ? (float)node.QTrendEWDebiased : float.NaN,
      RepDrawFraction = (float)node.RepDrawFraction,
      NumPolicyMoves = node.NumPolicyMoves,
      NumEdgesExpanded = node.NumEdgesExpanded,
      NumParentsInGraph = (ushort)Math.Min(node.NumParents, ushort.MaxValue),
      DepthFromSearchRoot = (ushort)Math.Clamp(SafeDepthFromSearchRoot(node, fallbackDepth), 0, ushort.MaxValue),

      EdgeP = (float)edge.P,
      ChildIndexInParent = childIndexInParent,
      EdgeN = (uint)edge.N,
      EdgeQ = (float)edge.Q,
      EdgeUncertaintyV = edge.UncertaintyV,
      EdgeUncertaintyP = edge.UncertaintyP,

      ParentN = (uint)parent.N,
      ParentQPure = parentQPure,
      CPUCTAtParent = cpuctAtParent,
      BestSiblingQGap = bestSiblingQGap,
      MilestoneBand = (byte)BitOperations.Log2(Math.Max(1u, (uint)node.N)),
      DrawKnownBelow = node.DrawKnownToExistAmongChildren,
    };
  }


  /// <summary>
  /// Fills one 38-feature row from the raw capture (NaN-indicator flags recorded
  /// first, then NaN-to-zero fill). This is the single shared implementation of the
  /// feature formulas; the TestSuite exporter delegates here.
  /// </summary>
  public static void FillFeatures(in QUncRawSnapshot r, Span<float> dest)
  {
    float sigmaHat = r.LeafVolatilityDebiased / (float)Math.Sqrt(r.N0 + 1.0);

    dest[0] = r.N0;
    dest[1] = (float)Math.Log2(Math.Max(1, r.N0));
    dest[2] = r.QPure;
    dest[3] = r.D;
    dest[4] = r.V;
    dest[5] = r.WinP;
    dest[6] = r.LossP;
    dest[7] = r.MLeft;
    dest[8] = r.UncertaintyV;
    dest[9] = r.UncertaintyP;
    dest[10] = r.LeafVolatilityDebiased;
    dest[11] = sigmaHat;
    dest[12] = r.QTrendEWDebiased;
    dest[13] = r.RepDrawFraction;
    dest[14] = r.NumPolicyMoves;
    dest[15] = r.NumEdgesExpanded;
    dest[16] = r.NumParentsInGraph;
    dest[17] = r.DepthFromSearchRoot;
    dest[18] = r.V - r.QPure;

    dest[19] = r.EdgeP;
    dest[20] = r.ChildIndexInParent;
    dest[21] = r.EdgeN;
    dest[22] = r.ParentN > 0 ? (float)r.EdgeN / r.ParentN : 0;
    dest[23] = r.EdgeQ;
    dest[24] = r.EdgeUncertaintyV;
    dest[25] = r.EdgeUncertaintyP;

    dest[26] = r.ParentN;
    dest[27] = (float)Math.Log2(Math.Max(1, r.ParentN));
    dest[28] = r.ParentQPure;
    dest[29] = r.CPUCTAtParent;
    dest[30] = r.BestSiblingQGap;
    dest[31] = r.ParentQPure + r.EdgeQ; // parent QPure minus child value from parent perspective

    dest[32] = r.MilestoneBand;
    dest[33] = r.DrawKnownBelow ? 1 : 0;
    dest[34] = float.IsNaN(r.UncertaintyV) || float.IsNaN(r.UncertaintyP) ? 1 : 0;
    dest[35] = float.IsNaN(r.LeafVolatilityDebiased) ? 1 : 0;
    dest[36] = float.IsNaN(r.QTrendEWDebiased) ? 1 : 0;
    dest[37] = float.IsNaN(r.BestSiblingQGap) ? 1 : 0;

    for (int f = 0; f < NUM_FEATURES; f++)
    {
      if (float.IsNaN(dest[f]))
      {
        dest[f] = 0;
      }
    }
  }


  /// <summary>
  /// Single scan of a parent's expanded edges (no edge-type filter, matching the
  /// harvest's ScanParentEdges) returning the best and second-best visited
  /// parent-perspective edge Q and the index of the best (first strict maximum).
  /// bestIndex is -1 when no edge is visited; secondBestQ is negative infinity when
  /// fewer than two are visited. Feeds both the per-child BestSiblingQGap
  /// (SiblingGapForChild) and the M2 expected-improvement qBest/second-best.
  /// </summary>
  public static void ComputeSiblingBests(GNode parent, out int bestIndex, out double bestQ, out double secondBestQ)
  {
    int numExpanded = parent.NumEdgesExpanded;
    bestIndex = -1;
    bestQ = double.NegativeInfinity;
    secondBestQ = double.NegativeInfinity;

    for (int i = 0; i < numExpanded; i++)
    {
      GEdge edge = parent.ChildEdgeAtIndex(i);
      if (edge.N > 0)
      {
        double parentPerspQ = -edge.Q;
        if (parentPerspQ > bestQ)
        {
          secondBestQ = bestQ;
          bestQ = parentPerspQ;
          bestIndex = i;
        }
        else if (parentPerspQ > secondBestQ)
        {
          secondBestQ = parentPerspQ;
        }
      }
    }
  }


  /// <summary>
  /// Per-child BestSiblingQGap from the ComputeSiblingBests results, replicating the
  /// harvest expression (float)(bestOtherQ - (-selfEdgeQ)) where selfEdgeQ is the
  /// already-float-rounded (float)edge.Q of the child itself. NaN when no OTHER
  /// visited sibling exists (or selfEdgeQ is NaN).
  /// </summary>
  public static float SiblingGapForChild(int childIndex, float selfEdgeQ, int bestIndex, double bestQ, double secondBestQ)
  {
    double bestOtherQ = childIndex == bestIndex ? secondBestQ : bestQ;
    bool haveOther = childIndex == bestIndex ? !double.IsNegativeInfinity(secondBestQ) : bestIndex >= 0;
    return (haveOther && !float.IsNaN(selfEdgeQ)) ? (float)(bestOtherQ - (-selfEdgeQ)) : float.NaN;
  }
}
