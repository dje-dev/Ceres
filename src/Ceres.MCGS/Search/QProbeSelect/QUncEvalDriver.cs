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
using System.Numerics;
using System.Threading;

using Ceres.Chess;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PUCT;

#endregion

namespace Ceres.MCGS.Search.QProbeSelect;

/// <summary>
/// The select-time glue for the Q-uncertainty methods M1-M4: gating, per-child
/// forecast retrieval (cache-first, then ONE batched inference for all misses of the
/// parent), and computation of the per-child score bonuses (M1 sigma bonus, M2
/// expected improvement, M3 mu correction) and U-term multipliers (M4) that
/// PUCTScoreCalcVector applies. Children outside model coverage (unexpanded edges,
/// unvisited, terminal, unevaluated, checkmate-pinned or old-generation child nodes)
/// receive neutral adjustments and keep their stock PUCT scores.
///
/// Feature extraction reads only stored node/edge fields (quiescent semantics,
/// matching harvest capture); train/serve parity is guarded by the qprobe tt6 gate.
/// </summary>
internal static class QUncEvalDriver
{
  [ThreadStatic] static QUncScratch threadScratch;


  /// <summary>Per-thread working buffers (about 25 KB per thread at MAX_CHILDREN=64).</summary>
  sealed class QUncScratch
  {
    public readonly QProbeSelectScratch NNScratch = new QProbeSelectScratch();
    public float[] InputRows = Array.Empty<float>();
    public float[] MuOut = Array.Empty<float>();
    public float[] SigmaOut = Array.Empty<float>();
    public readonly float[] MuS = new float[PUCTScoreCalcVector.MAX_CHILDREN];
    public readonly float[] SigS = new float[PUCTScoreCalcVector.MAX_CHILDREN];
    public readonly float[] MuL = new float[PUCTScoreCalcVector.MAX_CHILDREN];
    public readonly float[] SigL = new float[PUCTScoreCalcVector.MAX_CHILDREN];
    public readonly double[] EdgeQ = new double[PUCTScoreCalcVector.MAX_CHILDREN];
    public readonly bool[] Covered = new bool[PUCTScoreCalcVector.MAX_CHILDREN];
    public readonly int[] MissChild = new int[PUCTScoreCalcVector.MAX_CHILDREN];

    // Adjustment arrays handed to PUCTScoreCalcVector. Oversized past MAX_CHILDREN so
    // SIMD full-vector loads at any block offset never run short; the padding tail is
    // permanently neutral (0 bonus / 1 multiplier) because only [0, numToProcess) is
    // ever written.
    public readonly double[] ScoreBonus = new double[PUCTScoreCalcVector.MAX_CHILDREN + 8];
    public readonly double[] UMultiplier = new double[PUCTScoreCalcVector.MAX_CHILDREN + 8];

    public QUncScratch()
    {
      Array.Fill(UMultiplier, 1.0);
    }

    public void EnsureRowsSized(int numFloats, int numRows)
    {
      if (InputRows.Length < numFloats)
      {
        InputRows = new float[numFloats];
      }
      if (MuOut.Length < numRows)
      {
        MuOut = new float[numRows];
        SigmaOut = new float[numRows];
      }
    }
  }


  /// <summary>
  /// Computes the per-child adjustments for one qualifying parent gather. Returns
  /// false (and null adjustments) when gated out or no child is model-covered.
  /// scoreBonus is non-null when any of M1/M2/M3 is active; uMultiplier is non-null
  /// when M4 applied this gather. Both arrays are per-thread scratch valid only
  /// until the next call on this thread.
  /// </summary>
  internal static bool PrepareAdjustments(QUncSelectContext context, GNode node, in GNodeStruct nodeRef,
                                          ParamsSelect paramsSelect, int numToProcess, int pathDepth,
                                          out double[] scoreBonus, out double[] uMultiplier)
  {
    scoreBonus = null;
    uMultiplier = null;
    QUncStats stats = context.Stats;

    if (nodeRef.N < context.MinParentN)
    {
      Interlocked.Increment(ref stats.ParentsElidedMinN);
      return false;
    }

    int numCandidates = Math.Min(Math.Min(numToProcess, nodeRef.NumEdgesExpanded), context.MaxChildren);
    if (numCandidates < 2 || nodeRef.CheckmateKnownToExistAmongChildren)
    {
      Interlocked.Increment(ref stats.ParentsElidedFast);
      return false;
    }

    long ticksScanStart = Stopwatch.GetTimestamp();

    QUncFeatures.ComputeSiblingBests(node, out int bestIndex, out double bestQ, out double secondBestQ);
    if (bestIndex < 0 || double.IsNegativeInfinity(secondBestQ))
    {
      Interlocked.Increment(ref stats.ParentsElidedFast);
      return false;
    }

    QUncScratch scratch = threadScratch ??= new QUncScratch();
    int generation = context.CurrentGeneration;
    int numMisses = 0;
    int numCovered = 0;
    int numCacheHits = 0;

    for (int i = 0; i < numCandidates; i++)
    {
      scratch.Covered[i] = false;
      GEdge edge = node.ChildEdgeAtIndex(i);
      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNode.IsNull || edge.N < 1)
      {
        continue;
      }
      GNode child = edge.ChildNode;
      if (!child.IsEvaluated || child.Terminal.IsTerminal()
       || child.CheckmateKnownToExistAmongChildren || child.IsOldGeneration)
      {
        continue;
      }

      scratch.EdgeQ[i] = edge.Q;
      if (context.Cache.TryGet(child.Index.Index, child.N, generation, context.NeedLongHorizon,
                               out float muS, out float sigS, out float muL, out float sigL))
      {
        scratch.MuS[i] = muS;
        scratch.SigS[i] = sigS;
        scratch.MuL[i] = muL;
        scratch.SigL[i] = sigL;
        scratch.Covered[i] = true;
        numCovered++;
        numCacheHits++;
      }
      else
      {
        scratch.MissChild[numMisses++] = i;
      }
    }

    Interlocked.Add(ref stats.FeatureTicks, Stopwatch.GetTimestamp() - ticksScanStart);

    if (numMisses > 0)
    {
      float parentQPure = (float)node.ComputeQPure();
      float cpuctAtParent = (float)paramsSelect.CalcCPUCT(node.IsSearchRoot, node.N);

      int rowsPerChild = context.NeedLongHorizon ? 2 : 1;
      int numRows = numMisses * rowsPerChild;
      int numInputs = context.Model.NumInputs;
      scratch.EnsureRowsSized(numRows * numInputs, numRows);

      float dnShortLog = MathF.Log(1f + context.DnShort);
      float dnShortScale = context.DnShort / 64f;
      float dnLongLog = MathF.Log(1f + context.DnLong);
      float dnLongScale = context.DnLong / 64f;

      long ticksFillStart = Stopwatch.GetTimestamp();
      for (int m = 0; m < numMisses; m++)
      {
        int i = scratch.MissChild[m];
        GEdge edge = node.ChildEdgeAtIndex(i);
        GNode child = edge.ChildNode;
        float gap = QUncFeatures.SiblingGapForChild(i, (float)edge.Q, bestIndex, bestQ, secondBestQ);
        QUncFeatures.Capture(child, edge, node, (byte)i, parentQPure, cpuctAtParent, gap,
                             context.TrackVolatility, pathDepth + 1, out QUncRawSnapshot raw);

        Span<float> row = scratch.InputRows.AsSpan(m * rowsPerChild * numInputs, numInputs);
        QUncFeatures.FillFeatures(in raw, row);
        row[numInputs - 2] = dnShortLog;
        row[numInputs - 1] = dnShortScale;
        if (rowsPerChild == 2)
        {
          Span<float> rowLong = scratch.InputRows.AsSpan((m * rowsPerChild + 1) * numInputs, numInputs);
          row.CopyTo(rowLong);
          rowLong[numInputs - 2] = dnLongLog;
          rowLong[numInputs - 1] = dnLongScale;
        }
      }

      long ticksEvalStart = Stopwatch.GetTimestamp();
      QProbeSelectNNEval.EvaluateBatch(context.Model, scratch.NNScratch, scratch.InputRows, numRows,
                                       scratch.MuOut, scratch.SigmaOut);
      long ticksEvalEnd = Stopwatch.GetTimestamp();
      Interlocked.Add(ref stats.FeatureTicks, ticksEvalStart - ticksFillStart);
      Interlocked.Add(ref stats.EvalTicks, ticksEvalEnd - ticksEvalStart);
      Interlocked.Increment(ref stats.BatchEvals);
      Interlocked.Add(ref stats.ChildRowsEvaluated, numRows);

      for (int m = 0; m < numMisses; m++)
      {
        int i = scratch.MissChild[m];
        int rowIndex = m * rowsPerChild;
        float muS = scratch.MuOut[rowIndex];
        float sigS = scratch.SigmaOut[rowIndex];
        float muL = rowsPerChild == 2 ? scratch.MuOut[rowIndex + 1] : 0f;
        float sigL = rowsPerChild == 2 ? scratch.SigmaOut[rowIndex + 1] : 0f;

        scratch.MuS[i] = muS;
        scratch.SigS[i] = sigS;
        scratch.MuL[i] = muL;
        scratch.SigL[i] = sigL;
        scratch.Covered[i] = true;
        numCovered++;

        GNode child = node.ChildEdgeAtIndex(i).ChildNode;
        context.Cache.Store(child.Index.Index, child.N, generation, rowsPerChild == 2, muS, sigS, muL, sigL);
      }
    }

    Interlocked.Add(ref stats.ChildCacheHits, numCacheHits);
    Interlocked.Add(ref stats.ChildCacheMisses, numMisses);

    if (numCovered == 0)
    {
      Interlocked.Increment(ref stats.ParentsElidedFast);
      return false;
    }

    // Per-child adjustments over covered children only (uncovered slots stay neutral).
    bool anyBonusMethod = context.SigmaBonusCoeff != 0 || context.ExpectedImprovementCoeff != 0
                       || context.MuCorrectionCoeff != 0;
    if (anyBonusMethod)
    {
      Array.Clear(scratch.ScoreBonus, 0, numToProcess);
      for (int i = 0; i < numCandidates; i++)
      {
        if (!scratch.Covered[i])
        {
          continue;
        }

        double bonus = 0;
        if (context.SigmaBonusCoeff != 0)
        {
          bonus += context.SigmaBonusCoeff * scratch.SigS[i];
        }
        if (context.MuCorrectionCoeff != 0)
        {
          // Child-perspective mu enters the parent-perspective score NEGATED.
          bonus += -context.MuCorrectionCoeff * scratch.MuS[i];
        }
        if (context.ExpectedImprovementCoeff != 0)
        {
          double qBestOther = i == bestIndex ? secondBestQ : bestQ;
          double qHatLong = -(scratch.EdgeQ[i] + scratch.MuL[i]);
          double sigLong = scratch.SigL[i];
          double ei;
          if (sigLong <= 1e-6)
          {
            ei = Math.Max(0.0, qHatLong - qBestOther);
          }
          else
          {
            double z = (qHatLong - qBestOther) / sigLong;
            ei = sigLong * (z * NormalCdf(z) + NormalPdf(z));
          }
          bonus += context.ExpectedImprovementCoeff * ei;
        }

        scratch.ScoreBonus[i] = bonus;
        if (context.EnableStats)
        {
          stats.RecordBonus(bonus);
        }
      }
      scoreBonus = scratch.ScoreBonus;
    }

    if (context.UTermSigmaExponent != 0 && numCovered >= 2)
    {
      Span<float> sigmas = stackalloc float[PUCTScoreCalcVector.MAX_CHILDREN];
      int count = 0;
      for (int i = 0; i < numCandidates; i++)
      {
        if (scratch.Covered[i])
        {
          // Insertion sort ascending (count <= 64).
          int pos = count++;
          while (pos > 0 && sigmas[pos - 1] > scratch.SigS[i])
          {
            sigmas[pos] = sigmas[pos - 1];
            pos--;
          }
          sigmas[pos] = scratch.SigS[i];
        }
      }
      float median = sigmas[(count - 1) / 2];
      if (median > 0)
      {
        Array.Fill(scratch.UMultiplier, 1.0, 0, numToProcess);
        for (int i = 0; i < numCandidates; i++)
        {
          if (scratch.Covered[i])
          {
            scratch.UMultiplier[i] = Math.Clamp(Math.Pow(scratch.SigS[i] / median, context.UTermSigmaExponent),
                                                0.25, 4.0);
          }
        }
        uMultiplier = scratch.UMultiplier;
        Interlocked.Add(ref stats.UMultApplications, numCovered);
      }
    }

    Interlocked.Increment(ref stats.ParentsActive);
    return scoreBonus != null || uMultiplier != null;
  }


  static double NormalPdf(double z) => Math.Exp(-0.5 * z * z) * 0.3989422804014327;


  static double NormalCdf(double z) => 0.5 * Erfc(-z * 0.7071067811865476);


  static double Erfc(double x) => 1.0 - Erf(x);


  /// <summary>
  /// Abramowitz-Stegun 7.1.26 rational approximation (max abs error ~1.5e-7).
  /// Consistency matters here, not exactness (spec section 5).
  /// </summary>
  static double Erf(double x)
  {
    double ax = Math.Abs(x);
    double t = 1.0 / (1.0 + 0.3275911 * ax);
    double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
                     * t * Math.Exp(-ax * ax);
    return x >= 0 ? y : -y;
  }
}
