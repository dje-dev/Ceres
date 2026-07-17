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
using System.Threading;

using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.QProbeSelect;

/// <summary>
/// Per-engine root object for the Q-uncertainty select methods: the loaded model,
/// gate values copied from ParamsSelect, the per-child forecast cache, per-search
/// stats and the cache generation stamp. Constructed once per engine (lazily, at
/// strategy creation) when ParamsSelect.QUncAnyMethodActive; shared by all search
/// threads (the model is immutable, the cache seqlocked, the stats Interlocked).
/// </summary>
public sealed class QUncSelectContext
{
  public readonly QProbeSelectNNModel Model;

  /// <summary>Resolved short query horizon (param override or model header HorizonN1).</summary>
  public readonly int DnShort;

  /// <summary>Resolved long query horizon (param override or model header HorizonN2).</summary>
  public readonly int DnLong;

  public readonly float SigmaBonusCoeff;           // M1
  public readonly float ExpectedImprovementCoeff;  // M2
  public readonly float MuCorrectionCoeff;         // M3
  public readonly float UTermSigmaExponent;        // M4
  public readonly bool TPSLearnedSigma;            // M5a
  public readonly bool TPSLearnedPrior;            // M5b

  public readonly int MinParentN;
  public readonly int MaxChildren;
  public readonly bool EnableStats;

  /// <summary>From ParamsSearch, for capture parity with harvest (volatility/trend features).</summary>
  public readonly bool TrackVolatility;

  /// <summary>Any of M1..M4 configured (the select-path methods).</summary>
  public readonly bool SelectMethodsActive;

  /// <summary>M5a or M5b configured (the TPS backup methods).</summary>
  public readonly bool BackupMethodsActive;

  /// <summary>The long horizon is queried only when M2 is on.</summary>
  public readonly bool NeedLongHorizon;

  internal readonly QUncChildCache Cache = new QUncChildCache();
  public readonly QUncStats Stats = new QUncStats();

  int currentGeneration;

  public int CurrentGeneration => Volatile.Read(ref currentGeneration);


  public QUncSelectContext(QProbeSelectNNModel model, ParamsSelect paramsSelect, ParamsSearch paramsSearch)
  {
    Model = model;
    DnShort = paramsSelect.QUncDnShort > 0 ? paramsSelect.QUncDnShort : model.HorizonN1;
    DnLong = paramsSelect.QUncDnLong > 0 ? paramsSelect.QUncDnLong : model.HorizonN2;
    SigmaBonusCoeff = paramsSelect.QUncSigmaBonusCoeff;
    ExpectedImprovementCoeff = paramsSelect.QUncExpectedImprovementCoeff;
    MuCorrectionCoeff = paramsSelect.QUncMuCorrectionCoeff;
    UTermSigmaExponent = paramsSelect.QUncUTermSigmaExponent;
    TPSLearnedSigma = paramsSelect.QUncTPSLearnedSigma;
    TPSLearnedPrior = paramsSelect.QUncTPSLearnedPrior;
    MinParentN = paramsSelect.QUncMinParentN;
    MaxChildren = Math.Min(paramsSelect.QUncMaxChildren, 64);
    EnableStats = paramsSelect.QUncEnableStats;
    TrackVolatility = paramsSearch.TrackLeafValueVolatility;

    SelectMethodsActive = SigmaBonusCoeff != 0 || ExpectedImprovementCoeff != 0
                       || MuCorrectionCoeff != 0 || UTermSigmaExponent != 0;
    BackupMethodsActive = TPSLearnedSigma || TPSLearnedPrior;
    NeedLongHorizon = ExpectedImprovementCoeff != 0;
  }


  /// <summary>
  /// Called once at the start of every search: invalidates all cached forecasts
  /// (node indices may be recycled across searches under graph reuse) and resets
  /// the per-search counters.
  /// </summary>
  public void OnSearchStart()
  {
    Interlocked.Increment(ref currentGeneration);
    Stats.Reset();
  }


  /// <summary>
  /// Cache-ONLY lookup of the short-horizon forecast for a child node, for the TPS
  /// backup methods (M5a/M5b). NEVER evaluates the model - the backup path must stay
  /// cheap and lock-free; on a miss the caller falls back to its closed forms.
  /// </summary>
  public bool TryGetShortHorizon(int childNodeIndex, int childN, out float mu, out float sigma)
  {
    Interlocked.Increment(ref Stats.M5Lookups);
    bool hit = Cache.TryGet(childNodeIndex, childN, CurrentGeneration, needLong: false,
                            out mu, out sigma, out float muLongUnused, out float sigmaLongUnused);
    if (hit)
    {
      Interlocked.Increment(ref Stats.M5Hits);
    }
    return hit;
  }
}
