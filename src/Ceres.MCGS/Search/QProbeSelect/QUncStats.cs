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
using System.Threading;

#endregion

namespace Ceres.MCGS.Search.QProbeSelect;

/// <summary>
/// Per-search Q-uncertainty select counters (Interlocked; reset at search start via
/// QUncSelectContext.OnSearchStart, printed at search end when
/// ParamsSelect.QUncEnableStats). The overhead figure (feature extraction plus
/// inference as a fraction of search wall time) is the primary deployment budget
/// gate (target under 3%).
/// </summary>
public sealed class QUncStats
{
  public long ParentsActive;
  public long ParentsElidedMinN;
  public long ParentsElidedFast;
  public long ChildCacheHits;
  public long ChildCacheMisses;
  public long BatchEvals;
  public long ChildRowsEvaluated;
  public long FeatureTicks;
  public long EvalTicks;
  public long M5Lookups;
  public long M5Hits;
  public long BonusCount;
  public long BonusAbsSumMicroQ;
  long bonusAbsMaxBits;
  public long UMultApplications;


  public double BonusAbsMax => BitConverter.Int64BitsToDouble(Interlocked.Read(ref bonusAbsMaxBits));


  public void Reset()
  {
    Interlocked.Exchange(ref ParentsActive, 0);
    Interlocked.Exchange(ref ParentsElidedMinN, 0);
    Interlocked.Exchange(ref ParentsElidedFast, 0);
    Interlocked.Exchange(ref ChildCacheHits, 0);
    Interlocked.Exchange(ref ChildCacheMisses, 0);
    Interlocked.Exchange(ref BatchEvals, 0);
    Interlocked.Exchange(ref ChildRowsEvaluated, 0);
    Interlocked.Exchange(ref FeatureTicks, 0);
    Interlocked.Exchange(ref EvalTicks, 0);
    Interlocked.Exchange(ref M5Lookups, 0);
    Interlocked.Exchange(ref M5Hits, 0);
    Interlocked.Exchange(ref BonusCount, 0);
    Interlocked.Exchange(ref BonusAbsSumMicroQ, 0);
    Interlocked.Exchange(ref bonusAbsMaxBits, 0);
    Interlocked.Exchange(ref UMultApplications, 0);
  }


  /// <summary>
  /// Records one per-child score bonus (any of M1/M2/M3 contributions summed).
  /// </summary>
  public void RecordBonus(double bonus)
  {
    double absBonus = Math.Abs(bonus);
    Interlocked.Increment(ref BonusCount);
    Interlocked.Add(ref BonusAbsSumMicroQ, (long)(absBonus * 1e6));

    long observedBits = Interlocked.Read(ref bonusAbsMaxBits);
    while (absBonus > BitConverter.Int64BitsToDouble(observedBits))
    {
      long newBits = BitConverter.DoubleToInt64Bits(absBonus);
      long priorBits = Interlocked.CompareExchange(ref bonusAbsMaxBits, newBits, observedBits);
      if (priorBits == observedBits)
      {
        break;
      }
      observedBits = priorBits;
    }
  }


  public string SummaryLine(double searchSeconds)
  {
    long hits = Interlocked.Read(ref ChildCacheHits);
    long misses = Interlocked.Read(ref ChildCacheMisses);
    long rows = Interlocked.Read(ref ChildRowsEvaluated);
    long evals = Interlocked.Read(ref BatchEvals);
    long featTicks = Interlocked.Read(ref FeatureTicks);
    long evalTicks = Interlocked.Read(ref EvalTicks);
    long bonusCount = Interlocked.Read(ref BonusCount);
    long m5Lookups = Interlocked.Read(ref M5Lookups);

    double hitRate = hits + misses > 0 ? 100.0 * hits / (hits + misses) : 0;
    double usPerRow = rows > 0 ? (evalTicks * 1e6 / Stopwatch.Frequency) / rows : 0;
    double overheadPct = searchSeconds > 0
        ? 100.0 * (featTicks + evalTicks) / ((double)Stopwatch.Frequency * searchSeconds) : 0;
    double bonusMean = bonusCount > 0 ? Interlocked.Read(ref BonusAbsSumMicroQ) * 1e-6 / bonusCount : 0;
    double m5HitRate = m5Lookups > 0 ? 100.0 * Interlocked.Read(ref M5Hits) / m5Lookups : 0;

    return $"[QUNC] parents active {Interlocked.Read(ref ParentsActive):N0} "
         + $"(elided minN {Interlocked.Read(ref ParentsElidedMinN):N0} fast {Interlocked.Read(ref ParentsElidedFast):N0}); "
         + $"child cache hit {hitRate:F1}%; batches {evals:N0} rows {rows:N0} ({usPerRow:F1} us/row); "
         + $"overhead {overheadPct:F2}%; |bonus| mean {bonusMean:F4} max {BonusAbsMax:F4} (n={bonusCount:N0}); "
         + $"uMult n={Interlocked.Read(ref UMultApplications):N0}; "
         + $"M5 lookups {m5Lookups:N0} hit {m5HitRate:F1}%";
  }
}
