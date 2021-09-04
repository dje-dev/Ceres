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
using System.Threading.Tasks;

using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;

#endregion

namespace Ceres.MCTS.Params
{
  /// <summary>
  /// A set of possibly multiple NNEvaluator instances used together as a set.
  /// 
  /// For example, an MCTS search may require:
  ///   - a primary evaluator,
  ///   - possibly a second evaluator (sharing the same definition) 
  ///     to support overlapped (concurrent) evaluations,
  ///   - possibly a supplemental evaluator (with a possibly different definition)
  ///     used for ancillary purposes.
  /// </summary>
  public partial class NNEvaluatorSet : IDisposable
  {
    /// <summary>
    /// Definition of the evaluators to be used for Evaluator1 and Evaluator2.
    /// </summary>
    public readonly NNEvaluatorDef EvaluatorDef;

    /// <summary>
    /// If the overlapping evaluator (#2) will be required.
    /// </summary>
    public readonly bool UsesEvaluator2;

    /// <summary>
    /// Definition of the evaluator to be used for EvaluatorSecondary (optional).
    /// </summary>
    public readonly NNEvaluatorDef EvaluatorDefSecondary;


    #region Internal data

    [NonSerialized]
    NNEvaluator evaluator1 = null;

    [NonSerialized]
    NNEvaluator evaluator2 = null;

    // TODO: move this elsewhere
    [NonSerialized]
    NNEvaluator evaluatorSecondary = null;

    /// <summary>
    /// Lock to enforce serialized access to evaluator1
    /// </summary>
    object makeEvaluator1Lock = new object();

    /// <summary>
    /// Lock to enforce serialized access to evaluator2
    /// </summary>
    object makeEvaluator2Lock = new object();

    /// <summary>
    /// Lock to enforce serialized access to evaluatorSecondary
    /// </summary>
    object makeEvaluatorSecondaryLock = new object();

    #endregion


    /// <summary>
    /// Standard constructor which builds the evaluator
    /// (or two evaluators, if running with overlapping)
    /// according to a specified definition.
    /// </summary>
    /// <param name="evaluatorDef"></param>
    /// <param name="usesEvaluator2"></param>
    /// <param name="evaluatorDefSecondary"></param>
    public NNEvaluatorSet(NNEvaluatorDef evaluatorDef, bool usesEvaluator2, 
                          NNEvaluatorDef evaluatorDefSecondary = null)
    {
      if (evaluatorDef == null)
      {
        throw new ArgumentNullException(nameof(evaluatorDef));
      }

      EvaluatorDef = evaluatorDef;
      UsesEvaluator2 = usesEvaluator2;
      EvaluatorDefSecondary = evaluatorDefSecondary;
    }


    public bool IsWDL => evaluator1.IsWDL;
    public bool HasM => evaluator1.HasM;

    public bool PolicyReturnedSameOrderMoveList => evaluator1.PolicyReturnedSameOrderMoveList;

    /// <summary>
    /// Returns the maximum batch size supported by the NNEvaluatorSet.
    /// </summary>
    public int MaxBatchSize
    {
      get
      {
        int max = evaluator1.MaxBatchSize;
        if (evaluator2 != null) max = Math.Max(max, evaluator2.MaxBatchSize);
        if (evaluatorSecondary != null) max = Math.Max(max, evaluatorSecondary.MaxBatchSize);
        return max;
      }
    }


    public void Warmup(bool alsoCalcStatistics)
    {
      if (alsoCalcStatistics)
      {
        Parallel.Invoke(
          () => CalcStatistics(true),
          () => { if (UsesEvaluator2) NNEvaluatorBenchmark.Warmup(Evaluator2); },
          () => { if (EvaluatorSecondary is not null) NNEvaluatorBenchmark.Warmup(EvaluatorSecondary); }
          );
      }
      else
      {
        Parallel.Invoke(
          () => NNEvaluatorBenchmark.Warmup(Evaluator1),
          () => { if (UsesEvaluator2) NNEvaluatorBenchmark.Warmup(Evaluator2); },
          () => { if (EvaluatorSecondary is not null) NNEvaluatorBenchmark.Warmup(EvaluatorSecondary); }
          );
      }
    }

    NNEvaluatorPerformanceStats perfStatsPrimary = null;

    public NNEvaluatorPerformanceStats PerfStatsPrimary =>  perfStatsPrimary;

    public void CalcStatistics(bool computeBreaks, float maxSeconds = 1.0f)
    {
      if (Evaluator1.PerformanceStats == null)
      {
        evaluator1.CalcStatistics(computeBreaks, maxSeconds);
      }
      perfStatsPrimary = evaluator1.PerformanceStats;
    }


    /// <summary>
    /// Returns the first evaluator in the set.
    /// </summary>
    public NNEvaluator Evaluator1
    {
      get
      {
        if (evaluator1 == null)
        {
          lock (makeEvaluator1Lock)
          {
            if (evaluator1 == null)
            {
              evaluator1 = NNEvaluatorFactory.BuildEvaluator(EvaluatorDef, null);
            }
          }
        }

        return evaluator1;
      }
    }


    /// <summary>
    /// Returns the second evaluator in the set.
    /// </summary>
    public NNEvaluator Evaluator2
    {
      get
      {
        if (evaluator2 == null)
        {
          lock (makeEvaluator2Lock)
          {
            if (evaluator2 == null)
            {
              Debug.Assert(Evaluator1 != null);
              evaluator2 = NNEvaluatorFactory.BuildEvaluator(EvaluatorDef, Evaluator1);
            }
          }
        }

        return evaluator2;
      }
    }


    /// <summary>
    /// Returns the optional secondary evaluator in the set
    /// (often used for comparison purposes).
    /// </summary>
    public NNEvaluator EvaluatorSecondary
    {
      get
      {
        if (EvaluatorDefSecondary == null)
        {
          return null;
        }

        if (evaluatorSecondary == null)
        {
          lock (makeEvaluatorSecondaryLock)
          {
            if (evaluatorSecondary == null)
            {
              evaluatorSecondary = NNEvaluatorFactory.BuildEvaluator(EvaluatorDefSecondary, null);
            }
          }
        }
        return evaluatorSecondary;
      }
    }


    /// <summary>
    /// Disposes of underlying evaluators.
    /// </summary>
    public void Dispose()
    {
      evaluator1?.Dispose();
      evaluator2?.Dispose();
      evaluatorSecondary?.Dispose();

      evaluator1 = null;
      evaluator2 = null;
      evaluatorSecondary = null;
    }

  }

}