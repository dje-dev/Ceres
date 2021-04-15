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
  ///   - possibly a second evaluator to support overlapped (concurrent) evaluations,
  ///   - and possibly a supplemental evaluator used for ancillary purposes.
  /// if 
  /// </summary>
  public partial class NNEvaluatorSet : IDisposable
  {
    public readonly NNEvaluatorDef EvaluatorDef;

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
    public NNEvaluatorSet(NNEvaluatorDef evaluatorDef)
    {
      if (evaluatorDef == null) throw new ArgumentNullException(nameof(evaluatorDef));

      EvaluatorDef = evaluatorDef;
    }

    public bool IsWDL => evaluator1.IsWDL;
    public bool HasM => evaluator1.HasM;

    public bool PolicyReturnedSameOrderMoveList => evaluator1.PolicyReturnedSameOrderMoveList;


    public void Warmup(bool alsoCalcStatistics)
    {
      if (alsoCalcStatistics)
      {
        Parallel.Invoke(
          () => CalcStatistics(true),
          () => { if (Evaluator2 is not null) NNEvaluatorBenchmark.Warmup(Evaluator2); }
//          () => { if (EvaluatorSecondary is not null) NNEvaluatorBenchmark.Warmup(EvaluatorSecondary); }
          );
      }
      else
      {
        Parallel.Invoke(
          () => NNEvaluatorBenchmark.Warmup(Evaluator1),
          () => { if (Evaluator2 is not null) NNEvaluatorBenchmark.Warmup(Evaluator2); }
//          () => { if (EvaluatorSecondary is not null) NNEvaluatorBenchmark.Warmup(EvaluatorSecondary); }
          );
      }
    }

    NNEvaluatorPerformanceStats perfStatsPrimary = null;

    public NNEvaluatorPerformanceStats PerfStatsPrimary =>  perfStatsPrimary;

    public void CalcStatistics(bool computeBreaks, float maxSeconds = 1.0f)
    {
      if (Evaluator1.PerformanceStats == null)
        evaluator1.CalcStatistics(computeBreaks, maxSeconds);
      perfStatsPrimary = evaluator1.PerformanceStats;
    }


    NNEvaluator MakeEvaluator()
    {
      return NNEvaluatorFactory.BuildEvaluator(EvaluatorDef);
    }

    public NNEvaluator Evaluator1
    {
      get
      {
        const bool SHARED = true;
        if (evaluator1 == null)
          lock (makeEvaluator1Lock)
            if (evaluator1 == null)
              evaluator1 = MakeEvaluator();

        return evaluator1;
      }
    }

    public NNEvaluator Evaluator2
    {
      get
      {
        const bool SHARED = false; // we need to overlap concurrent execution, therefore must be a different session
        if (SHARED) return Evaluator1;

        if (evaluator2 == null)
          lock (makeEvaluator2Lock)
            if (evaluator2 == null)
              evaluator2 = MakeEvaluator();

        return evaluator2;
      }
    }

    public NNEvaluator EvaluatorSecondary
    {
      get
      {
        const bool SHARED = false; // ??
        if (evaluatorSecondary == null)
        {
          lock (makeEvaluatorSecondaryLock)
          {
            if (evaluatorSecondary == null)
            {
              // TODO: someday enable fancy combo nets like in Evaluator1 above
              throw new NotImplementedException("Needs remediation");
              //evaluatorSecondary = MakeLocalNNEvaluator(0, 0, SHARED, Params.SECONDARY_NETWORK_ID);
              //if (Params.EstimatePerformanceCharacteristics) evaluatorSecondary.CalcStatistics(true);
              return evaluatorSecondary;
            }
          }
        }
        return evaluatorSecondary;
      }
    }


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