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
using System.Threading.Tasks;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;

#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// A set of possibly multiple NNEvaluator instances used together as a set.
/// 
/// For example, a search may require:
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
  /// If the overlapping evaluator (#1) will be required.
  /// </summary>
  public readonly bool UsesEvaluator1;

  /// <summary>
  /// Definition of the evaluator to be used for EvaluatorSecondary (optional).
  /// </summary>
  public readonly NNEvaluatorDef EvaluatorDefSecondary;

  #region Internal data

  [NonSerialized]
  NNEvaluator evaluator0 = null;

  [NonSerialized]
  NNEvaluator evaluator1 = null;

  // TODO: move this elsewhere
  [NonSerialized]
  NNEvaluator evaluatorSecondary = null;

  /// <summary>
  /// Lock to enforce serialized access to evaluator1
  /// </summary>
  readonly Lock makeEvaluator0Lock = new();

  /// <summary>
  /// Lock to enforce serialized access to evaluator2
  /// </summary>
  readonly Lock makeEvaluator1Lock = new();

  /// <summary>
  /// Lock to enforce serialized access to evaluatorSecondary
  /// </summary>
  readonly Lock makeEvaluatorSecondaryLock = new();

  #endregion


  /// <summary>
  /// Standard constructor which builds the evaluator
  /// (or two evaluators, if running with overlapping)
  /// according to a specified definition.
  /// </summary>
  /// <param name="evaluatorDef"></param>
  /// <param name="usesEvaluator1"></param>
  /// <param name="evaluatorDefSecondary"></param>
  public NNEvaluatorSet(NNEvaluatorDef evaluatorDef, bool usesEvaluator1,
                        NNEvaluatorDef evaluatorDefSecondary = null)
  {
    EvaluatorDef = evaluatorDef ?? throw new ArgumentNullException(nameof(evaluatorDef));
    UsesEvaluator1 = usesEvaluator1;
    EvaluatorDefSecondary = evaluatorDefSecondary;
  }


  public bool IsWDL => Evaluator0.IsWDL;
  public bool HasM => Evaluator0.HasM;

  public bool HasState => Evaluator0.HasState;

  public bool HasAction => Evaluator0.HasAction;


  public bool PolicyReturnedSameOrderMoveList => Evaluator0.PolicyReturnedSameOrderMoveList;


  /// <summary>
  /// Overrides the evaluators to be used by the NNEvaluatorSet.
  /// Intended mainly for testing purposes.
  /// </summary>
  /// <param name="evaluator1"></param>
  /// <param name="evaluator2"></param>
  /// <param name="evaluatorSecondary"></param>
  public void OverrideEvaluators(NNEvaluator evaluator1, NNEvaluator evaluator2, NNEvaluator evaluatorSecondary)
  {
    this.evaluator0 = evaluator1;
    this.evaluator1 = evaluator2;
    this.evaluatorSecondary = evaluatorSecondary;
  }


  /// <summary>
  /// Returns the maximum batch size supported by the NNEvaluatorSet.
  /// </summary>
  public int MaxBatchSize
  {
    get
    {
      int max = Evaluator0.MaxBatchSize;
      if (UsesEvaluator1) max = Math.Max(max, Evaluator1.MaxBatchSize);
      if (EvaluatorSecondary != null) max = Math.Max(max, EvaluatorSecondary.MaxBatchSize);
      return max;
    }
  }


  /// <summary>
  /// Initializes the evaluators and performs some 
  /// minimal evaluation to insure any startup overhead is completed.
  /// </summary>
  public void Warmup(int maxBatchSize)
  {
    Evaluator0.Warmup();
    Evaluator1?.Warmup();
    EvaluatorSecondary?.Warmup(); 
  }


  NNEvaluatorPerformanceStats perfStatsPrimary = null;

  public NNEvaluatorPerformanceStats PerfStatsPrimary => perfStatsPrimary;

  public void CalcStatistics(bool computeBreaks, float maxSeconds = 1.0f)
  {
    if (Evaluator0.PerformanceStats == null)
    {
      Evaluator0.CalcStatistics(computeBreaks, maxSeconds);
    }
    perfStatsPrimary = Evaluator0.PerformanceStats;
  }


  /// <summary>
  /// Returns the first evaluator in the set.
  /// </summary>
  public NNEvaluator Evaluator0
  {
    get
    {
      if (evaluator0 == null)
      {
        lock (makeEvaluator0Lock)
        {
          evaluator0 ??= NNEvaluatorFactory.BuildEvaluator(EvaluatorDef, null);
        }
      }

      return evaluator0;
    }
  }


  /// <summary>
  /// Returns the second evaluator in the set.
  /// </summary>
  public NNEvaluator Evaluator1
  {
    get
    {
      if (!UsesEvaluator1)
      {
        return null;
      }

      if (evaluator1 == null)
      {
        lock (makeEvaluator1Lock)
        {
          if (evaluator1 == null)
          {
            // Pass the first evaluator as the "reference" evaluator
            // from which the already loaded network weights can be reused.
            Debug.Assert(Evaluator0 != null);
            evaluator1 = NNEvaluatorFactory.BuildEvaluator(EvaluatorDef, Evaluator0);
          }
        }
      }

      return evaluator1;
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
          evaluatorSecondary ??= NNEvaluatorFactory.BuildEvaluator(EvaluatorDefSecondary, null);
        }
      }
      return evaluatorSecondary;
    }
  }

  #region Dispose

  private bool disposedValue;

  protected virtual void Dispose(bool disposing)
  {
    if (!disposedValue)
    {
      if (disposing)
      {
        evaluator0?.Dispose();
        evaluator1?.Dispose();
        evaluatorSecondary?.Dispose();

        evaluator0 = null;
        evaluator1 = null;
        evaluatorSecondary = null;
      }

      disposedValue = true;
    }
  }

  public void Dispose()
  {
    Dispose(disposing: true);
  }

  #endregion

  /// <summary>
  /// Returns a string representation of the NNEvaluatorSet.
  /// </summary>
  /// <returns></returns>
  public override string ToString() 
    =>  $"<NNEvaluatorSet: Def={EvaluatorDef}, UsesEv1={UsesEvaluator1}, DefSecon={EvaluatorDefSecondary}>";
}
