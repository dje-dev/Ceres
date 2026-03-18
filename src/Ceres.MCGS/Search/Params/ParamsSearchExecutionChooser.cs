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
using Ceres.Chess;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.MCGS.Search.Params;


#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Implements logic to choose the best execution parameters
/// for a given batch size and set of search parameters.
/// 
/// The selected execution parameters mostly relate to the 
/// implementation details such as multithreading strategy
/// and prefetching.
/// calling 
/// </summary>
public class ParamsSearchExecutionChooser
{
  public const int THRESHOLD_BEGIN_OVERLAPPING = 5000;

  /// <summary>
  /// Neural network evaluator being used for search.
  /// </summary>
  public NNEvaluatorDef NNEvaluatorDef;

  /// <summary>
  /// Search parameters used for search.
  /// </summary>
  public ParamsSearch ParamsSearch;

  /// <summary>
  /// Leaf selection parameters being for search.
  /// </summary>
  public ParamsSelect ParamsSelect;

  /// <summary>
  /// Resource limit (time or nodes) being used for search.
  /// </summary>
  public SearchLimit SearchLimit;


  private static ParamsSearchExecution DEFAULT_PARAMS = new();



  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="nnEvaluatorDef"></param>
  /// <param name="paramsSearch"></param>
  /// <param name="paramsSelect"></param>
  /// <param name="searchLimit"></param>
  public ParamsSearchExecutionChooser(NNEvaluatorDef nnEvaluatorDef,
                                      ParamsSearch paramsSearch,
                                      ParamsSelect paramsSelect,
                                      SearchLimit searchLimit)
  {
    // Make sure params arguments look initialized
    if (nnEvaluatorDef == null) throw new ArgumentNullException(nameof(nnEvaluatorDef));

    NNEvaluatorDef = nnEvaluatorDef;
    ParamsSearch = paramsSearch;
    ParamsSelect = paramsSelect;
    SearchLimit = searchLimit with { };
  }


  /// <summary>
  /// Primary method that sets the optimal parameters.
  /// </summary>
  /// <param name="estNumNodes"></param>
  /// <param name="postprocessor">optional delegated called after parameters are set allowing modifications</param>
  public void ChooseOptimal(int estNumNodes)
  {
    if (ParamsSearch.AutoOptimizeEnabled)
    {
      // Start by using a clone of existing settings (including possibly set by user and carried over).
      ParamsSearch.Execution = ParamsSearch.Execution with { };

      // Then choose optimal values.
      DoChooseOptimal(estNumNodes);
    }

    if (ParamsSearch.ExecutionModifierID != null)
    {
      ParamsSearchExecutionModifier.Invoke(ParamsSearch.ExecutionModifierID, ParamsSearch.Execution);
    }
  }


  /// <summary>
  /// Internal worker method to that actually computes and updates the optimal parameters.
  /// </summary>
  /// <param name="estNumNodes"></param>
  void DoChooseOptimal(int estNumNodes)
  {
    throw new Exception("Not currently used in MCGS. Maybe some should be, e.g. AdjustForPooled?");

    // TODO: lift restriction that SMART_SIZE only works with single device
    ParamsSearch.Execution.SmartSizeBatches = ParamsSearchExecution.DEFAULT_USE_SMART_SIZE_BATCHES
                                           && estNumNodes > 1000
                                           && NNEvaluatorDef.NumDevices == 1;

    if (NNEvaluatorDef.DeviceCombo == NNEvaluatorDeviceComboType.Pooled)
    {
      AdjustForPooled();
    }

    // Turn off some features if search is very small (overhead of initializing them not worth it)
    const int CUTOVER_NUM_NODES_SMALL = 20_000;

    ParamsSearch.Execution.DualOverlappedIterators = DEFAULT_PARAMS.DualOverlappedIterators && estNumNodes > THRESHOLD_BEGIN_OVERLAPPING;

    if (estNumNodes < CUTOVER_NUM_NODES_SMALL)
    {
      ParamsSearch.Execution.SetPoliciesParallelEnabled = false;
    }
    else
    {
      ParamsSearch.Execution.SetPoliciesParallelEnabled = DEFAULT_PARAMS.SetPoliciesParallelEnabled;
    }

    // TODO: set the GPU fractions if multiple
  }


  /// <summary>
  /// Makes adjustmens to parmaeters that are appropriate when 
  /// a pooled evaluator is being used.
  /// </summary>
  private void AdjustForPooled()
  {
    // Reduce maximum batch size to avoid overflow in multibatch evaluator
    // where some pooling with other batches may be unavoidable.
    int maxBatchSize = 800;
    ParamsSearch.Execution.MaxBatchSize = Math.Min(ParamsSearch.Execution.MaxBatchSize, maxBatchSize);
  }

}
