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
using Ceres.Base.OperatingSystem;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Type of backup algorithm used in the search.
/// N.B. The ordering of the enum values is important (see IsReduction extension method). 
/// </summary>
public enum BackupMethodEnum
{
  /// <summary>
  /// Backup via single threaded graph reduction (leaf toward root until merge node encountered).
  /// </summary>
  ReductionSingleThread,

  /// <summary>
  /// Backup via parallel thread reduction (leaf toward root until merge node encountered).
  /// </summary>
  ReductionMultiThread,
}


/// <summary>
/// Defines the parameters relating to the implementation of the search
/// such as batch size ore degree of parallelism.
/// 
/// Optimal values are computed dynamically based on the characteristics
/// of the current search. For example, parallelism may be 
/// disabled for very small searches because it is not beneficial.
/// </summary>
[Serializable]
public record ParamsSearchExecution
{
  /// <summary>
  /// Scaling factor with higher values decreasing 
  /// the degree of fine-grained parallelism employed
  /// in various places (ParallelFor granularity).
  /// Higher values may be optimal for Linux.
  /// </summary>
  public const int ParallelMultiplier = 1;

  internal const bool DEFAULT_USE_SMART_SIZE_BATCHES = true;

  /// <summary>
  /// If the batch size is dynamically computed each time based on
  /// search characteristics  (e.g. larger batches when the tree is already large).
  /// </summary>
  public bool SmartSizeBatches = DEFAULT_USE_SMART_SIZE_BATCHES;

  /// <summary>
  /// Optional a set batch sizes at which the NNEvaluator
  /// is known to suffer a local maximum (falling off sharply in speed just above this value).
  /// </summary>
  public int[] NNEvaluatorBatchSizeBreakHints = null;

  /// <summary>
  /// If two batches should processed used in flight simultaneously,
  /// with a second batch being assembled concurrently while the
  /// prior batch is being evaluated by the network evaluator.
  /// </summary>
  public bool DualOverlappedIterators = CeresUserSettingsManager.Settings.EnableOverlappingExecutors;

  /// <summary>
  /// If two NNEvaluators should be used when OverlappedIterators is true.
  /// This modestly increases search parallelism (and therefore speed)
  /// at the cost of slower startup and more GPU VRAM usage.
  /// </summary>
  public bool DualEvaluators = true;

  /// <summary>
  /// Optional additional hard limit on size of gathered batch 
  /// of nodes (not all of which are necessarily destined for neural network evaluation).
  /// </summary>
  public int MaxBatchSize = CeresUserSettingsManager.Settings.MaxBatchSize;

  /// <summary>
  /// Improves utilization by padding device batch sizes to be
  /// aligned with (or just below) multiples of specified value (typically a power of 2).
  /// NPS improvements for large networks are typically on the order of:
  ///   25% at 100 nodes/move,
  ///   5-10% at 1000 nodes/move.
  /// Value of 0 disables.
  /// TODO: Instead of divisor-based alignment, use exact fixed batch sizes used by NNEvaluator.
  /// </summary>
  public int NNBatchSizeAlignmentTarget = 0;


  /// <summary>
  /// If we are running dual iterators it is possible that some nodes
  /// that a selector will choose the same leaf nodes as already 
  /// selected by the other selector. This duplicate will be 
  /// detected and aborted with some resulting loss in runtime efficiency.
  /// To mitigate this can allow some repelling force between the two selectors
  /// by applying some virtual loss borrowed from the other selector.
  /// 
  /// Larger values can make search speed better (fewer duplicated nodes)
  /// but come with a clear loss in play quality.
  /// </summary>
  public float DualIteratorAlternateCollisionFraction = 0.25f;

  /// <summary>
  /// Level above which separate threads are spawned for selecting nodes. 
  /// Use int.MaxValue to disable.
  /// Note that this value may be adjusted downward somewhat
  /// dynamically at runtime if graph size is very large.
  /// </summary>
  public int SelectOperationParallelThresholdNumVisits = 22;

 /// <summary>
  /// If the initialization of policies in tree nodes (after retrieval from NN)
  /// is (potentially) done in parallel.
  /// </summary>
  public bool SetPoliciesParallelEnabled = System.Environment.ProcessorCount > 2;

  /// <summary>
  /// Algorithm used for the backup stage.
  /// </summary>
  public BackupMethodEnum BackupMode = BackupMethodEnum.ReductionMultiThread;

  /// <summary>
  /// If overlapping is enabled and greater than zero,
  /// both iterators will pause and synchronize
  /// after every N completed batches (post-backup).
  /// </summary>
  public int SyncEveryNBatches;
}
