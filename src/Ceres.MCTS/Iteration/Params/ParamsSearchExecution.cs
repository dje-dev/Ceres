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

namespace Ceres.MCTS.Params
{
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

    /// <summary>
    /// Method used for explotiing transpositions within the tree
    /// to avoid reevaluation of the positions by the neural network.
    /// </summary>
    public TranspositionMode TranspositionMode = TranspositionMode.SingleNodeDeferredCopy;

    /// <summary>
    /// If attempt is made to keep each transposition root as having
    /// N large relative to other nodes in the same equivalence class.
    /// Benefit is unclear in testing, though currently on maintained on tree rewrite.
    /// </summary>
    public bool TranspositionMaximizeRootN = false;

    /// <summary>
    /// If transpositions are detected and copied within a single batch,
    /// thereby reducing the number of NN evaluations needed.
    /// However disabled by default because at most 0.5% of NN evaluations are obviated,
    /// and incompatible with NNEvaluatorBatchSizeBreakHints.
    /// </summary>
    public bool InFlightThisBatchLinkageEnabled = false;

    /// <summary>
    /// If transpositions are detected and copied
    /// between the batch being assembled and the prior batch
    /// being evaluated in parallel due to overlapping.
    /// TODO: This seems not faster and is possibly buggy, fix up or possibly eliminate.
    /// </summary>
    public bool InFlightOtherBatchLinkageEnabled = false;

    /// <summary>
    /// Size of the cache used to store miscellaneous supplemental
    /// information (such as the actual Position) relating to recently visited nodes.
    /// </summary>
    public int NodeAnnotationCacheSize = CeresUserSettingsManager.Settings.ReducedMemoryMode ? 350_000 : 1_000_000;


    internal const bool DEFAULT_USE_SMART_SIZE_BATCHES = true;

    /// <summary>
    /// If the batch size is dynamically computed each time based on
    /// search characateristics  (e.g. larger batches when the tree is already large).
    /// </summary>
    public bool SmartSizeBatches = DEFAULT_USE_SMART_SIZE_BATCHES;

    /// <summary>
    /// Optional a set batch sizes at which the NNEvaluator
    /// is known to suffer a local maximum (falling off sharply in speed just above this value).
    /// </summary>
    public int[] NNEvaluatorBatchSizeBreakHints = null;

    /// <summary>
    /// If two batches should procesed used in flight simultaneously,
    /// with a second batch being assembled concurrently while the
    /// prior batch is being evaluated by the network evaluator.
    /// </summary>
    public bool FlowDirectOverlapped = CeresUserSettingsManager.Settings.EnableOverlappingExecutors;

    /// <summary>
    /// If two selectors should be used in batch gathering,
    /// each taking CPUCT values slightly offset from the baseline value.
    /// </summary>
    public bool FlowDualSelectors = true;

    /// <summary>
    /// If each batch should be gathered in two passes 
    /// of the tree from root to leaves.
    /// 
    /// If true, the immediate nodes are applied after
    /// the first pass and the second pass is sometimes
    /// skipped if the yield is low indicating 
    /// many collisions and likely lower node selection purity.
    /// </summary>
    public bool FlowSplitSelects = true;

    /// <summary>
    /// Optional additional hard limit on size of gathered batch 
    /// of nodes (not all of which are necessarily destined for neural network evaluation).
    /// </summary>
    public int MaxBatchSize = CeresUserSettingsManager.Settings.MaxBatchSize;

    /// <summary>
    /// If we are running dual selectors it is possible that some nodes
    /// that a selector will choose the same leaf nodes as already 
    /// selected by the other selector. This duplicate will be 
    /// detected and aborted with some resulting loss in runtine efficiency.
    /// To mitigate this can allow some repelling force between the two selectors
    /// by applying some virtual loss borrowed from the other selector.
    /// 
    /// Larger values can make search speed better (fewer dupliated nodes)
    /// but come with a clear loss in play quality.
    /// </summary>
    public float DualSelectorAlternateCollisionFraction = 0.25f;

    
    /// <summary>
    /// If MCTS leaf selection is (potentially) conducted by parallel threads
    /// (over non-overlapping subtrees).
    /// </summary>
    public bool SelectParallelEnabled = true;

    /// <summary>
    /// Minimum number of targeted leaf visits which must be present
    /// for parallel subthreads to be allocated.
    /// </summary>
    public int SelectParallelThreshold = ParallelMultiplier * (SoftwareManager.IsLinux ? 8 : 5);

    /// <summary>
    /// If the initialization of poliices in tree nodes (after retrieval from NN)
    /// is (potentially) done in parallel.
    /// </summary>
    public bool SetPoliciesParallelEnabled = true;

    /// <summary>
    /// Target number of policies to be initialized by each thread.
    /// </summary>
    public int SetPoliciesNumPoliciesPerThread = 32 * ParallelMultiplier;


  }
}
