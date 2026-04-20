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
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.MCGS.Search.Params;

/// <summary>
/// Specifies the mode (rules used for establishing equivalence classes over positions/histories)
/// used for transposition matching.
/// </summary>
public enum PathMode
{
  /// <summary>
  /// Only the current position is used (no history).
  /// 
  /// This mode suffers from the GHI (graph history inconsistency) problem because it 
  /// records information keyed only by the position, while the game’s rules make game state
  /// (draw by repetition detection) depend on prior occurrences along the current line.
  /// 
  /// By aliasing together all positions that are identical in board position,
  /// this mode causes an averaging of visits and Q values across all such positions,
  /// some of which may have triggered draws by repetition due to their particular histories.
  /// </summary>
  PositionEquivalence,

  /// <summary>
  /// The current position and accumulated history (sufficient to detect draws by repetition)
  /// are used to avoid GHI problems.
  /// 
  /// The benefit is correct recognition of draw by repetition situations.
  /// However creating separate nodes for each distinct path increase engine overhead
  /// and fails to share information between positions that are identical except for history
  /// (although the pseudotransposition blending feature can partly correct for that).
  /// </summary>
  PositionAndHistoryEquivalence
};


/// <summary>
/// MCGS coefficients and parameters related to core search algorithm.
/// The nested object ParamsSearchExecution contains parameters 
/// related to implementation (such as batch sizing and parallelization)
/// that may be dynamically changed tuned for each batch based on its characteristics.
/// </summary>
[Serializable]
public record ParamsSearch
{
  /// <summary>
  /// Default fraction of physical RAM to use as maximum memory for each search
  /// (for machines with <= 512GB of RAM).
  /// </summary>
  public const double DEFAULT_MAX_MEMORY_FRACTION_LE_512GB = 0.45;

  /// <summary>
  /// Default fraction of physical RAM to use as maximum memory for each search
  /// (for machines with > 512GB of RAM).
  /// </summary>
  public const double DEFAULT_MAX_MEMORY_FRACTION_GT_512GB = 0.32;

  /// <summary>
  /// Default value used by LC0 for smart pruning.
  /// This corresponds to somewhat to the Ceres MoveFutilityPruningAggressiveness value.
  /// For compatability with LC0 we accept only two value in UCI interface,
  /// interpreting 0 as "off" and 1.33 meaning "default" for both programs.
  /// </summary>
  public const float LC0_DEFAULT_SMART_PRUNING_FACTOR = 1.33f;

  public enum BestMoveModeEnum
  {
    /// <summary>
    /// The move receiving the most number of visits (N) is chosen (traditional AlphaZero).
    /// </summary>
    TopN,

    /// <summary>
    /// The move having the best Q value is chosen, subject to 
    /// also having certain minimum number of visits.
    /// </summary>
    TopQIfSufficientN,

    /// <summary>
    /// The move having the best Q value is chosen, subject to
    /// also having certain (less permissive) minimum number of visits
    /// </summary>
    TopQIfSufficientNPermissive,

    /// <summary>
    /// Use policy regularization (low intensity) using the method described in:
    /// "Monte-Carlo tree search as regularized policy optimization" by Grill et al (2020)
    /// </summary>
    RegularizedPolicyOptimizationLow,

    /// <summary>
    /// Use policy regularization (high intensity) using the method described in:
    /// "Monte-Carlo tree search as regularized policy optimization" by Grill et al (2020)
    /// </summary>
    RegularizedPolicyOptimizationHigh
  };

  /// <summary>
  /// Method for choosing which move is best to make at end of search.
  /// Using Q instead of the AlphaZero approach of N (in most situations)
  /// seems to slightly improve play quality and opens the possibilty
  /// of experimenting with leaf selection strategies which are more exploratory.
  /// </summary>
  public BestMoveModeEnum BestMoveMode = BestMoveModeEnum.TopQIfSufficientN;


  /// <summary>
  /// Implementation related parameters related to execution of a batch
  /// which are potentially recomputed based on the characterstic of the batch.
  /// </summary>
  public ParamsSearchExecution Execution;

  /// <summary>
  /// If graph features should be enabled (allowing cycles).
  /// </summary>
  public bool EnableGraph = true;

  /// <summary>
  /// Mode to be used when processing transpositions.
  /// </summary>
  public PathMode PathTranspositionMode = PathMode.PositionEquivalence;


  /// <summary>
  /// If backed up value score from first visit to a leaf nodes is potentially automatically extended
  /// to blend in the value score from the position resulting from its top policy move.
  /// Preliminary test results not encouraging:
  ///   - numerous matches show slightly lower Elo (circa 10)
  ///   - suite test at 10sec on H100 inferior: fewer solved, longer solve time: (257   4.81   4.88  1205.06 1160.85)
  /// </summary>
  public bool EnablePVAutoExtend = false;

  /// <summary>
  /// If full graph validation should be performed after each search.
  /// </summary>
  public bool ValidateAfterSearch = false;


  /// <summary>
  /// If PTB (pseudo-transposition blending) should be enabled where
  /// Q-values are blended across nodes corresponding with identical standalone hash values,
  /// i.e. identical board positions totally ignoring history.
  /// 
  /// This feature allows utilizing some of the information in the subgraphs
  /// corresponding to pseudo-transpositions, even though 
  /// they differ with respect to draw by repetition detection 
  /// and cannot be used to actually merge the subgraphs.
  /// 
  /// It was found necessary (especially at longer time controls) to carefully
  /// adjust the weight of the pseudotranspositions in the value blend to avoid
  /// distortion arising from the fact that the pseudotransposition subgraphs
  /// sometimes have much larger visit counts (N) than the node being updated
  /// (and surrounding nodes). It might naively seem that using a higher weight
  /// in the blend for large N subgraphs would be optimal because they contain
  /// the most information, but in practice this does not seem to be the case.
  /// However in fact it apparently distorts the relative evaluations being performed 
  /// at the current node and surrounding nodes (e.g. the parent node).
  /// Therefore the weight of large-N pseudotranspositions is tightly constrained.
  /// </summary>
  public bool EnablePseudoTranspositionBlendingInPositionAndEquivalenceMode = true;

  /// <summary>
  /// If pseudotransposition blending is enabled.
  /// This is only applicable in PositionAndHistoryEquivalence
  /// (otherwise the nodes sharing same position are already coalesced).
  /// </summary>
  public bool EnablePseudoTranspositionBlending => EnablePseudoTranspositionBlendingInPositionAndEquivalenceMode
                                                && PathTranspositionMode == PathMode.PositionAndHistoryEquivalence;

  // Optionally stop descent at a transposition node only if it has a
  // sufficiently large number of visits. 
  // This parameter is the cutoff value of a transposed child’s visit count
  // relative to to the edge's requested visits (as a ratio)
  // triggering stop of descent and instead directly backing up the child’s value.
  // Larger values encourages exploring more deeply those nodes that are often visited
  // from many parents (and therefore may have more importance to accurately evaluate).
  // Tests with T81 on HOP at 240s+4s versus Stockfish at various levels (Elo +/-18):
  //   1.5 --> -70 Elo
  //   3.0 --> -42 Elo
  //   5.0 --> -50 Elo
  //  20.0 --> -52 Elo
  public float TranspositionStopMinSupportRatioPositionAndHistoryMode = 3f;


  /// <summary>
  /// MinSupportRatio when running in Position mode.
  /// Because visits are coalesced across all transposition paths,
  /// the expectation for subgraph size is typically higher than with PositionAndHistory mode.
  /// </summary>
  public float TranspositionStopMinSupportRatioPositionMode = 5f;


  /// <summary>
  /// Threshold after which suboptimal visit choices (according to PUCT)
  /// are rejected as being too suboptimal.
  /// </summary>
  public float? VisitSuboptimalityRejectThreshold = null;

  public enum MoveOrderingPhaseEnum
  {
    /// <summary>
    /// No move reordering is applied.
    /// </summary>
    None,

    /// <summary>
    /// Reordering is applied once at time of node initialization (first leaf visit).
    /// </summary>
    NodeInitialization,

    /// <summary>
    /// Reordering is applied each time child selection is performed on a node.
    /// </summary>
    ChildSelection,

    /// <summary>
    /// Reordering applied at both.
    /// </summary>
    NodeInitializationAndChildSelect,
  }

  /// <summary>
  /// Specifies the phase in which to apply possible move reordering (if any).
  /// The idea is to enhance information reuse within the graph.
  /// 
  /// During selection, moves which are candidates for first visit
  /// may link to an existing subgraph via the standalone hashcode.
  /// The Q from the root of this subgraph can be compared to other 
  /// not-yet-visited children to possibly reorder the move list
  /// so that the better Q children are visited earlier.
  /// 
  /// Extensive tests suggest that enabling move reordering does gain Elo but 
  /// only very modestly (by perhaps 5 to 7 Elo).
  /// This is probably because:
  ///   - the policy prior is already quite accurate
  ///   - this leaks information from dissimilar graph depths perhaps making comparisons difficult
  ///   - search speed is somewhat slowed due to additional computation (transposition table lookups)
  /// </summary>
  public MoveOrderingPhaseEnum MoveOrderingPhase = MoveOrderingPhaseEnum.None;


  /// <summary>
  /// If backup updates should potentially be applied to antecednet nodes
  /// beyond those directly on the visit path that gave rise to a leaf visit
  /// (recursively upward to specified number of levels).
  /// Value of 0 disables the feature.
  /// Value of 1 propagates only to immediate parents and is known threadsafe (but does not seem to gain Elo).
  /// Values greater than 1 should not be used due to threadsafety concerns.
  /// </summary>
  public int OffPathBackupNumAdditionalLevelsToPropagate = 0;

  /// <summary>
  /// If the exploration bonus mulitplier (CPUCT) should be attenuated
  /// when a path already contains one or more highly exploratory 
  /// ancestor selections (i.e., selections influenced heavily by the U term).
  /// 
  /// Motivation: prevents compounding exploration noise along speculative lines,
  /// so that two or more noisy choices don't drown out true signal.
  /// </summary>
  public bool EnablePathDependentCPUCTScaling = false;

  /// <summary>
  /// Maximum number of nodes allowed in the search graph.
  /// Some internal efficiencies may result if a smaller value 
  /// than the large default is specified.
  /// </summary>
  public int MaxNodes = 1_100_000_000; // pending testing expand to at least: 2_001_000_000;

  /// <summary>
  /// Maximum RAM (in bytes) that the engine should use, 0 for no limit.
  /// </summary>
  public long MaxMemoryBytes = HardwareManager.MemorySize > (512L * 1024L * 1024L * 1024L) 
                                 ? (long)(HardwareManager.MemorySize * DEFAULT_MAX_MEMORY_FRACTION_GT_512GB)
                                 : (long)(HardwareManager.MemorySize * DEFAULT_MAX_MEMORY_FRACTION_LE_512GB);
  /// <summary>
  /// If the use of parent state information should be enabled (if supported by the evaluator).
  /// </summary>
  public bool EnableState = false;

  /// <summary>
  /// Set of prefetch parameters (optional).
  /// </summary>
  public ParamsPrefetch PrefetchParams;

  /// <summary>
  /// If supplemental progress logging to Console and verification steps are
  /// performed (useful for debugging/statistics collection).
  /// </summary>
  public bool DebugDumpVerifyMode = false;

  public bool EnableEarlySmallBatchSizes = false;

  /// <summary>
  /// Optionally the ID of a delegate registered with ParamsSearchExecutionModifier
  /// which will be called before each batch to allow customization
  /// of the nested Execution field members.
  /// </summary>
  public string ExecutionModifierID = null;


  /// <summary>
  /// Parameters relating to the secondary neural network evaluator (if in use).
  /// </summary>
  public ParamsSearchSecondaryEvaluator ParamsSecondaryEvaluator;

  /// <summary>
  /// Temperature to be applied to the value head output from the neural network evaluations.
  /// Values less than 1 sharpen the distribution, values greater than 1 flatten it.
  /// </summary>
  public float ValueTemperature = 1.0f;


  /// <summary>
  /// If the ParamsSearchExecution should be automatically
  /// reoptimized based on the characteristics of each batch
  /// (otherwise left unchanged).
  /// </summary>
  public bool AutoOptimizeEnabled = true;

  /// <summary>
  /// If the graph generated from a prior move will potentially
  /// be carried forward as a starting point for the subsequent search.
  /// 
  /// Note that the search parameters used may also be impacted by this setting,
  /// (for example, investing in more visits for the best move represents a deferred asset).
  /// Therefore this setting should only be enabled when graph reuse is possible 
  /// (i.e. games rather than single test positions).
  /// 
  /// TODO: Consider speeding up via one or both of:
  ///   - allocate a second store (transiently) and just copy nodes over - possibly much faster
  ///   - just keep the old nodes in the graph and change the root (at least until/if it has too much wasted space)
  /// </summary>
  public bool GraphReuseEnabled = true;

  /// <summary>
  /// If graph reuse may possibly make use of swapping root node into place
  /// rather than rewriting entire graph. This can consume additional memory
  /// but reduce time spent preparing graph for reuse.
  /// </summary>
  public bool GraphReuseRewriteEnabled = false;// !CeresUserSettingsManager.Settings.ReducedMemoryMode;

  /// <summary>
  /// If another search graph should be consulted to possibly reuse NN evaluations.
  /// The other graph might for example come from the opponent when playing a tournament, 
  /// or a comparison graph when doing suite testing.
  /// This will only take effect if the neural networks and certain other parameters are compatible/identical.
  /// </summary>
  /// TODO: move this into SuiteTestDef and TournamentDef instead.
  public bool ReusePositionEvaluationsFromOtherGraph = false;

  /// <summary>
  /// Scaling factor to batch sizes:
  ///    - first a good default batch size is estimated based on 
  ///      characteristics of hardware and search,
  ///    - then this default value is scaled up/or down by the BATCH_SIZE_MULTIPLIER
  /// </summary>
  public float BatchSizeMultiplier = 1.0f;

  /// <summary>
  /// In the case that only a single position (without any history)
  /// is being searched (e.g. from a FEN or at the beginning of a game),
  /// this flag controls if the single position should be replicated across
  /// all 8 history planes. For some networks this can influence NN output considerably,
  /// and it is believed that this is best set a true, i.e. that having
  /// "all zeros" for the history planes  is a bad idea because it
  /// generates inputs discontinuous with what the network saw in training 
  /// (except possibly at the start position).
  /// </summary>
  [CeresOption(Name = "history-fill-in", Desc = "If the history planes ", Default = "true")]
  public bool HistoryFillIn = true;


  /// <summary>
  /// If tablebases are enabled.
  /// This will be initialized according to the value specifid by user in Ceres user settings
  /// (if DirTablebases is empty or not.)
  /// </summary>
  [CeresOption(Name = "tablebases", Desc = "Enable external endgame tablebases", Default = "true")]
  public bool EnableTablebases = true;

  /// <summary>
  /// Path(s) to tablebase files.
  /// </summary>
  public string TablebasePaths = null;

  /// <summary>
  /// If CheckmateKnownToExist at chlid should cause 
  /// greatly reduced exploration at nodes with checkmate children
  /// so that few visits are wasted expanding the graph when outcome already known.
  /// NOTE: const for efficiency reasons.
  /// </summary>
  public const bool CheckmateCertaintyPropagationEnabled = true;

  /// <summary>
  /// If search considers positions arising twice as already a draw.
  /// Seems to slightly improve play quality due to early detection of draw by repetition subgraphs.
  /// </summary>
  public bool TwofoldDrawEnabled = true;

  /// <summary>
  /// If nodes should apply supplemental updates.
  /// </summary>
  public bool EnableGraphCatchUp = false;

  /// <summary>
  /// If searches are possibly terminated early if it is determined the top move
  /// is unlikely or impossible to change before search ends due to time or nodes limit.
  /// Note that when using nodes or time per move will be strictly inferior if graph reuse is enabled.
  /// </summary>
  [CeresOption(Name = "early-stop-search-enabled", Desc = "If searches are possibly exited early due to leading move being ahead.", Default = "true")]
  public bool FutilityPruningStopSearchEnabled = true;


  /// <summary>
  /// Aggressiveness with which searches from moves at the root of the search are pruned
  /// (including the best/only remaining move if FutilityPruningStopSearchEnabled is true)
  /// from further visits due to impossibility or implausability that they will be the best move.
  /// </summary>
  [CeresOption(Name = "move-futility-pruning-aggressiveness", Desc = "Aggresiveness for early termination of searches to less promising root search subgraphs in range [0..1.5], 0 disables.", Default = "0.4")]
  public float MoveFutilityPruningAggressiveness = 0.4f;


  /// <summary>
  /// Aggressiveness with which limited search resource (time or nodes) is consumed.
  /// </summary>
  [CeresOption(Name = "time-management-aggressiveness", Desc = "Aggressiveness with which limited search resource (time or nodes) is consumed.", Default = "1.0")]
  public float GameLimitUsageAggressiveness = 1.0f;


  /// <summary>
  /// If moves are possibly made more quickly if a large graph 
  /// from reuse is already available and the best move also seems relatively clear.
  /// QuickMoves are only made if FutilityPruningStopSearchEnabled is also true.
  /// </summary>
  public bool EnableQuickMoves = true;

  /// <summary>
  /// If nonzero, top-level moves within this Q distance (e.g. 0.05) of the best move
  /// are considered as alternate choices if they appear to lead to
  /// more concrete progress (irreverisble moves in the principal variation)
  /// than the otherwise best move.
  /// </summary>
  public float AntiShufflingQThreshold = 0f;

  /// <summary>
  /// If the limits (time or nodes) initially allocated for a search may
  /// possibly be extended when multiple moves are close to equal in score (N and/or Q).
  /// </summary>
  public bool EnableSearchExtension = false; // TODO: consider enabling this


  // Due to immediate nodes (cache hits, tablebase hit), or collisions or terminals the number of nodes 
  // sent to the NN will generally be less than the requested batch size.
  // The optional PADDED feature expands the batch size somewhat to compensate for this
  // This can help fill out batches toward the full size (which may be set at level optimized for the hardware)
  // Any nodes above the requested batch size are set to CacheOnly
  // Although performance for small to medium searches is typically improved by 5% to 10%,
  // there is typically some reduction in search purity/quality due to "out-of-order" effect
  public bool PaddedBatchSizing = false;
  public int PaddedExtraNodesBase = 5;
  public float PaddedExtraNodesMultiplier = 0.03f;


  /// <summary>
  /// If nonzero, the V stored for the node is no longer the exact value from the value head,
  /// but rather a partial blend with the action head value.
  /// </summary>
  public float ActionHeadWeightInV = 0f;

  /// <summary>
  /// Nonzero values cause greater exploration at nodes with high uncertainty
  /// (applied based on node uncertainty).
  /// </summary>
  public float SelectExplorationForUncertaintyAtNode = 0;

  /// <summary>
  /// Nonzero values cause greater exploration at nodes with high uncertainty
  /// (applied based on edge uncertainty).
  /// </summary>
  public float SelectExplorationForUncertaintyAtEdge = 0;

  /// <summary>
  /// If the "uncertainty boosting" feature should be enabled.
  /// It adds a small uncertainty-related tweak to the leaf selection formula 
  /// (similar ideas such as UCB-V for classical UCB has already been explored in other domains). 
  ///  It optimistically slightly incentivizes exploration at nodes with
  ///  high uncertainty by inserting a scaling factor in the exploration term which is 
  ///  calibrated to have mean 1 (with outlier truncation and shrinkage) 
  ///  and increasing in the uncertainty (standard deviation) of the backed-up nodes seen so far. 
  /// </summary>
  /// 
  public bool EnableUncertaintyBoosting = false;

  /// <summary>
  /// If the policy uncertainty value should be used to 
  /// adjust the per-position temperature on the policy.
  /// </summary>
  public bool EnablePolicyUncertaintyTemperatureBoosting = false;


  /// <summary>
  /// Amount of time subtracted from time allotments to 
  /// compensate for lag or various unpredictable latencies.
  /// </summary>
  public float MoveOverheadSeconds = 0.25f;


  /// <summary>
  /// Optional flag that can be defined by developers for ad-hoc testing.
  /// </summary>
  public bool TestFlag = false;

  /// <summary>
  /// Optional (secondary)flag that can be defined by developers for ad-hoc testing.
  /// </summary>
  public bool TestFlag2 = false;


  /// <summary>
  /// Optional scalar that can be defined by developers for ad-hoc testing.
  /// </summary>
  public float TestScalar = 0.0f;


  /// <summary>
  /// Constructor (uses default values for the class unless overridden in settings file).
  /// </summary>
  public ParamsSearch()
  {
    if (CeresUserSettingsManager.Settings.SmartPruningFactor.HasValue)
    {
      float pruningFactorValue = CeresUserSettingsManager.Settings.SmartPruningFactor.Value;
      if (pruningFactorValue == 0)
      {
        FutilityPruningStopSearchEnabled = false;
        MoveFutilityPruningAggressiveness = 0;
      }
      else if (pruningFactorValue == LC0_DEFAULT_SMART_PRUNING_FACTOR)
      {
        FutilityPruningStopSearchEnabled = new ParamsSearch().FutilityPruningStopSearchEnabled;
        MoveFutilityPruningAggressiveness = new ParamsSearch().MoveFutilityPruningAggressiveness;
      }
      else
      {
        throw new Exception("SmartPruningFactor in Ceres.json must either have value 0, 1.33 or be absent.");
      }
    }

    if (CeresUserSettingsManager.Settings.EnableUncertaintyBoosting.HasValue)
    {
      EnableUncertaintyBoosting = CeresUserSettingsManager.Settings.EnableUncertaintyBoosting.Value;
    }

    if (CeresUserSettingsManager.Settings.ValueTemperature.HasValue)
    {
      ValueTemperature = CeresUserSettingsManager.Settings.ValueTemperature.Value;
    }

    TablebasePaths = CeresUserSettingsManager.Settings.TablebaseDirectory;

    // Start with default execution params,
    // but these may be updated dynamicaly during search
    // based on search state.
    Execution = new ParamsSearchExecution();


    ParamsSecondaryEvaluator = new();
  }


  /// <summary>
  /// Validates all settigs for self-consistency.
  /// </summary>
  public void Validate()
  {
    if (OffPathBackupNumAdditionalLevelsToPropagate > 1
     && Execution.BackupMode == BackupMethodEnum.ReductionMultiThread)
    {
      // Using 1 level is always safe because we merely set the IsStale flag (all updates happen in select phase).
      // But more than 1 level would require locking nodes which could likely cause deadlock during backup.
      throw new Exception("Off-path backup propagation not supported with multithreaded backup");
    }

    if (!EnableGraph && PathTranspositionMode == PathMode.PositionAndHistoryEquivalence)
    {
      throw new Exception("Cannot use PathTranspositionMode PositionAndHistoryEquivalence when graph is disabled.");
    }

    if (EnablePseudoTranspositionBlending &&
        !MCGSParamsFixed.REFRESH_SIBLING_DURING_SELECT_PHASE
      && !MCGSParamsFixed.REFRESH_SIBLING_DURING_BACKUP_PHASE)
    {
      throw new Exception("PseudoTranspositionBlending requires sibling info to be updated during select or backup.");
    }
  
    if (SelectExplorationForUncertaintyAtNode > 0 && !MCGSParamsFixed.TRACK_NODE_EDGE_UNCERTAINTY)
    {
      throw new NotImplementedException("Nonzero SelectExplorationForUncertaintyAtNode requires UNCERTAINTY_TESTS_ENABLED to be true.");
    }

    if (SelectExplorationForUncertaintyAtEdge > 0 && !MCGSParamsFixed.TRACK_NODE_EDGE_UNCERTAINTY)
    {
      throw new NotImplementedException("Nonzero SelectExplorationForUncertaintyAtEdge requires UNCERTAINTY_TESTS_ENABLED to be true.");
    }
  }
}
