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
using Ceres.Base.Math;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Search.IteratedMCTS;

#endregion

namespace Ceres.MCTS.Params
{
  /// <summary>
  /// MCTS coefficients and parameters related to core search algorithm.
  /// The nested object ParamsSearchExecution contains parameters 
  /// related to implementation (such as batch sizing and parallelization)
  /// that may be dynamically changed tuned for each batch based on its characteristics.
  /// </summary>
  [Serializable]
  public record ParamsSearch
  {
    /// <summary>
    /// Default value used by LC0 for smart pruning.
    /// This corresponds to somewhat to the Ceres MoveFutilityPruningAggressiveness value.
    /// For compatability with LC0 we accpet only two value in UCI inerface,
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
      /// The move having the best Q value is chose, subject to 
      /// also having certain minimum minimum number of visits.
      /// </summary>
      TopQIfSufficientN,
    };

    /// <summary>
    /// Method for choosing which move is best to make at end of search.
    /// Using Q instead of the AlphaZero approach of N (in most situations)
    /// seems to slightly improve play quality and opens the possibilty
    /// of experimenting with leaf selection strategies which are more exploratory.
    /// </summary>
    public BestMoveModeEnum BestMoveMode = BestMoveModeEnum.TopQIfSufficientN;

    /// <summary>
    /// Experimental. If a bonus should be given to children which have a 
    /// positive trend in their recent backed-up V values.
    /// This is found to help modestly with position testing and Ceres tests,
    /// but probably perform meaningfully worse against Stockfish (up to 50 Elo).
    /// </summary>
    public bool ApplyTrendBonus = false;


    /// <summary>
    /// Experimental. If a bonus should be applied to favor low M (moves left)
    /// when not winning by at least contempt threshold, otherwise favor high M.
    /// 0.003f
    /// </summary>
    public float MLHBonusFactor = 0;

    /// <summary>
    /// Implementation related parameters related to execution of a batch
    /// which are potentially recomputed based on the characterstic of the batch.
    /// </summary>
    public ParamsSearchExecution Execution;

    /// <summary>
    /// Optionally the ID of a delegate registered with ParamsSearchExecutionModifier
    /// which will be called before each batch to allow customization
    /// of the nested Execution field members.
    /// </summary>
    public string ExecutionModifierID = null;

    /// <summary>
    /// Experimental. If an "iterated" MCTS strategy should be followed in search,
    /// involving building trees of intermediate size and then resetting them
    /// and rewriting the P priors from the neural network partly based on
    /// the emprical policy distribution (to reduce memory consumption or possibly improve play).
    /// </summary>
    public IteratedMCTSDef IMCTSSchedule = null;

    /// <summary>
    /// Parameters relating to the secondary neural network evaluator (if in use).
    /// </summary>
    public ParamsSearchSecondaryEvaluator ParamsSecondaryEvaluator;

    /// <summary>
    /// Optional policy used for applying Dirichlet noise at root.
    /// </summary>
    public SearchNoisePolicyDef SearchNoisePolicy;

    /// <summary>
    /// Parameters of the Dirichlet noise process.
    /// </summary>
    public SearchNoiseBestMoveSamplingDef SearchNoiseBestMoveSampling;

    /// <summary>
    /// If the ParamsSearchExecution should be automatically
    /// reoptimized based on the characteristics of each batch
    /// (otherwise left unchanged).
    /// </summary>
    public bool AutoOptimizeEnabled = true;

    /// <summary>
    /// If the tree generated from a prior move will potentially
    /// be carried forward as a starting point for the subsequent search.
    /// 
    /// Note that the search parameters used may also be impacted by this setting,
    /// (for example, investing in more visits for the best move represents a deferred asset).
    /// Therefore this setting should only be enabled when tree reuse is possible 
    /// (i.e. games rather than single test positions).
    /// 
    /// TODO: Consider speeding up via one or both of:
    ///   - allocate a second store (transiently) and just copy nodes over - possibly much faster
    ///   - just keep the old nodes in the tree and change the root (at least until/if it has too much wasted space)
    /// </summary>
    public bool TreeReuseEnabled = true;

    /// <summary>
    /// If tree reuse may possibly make use of swapping root node into place
    /// rather than rewriting entire tree. This can consume additional memory
    /// but reduce time spent preparing tree for reuse.
    /// </summary>
    public bool TreeReuseSwapRootEnabled = !CeresUserSettingsManager.Settings.ReducedMemoryMode;

    /// <summary>
    /// If reachable nodes in search tree undergoing rebuild are retained in a separate cache.
    /// However seemingly not useful because only about 4% of positions saved are actually subsequently used.
    /// </summary>
    public bool TreeReuseRetainedPositionCacheEnabled = false;

    /// <summary>
    /// If another serach tree should be consulted to possibly reuse NN evaluations.
    /// The other tree might for example come from the opponent when playing a tournament, 
    /// or a comparision tree when doing suite testing.
    /// This will only take effect if the neural networks and certain other parameters are compatible/identical.
    /// </summary>
    /// TODO: move this into SuiteTestDef and TournamentDef instead.
    public bool ReusePositionEvaluationsFromOtherTree = true;

    /// <summary>
    /// Contempt value penalizes draw outcome (if nonzero) 
    /// potentially improving average win rate against lessor opponents
    /// or possibly reducing draw rate in self-play testing scenarios.
    /// 
    /// The actual contempt applied to positions is a penalty applied to the 
    /// the value score of evaluted nodes which is the product of the
    /// Contempt and the node's draw probability.
    /// </summary>
    [CeresOption(Name = "contempt", Desc = "Contempt coefficient which penalizes draw outcomes", Default = "0.0")]
    public float Contempt = 0.0f;

    /// <summary>
    /// Experimental. The fraction of the final contempt which is derived from an
    /// auto scaled value estimated continuously and dynamically,
    /// based on degree of agreement between opponents moves and
    /// estimated  suboptimality relative to Ceres estimated optimal move.
    /// The remainder of the weight comes from the baseline CONTEMPT factor.
    /// 
    /// Nonzero values are probably only appropriate when it is believed
    /// that Ceres will be stronger than opponents
    /// (otherwise she is in no position to accurately assess suboptimality).
    /// </summary>
    public float ContemptAutoScaleWeight = 0.0f;

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
    /// The number of plies to look back during search when evaluating if
    /// a node should be draw by repetition.
    /// 
    /// To detect draws by 3-fold repetition, according to the rules of chess
    /// we need to look back over full game, but for efficiency reasons we 
    /// typically restrict lookback to a limited window.
    /// </summary>
    public const int DrawByRepetitionLookbackPlies = 22;


    /// <summary>
    /// If CheckmateKnownToExist at chlid should cause 
    /// suspension of exploration at nodes with checkmate children.
    /// NOTE: const for efficiency reasons.
    /// </summary>
    public const bool CheckmateCertaintyPropagationEnabled = true;

    /// <summary>
    /// If search considers positions arising twice as already a draw.
    /// Seems to slightly improve play quality due to early detection of draw by repetition subtrees.
    /// </summary>
    public bool TwofoldDrawEnabled = true;


    /// <summary>
    /// If searches are possibily terminated early if it is determined the top move
    /// is unlikely or impossible to change before search ends due to time or nodes limit.
    /// Note that when using nodes or time per move will be strictly inferior if tree reuse is enabled.
    /// </summary>
    [CeresOption(Name = "early-stop-search-enabled", Desc = "If searches are possibly exited early due to leading move being ahead.", Default = "true")]
    public bool FutilityPruningStopSearchEnabled = true;


    /// <summary>
    /// Aggressiveness with which searches from moves at the root of the search are pruned
    /// (including the best/only remaining move if FutilityPruningStopSearchEnabled is true)
    /// from further visits due to impossibility or implausability that they will be the best move.
    /// </summary>
    [CeresOption(Name = "move-futility-pruning-aggressiveness", Desc = "Aggresiveness for early termination of searches to less promising root search subtrees in range [0..1.5], 0 disables.", Default = "0.4")]
    public float MoveFutilityPruningAggressiveness = 0.4f;


    /// <summary>
    /// Aggressiveness with which limited search resource (time or nodes) is consumed.
    /// </summary>
    [CeresOption(Name = "time-management-aggressiveness", Desc = "Aggressiveness with which limited search resource (time or nodes) is consumed.", Default = "1.0")]
    public float GameLimitUsageAggressiveness = 1.0f;


    /// <summary>
    /// If moves are possibly made instantly (with no search) if a large tree 
    /// from reuse is already available and the best move also seems relatively clear.
    /// Instamoves are only made if FutilityPruningStopSearchEnabled is also true.
    /// </summary>
    public bool EnableInstamoves = true;

    /// <summary>
    /// If the limits (time or nodes) initially allocated for a search may
    /// possibly be extended when multiple moves are close to equal in score (N and/or Q).
    /// </summary>
    public bool EnableSearchExtension = true;


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
    /// Hard fixed limit on maximum number of times (up to 3)
    /// evaluations will be taken from a transposition root
    /// (used only if transposition mode is SingleNodeDeferredCopy).
    /// 
    /// This feature offers two benefits:
    ///   - the nodes are not instantiated in the tree until this value is exceeded
    ///     (instead the data neeeded during searc visits is 
    ///      plucked from the transposition root subtree).
    ///   - optionally some of the Q (subtree average) from the transposition root
    ///     can be mixed in with these visits (see TranspositionRootQFraction).
    /// </summary>
    public int MaxTranspositionRootUseCount => StatUtils.CountNonNaN(TranspositionRootBackupSubtreeFracs);

    
    /// <summary>
    /// Fractional weight given to transposition root when sending
    /// backup value added into node accumulators (e.g. W) in the tree.
    /// 
    /// (rather than directly to node being visited at or under transposition root)
    /// when computing value to be backed up tree after a visit
    /// to a node still linked to a transposition root.
    /// a transposition root.
    ///
    /// Since up to 3 values can be borrowed from transposition root,
    /// an array of 3 values is provided indicating weight for each corresponding visit.
    /// 
    /// NaN values indicate no transposition root reuse is used for the corresponding N.
    /// 
    /// Nonzero values have the benefit of sharing information from possibly large
    /// subtrees already explored below the transposition root.
    /// 
    /// However large values have the disadvantage of distorting the evaluations
    /// at and above the node, effectively overweighting nodes deeper in the tree.
    /// 
    /// </summary>
    public float[] TranspositionRootBackupSubtreeFracs = new float[] { 1, 1, 1 };

    /// <summary>
    /// Fractional weight given to subtree averages (e.g. Q) from node being 
    /// copied (cloned) from transposition root subtree when materializing.
    /// </summary>
    public float[] TranspositionCloneNodeSubtreeFracs = new float[] { 1, 1, 1 };


    /// <summary>
    /// Experiemental feature that a initializes a new leaf
    /// with the averge V or Q across all nodes in tree which
    /// are equivalent.
    /// </summary>
    public bool TranspositionUseCluster = false; // not sure if better in games

    /// <summary>
    /// If values applied during backup are potentially taken from the root
    /// of corresponding transposition subtree (if any) in cases when
    /// that subtree is larger than the current node's subtree.
    /// Potentially highly beneficial for large searches (>100k and especially >1000k nodes).
    /// </summary>
    public bool EnableDeepTranspositionBackup = false;


    /// <summary>
    /// If the transposition table is maintained to enforce condition that 
    /// each node serving as root is the node with  maximal N 
    /// among all nodes in the tree in the same equivalence class
    /// (allows deep transposition backup to always use the most informed root).
    /// Seems beneficial (+7Elo +/-7, 83%, with T60 at 75knpm).
    /// </summary>
    public bool TranspositionRootMaxN = false;

    /// <summary>
    /// If evaluations of siblings not yet visited (derived from transpositions)
    /// should possibly be blended into backed up evaluations.
    /// Believed to have a small positive impact, especially for longer searches
    /// (although the benefit may be limited when deep transposition backup is also in use).
    /// </summary>
    public bool EnableUseSiblingEvaluations = false;

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

      if (CeresUserSettingsManager.Settings.EnableSiblingEval.HasValue)
      {
        EnableUseSiblingEvaluations = CeresUserSettingsManager.Settings.EnableSiblingEval.Value;
      }

      if (CeresUserSettingsManager.Settings.EnableUncertaintyBoosting.HasValue)
      {
        EnableUncertaintyBoosting = CeresUserSettingsManager.Settings.EnableUncertaintyBoosting.Value;
      }

      // Check user settings to see if tablebases are configured.
      EnableTablebases = CeresUserSettingsManager.Settings.TablebaseDirectory is not null;

      // Start with default execution params,
      // but these may be updated dynamicaly during search
      // based on search state.
      Execution = new ParamsSearchExecution();


      ParamsSecondaryEvaluator = new();
    }


  }
}
