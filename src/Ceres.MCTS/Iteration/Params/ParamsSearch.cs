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

using Ceres.Chess.UserSettings;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Search.IteratedMCTS;
using LINQPad.Controls;
using System;

#endregion

namespace Ceres.MCTS.Params
{
  /// <summary>
  /// MCTS coefficients and parameters related to core search algorithm
  /// and implementation details (nested within the ParamsSearchExecution field).
  /// </summary>
  [Serializable]
  public class ParamsSearch
  {
    /// <summary>
    /// If a set of small adjustments to parameters resulting from tuning tests
    /// should be applied (otherwise baseline LC0-like parameters retained).
    /// </summary>
    public const bool USE_CERES_ADJUSTMENTS = true;

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

    public BestMoveModeEnum BestMoveMode = BestMoveModeEnum.TopQIfSufficientN;

    /// <summary>
    /// If a bonus should be given to children which have a 
    /// positive trend in their recent backed-up V values.
    /// This is found to help modestly with position testing and Ceres tests,
    /// but probably perform meaningfully worse against Stockfish (up to 50 Elo).
    /// </summary>
    public bool ApplyTrendBonus = false;


    /// <summary>
    /// If a bonus should be applied to favor low M (moves left)
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
    /// If an "iterated" MCTS strategy should be followed in search,
    /// involving building trees of intermediate size and then resetting them
    /// and rewriting the P priors from the neural network partly based on
    /// the emprical policy distribution (to reduce memory consumption or possibly improve play).
    /// </summary>
    public IteratedMCTSDef IMCTSSchedule = null;

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
    ///         - allocate a second store (transiently) and just copy nodes over - possibly much faster
    ///         - just keep the old nodes in the tree and change the root (at least until/if it has too much wasted space)
    /// </summary>
    public bool TreeReuseEnabled = true;

    // SLOW AND BUGGY ***
    // Also we find that only about 2% of positions saved are actually subsequently used, 
    // so this feature is probably not useful
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
    [CeresOption(Name = "contempt", Desc = "Contempt coefficient which penalizes draw outcomes", Default = "0")]
    public float Contempt = 0.01f;

    /// <summary>
    /// The fraction of the final contempt which is derived from an
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
    ///    - first a good default batch size is estimated based on characteristics of hardware and search,
    ///    - then this default value is scaled up/or down by the BATCH_SIZE_MULTIPLIER
    /// </summary>
    public float BatchSizeMultiplier = 1.0f;

    [CeresOption(Name = "history-fill-in", Desc = "If the history planes ", Default = "true")]
    public bool HistoryFillIn = true;

    [CeresOption(Name = "tablebases", Desc = "Enable external endgame tablebases", Default = "false")]
    public bool EnableTablebases = true;

    // To detect draws by 3-fold repetition, in theory we need to look back over full game
    // but for efficiency reasons we restrict to last 22 positions

    /// <summary>
    /// The number of plies to look back during search when evaluating if
    /// a node should be draw by repetition.
    /// 
    /// According to the rules of chess the lookback should go to the beginning of game.
    /// However for efficiency reasons typically a limited window is used.
    /// </summary>
    public int DrawByRepetitionLookbackPlies = 22;

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
    /// from further visits due to impossibility or implausability that they will be the best move.
    /// TODO: make this smarter, possibly look at recent trends
    /// </summary>
    [CeresOption(Name = "move-futility-pruning-aggressiveness", Desc = "Aggresiveness for early termination of searches to less promising root search subtrees in range [0..1], 0 disables.", Default = "0.0")]
    public float MoveFutilityPruningAggressiveness = 1.0f;

    /// <summary>
    /// Aggressiveness with which limited search resource (time or nodes) is consumed.
    /// </summary>
    [CeresOption(Name = "time-management-aggressiveness", Desc = "Aggressiveness with which limited search resource (time or nodes) is consumed.", Default = "1.0")]
    public float GameLimitUsageAggressiveness = 1.0f;

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
    /// Optionally leaf selection can be influenced
    /// by a specified amount of noise applied to Q
    /// (applied only in case child already has N > 1000).
    /// Suggested value: 0.04.
    /// </summary>
    public float QNoiseFactorRoot = 0.0f;

    /// <summary>
    /// Optionally leaf selection can be influenced
    /// by a specified amount of noise applied to Q
    /// (applied only in case child already has N > 1000).
    /// Suggested value: 0.02.
    /// </summary>
    public float QNoiseFactorNonRoot = 0.0f;


    /// <summary>
    /// If the V take from a node found to be a transposition
    /// should be the Q from the whole transposition subtree 
    /// rather than just the transposed node.
    /// </summary>
    public bool TranspositionUseTransposedQ = true; 

                                                   
    public bool TranspositionUseCluster = false; // not sure if better in games


    /// <summary>
    /// Optinal flag that can be defined by developers for ad-hoc testing.
    /// </summary>
    public bool TestFlag = false;

    /// <summary>
    /// Constructor.
    /// </summary>
    public ParamsSearch()
    {
      // Check user settings to see if tablebases are configured.
      EnableTablebases = CeresUserSettingsManager.Settings.DirTablebases != "";

      // Start with default execution params,
      // but these may be updated dynamicaly during search
      // based on search state.
      Execution = new ParamsSearchExecution();
    }
   

  }
}
