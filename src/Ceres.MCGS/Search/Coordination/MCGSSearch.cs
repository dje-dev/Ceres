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
using System.Collections.Generic;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Managers.Limits;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PathEvaluators;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Phases.Evaluation;

using static Ceres.MCGS.Search.Coordination.MCGSManager;
using static Ceres.MCGS.Search.Phases.MCGSSelect;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Entry point for launching a search and capturing the results.

/// Note that in most situations instead the class GameEngineCeresInProcess 
/// is preferrable as the entry point for launching searches.
/// </summary>
public partial class MCGSSearch
{
  /// <summary>
  /// Optional delegate that registers to receive informational messages that should be logged.
  /// </summary>
  /// <param name="infoMessage"></param>
  public delegate void MCGSInfoLogger(string infoMessage);

  /// <summary>
  /// The underlying serach manager.
  /// </summary>
  public MCGSManager Manager { get; internal set; }

  /// <summary>
  /// Selected best move from last search.
  /// </summary>
  public MGMove BestMove { get; private set; }

  /// <summary>
  /// Node within the graph from which the search starts.
  /// </summary>
  public GNode SearchRootNode => Manager.Engine.SearchRootNode;

  /// <summary>
  /// N of the SearchRootNode when at beginning of search.
  /// </summary>
  public int StartSearchN { get; private set; }

  /// <summary>
  /// Total number of searches conducted.
  /// </summary>
  public static int SearchCount { get; internal set; }


  #region Graph reuse related

  /// <summary>
  /// The number of times a search from this tree
  /// has been satisfied out of tree reuse (no actual search).
  /// </summary>
  public int CountSearchContinuations { get; private set; }


  /// <summary>
  /// Optional delegate that registers to receive informational messages that should be logged.
  /// </summary>
  /// <param name="search"></param>
  /// <param name="infoMessage"></param>
  public readonly MCGSInfoLogger InfoLogger;

  #endregion


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="infoLogger"></param>
  public MCGSSearch(MCGSInfoLogger infoLogger = null)
  {
    InfoLogger = infoLogger;
  }

  /// <summary>
  /// Runs a new search.
  /// </summary>
  /// <param name="nnEvaluators"></param>
  /// <param name="graphToPossiblyReuse"></param>
  /// <param name="paramsSelect"></param>
  /// <param name="paramsSearch"></param>
  /// <param name="limitManager"></param>
  /// <param name="priorMoves"></param>
  /// <param name="searchLimit"></param>
  /// <param name="verbose"></param>
  /// <param name="startTime"></param>
  /// <param name="gameMoveHistory"></param>
  /// <param name="progressCallback"></param>
  /// <param name="isFirstMoveOfGame"></param>
  /// <param name="fixedSearchLimit"></param>
  public void Search(NNEvaluatorSet nnEvaluators,
                     Graph graphToPossiblyReuse,
                     WorkerPool<ExtendPathsWorkerInfo>[] selectWorkerPools,
                     ParamsSelect paramsSelect,
                     ParamsSearch paramsSearch,
                     IManagerGameLimit limitManager,
                     PositionWithHistory priorMoves,
                     SearchLimit searchLimit,
                     bool verbose,
                     DateTime startTime,
                     List<GameMoveStat> gameMoveHistory,
                     MCGSProgressCallback progressCallback = null,
                     PositionEvalCache positionEvalCache = null,
                     bool isFirstMoveOfGame = false,
                     bool moveImmediateIfOnlyOneMove = false,
                     MGMove forcedMove = default,
                     SearchLimit fixedSearchLimit = null)
  {
    if (searchLimit == null)
    {
      throw new ArgumentNullException(nameof(searchLimit));
    }

    if (searchLimit.SearchCanBeExpanded && !MCGSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC)
    {
      throw new Exception("STORAGE_USE_INCREMENTAL_ALLOC must be true when SearchCanBeExpanded.");
    }

    if (!MCGSParamsFixed.STORAGE_USE_INCREMENTAL_ALLOC && !searchLimit.IsNodesLimit)
    {
      throw new Exception("SearchLimit must be NodesPerMove or NodesPerGame when STORAGE_USE_INCREMENTAL_ALLOC is false");
    }

    paramsSearch.Validate();
    paramsSelect.Validate();

    if (paramsSelect.RPOBackupLambda == 0 &&
      (paramsSearch.SelectExplorationForUncertaintyAtNode > 0
     || paramsSearch.SelectExplorationForUncertaintyAtNode > 0
     || MCGSParamsFixed.TRACK_NODE_EDGE_UNCERTAINTY))
    {
      throw new Exception("Currently uncertainty tracking only works in RPO mode (when select updates disabled)");
    }

    // TODO: clean this up?
    string tablebasePaths = paramsSearch.EnableTablebases ? paramsSearch.TablebasePaths : null;
    bool forceNoTablebaseTerminals = EvaluatorSyzygy.PosIsTablebaseWinWithNoDTZAvailable(tablebasePaths, priorMoves);

#if NOT
In MCGS version we build Sygyzy evaluator but tell it to instead return
neural network eval because WDL and not DTZ available
good example position: 8/5k2/8/6P1/8/7P/pBp1K3/8 b - - 1 51


    else
    {
      node.InfoRef.EvalResultAuxilliary = (FP16)result.V;
      return default;
    }

This isn't currently easy in the MCGS engine.
As a workaround, EvaluatorSygyzy will just return as if no hit.

#endif
    SearchLimit searchLimitAdjusted = AdjustedSearchLimit(searchLimit, paramsSearch);

#if NOT
// Have to disable this to avoid overflows
// The problem is probably that the sizing here doesn't account for the fact
// that the graph can grow thru reuse very large.

    int maxNodes;
    if (!searchLimitAdjusted.SearchCanBeExpanded && searchLimitAdjusted.IsNodesLimit)
    {
      maxNodes = (int)(searchLimitAdjusted.Value + searchLimitAdjusted.ValueIncrement + 5000);
    }
    else
    {
      // In this mode, we are just reserving virtual address space
      // from a very large pool (e.g. 256TB for Windows).
      // Therefore it is safe to reserve a very large block.
      if (searchLimitAdjusted.MaxTreeNodes != null)
      {
        // Reseve somewhat more storage than the maximum requested tree nodes
        // if the search can be expenaded because during tree rewrite
        // a preparatory step (MaterializeNodesWithNonRetainedTranspositionRoots) 
        // will initially make the store larger (before it is subsequently compacted).
        double NODES_BUFFER_MULTIPLIER = searchLimitAdjusted.SearchCanBeExpanded ? 1.2 : 1.0;
        long maxNodesLong = (long)(NODES_BUFFER_MULTIPLIER * searchLimitAdjusted.MaxTreeNodes.Value) + 100_000;
        maxNodes = (int)Math.Min(maxNodesLong, int.MaxValue - 100_000);
      }
      else
      {
        maxNodes = GraphStore.MAX_NODES;
      }
    }
#endif

    // Try to reuse the prior graph
    Graph graphToReuse = GraphReuseManager.TryReuseGraph(paramsSearch, priorMoves, graphToPossiblyReuse,
                                                        out GraphRootToSearchRootNodeInfo searchRootNodeInfo,
                                                        out List<GraphRootToSearchRootNodeInfo> searchRootPathFromGraphRoot);

    SearchLimit searchLimitToUse;
    ManagerGameLimitInputs gameLimitsInputs = null;
    ManagerGameLimitOutputs gameLimitsOutputs = null;
    if (!searchLimit.IsPerGameLimit)
    {
      searchLimitToUse = searchLimit;
    }
    else
    {
      SearchLimitType targetType = searchLimit.IsTimeLimit ? SearchLimitType.SecondsPerMove
                                                           : SearchLimitType.NodesPerMove;

      int searchRootNodeN = searchRootNodeInfo == default ? 0 : searchRootNodeInfo.ChildNode.N;
      double searchRootNodeQ = searchRootNodeInfo == default ? 0 : searchRootNodeInfo.ChildNode.Q;
      gameLimitsInputs = new(priorMoves.FinalPosition,
                           paramsSearch, gameMoveHistory,
                           targetType, searchRootNodeN, (float)searchRootNodeQ,
                           searchLimit.Value, searchLimit.ValueIncrement,
                           searchLimit.MaxTreeNodes, searchLimit.MaxTreeVisits,
                           float.NaN, float.NaN,
                           maxMovesToGo: searchLimit.MaxMovesToGo,
                           isFirstMoveOfGame: isFirstMoveOfGame, paramsSearch.EnableQuickMoves);

      gameLimitsOutputs = limitManager.ComputeMoveAllocation(gameLimitsInputs);
      searchLimitToUse = gameLimitsOutputs.LimitTarget;
    }


    List<MGMove> searchMovesTablebaseRestricted = null;
    if (searchLimit.SearchMoves != null)
    {
      Position startPos = priorMoves.FinalPosition;
      foreach (Move move in searchLimit.SearchMoves)
      {
        searchMovesTablebaseRestricted.Add(MGMoveConverter.MGMoveFromPosAndMove(startPos, move));
      }
    }

    Manager = new(nnEvaluators,
                  paramsSearch, paramsSelect,
                  searchLimitToUse, limitManager, startTime,
                  gameMoveHistory, isFirstMoveOfGame,
                  forceNoTablebaseTerminals,
                  searchMovesTablebaseRestricted, priorMoves.FinalPosition.IsWhite,
                  fixedSearchLimit)
    {
      LastGameLimitInputs = gameLimitsInputs,
      LastGameLimitOutputs = gameLimitsOutputs
    };



    // Process graph rewrite if needed (handles memory pressure, low ratio triggers)
    Graph graphToUse = GraphReuseManager.ProcessGraphRewrite(graphToReuse, searchRootNodeInfo, ref searchRootPathFromGraphRoot, paramsSearch, searchLimitToUse, priorMoves);

    // Create new graph if needed (either no prior graph, or prior graph was abandoned)
    if (graphToUse == null)
    {
      const long MAX_NODES = 1_100_000_000L;

      // Attempt to find some safe (hopefully lower value)
      // to use for max nodes than MAX_NODES to reduce virtual memory reservation length.
      // Note that if GraphReuseRewriteEnabled then we can be more aggressive in allowing more nodes
      // because we will rewrite the graph to reduce if approaches the limit.
      // TODO: possibly unify this with code already in SearchLimit
      long MAX_MOVES_PER_GRAPH = paramsSearch.GraphReuseEnabled ? (paramsSearch.GraphReuseRewriteEnabled ? 150L : 400L) : 5L;
      const long MAX_NODES_PER_SECOND = 500_000L;
      long maxNodes = searchLimit.Type switch
      {
        SearchLimitType.BestValueMove => 1,
        SearchLimitType.BestActionMove => 1,

        SearchLimitType.NodesPerMove => (long)(MAX_MOVES_PER_GRAPH * (float)(searchLimit.Value + searchLimit.ValueIncrement)),
        SearchLimitType.NodesForAllMoves => (long)(searchLimit.Value + searchLimit.ValueIncrement * MAX_MOVES_PER_GRAPH),

        SearchLimitType.SecondsPerMove => (long)((searchLimit.Value + searchLimit.ValueIncrement) * MAX_NODES_PER_SECOND * MAX_MOVES_PER_GRAPH),
        SearchLimitType.SecondsForAllMoves => (long)((searchLimit.Value + (searchLimit.ValueIncrement * MAX_MOVES_PER_GRAPH)) * MAX_NODES_PER_SECOND),
        _ => MAX_NODES
      };

      maxNodes = Math.Min(MAX_NODES, maxNodes);

      // Possibly apply tighter constraint based on fixedSearchLimit
      if (fixedSearchLimit != null && fixedSearchLimit.IsNodesLimit)
      {
        long fixedMax = MAX_MOVES_PER_GRAPH * (long)(fixedSearchLimit.Value + fixedSearchLimit.ValueIncrement);
        maxNodes = Math.Min(maxNodes, fixedMax);
      }

      // Possibly apply constraint based on max memory
      long maxBytes = paramsSearch.MaxMemoryBytes;
      const long MIN_BYTES_PER_NODE = 200; // average probably more typically 400 considering all associated data structures
      long maxNodesAllowedInMemory = maxBytes / MIN_BYTES_PER_NODE;
      maxNodes = Math.Min(maxNodes, maxNodesAllowedInMemory);

      int maxNodesInt = (int)Math.Min(maxNodes + 1000, MAX_NODES);

      bool hasAction = Manager.NNEvaluator0.HasAction;

      if ((Manager.ParamsSelect.FPUMode == ParamsSelect.FPUType.ActionHead
        || Manager.ParamsSelect.FPUModeAtRoot == ParamsSelect.FPUType.ActionHead)
       && !hasAction)
      {
        throw new Exception("FPUType.ActionHead requires a neural network with an action head output.");
      }

      graphToUse = new(maxNodesInt, hasAction,
                       Manager.ParamsSearch.EnableState,
                       Manager.ParamsSearch.EnableGraph,
                       Manager.ParamsSearch.PathTranspositionMode == PathMode.PositionEquivalence,
                       MCGSParamsFixed.TryEnableLargePages,
                       Manager.ParamsSearch.EnablePseudoTranspositionBlending,
                       priorMoves,
                       Manager.ParamsSearch.TestFlag);
    }

    Manager.Engine = new MCGSEngine(Manager, selectWorkerPools, graphToUse, searchRootPathFromGraphRoot == null ? []
                                                                                              : [.. searchRootPathFromGraphRoot]);

    StartSearchN = SearchRootNode.N;
    (MGMove bestMove, BestMoveInfoMCGS moveInfo) = DoSearch(Manager, verbose, progressCallback,
                                                            moveImmediateIfOnlyOneMove, forcedMove);
    BestMove = bestMove;
  }


  /// <summary>
  /// Returns new SearchLimit, possibly adjusted for time overhead and max graph nodes.
  /// </summary>
  /// <param name="limit"></param>
  /// <param name="paramsSearch"></param>
  /// <returns></returns>
  SearchLimit AdjustedSearchLimit(SearchLimit limit, ParamsSearch paramsSearch)
  {
    // Determine maximum number of tree nodes to allow store to grow to.
    int? maxTreeNodes;
    if (limit.MaxTreeNodes is not null)
    {
      // Use explicit value specified.
      maxTreeNodes = limit.MaxTreeNodes;
    }
    else if (CeresUserSettingsManager.Settings.MaxTreeNodes is not null)
    {
      // Use value explicitly set in Ceres.json.
      maxTreeNodes = CeresUserSettingsManager.Settings.MaxTreeNodes;
    }
    else
    {
      // Use default value based on amount of physical memory.
      maxTreeNodes = GraphStore.MAX_NODES - 2500;
    }

    if (limit.IsTimeLimit)
    {
      return limit with
      {
        MaxTreeNodes = maxTreeNodes,
        Value = Math.Max(0.01f, limit.Value - paramsSearch.MoveOverheadSeconds)
      };
    }
    else
    {
      return limit with { MaxTreeNodes = maxTreeNodes };
    }
  }
}
