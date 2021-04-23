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

using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.MCTS.Evaluators;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Managers.Limits;
using Ceres.MCTS.Params;

using Ceres.Features.UCI;
using System.IO;
using System.Text;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Environment;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Sublcass of GameEngine specialized for Ceres engine (in-process).
  /// </summary>
  public class GameEngineCeresInProcess : GameEngine
  {
    /// <summary>
    /// Optinal modifier action to be applied to ParamsSearchExecution before each search batch iteration.
    /// </summary>
    public readonly ParamsSearchExecutionModifier ParamsSearchExecutionModifier;

    /// <summary>
    /// Definition of neural network evaluator used for execution.
    /// </summary>
    public readonly NNEvaluatorDef EvaluatorDef;

    /// <summary>
    /// General search parameters used.
    /// </summary>
    public readonly ParamsSearch SearchParams;

    /// <summary>
    /// MCTS leaf selection parameters used.
    /// </summary>
    public readonly ParamsSelect ChildSelectParams;

    /// <summary>
    /// Manager used for approprtioning node or time limits at the game
    /// level to individual moves.
    /// </summary>
    public readonly IManagerGameLimit GameLimitManager;

    /// <summary>
    /// Current or most recently executed search.
    /// </summary>
    public MCTSearch Search;

    /// <summary>
    /// Search conducted prior to current search.
    /// </summary>
    public MCTSearch LastSearch;

    /// <summary>
    /// Optional name of file to which detailed log information 
    /// will be written after each move.
    /// </summary>
    public string SearchLogFileName;

    /// <summary>
    /// If detailed information relating to search status of
    /// moves at root should be output at end of a search.
    /// </summary>
    public bool VerboseMoveStats;

    /// <summary>
    /// Optional descriptive information for current game.
    /// </summary>
    public string CurrentGameID;


    #region Internal data

    /// <summary>
    /// Once created the NN evaluator pair is reused (until Dispose is called).
    /// </summary>
    NNEvaluatorSet evaluators = null;

    #endregion


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="evaluatorDef"></param>
    /// <param name="searchParams"></param>
    /// <param name="childSelectParams"></param>
    /// <param name="gameLimitManager"></param>
    /// <param name="paramsSearchExecutionModifier"></param>
    /// <param name="logFileName"></param>
    public GameEngineCeresInProcess(string id, NNEvaluatorDef evaluatorDef,
                                    ParamsSearch searchParams = null,
                                    ParamsSelect childSelectParams = null,
                                    IManagerGameLimit gameLimitManager = null,
                                    ParamsSearchExecutionModifier paramsSearchExecutionModifier = null,
                                    string logFileName = null) : base(id)
    {
      if (evaluatorDef == null) throw new ArgumentNullException(nameof(evaluatorDef));

      // TODO: remove, temporary testing 
      //InstallCUSTOM1AsDynamicByPosition();

      // Use default settings for search and select params if not specified.
      if (searchParams == null) searchParams = new ParamsSearch();
      if (childSelectParams == null) childSelectParams = new ParamsSelect();

      // Use default limit manager if not specified.
      if (gameLimitManager == null) gameLimitManager = new ManagerGameLimitCeres();

      ParamsSearchExecutionModifier = paramsSearchExecutionModifier;
      EvaluatorDef = evaluatorDef;
      SearchParams = searchParams;
      GameLimitManager = gameLimitManager;
      ChildSelectParams = childSelectParams;
      SearchLogFileName = logFileName;
      VerboseMoveStats = CeresUserSettingsManager.Settings.VerboseMoveStats;

      if (!string.IsNullOrEmpty(CeresUserSettingsManager.Settings.SearchLogFile))
      {
        SearchLogFileName = CeresUserSettingsManager.Settings.SearchLogFile;
      }
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public override bool SupportsNodesPerGameMode => true;


    bool isFirstMoveOfGame = true;

    /// <summary>
    /// Resets all state between games.
    /// </summary>
    /// <param name="gameID">optional game descriptive string</param>
    public override void ResetGame(string gameID = null)
    {
      Search = null;
      LastSearch = null;
      isFirstMoveOfGame = true;
      CurrentGameID = gameID;
    }

    /// <summary>
    /// Executes any preparatory steps (that should not be counted in thinking time) before a search.
    /// </summary>
    protected override void DoSearchPrepare()
    {
    }


    static readonly object logFileWriteObj = new object();

    /// <summary>
    /// Runs a search, calling DoSearch and adjusting the cumulative search time
    /// (convenience method with same functionality but returns the as the subclass
    /// GameEngineSearchResultCeres.
    /// </summary>
    /// <param name="curPositionAndMoves"></param>
    /// <param name="searchLimit"></param>
    /// <param name="callback"></param>
    /// <returns></returns>
    public GameEngineSearchResultCeres SearchCeres(PositionWithHistory curPositionAndMoves,
                                                   SearchLimit searchLimit,
                                                   List<GameMoveStat> gameMoveHistory = null,
                                                   ProgressCallback callback = null,
                                                   bool verbose = false)
    {
      return Search(curPositionAndMoves, searchLimit, gameMoveHistory, callback, verbose) as GameEngineSearchResultCeres;
    }

    /// <summary>
    /// Overriden virtual method which executes search.
    /// </summary>
    /// <param name="curPositionAndMoves"></param>
    /// <param name="searchLimit"></param>
    /// <param name="gameMoveHistory"></param>
    /// <param name="callback"></param>
    /// <returns></returns>
    protected override GameEngineSearchResult DoSearch(PositionWithHistory curPositionAndMoves,
                                                       SearchLimit searchLimit,
                                                       List<GameMoveStat> gameMoveHistory,
                                                       ProgressCallback callback,
                                                       bool verbose)
    {
      if (LastSearch != null && curPositionAndMoves.InitialPosMG != LastSearch.Manager.Context.StartPosAndPriorMoves.InitialPosMG)
        throw new Exception("ResetGame must be called if not continuing same line");

      MCTSearch searchResult;

      // Set up callback passthrough if provided
      MCTSManager.MCTSProgressCallback callbackMCTS = null;
      if (callback != null) callbackMCTS = callbackContext => callback((MCTSManager)callbackContext);

      // Possibly use the context of opponent to reuse position evaluations
      MCTSIterator shareContext = null;
      if (OpponentEngine is GameEngineCeresInProcess)
      {
        GameEngineCeresInProcess ceresOpponentEngine = OpponentEngine as GameEngineCeresInProcess;

        if (LastSearch is not null
         && LastSearch.Manager.Context.ParamsSearch.ReusePositionEvaluationsFromOtherTree
         && ceresOpponentEngine?.LastSearch.Manager != null
         && LeafEvaluatorReuseOtherTree.ContextsCompatibleForReuse(LastSearch.Manager.Context, ceresOpponentEngine.LastSearch.Manager.Context))
        {
          shareContext = ceresOpponentEngine.LastSearch.Manager.Context;

          // Clear any prior shared context from the shared context
          // to prevent unlimited backward chaining (keeping unneeded prior contexts alive)
          shareContext.ClearSharedContext();
        }
      }

      void InnerCallback(MCTSManager manager)
      {
        callbackMCTS?.Invoke(manager);
      }

      // Run the search
      searchResult = RunSearchPossiblyTreeReuse(shareContext, curPositionAndMoves, gameMoveHistory,
                                                  searchLimit, InnerCallback, verbose);

      int scoreCeresCP = (int)Math.Round(EncodedEvalLogistic.LogisticToCentipawn((float)searchResult.Manager.Root.Q), 0);

      MGMove bestMoveMG = searchResult.BestMove;

      int N = (int)searchResult.SearchRootNode.N;

      // Save (do not dispose) last search in case we can reuse it next time
      LastSearch = searchResult;

      isFirstMoveOfGame = false;

      // TODO is the RootNWhenSearchStarted correct because we may be following a continuation (BestMoveRoot)
      GameEngineSearchResultCeres result = 
        new GameEngineSearchResultCeres(bestMoveMG.MoveStr(MGMoveNotationStyle.LC0Coordinate),
                                        (float)searchResult.SearchRootNode.Q, scoreCeresCP, searchResult.SearchRootNode.MAvg, searchResult.Manager.SearchLimit, default,
                                        searchResult.Manager.RootNWhenSearchStarted, N, (int)searchResult.Manager.Context.AvgDepth, searchResult);

      // Append search result information to log file (if any).
      StringWriter dumpInfo = new StringWriter();
      if (SearchLogFileName != null)
      {
        result.Search.Manager.DumpFullInfo(bestMoveMG, result.Search.SearchRootNode, dumpInfo, CurrentGameID);
        lock (logFileWriteObj)
        {
          File.AppendAllText(SearchLogFileName, dumpInfo.GetStringBuilder().ToString());
        }
      }

      if (VerboseMoveStats)
      {
        result.Search.Manager.Context.Root.Dump(1, 1);
      }

      return result;
    }


    /// <summary>
    /// Launches search, possibly as continuation from last search.
    /// </summary>
    /// <param name="reuseOtherContextForEvaluatedNodes"></param>
    /// <param name="curPositionAndMoves"></param>
    /// <param name="gameMoveHistory"></param>
    /// <param name="searchLimit"></param>
    /// <param name="VERBOSE"></param>
    /// <param name="callback"></param>
    /// <returns></returns>
    private MCTSearch RunSearchPossiblyTreeReuse(MCTSIterator reuseOtherContextForEvaluatedNodes,
                                                 PositionWithHistory curPositionAndMoves,
                                                 List<GameMoveStat> gameMoveHistory,
                                                 SearchLimit searchLimit,
                                                 MCTSManager.MCTSProgressCallback callback,
                                                 bool verbose)
    {
      DateTime startTime = DateTime.Now;

      PositionEvalCache positionCacheOpponent = null;

      Search = new MCTSearch();

      if (LastSearch == null)
      {
        if (evaluators == null) evaluators = new NNEvaluatorSet(EvaluatorDef);

        Search.Search(evaluators, ChildSelectParams, SearchParams, GameLimitManager,
                      ParamsSearchExecutionModifier,
                      reuseOtherContextForEvaluatedNodes,
                      curPositionAndMoves, searchLimit, verbose, startTime,
                      gameMoveHistory, callback, false, isFirstMoveOfGame);
      }
      else
      {
        if (LastSearch.Manager.Context.StartPosAndPriorMoves.InitialPosMG != curPositionAndMoves.InitialPosMG)
          throw new Exception("Internal error: not same starting position");

        List<MGMove> forwardMoves = new List<MGMove>();
        List<MGMove> lastMoves = LastSearch.Manager.Context.StartPosAndPriorMoves.Moves;
        for (int i = 0; i < curPositionAndMoves.Moves.Count; i++)
        {
          if (i < lastMoves.Count)
          {
            if (lastMoves[i] != curPositionAndMoves.Moves[i])
              throw new Exception("Internal error: move sequence is not a prefix");
          }
          else
            forwardMoves.Add(curPositionAndMoves.Moves[i]);
        }

        // Determine the minimum fraction of tree that would need to be useful
        // before we would possibly reuse part of tree from prior search.
        // Below a certain level the tree reuse would increase memory consumption
        // (because unused nodes remain in node store) but not appreciably improve search speed.
        //TODO: tweak this parameter, possibly make it larger if small hardwre config is in use
        float THRESHOLD_FRACTION_NODES_REUSABLE = 0.05f;
        Search.SearchContinue(LastSearch, reuseOtherContextForEvaluatedNodes,
                                     forwardMoves, curPositionAndMoves,
                                     gameMoveHistory, searchLimit, verbose, startTime,
                                     callback, THRESHOLD_FRACTION_NODES_REUSABLE,
                                     isFirstMoveOfGame);
      }

      // Update the statistic on possible overshoot of internally alloted time and actual
      // (unless first move, since extra overhead is perhaps unavoidable in that situation).
      if (!isFirstMoveOfGame)
      {
        TimeSpan elapsedTime = DateTime.Now - startTime;
        if (Search.Manager.SearchLimit.IsTimeLimit)
        {
          if (Search.Manager.SearchLimit.IsPerGameLimit) throw new NotImplementedException();
          float timeOvershoot = (float)elapsedTime.TotalSeconds - Search.Manager.SearchLimit.Value;
          MCTSEventSource.MaximumTimeAllotmentOvershoot = Math.Max(MCTSEventSource.MaximumTimeAllotmentOvershoot, timeOvershoot);
        }
      }

      return Search;
    }


    public override void DumpMoveHistory(List<GameMoveStat> gameMoveHistory, SideType? side)
    {
      // TODO: fill in increemental time below (last argument)
      ManagerGameLimitInputs timeManagerInputs = new(LastSearch.Manager.Context.StartPosAndPriorMoves.FinalPosition,
                                                LastSearch.Manager.Context.ParamsSearch,
                                                gameMoveHistory, SearchLimitType.SecondsPerMove,
                                                LastSearch.Manager.Root.N, (float)LastSearch.Manager.Root.Q,
                                                LastSearch.Manager.SearchLimit.Value, 0, 0, 0);
      timeManagerInputs.Dump(side);
    }


    /// <summary>
    /// Returns UCI information string 
    /// (such as would appear in a chess GUI describing search progress) 
    /// based on last state of search.
    /// </summary>
    public override UCISearchInfo UCIInfo
    {
      get
      {
        if (LastSearch != null)
          return new UCISearchInfo(MCTS.Utils.UCIInfo.UCIInfoString(Search.Manager, Search.SearchRootNode));
        else
          return null;
      }
    }


    /// <summary>
    /// Diposes underlying search engine.
    /// </summary>
    public override void Dispose()
    {
      evaluators?.Dispose();
      evaluators = null;

      Search?.Manager.Dispose();
      Search = null;

      LastSearch?.Manager.Dispose();
      LastSearch = null;
    }


    static bool HAVE_NOTIFIED = false;
    /// <summary>
    /// TODO: remove, temporary testing code
    /// </summary>
    public static void InstallCUSTOM1AsDynamicByPosition()
    {
      NNEvaluator Build(string netID, int gpuID, NNEvaluator referenceEvaluator)
      {
        // ************************************************** NET SELECTION HERE
        const string gpuFast = "GPU:0";
        const string gpuSlow = "GPU:0";

        const string netFast = "LC0:jj_ensemble";
        const string netSlow = "LC0:j94-100";

        // *************************************************** END NET SELECTION

        // Construct a compound evaluator which does both fast and slow (potentially parallel)
        NNEvaluator[] evaluators = new NNEvaluator[] {NNEvaluator.FromSpecification(netFast, gpuFast),
                                                      NNEvaluator.FromSpecification(netSlow, gpuSlow)};
        NNEvaluatorDynamicByPos.DynamicEvaluatorIndexPredicate chooserFunc = delegate (NNPositionEvaluationBatchMember[] evaluatorResults)
        {
          if (!HAVE_NOTIFIED)
          {
            Console.WriteLine($"\r\n*** NOTE: InstallCUSTOM1AsDynamicByPosition is in use for CUSTOM1 with fast/slow net combination logic (see GameEngineCeresInProcess) using {netFast} and {netSlow}\r\n");
            HAVE_NOTIFIED = true;
          }

          NNPositionEvaluationBatchMember resultFast = evaluatorResults[0];
          NNPositionEvaluationBatchMember resultSlow = evaluatorResults[1];

          // ************************************************** LOGIC GOES HERE
          // Determine which network to use
          float absVFromFast = System.Math.Abs(resultFast.V);
          bool useFast = absVFromFast > 0.33f;
          // *************************************************** END LOGIC

          if (useFast)
          {
            // Add a statistic showing eval comes from fast net
            MCTSEventSource.TestCounter1++;
          }
          return useFast ? 0 : 1;
        };

        NNEvaluatorDynamicByPos evalChoose = new(evaluators, chooserFunc);
        return evalChoose;
      }
      NNEvaluatorFactory.Custom1Factory = Build;
    }


  }
}
