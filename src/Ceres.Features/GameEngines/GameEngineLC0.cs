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

using Ceres.Base.Math;

using Ceres.Chess;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.LC0.Engine;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.NNFiles;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.GameEngines;
using Ceres.Chess.Positions;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Subclass of GameEngine for Leela Zero chess engine accessed via UCI protocol.
  /// </summary>
  public class GameEngineLC0 : GameEngine, IUCIGameRunnerProvider
  {
    /// <summary>
    /// Underlying LC0 engine object.
    /// </summary>
    public readonly LC0Engine LC0Engine;

    /// <summary>
    /// Underlying action to be applied as part of initialization.
    /// </summary>
    public readonly Action SetupAction = null;

    /// <summary>
    /// Underlying UCIRunner object which manages 
    /// launcing external process and communicating via the UCI protocol.
    /// </summary>
    public UCIGameRunner UCIRunner => LC0Engine.Runner;

    public bool DisableTreeReuse;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="networkID"></param>
    /// <param name="forceDisableSmartPruning"></param>
    /// <param name="emulateCeresSettings"></param>
    /// <param name="searchParams"></param>
    /// <param name="selectParams"></param>
    /// <param name="paramsNN"></param>
    /// <param name="setupAction"></param>
    /// <param name="overrideEXE"></param>
    /// <param name="verbose"></param>
    /// <param name="alwaysFillHistory"></param>
    /// <param name="overrideBatchSize"></param>
    /// <param name="overrideCacheSize"></param>
    /// <param name="extraCommandLineArgs"></param>
    /// <param name="processorGroupID"></param>
    public GameEngineLC0(string id, string networkID, bool forceDisableSmartPruning = false,
                         bool emulateCeresSettings = false,
                         ParamsSearch searchParams = null, ParamsSelect selectParams = null,
                         NNEvaluatorDef paramsNN = null,
                         Action setupAction = null,
                         string overrideEXE = null,
                         bool verbose = false,
                         bool alwaysFillHistory = true,
                         int? overrideBatchSize = null,
                         int? overrideCacheSize = null,
                         string extraCommandLineArgs = null,
                         int processorGroupID = 0,
                         string? overrideBackendString = null,
                         bool disableTreeReuse = false) : base(id, processorGroupID)
    {
      SetupAction = setupAction;
      if (SetupAction != null)
      {
        SetupAction();
      }

      bool resetStateAndCachesBeforeMoves = disableTreeReuse || (searchParams != null && !searchParams.TreeReuseEnabled);
      DisableTreeReuse = disableTreeReuse;

      LC0Engine = LC0EngineConfigured.GetLC0Engine(searchParams, selectParams, paramsNN,
                                                   NNWeightsFiles.LookupNetworkFile(networkID),
                                                   processorGroupID, resetStateAndCachesBeforeMoves,
                                                   emulateCeresSettings, verbose,
                                                   forceDisableSmartPruning, overrideEXE,
                                                   alwaysFillHistory, extraCommandLineArgs,
                                                   overrideBatchSize, overrideCacheSize, overrideBackendString);
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public override bool SupportsNodesPerGameMode => false;

    /// <summary>
    /// If the game engine should reset its state before each search.
    /// </summary>
    public override bool ResetGameAlwaysBeforeSearch => DisableTreeReuse;

    /// <summary>
    /// Returns UCI search information 
    /// (such as would appear in a chess GUI describing search progress) 
    /// based on last state of search.
    /// </summary>
    public override UCISearchInfo UCIInfo => new UCISearchInfo(UCIRunner.LastInfoString);


    /// <summary>
    /// Resets all state between games.
    /// </summary>
    /// <param name="gameID">optional game descriptive string</param>
    public override void ResetGame(string gameID = null) => UCIRunner.StartNewGame();


    /// <summary>
    /// Dispose method which releases underlying engine object.
    /// </summary>
    public override void Dispose() => LC0Engine.Dispose();


    /// <summary>
    /// Executes any preparatory steps (that should not be counted in thinking time) before a search.
    /// </summary>
    protected override void DoSearchPrepare()
    {
      LC0Engine.DoSearchPrepare();
    }

    /// <summary>
    /// Overridden virtual method that executs the search
    /// by issuing UCI commands to the LC0 engine with appropriate search limit parameters.
    /// </summary>
    /// <param name="curPositionAndMoves"></param>
    /// <param name="searchLimit"></param>
    /// <param name="gameMoveHistory"></param>
    /// <param name="callback"></param>
    /// <returns></returns>
    protected override GameEngineSearchResult DoSearch(PositionWithHistory curPositionAndMoves,
                                                       SearchLimit searchLimit,
                                                       List<GameMoveStat> gameMoveHistory,
                                                       ProgressCallback callback, bool verbose)
    {
      DoSearchPrepare();
      if (DisableTreeReuse)
      {
        UCIRunner.StartNewGame();
      }

      if (SetupAction != null) SetupAction();

      // Run the analysis
      VerboseMoveStats lc0Analysis = LC0Engine.AnalyzePositionFromFENAndMoves(curPositionAndMoves.FENAndMovesString, searchLimit);

      if (verbose)
      {
        lc0Analysis.Dump();
      }

      // TODO: can we somehow correctly set the staring N arugment here?
      float boundedCP = StatUtils.Bounded(lc0Analysis.ScoreCentipawns, -9999, 9999);
      float lc0Q = EncodedEvalLogistic.CentipawnToLogistic(boundedCP);
      return new GameEngineSearchResult(lc0Analysis.BestMove, lc0Q, boundedCP, float.NaN,
                                        searchLimit, default, 0, (int)lc0Analysis.NumNodes,
                                        lc0Analysis.UCIInfo.NPS, (int)lc0Analysis.UCIInfo.Depth,
                                        lc0Analysis.Moves);
    }

    /// <summary>
    /// Returns string summary of object.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<GameEngineLC0 {LC0Engine.Runner.EngineEXE} {LC0Engine.Runner.EngineExtraCommand}>";
    }

  }

}
