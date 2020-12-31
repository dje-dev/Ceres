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
using System.IO;

using Ceres.Base;
using Ceres.Base.Benchmarking;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.LC0.Positions;
using Ceres.Chess.MoveGen;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.Chess.GameEngines
{
  /// <summary>
  /// Implementation of GameEngine for an engine running 
  /// in an external process and supporting the UCI protocol.
  /// </summary>
  public class GameEngineUCI : GameEngine, IUCIGameRunnerProvider
  {
    /// <summary>
    /// Name of engine.
    /// </summary>
    public readonly string Name;

    /// <summary>
    /// Path to UCI executable file.
    /// </summary>
    public readonly string EXEPath;

    /// <summary>
    /// Optional number of threads to be allocated to engine.
    /// </summary>
    public readonly int? NumThreads;

    /// <summary>
    /// Option hash size (in megabytes) to be allocated to the engine.
    /// </summary>
    public readonly int? HashSizeMB;

    /// <summary>
    /// Optional path Syzyzy tablebase files for engine probing.
    /// </summary>
    public readonly string SyzygyPath;

    /// <summary>
    /// Optional set of supplementary UCI set option 
    /// commands to be issued to engine at startup.
    /// </summary>
    public string[] UCISetOptionCommands;

    /// <summary>
    /// Supplemental arguments to be provided to engine on command line.
    /// </summary>
    public readonly string ExtraArgs;


    /// <summary>
    /// UCI runner object that manages interaction with the engine.
    /// </summary>
    public readonly UCIGameRunner UCIRunner;

    /// <summary>
    /// Returns if the instance has been terminated.
    /// </summary>
    public bool IsShutdown => UCIRunner.IsShutdown;

    /// <summary>
    /// Returns the underlying UCIRunner associated with this instance.
    /// </summary>
    UCIGameRunner IUCIGameRunnerProvider.UCIRunner => UCIRunner;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="name"></param>
    /// <param name="executablePath"></param>
    /// <param name="numThreads"></param>
    /// <param name="hashSizeMB"></param>
    /// <param name="syzygyPath"></param>
    /// <param name="uciSetOptionCommands"></param>
    /// <param name="callback"></param>
    /// <param name="resetGameBetweenMoves"></param>
    /// <param name="extraArgs"></param>
    public GameEngineUCI(string name, string executablePath, 
                         int? numThreads = null, 
                         int? hashSizeMB = null, 
                         string syzygyPath = null,
                         string[] uciSetOptionCommands = null,
                         ProgressCallback callback = null,
                         bool resetGameBetweenMoves = false,
                         string extraArgs = null) : base(name)
    {
      if (callback is not null) throw new NotImplementedException("Internal error: Callbacks not currently supported");
      if (!File.Exists(executablePath)) throw new Exception($"UCI executable file not found {executablePath}");

      Name = name;
      EXEPath = executablePath;
      ExtraArgs = extraArgs;
      NumThreads = numThreads;
      HashSizeMB = hashSizeMB;
      SyzygyPath = syzygyPath;

      UCISetOptionCommands = uciSetOptionCommands;

      List<string> extraCommands = new List<string>();
      if (numThreads.HasValue) extraCommands.Add($"setoption name Threads value {numThreads}");
      if (hashSizeMB.HasValue) extraCommands.Add($"setoption name Hash value {hashSizeMB}");

      if (syzygyPath != null)
      {
        if (!Directory.Exists(syzygyPath)) throw new Exception($"Specified Syzygy path not found or inaccessilbe: {syzygyPath} ");
        extraCommands.Add($"setoption name SyzygyPath value {syzygyPath}");
      }

      // 
      if (uciSetOptionCommands != null)
      {
        foreach (string option in uciSetOptionCommands)
        {
          if (!option.ToLower().StartsWith("setoption name"))
            throw new Exception($"Supplemental UCI option expected to begin with 'setoption name' but see: {option}");

          extraCommands.Add(option);
        }
      }

      string[] extraCommandsArray = extraCommands.ToArray();

      UCIRunner = new UCIGameRunner(executablePath, resetGameBetweenMoves,
                                    extraCommandLineArguments : extraArgs,
                                    uciSetOptionCommands: extraCommandsArray);
    }

    public override void ResetGame()
    {
      UCIRunner.StartNewGame();
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
        return new UCISearchInfo(UCIRunner.LastInfoString, UCIRunner.LastBestMove);
      }
    }


    /// <summary>
    /// Executes any preparatory steps (that should not be counted in thinking time) before a search.
    /// </summary>
    protected override void DoSearchPrepare()
    {
      UCIRunner.EvalPositionPrepare();
    }

    protected override GameEngineSearchResult DoSearch(PositionWithHistory curPositionAndMoves, SearchLimit searchLimit,
                                                       List<GameMoveStat> gameMoveHistory, ProgressCallback callback,
                                                       bool verbose)
    {
      DoSearchPrepare();

      string startFEN = curPositionAndMoves.InitialPosition.FEN;
      string movesStr = curPositionAndMoves.MovesStr;

      UCISearchInfo gameInfo;
      switch (searchLimit.Type)
      {
        case SearchLimitType.SecondsPerMove:
          gameInfo = UCIRunner.EvalPositionToMovetime(startFEN, movesStr, (int)(searchLimit.Value * 1000));
          break;

        case SearchLimitType.NodesPerMove:
          gameInfo = UCIRunner.EvalPositionToNodes(startFEN, movesStr, (int)(searchLimit.Value));
          break;

        case SearchLimitType.NodesForAllMoves:
          throw new Exception("NodesForAllMoves not supported for generic UCI engine");

        case SearchLimitType.SecondsForAllMoves:
          bool weAreWhite = curPositionAndMoves.FinalPosition.MiscInfo.SideToMove == SideType.White;

          TimingStats stats = new TimingStats();
          using (new TimingBlock(stats, TimingBlock.LoggingType.None))
          {
            gameInfo = UCIRunner.EvalPositionRemainingTime(startFEN, movesStr,
                                                           weAreWhite,
                                                           searchLimit.MaxMovesToGo,
                                                           (int)(searchLimit.Value * 1000),
                                                           (int)(searchLimit.ValueIncrement * 1000));
          }

          break;

        default:
          throw new NotSupportedException($"Unsupported MoveType {searchLimit.Type}");
      }

      float q = EncodedEvalLogistic.CentipawnToLogistic(gameInfo.ScoreCentipawns);
      return new GameEngineSearchResult(gameInfo.BestMove, q, gameInfo.ScoreCentipawns, float.NaN, searchLimit, default, 0, (int)gameInfo.Nodes, gameInfo.Depth);
    }


    public override void Dispose()
    {
      UCIRunner.Shutdown();
    }
  }
}
