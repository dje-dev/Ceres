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
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Generic;

using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Base.OperatingSystem.NVML;

using Ceres.Chess;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.Positions;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNFiles;
using Ceres.Chess.SearchResultVerboseMoveInfo;

using Ceres.MCTS.Iteration;
using Ceres.MCTS.Params;
using Ceres.MCTS.MTCSNodes.Analysis;
using Ceres.MCTS.Utils;
using Ceres.MCTS.MTCSNodes;

using Ceres.Features.GameEngines;
using Ceres.Features.Visualization.TreePlot;
using Ceres.Features.Visualization.AnalysisGraph;
using Ceres.Chess.UserSettings;
using Ceres.Chess.MoveGen.Converters;

#endregion

namespace Ceres.Features.UCI
{
  /// <summary>
  /// Manager of UCI game loop, parsing and executing commands from Console
  /// and outputting appropriate UCI lines such as bestmove and info.
  /// </summary>
  public partial class UCIManager
  {
    /// <summary>
    /// Indicates if the UCI option "UCI_Chess960" is set to true for the current game.
    /// </summary>
    private bool IsChess960OptionSet = false;

    /// <summary>
    /// Input stream.
    /// </summary>
    public readonly TextReader InStream;

    /// <summary>
    /// Output stream.
    /// </summary>
    public readonly TextWriter OutStream;

    /// <summary>
    /// Action to be called upon end of Ceres search.
    /// </summary>
    public Action<MCTSManager> SearchFinishedEvent;

    /// <summary>
    /// Action to be called upon receipt of a request to log a message during search.
    /// </summary>
    public Action<(MCTSearch, string)> LogSearchInfoMessageEvent;


    public NNEvaluatorDef EvaluatorDef;

    /// <summary>
    /// Ceres engine instance used for current UCI game.
    /// </summary>
    public GameEngineCeresInProcess CeresEngine;

    volatile Task<GameEngineSearchResultCeres> taskSearchCurrentlyExecuting;

    bool haveInitializedEngine;

    /// <summary>
    /// Optional evaluator to call to benchmark neural network backend.
    /// </summary>
    Action<NNEvaluatorDef> BackendBenchEvaluator;

    /// <summary>
    /// Optional evaluator to call to search benchmark program (to run for specified number of seconds).
    /// </summary>
    Action<NNEvaluatorDef, int> BenchmarkSearchEvaluator;

    /// <summary>
    /// The position and history associated with the current evaluation.
    /// </summary>
    PositionWithHistory curPositionAndMoves;
    bool curPositionIsContinuationOfPrior;

    List<GameMoveStat> gameMoveHistory = new List<GameMoveStat>();

    bool stopIsPending;
    bool debug = false;

    /// <summary>
    /// Stream to which all UCI input and output is echoed.
    /// </summary>
    StreamWriter uciLogWriter;

    /// <summary>
    /// Specification string for which neural network to use.
    /// </summary>
    NNNetSpecificationString NetworkSpec;

    /// <summary>
    /// Specification string for which device to use for network inference.
    /// </summary>
    NNDevicesSpecificationString DeviceSpec;

    /// <summary>
    /// Optional delegate that can modify search parameters.
    /// </summary>
    public readonly Action<ParamsSearch> SearchModifier;

    /// <summary>
    /// Optional delegate that can modify select parameters.
    /// </summary>
    public readonly Action<ParamsSelect> SelectModifier;


    void CreateEvaluator()
    {
      EvaluatorDef = new NNEvaluatorDef(NetworkSpec.ComboType, NetworkSpec.NetDefs,
                                        DeviceSpec.ComboType, DeviceSpec.Devices, NetworkSpec.OptionsString, null);

      OutStream.WriteLine($"uci info Network evaluation configured to use: {EvaluatorDef}");
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="networkSpec"></param>
    /// <param name="deviceSpec"></param>
    /// <param name="searchModifier"></param>
    /// <param name="selectModifier"></param>
    /// <param name="inStream"></param>
    /// <param name="outStream"></param>
    /// <param name="searchFinishedEvent"></param>
    /// <param name="disablePruning"></param>
    /// <param name="uciLogFileName"></param>
    /// <param name="searchLogFileName"></param>
    /// <param name="backendBenchEvaluator"></param>
    /// <param name="benchmarkEvaluator"></param>
    public UCIManager(NNNetSpecificationString networkSpec,
                      NNDevicesSpecificationString deviceSpec,
                      Action<ParamsSearch> searchModifier = null,
                      Action<ParamsSelect> selectModifier = null,
                      TextReader inStream = null, TextWriter outStream = null,
                      Action<MCTSManager> searchFinishedEvent = null,
                      bool disablePruning = false,
                      string uciLogFileName = null,
                      string searchLogFileName = null,
                      Action<NNEvaluatorDef> backendBenchEvaluator = null,
                      Action<NNEvaluatorDef, int> benchmarkEvaluator = null)
    {
      InStream = inStream ?? Console.In;
      OutStream = outStream ?? Console.Out;
      SearchFinishedEvent = searchFinishedEvent;

      NetworkSpec = networkSpec;
      DeviceSpec = deviceSpec;
      SearchModifier = searchModifier;
      SelectModifier = selectModifier;

      BackendBenchEvaluator = backendBenchEvaluator;
      BenchmarkSearchEvaluator = benchmarkEvaluator;


      if (disablePruning) futilityPruningDisabled = true;

      this.searchLogFileName = searchLogFileName;

      // Possibly all UCI input/output to log file.
      if (uciLogFileName != null)
      {
        uciLogWriter = new StreamWriter(new FileStream(uciLogFileName, FileMode.Append, FileAccess.Write));
      }

      if (NetworkSpec != null)
      {
        CreateEvaluator();
      }
    }


    void LogWriteLine(string prefix, string line)
    {
      uciLogWriter.WriteLine(prefix +
                            " ["
                            + DateTime.Now.ToString("MM/dd/yyyy hh:mm:ss.fff tt")
                            + "]"
                            + line);
      uciLogWriter.Flush();
    }

    /// <summary>
    /// Outputs line to UCI.
    /// </summary>
    /// <param name="result"></param>
    void UCIWriteLine(string result = null)
    {
      OutStream.WriteLine(result);
      if (uciLogWriter != null)
      {
        LogWriteLine("OUT:", result);
      }
    }


    public ParamsSelect ParamsSelect
    {
      get
      {
        ParamsSelect parms = new ParamsSelect();
        parms.CPUCT = cpuct;
        parms.CPUCTBase = cpuctBase;
        parms.CPUCTFactor = cpuctFactor;
        parms.CPUCTAtRoot = cpuctAtRoot;
        parms.CPUCTBaseAtRoot = cpuctBaseAtRoot;
        parms.CPUCTFactorAtRoot = cpuctFactorAtRoot;
        parms.PolicySoftmax = policySoftmax;
        parms.FPUValue = fpu;
        parms.FPUValueAtRoot = fpuAtRoot;

        SelectModifier?.Invoke(parms);
        return parms;
      }
    }

    public ParamsSearch ParamsSearch
    {
      get
      {
        ParamsSearch parms = new ParamsSearch();
        parms.MoveOverheadSeconds = moveOverheadSeconds;

        SearchModifier?.Invoke(parms);
        return parms;
      }
    }


    void ResetGame()
    {
      curPositionAndMoves = PositionWithHistory.FromFENAndMovesUCI(Chess.Position.StartPosition.FEN);
      gameMoveHistory = new List<GameMoveStat>();
      CeresEngine?.ResetGame();
    }


    /// <summary>
    /// Runs the UCI loop.
    /// </summary>
    public void PlayUCI()
    {
      ResetGame();

      while (true)
      {
        string command = InStream.ReadLine();
        if (uciLogWriter != null)
        {
          LogWriteLine("IN:", command);
        }

        switch (command)
        {
          case null:
          case "":
            break;

          case "uci":
            UCIWriteLine($"id name Ceres {CeresVersion.VersionString}");
            UCIWriteLine("id author David Elliott and the Ceres Authors");
            UCIWriteLine(SetOptionUCIDescriptions);
            UCIWriteLine("uciok");
            break;

          case "commands":
            UCIWriteLine("download <ID>   - attempt to download Ceres network from CeresNets repository");
            UCIWriteLine("backendbench    - run backend benchmark to test nps speed of network evaluation");
            UCIWriteLine("benchmark       - run search benchmark with specified number of seconds per position, e.g. \"benchmark 10\"");
            UCIWriteLine("dump-fen        - shows FEN of most recently searched position");
            UCIWriteLine("dump-move-stats - dumps information top level candidate moves");
            UCIWriteLine("dump-pv         - dumps principal variation information from last search");
            UCIWriteLine("dump-pv-detail  - dumps principal variation information from last search (detailed)");
            UCIWriteLine("dump-info       - dump information about last search (top level candidate moves, principal variation, etc.)");
            UCIWriteLine("dump-time       - dump information about time manager's last decision");
            UCIWriteLine("dump-processor  - dump information about CPUs in this system");
            UCIWriteLine("dump-params     - dump configuration parameters currently in use for Ceres");
            UCIWriteLine("dump-store      - dumps full node store for tree from last search");
            UCIWriteLine("dump-nvidia     - dumps information about NVIDIA CUDA devices detected in the system");
            UCIWriteLine("show-tree-plot  - shows a graphical representation of full search tree");
            UCIWriteLine("graph [1-10]    - invokes graph feature to show the principal variations from last search (requires configuration), e.g. graph 7");
            UCIWriteLine("gamecomp        - invokes the game comparison feature to graph the divergence points in one or more games (requires configuration)");
            UCIWriteLine("");
            break;

          case string c when c.StartsWith("setoption"):
            ProcessSetOption(command);
            break;

          case "stop":
            if (taskSearchCurrentlyExecuting != null && !stopIsPending)
            {
              stopIsPending = true;

              // Avoid race condition by making sure the search is already created.
              while (CeresEngine.Search?.Manager == null)
              {
                Thread.Sleep(20);
              }

              CeresEngine.Search.Manager.ExternalStopRequested = true;
              if (taskSearchCurrentlyExecuting != null)
              {
                taskSearchCurrentlyExecuting.Wait();
                //                if (!debug && taskSearchCurrentlyExecuting != null) taskSearchCurrentlyExecuting.Result?.Search?.Manager?.Dispose();
                taskSearchCurrentlyExecuting = null;
              }

              stopIsPending = false;
            }

            break;

          case "ponderhit":
            throw new NotImplementedException("Ceres does not yet support UCI ponder mode.");
            return;

          case "xboard":
            // ignore
            break;

          case "debug on":
            debug = true;
            break;

          case "debug off":
            debug = false;
            break;

          case "isready":
            if (InitializeEngineIfNeeded())
            {
              UCIWriteLine("readyok");
            }
            break;

          case "ucinewgame":
            if (InitializeEngineIfNeeded())
            {
              ResetGame();
            }
            break;

          case "quit":
            UCIWriteLine("Ceres shutdown in progress....");

            if (taskSearchCurrentlyExecuting != null)
            {
              CeresEngine.Search.Manager.ExternalStopRequested = true;
              taskSearchCurrentlyExecuting?.Wait();
            }

            if (CeresEngine != null)
            {
              CeresEngine.Dispose();
            }

            // In rare cases we've seen a crash during the Environment.Exit() below,
            // presumably due to cleanup which was in progress.
            // To try to avoid this ugliness, do a GC first and pause briefly before exit.
            GC.Collect(4, GCCollectionMode.Forced, true);
            Thread.Sleep(500);

            System.Environment.Exit(0);
            break;

          case string c when c.StartsWith("go"):

            // Possibly another search is already executing.
            // The UCI specification is unclear about what to do in this situation.
            // Some engines seem to enqueue these for later execution (e.g. Stockfish)
            // whereas others (e.g. Python chess) report this as an error condition.
            // Currently Ceres waits only a short while for any possible pending search
            // to finish (e.g. to avoid a race condition if it is in the process of being shutdown)
            // and aborts with an error if search is still in progress.
            // It is not viable to wait indefinitely, since (among other reasons)
            // the engine needs to monitor for stop commands.
            const int MAX_MILLISECONDS_WAIT = 500;
            taskSearchCurrentlyExecuting?.Wait(MAX_MILLISECONDS_WAIT);

            if (taskSearchCurrentlyExecuting != null && !taskSearchCurrentlyExecuting.IsCompleted)
            {
              throw new Exception("Received go command when another search was running and not stopped first.");
            }

            if (InitializeEngineIfNeeded())
            {
              taskSearchCurrentlyExecuting = ProcessGo(command);
            }
            break;

          case string c when c.StartsWith("position"):
            try
            {
              ProcessPosition(c);
            }
            catch (Exception e)
            {
              UCIWriteLine($"Illegal position command: \"{c}\"" + System.Environment.NewLine + e.ToString());
            }
            break;

          // Proprietary commands
          case "dump-info":
            if (CeresEngine?.Search != null)
            {
              MCTSearch search = CeresEngine.Search;
              CeresEngine.Search.Manager.DumpFullInfo(search.BestMove, search.SearchRootNode,
                                                      search.LastReuseDecision, search.LastMakeNewRootTimingStats,
                                                      search.LastGameLimitInputs, Console.Out, "UCI");
            }
            else
              UCIWriteLine("info string No search manager created");
            break;

          case string c when c.StartsWith("graph"):
            // Command of the form such as "graph" or "graph 7" or "graph 3 ref 0.5"
            if (CeresEngine?.Search != null)
            {
              string[] partsGraph = c.TrimEnd().Split(" ");
              string optionsStr = partsGraph.Length > 1 ? c.Substring(6) : null;
              AnalysisGraphOptions options = AnalysisGraphOptions.FromString(optionsStr);
              AnalysisGraphGenerator graphGenerator = new AnalysisGraphGenerator(CeresEngine.Search, options);
              graphGenerator.Write(true);
            }
            else
            {
              UCIWriteLine("info string No search manager created");
            }
            break;

          case String c when c.StartsWith("gamecomp"):
            UCIManagerHelpers.ProcessGameComp(c, UCIWriteLine);
            break;

          case "dump-params":
            if (CeresEngine?.Search != null)
            {
              CeresEngine?.Search.Manager.DumpParams();
            }
            else
              UCIWriteLine("info string No search manager created");
            break;

          case "dump-processor":
            HardwareManager.DumpProcessorInfo();
            break;

          case string c when c.StartsWith("download"):
            UCIManagerHelpers.ProcessDownloadCommand(c, UCIWriteLine);
            break;

          case "backendbench":
            if (BackendBenchEvaluator == null)
            {
              Console.WriteLine("No BackendBenchEvaluator installed, cannot benchmark.");
            }
            else
            {
              if (InitializeEngineIfNeeded())
              {
                BackendBenchEvaluator(EvaluatorDef);
              }
            }
            break;

          case string c when c.StartsWith("benchmark"):
            string[] parts = c.Split(" ");
            int numSeconds = 30;
            if (parts.Length > 1)
            {
              if (!int.TryParse(parts[1], out numSeconds))
              {
                UCIWriteLine("info string Invalid number of seconds for benchmark");
                break;
              }
            }
            if (BenchmarkSearchEvaluator == null)
            {
              Console.WriteLine("No BenchmarkEvaluator installed, cannot benchmark.");
            }
            else
            {
              if (InitializeEngineIfNeeded())
              {
                BenchmarkSearchEvaluator(EvaluatorDef, numSeconds);
              }
            }
            break;

          case "dump-fen":
            if (CeresEngine?.Search != null)
            {
              Console.WriteLine("info string " + CeresEngine.Search.Manager.Context.StartPosAndPriorMoves.FinalPosition.FEN);
            }
            else
            {
              UCIWriteLine("info string No search manager created");
            }
            break;

          case "dump-time":
            if (CeresEngine?.Search != null)
            {
              CeresEngine.Search.Manager.DumpTimeInfo(OutStream, CeresEngine.Search.SearchRootNode);
            }
            else
            {
              UCIWriteLine("info string No search manager created");
            }
            break;

          case "dump-store":
            if (CeresEngine?.Search != null)
            {
              CeresEngine.Search.Manager.Context.Tree.Store.Dump(true);
            }
            else
            {
              UCIWriteLine("info string No search manager created");
            }

            break;

          case "validate-store":
            if (CeresEngine?.Search != null)
            {
              CeresEngine.Search.Manager.Context.Tree.Store.Validate(null);
            }
            else
            {
              UCIWriteLine("info string No search manager created");
            }


            break;

          case "dump-move-stats":
            if (CeresEngine?.Search != null)
            {
              OutputVerboseMoveStats();
            }
            else
            {
              UCIWriteLine("info string No search manager created");
            }
            break;

          case "dump-pv":
            DumpPV(false);
            break;

          case "dump-pv-detail":
            DumpPV(true);
            break;

          case "dump-nvidia":
            NVML.DumpInfo();
            break;

          case "show-tree-plot":
            if (CeresEngine?.Search != null)
              TreePlot.Show(CeresEngine.Search.Manager.Context.Root.StructRef);
            else
              UCIWriteLine("info string No search manager created");
            break;

          case string c when c.StartsWith("save-tree-plot"):
            if (CeresEngine?.Search != null)
            {
              string[] partsPlot = command.Split(" ");
              if (partsPlot.Length == 2)
              {
                string fileName = partsPlot[1];
                TreePlot.Save(CeresEngine.Search.Manager.Context.Root.StructRef, fileName);
              }
              else if (partsPlot.Length == 1)
              {
                UCIWriteLine("Filename was not provided");
              }
              else
              {
                UCIWriteLine("Filename cannot contain spaces");
              }
            }
            else
              UCIWriteLine("info string No search manager created");
            break;

          case "waitdone": // proprietary verb used for test driver
            taskSearchCurrentlyExecuting?.Wait();
            break;

          default:
            UCIWriteLine($"error Unknown command: {command}");
            break;
        }
      }
    }

    /// <summary>
    /// Dumps the PV (principal variation) to the output stream.
    /// </summary>
    /// <param name="withDetail"></param>
    private void DumpPV(bool withDetail)
    {
      if (CeresEngine?.Search != null)
      {
        if (withDetail)
        {
          MCTSPosTreeNodeDumper.DumpPV(CeresEngine.Search.SearchRootNode, true, this.OutStream);
        }
        else
        {
          SearchPrincipalVariation pv2 = new SearchPrincipalVariation(CeresEngine.Search.Manager.Root);
          UCIWriteLine(pv2.ShortStr(IsChess960OptionSet));
        }
      }
      else
      {
        UCIWriteLine("info string No search manager created");
      }
    }




    /// <summary>
    /// Releases existing search engine (if any)
    /// so that any subsequent search will rebuild (with current options).
    /// </summary>
    private void ReinitializeEngine()
    {
      if (haveInitializedEngine && CeresEngine != null)
      {
        CeresEngine.Dispose();
      }

      CeresEngine = null;
      haveInitializedEngine = false;
    }


    void InfoLogger(string infoMessage)
    {
      UCIWriteLine("info engine " + infoMessage);
    }

    private bool InitializeEngineIfNeeded()
    {
      bool success = TryInitializeEngineIfNeeded();
      if (!success)
      {
        UCIWriteLine("Cannot initialize engine.");
        UCIWriteLine();
      }
      return success;
    }


    private bool TryInitializeEngineIfNeeded()
    {
      try
      {
        return DoTryInitializeEngineIfNeeded();
      }
      catch (Exception e)
      {
        UCIWriteLine();
        UCIWriteLine(e.ToString());
        UCIWriteLine($"No evaluator created, ERROR encountered.");

        if (EvaluatorDef != null)
        {
          string netSpecString = NNNetSpecificationString.ToSpecificationString(EvaluatorDef.NetCombo, EvaluatorDef.Nets);
          string deviceSpecString = NNDevicesSpecificationString.ToSpecificationString(EvaluatorDef.DeviceCombo, EvaluatorDef.Devices);

          UCIWriteLine($"Attempted Network: {netSpecString}");
          UCIWriteLine($"Attempted Device: {deviceSpecString}");
        }

        EvaluatorDef = null;
        return false;
      }
    }


    private bool DoTryInitializeEngineIfNeeded()
    {
      if (!haveInitializedEngine)
      {
        if (EvaluatorDef == null)
        {
          // Use Ceres.json settings for default network/device if found.
          if (CeresUserSettingsManager.Settings.DefaultNetworkSpecString == null)
          {
            UCIWriteLine("info string No default network specified, cannot initialize engine.");
            UCIWriteLine("info string Use setoption with \"WeightsFile\" to specify neural network weights file to use.");
            UCIWriteLine("info string Example: setoption name WeightsFile value C1-256-10");
            return false;
          }
          else
          {
            NetworkSpec = new NNNetSpecificationString(CeresUserSettingsManager.Settings.DefaultNetworkSpecString);
            const string DEFAULT_DEVICE = "GPU:0";
            string deviceString = CeresUserSettingsManager.Settings.DefaultDeviceSpecString ?? DEFAULT_DEVICE;
            DeviceSpec = new NNDevicesSpecificationString(deviceString);

            CreateEvaluator();
          }
        }

        ShowWeightsFileInfo();

        // Create the engine (to be subsequently reused).
        CeresEngine = new GameEngineCeresInProcess("Ceres", EvaluatorDef, null, ParamsSearch, ParamsSelect,
                                                   logFileName: searchLogFileName, infoLogger: InfoLogger);

        // Disable verbose move stats from the engine since 
        // this class manages the possibly dumping of verbose move stats itself.
        CeresEngine.OutputVerboseMoveStats = false;

        // Initialize engine
        CeresEngine.Warmup();
        haveInitializedEngine = true;
        OutStream.WriteLine();
      }

      return true;
    }

    private void ShowWeightsFileInfo()
    {
      UCIWriteLine();
      if (EvaluatorDef.Nets.Length == 1)
      {
        ShowWeightsInfo(0);
      }
      else
      {
        OutStream.WriteLine();
        for (int i = 0; i < EvaluatorDef.Nets.Length; i++)
        {
          ShowWeightsInfo(i);
        }
      }
      UCIWriteLine();
    }

    private void ShowWeightsInfo(int i)
    {
      if (EvaluatorDef.Nets[i].Net.Type == NNEvaluatorType.LC0)
      {
        INNWeightsFileInfo net = NNWeightsFiles.LookupNetworkFile(EvaluatorDef.Nets[i].Net.NetworkID);
        string infoStr = net == null ? "(unknown)" : net.ShortStr;
        UCIWriteLine($"Loaded network weights: {i} {infoStr}");
      }
      else
      {
        UCIWriteLine($"Loaded network weights: " + EvaluatorDef.Nets[i].Net.NetworkID);
      }
    }


    /// <summary>
    /// Processes a specified position command, 
    /// with the side effect of resetting the curPositionAndMoves.
    /// </summary>
    /// <param name="command"></param>
    private void ProcessPosition(string command)
    {
      command = StringUtils.WhitespaceRemoved(command);

      string commandLower = command.ToLower();

      string posString;
      if (commandLower.StartsWith("position fen "))
        posString = command.Substring(13);
      else if (commandLower.StartsWith("position startpos"))
        posString = command.Substring(9);
      else
        throw new Exception($"Illegal position command, expected to start with position fen or position startpos");

      PositionWithHistory newPositionAndMoves = PositionWithHistory.FromFENAndMovesUCI(posString);      
      
      curPositionIsContinuationOfPrior = newPositionAndMoves.IsIdenticalToPriorToLastMove(curPositionAndMoves);
      if (!curPositionIsContinuationOfPrior && CeresEngine != null)
      {
        CeresEngine.ResetGame();
      }

      // Switch to the new position and moves
      curPositionAndMoves = newPositionAndMoves;
    }


    /// <summary>
    /// Processes the go command.
    /// </summary>
    /// <param name="command"></param>
    /// <returns></returns>
    private Task<GameEngineSearchResultCeres> ProcessGo(string command)
    {
      return Task.Run(() =>
      {
        try
        {
          SearchLimit searchLimit;

          // Parse the search limit
          searchLimit = UCIManagerHelpers.GetSearchLimit(command, curPositionAndMoves, UCIWriteLine);

          GameEngineSearchResultCeres result = null;
          if (searchLimit != null)
          {
            if (maxTreeVisits != null || maxTreeNodes != null)
            {
              searchLimit = searchLimit with { MaxTreeVisits = maxTreeVisits, MaxTreeNodes = maxTreeNodes };
            }

            if (searchLimit.Value > 0)
            {
              // Run the actual search
              result = RunSearch(searchLimit * searchLimitMultiplier);
            }
          }

          taskSearchCurrentlyExecuting = null;
          return result;
        }
        catch (Exception exc)
        {
          UCIWriteLine("Exception in Ceres engine execution:");
          UCIWriteLine(exc.ToString());
          UCIWriteLine(exc.StackTrace);

          System.Environment.Exit(3);
          return null;
        }
      });
    }



    /// <summary>
    /// Actually runs a search with specified limits.
    /// </summary>
    /// <param name="searchLimit"></param>
    /// <returns></returns>
    private GameEngineSearchResultCeres RunSearch(SearchLimit searchLimit)
    {
      DateTime firstInfoUpdate = DateTime.Now;
      DateTime lastInfoUpdate = DateTime.Now;

      int numUpdatesSent = 0;

      MCTSManager.MCTSProgressCallback callback =
        (manager) =>
        {
          // Choose an appropriate update interval, less frequent for longer searches.
          DateTime now = DateTime.Now;
          float timeSinceFirstUpdate = (float)(now - firstInfoUpdate).TotalSeconds;
          float timeSinceLastUpdate = (float)(now - lastInfoUpdate).TotalSeconds;
          bool isFirstUpdate = numUpdatesSent == 0;
          float UPDATE_INTERVAL_SECONDS = 0;
          if (isFirstUpdate)
          {
            UPDATE_INTERVAL_SECONDS = 0.1f;
          }
          else if (timeSinceFirstUpdate < 5)
          {
            UPDATE_INTERVAL_SECONDS = 0.5f;
          }
          else if (timeSinceFirstUpdate < 30)
          {
            UPDATE_INTERVAL_SECONDS = 1f;
          }
          else
          {
            UPDATE_INTERVAL_SECONDS = 3;
          }

          if (manager != null && timeSinceLastUpdate > UPDATE_INTERVAL_SECONDS && manager.Root.N > 0)
          {
            OutputUCIInfo(CeresEngine.Search.Manager, CeresEngine.Search.SearchRootNode);

            numUpdatesSent++;
            lastInfoUpdate = now;
          }
        };

      GameEngineCeresInProcess.ProgressCallback callbackPlain = obj => callback((MCTSManager)obj);

      // use this? movesSinceNewGame

      // Search from this position (possibly with tree reuse)
      GameEngineSearchResultCeres result = CeresEngine.Search(curPositionAndMoves, searchLimit, gameMoveHistory, callbackPlain) as GameEngineSearchResultCeres;

      GameMoveStat moveStat = new GameMoveStat(gameMoveHistory.Count,
                                               curPositionAndMoves.FinalPosition.MiscInfo.SideToMove,
                                               curPositionAndMoves.FinalPosition,
                                               result.ScoreQ, result.ScoreCentipawns,
                                               float.NaN, //engine1.CumulativeSearchTimeSeconds, 
                                               curPositionAndMoves.FinalPosition.PieceCount,
                                               result.MAvg, result.FinalN, result.FinalN - result.StartingN,
                                               searchLimit,
                                               (float)result.TimingStats.ElapsedTimeSecs, result.NPS);

      gameMoveHistory.Add(moveStat);

      if (SearchFinishedEvent != null) SearchFinishedEvent(result.Search.Manager);

      // Output the final UCI info line containing end of search information
      OutputUCIInfo(result.Search.Manager, result.Search.SearchRootNode, true);

      // Send the best move (using appropriate castling move style).
      UCIWriteLine("bestmove " + result.Search.BestMove.MoveStr(MGMoveNotationStyle.Coordinates, IsChess960OptionSet));

      return result;
    }


    void OutputUCIInfo(MCTSManager manager, MCTSNode searchRootNode, bool isFinalInfo = false)
    {
      BestMoveInfo best = searchRootNode.BestMoveInfo(false);

      if (numPV == 1)
      {
        UCIWriteLine(UCIInfo.UCIInfoString(manager, searchRootNode, best == null ? default : best.BestMoveNode,
                                  showWDL: showWDL, scoreAsQ: scoreAsQ, isChess960: IsChess960OptionSet));
      }
      else
      {
        // Send top move
        UCIWriteLine(UCIInfo.UCIInfoString(manager, searchRootNode, best.BestMoveNode, 1,
                                   showWDL: showWDL, useParentN: !perPVCounters, scoreAsQ: scoreAsQ, isChess960: IsChess960OptionSet));

        // Send other moves visited
        MCTSNode[] sortedN = searchRootNode.ChildrenSorted(s => -(float)s.N);
        int multiPVIndex = 2;
        for (int i = 0; i < sortedN.Length && i < numPV; i++)
        {
          if (sortedN[i] != best.BestMoveNode)
          {
            UCIWriteLine(UCIInfo.UCIInfoString(manager, searchRootNode, sortedN[i], multiPVIndex,
                                       showWDL: showWDL, useParentN: !perPVCounters, scoreAsQ: scoreAsQ));
            multiPVIndex++;
          }
        }

        // Finally show moves that had no visits
        float elapsedTimeSeconds = (float)(DateTime.Now - manager.StartTimeThisSearch).TotalSeconds;
        string timeStr = $"{elapsedTimeSeconds * 1000.0f:F0}";
        for (int i = multiPVIndex - 1; i < numPV; i++)
        {
          (MCTSNode node, EncodedMove move, FP16 p) info = searchRootNode.ChildAtIndexInfo(i);
          if (info.node.IsNull)
          {
            bool isWhite = searchRootNode.Annotation.Pos.MiscInfo.SideToMove == SideType.White;
            EncodedMove moveCorrectPerspective = isWhite ? info.move : info.move.Flipped;
            string str = $"info depth 0 seldepth 0 time {timeStr} nodes 1 score cp 0 tbhits 0 "
                       + $"multipv {multiPVIndex} pv {moveCorrectPerspective.AlgebraicStr} ";
            UCIWriteLine(str);
            multiPVIndex++;
          }
        }

      }
      if (verboseMoveStats && (logLiveStats || isFinalInfo)) OutputVerboseMoveStats();
    }

    /// <summary>
    /// Since "log live stats" is a LC0 only feature (used for example Nibbler)
    /// if in this mode we use the LC0 format instead of Ceres' own.
    /// </summary>
    bool ShouldUseLC0FormatForVerboseMoves => logLiveStats;

    void OutputVerboseMoveStats()
    {

      MCTSNode searchRootNode = CeresEngine.Search.SearchRootNode;
      if (ShouldUseLC0FormatForVerboseMoves)
      {
        foreach (VerboseMoveStat stat in VerboseMoveStatsFromMCTSNode.BuildStats(searchRootNode))
        {
          UCIWriteLine(stat.LC0String);
        }
      }
      else
      {
        CeresEngine.Search.Manager.Context.Root.Dump(1, 1, prefixString: "info string ");
      }
    }

  }
}
