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
using System.Globalization;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem;
using Ceres.Base.OperatingSystem.NVML;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.NNFiles;
using Ceres.Chess.Positions;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.UserSettings;

using Ceres.MCGS.Analysis;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Utils;
using Ceres.MCGS.Visualization.AnalysisGraph;
using Ceres.MCGS.Visualization.PlyVisualization;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.MCGS.UCI;

/// <summary>
/// Manager of UCI game loop, parsing and executing commands from Console
/// and outputting appropriate UCI lines such as bestmove and info.
/// </summary>
public partial class UCIManagerMCGS
{
  const string DEFAULT_DEVICE = "GPU:0";

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
  public Action<MCGSManager> SearchFinishedEvent;

  /// <summary>
  /// Action to be called upon receipt of a request to log a message during search.
  /// </summary>
  public Action<(MCGSSearch, string)> LogSearchInfoMessageEvent;


  public NNEvaluatorDef EvaluatorDef;

  /// <summary>
  /// Ceres engine instance used for current UCI game.
  /// </summary>
  public GameEngineCeresMCGSInProcess CeresEngine;

  /// <summary>
  /// Optional handler for the custom "tcec" command, which launches the live TCEC
  /// broadcast monitor (continuous analysis that auto-follows the live game). The
  /// implementation lives in Ceres.Features (which references Ceres.MCGS), so it is
  /// injected here by the host (DispatchCommands) rather than referenced directly to
  /// avoid a circular project dependency. Invoked with the initialized engine instance.
  /// </summary>
  public Action<GameEngineCeresMCGSInProcess> TCECMonitorHandler;

  GameEngineSearchResultCeresMCGS lastSearchResult;

  /// <summary>
  /// Principal position set collected by the most recent dump-pp command
  /// (used as the start nodes for the deep-rollout command).
  /// </summary>
  PrincipalPosSet lastDumpPPSet;

  /// <summary>
  /// Search manager from which lastDumpPPSet was collected
  /// (deep-rollout requires that the set be from the current search).
  /// </summary>
  MCGSManager lastDumpPPManager;

  /// <summary>
  /// Results of the most recent deep-rollout command.
  /// </summary>
  public DeepRolloutSet LastDeepRolloutSet { get; private set; }

  /// <summary>
  /// Results of the most recent revalue-root command.
  /// </summary>
  public PrincipalRevaluationResult LastRevaluation { get; private set; }

  volatile Task<GameEngineSearchResultCeresMCGS> taskSearchCurrentlyExecuting;

  bool haveInitializedEngine;

  /// <summary>
  /// Optional evaluator to call to benchmark neural network backend.
  /// </summary>
  readonly Action<NNEvaluatorDef, NNEvaluator, int, int, int, int> BackendBenchEvaluator;

  /// <summary>
  /// Optional evaluator to call to search benchmark program (to run for specified number of seconds).
  /// </summary>
  readonly Action<NNEvaluatorDef, int> BenchmarkSearchEvaluator;

  /// <summary>
  /// The position and history associated with the current evaluation.
  /// </summary>
  PositionWithHistory curPositionAndMoves;
  bool curPositionIsContinuationOfPrior;

  List<GameMoveStat> gameMoveHistory = [];

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


  /// <summary>
  /// Optional override search parameters.
  /// </summary>
  public ParamsSearch OverrideParamsSearch;

  /// <summary>
  /// Optional override select parameters.
  /// </summary>
  public ParamsSelect OverrideParamsSelect;


  /// <summary>
  /// Search parameters loaded from a file via the "load-params" UCI command (or null if none).
  /// When set, these take complete precedence: they are used verbatim for all future searches,
  /// bypassing the per-field setoption overlays normally applied by the ParamsSearch getter.
  /// </summary>
  ParamsSearch loadedParamsSearch;

  /// <summary>
  /// Select parameters loaded from a file via the "load-params" UCI command (or null if none).
  /// See <see cref="loadedParamsSearch"/> for precedence semantics.
  /// </summary>
  ParamsSelect loadedParamsSelect;



  void CreateEvaluator()
  {
    InitNetworkAndDeviceSpecsIfNeeded();

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
  public UCIManagerMCGS(NNNetSpecificationString networkSpec,
                        NNDevicesSpecificationString deviceSpec,
                        Action<ParamsSearch> searchModifier = null,
                        Action<ParamsSelect> selectModifier = null,
                        TextReader inStream = null,
                        TextWriter outStream = null,
                        Action<MCGSManager> searchFinishedEvent = null,
                        bool disablePruning = false,
                        string uciLogFileName = null,
                        string searchLogFileName = null,
                        Action<NNEvaluatorDef, NNEvaluator, int, int, int, int> backendBenchEvaluator = null,
                        Action<NNEvaluatorDef, int> benchmarkEvaluator = null,
                        ParamsSearch overrideParamsSearch = null,
                        ParamsSelect overrideParamsSelect = null)
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

    OverrideParamsSearch = overrideParamsSearch;
    OverrideParamsSelect = overrideParamsSelect;

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
  /// When a game-analyze capture writer is active (used to compute a position's output in the
  /// background without it interleaving with the console), the line is redirected there instead.
  /// </summary>
  /// <param name="result"></param>
  void UCIWriteLine(string result = null)
  {
    (gameAnalyzeCaptureWriter ?? OutStream).WriteLine(result);
    if (uciLogWriter != null)
    {
      LogWriteLine("OUT:", result);
    }
  }


  public ParamsSelect ParamsSelect
  {
    get
    {
      // Params loaded from a file are authoritative: use them verbatim, bypassing the overlays below.
      if (loadedParamsSelect != null)
      {
        return loadedParamsSelect;
      }

      ParamsSelect parms = OverrideParamsSelect ?? new ParamsSelect();
      parms.CPUCT = cpuct;
      parms.CPUCTBase = cpuctBase;
      parms.CPUCTFactor = cpuctFactor;
      parms.CPUCTAtRoot = cpuctAtRoot;
      parms.CPUCTBaseAtRoot = cpuctBaseAtRoot;
      parms.CPUCTFactorAtRoot = cpuctFactorAtRoot;
      parms.PolicySoftmax = policySoftmax;
      parms.FPUValue = fpu;
      parms.FPUValueAtRoot = fpuAtRoot;

      // Apply ActionHead settings if enabled
      if (enableActionHead)
      {
        parms.FPUMode = ParamsSelect.FPUType.ActionHead;
        parms.FPUModeAtRoot = ParamsSelect.FPUType.ActionHead;
      }

      SelectModifier?.Invoke(parms);

      return parms;
    }
  }

  public ParamsSearch ParamsSearch
  {
    get
    {
      // Params loaded from a file are authoritative: use them verbatim, bypassing the overlays below.
      if (loadedParamsSearch != null)
      {
        return loadedParamsSearch;
      }

      ParamsSearch parms = OverrideParamsSearch ?? new ParamsSearch();
      parms.MoveOverheadSeconds = moveOverheadSeconds;
      parms.EnableTablebases = parms.EnableTablebases && tablebaseDirectory != null;
      parms.TablebasePaths = tablebaseDirectory;
      parms.EnableGraph = MCGS;
      parms.PathTranspositionMode = pathMode;
      parms.MoveFutilityPruningAggressiveness = futilityPruningDisabled ? 0 : parms.MoveFutilityPruningAggressiveness;
      parms.MaxMemoryBytes = ramLimitMb == 0 ? parms.MaxMemoryBytes : (long)(ramLimitMb * (1024 * 1024));
      parms.ValueTemperature = valueTemperature;
      SearchModifier?.Invoke(parms);
      return parms;
    }
  }


  void ResetGame()
  {
    curPositionAndMoves = PositionWithHistory.FromFENAndMovesUCI(Chess.Position.StartPosition.FEN);
    gameMoveHistory = [];
    CeresEngine?.ResetGame();
  }


  /// <summary>
  /// Runs the UCI loop.
  /// </summary>
  /// <param name="gameAnalyzeStartupArgs">
  /// If non-null, a "game-analyze" specification ("&lt;pgn&gt; &lt;move&gt; &lt;time&gt;") which is
  /// executed once after reset and before the interactive loop begins (used by the
  /// "game-analyze" command-line verb to leave the console in UCI mode pre-positioned).
  /// </param>
  public void PlayUCI(string gameAnalyzeStartupArgs = null, string gameAnalyzeLC0StartupArgs = null)
  {
    ResetGame();

    if (gameAnalyzeStartupArgs != null)
    {
      ProcessGameAnalyzeInteractive("game-analyze " + gameAnalyzeStartupArgs);
    }
    else if (gameAnalyzeLC0StartupArgs != null)
    {
      ProcessGameAnalyzeLC0Interactive("game-analyze-lc0 " + gameAnalyzeLC0StartupArgs);
    }

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
          UCIWriteLine("backendbench    - benchmark of speed of network backend evaluator, optionally [from <int>] [to <int>] [by <int>]");
          UCIWriteLine("benchmark       - search benchmark with specified number of seconds per position, e.g. \"benchmark 10\"");
          UCIWriteLine("deep-rollout    - runs deep rollouts from principal positions of last dump-pp and redisplays table with rollout statistics (optional rollouts per position and exploration multiplier, e.g. \"deep-rollout 100 0.05\")");
          UCIWriteLine("dump-fen        - shows FEN of most recently searched position");
          UCIWriteLine("dump-move-stats - dumps information top level candidate moves");
          UCIWriteLine("dump-pv         - dumps principal variation information from last search");
          UCIWriteLine("dump-pv-detail  - dumps principal variation information from last search (detailed)");
          UCIWriteLine("dump-pp         - dumps principal positions from last search (optional visit percentage and max Q deviation, e.g. \"1% 0.02\")");
          UCIWriteLine("dump-pp-html    - dumps principal positions from last search (like dump-pp but to HTML page)");
          UCIWriteLine("revalue-root    - revalues root/move Q of last search via exploration-ladder deep rollouts from the visit frontier (optional rollouts/position/stage, ladder, dry-up rounds (0=off), e.g. \"revalue-root 50 0.15,0.04,0 12\")");
          UCIWriteLine("dump-info       - dump information about last search (top level candidate moves, principal variation, etc.)");
          UCIWriteLine("dump-info-block - like dump-info but prefixed with a process/GC/machine header and wrapped in begin/end markers (used for programmatic capture)");
          UCIWriteLine("game-analyze    - locate a position (or range) in a PGN by move number and analyze it (prompts for: <pgn file> <move number or range, e.g. 105, 105.., 1-9999, ..7-..12> <time, e.g. 10s>)");
          UCIWriteLine("game-analyze-lc0- like game-analyze but runs the analysis with Lc0 (streaming its UCI output), then loads the position into Ceres without searching");
          UCIWriteLine("dump-time       - dump information about time manager's last decision");
          UCIWriteLine("dump-processor  - dump information about CPUs in this system");
          UCIWriteLine("dump-params     - dump configuration parameters currently in use for Ceres");
          UCIWriteLine("save-params <f> - serialize the current ParamsSearch and ParamsSelect (as JSON) to the specified file");
          UCIWriteLine("load-params <f> - load ParamsSearch and ParamsSelect from a JSON file; these then override all options for future searches");
          UCIWriteLine("dump-store {d}  - dumps full node store for tree from last search (optionally with max depth specifier)");
          UCIWriteLine("dump-trans-pos  - dumps transpositions list (standalone hash)");
          UCIWriteLine("dump-nvidia     - dumps information about NVIDIA CUDA devices detected in the system");
          UCIWriteLine("forward <n>     - advances position by n ply along PV (e.g. \"forward 7\") for subsequent search with graph reuse");
          UCIWriteLine("show-graph-plot - shows a graphical representation of full search graph");
          UCIWriteLine("graph [1-10]    - invokes graph feature to show the principal variations from last search (requires configuration), e.g. graph 7");
          UCIWriteLine("gamecomp        - invokes the game comparison feature to graph the divergence points in one or more games (requires configuration)");
          UCIWriteLine("plyviz          - generates HTML visualization of PlyBin distributions for current position");
          UCIWriteLine("tcec            - live-analyze the current TCEC broadcast game, auto-following each new move (q/Esc to return)");
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
            while (CeresEngine.Search == null)
            {
              Thread.Sleep(20);
            }

            CeresEngine.Search.Manager.ExternalStopRequested = true;
            if (taskSearchCurrentlyExecuting != null)
            {
              taskSearchCurrentlyExecuting.Wait();
              // if (!debug && taskSearchCurrentlyExecuting != null) taskSearchCurrentlyExecuting.Result?.Search?.Manager?.Dispose();
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

          CeresEngine?.Dispose();

          // In rare cases we've seen a crash during the Environment.Exit() below,
          // presumably due to cleanup which was in progress.
          // To try to avoid this ugliness, do a GC first and pause briefly before exit.
          // TODO: possibly no longer needed, disabled. Remove.
          //GC.Collect(4, GCCollectionMode.Forced, true);
          //Thread.Sleep(500);

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


        // Custom commands
        case string c when c.StartsWith("dump-pp"):
          if (CeresEngine?.Search?.Manager != null)
          {
            bool useHTML = c.StartsWith("dump-pp-html");
            ProcessDumpPPCommand(c, useHTML);
          }
          else
          {
            UCIWriteLine("info string No search manager created");
          }
          break;

        case string c when c.StartsWith("deep-rollout"):
          if (CeresEngine?.Search?.Manager != null)
          {
            ProcessDeepRolloutCommand(c);
          }
          else
          {
            UCIWriteLine("info string No search manager created");
          }
          break;

        case string c when c.StartsWith("revalue-root"):
          if (CeresEngine?.Search?.Manager != null)
          {
            ProcessRevalueRootCommand(c);
          }
          else
          {
            UCIWriteLine("info string No search manager created");
          }
          break;

        case "dump-info":
          ProcessDumpInfoCommand(blockFormat: false);
          break;

        case "dump-info-block":
          ProcessDumpInfoCommand(blockFormat: true);
          break;

        case string c when c.StartsWith("game-analyze-lc0"):
          ProcessGameAnalyzeLC0Interactive(c);
          break;

        case string c when c.StartsWith("game-analyze"):
          ProcessGameAnalyzeInteractive(c);
          break;

        case string c when c.StartsWith("graph"):
          if (CeresEngine?.Search != null)
          {
            string[] partsGraph = c.TrimEnd().Split(" ");
            string optionsStr = partsGraph.Length > 1 ? c[6..] : null;
            AnalysisGraphOptions graphOptions = AnalysisGraphOptions.FromString(optionsStr);
            AnalysisGraphGenerator graphGenerator = new AnalysisGraphGenerator(CeresEngine.Search, graphOptions);
            graphGenerator.Write(true);
          }
          else
          {
            UCIWriteLine("info string No search manager created");
          }
          break;

        case string c when c.StartsWith("gamecomp"):
          UCIManagerHelpersMCGS.ProcessGameComp(c, UCIWriteLine);
          break;

        case "plyviz":
          if (InitializeEngineIfNeeded())
          {
            NNEvaluator evaluator = CeresEngine.Evaluators.Evaluator0;
            NNEvaluatorResult plyvizResult = evaluator.Evaluate(curPositionAndMoves);
            if (plyvizResult.PlyBinMoveProbs == null)
            {
              UCIWriteLine("info string PlyBin outputs not available for this network");
            }
            else
            {
              Position plyvizPos = curPositionAndMoves.FinalPosition;
              bool isBlack = plyvizPos.SideToMove == SideType.Black;
              (Half[,] moveProbs, Half[,] capProbs) = PlyBinVisualization.ConvertProbs(plyvizResult, isBlack);
              List<PVCandidate> pvs = PVExtractor.ExtractPVs(plyvizResult, plyvizPos);
              string fen = plyvizPos.FEN;
              string plyvizTitle = "PlyBin - " + fen;
              string plyvizPath = Path.Combine(Path.GetTempPath(), "ceres_plyviz.html");
              List<PlyBinEntry> plyvizEntries =
              [
                new PlyBinEntry(fen, plyvizTitle, moveProbs, capProbs, plyvizResult.ToString(),
                                PunimSelfProbs: plyvizResult.PunimSelfProbs,
                                PunimOpponentProbs: plyvizResult.PunimOpponentProbs,
                                ProjectedPVs: pvs, MLH: plyvizResult.M)
              ];
              string outputFile = PlyBinVisualization.GenerateMulti(plyvizEntries, plyvizTitle, plyvizPath);
              UCIWriteLine("info string Ply Visualization page output to " + outputFile);
              StringUtils.LaunchBrowserWithURL(outputFile);
            }
          }
          break;

        case "dump-params":
          if (CeresEngine?.Search?.Manager != null)
          {
            CeresEngine.Search.Manager.DumpParams();
          }
          else
          {
            // No search has run yet; dump the parameters that would be used for the next search
            // (this reflects any params loaded via "load-params").
            DumpPendingParams();
          }
          break;

        case string c when c.StartsWith("load-params"):
          ProcessLoadParamsCommand(c);
          break;

        case string c when c.StartsWith("save-params"):
          ProcessSaveParamsCommand(c);
          break;

        case string c when c.StartsWith("forward"):
          ProcessForward(c);
          break;

        case "dump-processor":
          HardwareManager.DumpProcessorInfo();
          break;

        case string c when c.StartsWith("download"):
          UCIManagerHelpersMCGS.ProcessDownloadCommand(c, UCIWriteLine);
          break;

        case string c when c.StartsWith("backendbench"):
          if (BackendBenchEvaluator == null)
          {
            Console.WriteLine("No BackendBenchEvaluator installed, cannot benchmark.");
          }
          else
          {
            if (InitializeEngineIfNeeded())
            {
              BackendbenchOptions options = BackendbenchOptionsParser.Parse(c);
              BackendBenchEvaluator(EvaluatorDef, CeresEngine.Evaluators.Evaluator0,
                                    options.StartIndex, options.EndIndex, options.StepSize, options.RepeatCount);
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
          if (CeresEngine?.Search?.Manager != null)
          {
            Console.WriteLine("info string " + CeresEngine.Search.Manager.StartPosAndPriorMoves.FinalPosition.FEN);
          }
          else
          {
            UCIWriteLine("info string No search manager created");
          }
          break;

        case "dump-time":
          if (CeresEngine?.Search?.Manager != null)
          {
            CeresEngine.Search.Manager.DumpTimeInfo(OutStream, CeresEngine.Search.Manager.Engine.SearchRootNode);
          }
          else
          {
            UCIWriteLine("info string No search manager created");
          }
          break;

        case string c when c.StartsWith("dump-store"):
          if (CeresEngine?.Search?.Manager != null)
          {
            // Default to depth 2, optional argument to provide another depth
            int maxDepth = 2;
            string[] dumpStoreParts = c.Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (dumpStoreParts.Length > 1 && int.TryParse(dumpStoreParts[1], out int parsedDepth))
            {
              maxDepth = parsedDepth;
            }
            CeresEngine.Search.Manager.Engine.Graph.DumpNodesStructure(maxDepth, 0);
            CeresEngine.DumpStoreUsageSummary();

            Console.WriteLine();
            Console.WriteLine("GC total pause " + GC.GetTotalPauseDuration().TotalSeconds + " sec");
            Console.WriteLine("Pause durations:");
            foreach (var p in GC.GetGCMemoryInfo().PauseDurations)
            {
              Console.WriteLine("  " + p.TotalSeconds);
            }

            long bytesTotalMemory = GC.GetTotalMemory(true);
            Console.WriteLine("GB total managed memory: " + Math.Round(bytesTotalMemory / (1024.0 * 1024.0 * 1024.0), 3));
          }
          else
          {
            UCIWriteLine("info string No search manager created");
          }
          break;

        case "dump-trans-pos":
          if (CeresEngine?.Search?.Manager != null)
          {
            CeresEngine.Search.Manager.Engine.Graph.DumpTranspositionsStandalone();
          }
          else
          {
            UCIWriteLine("info string No search manager created");
          }
          break;


        case "validate-store":
          if (CeresEngine?.Search?.Manager != null)
          {
            CeresEngine.Search.Manager.Engine.Graph.Validate();
          }
          else
          {
            UCIWriteLine("info string No search manager created");
          }


          break;

        case "dump-move-stats":
          if (CeresEngine?.Search?.Manager != null)
          {
            MCGSManager manager = CeresEngine.Search.Manager;
            ManagerChooseBestMoveMCGS bestMoveCalc = new(manager, manager.Engine.SearchRootNode, false, default, true);
            BestMoveInfoMCGS bestMoveInfo = bestMoveCalc.BestMoveCalc;
            OutputVerboseMoveStats(bestMoveInfo);
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

        case "show-graph-plot":
          UCIWriteLine("show-graph-plot not yet implemented");
          break;

#if NOT
          if (CeresEngine?.Search != null)
            TreePlot.Show(CeresEngine.Search.Manager.Context.Root.StructRef);
          else
            UCIWriteLine("info string No search manager created");
          break;
#endif
        case string c when c.StartsWith("save-graph-plot"):
          UCIWriteLine("save-graph-plot not yet implemented");
          break;
#if NOT
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
#endif

        case "waitdone": // custom verb used for test driver
          taskSearchCurrentlyExecuting?.Wait();
          break;

        case "tcec":
          // Launch the live TCEC broadcast monitor (implementation injected by the host
          // via TCECMonitorHandler). Runs synchronously and returns here when the user
          // quits (q/Esc), resuming normal UCI command processing.
          if (InitializeEngineIfNeeded())
          {
            if (TCECMonitorHandler != null)
            {
              TCECMonitorHandler(CeresEngine);
            }
            else
            {
              UCIWriteLine("info string TCEC monitor handler not registered");
            }
          }
          break;

        default:
          UCIWriteLine($"error Unknown command: {command}");
          break;
      }
    }
  }

  private void ProcessDumpPPCommand(string c, bool useHTML)
  {
    float minVisitsFraction = 0.01f; // 1%
    float maxQDeviation = 0.02f;

    // arguments: <fraction_as_percentage> <maxQDeviation>
    // where there can be none, just fraction_as_percentage, or both
    // when not specified uses the defaults above
    string[] partsPP = c.Split(" ", StringSplitOptions.RemoveEmptyEntries);

    // Parse optional arguments
    if (partsPP.Length > 1)
    {
      // Try to parse first argument as fraction percentage (e.g., "0.5%" or "0.5")
      string fractionStr = partsPP[1].TrimEnd('%');
      if (float.TryParse(fractionStr, out float parsedFraction))
      {
        // If it was specified as percentage (e.g., "0.5"), convert to fraction (0.005)
        minVisitsFraction = partsPP[1].Contains('%') ? parsedFraction / 100.0f : parsedFraction;
      }
    }

    if (partsPP.Length > 2)
    {
      // Try to parse second argument as maxQDeviation
      if (float.TryParse(partsPP[2], out float parsedDeviation))
      {
        maxQDeviation = parsedDeviation;
      }
    }

    MCGSEngine engine = CeresEngine.Search.Manager.Engine;
    MGPosition rootPos = engine.SearchRootNode.Graph.Store.PositionHistory.FinalPosMG;
    PrincipalPosSet pp = PrincipalPosSet.CollectNodesAboveVisitThreshold(rootPos, CeresEngine.Search.SearchRootNode,
                                                                         (int)(CeresEngine.Search.SearchRootNode.N * minVisitsFraction),
                                                                         maxQDeviation);

    // Retain for possible subsequent deep-rollout command.
    lastDumpPPSet = pp;
    lastDumpPPManager = CeresEngine.Search.Manager;

    if (useHTML)
    {
      PrincipalPosSetDumperHTML.DumpToGraphHTML(pp, CeresEngine.Search.BestMove);
    }
    else
    {
      PrincipalPosSetDumper.DumpToConsoleGraphical(pp, CeresEngine.Search.BestMove);
    }
  }


  /// <summary>
  /// Processes the deep-rollout command: runs deep rollouts from the principal positions
  /// collected by the most recent dump-pp command, then redisplays the principal position
  /// tables augmented with columns characterizing the rollout results.
  ///
  /// Arguments: deep-rollout [num rollouts per position] [exploration multiplier]
  /// </summary>
  /// <param name="c"></param>
  private void ProcessDeepRolloutCommand(string c)
  {
    const int DEFAULT_NUM_ROLLOUTS = 100;
    const float DEFAULT_EXPLORATION_MULTIPLIER = 0.05f;

    // Make sure any search still executing has completed (the graph must be quiescent).
    taskSearchCurrentlyExecuting?.Wait();

    if (lastDumpPPSet == null || lastDumpPPManager != CeresEngine.Search.Manager)
    {
      UCIWriteLine("info string No principal positions available for current search, run dump-pp first");
      return;
    }

    if (lastDumpPPSet.Members.Count == 0)
    {
      UCIWriteLine("info string Last dump-pp found no principal positions, nothing to roll out");
      return;
    }

    string[] partsDR = c.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    int numRollouts = DEFAULT_NUM_ROLLOUTS;
    if (partsDR.Length > 1
     && (!int.TryParse(partsDR[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out numRollouts) || numRollouts <= 0))
    {
      UCIWriteLine($"info string Invalid number of rollouts: {partsDR[1]}");
      return;
    }

    float explorationMultiplier = DEFAULT_EXPLORATION_MULTIPLIER;
    if (partsDR.Length > 2
     && (!float.TryParse(partsDR[2], NumberStyles.Float, CultureInfo.InvariantCulture, out explorationMultiplier) || explorationMultiplier < 0))
    {
      UCIWriteLine($"info string Invalid exploration multiplier: {partsDR[2]}");
      return;
    }

    UCIWriteLine($"info string Running deep rollouts: {lastDumpPPSet.Members.Count} principal positions x {numRollouts} rollouts,"
               + $" exploration multiplier {explorationMultiplier}");

    LastDeepRolloutSet = DeepRolloutSet.Run(lastDumpPPManager, lastDumpPPSet, numRollouts, explorationMultiplier);

    UCIWriteLine($"info string Deep rollouts complete in {LastDeepRolloutSet.TimingStats.ElapsedTimeSecs:F2} sec");
    UCIWriteLine();

    // Redisplay the principal position tables, now augmented with deep rollout statistics columns.
    DeepRolloutSet drSet = LastDeepRolloutSet;
    string StatText(PrincipalPos pos, Func<DeepRolloutNodeStats, string> formatter)
      => drSet.TryGetStats(pos.LeafNode.Index, out DeepRolloutNodeStats drStats) && drStats.NumDistinctPaths > 0
           ? formatter(drStats)
           : "";

    (string colName, Func<PrincipalPos, string> calcFunc)[] drColumns =
    [
      ("DRCount", pos => StatText(pos, s => s.NumDistinctPaths.ToString())),
      ("DRDepth", pos => StatText(pos, s => s.MedianDepthBelowNode.ToString("0.#"))),
      ("DRAvg", pos => StatText(pos, s =>
        {
          // Convert from the principal position's side-to-move perspective to the root player's.
          bool isRootPlayerPerspective = (pos.PathFromRoot.Count % 2) == 1;
          return PrincipalPosSetDumper.FormatQValue(isRootPlayerPerspective ? s.AvgLeafQ : -s.AvgLeafQ);
        })),
      ("DRStd", pos => StatText(pos, s => s.StdDevLeafQ.ToString("F3"))),
    ];

    PrincipalPosSetDumper.DumpToConsoleGraphical(lastDumpPPSet, CeresEngine.Search.BestMove, drColumns);

    Spectre.Console.AnsiConsole.MarkupLine("[yellow]DRCount[/]       - Number of distinct deep rollout lines explored below this principal position");
    Spectre.Console.AnsiConsole.MarkupLine("[yellow]DRDepth[/]       - Median depth (plies below the principal position) of those lines");
    Spectre.Console.AnsiConsole.MarkupLine("[yellow]DRAvg[/]         - Average leaf Q of those lines (root player's perspective)");
    Spectre.Console.AnsiConsole.MarkupLine("[yellow]DRStd[/]         - Standard deviation of leaf Q across those lines");
    Spectre.Console.AnsiConsole.WriteLine();
  }


  /// <summary>
  /// Processes the revalue-root command: collects the frontier of the well-visited subtree of
  /// the last search, runs an exploration ladder of deep rollouts from each frontier position,
  /// and displays revalued root and per-move Q estimates under a family of backup operators
  /// (visit-weighted average, negamax, soft-minimax) along with classifications of the
  /// frontier evidence.
  ///
  /// Arguments: revalue-root [rollouts per position per stage] [epsilon ladder, comma-separated] [dry-up rounds, 0=off]
  /// </summary>
  /// <param name="c"></param>
  private void ProcessRevalueRootCommand(string c)
  {
    const int DEFAULT_ROUNDS_PER_STAGE = 20;

    // For interactive analysis use a much more generous dry-up threshold than in play:
    // repeated rollouts of a known line still back up (sequential refinement) and may
    // re-route it, so saturation deserves more patience here.
    const int DEFAULT_DRY_UP_ROUNDS = PrincipalRevaluation.ANALYSIS_DRY_UP_ROUNDS;

    // Make sure any search still executing has completed (the graph must be quiescent).
    taskSearchCurrentlyExecuting?.Wait();

    MCGSManager manager = CeresEngine.Search.Manager;

    string[] parts = c.Split(' ', StringSplitOptions.RemoveEmptyEntries);

    int roundsPerStage = DEFAULT_ROUNDS_PER_STAGE;
    if (parts.Length > 1
     && (!int.TryParse(parts[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out roundsPerStage) || roundsPerStage <= 0))
    {
      UCIWriteLine($"info string Invalid rollouts per position per stage: {parts[1]}");
      return;
    }

    float[] epsilonLadder = null;
    if (parts.Length > 2)
    {
      string[] epsParts = parts[2].Split(',', StringSplitOptions.RemoveEmptyEntries);
      epsilonLadder = new float[epsParts.Length];
      for (int i = 0; i < epsParts.Length; i++)
      {
        if (!float.TryParse(epsParts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out epsilonLadder[i]) || epsilonLadder[i] < 0)
        {
          UCIWriteLine($"info string Invalid epsilon ladder entry: {epsParts[i]}");
          return;
        }
      }
    }

    int dryUpRounds = DEFAULT_DRY_UP_ROUNDS;
    if (parts.Length > 3
     && (!int.TryParse(parts[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out dryUpRounds) || dryUpRounds < 0))
    {
      UCIWriteLine($"info string Invalid dry-up rounds: {parts[3]}");
      return;
    }

    UCIWriteLine($"info string Running revaluation: ladder [{string.Join(", ", epsilonLadder ?? PrincipalRevaluation.DEFAULT_EPSILON_LADDER)}]"
               + $" x {roundsPerStage} rollouts/position/stage"
               + $" (dry-up after {(dryUpRounds == 0 ? "never" : dryUpRounds.ToString())} repeat rounds)");

    LastRevaluation = PrincipalRevaluation.Run(manager, roundsPerStage, epsilonLadder, dryUpRounds: dryUpRounds);

    UCIWriteLine($"info string Revaluation complete in {LastRevaluation.ElapsedSecs:F2} sec"
               + $" ({LastRevaluation.NumRolloutVisits} rollout visits over {LastRevaluation.NumFrontier} frontier positions)");
    UCIWriteLine();

    PrincipalRevaluationDumper.DumpToConsole(LastRevaluation, CeresEngine.Search.BestMove);

    // Purely informational: report whether the rollout evidence would prefer a different move
    // (the blended-Q comparison; move choice during play is never affected).
    GNode revalRootNode = manager.Engine.SearchRootNode;
    BestMoveInfoMCGS provisional = new ManagerChooseBestMoveMCGS(manager, revalRootNode, false, default, false).BestMoveCalc;
    RevaluationSwitchDecision decision = PrincipalRevaluation.CalcBlendedQSwitchDecision(
                                           manager, revalRootNode, provisional, LastRevaluation);
    UCIWriteLine(decision.WouldSwitch
      ? $"info string reval decision: rollout evidence would prefer {decision.CandidateMove} over {decision.BaselineMove} ({decision.Description})"
      : $"info string reval decision: rollout evidence keeps {decision.BaselineMove} ({decision.Description})");
  }


  /// <summary>
  /// Dumps the PV (principal variation) to the output stream.
  /// </summary>
  /// <param name="withDetail"></param>
  private void DumpPV(bool withDetail)
  {
    if (CeresEngine?.Search?.Manager != null)
    {
      GNode searchRootNode = CeresEngine.Search.Manager.Engine.SearchRootNode;

      // Avoid showing PV which extends to tiny fraction of the graph
      int minN = (int)Math.Max(1, (int)searchRootNode.N * 0.0001f);

      if (withDetail)
      {
        MCGSPosGraphNodeDumper.DumpPV(CeresEngine.Search.Manager, searchRootNode, true, this.OutStream, null, minN);
      }
      else
      {
        SearchPrincipalVariationMCGS pv2 = new(CeresEngine.Search.Manager, searchRootNode, default, true, minN);
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


  /// <summary>
  /// Handles the "dump-info" / "dump-info-block" commands. If a search is currently in progress the
  /// dump is deferred to the engine's next quiescent point (so the mutating graph is never read from
  /// this UCI input thread); otherwise it is emitted immediately. When blockFormat is true the dump
  /// is prefixed with a process/GC/machine header and wrapped in begin/end markers.
  /// </summary>
  void ProcessDumpInfoCommand(bool blockFormat)
  {
    // If a search is running, route through the engine's thread-safe (quiescent) dump mechanism.
    if (CeresEngine != null && CeresEngine.IsSearching)
    {
      CeresEngine.RequestDumpInfo("UCI",
        blockFormat ? GameEngineCeresMCGSInProcess.DumpInfoFormat.Block
                    : GameEngineCeresMCGSInProcess.DumpInfoFormat.Plain);
      return;
    }

    // No search active: safe to read and dump immediately on this thread.
    if (blockFormat)
    {
      DiagnosticsBlock.WriteBlock(Console.Out, w =>
      {
        if (CeresEngine?.Search?.Manager != null)
        {
          CeresEngine.Search.Manager.DumpFullInfo(lastSearchResult, w, "UCI");
        }
        else
        {
          w.WriteLine("info string No search manager created");
        }
      });
    }
    else
    {
      if (CeresEngine?.Search?.Manager != null)
      {
        CeresEngine.Search.Manager.DumpFullInfo(lastSearchResult, Console.Out, "UCI");
      }
      else
      {
        UCIWriteLine("info string No search manager created");
      }
    }
  }


  /// <summary>
  /// Dumps the ParamsSelect / ParamsSearch / ParamsSearchExecution that would be used for the next
  /// search (reflecting any params loaded via "load-params"). Used by "dump-params" when no search
  /// manager has yet been created.
  /// </summary>
  void DumpPendingParams()
  {
    ParamsSelect pendingSelect = ParamsSelect;
    ParamsSearch pendingSearch = ParamsSearch;
    Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSelect>(pendingSelect, new ParamsSelect(), false));
    Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearch>(pendingSearch, new ParamsSearch(), false));
    Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(pendingSearch.Execution, new ParamsSearchExecution(), false));
    Console.WriteLine(ObjUtils.FieldValuesDumpString<ParamsRootMinimaxBlend>(pendingSearch.RootMinimaxBlend, new ParamsRootMinimaxBlend(), false));
  }


  /// <summary>
  /// Handles the "load-params &lt;filename&gt;" command: loads serialized ParamsSearch and ParamsSelect
  /// from the specified JSON file, makes them authoritative for all future searches, and clears the
  /// currently-loaded search engine so the next search rebuilds with them.
  /// </summary>
  void ProcessLoadParamsCommand(string command)
  {
    string fileName = command.Substring("load-params".Length).Trim();
    if (string.IsNullOrEmpty(fileName))
    {
      UCIWriteLine("info string load-params requires a filename, e.g. load-params my.ceres.params");
      return;
    }

    try
    {
      ParamsFileContents contents = ParamsFileSerializer.Load(fileName);
      loadedParamsSearch = contents.ParamsSearch;
      loadedParamsSelect = contents.ParamsSelect;

      // Clear the currently-loaded search engine (if any) so the next search rebuilds with these params.
      ReinitializeEngine();

      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        $"Loaded search params from {fileName} - these now override all UCI options for future searches.");
    }
    catch (Exception exc)
    {
      UCIWriteLine($"info string load-params failed: {exc.Message}");
    }
  }


  /// <summary>
  /// Handles the "save-params &lt;filename&gt;" command: serializes the current ParamsSearch and
  /// ParamsSelect (those in use by the active search if any, else those that would be used next)
  /// to the specified JSON file. Useful for generating an editable template to later load-params.
  /// </summary>
  void ProcessSaveParamsCommand(string command)
  {
    string fileName = command.Substring("save-params".Length).Trim();
    if (string.IsNullOrEmpty(fileName))
    {
      UCIWriteLine("info string save-params requires a filename, e.g. save-params my.ceres.params");
      return;
    }

    try
    {
      ParamsSearch searchToSave;
      ParamsSelect selectToSave;
      if (CeresEngine?.Search?.Manager != null)
      {
        searchToSave = CeresEngine.Search.Manager.ParamsSearch;
        selectToSave = CeresEngine.Search.Manager.ParamsSelect;
      }
      else
      {
        searchToSave = ParamsSearch;
        selectToSave = ParamsSelect;
      }

      ParamsFileSerializer.Save(fileName, searchToSave, selectToSave);
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Saved current search params to {fileName}.");
    }
    catch (Exception exc)
    {
      UCIWriteLine($"info string save-params failed: {exc.Message}");
    }
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


  private void InitNetworkAndDeviceSpecsIfNeeded()
  {
    if (NetworkSpec == null)
    {
      NetworkSpec = new NNNetSpecificationString(CeresUserSettingsManager.Settings.DefaultNetworkSpecString);
    }

    if (DeviceSpec == null)
    {
      string deviceString = CeresUserSettingsManager.Settings.DefaultDeviceSpecString ?? DEFAULT_DEVICE;
      DeviceSpec = new NNDevicesSpecificationString(deviceString);
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
          CreateEvaluator();
        }
      }

      OutStream.WriteLine($"uci info Engine running in {(ParamsSearch.EnableGraph ? "MCGS" : "MCTS")} mode using PathMode {pathMode}");

      ShowWeightsFileInfo();

      // Create the engine (to be subsequently reused).
      CeresEngine = new GameEngineCeresMCGSInProcess("CeresV2", EvaluatorDef, ParamsSearch, ParamsSelect,
                                                        logFileName: searchLogFileName, infoLogger: InfoLogger,
                                                        emitMiniLog: enableMiniLog)
      {
        // Disable verbose move stats from the engine since 
        // this class manages the possibly dumping of verbose move stats itself.
        OutputVerboseMoveStats = false
      };

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
    {
      posString = command[13..];
    }
    else if (commandLower.StartsWith("position startpos"))
    {
      posString = command[9..];
    }
    else
    {
      throw new Exception($"Illegal position command, expected to start with position fen or position startpos");
    }

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
  /// Processes the forward command which advances position along the principal variation.
  /// </summary>
  /// <param name="command">The forward command with ply count argument (e.g., "forward 7")</param>
  private void ProcessForward(string command)
  {
    // Verify search manager exists
    if (CeresEngine?.Search?.Manager == null)
    {
      UCIWriteLine("info string No search manager created - run a search first");
      return;
    }

    // Parse the ply count argument
    string[] parts = command.Split(' ', StringSplitOptions.RemoveEmptyEntries);
    if (parts.Length != 2)
    {
      UCIWriteLine("info string forward command requires exactly one positive integer argument (number of ply)");
      return;
    }

    if (!int.TryParse(parts[1], out int numPly) || numPly < 1)
    {
      UCIWriteLine("info string forward command requires a positive integer argument");
      return;
    }

    // Get the principal variation from the current search
    GNode searchRootNode = CeresEngine.Search.Manager.Engine.SearchRootNode;

    // Minimum N for PV nodes (avoid noisy nodes)
    int minN = (int)Math.Max(1, searchRootNode.N * 0.0001f);

    SearchPrincipalVariationMCGS pv = new(CeresEngine.Search.Manager, searchRootNode, default, true, minN);

    // Check if we have enough PV nodes to advance
    if (pv.Nodes.Count <= numPly)
    {
      UCIWriteLine($"info string Cannot advance {numPly} ply - PV only has {pv.Nodes.Count - 1} moves");
      return;
    }

    // Build the list of moves from the PV to advance by
    List<string> movesToAdd = [];
    for (int i = 0; i < numPly; i++)
    {
      GNodeAndOptionalEdge nodeEdge = pv.Nodes[i];
      if (!nodeEdge.HasEdge || nodeEdge.Edge.IsNull)
      {
        UCIWriteLine($"info string Cannot advance {numPly} ply - PV terminates at ply {i}");
        return;
      }

      string moveStr = nodeEdge.Edge.MoveMG.MoveStr(MGMoveNotationStyle.Coordinates, IsChess960OptionSet);
      movesToAdd.Add(moveStr);
    }

    // Create a new position by cloning and applying the PV moves to the current position
    PositionWithHistory newPosition = new(curPositionAndMoves);
    foreach (string move in movesToAdd)
    {
      newPosition.AppendMove(move);
    }

    // Get the new search root node (the position after advancing)
    GNode newRootNode = pv.Nodes[numPly].ParentNode;
    string newFEN = newRootNode.CalcPosition().ToPosition.FEN;
    double newRootQ = newRootNode.Q;

    // Update the current position
    curPositionAndMoves = newPosition;
    curPositionIsContinuationOfPrior = true; // Signal that graph reuse should be attempted

    // Output the info line
    UCIWriteLine($"info string advanced {numPly} ply along the PV to {newFEN} with root Q = {newRootQ:F2}");
  }


  /// <summary>
  /// Processes the go command.
  /// </summary>
  /// <param name="command"></param>
  /// <returns></returns>
  private Task<GameEngineSearchResultCeresMCGS> ProcessGo(string command)
  {
    return Task.Run(() =>
    {
      try
      {
        SearchLimit searchLimit;

        // Parse the search limit
        searchLimit = UCIManagerHelpersMCGS.GetSearchLimit(command, curPositionAndMoves, UCIWriteLine);

        GameEngineSearchResultCeresMCGS result = null;
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
  private GameEngineSearchResultCeresMCGS RunSearch(SearchLimit searchLimit)
  {
    DateTime firstInfoUpdate = DateTime.Now;
    DateTime lastInfoUpdate = DateTime.Now;

    int numUpdatesSent = 0;

    MCGSManager.MCGSProgressCallback callback =
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

        if (manager != null && timeSinceLastUpdate > UPDATE_INTERVAL_SECONDS && manager.Engine.SearchRootNode.N > 0)
        {
          ManagerChooseBestMoveMCGS bestMoveCalc = new(manager, manager.Engine.SearchRootNode, false, default, false);
          BestMoveInfoMCGS bestMoveInfo = bestMoveCalc.BestMoveCalc;
          OutputUCIInfo(manager, bestMoveInfo, false);

          numUpdatesSent++;
          lastInfoUpdate = now;
        }
      };

    GameEngineCeresMCGSInProcess.ProgressCallback callbackPlain = obj => callback((MCGSManager)obj);

    // use this? movesSinceNewGame

    // Search from this position (possibly with tree reuse)
    GameEngineSearchResultCeresMCGS result = CeresEngine.Search(curPositionAndMoves, searchLimit, gameMoveHistory, callbackPlain) as GameEngineSearchResultCeresMCGS;
    lastSearchResult = result;

    GameMoveStat moveStat = new(gameMoveHistory.Count,
                                curPositionAndMoves.FinalPosition.MiscInfo.SideToMove,
                                curPositionAndMoves.FinalPosition,
                                result.ScoreQ, result.ScoreCentipawns,
                                float.NaN, //engine1.CumulativeSearchTimeSeconds, 
                                curPositionAndMoves.FinalPosition.PieceCount,
                                result.MAvg, result.FinalN, result.FinalN - result.StartingN,
                                searchLimit,
                                (float)result.TimingStats.ElapsedTimeSecs, result.NPS);

    gameMoveHistory.Add(moveStat);

    SearchFinishedEvent?.Invoke(result.Search.Manager);

    // Output the final UCI info line containing end of search information,
    OutputUCIInfo(CeresEngine.Search.Manager, result.BestMoveInfo, true);

    // Send the best move (using appropriate castling move style).
    UCIWriteLine("bestmove " + result.BestMoveInfo.BestMove.MoveStr(MGMoveNotationStyle.Coordinates, IsChess960OptionSet));

    return result;
  }


  void OutputUCIInfo(MCGSManager manager, BestMoveInfoMCGS bestMoveInfo, bool isFinalInfo = false)
  {
    if (numPV == 1)
    {
      UCIWriteLine(UCIInfoMCGS.UCIInfoString(manager,
                                             //                                               bestMoveInfo == null ? default : bestMoveInfo.BestMoveEdge,
                                             showWDL: showWDL, scoreAsQ: scoreAsQ, isChess960: IsChess960OptionSet));
    }
    else
    {
      // Send top move
      UCIWriteLine(UCIInfoMCGS.UCIInfoString(manager, bestMoveInfo.BestMoveEdge.ChildNode, 1,
                                             showWDL: showWDL, useParentN: !perPVCounters, scoreAsQ: scoreAsQ, isChess960: IsChess960OptionSet));

      // Send other moves visited
      GEdge[] sortedN = manager.Engine.SearchRootNode.EdgesSorted(s => -(float)s.N);
      int multiPVIndex = 2;
      for (int i = 0; i < sortedN.Length && i < numPV; i++)
      {
        if (sortedN[i] != bestMoveInfo.BestMoveEdge)
        {
          UCIWriteLine(UCIInfoMCGS.UCIInfoString(manager, sortedN[i].ChildNode, multiPVIndex,
                                                showWDL: showWDL, useParentN: !perPVCounters, scoreAsQ: scoreAsQ));
          multiPVIndex++;
        }
      }

      GNode searchRootNode = manager.Engine.SearchRootNode;

      // Finally show moves that had no visits
      float elapsedTimeSeconds = (float)(DateTime.Now - manager.StartTimeThisSearch).TotalSeconds;
      string timeStr = $"{elapsedTimeSeconds * 1000.0f:F0}";
      for (int i = multiPVIndex - 1; i < Math.Min(numPV, searchRootNode.NumPolicyMoves); i++)
      {
        MGMove mgMove;
        MGPosition mgPosition;
        if (i >= searchRootNode.NumEdgesExpanded)
        {
          mgPosition = searchRootNode.CalcPosition();
          EncodedMove move = searchRootNode.ChildEdgeHeaderAtIndex(i).Move;
          mgMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(move, mgPosition);
          string moveCorrectPerspectiveStr = MGMoveToString.AlgebraicMoveString(mgMove, mgPosition.ToPosition);
          string str = $"info depth 0 seldepth 0 time {timeStr} nodes 1 score cp 0 tbhits 0 "
                     + $"multipv {multiPVIndex} pv {moveCorrectPerspectiveStr} ";
          UCIWriteLine(str);
          multiPVIndex++;
        }
      }

    }

    if (verboseMoveStats && (logLiveStats || isFinalInfo))
    {
      OutputVerboseMoveStats(bestMoveInfo);
    }
  }

  /// <summary>
  /// Since "log live stats" is a LC0 only feature (used for example Nibbler)
  /// if in this mode we use the LC0 format instead of Ceres' own.
  /// </summary>
  bool ShouldUseLC0FormatForVerboseMoves => logLiveStats;

  internal void OutputVerboseMoveStats(BestMoveInfoMCGS bestMoveInfo)
  {
    GNode searchRootNode = CeresEngine.Search.Manager.Engine.SearchRootNode;

    if (ShouldUseLC0FormatForVerboseMoves)
    {
      foreach (VerboseMoveStat stat in VerboseMoveStatsFromMCGSNode.BuildStats(CeresEngine.Search.Manager, bestMoveInfo))
      {
        UCIWriteLine(stat.LC0String);
      }
    }
    else
    {
      MCGSPosGraphNodeDumper.DumpNodeStr(CeresEngine.Search.Manager, 0, searchRootNode, default, searchRootNode, default, 0, true);
      //        CeresEngine.Manager.Engine.SearchRootNode.Dump(1, 1, prefixString: "info string ");
    }
  }

  public readonly record struct BackendbenchOptions(int StartIndex = 1, int EndIndex = 1024, int StepSize = 8, int RepeatCount = 4);

  public static class BackendbenchOptionsParser
  {
    public static BackendbenchOptions Parse(string input)
    {
      string[] parts = input.Split(' ', StringSplitOptions.RemoveEmptyEntries);
      if (parts.Length == 0)
      {
        throw new ArgumentException("Input cannot be empty.", nameof(input));
      }

      if (!string.Equals(parts[0], "backendbench", StringComparison.OrdinalIgnoreCase))
      {
        throw new FormatException("Command must start with 'backendbench'.");
      }

      int defaultStartIndex = 1;
      int defaultEndIndex = 1024;
      int defaultStepSize = 8;
      int defaultRepeatCount = 4;

      int index = 1;
      while (index < parts.Length)
      {
        string token = parts[index];

        if (string.Equals(token, "from", StringComparison.OrdinalIgnoreCase))
        {
          index++;
          if (index >= parts.Length)
          {
            throw new FormatException("Expected integer after 'from'.");
          }

          defaultStartIndex = ParseInt(parts[index], "from");
          index++;
        }
        else if (string.Equals(token, "to", StringComparison.OrdinalIgnoreCase))
        {
          index++;
          if (index >= parts.Length)
          {
            throw new FormatException("Expected integer after 'to'.");
          }

          defaultEndIndex = ParseInt(parts[index], "to");
          index++;
        }
        else if (string.Equals(token, "by", StringComparison.OrdinalIgnoreCase))
        {
          index++;
          if (index >= parts.Length)
          {
            throw new FormatException("Expected integer after 'by'.");
          }

          defaultStepSize = ParseInt(parts[index], "by");
          index++;
        }
        else if (string.Equals(token, "repeat", StringComparison.OrdinalIgnoreCase))
        {
          index++;
          if (index >= parts.Length)
          {
            throw new FormatException("Expected integer after 'by'.");
          }

          defaultRepeatCount = ParseInt(parts[index], "repeat");
          index++;
        }
        else
        {
          throw new FormatException("Unexpected token '" + token + "'.");
        }
      }

      return new BackendbenchOptions(defaultStartIndex, defaultEndIndex, defaultStepSize, defaultRepeatCount);
    }

    private static int ParseInt(string token, string context)
    {
      int value;
      if (!int.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
      {
        throw new FormatException("Invalid integer '" + token + "' after '" + context + "'.");
      }

      return value;
    }
  }

}
