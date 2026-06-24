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
using System.Text;
using System.Threading.Tasks;

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.GameEngines;
using Ceres.Chess.Games.Utils;
using Ceres.Chess.LC0.Engine;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNFiles;
using Ceres.Features.GameEngines;
using Ceres.Chess.Positions;
using Ceres.Base.Misc;
using Ceres.Base.Benchmarking;
using System.Collections.Concurrent;
using Ceres.Base.Math;
using Ceres.Chess.NNEvaluators.Internals;
using System.Linq;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Search.Params;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.NNEvaluators.Defs;

#endregion

namespace Ceres.Features.Suites
{
  public class SuiteTestRunner
  {
    /// <summary>
    /// Definition of tournament used.
    /// </summary>
    public SuiteTestDef Def;

    /// <summary>
    /// Number of suite positions to evaluate concurrently.
    /// </summary>
    public int NumConcurrent;

    /// <summary>
    /// Optionally a flat sequence of GPU device IDs over which the concurrent suite
    /// workers are allocated. The IDs are partitioned into consecutive chunks (one chunk
    /// per worker), where the chunk size equals the number of devices in the engine
    /// specification. When null the engines are used exactly as specified
    /// (no per-worker device assignment).
    /// </summary>
    public int[] DeviceIDs;

    /// <summary>
    /// Pool of (paired) Ceres engine instances available to the concurrent suite workers.
    /// Each entry holds the first Ceres engine and (optionally) the second Ceres engine
    /// for one worker; the two engines in a pair share the same assigned device(s), so a
    /// worker always runs both engines on the same GPU(s).
    /// </summary>
    public ConcurrentBag<(GameEngine Engine1, GameEngine Engine2)> EngineSets = new();

    /// <summary>
    /// Engine for optional external UCI engine.
    /// </summary>
    public GameEngine EngineExternal { get; private set; }

    /// <summary>
    /// When Run is invoked with useMultiEngineMode (or the definition was already in multiengine
    /// mode), holds the comprehensive multiengine result produced by the run (Run itself returns
    /// null in that case, since its return type is the two-engine SuiteTestResult).
    /// </summary>
    public MultiEngineSuiteResult MultiEngineResult { get; private set; }

    /// <summary>
    /// Pool of engine instance arrays for multiengine mode (one array per concurrent worker,
    /// with one engine instance per engine in the multiengine definition). The Ceres engines
    /// within a worker's array all share that worker's assigned device(s), and for any position
    /// a worker runs all its engines sequentially on those device(s) (preserving the fair-timing
    /// property of the two-engine mode). Used only when Def.IsMultiEngine.
    /// </summary>
    public ConcurrentBag<GameEngine[]> EngineArrays = new();


    int numConcurrentSuiteThreads;


    /// <summary>
    /// Creates and warms up a single engine instance for a worker, optionally rewriting
    /// the device indices (of an isolated clone of the engine definition) so the worker
    /// runs on a specific set of GPUs. The engine is warmed up using the supplied search
    /// limit (which may be null, in which case a default warmup is performed).
    /// </summary>
    static GameEngine CreateWorkerEngine(GameEngineDef engineDef, SearchLimit warmupLimit, int[] deviceIDsForWorker)
    {
      if (engineDef == null)
      {
        return null;
      }

      GameEngineDef defForWorker = engineDef;
      if (deviceIDsForWorker != null)
      {
        // Clone so the device rewrite is isolated to this worker
        // (binary deep clone, the same mechanism used by TournamentDef.Clone).
        defForWorker = ObjUtils.DeepClone(engineDef);
        defForWorker.TrySetDeviceIndicesIfNotPooled(deviceIDsForWorker);
      }

      GameEngine engine = defForWorker.CreateEngine();

      // Enable per-move (root) visit statistics so the policy-difference (KLD) metric can be
      // computed. MCGS only populates GameEngineSearchResult.VerboseMoveStats when this flag is
      // set; it is a cheap post-search root walk and does not change search behavior or output.
      if (engine is GameEngineCeresMCGSInProcess mcgsEngine)
      {
        mcgsEngine.GatherVerboseMoveStats = true;
      }

      engine.Warmup(warmupLimit?.KnownMaxNumNodes);
      return engine;
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="def"></param>
    public SuiteTestRunner(SuiteTestDef def)
    {
      Def = def;
    }


    /// <summary>
    /// Constructor specifying a level of concurrency and (optionally) a flat set of GPU
    /// device IDs over which the concurrent suite workers are spread.
    ///
    /// The deviceIDs are treated as a flat pool partitioned into consecutive chunks (one
    /// chunk per concurrent worker), where the chunk size equals the number of devices in
    /// the engine's device specification. For example, with a "GPU:0,1" specification,
    /// numConcurrent 2 and deviceIDs [0,1,2,3], one worker runs on GPUs [0,1] and the
    /// other on GPUs [2,3]. (Modeled after TournamentManager.)
    /// </summary>
    /// <param name="def"></param>
    /// <param name="numConcurrent"></param>
    /// <param name="deviceIDs"></param>
    public SuiteTestRunner(SuiteTestDef def, int numConcurrent, int[] deviceIDs = null)
    {
      if (numConcurrent > 1 && deviceIDs == null)
      {
        throw new Exception("Must specify deviceIDs if numConcurrent > 1.");
      }

      Def = def;
      NumConcurrent = numConcurrent;
      DeviceIDs = deviceIDs;
    }


    /// <summary>
    /// Returns the set of (absolute) GPU device IDs assigned to a given concurrent worker,
    /// or null if no explicit device assignment is in effect.
    /// </summary>
    int[] DeviceSliceForWorker(int workerIndex, int numDevicesPerWorker)
    {
      if (DeviceIDs == null)
      {
        return null;
      }

      // Each worker gets its own disjoint chunk of the device pool (no wrap-around), so that
      // concurrent workers never share a GPU. This is validated in ComputeAndValidateDevicePartitioning.
      int start = workerIndex * numDevicesPerWorker;
      int[] ret = new int[numDevicesPerWorker];
      Array.Copy(DeviceIDs, start, ret, 0, numDevicesPerWorker);
      return ret;
    }


    void Init(int numDevicesPerWorker)
    {
      // Create and warm up the engine pairs (one pair per concurrent worker).
      for (int i = 0; i < numConcurrentSuiteThreads; i++)
      {
        int[] deviceIDsForWorker = DeviceSliceForWorker(i, numDevicesPerWorker);
        GameEngine engine1 = CreateWorkerEngine(Def.Engine1Def, Def.CeresEngine1Def?.SearchLimit, deviceIDsForWorker);
        GameEngine engine2 = CreateWorkerEngine(Def.Engine2Def, Def.CeresEngine2Def?.SearchLimit, deviceIDsForWorker);
        EngineSets.Add((engine1, engine2));
      }

      Def.ExternalEngineDef?.EngineDef.CreateEngine(); EngineExternal?.Warmup(Def.ExternalEngineDef.SearchLimit.KnownMaxNumNodes);
    }


    /// <summary>
    /// Validates that the supplied DeviceIDs pool can be evenly partitioned into one chunk
    /// per concurrent worker, and returns the number of devices per worker (the number of
    /// devices in the engine specification).
    /// </summary>
    int ComputeAndValidateDevicePartitioning()
    {
      Chess.NNEvaluators.Defs.NNEvaluatorDef evalDef1 = Def.Engine1Def.GetEvaluatorDef();
      if (evalDef1 == null)
      {
        throw new Exception("DeviceIDs were specified but the first engine is not a Ceres engine with an evaluator.");
      }

      int numDevicesPerWorker = evalDef1.NumDevices;
      if (numDevicesPerWorker < 1)
      {
        throw new Exception("Engine specification must reference at least one device.");
      }

      Chess.NNEvaluators.Defs.NNEvaluatorDef evalDef2 = Def.Engine2Def?.GetEvaluatorDef();
      if (evalDef2 != null && evalDef2.NumDevices != numDevicesPerWorker)
      {
        throw new Exception($"Both Ceres engines must reference the same number of devices when using DeviceIDs "
                          + $"(engine1 has {numDevicesPerWorker}, engine2 has {evalDef2.NumDevices}).");
      }

      // Require enough device IDs that every concurrent worker is assigned its OWN disjoint
      // set of GPUs (no sharing across workers). This is essential for fairness: for any
      // position the two engines run sequentially on the worker's GPU(s), so if a worker had
      // exclusive use of its GPU(s) the measured per-engine execution times are directly
      // comparable; sharing GPUs between concurrent workers would distort those timings.
      int numWorkers = NumConcurrent;
      if (DeviceIDs.Length < numWorkers * numDevicesPerWorker)
      {
        throw new Exception($"Insufficient DeviceIDs ({DeviceIDs.Length}): need at least "
                          + $"NumConcurrent ({numWorkers}) * devices-per-engine ({numDevicesPerWorker}) "
                          + $"= {numWorkers * numDevicesPerWorker}, so each concurrent worker has its own GPU(s).");
      }

      return numDevicesPerWorker;
    }


    int numSearches = 0;
    int numSearchesBothFound = 0;
    int accCeres1 = 0, accCeres2 = 0, accWCeres1 = 0, accWCeres2 = 0, avgOther = 0;
    int sumCeres1NumNodesWhenChoseTopNode, sumCeres2NumNodesWhenChoseTopNode;

    List<float> solvedPct1MinusPct2Samples = new();

    // Per-position graded-score difference (engine 1 minus engine 2), in 0-10 points.
    // Accumulated only when a second engine is present; used to compute the headline
    // solution-quality z-score (a paired-difference significance test over the suite).
    List<float> scoreDiffSamples = new();

    float totalTimeOther = 0;
    float totalTimeCeres1 = 0;
    float totalTimeCeres2 = 0;

    long totalNodesOther = 0;
    long totalNodes1 = 0;
    long totalNodes2 = 0;

    int sumEvalNumPosOther;
    int sumEvalNumBatches1;
    int sumEvalNumPos1;
    int sumEvalNumBatches2;
    int sumEvalNumPos2;

    long sumTablebaseHits1;
    long sumTablebaseHits2;

    // Correct-move-visit counters: over positions where both engines exposed root visit stats,
    // how often each engine put a (meaningfully, by > 3 percentage points) larger fraction of
    // its final visits on the correct move(s), and how often the two were within 3 points.
    int countEngine1MoreCorrectVisits;
    int countEngine2MoreCorrectVisits;
    int countCorrectVisitsEqual;
    int countCorrectVisitsCompared;

    // Accumulates per-column statistics so a summary row (average of each column) can be
    // emitted at end of suite, and used to populate the comprehensive SuiteTestResult.
    readonly ColumnAccumulator columnAcc = new();

    // Whether the per-position detail table is being printed (governs the summary row).
    bool outputDetailRun;

    // Policy difference: symmetric KLD of the two engines' root visit distributions, summed
    // over positions where both engines exposed per-move visit statistics.
    double sumPolicyKLD;
    int countPolicyKLD;

    // Per-engine evaluations-per-second (EPS), accumulated from each search result.
    long sumEPS1;
    long sumEPS2;

    // Per-engine device backend ("in C++ interop") busy seconds and the matching search-loop
    // elapsed seconds (denominator), accumulated only over moves where the metric is supported.
    double sumBackendWait1;
    double sumBackendWait2;
    double sumBackendSearch1;
    double sumBackendSearch2;

    // Number of positions in the test set actually run (after filtering/slicing).
    int numPositionsInRun;


#if NOT
    void DumpParams(TextWriter writer, bool differentOnly)
    {
      // Consider instead emulating/consolidating code in used by TournamentDef dumping
      ParamsDump.DumpParams(writer, differentOnly,
                  null, null,
                  Def.Engine1Def.GetEvaluatorDef(), Def.Engine2Def?.GetEvaluatorDef(),
                  Def.CeresEngine1Def.SearchLimit, Def.CeresEngine2Def?.SearchLimit,
                  null, null,
                  null, null,
                  null, null);
    }
#endif


    /// <summary>
    /// If both Ceres engines are in-process MCGS engines, writes to the output any
    /// differences in their ParamsSearch (including the nested ParamsSearchExecution)
    /// and ParamsSelect. Nothing is written if the engines are not both MCGS engines
    /// or if their parameters are identical.
    /// </summary>
    void DumpMCGSEngineParamsDifferences()
    {
      if (!Def.RunCeres2Engine)
      {
        return;
      }

      // The engine instances are created during Init (which has already run when this is
      // called); peek at one warmed-up worker pair to inspect the effective parameters.
      if (!EngineSets.TryPeek(out var engineSet))
      {
        return;
      }

      if (engineSet.Engine1 is not GameEngineCeresMCGSInProcess mcgs1
       || engineSet.Engine2 is not GameEngineCeresMCGSInProcess mcgs2)
      {
        return;
      }

      string searchDiff = ObjUtils.FieldValuesDumpString<ParamsSearch>(mcgs1.SearchParams, mcgs2.SearchParams, true);
      string executionDiff = ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(mcgs1.SearchParams.Execution, mcgs2.SearchParams.Execution, true);
      string selectDiff = ObjUtils.FieldValuesDumpString<ParamsSelect>(mcgs1.SelectParams, mcgs2.SelectParams, true);

      // FieldValuesDumpString returns null for a section when there are no differences.
      if (searchDiff == null && executionDiff == null && selectDiff == null)
      {
        return;
      }

      Def.Output.WriteLine();
      Def.Output.WriteLine("MCGS1 (C1) vs MCGS2 (C2) parameter differences");
      if (searchDiff != null)
      {
        Def.Output.WriteLine(searchDiff);
      }
      if (executionDiff != null)
      {
        Def.Output.WriteLine(executionDiff);
      }
      if (selectDiff != null)
      {
        Def.Output.WriteLine(selectDiff);
      }
      Def.Output.WriteLine();
    }


    /// <summary>
    /// Runs the suite test.
    ///
    /// When useMultiEngineMode is true a legacy (one/two/three engine) definition is converted
    /// on the fly into multiengine mode (with the first Ceres engine as the baseline) and run via
    /// the multiengine path — i.e. the single in-place-refreshed statistics block instead of the
    /// per-position detail rows. In that case (or when the definition was already in multiengine
    /// mode) the comprehensive result is exposed via the MultiEngineResult property and this
    /// method returns null.
    /// </summary>
    public SuiteTestResult Run(int numConcurrentSuiteThreads = 1,
                               bool outputDetail = true,
                               bool saveCacheWhenDone = true,
                               bool enableCancelVialCtrlC = true,
                               bool useMultiEngineMode = false)
    {
      // Optionally convert a legacy definition into multiengine mode (Engine1 = baseline).
      if (useMultiEngineMode && !Def.IsMultiEngine)
      {
        Def.ConfigureMultiEngineFromLegacy();
      }

      if (Def.IsMultiEngine)
      {
        MultiEngineResult = RunMultiEngine(numConcurrentSuiteThreads, outputDetail, enableCancelVialCtrlC);
        return null;
      }

      outputDetailRun = outputDetail;

      // Tree reuse is no help, indicate that we won't need it
      Def.Engine1Def.DisableTreeReuse();
      Def.Engine2Def?.DisableTreeReuse();

      // Disable dump for now, the execution parameters are modified
      // for the warmup which is confusing because different parameters
      // will be chosen for the actual search.
      //DumpParams(Def.Output, true);

      // If a device pool was supplied (via the concurrency constructor) then it governs
      // the number of concurrent workers, and each worker is assigned a distinct set of GPUs.
      int numDevicesPerWorker = 0;
      if (DeviceIDs != null)
      {
        numConcurrentSuiteThreads = NumConcurrent;
        numDevicesPerWorker = ComputeAndValidateDevicePartitioning();
      }

      this.numConcurrentSuiteThreads = numConcurrentSuiteThreads;

      Init(numDevicesPerWorker);

      // Install Ctrl-C handler to allow ad hoc clean termination (with stats).
      bool stopRequested = false;
      if (enableCancelVialCtrlC)
      {
        ConsoleCancelEventHandler ctrlCHandler = new ConsoleCancelEventHandler((object sender, ConsoleCancelEventArgs args) =>
        {
          Console.WriteLine("Suite pending shutdown....");
          stopRequested = true;
          args.Cancel = true;
        });
        Console.CancelKeyPress += ctrlCHandler;
      }

      int timerFiredCount = 0;

      // TODO: add path automatically
      List<EPDEntry> epds = EPDEntry.EPDEntriesInEPDFile(Def.EPDFileName, Def.MaxNumPositions,
                                                         Def.EPDLichessPuzzleFormat,
                                                         Def.EPDRawLineFilter, Def.EPDFilter);

      // Possibly skip some of positions at beginning of file.
      if (Def.SkipNumPositions > 0)
      {
        if (Def.SkipNumPositions <= epds.Count)
        {
          throw new Exception("Insufficient positions in " + Def.EPDFileName + " to skip " + Def.SkipNumPositions);
        }
        epds = epds.GetRange(Def.SkipNumPositions, epds.Count - Def.SkipNumPositions);
      }

      // Position filter EPDs based on the starting position.
      if (Def.AcceptPosPredicate != null)
      {
        epds = epds.Where(s => Def.AcceptPosPredicate(s.Position)).ToList();
      }

      if (Def.MaxNumPositions == 0)
      {
        Def.MaxNumPositions = epds.Count;
      }

      Def.Output.WriteLine();
      Def.Output.WriteLine("C1 = " + Def.CeresEngine1Def.EngineDef);
      if (Def.RunCeres2Engine)
      {
        Def.Output.WriteLine("C2 = " + Def.CeresEngine2Def.EngineDef);
      }

      if (Def.ExternalEngineDef != null)
      {
        Def.Output.WriteLine("EX = " + Def.ExternalEngineDef.EngineDef);
      }

      // If both engines are in-process MCGS engines, dump any differences in their
      // search/selection parameters as part of the header (so A/B parameter comparisons
      // are self-documenting in the suite output).
      DumpMCGSEngineParamsDifferences();

#if NOT
      // To make up for the fact that LZ0 "cheats" by sometimes running over specified number of nodes
      // (she seems to always fill the batch even if reached limit), add half a batch extra for Ceres as compensation
      if (searchLimitCeres1.Type == SearchLimit.LimitType.NodesPerMove)
      {
        searchLimitCeres1 = new SearchLimit(searchLimit.Type, searchLimit.Value + paramsSearch1.BATCH_SIZE_PRIMARY / 2);
        searchLimitCeres2 = new SearchLimit(searchLimit.Type, searchLimit.Value + paramsSearch2.BATCH_SIZE_PRIMARY / 2);
      }
#endif

      //Def.Output.WriteLine($"MAX_CERES_GAME_THREADS {numConcurrentCeresGames} MAX_LEELA_GAME_THREADS {MAX_LEELA_GAME_THREADS}");

      // Turn off position reuse if evaluators produce different results
      if (Def.RunCeres2Engine)
      {
        Chess.NNEvaluators.Defs.NNEvaluatorDef evalDef1 = Def.Engine1Def.GetEvaluatorDef();
        Chess.NNEvaluators.Defs.NNEvaluatorDef evalDef2 = Def.Engine2Def.GetEvaluatorDef();
        if (evalDef1 != null && evalDef2 != null && !evalDef1.NetEvaluationsIdentical(evalDef2))
        {
          Def.Engine1Def.SetReusePositionEvaluationsFromOther(false);
          Def.Engine2Def.SetReusePositionEvaluationsFromOther(false);
        }
      }

      if (Def.RunCeres2Engine && (Def.Engine1Def.GetReusePositionEvaluationsFromOther() ||
          Def.Engine2Def.GetReusePositionEvaluationsFromOther()))
      {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\r\nWARNING: REUSE_POSITION_EVALUATIONS_FROM_OTHER_TREE is turned on for one or both evaluators\r\n"
                         + "(alternating between the two evaluators). This may cause slight differences in search behavior and speed.\r\n");
        Console.ForegroundColor = ConsoleColor.White;
      }

      Def.Output.WriteLine();

      if (Def.MaxNumPositions > epds.Count) Def.MaxNumPositions = epds.Count;
      epds = epds.GetRange(Def.FirstTestPosition, Def.MaxNumPositions);

      numPositionsInRun = epds.Count;

      int numExternalGameProcesses = 1;

      numConcurrentSuiteThreads = Math.Min(Def.MaxNumPositions, numConcurrentSuiteThreads);

      if (numConcurrentSuiteThreads > 1)
      {
        if (Def.ExternalEngineDef != null)
        {
          // For safety (to not overflow main or GPU memory) we limit number of LC0 processes.
          const int MAX_LC0_PROCESSES = 4;
          numExternalGameProcesses = Math.Min(MAX_LC0_PROCESSES, numConcurrentSuiteThreads);
        }
      }

      bool leelaVerboseMovesStats = true;//xxx Def.NumTestPos == 1;
      Func<int, object> makeExternalEngine = null;

      // TODO: eventually create clones of engine definitions
      //       then set each one to a different processorGroupID
      const int processorGroupID = 0;

      if (Def.ExternalEngineDef != null)
      {
        if (Def.ExternalEngineDef.EngineDef is GameEngineDefLC0)
        {
          GameEngineDefLC0 lc0EngineDef = Def.ExternalEngineDef.EngineDef as GameEngineDefLC0;
          bool forceDisableSmartPruning = lc0EngineDef.ForceDisableSmartPruning;
          const bool FILL_HISTORY = true;
          NNEvaluatorDef engine1EvalDef = lc0EngineDef.EvaluatorDef;
          makeExternalEngine = (int processorGroupID) =>
          {
            LC0Engine engine = LC0EngineConfigured.GetLC0Engine(null, null, engine1EvalDef,
                                                                NNWeightsFiles.LookupNetworkFile(engine1EvalDef.Nets[0].Net.NetworkID),
                                                                processorGroupID, true,
                                                                false, leelaVerboseMovesStats, forceDisableSmartPruning,
                                                                lc0EngineDef.OverrideEXE, FILL_HISTORY, lc0EngineDef.ExtraCommandLineArgs);
            // WARMUP
            engine.AnalyzePositionFromFEN(Position.StartPosition.FEN, SearchLimit.NodesPerMove(1));
            return engine;
          };
        }
        else
        {
          makeExternalEngine = delegate (int processorGroupID)
          {
            GameEngine engine = Def.ExternalEngineDef.EngineDef.CreateEngine();
            engine.Warmup(Def.ExternalEngineDef.SearchLimit.KnownMaxNumNodes);
            return engine;
          };
        }
      }

      // Don't create too many non_Ceres threads since each one will consume separate GPU memory or threads.
      int maxLeelaThreads = Math.Min(numExternalGameProcesses, numConcurrentSuiteThreads);
      ObjectPool<object> externalEnginePool = new ObjectPool<object>(() => makeExternalEngine(processorGroupID), maxLeelaThreads);

      using (new TimingBlock("EPDS", Def.Output == Console.Out ? TimingBlock.LoggingType.Console : TimingBlock.LoggingType.None))
      {
        Parallel.For(0, epds.Count,
                     new ParallelOptions() { MaxDegreeOfParallelism = numConcurrentSuiteThreads },
                     delegate (int gameNum)
                     {
                       if (stopRequested)
                       {
                         return;
                       }

                       try
                       {
                         EPDEntry epd = epds[gameNum];

                         // Skip positions which are already draws
                         if (epd.Position.CheckDrawBasedOnMaterial == Position.PositionDrawStatus.DrawByInsufficientMaterial)
                         {
                           return;
                         }
                         // TODO: also do this for checkmate?

                         ProcessEPD(gameNum, epds[gameNum], outputDetail, externalEnginePool);

                       }
                       catch (Exception exc)
                       {
                         Def.Output.WriteLine("Error in ProcessEPD " + exc);
                         throw;
                       }
                     });
      }

      WriteSummaries();

      Shutdown(externalEnginePool);

      // Mean of a per-position column (from the accumulator), coalescing NaN (absent column) to 0.
      float ColAvg(string id)
      {
        double v = columnAcc.Average(id);
        return double.IsNaN(v) ? 0.0f : (float)v;
      }

      float avgEPS1 = numSearches > 0 ? (float)sumEPS1 / numSearches : 0;
      float avgEPS2 = numSearches > 0 ? (float)sumEPS2 / numSearches : 0;

      // Graded solution-quality difference (engine 1 minus engine 2) and its significance.
      float[] scoreDiffArr = scoreDiffSamples.ToArray();
      float scoreDiffMean = scoreDiffArr.Length > 0 ? (float)StatUtils.Average(scoreDiffArr) : 0;
      float scoreDiffSE = scoreDiffArr.Length > 1 ? (float)StatUtils.StdDev(scoreDiffArr) / MathF.Sqrt(scoreDiffArr.Length) : 0;

      // Nodes-to-solution speed difference (difference in fraction of nodes when top move chosen).
      float[] fasterArr = solvedPct1MinusPct2Samples.ToArray();
      float fasterMean = fasterArr.Length > 0 ? (float)StatUtils.Average(fasterArr) : 0;
      float fasterSE = fasterArr.Length > 1 ? (float)StatUtils.StdDev(fasterArr) / MathF.Sqrt(fasterArr.Length) : 0;

      return new SuiteTestResult(Def)
      {
        AvgScore1 = (float)accCeres1 / numSearches,
        AvgScore2 = (float)accCeres2 / numSearches,
        AvgWScore1 = (float)accWCeres1 / numSearches,
        AvgWScore2 = (float)accWCeres2 / numSearches,
        AvgScoreLC0 = (float)avgOther / numSearches,

        TotalRuntimeLC0 = totalTimeOther,
        TotalRuntime1 = totalTimeCeres1,
        TotalRuntime2 = totalTimeCeres2,

        TotalBackendWait1 = sumBackendWait1,
        TotalBackendWait2 = sumBackendWait2,
        TotalBackendSearch1 = sumBackendSearch1,
        TotalBackendSearch2 = sumBackendSearch2,

        FinalQ1 = finalQ1.ToArray(),
        FinalQ2 = finalQ2?.ToArray(),

        TotalNodesLC0 = totalNodesOther,
        TotalNodes1 = totalNodes1,
        TotalNodes2 = totalNodes2,

        AvgAbsQDifference = finalQ2.Count == 0 ? 0 : StatUtils.Average(StatUtils.AbsDiff(finalQ1.ToArray(), finalQ2.ToArray())),

        // Per-position averages (mirroring the console columns).
        AvgDepth1 = ColAvg("ADep"),
        AvgDepth2 = ColAvg("ADep2"),
        MaxDepth1 = ColAvg("MDep"),
        MaxDepth2 = ColAvg("MDep2"),
        VisitEntropy1 = ColAvg("VEnt"),
        VisitEntropy2 = ColAvg("VEnt2"),
        YieldFrac1 = ColAvg("Yld"),
        YieldFrac2 = ColAvg("Yld2"),
        CorrectMoveVisitFracPct1 = ColAvg("Fr"),
        CorrectMoveVisitFracPct2 = ColAvg("Fr2"),
        AvgQ1 = ColAvg("QC"),
        AvgQ2 = ColAvg("QC2"),
        AvgQLC0 = ColAvg("QEx"),

        // Totals (sums) over all positions.
        TotalNNBatches1 = sumEvalNumBatches1,
        TotalNNBatches2 = sumEvalNumBatches2,
        TotalNNEvals1 = sumEvalNumPos1,
        TotalNNEvals2 = sumEvalNumPos2,
        TotalTablebaseHits1 = sumTablebaseHits1,
        TotalTablebaseHits2 = sumTablebaseHits2,
        TotalNodesWhenChoseTopN1 = sumCeres1NumNodesWhenChoseTopNode,
        TotalNodesWhenChoseTopN2 = sumCeres2NumNodesWhenChoseTopNode,

        // Correct-move-visit comparison (positions where both engines exposed root visit stats).
        CountEngine1MoreCorrectVisits = countEngine1MoreCorrectVisits,
        CountEngine2MoreCorrectVisits = countEngine2MoreCorrectVisits,
        CountCorrectVisitsEqual = countCorrectVisitsEqual,
        CountCorrectVisitsCompared = countCorrectVisitsCompared,

        // Suite identity.
        ID = Def.ID,
        EPDFileName = Def.EPDFileName,
        NumPositionsTested = numPositionsInRun,
        SearchLimit1 = Def.CeresEngine1Def?.SearchLimit,
        SearchLimit2 = Def.CeresEngine2Def?.SearchLimit,
        MachineName = Environment.MachineName,
        RunDateTime = DateTime.Now,

        // Performance summary metrics.
        MeanPolicyKLD = countPolicyKLD > 0 ? (float)(sumPolicyKLD / countPolicyKLD) : float.NaN,
        CountPolicyKLDPositions = countPolicyKLD,
        AvgEPS1 = avgEPS1,
        AvgEPS2 = avgEPS2,
        RelativeEPSPct = avgEPS2 > 0 ? (avgEPS1 / avgEPS2 - 1.0f) * 100.0f : 0,

        // Graded solution-quality difference (the headline decision statistic).
        ScoreDiffMean = scoreDiffMean,
        ScoreDiffStdErr = scoreDiffSE,
        ScoreDiffNumSamples = scoreDiffArr.Length,
        ScoreDiffZ = scoreDiffSE > 0 ? scoreDiffMean / scoreDiffSE : 0,

        // Nodes-to-solution speed difference (negative favors engine 1).
        NodesToFindFasterMean = fasterMean,
        NodesToFindFasterStdErr = fasterSE,
        NodesToFindFasterZ = fasterSE > 0 ? fasterMean / fasterSE : 0
      };

    }

    #region Multiengine mode

    // Index (into Def.MultiEngineDefs) of the baseline engine in multiengine mode.
    int baselineIndexMulti;


    /// <summary>
    /// Runs the suite in "multiengine mode": an arbitrary number of engines (each a standard
    /// in-process Ceres engine or an external engine) are compared at once on the same set of
    /// positions. Results are shown in a single statistics block which is refreshed in place
    /// as the suite runs (best value per row green, worst red), and returned as a comprehensive
    /// MultiEngineSuiteResult. Requires a SuiteTestDef built with the multiengine constructor.
    ///
    /// As in the two-engine mode, each concurrent worker holds its own instance of every engine;
    /// for any position a worker runs all its engines sequentially on its own GPU(s) (so the
    /// measured per-engine timings/EPS remain directly comparable). Multi-GPU spreading via the
    /// SuiteTestRunner(def, numConcurrent, deviceIDs) constructor is supported.
    /// </summary>
    public MultiEngineSuiteResult RunMultiEngine(int numConcurrentSuiteThreads = 1,
                                                 bool outputDetail = true,
                                                 bool enableCancelViaCtrlC = true)
    {
      if (!Def.IsMultiEngine)
      {
        throw new Exception("RunMultiEngine requires a SuiteTestDef constructed with the multiengine constructor.");
      }

      List<MultiEngineEntry> entries = Def.MultiEngineDefs;
      baselineIndexMulti = entries.FindIndex(e => e.IsBaseline);
      if (baselineIndexMulti < 0)
      {
        baselineIndexMulti = 0;
      }

      // Tree reuse is no help in suite testing.
      foreach (MultiEngineEntry e in entries)
      {
        e.PlayerDef.EngineDef.DisableTreeReuse();
      }

      // If a device pool was supplied (via the concurrency constructor) it governs the number
      // of concurrent workers, and each worker is assigned a distinct set of GPUs.
      int numDevicesPerWorker = 0;
      if (DeviceIDs != null)
      {
        numConcurrentSuiteThreads = NumConcurrent;
        numDevicesPerWorker = ComputeAndValidateDevicePartitioningMulti();
      }
      this.numConcurrentSuiteThreads = numConcurrentSuiteThreads;

      InitMultiEngine(numDevicesPerWorker);

      // Install Ctrl-C handler to allow ad hoc clean termination (with stats so far).
      bool stopRequested = false;
      if (enableCancelViaCtrlC)
      {
        ConsoleCancelEventHandler ctrlCHandler = new ConsoleCancelEventHandler((object sender, ConsoleCancelEventArgs args) =>
        {
          Console.WriteLine("Suite pending shutdown....");
          stopRequested = true;
          args.Cancel = true;
        });
        Console.CancelKeyPress += ctrlCHandler;
      }

      List<EPDEntry> epds = LoadAndSliceEPDsMulti();
      numPositionsInRun = epds.Count;

      this.numConcurrentSuiteThreads = Math.Min(Math.Max(1, epds.Count), this.numConcurrentSuiteThreads);

      WriteMultiEngineHeader();

      MultiEngineAccumulator acc = new MultiEngineAccumulator(entries, baselineIndexMulti);
      MultiEngineLiveDisplay display = new MultiEngineLiveDisplay(Def.Output, entries.ToArray(), epds.Count);

      using (new TimingBlock("EPDS", Def.Output == Console.Out ? TimingBlock.LoggingType.Console : TimingBlock.LoggingType.None))
      {
        Parallel.For(0, epds.Count,
                     new ParallelOptions() { MaxDegreeOfParallelism = this.numConcurrentSuiteThreads },
                     delegate (int gameNum)
                     {
                       if (stopRequested)
                       {
                         return;
                       }

                       try
                       {
                         EPDEntry epd = epds[gameNum];

                         // Skip positions which are already draws.
                         if (epd.Position.CheckDrawBasedOnMaterial == Position.PositionDrawStatus.DrawByInsufficientMaterial)
                         {
                           return;
                         }

                         ProcessEPDMultiEngine(gameNum, epd, acc, display);
                       }
                       catch (Exception exc)
                       {
                         Def.Output.WriteLine("Error in ProcessEPDMultiEngine " + exc);
                         throw;
                       }
                     });
      }

      // Final frozen statistics block.
      MultiEngineEngineResult[] snapshot = acc.Snapshot();
      display.RenderFinal(snapshot, acc.PositionsDone);

      if (outputDetail)
      {
        WriteMultiEngineDetailDump(snapshot);
      }

      ShutdownMultiEngine();

      return new MultiEngineSuiteResult(Def)
      {
        Engines = snapshot,
        BaselineIndex = baselineIndexMulti,
        ID = Def.ID,
        EPDFileName = Def.EPDFileName,
        NumPositionsTested = acc.PositionsDone,
        MachineName = Environment.MachineName,
        RunDateTime = DateTime.Now
      };
    }


    /// <summary>
    /// Loads, filters and slices the EPD test positions for a multiengine run
    /// (Lichess puzzle format is not currently supported in multiengine mode).
    /// </summary>
    List<EPDEntry> LoadAndSliceEPDsMulti()
    {
      if (Def.EPDLichessPuzzleFormat)
      {
        throw new NotSupportedException("Lichess puzzle format EPD files are not yet supported in multiengine mode.");
      }

      List<EPDEntry> epds = EPDEntry.EPDEntriesInEPDFile(Def.EPDFileName, Def.MaxNumPositions,
                                                         Def.EPDLichessPuzzleFormat,
                                                         Def.EPDRawLineFilter, Def.EPDFilter);

      if (Def.SkipNumPositions > 0)
      {
        if (Def.SkipNumPositions >= epds.Count)
        {
          throw new Exception("Insufficient positions in " + Def.EPDFileName + " to skip " + Def.SkipNumPositions);
        }
        epds = epds.GetRange(Def.SkipNumPositions, epds.Count - Def.SkipNumPositions);
      }

      if (Def.AcceptPosPredicate != null)
      {
        epds = epds.Where(s => Def.AcceptPosPredicate(s.Position)).ToList();
      }

      int first = Math.Min(Def.FirstTestPosition, epds.Count);
      int max = (Def.MaxNumPositions <= 0 || Def.MaxNumPositions == int.MaxValue)
              ? epds.Count - first
              : Math.Min(Def.MaxNumPositions, epds.Count - first);
      epds = epds.GetRange(first, max);
      return epds;
    }


    /// <summary>
    /// Validates that the supplied DeviceIDs pool can be partitioned into one chunk per
    /// concurrent worker (one chunk per worker, chunk size = number of devices referenced by
    /// the Ceres engines, which must all match), and returns the number of devices per worker.
    /// </summary>
    int ComputeAndValidateDevicePartitioningMulti()
    {
      int? numDevicesPerWorker = null;
      foreach (MultiEngineEntry e in Def.MultiEngineDefs)
      {
        Chess.NNEvaluators.Defs.NNEvaluatorDef evalDef = e.PlayerDef.EngineDef.GetEvaluatorDef();
        if (evalDef == null)
        {
          continue; // external engine: not assigned a GPU slice
        }

        int n = evalDef.NumDevices;
        if (n < 1)
        {
          throw new Exception($"Engine {e.ID} specification must reference at least one device.");
        }

        if (numDevicesPerWorker == null)
        {
          numDevicesPerWorker = n;
        }
        else if (n != numDevicesPerWorker.Value)
        {
          throw new Exception($"All Ceres engines must reference the same number of devices when using DeviceIDs "
                            + $"(engine {e.ID} has {n}, expected {numDevicesPerWorker.Value}).");
        }
      }

      if (numDevicesPerWorker == null)
      {
        throw new Exception("DeviceIDs were specified but none of the engines is a Ceres engine with an evaluator.");
      }

      int numWorkers = NumConcurrent;
      if (DeviceIDs.Length < numWorkers * numDevicesPerWorker.Value)
      {
        throw new Exception($"Insufficient DeviceIDs ({DeviceIDs.Length}): need at least "
                          + $"NumConcurrent ({numWorkers}) * devices-per-engine ({numDevicesPerWorker.Value}) "
                          + $"= {numWorkers * numDevicesPerWorker.Value}, so each concurrent worker has its own GPU(s).");
      }

      return numDevicesPerWorker.Value;
    }


    /// <summary>
    /// Creates and warms up one array of engine instances per concurrent worker (one instance
    /// per engine in the multiengine definition). Ceres engines are cloned onto the worker's
    /// assigned device slice; external engines are created as-is (each worker gets its own).
    /// </summary>
    void InitMultiEngine(int numDevicesPerWorker)
    {
      List<MultiEngineEntry> entries = Def.MultiEngineDefs;
      for (int i = 0; i < numConcurrentSuiteThreads; i++)
      {
        int[] deviceIDsForWorker = DeviceSliceForWorker(i, numDevicesPerWorker);
        GameEngine[] arr = new GameEngine[entries.Count];
        for (int k = 0; k < entries.Count; k++)
        {
          MultiEngineEntry e = entries[k];
          int[] slice = e.IsCeresEngine ? deviceIDsForWorker : null;
          arr[k] = CreateWorkerEngine(e.PlayerDef.EngineDef, e.Limit, slice);
        }
        EngineArrays.Add(arr);
      }
    }


    /// <summary>
    /// Evaluates one position with every engine (sequentially, on the worker's own device(s)),
    /// merges the results into the accumulator and refreshes the live statistics block.
    /// </summary>
    void ProcessEPDMultiEngine(int epdNum, EPDEntry epd, MultiEngineAccumulator acc, MultiEngineLiveDisplay display)
    {
      if (!EngineArrays.TryTake(out GameEngine[] engines))
      {
        throw new Exception("No engine array available");
      }

      int n = engines.Length;
      GameEngineSearchResult[] results = new GameEngineSearchResult[n];
      PositionWithHistory pos = epd.PosWithHistory;

      try
      {
        // Rotate the starting engine each position to avoid any systematic ordering effect.
        int start = epdNum % n;
        for (int j = 0; j < n; j++)
        {
          int k = (start + j) % n;
          GameEngine eng = engines[k];
          eng.ResetGame();
          SearchLimit limit = Def.MultiEngineDefs[k].Limit.ConvertedGameToMoveLimit;
          results[k] = eng.Search(pos, limit);
        }
      }
      finally
      {
        // Restore the array to the pool (so this worker keeps its assigned device(s)).
        EngineArrays.Add(engines);
      }

      lock (lockObj)
      {
        acc.AddPosition(results, epd);
        MultiEngineEngineResult[] snapshot = acc.Snapshot();
        display.Refresh(snapshot, acc.PositionsDone);
      }
    }


    /// <summary>
    /// Writes the multiengine header block (suite identity and the participating engines), and
    /// any MCGS search/selection parameter differences of each engine versus the baseline.
    /// </summary>
    void WriteMultiEngineHeader()
    {
      Def.Output.WriteLine();
      Def.Output.WriteLine("MULTIENGINE SUITE TEST: " + Def.ID);
      Def.Output.WriteLine("  Machine      : " + Environment.MachineName);
      Def.Output.WriteLine("  Date/Time    : " + DateTime.Now);
      Def.Output.WriteLine("  EPD file     : " + Def.EPDFileName);
      Def.Output.WriteLine("  Positions    : " + numPositionsInRun);
      Def.Output.WriteLine("  Concurrency  : " + numConcurrentSuiteThreads
                         + (DeviceIDs != null ? "   Devices [" + string.Join(",", DeviceIDs) + "]" : ""));
      Def.Output.WriteLine();

      foreach (MultiEngineEntry e in Def.MultiEngineDefs)
      {
        string kind = !e.IsCeresEngine ? "external"
                    : (e.PlayerDef.EngineDef is GameEngineDefCeresMCGS ? "MCGS" : "MCTS");
        Def.Output.WriteLine($"  {e.ID,-12}{(e.IsBaseline ? " *BASELINE" : "         ")}  [{kind,-8}]  limit={e.Limit}   {e.PlayerDef.EngineDef}");
      }

      DumpMultiEngineParamDifferences();

      Def.Output.WriteLine();
    }


    /// <summary>
    /// For each in-process MCGS engine, writes any differences in its ParamsSearch / ParamsSelect
    /// versus the baseline engine (when the baseline is also an MCGS engine). Nothing is written
    /// when the engines are not MCGS or their parameters are identical.
    /// </summary>
    void DumpMultiEngineParamDifferences()
    {
      if (!EngineArrays.TryPeek(out GameEngine[] arr))
      {
        return;
      }

      if (arr[baselineIndexMulti] is not GameEngineCeresMCGSInProcess baseMCGS)
      {
        return;
      }

      for (int k = 0; k < arr.Length; k++)
      {
        if (k == baselineIndexMulti || arr[k] is not GameEngineCeresMCGSInProcess mcgs)
        {
          continue;
        }

        string searchDiff = ObjUtils.FieldValuesDumpString<ParamsSearch>(mcgs.SearchParams, baseMCGS.SearchParams, true);
        string selectDiff = ObjUtils.FieldValuesDumpString<ParamsSelect>(mcgs.SelectParams, baseMCGS.SelectParams, true);
        if (searchDiff == null && selectDiff == null)
        {
          continue;
        }

        Def.Output.WriteLine();
        Def.Output.WriteLine($"  {Def.MultiEngineDefs[k].ID} vs baseline {Def.MultiEngineDefs[baselineIndexMulti].ID} parameter differences:");
        if (searchDiff != null)
        {
          Def.Output.WriteLine(searchDiff);
        }
        if (selectDiff != null)
        {
          Def.Output.WriteLine(selectDiff);
        }
      }
    }


    /// <summary>
    /// Writes a per-engine detail recap (all available aggregate statistics) at end of run,
    /// after the live statistics block has been frozen.
    /// </summary>
    void WriteMultiEngineDetailDump(MultiEngineEngineResult[] engines)
    {
      Def.Output.WriteLine();
      Def.Output.WriteLine("PER-ENGINE DETAIL");
      foreach (MultiEngineEngineResult r in engines)
      {
        string Fmt(float v, string fmt) => float.IsNaN(v) ? "n/a" : v.ToString(fmt);
        Def.Output.WriteLine($"  {r.ID}{(r.IsBaseline ? "*" : "")} [{r.Kind}]  positions={r.NumPositions}");
        Def.Output.WriteLine($"      solve={Fmt(r.AvgSolveScorePct, "F1")}%  EPS={Fmt(r.AvgEPS, "N0")}  Q={Fmt(r.AvgQ, "F3")}  "
                           + $"|dQ|vsBase={Fmt(r.AvgAbsQDiffVsBaseline, "F3")}  KLDvsBase={Fmt(r.AvgPolicyKLDVsBaseline, "F4")}  "
                           + $"correctVisits={Fmt(r.AvgCorrectMoveVisitPct, "F1")}%");
        Def.Output.WriteLine($"      nodes={r.TotalNodes:N0}  time={r.TotalTimeSecs:F1}s  NNevals={r.TotalNNEvals:N0}  "
                           + $"TBhits={r.TotalTablebaseHits:N0}  avgDepth={Fmt(r.AvgDepth, "F1")}  visitEntropy={Fmt(r.AvgVisitEntropy, "F2")}");
      }
      Def.Output.WriteLine();
    }


    void ShutdownMultiEngine()
    {
      // Note: like the two-engine Shutdown, engine disposal is currently skipped (the underlying
      // dispose path has a known issue); we simply release the engine references.
      EngineArrays.Clear();
    }


    /// <summary>
    /// Accumulates per-engine aggregate statistics across all positions in a multiengine run,
    /// including the baseline-relative difference statistics. A position's full set of engine
    /// results is added atomically (caller holds the suite lock).
    /// </summary>
    internal sealed class MultiEngineAccumulator
    {
      readonly int n;
      readonly int baselineIndex;
      readonly MultiEngineEntry[] entries;
      readonly MultiEngineKind[] kinds;

      public int PositionsDone { get; private set; }
      public int BaselineIndex => baselineIndex;

      readonly double[] sumSolve; readonly int[] cntSolve;
      readonly double[] sumEPS;   readonly int[] cntEPS;
      readonly double[] sumQ;     readonly int[] cntQ;
      readonly double[] sumCV;    readonly int[] cntCV;
      readonly double[] sumQDiff; readonly int[] cntQDiff;
      readonly double[] sumKLD;   readonly int[] cntKLD;
      // Paired graded solution-quality difference vs the baseline (this engine minus baseline, 0-10
      // graded points). Running sum and sum-of-squares per engine let Snapshot() compute the
      // per-column z-score (mean / standard-error) without retaining the full per-position samples.
      readonly double[] sumScoreDiff; readonly double[] sumScoreDiffSq; readonly int[] cntScoreDiff;
      readonly long[] totNodes;
      readonly double[] totTime;
      readonly long[] totNNEvals;
      readonly long[] totTB;
      readonly double[] sumDepth; readonly int[] cntDepth;
      readonly double[] sumEnt;   readonly int[] cntEnt;
      readonly double[] totBackendWait;   // device backend busy seconds (supported moves only)
      readonly double[] totBackendSearch; // matching search-loop elapsed seconds (denominator)

      public MultiEngineAccumulator(List<MultiEngineEntry> entriesList, int baselineIndex)
      {
        entries = entriesList.ToArray();
        this.baselineIndex = baselineIndex;
        n = entries.Length;

        kinds = new MultiEngineKind[n];
        for (int k = 0; k < n; k++)
        {
          MultiEngineEntry e = entries[k];
          kinds[k] = !e.IsCeresEngine ? MultiEngineKind.External
                   : (e.PlayerDef.EngineDef is GameEngineDefCeresMCGS ? MultiEngineKind.CeresMCGS : MultiEngineKind.CeresMCTS);
        }

        sumSolve = new double[n]; cntSolve = new int[n];
        sumEPS = new double[n];   cntEPS = new int[n];
        sumQ = new double[n];     cntQ = new int[n];
        sumCV = new double[n];    cntCV = new int[n];
        sumQDiff = new double[n]; cntQDiff = new int[n];
        sumKLD = new double[n];   cntKLD = new int[n];
        sumScoreDiff = new double[n]; sumScoreDiffSq = new double[n]; cntScoreDiff = new int[n];
        totNodes = new long[n];
        totTime = new double[n];
        totNNEvals = new long[n];
        totTB = new long[n];
        sumDepth = new double[n]; cntDepth = new int[n];
        sumEnt = new double[n];   cntEnt = new int[n];
        totBackendWait = new double[n];
        totBackendSearch = new double[n];
      }

      public void AddPosition(GameEngineSearchResult[] results, EPDEntry epd)
      {
        PositionsDone++;

        Position position = epd.Position;
        GameEngineSearchResult baseR = (baselineIndex >= 0 && baselineIndex < results.Length) ? results[baselineIndex] : null;

        // Baseline's graded solve score (0..10) for this position, computed once so each engine's
        // paired solution-quality difference (engine minus baseline) can be accumulated below
        // regardless of engine ordering within the loop.
        int baselineScore = 0;
        bool haveBaselineScore = false;
        if (baseR != null)
        {
          Move baseBM = default;
          try
          {
            baseBM = Move.FromUCI(in position, baseR.MoveString);
          }
          catch
          {
            // Unparseable move (counts as not solved).
          }
          baselineScore = baseBM.IsNull ? 0 : epd.CorrectnessScore(baseBM, 10);
          haveBaselineScore = true;
        }

        for (int k = 0; k < n; k++)
        {
          GameEngineSearchResult r = results[k];
          if (r == null)
          {
            continue;
          }

          // Solve score (correctness 0..10), best move parsed uniformly from the UCI move string.
          Move bm = default;
          try
          {
            bm = Move.FromUCI(in position, r.MoveString);
          }
          catch
          {
            // Unparseable move (counts as not solved).
          }
          int score = bm.IsNull ? 0 : epd.CorrectnessScore(bm, 10);
          sumSolve[k] += score; cntSolve[k]++;

          // EPS (only counted when the engine reported a value).
          if (r.EPS > 0)
          {
            sumEPS[k] += r.EPS; cntEPS[k]++;
          }

          // Root Q (best-move Q, populated by all engine types).
          sumQ[k] += r.ScoreQ; cntQ[k]++;

          // Correct-move visit fraction (requires per-move visit statistics).
          double cv = CorrectMoveVisitFraction(r, epd);
          if (!double.IsNaN(cv))
          {
            sumCV[k] += cv; cntCV[k]++;
          }

          // Totals / additional aggregates.
          totNodes[k] += r.FinalN;
          totTime[k] += r.TimingStats.ElapsedTimeSecs;
          totNNEvals[k] += r.NumNNNodes;
          totTB[k] += r.CountSearchContinuations > 0 ? 0 : r.CountTablebaseHits;

          // Device backend ("in C++ interop") busy time, accumulated only over moves where the
          // metric is supported (NNEvaluatorTensorRT / NNEvaluatorCUDA), with the matching search-loop
          // elapsed time as the denominator for the backend-busy fraction.
          if (!double.IsNaN(r.TimeDeviceBackendWaitSeconds))
          {
            totBackendWait[k] += r.TimeDeviceBackendWaitSeconds;
            totBackendSearch[k] += r.TimeElapsedTotalSeconds;
          }
          if (r.AvgDepth > 0)
          {
            sumDepth[k] += r.AvgDepth; cntDepth[k]++;
          }
          if (r.VisitEntropy > 0)
          {
            sumEnt[k] += r.VisitEntropy; cntEnt[k]++;
          }

          // Baseline-relative difference statistics (skipped for the baseline engine itself).
          if (k != baselineIndex && baseR != null)
          {
            // Paired graded solution-quality difference (this engine minus baseline, 0-10 points).
            if (haveBaselineScore)
            {
              double scoreDiff = score - baselineScore;
              sumScoreDiff[k] += scoreDiff;
              sumScoreDiffSq[k] += scoreDiff * scoreDiff;
              cntScoreDiff[k]++;
            }

            sumQDiff[k] += Math.Abs(r.ScoreQ - baseR.ScoreQ); cntQDiff[k]++;
            if (r.VerboseMoveStats != null && baseR.VerboseMoveStats != null)
            {
              double kld = PolicySymmetricKLD(r.VerboseMoveStats, baseR.VerboseMoveStats);
              if (!double.IsNaN(kld))
              {
                sumKLD[k] += kld; cntKLD[k]++;
              }
            }
          }
        }
      }

      public MultiEngineEngineResult[] Snapshot()
      {
        MultiEngineEngineResult[] outArr = new MultiEngineEngineResult[n];
        for (int k = 0; k < n; k++)
        {
          // Per-column z-score of the paired graded solution-quality difference vs the baseline.
          // Uses population standard deviation (/cnt) divided by sqrt(cnt) for the standard error,
          // matching the headline two-engine statistic (StatUtils.StdDev / sqrt(n)).
          float gradedScoreDiffZ = float.NaN;
          if (k != baselineIndex && cntScoreDiff[k] > 1)
          {
            double cnt = cntScoreDiff[k];
            double mean = sumScoreDiff[k] / cnt;
            double ss = sumScoreDiffSq[k] - sumScoreDiff[k] * sumScoreDiff[k] / cnt;
            double sd = ss > 0 ? Math.Sqrt(ss / cnt) : 0;
            double se = sd / Math.Sqrt(cnt);
            gradedScoreDiffZ = se > 0 ? (float)(mean / se) : 0f;
          }

          outArr[k] = new MultiEngineEngineResult
          {
            ID = entries[k].ID,
            IsBaseline = k == baselineIndex,
            Kind = kinds[k],
            SearchLimit = entries[k].Limit,
            NumPositions = cntSolve[k],
            AvgSolveScorePct = cntSolve[k] > 0 ? (float)(sumSolve[k] / cntSolve[k]) * 10f : float.NaN,
            AvgEPS = cntEPS[k] > 0 ? (float)(sumEPS[k] / cntEPS[k]) : float.NaN,
            AvgTimeSecs = cntSolve[k] > 0 ? (float)(totTime[k] / cntSolve[k]) : float.NaN,
            AvgQ = cntQ[k] > 0 ? (float)(sumQ[k] / cntQ[k]) : float.NaN,
            AvgAbsQDiffVsBaseline = (k == baselineIndex || cntQDiff[k] == 0) ? float.NaN : (float)(sumQDiff[k] / cntQDiff[k]),
            AvgPolicyKLDVsBaseline = (k == baselineIndex || cntKLD[k] == 0) ? float.NaN : (float)(sumKLD[k] / cntKLD[k]),
            GradedScoreDiffZVsBaseline = gradedScoreDiffZ,
            AvgCorrectMoveVisitPct = cntCV[k] > 0 ? (float)(sumCV[k] / cntCV[k]) : float.NaN,
            TotalNodes = totNodes[k],
            TotalTimeSecs = (float)totTime[k],
            TotalNNEvals = totNNEvals[k],
            TotalTablebaseHits = totTB[k],
            AvgDepth = cntDepth[k] > 0 ? (float)(sumDepth[k] / cntDepth[k]) : float.NaN,
            AvgVisitEntropy = cntEnt[k] > 0 ? (float)(sumEnt[k] / cntEnt[k]) : float.NaN,
            BackendBusyFraction = totBackendSearch[k] > 0 ? (float)(totBackendWait[k] / totBackendSearch[k]) : float.NaN
          };
        }
        return outArr;
      }
    }

    #endregion


    private void Shutdown(ObjectPool<object> externalEnginePool)
    {
      return;

      // TODO: restore this, currently buggy (stack overflow)
      foreach (var engineSet in EngineSets)
      {
        engineSet.Engine1?.Dispose();
        engineSet.Engine2?.Dispose();
      }
      EngineSets.Clear();

      EngineExternal?.Dispose();

      externalEnginePool.Shutdown(engineObj => (engineObj as IDisposable)?.Dispose());
    }

    private void WriteSummaries()
    {
      // Averaged summary row, aligned directly beneath the per-position detail table.
      if (outputDetailRun)
      {
        WriteSummaryRow();
      }

      Def.Output.WriteLine();

      Def.Output.WriteLine();
      if (Def.ExternalEngineDef != null)
      {
        Def.Output.WriteLine($"Total {Def.ExternalEngineDef.ID} Time {totalTimeOther,6:F2}");
      }

      Def.Output.WriteLine($"Total C1 Time {totalTimeCeres1,6:F2}");
      if (Def.CeresEngine2Def != null)
      {
        Def.Output.WriteLine($"Total C2 Time {totalTimeCeres2,6:F2}");
      }

      Def.Output.WriteLine();
      if (Def.ExternalEngineDef != null)
      {
        Def.Output.WriteLine($"Avg {Def.ExternalEngineDef.ID} pos/sec    {totalNodesOther / totalTimeOther,8:F2}");
      }

      Def.Output.WriteLine($"Avg Ceres    pos/sec    {totalNodes1 / totalTimeCeres1,8:F2}");
      if (Def.CeresEngine2Def != null)
      {
        Def.Output.WriteLine($"Avg Ceres2    pos/sec    {totalNodes2 / totalTimeCeres2,8:F2}");
      }

      Def.Output.WriteLine();
      // Device backend busy fraction (time inside C++ interop / search-loop wall-clock; target 1.0).
      // Only reported where supported (NNEvaluatorTensorRT / NNEvaluatorCUDA); otherwise shown as "-".
      Def.Output.WriteLine($"Backend busy C1         {(sumBackendSearch1 > 0 ? (sumBackendWait1 / sumBackendSearch1).ToString("F3") : "-"),8}");
      if (Def.CeresEngine2Def != null)
      {
        Def.Output.WriteLine($"Backend busy C2         {(sumBackendSearch2 > 0 ? (sumBackendWait2 / sumBackendSearch2).ToString("F3") : "-"),8}");
      }

      Def.Output.WriteLine();

      Def.Output.WriteLine($"Num all evaluations      :   {NNEvaluatorStats.TotalPosEvaluations,12:N0}");

      Def.Output.WriteLine();
      Def.Output.WriteLine($"Ceres1 total nodes to solve {sumCeres1NumNodesWhenChoseTopNode,12:N0}");
      Def.Output.WriteLine($"Ceres2 total nodes to solve {sumCeres2NumNodesWhenChoseTopNode,12:N0}");

      if (Def.CeresEngine2Def != null)
      {
        Def.Output.WriteLine();
        Def.Output.WriteLine($"Average absolute difference in final Q {StatUtils.Average(StatUtils.AbsDiff(finalQ1.ToArray(), finalQ2.ToArray())),5:F3}");
      }

      Def.Output.WriteLine();
      float avgFaster = (float) StatUtils.Average(solvedPct1MinusPct2Samples);
      float stdFaster = (float)StatUtils.StdDev(solvedPct1MinusPct2Samples) / MathF.Sqrt(solvedPct1MinusPct2Samples.Count);
      Def.Output.WriteLine($"Ceres1 time required to solve vs. Ceres2 (%) {(100* avgFaster),5:F2} +/-{(100 * stdFaster),5:F2}");

      if (Def.CeresEngine2Def != null && scoreDiffSamples.Count > 0)
      {
        // Headline decision statistic: paired graded-score difference (Ceres1 - Ceres2) over the suite.
        // With weighted EPDs this is a continuous solution-quality signal; |z| >~ 2 is significant.
        float sdMean = (float)StatUtils.Average(scoreDiffSamples);
        float sdSE = scoreDiffSamples.Count > 1 ? (float)StatUtils.StdDev(scoreDiffSamples) / MathF.Sqrt(scoreDiffSamples.Count) : 0;
        float sdZ = sdSE > 0 ? sdMean / sdSE : 0;
        Def.Output.WriteLine();
        Def.Output.WriteLine($"Graded solution-quality difference (Ceres1 - Ceres2, 0-10 pts) {sdMean,6:F3} +/-{sdSE,5:F3}  z= {sdZ,5:F2}  (n={scoreDiffSamples.Count})");
      }

      if (Def.CeresEngine2Def != null)
      {
        Def.Output.WriteLine();
        Def.Output.WriteLine($"Correct move visits (fraction of root visits on the correct move(s), ~equal = within 3 points):");
        Def.Output.WriteLine($"  Ceres1 more visits {countEngine1MoreCorrectVisits,5}   "
                           + $"Ceres2 more visits {countEngine2MoreCorrectVisits,5}   "
                           + $"~equal {countCorrectVisitsEqual,5}   of {countCorrectVisitsCompared,5}");
      }

      // Recap header block followed by the boxed performance summary, at the very end of output.
      Def.Output.WriteLine();
      WriteSuiteHeaderBlock();
      if (Def.CeresEngine2Def != null)
      {
        Def.Output.WriteLine();
        WritePerformanceSummary();
      }

      Def.Output.WriteLine();
    }


    /// <summary>
    /// Emits a final "summary row" beneath the per-position detail table. Each column shows
    /// the average of its per-position values, except the running cumulative-average score
    /// columns (rendered as their final converged value) and non-numeric columns (blank).
    /// The row reuses the exact column ids and widths captured during the run, so it aligns
    /// directly under the table headers.
    /// </summary>
    private void WriteSummaryRow()
    {
      if (columnAcc.RowCount == 0)
      {
        return;
      }

      Writer w = new Writer(true);
      foreach ((string id, int width, CellAgg agg, string summaryFormat) in columnAcc.Order)
      {
        string display;
        if (id == "#")
        {
          display = "AVG";
        }
        else if (agg == CellAgg.None)
        {
          display = "";
        }
        else if (agg == CellAgg.Final)
        {
          display = FormatFinal(id, summaryFormat);
        }
        else
        {
          double avg = columnAcc.Average(id);
          display = double.IsNaN(avg) ? "" : string.Format(summaryFormat, avg);
        }

        w.Add(id, display, width);
      }

      Def.Output.WriteLine(w.dividers.ToString());
      Def.Output.WriteLine(w.text.ToString());
    }


    /// <summary>
    /// Renders the final converged value of a running cumulative-average "score" column,
    /// computed directly from the authoritative accumulators (matching the per-position
    /// interpolation), using the supplied composite format string.
    /// </summary>
    private string FormatFinal(string id, string summaryFormat)
    {
      double v = id switch
      {
        "CEx" => numSearches == 0 ? 0 : (double)avgOther / numSearches,
        "CC"  => numSearches == 0 ? 0 : (double)accCeres1 / numSearches,
        "CC2" => numSearches == 0 ? 0 : (double)accCeres2 / numSearches,
        "P"   => numSearchesBothFound == 0 ? 0 : 0.001 * accWCeres1 / numSearchesBothFound,
        "P2"  => numSearchesBothFound == 0 ? 0 : 0.001 * accWCeres2 / numSearchesBothFound,
        _     => double.NaN
      };

      return double.IsNaN(v) ? "" : string.Format(summaryFormat, v);
    }


    /// <summary>
    /// Converts a list of per-move verbose stats into an empirical policy distribution
    /// (move -> probability) using final visit counts, excluding the "node" pseudo-entry.
    /// Returns null if there were no visits.
    /// </summary>
    private static Dictionary<string, double> EmpiricalPolicyDistribution(List<VerboseMoveStat> stats)
    {
      Dictionary<string, long> counts = new();
      long total = 0;
      foreach (VerboseMoveStat s in stats)
      {
        if (s.MoveString == null || s.MoveString == "node")
        {
          continue;
        }
        counts.TryGetValue(s.MoveString, out long c);
        counts[s.MoveString] = c + s.VisitCount;
        total += s.VisitCount;
      }

      if (total <= 0 || counts.Count == 0)
      {
        return null;
      }

      Dictionary<string, double> dist = new(counts.Count);
      foreach (KeyValuePair<string, long> kv in counts)
      {
        dist[kv.Key] = (double)kv.Value / total;
      }
      return dist;
    }


    /// <summary>
    /// Symmetric KL divergence 0.5*(KL(P1||P2) + KL(P2||P1)) between the two engines' empirical
    /// root-move visit distributions. Returns NaN if either distribution is unavailable.
    /// A small probability floor avoids log(0) for moves visited by only one engine.
    /// </summary>
    private static double PolicySymmetricKLD(List<VerboseMoveStat> stats1, List<VerboseMoveStat> stats2)
    {
      Dictionary<string, double> d1 = EmpiricalPolicyDistribution(stats1);
      Dictionary<string, double> d2 = EmpiricalPolicyDistribution(stats2);
      if (d1 == null || d2 == null)
      {
        return double.NaN;
      }

      HashSet<string> moves = new(d1.Keys);
      moves.UnionWith(d2.Keys);

      const double FLOOR = 1e-9;
      double KL(Dictionary<string, double> p, Dictionary<string, double> q)
      {
        double acc = 0;
        foreach (string m in moves)
        {
          p.TryGetValue(m, out double pv);
          if (pv <= 0)
          {
            continue;
          }
          q.TryGetValue(m, out double qv);
          acc += pv * Math.Log(pv / Math.Max(qv, FLOOR));
        }
        return acc;
      }

      return 0.5 * (KL(d1, d2) + KL(d2, d1));
    }


    /// <summary>
    /// Returns the percentage of the search's total root visits (FinalN) that landed on the
    /// correct move(s) for this position, summed over all correct moves (handles multi-best and
    /// avoid-move EPDs uniformly via EPDEntry.CorrectnessScore). Returns NaN if per-move visit
    /// statistics are unavailable or the search had no visits.
    /// </summary>
    private static double CorrectMoveVisitFraction(GameEngineSearchResult search, EPDEntry epd)
    {
      if (search?.VerboseMoveStats == null || search.FinalN <= 0)
      {
        return double.NaN;
      }

      Position position = epd.Position;
      long correctVisits = 0;
      foreach (VerboseMoveStat stat in search.VerboseMoveStats)
      {
        if (stat.MoveString == null || stat.MoveString == "node")
        {
          continue;
        }

        Move move;
        try
        {
          move = Move.FromUCI(in position, stat.MoveString);
        }
        catch
        {
          continue;
        }

        if (epd.CorrectnessScore(move, 10) == 10)
        {
          correctVisits += stat.VisitCount;
        }
      }

      return 100.0 * correctVisits / search.FinalN;
    }


    /// <summary>
    /// Writes a recap header block (loosely patterned after TournamentDef.DumpParams) identifying
    /// the suite, machine, positions, EPD file, and the engine player definitions.
    /// </summary>
    private void WriteSuiteHeaderBlock()
    {
      void Line(string label, object value) => Def.Output.WriteLine($"  {label,-20}: {value}");

      Def.Output.WriteLine("SUITE TEST");
      Line("ID", Def.ID);
      Line("Machine Name", Environment.MachineName);
      Line("Date/Time", DateTime.Now);
      Line("Test Positions", numPositionsInRun);
      Line("EPD file name", Def.EPDFileName);
      Def.Output.WriteLine($"  Player 1 : {Def.CeresEngine1Def}");
      if (Def.RunCeres2Engine)
      {
        Def.Output.WriteLine($"  Player 2 : {Def.CeresEngine2Def}");
      }
      if (Def.ExternalEngineDef != null)
      {
        Def.Output.WriteLine($"  External : {Def.ExternalEngineDef}");
      }
    }


    /// <summary>
    /// Writes the boxed performance-summary table of head-to-head metrics (engine 1 vs engine 2).
    /// </summary>
    private void WritePerformanceSummary()
    {
      float meanAbsQ = (finalQ2 == null || finalQ2.Count == 0)
                     ? 0
                     : (float)StatUtils.Average(StatUtils.AbsDiff(finalQ1.ToArray(), finalQ2.ToArray()));

      float avgScore1 = numSearches > 0 ? (float)accCeres1 / numSearches : 0;
      float avgScore2 = numSearches > 0 ? (float)accCeres2 / numSearches : 0;

      float avgEPS1 = numSearches > 0 ? (float)sumEPS1 / numSearches : 0;
      float avgEPS2 = numSearches > 0 ? (float)sumEPS2 / numSearches : 0;
      float relEPS = avgEPS2 > 0 ? (avgEPS1 / avgEPS2 - 1.0f) * 100.0f : 0;

      string policyStr = countPolicyKLD > 0
                       ? (sumPolicyKLD / countPolicyKLD).ToString("F4")
                       : "n/a";

      List<(string, string)> rows = new()
      {
        ("Q difference (mean abs)",       meanAbsQ.ToString("F4")),
        ("Policy difference (sym KLD)",   policyStr),
        ("Evaluation intensity (EPS)",    $"{relEPS.ToString("+0.0;-0.0")}%  ({avgEPS1:N0} vs {avgEPS2:N0})"),
        ("Solve score difference",        $"{(avgScore1 - avgScore2).ToString("+0.00;-0.00")} ({avgScore1:F2} vs {avgScore2:F2})"),
        ("Solve correct move visits (3%)", $"{countEngine1MoreCorrectVisits} better vs {countEngine2MoreCorrectVisits} better"),
      };

      WriteBoxedTable("PERFORMANCE SUMMARY  (C1 vs C2)", rows);
    }


    /// <summary>
    /// Writes a simple ASCII boxed table: a title row, then one "label : value" row per entry,
    /// auto-sized to the widest content. ASCII borders are used for cross-platform/log safety.
    /// </summary>
    private void WriteBoxedTable(string title, List<(string Label, string Value)> rows)
    {
      int labelWidth = 0;
      foreach ((string label, _) in rows)
      {
        labelWidth = Math.Max(labelWidth, label.Length);
      }

      List<string> contentLines = new(rows.Count);
      int innerWidth = title.Length;
      foreach ((string label, string value) in rows)
      {
        string line = $"{label.PadRight(labelWidth)} : {value}";
        contentLines.Add(line);
        innerWidth = Math.Max(innerWidth, line.Length);
      }

      string border = "+" + new string('-', innerWidth + 2) + "+";

      Def.Output.WriteLine(border);
      Def.Output.WriteLine($"| {title.PadRight(innerWidth)} |");
      Def.Output.WriteLine(border);
      foreach (string line in contentLines)
      {
        Def.Output.WriteLine($"| {line.PadRight(innerWidth)} |");
      }
      Def.Output.WriteLine(border);
    }



    List<float> finalQ1 = new();
    List<float> finalQ2 = new();

    readonly object lockObj = new();

    void ProcessEPD(int epdNum, EPDEntry epd, bool outputDetail, ObjectPool<object> otherEngines)
    {
      if (!EngineSets.TryTake(out var engineSet))
      {
        throw new Exception("No engine available");
      }

      GameEngine EngineCeres1 = engineSet.Engine1;
      GameEngine EngineCeres2 = engineSet.Engine2;

      EngineCeres1.ResetGame();
      EngineCeres2?.ResetGame();
      EngineExternal?.ResetGame();

      UCISearchInfo otherEngineAnalysis2 = default;


      GameEngineSearchResult search1 = null;
      GameEngineSearchResult search2 = null;
      int scoreCeres1 = 0, scoreCeres2 = 0, scoreOtherEngine = 0;
      float otherEngineTime = 0;

      // The case of Lichess puzzles has to be handled specially.
      // We need to sequentially play out all the moves in the puzzle.
      // The engine gets all or no credit depending if it finds every one of the correct moves.
      LichessDatabaseRecord lichessPuzzle = epd.IsLichessPuzzle ? new LichessDatabaseRecord(epd.LichessRawLine) : default;
      int count = epd.IsLichessPuzzle ? lichessPuzzle.NumPuzzleMoves : 1;
      int countFailuresCeres1 = 0;
      int countFailuresCeres2 = 0;
      int countFailuresOtherEngine = 0;
      for (int i = 0; i < count; i++)
      {
        EPDEntry epdInSequence = epd.IsLichessPuzzle ? lichessPuzzle.EPDForPuzzleMoveAtIndex(i) : epd;

        bool isLastEntry = i == count - 1;
        otherEngineAnalysis2 = RunSearch(epdNum, epdInSequence,
                                         otherEngines, EngineCeres1, EngineCeres2,
                                         otherEngineAnalysis2,
                                         isLastEntry,
                                         out search1, out search2,
                                         out scoreCeres1, out scoreCeres2,
                                         out scoreOtherEngine,
                                         out otherEngineTime);

        if (scoreCeres1 < 10)
        {
          countFailuresCeres1++;
        }
        if (scoreCeres2 < 10)
        {
          countFailuresCeres2++;
        }
        if (scoreOtherEngine < 10)
        {
          countFailuresOtherEngine++;
        }
      }

      if (epd.IsLichessPuzzle && lichessPuzzle.NumPuzzleMoves > 1)
      {
        if (countFailuresCeres1 > 0)
        {
          scoreCeres1 = 0;
        }
        if (countFailuresCeres2 > 0)
        {
          scoreCeres2 = 0;
        }
        if (countFailuresOtherEngine > 0)
        {
          scoreOtherEngine = 0;
        }
      }

      float avgCeres1 = (float)accCeres1 / numSearches;
      float avgCeres2 = (float)accCeres2 / numSearches;
      float avgWCeres1 = (float)accWCeres1 / numSearchesBothFound;
      float avgWCeres2 = (float)accWCeres2 / numSearchesBothFound;

      float avgOther = (float)this.avgOther / numSearches;

      string MoveIfWrong(Move m) => m.IsNull || epd.CorrectnessScore(m, 10) == 10 ? "    " : m.ToString().ToLower();

      int diff1 = scoreCeres1 - scoreOtherEngine;

      //NodeEvaluatorNeuralNetwork
      int evalNumBatches1 = search1.NumNNBatches;
      int evalNumPos1 = search1.NumNNNodes;
      int evalNumBatches2 = search2 == null ? 0 : search2.NumNNBatches;
      int evalNumPos2 = search2 == null ? 0 : search2.NumNNNodes;

      // Fraction (percent) of total root visits (FinalN) each engine placed on the correct
      // move(s) at the end of search (NaN if per-move visit statistics are unavailable).
      double correctVisitFrac1 = CorrectMoveVisitFraction(search1, epd);
      double correctVisitFrac2 = search2 == null ? double.NaN : CorrectMoveVisitFraction(search2, epd);

      string correctMove = null;
      if (epd.AMMoves != null)
      {
        correctMove = "-" + epd.AMMoves[0];
      }
      else if (epd.BMMoves != null)
      {
        correctMove = epd.BMMoves[0];
      }

      lock (lockObj)
      {
        totalTimeOther += otherEngineTime;
        totalTimeCeres1 += (float)search1.TimingStats.ElapsedTimeSecs;

        totalNodesOther += otherEngineAnalysis2 == null ? 0 : (int)otherEngineAnalysis2.Nodes;
        totalNodes1 += search1.FinalN;

        sumEvalNumPosOther += otherEngineAnalysis2 == null ? 0 : (int)otherEngineAnalysis2.Nodes;
        sumEvalNumBatches1 += evalNumBatches1;
        sumEvalNumPos1 += evalNumPos1;
        sumTablebaseHits1 += search1.CountSearchContinuations > 0 ? 0 : search1.CountTablebaseHits;
        sumEPS1 += search1.EPS;
        if (!double.IsNaN(search1.TimeDeviceBackendWaitSeconds))
        {
          sumBackendWait1 += search1.TimeDeviceBackendWaitSeconds;
          sumBackendSearch1 += search1.TimeElapsedTotalSeconds;
        }

        if (Def.RunCeres2Engine)
        {
          totalTimeCeres2 += (float)search2.TimingStats.ElapsedTimeSecs;
          totalNodes2 += search2.FinalN;
          sumEvalNumBatches2 += evalNumBatches2;
          sumEvalNumPos2 += evalNumPos2;
          sumTablebaseHits2 += search2.CountSearchContinuations > 0 ? 0 : search2.CountTablebaseHits;
          sumEPS2 += search2.EPS;
          if (!double.IsNaN(search2.TimeDeviceBackendWaitSeconds))
          {
            sumBackendWait2 += search2.TimeDeviceBackendWaitSeconds;
            sumBackendSearch2 += search2.TimeElapsedTotalSeconds;
          }
        }

        // Correct-move-visit comparison: which engine placed a larger fraction of its visits on
        // the correct move(s), counting the two as tied when within 3 percentage points.
        if (search2 != null && !double.IsNaN(correctVisitFrac1) && !double.IsNaN(correctVisitFrac2))
        {
          double diffPts = correctVisitFrac1 - correctVisitFrac2;
          if (Math.Abs(diffPts) <= 3.0)
          {
            countCorrectVisitsEqual++;
          }
          else if (diffPts > 0)
          {
            countEngine1MoreCorrectVisits++;
          }
          else
          {
            countEngine2MoreCorrectVisits++;
          }
          countCorrectVisitsCompared++;
        }
      }

      string worker1PickedNonTopNMoveStr = search1.PickedNonTopNMoveStr;
      string worker2PickedNonTopNMoveStr = search2?.PickedNonTopNMoveStr;

      bool ex = otherEngineAnalysis2 != null;
      bool c2 = search2 != null;

      Writer writer = new Writer(epdNum == 0);

      // Every column is emitted via this single helper, which both writes the per-position
      // cell to the Writer (byte-identical to the historical output) and records a Cell so
      // the same column layout can be replayed as an averaged summary row at end of suite.
      List<Cell> cells = new();
      void Emit(string id, string display, int width, CellAgg agg, double value = 0, string summaryFormat = null)
      {
        writer.Add(id, display, width);
        cells.Add(new Cell(id, width, display, agg, value, summaryFormat));
      }

      // Renders a correct-move-visit-fraction cell (NaN -> "n/a"), prefixed by the non-top-N flag.
      string FrCell(string prefix, double frac) => double.IsNaN(frac) ? $"{prefix}n/a" : $"{prefix}{frac,3:F0}%";

      Emit("#", $"{epdNum,4}", 6, CellAgg.None);

      if (ex)
      {
        Emit("CEx", $"{avgOther,5:F2}", 7, CellAgg.Final, summaryFormat: "{0,5:F2}");
      }

      Emit("CC", $"{avgCeres1,5:F2}", 7, CellAgg.Final, summaryFormat: "{0,5:F2}");
      if (c2)
      {
        Emit("CC2", $"{avgCeres2,5:F2}", 7, CellAgg.Final, summaryFormat: "{0,5:F2}");
      }

      Emit("P", $"{0.001f * avgWCeres1,6:f2}", 8, CellAgg.Final, summaryFormat: "{0,6:F2}");
      if (c2)
      {
        Emit("P2", $"{0.001f * avgWCeres2,6:f2}", 8, CellAgg.Final, summaryFormat: "{0,6:F2}");
      }

      if (ex)
      {
        Emit("SEx", $" {scoreOtherEngine,3}", 5, CellAgg.Average, scoreOtherEngine, " {0,3:F1}");
      }

      Emit("SC", $" {scoreCeres1,3}", 5, CellAgg.Average, scoreCeres1, " {0,3:F1}");
      if (c2)
      {
        Emit("SC2", $" {scoreCeres2,3}", 5, CellAgg.Average, scoreCeres2, " {0,3:F1}");
      }

      if (ex)
      {
        Emit("MEx", $"{otherEngineAnalysis2.BestMove,7}", 9, CellAgg.None);
      }

      Emit("MC", $"{search1.BestMoveMG,7}", 9, CellAgg.None);
      if (c2)
      {
        Emit("MC2", $"{search2.BestMoveMG,7}", 9, CellAgg.None);
      }

      // Fr / Fr2: percentage of total root visits placed on the correct move(s) at end of search.
      Emit("Fr", FrCell(worker1PickedNonTopNMoveStr, correctVisitFrac1), 8,
           CellAgg.Average, correctVisitFrac1, "{0,3:F0}%");
      if (c2)
      {
        Emit("Fr2", FrCell(worker2PickedNonTopNMoveStr, correctVisitFrac2), 8,
             CellAgg.Average, correctVisitFrac2, "{0,3:F0}%");
      }

      Emit("Yld", $"{search1.NodeSelectionYieldFrac,6:f3}", 9, CellAgg.Average, search1.NodeSelectionYieldFrac, "{0,6:F3}");
      if (c2)
      {
        Emit("Yld2", $"{search2.NodeSelectionYieldFrac,6:f3}", 9, CellAgg.Average, search2.NodeSelectionYieldFrac, "{0,6:F3}");
      }

      // Search time
      if (ex)
      {
        Emit("TimeEx", $"{otherEngineTime,7:F2}", 9, CellAgg.Average, otherEngineTime, "{0,7:F2}");
      }

      Emit("TimeC", $"{search1.TimingStats.ElapsedTimeSecs,7:F2}", 9, CellAgg.Average, search1.TimingStats.ElapsedTimeSecs, "{0,7:F2}");
      if (c2)
      {
        Emit("TimeC2", $"{search2.TimingStats.ElapsedTimeSecs,7:F2}", 9, CellAgg.Average, search2.TimingStats.ElapsedTimeSecs, "{0,7:F2}");
      }

      Emit("ADep", $"{search1.AvgDepth,5:f1}", 7, CellAgg.Average, search1.AvgDepth, "{0,5:F1}");
      if (c2)
      {
        Emit("ADep2", $"{search2.AvgDepth,5:f1}", 7, CellAgg.Average, search2.AvgDepth, "{0,5:F1}");
      }

      Emit("MDep", $"{search1.MaxDepth,5:f1}", 7, CellAgg.Average, search1.MaxDepth, "{0,5:F1}");
      if (c2)
      {
        Emit("MDep2", $"{search2.MaxDepth,5:f1}", 7, CellAgg.Average, search2.MaxDepth, "{0,5:F1}");
      }

      Emit("VEnt", $"{search1.VisitEntropy,5:f2}", 7, CellAgg.Average, search1.VisitEntropy, "{0,5:F2}");
      if (c2)
      {
        Emit("VEnt2", $"{search2.VisitEntropy,5:f2}", 7, CellAgg.Average, search2.VisitEntropy, "{0,5:F2}");
      }

      // Nodes
      if (ex)
      {
        Emit("NEx", $"{otherEngineAnalysis2.Nodes,12:N0}", 14, CellAgg.Average, otherEngineAnalysis2.Nodes, "{0,12:N0}");
      }
      Emit("Nodes", $"{search1.FinalN,12:N0}", 14, CellAgg.Average, search1.FinalN, "{0,12:N0}");
      if (c2)
      {
        Emit("Nodes2", $"{search2.FinalN,12:N0}", 14, CellAgg.Average, search2.FinalN, "{0,12:N0}");
      }

      // Score (Q)
      if (ex)
      {
        Emit("QEx", $"{otherEngineAnalysis2.ScoreLogistic,6:F3}", 8, CellAgg.Average, otherEngineAnalysis2.ScoreLogistic, "{0,6:F3}");
      }

      Emit("QC", $"{search1.ScoreQRoot,6:F3}", 8, CellAgg.Average, search1.ScoreQRoot, "{0,6:F3}");
      if (c2)
      {
        Emit("QC2", $"{search2.ScoreQRoot,6:F3}", 8, CellAgg.Average, search2.ScoreQRoot, "{0,6:F3}");
      }

      // Num batches&positions
      Emit("Batches", $"{evalNumBatches1,8:N0}", 10, CellAgg.Average, evalNumBatches1, "{0,8:N0}");
      Emit("NNEvals", $"{evalNumPos1,11:N0}", 13, CellAgg.Average, evalNumPos1, "{0,11:N0}");
      if (c2)
      {
        Emit("Batches2", $"{evalNumBatches2,8:N0}", 10, CellAgg.Average, evalNumBatches2, "{0,8:N0}");
        Emit("NNEvals2", $"{evalNumPos2,11:N0}", 13, CellAgg.Average, evalNumPos2, "{0,11:N0}");
      }

      // Tablebase hits
      Emit("TBase", $"{(search1.CountSearchContinuations > 0 ? 0 : search1.CountTablebaseHits),8:N0}", 10,
           CellAgg.Average, search1.CountSearchContinuations > 0 ? 0 : search1.CountTablebaseHits, "{0,8:N0}");
      if (c2)
      {
        Emit("TBase2", $"{(search2.CountSearchContinuations > 0 ? 0 : search2.CountTablebaseHits),8:N0}", 10,
             CellAgg.Average, search2.CountSearchContinuations > 0 ? 0 : search2.CountTablebaseHits, "{0,8:N0}");
      }

      if (Def.DumpEPDInfo)
      {
        Emit("FEN", epd.FEN, -1, CellAgg.None);
      }

      // Merge this row's columns into the run-level accumulator (atomically, under the lock),
      // so the summary row and the comprehensive SuiteTestResult can be produced at the end.
      lock (lockObj)
      {
        columnAcc.Merge(cells);
      }

      if (outputDetail)
      {
        if (epdNum == 0)
        {
          Def.Output.WriteLine(writer.ids.ToString());
          Def.Output.WriteLine(writer.dividers.ToString());
        }
        Def.Output.WriteLine(writer.text.ToString());
      }
    }


    private UCISearchInfo RunSearch(int epdNum, EPDEntry epd,
                                    ObjectPool<object> otherEngines,
                                    GameEngine engineCeres1,
                                    GameEngine engineCeres2,
                                    UCISearchInfo otherEngineAnalysis2,
                                    bool restoreEnginesToPoolWhenDone,
                                    out GameEngineSearchResult search1,
                                    out GameEngineSearchResult search2,
                                    out int scoreCeres1, out int scoreCeres2, out int scoreOtherEngine,
                                    out float otherEngineTime)
    {
      Task RunNonCeres()
      {
        if (Def.ExternalEngineDef != null)
        {
          object engineObj = otherEngines.GetFromPool();

          SearchLimit adjustedLimit = Def.ExternalEngineDef.SearchLimit.ConvertedGameToMoveLimit;
          if (engineObj is LC0Engine)
          {
            LC0Engine le = (LC0Engine)engineObj;

            // Run test 2 first since that's the one we dump in detail, to avoid any possible caching effect from a prior run
            otherEngineAnalysis2 = le.AnalyzePositionFromFEN(epd.FENAndMoves, adjustedLimit);

            if (restoreEnginesToPoolWhenDone)
            {
              otherEngines.RestoreToPool(le);
            }
          }
          else if (engineObj is GameEngine gameEngine)
          {
            GameEngineSearchResult result = gameEngine.Search(epd.PosWithHistory, adjustedLimit);
            UCISearchInfo uciInfo = new UCISearchInfo(null, result.MoveString, null);
            uciInfo.Nodes = (ulong)result.FinalN;
            uciInfo.EngineReportedSearchTime = (int)(1000.0f * result.TimingStats.ElapsedTimeSecs);
            uciInfo.ScoreCentipawns = (int)MathF.Round(result.ScoreCentipawns);
            uciInfo.BestMove = result.MoveString;
            otherEngineAnalysis2 = uciInfo;
            if (restoreEnginesToPoolWhenDone)
            {
              otherEngines.RestoreToPool(engineObj);
            }
          }
          else if (engineObj is UCIGameRunner runner)
          {
            string moveType = Def.ExternalEngineDef.SearchLimit.Type == SearchLimitType.NodesPerMove ? "nodes" : "movetime";
            int moveValue = moveType == "nodes" ? (int)Def.ExternalEngineDef.SearchLimit.Value : (int)adjustedLimit.Value * 1000;
            runner.EvalPositionPrepare();
            otherEngineAnalysis2 = runner.EvalPosition(epd.FEN, epd.StartMoves, moveType, moveValue, null);
            if (restoreEnginesToPoolWhenDone)
            {
              otherEngines.RestoreToPool(runner);
            }
          }
        }
        return Task.CompletedTask;
      }

      bool EXTERNAL_CONCURRENT = numConcurrentSuiteThreads > 1;

      Task lzTask = EXTERNAL_CONCURRENT ? Task.Run(RunNonCeres) : RunNonCeres();

      // Compute search limit
      // If possible, adjust for the fact that LC0 "cheats" by going slightly over node budget
      SearchLimit ceresSearchLimit1 = Def.CeresEngine1Def.SearchLimit.ConvertedGameToMoveLimit;
      SearchLimit ceresSearchLimit2 = Def.CeresEngine2Def?.SearchLimit.ConvertedGameToMoveLimit;

      if (Def.CeresEngine1Def.SearchLimit.Type == SearchLimitType.NodesPerMove
       && otherEngineAnalysis2 != null
       && !Def.Engine1Def.GetFutilityPruningStopSearchEnabled())
      {
        if (Def.CeresEngine1Def.SearchLimit.Type == SearchLimitType.NodesPerMove)
        {
          ceresSearchLimit1 = new SearchLimit(SearchLimitType.NodesPerMove, otherEngineAnalysis2.Nodes);
        }
        if (Def.CeresEngine1Def.SearchLimit.Type == SearchLimitType.NodesPerMove)
        {
          ceresSearchLimit2 = new SearchLimit(SearchLimitType.NodesPerMove, otherEngineAnalysis2.Nodes);
        }
      }

      PositionWithHistory pos = epd.PosWithHistory;

      // Note that if we are running both Ceres1 and Ceres2 we alternate which search goes first.
      // This prevents any systematic difference/benefit that might come from order
      // (for example if we reuse position evaluations from the other tree, which can benefit only one of the two searches).
      search1 = null;
      search2 = null;
      if (epdNum % 2 == 0 || Def.CeresEngine2Def == null)
      {
        engineCeres1.ResetGame();
        search1 = engineCeres1.Search(pos, ceresSearchLimit1);

        if (Def.RunCeres2Engine)
        {
          search2 = engineCeres2.Search(pos, ceresSearchLimit2);
        }

      }
      else
      {
        engineCeres2.ResetGame();
        search2 = engineCeres2.Search(pos, ceresSearchLimit2);

        search1 = engineCeres1.Search(pos, ceresSearchLimit1);

      }

      if (restoreEnginesToPoolWhenDone)
      {
        // Restore engines to pool (as a pair, so a worker keeps its assigned device(s)).
        EngineSets.Add((engineCeres1, engineCeres2));
      }

      lock (lockObj)
      {
        while (finalQ1.Count <= epdNum)
        {
          finalQ1.Add(float.NaN);
        }

        finalQ1[epdNum] = (float)search1.ScoreQ;

        if (search2 != null)
        {
          while (finalQ2.Count <= epdNum)
          {
            finalQ2.Add(float.NaN);
          }

          finalQ2[epdNum] = (float)search2.ScoreQ;
        }
      }

      // Wait for LZ analysis
      if (EXTERNAL_CONCURRENT) lzTask.Wait();

      Move bestMoveOtherEngine = default;

      if (Def.ExternalEngineDef != null)
      {
        MGPosition thisPosX = PositionWithHistory.FromFENAndMovesUCI(epd.FEN, epd.StartMoves).FinalPosMG;

        MGMove lzMoveMG1 = MGMoveFromString.ParseMove(thisPosX, otherEngineAnalysis2.BestMove);
        bestMoveOtherEngine = MGMoveConverter.ToMove(lzMoveMG1);
      }

      Move bestMoveCeres1 = MGMoveConverter.ToMove(search1.BestMoveMG);
      Move bestMoveCeres2 = search2 == null ? default : MGMoveConverter.ToMove(search2.BestMoveMG);

      char CorrectStr(Move move) => epd.CorrectnessScore(move, 10) == 10 ? '+' : '.';

      scoreCeres1 = epd.CorrectnessScore(bestMoveCeres1, 10);
      scoreCeres2 = epd.CorrectnessScore(bestMoveCeres2, 10);
      scoreOtherEngine = epd.CorrectnessScore(bestMoveOtherEngine, 10);

      // Possibly invoke callback method.
      Def.Callback?.Invoke(epd, scoreCeres1, search1);

      otherEngineTime = otherEngineAnalysis2 == null ? 0 : (float)otherEngineAnalysis2.EngineReportedSearchTime / 1000.0f;
      lock (lockObj)
      {
        accCeres1 += scoreCeres1;
        accCeres2 += scoreCeres2;

        // Paired graded-score difference (only meaningful when a second engine is present).
        // scoreCeres1/scoreCeres2 are graded (0-10) when the EPD carries weighted ScoredMoves,
        // so this captures solution-quality differences continuously rather than just solved/unsolved.
        if (search2 != null)
        {
          scoreDiffSamples.Add(scoreCeres1 - scoreCeres2);
        }

        // Accumulate how many nodes were required to find one of the correct moves
        // (only in cases where both engines found correct move at some point).
        if (scoreCeres1 > 0 && (search2 == null || scoreCeres2 > 0))
        {
          sumCeres1NumNodesWhenChoseTopNode += search1.NumNodesWhenChoseTopNNode;
          if (search2 != null)
          {
            sumCeres2NumNodesWhenChoseTopNode += search2.NumNodesWhenChoseTopNNode;
          }
        }

        // Accumulate how many nodes were required to find one of the correct moves
        // (in the cases where both succeeded)
        if (scoreCeres1 > 0 && (search2 == null || scoreCeres2 > 0))
        {
          accWCeres1 += (scoreCeres1 == 0) ? search1.FinalN : search1.NumNodesWhenChoseTopNNode;
          if (search2 != null)
          {
            accWCeres2 += (scoreCeres2 == 0) ? search2.FinalN : search2.NumNodesWhenChoseTopNNode;
            solvedPct1MinusPct2Samples.Add(search1.FractionNumNodesWhenChoseTopNNode - search2.FractionNumNodesWhenChoseTopNNode);
          }
          numSearchesBothFound++;
        }

        // Policy difference: symmetric KLD between the two engines' root visit distributions
        // (only when both engines exposed per-move visit statistics for this position).
        if (search2 != null && search1.VerboseMoveStats != null && search2.VerboseMoveStats != null)
        {
          double kld = PolicySymmetricKLD(search1.VerboseMoveStats, search2.VerboseMoveStats);
          if (!double.IsNaN(kld))
          {
            sumPolicyKLD += kld;
            countPolicyKLD++;
          }
        }

        this.avgOther += scoreOtherEngine;

        numSearches++;
      }

      return otherEngineAnalysis2;
    }
  }

  internal class Writer
  {
    public readonly bool WithHeader;

    public StringBuilder text = new StringBuilder();
    public StringBuilder ids;
    public StringBuilder dividers;

    public Writer(bool withHeader)
    {
      WithHeader = withHeader;

      if (withHeader)
      {
        ids = new StringBuilder();
        dividers = new StringBuilder();
      }
    }

    public void Add(string id, string value, int width)
    {
      if (WithHeader)
      {
        if (width != -1 && id.Length > width)
        {
          id = id.Substring(width);
        }

        ids.Append(width == -1 ? id : Center(id, width));

        if (width != -1)
        {
          for (int i = 0; i < width - 2; i++)
          {
            dividers.Append("-");
          }
        }
        dividers.Append("  ");
      }

      text.Append(width == -1 ? value : StringUtils.Sized(value, width));
    }

    static string Center(string str, int width)
    {
      int pad = width - str.Length;
      int padLeft = pad / 2 + str.Length;
      return str.PadLeft(padLeft, ' ').PadRight(width, ' ');
    }
  }


  /// <summary>
  /// Aggregation behavior for a console column when building the summary row at end of suite.
  /// </summary>
  internal enum CellAgg
  {
    /// <summary>Non-numeric column (e.g. a move or FEN); blank in the summary row.</summary>
    None,

    /// <summary>Summary cell is the average of the per-position numeric values.</summary>
    Average,

    /// <summary>
    /// Summary cell is a single converged value (used by the running cumulative-average
    /// "score" columns, which are rendered from the authoritative accumulators).
    /// </summary>
    Final
  }


  /// <summary>
  /// One console table cell, capturing both the exact display string emitted per position
  /// and the metadata needed to render the corresponding cell in the final summary row.
  /// </summary>
  internal readonly struct Cell
  {
    /// <summary>Column id (header).</summary>
    public readonly string Id;

    /// <summary>Column width (passed to the Writer; -1 means unbounded/raw).</summary>
    public readonly int Width;

    /// <summary>Exact string emitted for this cell in the per-position row.</summary>
    public readonly string Display;

    /// <summary>How this column is aggregated into the summary row.</summary>
    public readonly CellAgg Agg;

    /// <summary>Numeric payload used when Agg == Average (ignored otherwise).</summary>
    public readonly double Value;

    /// <summary>
    /// Composite format string (including any literal prefix/suffix) used to render the
    /// summary cell so that its width matches the per-position cell exactly.
    /// </summary>
    public readonly string SummaryFormat;

    public Cell(string id, int width, string display, CellAgg agg,
                double value = 0, string summaryFormat = null)
    {
      Id = id;
      Width = width;
      Display = display;
      Agg = agg;
      Value = value;
      SummaryFormat = summaryFormat;
    }
  }


  /// <summary>
  /// Accumulates per-column statistics across all suite positions so that a single averaged
  /// summary row (aligned under the existing columns) can be emitted at the end of the run.
  ///
  /// A full row's cell list is merged atomically (caller holds the lock); the first merged
  /// row establishes the canonical ordered set of columns. This is safe because the present
  /// column set is constant for a given run (the external engine and second Ceres engine are
  /// configured once, so the ex/c2 conditional columns never change between rows).
  /// </summary>
  internal sealed class ColumnAccumulator
  {
    readonly List<(string Id, int Width, CellAgg Agg, string SummaryFormat)> order = new();
    readonly Dictionary<string, double> sumById = new();
    readonly Dictionary<string, int> countById = new();
    bool orderEstablished = false;

    /// <summary>Number of rows (positions) merged.</summary>
    public int RowCount { get; private set; }

    /// <summary>The canonical ordered column metadata (from the first merged row).</summary>
    public IReadOnlyList<(string Id, int Width, CellAgg Agg, string SummaryFormat)> Order => order;

    /// <summary>
    /// Merges one position's complete ordered cell list. Caller must hold the lock.
    /// </summary>
    public void Merge(IReadOnlyList<Cell> cells)
    {
      if (!orderEstablished)
      {
        foreach (Cell c in cells)
        {
          order.Add((c.Id, c.Width, c.Agg, c.SummaryFormat));
        }
        orderEstablished = true;
      }

      foreach (Cell c in cells)
      {
        if (c.Agg == CellAgg.Average && !double.IsNaN(c.Value))
        {
          sumById.TryGetValue(c.Id, out double s);
          countById.TryGetValue(c.Id, out int n);
          sumById[c.Id] = s + c.Value;
          countById[c.Id] = n + 1;
        }
      }

      RowCount++;
    }

    /// <summary>
    /// Returns the mean of the per-position values for a column, or NaN if the column was
    /// never seen or had no values (e.g. an absent external-engine or second-Ceres column).
    /// </summary>
    public double Average(string id)
      => countById.TryGetValue(id, out int n) && n > 0 ? sumById[id] / n : double.NaN;
  }

}
