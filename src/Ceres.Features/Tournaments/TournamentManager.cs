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
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.UserSettings;
using Ceres.Features.Tournaments.Streaming;
using Ceres.MCGS.GameEngines;
#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Manages execution of a tournament (match) of games
  /// of one engine against another, possibly multithreaded.
  /// </summary>
  public partial class TournamentManager
  {
    /// <summary>
    /// Number of touranment games to play in parallel.
    /// </summary>
    public readonly int NumConcurrent = 1;

    /// <summary>
    ///  Optionally a pool of device IDs partitioned into consecutive chunks, one per
    ///  concurrent worker (chunk size = number of devices each engine's evaluator spans).
    ///  If the pool is smaller than NumConcurrent * devices-per-engine, IDs are reused
    ///  cyclically (deliberate GPU oversubscription, workers share devices; a warning is emitted).
    /// </summary>
    public int[] DeviceIDs = null;

    /// <summary>
    /// Definition of parameters of the tournament.
    /// </summary>
    public TournamentDef Def;


    /// <summary>
    /// Optionally a queue manager object which is used when running
    /// distributed model (running either as a coordinator or worker process).
    /// </summary>
    public TournamentGameQueueManager QueueManager;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="def"></param>
    /// <param name="numConcurrent"></param>
    /// <param name="deviceIDs"></param>
    public TournamentManager(TournamentDef def, int numConcurrent = 1, int [] deviceIDs = null)
    {
      if  (  (def.Engines == null || def.Engines.Length == 0)
          && (def.Player1Def == null || def.Player2Def == null))
      { 
        throw new Exception("Missing engine(s). Assign to def.Player1Def and def.Player2Def or call def.AddEngine multiple times."); 
      }

      if (numConcurrent > 1 && deviceIDs == null)
      {
        throw new Exception("Must specify deviceIDs if numConcurrent > 1.");
      }
      
      Def = def;
      NumConcurrent = Math.Min(numConcurrent, def.NumGamePairs ?? int.MaxValue);
      DeviceIDs = deviceIDs;

      // Turn off showing game moves if running parallel,
      // since the moves of various games would be intermixed.
      if (numConcurrent > 1)
      {
        def.ShowGameMoves = false;
      }

#if NOT
      if (def.EvaluatorDef2.DeviceCombo == NNEvaluators.Defs.NNEvaluatorDeviceComboType.Pooled)
      {
        // If already pooled then having separate GPUs does not make sense
        SeparateGPUs = false;
      }
#endif
    }

    int numGamePairsLaunched = 0;
    readonly object lockObj = new();

    int[] shuffledOpeningIndices = null;
    int[] randomizedOpeningIndices = null;


    /// <summary>
    /// Method called by threads to get the next available opening to be played.
    /// Returns an opening index, or -1 when the requested number of game pairs has been reached.
    /// Throws if NumGamePairs exceeds available openings (repeating openings is not supported).
    /// </summary>
    int GetNextOpeningIndexForLocalThread(int numOpeningsAvailable, int maxOpenings)
    {
      if (numOpeningsAvailable <= 0)
      {
        throw new ArgumentException("No openings available.");
      }

      if (maxOpenings > numOpeningsAvailable)
      {
        throw new Exception($"NumGamePairs ({maxOpenings}) exceeds available openings ({numOpeningsAvailable}). "
                          + "Reduce NumGamePairs or provide a larger opening book.");
      }

      lock (lockObj)
      {
        if (numGamePairsLaunched >= maxOpenings)
        {
          return -1;
        }

        int pairIndex = numGamePairsLaunched++;

        switch (Def.OpeningRandomization)
        {
          case OpeningRandomizationEnum.None:
            return pairIndex;

          case OpeningRandomizationEnum.ShuffleDeterministic:
            if (shuffledOpeningIndices == null)
            {
              shuffledOpeningIndices = new int[numOpeningsAvailable];
              for (int i = 0; i < numOpeningsAvailable; i++)
              {
                shuffledOpeningIndices[i] = i;
              }

              Random rng = new Random(0); // Use fixed seed for deterministic shuffle
              int n = shuffledOpeningIndices.Length;
              while (n > 1)
              {
                n--;
                int k = rng.Next(n + 1);
                (shuffledOpeningIndices[k], shuffledOpeningIndices[n]) = (shuffledOpeningIndices[n], shuffledOpeningIndices[k]);
              }
            }
            return shuffledOpeningIndices[pairIndex];

          case OpeningRandomizationEnum.Randomize:
            if (randomizedOpeningIndices == null)
            {
              randomizedOpeningIndices = new int[numOpeningsAvailable];
              for (int i = 0; i < numOpeningsAvailable; i++)
              {
                randomizedOpeningIndices[i] = i;
              }

              Random rng = new Random(); // Non-deterministic seed
              int n = randomizedOpeningIndices.Length;
              while (n > 1)
              {
                n--;
                int k = rng.Next(n + 1);
                (randomizedOpeningIndices[k], randomizedOpeningIndices[n]) = (randomizedOpeningIndices[n], randomizedOpeningIndices[k]);
              }
            }
            return randomizedOpeningIndices[pairIndex];

          default:
            throw new Exception($"Unknown OpeningRandomization mode: {Def.OpeningRandomization}");
        }
      }
    }


    /// <summary>
    /// If running multiple match threads then the neural network evaluators must be suitable
    /// for parallel execution, either by:
    ///   - being of type Pooled, so that all evaluator definitions reference the same underlying
    ///     evaluator which pools positions from many threads into one batch, or
    ///   - each concurrent worker being assigned its own (disjoint) set of GPUs.
    ///
    /// This method handles the latter case: the DeviceIDs sequence is treated as a flat pool of
    /// GPU IDs partitioned into disjoint consecutive chunks, one per worker, of length equal to
    /// the number of devices each engine's evaluator spans (e.g. pool [0,1,2,3] with two workers
    /// each spanning two GPUs => worker 0 on [0,1], worker 1 on [2,3]). Every engine assigned a
    /// slice (in-process Ceres engines and external NN engines such as LC0 alike) must reference
    /// the same number of devices; Pooled evaluators and engines without an evaluator are left
    /// unchanged. If the pool is smaller than NumConcurrent * devices-per-worker the IDs wrap
    /// around cyclically, so workers share GPUs (deliberate oversubscription; warned, not fatal).
    /// </summary>
    /// <param name="def"></param>
    /// <param name="workerIndex"></param>
    /// <param name="devicesPerWorker"></param>
    void AssignWorkerDevicesIfNotPooled(TournamentDef def, int workerIndex, int devicesPerWorker)
    {
      if (DeviceIDs == null || devicesPerWorker < 1)
      {
        return;
      }

      int[] deviceSlice = new int[devicesPerWorker];
      for (int j = 0; j < devicesPerWorker; j++)
      {
        deviceSlice[j] = DeviceIDs[(workerIndex * devicesPerWorker + j) % DeviceIDs.Length];
      }

      foreach (GameEngineDef engineDef in EnginesOf(def))
      {
        // No-op for Pooled evaluators and engines without an evaluator.
        engineDef.TrySetDeviceIndicesIfNotPooled(deviceSlice);
      }
    }


    /// <summary>
    /// Determines the number of devices per worker (the device count of the engines that will be
    /// assigned a slice, which must all match). Returns 0 if no engine is assigned a slice (all
    /// engines are Pooled or have no evaluator), in which case no pool-based partitioning is done.
    ///
    /// Throws if the assigned engines reference differing device counts. If the DeviceIDs pool is
    /// smaller than NumConcurrent * devices-per-worker only a warning is emitted (device IDs are
    /// then reused cyclically, i.e. deliberate GPU oversubscription across workers).
    /// </summary>
    int ComputeAndValidateDevicePartitioning()
    {
      int devicesPerWorker = 0;

      foreach (GameEngineDef engineDef in EnginesOf(Def))
      {
        Chess.NNEvaluators.Defs.NNEvaluatorDef evalDef = engineDef.GetEvaluatorDef();
        if (evalDef == null || evalDef.DeviceCombo == Chess.NNEvaluators.Defs.NNEvaluatorDeviceComboType.Pooled)
        {
          continue; // no evaluator (e.g. an external non-NN engine), or Pooled (shared) -> not sliced
        }

        int n = evalDef.NumDevices;
        if (n < 1)
        {
          throw new Exception($"Engine {engineDef.ID} specification must reference at least one device.");
        }

        if (devicesPerWorker == 0)
        {
          devicesPerWorker = n;
        }
        else if (n != devicesPerWorker)
        {
          throw new Exception($"All engines assigned GPUs from DeviceIDs must reference the same number "
                            + $"of devices (found both {devicesPerWorker} and {n}).");
        }
      }

      if (devicesPerWorker > 0 && DeviceIDs.Length < NumConcurrent * devicesPerWorker)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
            $"WARNING: DeviceIDs ({DeviceIDs.Length}) fewer than NumConcurrent ({NumConcurrent}) "
          + $"* devices-per-engine ({devicesPerWorker}) = {NumConcurrent * devicesPerWorker}; "
          + $"device IDs will be reused cyclically so concurrent workers share GPU(s) "
          + $"(deliberate oversubscription; per-engine timing comparability may be reduced).");
      }

      return devicesPerWorker;
    }


    /// <summary>
    /// Enumerates the engine definitions of a tournament (the Engines list, or the
    /// Player1Def/Player2Def pair when no Engines list is populated).
    /// </summary>
    static IEnumerable<GameEngineDef> EnginesOf(TournamentDef def)
    {
      if (def.Engines != null && def.Engines.Length > 0)
      {
        foreach (EnginePlayerDef engine in def.Engines)
        {
          yield return engine.EngineDef;
        }
      }
      else
      {
        yield return def.Player1Def.EngineDef;
        yield return def.Player2Def.EngineDef;
      }
    }

    void VerifyEnginesCompatible()
    {
      if (Def.Engines != null)
      {
        foreach (EnginePlayerDef engine in Def.Engines)
        {
          if (engine.SearchLimit.Type == SearchLimitType.NodesForAllMoves
              && !engine.EngineDef.SupportsNodesPerGameMode)
          {
            throw new Exception($"Requested NodesPerGame mode is not supported by engine: {engine.EngineDef.ID}");
          }
        }
      }

      else
      {
        if (Def.Player1Def.SearchLimit.Type == SearchLimitType.NodesForAllMoves
         && !Def.Player1Def.EngineDef.SupportsNodesPerGameMode)
        {
          throw new Exception($"Requested NodesPerGame mode is not supported by engine 1: {Def.Player1Def.EngineDef.ID}");
        }

        if (Def.Player2Def.SearchLimit.Type == SearchLimitType.NodesForAllMoves
         && !Def.Player2Def.EngineDef.SupportsNodesPerGameMode)
        {
          throw new Exception($"Requested NodesPerGame mode is not supported by engine 2: {Def.Player2Def.EngineDef.ID}");
        }
      }
    }


    internal ManualResetEvent shutdownComplete = new (false);

    /// <summary>
    /// 
    /// </summary>
    /// <param name="queueManager">optional associated queue manager if running in distributed mode</param>
    /// <returns></returns>
    public TournamentResultStats RunTournament(TournamentGameQueueManager queueManager = null, 
                                               bool enableCancelVialCtrlC = true)
    {
      shutdownComplete.Reset();
      Def.ShouldShutDown = false;

      if (Def.Engines.Length > 0)
      {
        foreach (EnginePlayerDef engine in Def.Engines)
        {
          if (engine == null)
          {
            throw new ArgumentNullException("engine is null)");
          }
        }
      }
      else
      {
        if (Def.Player1Def == null)
        {
          throw new ArgumentNullException("Def.Player1Def is null)");
        }
        if (Def.Player2Def == null)
        {
          throw new ArgumentNullException("Def.Player2Def is null)");
        }
      }

      if (Def.OpeningsFileName == null)
      {
        Def.NumGamePairs = 1;
      }

      VerifyEnginesCompatible();

      if (enableCancelVialCtrlC)
      {
        // Install Ctrl-C handler to allow ad hoc clean termination of tournament (with stats).
        ConsoleCancelEventHandler ctrlCHandler = new ConsoleCancelEventHandler((object sender,
          ConsoleCancelEventArgs args) =>
        {
          Console.WriteLine("Tournament pending shutdown....");
          Def.parentDef.ShouldShutDown = true;
          // Release any threads parked by a Ctrl-P pause so they observe the shutdown and drain.
          Def.parentDef.PauseController?.Resume();
          shutdownComplete.WaitOne();
        }); ;
        Console.CancelKeyPress += ctrlCHandler;
      }

      QueueManager = queueManager;
      TournamentResultStats parentTest = new();


      // Show differences between engine 1 and engine 2
      Def.DumpParams(Def.Logger);

      // Prepare to create a set of task threads to run games
      List<Task> tasks = new List<Task>();
      List<TournamentGameThread> gameThreads = new List<TournamentGameThread>();

      Directory.CreateDirectory(CeresUserSettingsManager.Settings.DirCeresOutput);

      // If we are a worker process then run only a single game thread.
      int numConcurrent = queueManager != null && !queueManager.IsCoordinator ? 1 : NumConcurrent;

      // Set up interactive console control (Ctrl-P pause/resume and Ctrl-D dump-info) for local
      // in-process tournaments when stdin is an interactive console. The pause controller must be
      // created before any worker thread starts (threads register with it as they enter their game
      // loop); the key-monitor thread itself is started below, after the worker tasks are launched.
      CancellationTokenSource keyMonitorStop = null;
      bool enableInteractiveKeys = enableCancelVialCtrlC && QueueManager == null && !Console.IsInputRedirected;
      if (enableInteractiveKeys)
      {
        Def.parentDef.PauseController = new TournamentPauseController();
      }

      // Optionally start the live streaming publisher (default-on; see TournamentDef.EnableLiveStreaming),
      // so any remote consumer (e.g. the EngineBattle GUI in REMOTE mode) can connect and watch games live.
      if (Def.parentDef.EnableLiveStreaming && Def.parentDef.Observer == null)
      {
        Def.parentDef.Observer = new TournamentStreamPublisher(Def.parentDef.LiveStreamPort, numConcurrent);
      }
      Def.parentDef.Observer?.OnTournamentStart(DtoMappers.ToTournamentMeta(Def, numConcurrent));

      // Validate up front that the device pool can give every concurrent worker its own
      // disjoint set of GPUs, and determine the per-worker device-slice size.
      int devicesPerWorker = (NumConcurrent > 1 && DeviceIDs != null)
                             ? ComputeAndValidateDevicePartitioning()
                             : 0;

      // First loop instantiats/warms up all the engines
      // (without concurrency due to potential CUDA synchronization conflicts).
      for (int i = 0; i < numConcurrent; i++)
      {
        TournamentDef tournamentDefClone = Def.Clone();

        // Make sure the threads will use either different or pooled evaluators
        if (NumConcurrent > 1)
        {
          AssignWorkerDevicesIfNotPooled(tournamentDefClone, i, devicesPerWorker);

          // Spread instances across different devices and
          // and processor groups to distribute computation.
          tournamentDefClone.ProcessGroupIndex = i;
        }

        TournamentGameThread gameTest = new TournamentGameThread(tournamentDefClone, parentTest);
        gameThreads.Add(gameTest);
      }

      // Second loop creates and launches tasks for each.
      for (int i = 0; i < numConcurrent; i++)
      {
        int threadIndex = i; // Local copy to avoid closure-over-loop-variable bug.
        TournamentGameThread gameTest = gameThreads[i];

        Action action;
        if (QueueManager == null)
        {
          // Everything happens locally, data structures updated as part of processing.
          action = () => ThreadProcLocalWorker(threadIndex, gameTest);
        }
        else if (QueueManager.IsCoordinator)
        {
          // We are coordinator. Repeatedly enqueue request to play game pairs and retireve/show results.
          action = () => ThreadProcCoordinator(threadIndex, gameTest);
        }
        else
        {
          // Worker method (which will forward result data structure back to coordinator).
          action = () => ThreadProcDistributedWorker(threadIndex, gameTest);
        }

        Task thisTask = new Task(action, default, TaskCreationOptions.LongRunning);
        tasks.Add(thisTask);
        thisTask.Start();
      }

      // Now that all worker tasks are running, start the interactive key-monitor (if enabled).
      if (enableInteractiveKeys)
      {
        keyMonitorStop = new CancellationTokenSource();
        StartKeyMonitorThread(gameThreads, parentTest, keyMonitorStop.Token);
      }

      Task.WaitAll(tasks.ToArray());

      // Stop the key monitor and release any parked threads (no-op if not currently paused).
      keyMonitorStop?.Cancel();
      Def.parentDef.PauseController?.Resume();

      Def.parentDef.Observer?.OnTournamentEnd();

      parentTest.DumpTournamentSummary(Def);

      shutdownComplete.Set();

      return parentTest;
    }


    /// <summary>
    /// Starts a background thread that monitors the console for the interactive tournament control
    /// keys: Ctrl-P (pause/resume) and Ctrl-D (dump search info for currently-searching MCGS engines).
    /// Ctrl-C is handled separately (via Console.CancelKeyPress) and is unaffected.
    /// </summary>
    /// <param name="gameThreads">the worker game threads (engines polled for Ctrl-D dumps)</param>
    /// <param name="parentTest">the shared result stats (summary printed on Ctrl-P pause)</param>
    /// <param name="stopToken">cancellation signal driven when the tournament ends</param>
    private void StartKeyMonitorThread(List<TournamentGameThread> gameThreads,
                                       TournamentResultStats parentTest,
                                       CancellationToken stopToken)
    {
      bool paused = false;

      void HandleCtrlP()
      {
        TournamentPauseController controller = Def.parentDef.PauseController;
        if (controller == null)
        {
          return;
        }

        if (!paused)
        {
          Console.WriteLine("Starting pause...");
          // Block until every active worker thread has parked at an end-of-game boundary (or the
          // tournament begins shutting down). This intentionally ignores further key presses until
          // the pause completes.
          bool allParked = controller.RequestPauseAndWaitAllParked(
                             () => Def.parentDef.ShouldShutDown || stopToken.IsCancellationRequested);
          if (allParked)
          {
            // Emit the same summary as at the end of the tournament, then announce the freeze.
            parentTest.DumpTournamentSummary(Def);
            ConsoleUtils.WriteLineColored(ConsoleColor.Green, "Paused...");
            paused = true;
          }
          else
          {
            // Aborted before fully paused (tournament shutting down); release any parked threads.
            controller.Resume();
          }
        }
        else
        {
          controller.Resume();
          Console.WriteLine("Resuming...");
          paused = false;
        }
      }

      void HandleCtrlD()
      {
        // Flag every currently-searching in-process MCGS engine to dump its search diagnostics,
        // labeling the dump with the originating console command (the engine stays agnostic to it).
        // For external Ceres engines accessed over UCI, request a marker-delimited dump-info-block
        // over the UCI channel, capture it, and display it here (mirroring the in-process dump).
        const string DUMP_DESCRIPTION = "Ctrl-D DUMP-INFO";
        foreach (TournamentGameThread t in gameThreads)
        {
          foreach (GameEngine engine in t.Run.Engines)
          {
            RequestDumpInfoForEngine(engine, DUMP_DESCRIPTION);
          }
          RequestDumpInfoForEngine(t.Run.Engine2CheckEngine, DUMP_DESCRIPTION);
        }
      }

      void RequestDumpInfoForEngine(GameEngine engine, string description)
      {
        if (engine is GameEngineCeresMCGSInProcess mcgs && mcgs.IsSearching)
        {
          mcgs.RequestDumpInfo(description);
        }
        else if (engine is GameEngineCeresMCGSUCI uci && uci.IsSearching)
        {
          // Capturing the block requires waiting for the subprocess to respond (up to a few seconds),
          // so run it off the key-monitor thread to keep that thread responsive.
          Task.Run(() =>
          {
            try
            {
              uci.DumpInfoBlockToConsole(description, Console.Out);
            }
            catch (Exception)
            {
              // Never let a diagnostics dump failure disrupt the tournament.
            }
          });
        }
      }

      Thread keyMonitorThread = new Thread(() =>
      {
        while (!stopToken.IsCancellationRequested)
        {
          try
          {
            if (!Console.KeyAvailable)
            {
              Thread.Sleep(40);
              continue;
            }

            ConsoleKeyInfo k = Console.ReadKey(intercept: true);
            bool ctrl = (k.Modifiers & ConsoleModifiers.Control) != 0;
            // Detect by key+modifier, with a fallback on the raw control character because some
            // terminals (notably on Linux) deliver the control char without setting the modifier.
            bool isCtrlP = (ctrl && k.Key == ConsoleKey.P) || k.KeyChar == '\u0010';
            bool isCtrlD = (ctrl && k.Key == ConsoleKey.D) || k.KeyChar == '\u0004';

            if (isCtrlP)
            {
              HandleCtrlP();
            }
            else if (isCtrlD)
            {
              HandleCtrlD();
            }
          }
          catch (Exception)
          {
            // A console quirk (e.g. an input-mode change) -- back off briefly and continue.
            Thread.Sleep(200);
          }
        }
      });
      keyMonitorThread.IsBackground = true;
      keyMonitorThread.Name = "TournamentKeyMonitor";
      keyMonitorThread.Start();
    }

  }
}
