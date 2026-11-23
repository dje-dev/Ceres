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
using System.Linq;
using System.Threading.Tasks;
using System.Collections.Generic;
using System.Threading;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Remote;
using Ceres.Chess.NNEvaluators.Specifications;
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
    /// Optionally the device slots (parsed from TournamentDef.DeviceSlots) over which the concurrent
    /// games are spread. This is the more general alternative to DeviceIDs: slots may be on remote
    /// hosts, may bundle several GPUs and may each host several concurrent games (see NNDeviceSlots).
    /// </summary>
    public readonly NNDeviceSlots DeviceSlots;

    /// <summary>
    /// Definition of parameters of the tournament.
    /// </summary>
    public TournamentDef Def;

    /// <summary>
    /// After a tournament has run: the number of games completed by each concurrent game thread.
    /// </summary>
    public int[] NumGamesPerThread { get; private set; }

    /// <summary>
    /// After a tournament has run: the number of game threads which failed and were retired before the end.
    /// </summary>
    public int NumThreadsRetired { get; private set; }

    /// <summary>
    /// After a tournament has run: the number of games discarded and replayed because the connection
    /// to a remote evaluation server was disrupted during the game (time-based search limits only).
    /// </summary>
    public int NumGamesReplayedAfterDisruption { get; private set; }


    /// <summary>
    /// Optionally a queue manager object which is used when running
    /// distributed model (running either as a coordinator or worker process).
    /// </summary>
    public TournamentGameQueueManager QueueManager;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="def"></param>
    /// <param name="numConcurrent">number of games to play concurrently; if null then the number of streams
    /// declared by def.DeviceSlots (if specified), otherwise 1</param>
    /// <param name="deviceIDs">optional flat pool of local GPU device IDs over which to spread concurrent games
    /// (the alternative to def.DeviceSlots)</param>
    public TournamentManager(TournamentDef def, int? numConcurrent = null, int [] deviceIDs = null)
    {
      if  (  (def.Engines == null || def.Engines.Length == 0)
          && (def.Player1Def == null || def.Player2Def == null))
      { 
        throw new Exception("Missing engine(s). Assign to def.Player1Def and def.Player2Def or call def.AddEngine multiple times."); 
      }

      if (def.DeviceSlots != null)
      {
        if (deviceIDs != null)
        {
          throw new Exception("Specify either TournamentDef.DeviceSlots or deviceIDs, not both.");
        }
        DeviceSlots = new NNDeviceSlots(def.DeviceSlots);
      }

      // By default play as many concurrent games as streams were declared by the device slots (if any).
      int concurrency = numConcurrent ?? DeviceSlots?.TotalStreams ?? 1;
      if (concurrency < 1)
      {
        throw new ArgumentOutOfRangeException(nameof(numConcurrent), "numConcurrent must be at least 1.");
      }

      if (concurrency > 1 && deviceIDs == null && DeviceSlots == null)
      {
        throw new Exception("Must specify deviceIDs (or TournamentDef.DeviceSlots) if numConcurrent > 1.");
      }
      
      Def = def;
      NumConcurrent = Math.Min(concurrency, def.NumGamePairs ?? int.MaxValue);
      DeviceIDs = deviceIDs;

      // Turn off showing game moves if running parallel,
      // since the moves of various games would be intermixed.
      if (concurrency > 1)
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

    /// <summary>
    /// Constructor which spreads the concurrent games over the specified device slots
    /// (equivalent to assigning def.DeviceSlots), for example:
    ///   new TournamentManager(def, 6, "GPU:0+GPU:(0,1)*2@HOP+GPU:3*3@station")
    /// </summary>
    /// <param name="def"></param>
    /// <param name="numConcurrent">number of games to play concurrently, or null for the number of streams declared by the slots</param>
    /// <param name="deviceSlots">device slots specification (see NNDeviceSlots)</param>
    public TournamentManager(TournamentDef def, int? numConcurrent, string deviceSlots)
      : this(WithDeviceSlots(def, deviceSlots), numConcurrent, (int[])null)
    {
    }

    static TournamentDef WithDeviceSlots(TournamentDef def, string deviceSlots)
    {
      ArgumentNullException.ThrowIfNull(def);
      def.DeviceSlots = deviceSlots ?? throw new ArgumentNullException(nameof(deviceSlots));
      return def;
    }


    int numGamePairsLaunched = 0;
    readonly object lockObj = new();

    /// <summary>
    /// Openings which were handed to a game thread that subsequently failed before recording
    /// any game from them. They are handed out again (ahead of new openings) to the next thread asking.
    /// Accessed only under lockObj.
    /// </summary>
    readonly Queue<int> openingsToRetry = new();

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
        // First hand out again any opening which a failed thread returned unplayed.
        if (openingsToRetry.Count > 0)
        {
          return openingsToRetry.Dequeue();
        }

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

    /// <summary>
    /// Outputs a table showing, for each device slot, the number of game streams it hosted,
    /// the games they completed and the search speed achieved. This makes visible how much
    /// each slot contributed (for example whether placing several streams on a GPU paid off,
    /// or whether a remote slot ran materially slower than a local one).
    /// </summary>
    /// <param name="writer"></param>
    /// <param name="gameThreads">the game threads (element i ran on deviceStreams[i])</param>
    /// <param name="deviceStreams"></param>
    void DumpDeviceSlotSummary(TextWriter writer, List<TournamentGameThread> gameThreads, NNDeviceStream[] deviceStreams)
    {
      // The score column (from the perspective of the first engine) is meaningful only with two engines.
      bool showScore = Def.Engines != null && Def.Engines.Length == 2;

      writer.WriteLine();
      writer.WriteLine("Device slot summary (NPS is nodes per second of search time, over all engines):");
      writer.WriteLine($"  {"Slot",-34} {"Streams",7} {"Games",7} "
                     + (showScore ? $"{"Score (+ = -)",-16} " : "")
                     + $"{"NPS total",12} {"NPS/stream",12}");

      foreach (IGrouping<int, int> slotGroup in Enumerable.Range(0, deviceStreams.Length).GroupBy(i => deviceStreams[i].SlotIndex))
      {
        int numStreams = 0;
        float numGames = 0;
        int wins = 0, draws = 0, losses = 0;
        double npsTotal = 0;

        foreach (int i in slotGroup)
        {
          TournamentGameThread thread = gameThreads[i];
          numStreams++;
          numGames += thread.NumGames;
          wins += thread.NumWinsEngine1;
          draws += thread.NumDrawsEngine1;
          losses += thread.NumLossesEngine1;

          // The engines of a stream search alternately, so total nodes over total search time
          // is the average search speed of the stream; concurrent streams add up.
          double seconds = thread.TotalTimeEngine1 + thread.TotalTimeEngine2;
          if (seconds > 0)
          {
            npsTotal += (thread.TotalNodesEngine1 + thread.TotalNodesEngine2) / seconds;
          }
        }

        string scoreStr = showScore ? $"{$"+{wins} ={draws} -{losses}",-16} " : "";
        string retiredStr = slotGroup.Any(i => gameThreads[i].WasRetired) ? "  (stream retired after failure)" : "";
        writer.WriteLine($"  {deviceStreams[slotGroup.First()].Slot.ToString(),-34} {numStreams,7} {numGames,7:F0} "
                       + scoreStr
                       + $"{npsTotal,12:N0} {npsTotal / numStreams,12:N0}{retiredStr}");
      }
      writer.WriteLine();
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

      // Install Ctrl-C handler to allow ad hoc clean termination of tournament (with stats).
      // Scoped registration: unsubscribed on every exit path from this method, so a completed
      // tournament is not kept reachable forever via the static Console.CancelKeyPress event.
      using ConsoleCancelHandlerScope ctrlCScope = !enableCancelVialCtrlC ? null
        : new ConsoleCancelHandlerScope((object sender, ConsoleCancelEventArgs args) =>
        {
          Console.WriteLine("Tournament pending shutdown....");
          Def.parentDef.ShouldShutDown = true;
          // Release any threads parked by a Ctrl-P pause so they observe the shutdown and drain.
          Def.parentDef.PauseController?.Resume();
          shutdownComplete.WaitOne();
        });

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

      static void Warn(string message) => ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, message);

      // If device slots were specified, determine the stream (slot) on which each concurrent game
      // thread runs and validate up front (before any engine is built) that the engines can run there
      // and that the remote evaluation servers are reachable.
      NNDeviceStream[] deviceStreams = null;
      int priorReconnectPatienceSeconds = NNEvaluatorRemote.ReconnectPatienceSeconds;
      if (DeviceSlots != null)
      {
        deviceStreams = DeviceSlots.StreamOrder(numConcurrent);
        DeviceSlots.Validate(EnginesOf(Def), numConcurrent, Warn);

        if (deviceStreams.Any(s => s.Slot.IsRemote))
        {
          DeviceSlots.VerifyRemoteHostsReachable(numConcurrent);

          // Let a game thread ride out a restart of its server or a network outage
          // (its searches stall meanwhile) rather than fail after the short default period.
          NNEvaluatorRemote.ReconnectPatienceSeconds = Math.Max(priorReconnectPatienceSeconds, Def.RemoteReconnectPatienceSeconds);
        }

        Def.Logger.WriteLine($"  Device streams      : {numConcurrent} concurrent game thread(s) on device slots {DeviceSlots.SpecificationString}");
        Def.Logger.Write(NNDeviceSlots.DescribeStreams(deviceStreams, "    ST#"));
        Def.Logger.WriteLine();
      }

      // Otherwise, if a flat pool of device IDs was specified, validate up front that it can give
      // every concurrent worker its own set of GPUs and determine the per-worker device-slice size.
      int devicesPerWorker = (deviceStreams == null && NumConcurrent > 1 && DeviceIDs != null)
                             ? NNDeviceSlots.DevicesPerWorkerForPool(EnginesOf(Def), DeviceIDs, NumConcurrent, Warn)
                             : 0;

      // First loop instantiats/warms up all the engines
      // (without concurrency due to potential CUDA synchronization conflicts).
      for (int i = 0; i < numConcurrent; i++)
      {
        TournamentDef tournamentDefClone = Def.Clone();

        // Record the effective concurrency (not the NumConcurrent field, which ignores the
        // distributed-worker override above) so each game thread knows whether its engines share
        // their minilog files with the engines of other threads.
        tournamentDefClone.NumConcurrentGameThreads = numConcurrent;

        if (deviceStreams != null)
        {
          // Place the engines of this game thread on its device slot (local or remote).
          NNDeviceStream stream = deviceStreams[i];
          foreach (GameEngineDef engineDef in EnginesOf(tournamentDefClone))
          {
            // No-op for Pooled evaluators and engines without an evaluator.
            engineDef.TrySetDevicesIfNotPooled(stream.Slot, stream.StreamIndex);
          }
          tournamentDefClone.DeviceStream = stream;
        }
        else if (NumConcurrent > 1)
        {
          // Make sure the threads will use either different or pooled evaluators:
          // each worker is assigned a slice of the flat pool of device IDs (if any).
          int[] deviceSlice = NNDeviceSlots.DeviceSliceForWorker(DeviceIDs, i, devicesPerWorker);
          if (deviceSlice != null)
          {
            foreach (GameEngineDef engineDef in EnginesOf(tournamentDefClone))
            {
              // No-op for Pooled evaluators and engines without an evaluator.
              engineDef.TrySetDeviceIndicesIfNotPooled(deviceSlice);
            }
          }
        }

        if (NumConcurrent > 1)
        {
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

      NNEvaluatorRemote.ReconnectPatienceSeconds = priorReconnectPatienceSeconds;

      NumGamesPerThread = gameThreads.Select(t => (int)t.NumGames).ToArray();
      NumThreadsRetired = gameThreads.Count(t => t.WasRetired);
      NumGamesReplayedAfterDisruption = gameThreads.Sum(t => t.NumGamesReplayedAfterDisruption);

      parentTest.DumpTournamentSummary(Def);

      if (deviceStreams != null)
      {
        DumpDeviceSlotSummary(Def.Logger, gameThreads, deviceStreams);
      }

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
