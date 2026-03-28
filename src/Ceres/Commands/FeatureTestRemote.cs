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
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Remote;

#endregion

namespace Ceres.Commands
{
  /// <summary>
  /// Test driver for NNEvaluatorRemote.
  ///
  /// Spawns a single local server process, connects to it, and runs
  /// correctness and performance tests at various batch sizes.
  /// Tests evaluator caching: second connection reuses the cached evaluator.
  ///
  /// Usage: Ceres TEST_REMOTE [net=~T81] [device=GPU:0] [port=50055]
  /// </summary>
  public static class FeatureTestRemote
  {
    // Batch sizes to test.
    static readonly int[] TEST_BATCH_SIZES = { 1, 4, 16, 64, 128, 256, 512 };

    // Number of warmup iterations before timing.
    const int NUM_WARMUP = 3;

    // Number of timed iterations per batch size.
    const int NUM_TIMED_RUNS = 8;


    /// <summary>
    /// Entry point: parses arguments and runs the full test suite.
    /// </summary>
    public static void ParseAndExecute(string args)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args,
        new System.Collections.Generic.List<string>
        {
          "NET", "NET2", "DEVICE", "PORT"
        });

      string net1 = keys.GetValueOrDefault("NET", "~T81", false);
      string net2 = keys.GetValueOrDefault("NET2", "Ceres:C1-640-34-I8", false);
      string device = keys.GetValueOrDefault("DEVICE", "GPU:0#TensorRTNative", false);
      string portStr = keys.GetValueOrDefault("PORT", NNRemoteProtocol.DEFAULT_PORT.ToString(), false);
      int port = int.Parse(portStr);

      Console.WriteLine();
      Console.WriteLine("================================================================");
      Console.WriteLine("  NNEvaluatorRemote Test Driver");
      Console.WriteLine("================================================================");
      Console.WriteLine($"  Host:     {Environment.MachineName} (localhost)");
      Console.WriteLine($"  Device:   {device}");
      Console.WriteLine($"  Port:     {port}");
      Console.WriteLine($"  Network1: {net1} (LC0)");
      Console.WriteLine($"  Network2: {net2} (Ceres/TPG)");
      Console.WriteLine("================================================================");
      Console.WriteLine();

      // Step 1: Spawn a single server process (handles all networks via caching).
      Console.Write("  Starting server process... ");
      Process serverProcess = StartServerProcess(device, port);
      if (serverProcess == null)
      {
        Console.WriteLine("FAILED to start server process.");
        return;
      }
      Console.WriteLine($"OK (PID {serverProcess.Id})");

      Console.Write("  Waiting for server startup... ");
      Thread.Sleep(2000);
      Console.WriteLine("OK");
      Console.WriteLine();

      // Step 2: Test each network against the same server.
      TestNetwork(net1, device, port);
      TestNetwork(net2, device, port);

      // Step 3: Cleanup.
      Console.WriteLine();
      try
      {
        serverProcess.Kill();
        serverProcess.WaitForExit(5000);
      }
      catch { }
      Console.WriteLine("  Server process terminated.");

      Console.WriteLine();
      Console.WriteLine("================================================================");
      Console.WriteLine("  All tests completed.");
      Console.WriteLine("================================================================");
    }


    /// <summary>
    /// Tests a single network: connects, runs correctness, performance, and cache reuse tests.
    /// </summary>
    static void TestNetwork(string netSpec, string deviceSpec, int port)
    {
      Console.WriteLine($"--------------------------------------------------------------");
      Console.WriteLine($"  Testing network: {netSpec}");
      Console.WriteLine($"--------------------------------------------------------------");
      Console.WriteLine();

      // Step 1: Build a local evaluator for reference (correctness checks).
      Console.Write($"  Building local evaluator for reference... ");
      NNEvaluator localEvaluator;
      try
      {
        localEvaluator = NNEvaluator.FromSpecification(netSpec, deviceSpec);
        Console.WriteLine("OK");
      }
      catch (Exception ex)
      {
        Console.WriteLine($"FAILED: {ex.Message}");
        Console.WriteLine($"  Skipping network {netSpec}.");
        Console.WriteLine();
        return;
      }

      // Step 2: Connect remote evaluator (first connection -- will build evaluator on server).
      Console.Write("  Connecting remote evaluator (1st connection, cold)... ");
      Stopwatch connectSW = Stopwatch.StartNew();
      NNEvaluatorRemote remoteEvaluator;
      try
      {
        remoteEvaluator = new NNEvaluatorRemote(
          "localhost", port, netSpec, deviceSpec,
          useCompression: true, maxBatchSize: 1024);
        connectSW.Stop();
        Console.WriteLine($"OK ({connectSW.ElapsedMilliseconds} ms)");
      }
      catch (Exception ex)
      {
        Console.WriteLine($"FAILED: {ex.Message}");
        localEvaluator.Shutdown();
        return;
      }

      Console.WriteLine($"  Remote capabilities: WDL={remoteEvaluator.IsWDL}, M={remoteEvaluator.HasM}, " +
                         $"Action={remoteEvaluator.HasAction}, UncV={remoteEvaluator.HasUncertaintyV}");
      Console.WriteLine();

      // Step 3: Correctness test.
      Console.WriteLine("  --- Correctness Test ---");
      RunCorrectnessTest(localEvaluator, remoteEvaluator);
      Console.WriteLine();

      // Step 4: Performance test.
      Console.WriteLine("  --- Performance Test (compressed) ---");
      RunPerformanceTest(localEvaluator, remoteEvaluator);
      Console.WriteLine();

      // Step 5: Test without compression.
      Console.Write("  Connecting without compression (2nd connection)... ");
      connectSW.Restart();
      NNEvaluatorRemote remoteNoCompress;
      try
      {
        remoteNoCompress = new NNEvaluatorRemote(
          "localhost", port, netSpec, deviceSpec,
          useCompression: false, maxBatchSize: 1024);
        connectSW.Stop();
        Console.WriteLine($"OK ({connectSW.ElapsedMilliseconds} ms)");
        Console.WriteLine();
        Console.WriteLine("  --- Performance Test (no compression) ---");
        RunPerformanceTest(localEvaluator, remoteNoCompress);
        remoteNoCompress.Shutdown();
      }
      catch (Exception ex)
      {
        Console.WriteLine($"FAILED: {ex.Message}");
      }
      Console.WriteLine();

      // Step 6: Cache reuse test -- disconnect and reconnect.
      // The server should reuse the cached evaluator, so connection should be nearly instant.
      Console.WriteLine("  --- Evaluator Cache Reuse Test ---");
      remoteEvaluator.Shutdown();
      Thread.Sleep(500); // Brief pause to let server process the disconnect.

      Console.Write("  Reconnecting (3rd connection, should reuse cached evaluator)... ");
      connectSW.Restart();
      try
      {
        NNEvaluatorRemote remoteReconnect = new NNEvaluatorRemote(
          "localhost", port, netSpec, deviceSpec,
          useCompression: true, maxBatchSize: 1024);
        connectSW.Stop();
        long reconnectMs = connectSW.ElapsedMilliseconds;
        Console.WriteLine($"OK ({reconnectMs} ms)");

        // Quick sanity eval to verify the cached evaluator works.
        EncodedPositionBatchFlat batch = NNEvaluatorBenchmark.MakeTestBatch(remoteReconnect, 16);
        remoteReconnect.EvaluateIntoBuffers(batch);

        if (reconnectMs < 500)
        {
          Console.WriteLine($"    PASS: Reconnection took {reconnectMs} ms (evaluator was cached on server)");
        }
        else
        {
          Console.WriteLine($"    WARN: Reconnection took {reconnectMs} ms (expected < 500 ms for cached evaluator)");
        }

        remoteReconnect.Shutdown();
      }
      catch (Exception ex)
      {
        Console.WriteLine($"FAILED: {ex.Message}");
      }
      Console.WriteLine();

      // Cleanup local evaluator.
      localEvaluator.Shutdown();
    }


    /// <summary>
    /// Verifies that remote evaluation produces identical results to local evaluation.
    /// </summary>
    static void RunCorrectnessTest(NNEvaluator localEval, NNEvaluator remoteEval)
    {
      string[] testFENs =
      {
        Position.StartPosition.FEN,
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkbnr/pppppppp/2n5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2",
        "r1bqkb1r/pppppppp/2n2n2/8/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
        "rnbqkb1r/pp2pppp/5n2/2ppP3/8/2N5/PPPP1PPP/R1BQKBNR w KQkq d6 0 4",
        "r3k2r/ppp2ppp/2n1bn2/2bpp3/4P3/2NP1N2/PPP1BPPP/R1BQK2R w KQkq - 4 7",
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
        "1k1r2r1/1p3p2/p2pbb2/2q4p/1PP1P3/5N1P/P2QB1P1/3R1R1K b - - 0 30",
      };

      int batchSize = testFENs.Length;
      EncodedPositionBatchFlat batch = MakeTestBatchFromFENs(localEval, testFENs);
      NNEvaluatorResult[] localResults = localEval.EvaluateBatch(batch);

      EncodedPositionBatchFlat remoteBatch = MakeTestBatchFromFENs(remoteEval, testFENs);
      NNEvaluatorResult[] remoteResults = remoteEval.EvaluateBatch(remoteBatch);

      bool allMatch = true;
      float maxValueDiff = 0;
      float maxPolicyDiff = 0;

      for (int i = 0; i < batchSize; i++)
      {
        float vDiff = MathF.Abs(localResults[i].V - remoteResults[i].V);
        maxValueDiff = MathF.Max(maxValueDiff, vDiff);

        float wDiff = MathF.Abs(localResults[i].W - remoteResults[i].W);
        float lDiff = MathF.Abs(localResults[i].L - remoteResults[i].L);

        if (vDiff > 0.01f || wDiff > 0.01f || lDiff > 0.01f)
        {
          Console.WriteLine($"    MISMATCH at position {i}: local V={localResults[i].V:F4}, remote V={remoteResults[i].V:F4}, diff={vDiff:F6}");
          allMatch = false;
        }

        var localIndices = localResults[i].Policy.MoveIndicesSpan;
        var remoteIndices = remoteResults[i].Policy.MoveIndicesSpan;
        var localProbs = localResults[i].Policy.ProbabilitiesSpan;
        var remoteProbs = remoteResults[i].Policy.ProbabilitiesSpan;

        for (int j = 0; j < CompressedPolicyVector.NUM_MOVE_SLOTS; j++)
        {
          if (localIndices[j] != remoteIndices[j])
          {
            Console.WriteLine($"    POLICY INDEX MISMATCH at pos {i}, slot {j}: local={localIndices[j]}, remote={remoteIndices[j]}");
            allMatch = false;
            break;
          }
          float pDiff = MathF.Abs(
            CompressedPolicyVector.DecodedProbability(localProbs[j]) -
            CompressedPolicyVector.DecodedProbability(remoteProbs[j]));
          maxPolicyDiff = MathF.Max(maxPolicyDiff, pDiff);
        }
      }

      if (allMatch)
      {
        Console.WriteLine($"    PASS: All {batchSize} positions match (max V diff={maxValueDiff:F6}, max policy diff={maxPolicyDiff:F6})");
      }
      else
      {
        Console.WriteLine($"    FAIL: Some positions had mismatches");
      }
    }


    /// <summary>
    /// Runs performance benchmarks at various batch sizes, comparing local vs remote.
    /// </summary>
    static void RunPerformanceTest(NNEvaluator localEval, NNEvaluator remoteEval)
    {
      Console.WriteLine($"    {"Batch",6} {"Local ms",10} {"Remote ms",10} {"Overhead ms",12} {"Overhead %",11} {"Local NPS",10} {"Remote NPS",11}");
      Console.WriteLine($"    {"-----",6} {"--------",10} {"---------",10} {"-----------",12} {"----------",11} {"---------",10} {"----------",11}");

      foreach (int batchSize in TEST_BATCH_SIZES)
      {
        if (batchSize > localEval.MaxBatchSize || batchSize > remoteEval.MaxBatchSize)
        {
          continue;
        }

        EncodedPositionBatchFlat localBatch = NNEvaluatorBenchmark.MakeTestBatch(localEval, batchSize);
        EncodedPositionBatchFlat remoteBatch = NNEvaluatorBenchmark.MakeTestBatch(remoteEval, batchSize);

        // Warmup.
        for (int w = 0; w < NUM_WARMUP; w++)
        {
          localEval.EvaluateIntoBuffers(localBatch);
          remoteEval.EvaluateIntoBuffers(remoteBatch);
        }

        // Time local.
        double[] localTimes = new double[NUM_TIMED_RUNS];
        for (int r = 0; r < NUM_TIMED_RUNS; r++)
        {
          Stopwatch sw = Stopwatch.StartNew();
          localEval.EvaluateIntoBuffers(localBatch);
          sw.Stop();
          localTimes[r] = sw.Elapsed.TotalMilliseconds;
        }

        // Time remote.
        double[] remoteTimes = new double[NUM_TIMED_RUNS];
        for (int r = 0; r < NUM_TIMED_RUNS; r++)
        {
          Stopwatch sw = Stopwatch.StartNew();
          remoteEval.EvaluateIntoBuffers(remoteBatch);
          sw.Stop();
          remoteTimes[r] = sw.Elapsed.TotalMilliseconds;
        }

        Array.Sort(localTimes);
        Array.Sort(remoteTimes);
        double localMedian = localTimes[NUM_TIMED_RUNS / 2];
        double remoteMedian = remoteTimes[NUM_TIMED_RUNS / 2];
        double overheadMs = remoteMedian - localMedian;
        double overheadPct = (overheadMs / localMedian) * 100.0;
        double localNPS = batchSize / (localMedian / 1000.0);
        double remoteNPS = batchSize / (remoteMedian / 1000.0);

        Console.WriteLine($"    {batchSize,6} {localMedian,10:F3} {remoteMedian,10:F3} {overheadMs,12:F3} {overheadPct,10:F1}% {localNPS,10:N0} {remoteNPS,10:N0}");
      }
    }


    /// <summary>
    /// Creates a test batch from an array of FEN strings.
    /// </summary>
    static EncodedPositionBatchFlat MakeTestBatchFromFENs(NNEvaluator evaluator, string[] fens)
    {
      int count = fens.Length;
      EncodedPositionWithHistory[] positions = new EncodedPositionWithHistory[count];

      for (int i = 0; i < count; i++)
      {
        Position pos = Position.FromFEN(fens[i]);
        positions[i] = EncodedPositionWithHistory.FromPosition(pos);
      }

      bool hasPositions = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Positions);
      bool hasMoves = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Moves);
      bool hasHashes = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Hashes);

      EncodedPositionBatchFlat batch = new EncodedPositionBatchFlat(positions, count, hasPositions);

      if (hasPositions) batch.Positions = new MGPosition[count];
      if (hasHashes) batch.PositionHashes = new ulong[count];
      if (hasMoves) batch.Moves = new MGMoveList[count];

      for (int i = 0; i < count; i++)
      {
        Position pos = Position.FromFEN(fens[i]);
        MGPosition mgPos = pos.ToMGPosition;
        if (hasPositions) batch.Positions[i] = mgPos;
        if (hasHashes) batch.PositionHashes[i] = (ulong)i + (ulong)mgPos.GetHashCode();
        if (hasMoves)
        {
          MGMoveList moves = new MGMoveList();
          MGMoveGen.GenerateMoves(in mgPos, moves);
          batch.Moves[i] = moves;
        }
      }

      return batch;
    }


    /// <summary>
    /// Starts a Ceres server process in the background.
    /// The server handles any network via its evaluator cache.
    /// </summary>
    static Process StartServerProcess(string deviceSpec, int port)
    {
      string ceresPath = Process.GetCurrentProcess().MainModule?.FileName;
      if (ceresPath == null)
      {
        Console.WriteLine("    Cannot determine Ceres executable path.");
        return null;
      }

      try
      {
        // Start server without a default network -- clients will specify their own.
        // We pass a dummy NET since the SERVER command requires it, but the client
        // will override with its own network spec during handshake.
        ProcessStartInfo psi = new ProcessStartInfo
        {
          FileName = ceresPath,
          Arguments = $"SERVER NET=NONE DEVICE={deviceSpec} PORT={port}",
          UseShellExecute = false,
          RedirectStandardOutput = true,
          RedirectStandardError = true,
          CreateNoWindow = true
        };

        Process process = Process.Start(psi);

        Task.Run(() =>
        {
          while (!process.HasExited)
          {
            string line = process.StandardOutput.ReadLine();
            if (line != null) Console.WriteLine($"    [SERVER] {line}");
          }
        });

        Task.Run(() =>
        {
          while (!process.HasExited)
          {
            string line = process.StandardError.ReadLine();
            if (line != null) Console.WriteLine($"    [SERVER-ERR] {line}");
          }
        });

        return process;
      }
      catch (Exception ex)
      {
        Console.WriteLine($"    Failed to start server: {ex.Message}");
        return null;
      }
    }
  }
}
