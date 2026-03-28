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
  /// Client-only test for NNEvaluatorRemote against an already-running server.
  /// Tests multiple networks, batch sizes, compression modes, and cache reuse.
  ///
  /// Usage: Ceres TEST_REMOTE_CLIENT SERVER=spark2-fast [PORT=50055]
  ///        [NET=~T81] [NET2=Ceres:C1-640-34-I8] [DEVICE=GPU:0#TensorRTNative]
  /// </summary>
  public static class FeatureTestRemoteClient
  {
    static readonly int[] BATCH_SIZES = { 1, 4, 16, 64, 128, 256, 512 };
    const int NUM_WARMUP = 3;
    const int NUM_RUNS = 10;


    public static void ParseAndExecute(string args)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args,
        new System.Collections.Generic.List<string>
        {
          "SERVER", "PORT", "NET", "NET2", "DEVICE"
        });

      string server = keys.GetValue("SERVER");
      if (server == null)
      {
        DispatchCommands.ShowErrorExit("TEST_REMOTE_CLIENT requires SERVER=hostname");
        return;
      }
      string portStr = keys.GetValueOrDefault("PORT", NNRemoteProtocol.DEFAULT_PORT.ToString(), false);
      int port = int.Parse(portStr);
      string net1 = keys.GetValueOrDefault("NET", "~T81", false);
      string net2 = keys.GetValueOrDefault("NET2", "Ceres:C1-640-34-I8", false);
      string device = keys.GetValueOrDefault("DEVICE", "GPU:0#TensorRTNative", false);

      Console.WriteLine();
      Console.WriteLine("================================================================");
      Console.WriteLine("  NNEvaluatorRemote Cross-Machine Test Client");
      Console.WriteLine("================================================================");
      Console.WriteLine($"  Client:   {Environment.MachineName}");
      Console.WriteLine($"  Server:   {server}:{port}");
      Console.WriteLine($"  Device:   {device}");
      Console.WriteLine($"  Network1: {net1}");
      Console.WriteLine($"  Network2: {net2}");
      Console.WriteLine("================================================================");
      Console.WriteLine();

      TestNetwork(net1, device, server, port);
      if (net2 != null && net2.Length > 0)
      {
        TestNetwork(net2, device, server, port);
      }

      Console.WriteLine("================================================================");
      Console.WriteLine("  All cross-machine tests completed.");
      Console.WriteLine("================================================================");
    }


    static void TestNetwork(string netSpec, string deviceSpec, string server, int port)
    {
      Console.WriteLine($"--------------------------------------------------------------");
      Console.WriteLine($"  Network: {netSpec}");
      Console.WriteLine($"  Server:  {server}:{port}");
      Console.WriteLine($"--------------------------------------------------------------");
      Console.WriteLine();

      // 1. Build local evaluator for reference timing.
      Console.Write("  Building local evaluator for reference... ");
      NNEvaluator localEval;
      try
      {
        localEval = NNEvaluator.FromSpecification(netSpec, deviceSpec);
        Console.WriteLine("OK");
      }
      catch (Exception ex)
      {
        Console.WriteLine($"FAILED: {ex.Message}");
        Console.WriteLine("  Will run remote-only tests (no overhead comparison).");
        localEval = null;
      }

      // 2. Cold connection (server builds evaluator if not cached).
      Console.Write("  Connecting remote (may build evaluator on server)... ");
      Stopwatch sw = Stopwatch.StartNew();
      NNEvaluatorRemote remote;
      try
      {
        remote = new NNEvaluatorRemote(server, port, netSpec, deviceSpec,
          useCompression: true, maxBatchSize: 1024);
        sw.Stop();
        Console.WriteLine($"OK ({sw.ElapsedMilliseconds} ms)");
      }
      catch (Exception ex)
      {
        Console.WriteLine($"FAILED: {ex.Message}");
        Console.WriteLine($"  Skipping network {netSpec}.");
        Console.WriteLine();
        localEval?.Shutdown();
        return;
      }

      Console.WriteLine($"  Capabilities: WDL={remote.IsWDL}, M={remote.HasM}, " +
                         $"Action={remote.HasAction}, UncV={remote.HasUncertaintyV}");

      // 3. Quick correctness check.
      Console.Write("  Quick eval check (startpos)... ");
      try
      {
        var batch1 = NNEvaluatorBenchmark.MakeTestBatch(remote, 1);
        var result = remote.EvaluateBatch(batch1);
        Console.WriteLine($"V={result[0].V:F4}, W={result[0].W:F4}, L={result[0].L:F4}, M={result[0].M:F1}");
      }
      catch (Exception ex)
      {
        Console.WriteLine($"FAILED: {ex.Message}");
        remote.Shutdown();
        localEval?.Shutdown();
        return;
      }

      // 4. Performance with compression + overhead comparison.
      Console.WriteLine();
      Console.WriteLine("  --- Performance (compressed) ---");
      RunPerfTestWithOverhead(localEval, remote);

      // 5. No compression.
      Console.Write("\n  Connecting (no compression, cached)... ");
      sw.Restart();
      NNEvaluatorRemote remoteNC;
      try
      {
        remoteNC = new NNEvaluatorRemote(server, port, netSpec, deviceSpec,
          useCompression: false, maxBatchSize: 1024);
        sw.Stop();
        Console.WriteLine($"OK ({sw.ElapsedMilliseconds} ms)");
      }
      catch (Exception ex)
      {
        Console.WriteLine($"FAILED: {ex.Message}");
        remote.Shutdown();
        localEval?.Shutdown();
        return;
      }

      Console.WriteLine();
      Console.WriteLine("  --- Performance (no compression) ---");
      RunPerfTestWithOverhead(localEval, remoteNC);
      remoteNC.Shutdown();

      // 6. Cache reuse test.
      Console.WriteLine();
      Console.WriteLine("  --- Cache Reuse Test ---");
      remote.Shutdown();
      Thread.Sleep(500);

      Console.Write("  Reconnecting (should reuse cached evaluator)... ");
      sw.Restart();
      try
      {
        var remoteR = new NNEvaluatorRemote(server, port, netSpec, deviceSpec,
          useCompression: true, maxBatchSize: 1024);
        sw.Stop();
        long reconnMs = sw.ElapsedMilliseconds;
        Console.WriteLine($"OK ({reconnMs} ms)");

        var batchR = NNEvaluatorBenchmark.MakeTestBatch(remoteR, 16);
        remoteR.EvaluateIntoBuffers(batchR);
        Console.WriteLine($"    {(reconnMs < 500 ? "PASS" : "WARN")}: Reconnect took {reconnMs} ms");
        remoteR.Shutdown();
      }
      catch (Exception ex)
      {
        Console.WriteLine($"FAILED: {ex.Message}");
      }
      Console.WriteLine();

      localEval?.Shutdown();
    }


    static void RunPerfTestWithOverhead(NNEvaluator localEval, NNEvaluator remoteEval)
    {
      bool hasLocal = localEval != null;

      if (hasLocal)
      {
        Console.WriteLine($"    {"Batch",6} {"Local ms",10} {"Remote ms",10} {"Overhead",10} {"% of local",11}");
        Console.WriteLine($"    {"-----",6} {"--------",10} {"---------",10} {"--------",10} {"----------",11}");
      }
      else
      {
        Console.WriteLine($"    {"Batch",6} {"Remote ms",10} {"NPS",10}");
        Console.WriteLine($"    {"-----",6} {"---------",10} {"---",10}");
      }

      foreach (int bs in BATCH_SIZES)
      {
        if (bs > remoteEval.MaxBatchSize) continue;
        if (hasLocal && bs > localEval.MaxBatchSize) continue;

        var remoteBatch = NNEvaluatorBenchmark.MakeTestBatch(remoteEval, bs);
        EncodedPositionBatchFlat localBatch = hasLocal
          ? NNEvaluatorBenchmark.MakeTestBatch(localEval, bs) : null;

        // Warmup.
        for (int w = 0; w < NUM_WARMUP; w++)
        {
          if (hasLocal) localEval.EvaluateIntoBuffers(localBatch);
          remoteEval.EvaluateIntoBuffers(remoteBatch);
        }

        // Time local.
        double localMedian = 0;
        if (hasLocal)
        {
          double[] localTimes = new double[NUM_RUNS];
          for (int r = 0; r < NUM_RUNS; r++)
          {
            var s = Stopwatch.StartNew();
            localEval.EvaluateIntoBuffers(localBatch);
            s.Stop();
            localTimes[r] = s.Elapsed.TotalMilliseconds;
          }
          Array.Sort(localTimes);
          localMedian = localTimes[NUM_RUNS / 2];
        }

        // Time remote.
        double[] remoteTimes = new double[NUM_RUNS];
        for (int r = 0; r < NUM_RUNS; r++)
        {
          var s = Stopwatch.StartNew();
          remoteEval.EvaluateIntoBuffers(remoteBatch);
          s.Stop();
          remoteTimes[r] = s.Elapsed.TotalMilliseconds;
        }
        Array.Sort(remoteTimes);
        double remoteMedian = remoteTimes[NUM_RUNS / 2];

        if (hasLocal)
        {
          double overheadMs = remoteMedian - localMedian;
          double overheadPct = localMedian > 0 ? (overheadMs / localMedian) * 100.0 : 0;
          Console.WriteLine($"    {bs,6} {localMedian,10:F3} {remoteMedian,10:F3} {overheadMs,9:F3} ms {overheadPct,9:F1}%");
        }
        else
        {
          double nps = bs / (remoteMedian / 1000.0);
          Console.WriteLine($"    {bs,6} {remoteMedian,10:F3} {nps,10:N0}");
        }
      }
    }
  }
}
