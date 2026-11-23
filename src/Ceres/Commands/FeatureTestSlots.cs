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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators;
using Chess.Ceres.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Remote;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.Positions;
using Ceres.Features.Suites;
using Ceres.Features.Tournaments;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.Commands
{
  /// <summary>
  /// Test driver for device slots (NNDeviceSlots), which spread the concurrent games of a tournament
  /// or the concurrent workers of a suite over local and/or remote GPUs.
  ///
  /// The first group of tests needs no GPU: slot specification parsing, the order in which streams
  /// are handed out, re-targeting of evaluator definitions and validation.
  ///
  /// The live tests (LIVE=true, the default) start a local server process and then run a small tournament
  /// and suites with one local stream and two streams on the "remote" (localhost) server.
  /// With RESTART=true the server is also killed and restarted in the middle of a time-limited
  /// tournament, to exercise reconnection and the replay of disrupted games.
  ///
  /// With LC0NET=... one search of an MCGS engine using that LC0 network is also run twice, with the
  /// network evaluated locally and then on the server, and the results are compared.
  /// BASIC=false skips the tournament and suite tests (to run only the RESTART and/or LC0NET tests).
  ///
  /// Usage: Ceres TEST_SLOTS [net=...] [device=GPU:0#TensorRTNative] [port=50071]
  ///                         [live=true] [basic=true] [restart=false] [nodes=200] [lc0net=...]
  /// </summary>
  public static class FeatureTestSlots
  {
    static int numPassed;
    static int numFailed;

    static void Check(bool condition, string description)
    {
      if (condition)
      {
        numPassed++;
        Console.WriteLine($"    PASS  {description}");
      }
      else
      {
        numFailed++;
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"    FAIL  {description}");
        Console.ResetColor();
      }
    }

    static void CheckThrows(Action action, string description)
    {
      bool threw = false;
      try
      {
        action();
      }
      catch (Exception)
      {
        threw = true;
      }
      Check(threw, description);
    }

    static void Section(string title)
    {
      Console.WriteLine();
      Console.WriteLine("--------------------------------------------------------------");
      Console.WriteLine("  " + title);
      Console.WriteLine("--------------------------------------------------------------");
    }


    /// <summary>
    /// Entry point: parses arguments and runs the tests.
    /// </summary>
    public static void ParseAndExecute(string args)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args,
        new List<string> { "NET", "DEVICE", "PORT", "LIVE", "BASIC", "RESTART", "NODES", "LC0NET" });

      string net = keys.GetValueOrDefault("NET", "Ceres:C1-640-34-I8", false);
      string device = keys.GetValueOrDefault("DEVICE", "GPU:0#TensorRTNative", false);
      int port = int.Parse(keys.GetValueOrDefault("PORT", "50071", false));
      bool live = bool.Parse(keys.GetValueOrDefault("LIVE", "true", false));
      bool basic = bool.Parse(keys.GetValueOrDefault("BASIC", "true", false));
      bool restart = bool.Parse(keys.GetValueOrDefault("RESTART", "false", false));
      int nodes = int.Parse(keys.GetValueOrDefault("NODES", "200", false));
      string lc0Net = keys.GetValueOrDefault("LC0NET", null, false);

      Console.WriteLine();
      Console.WriteLine("================================================================");
      Console.WriteLine("  Device Slots Test Driver");
      Console.WriteLine("================================================================");

      TestParsing();
      TestStreamOrder();
      TestRetargeting();
      TestValidation();

      if (live)
      {
        RunLiveTests(net, device, port, nodes, basic, restart, lc0Net);
      }

      Console.WriteLine();
      Console.WriteLine("================================================================");
      Console.WriteLine($"  Device slots tests completed: {numPassed} passed, {numFailed} failed.");
      Console.WriteLine("================================================================");
    }


    #region Tests which need no GPU

    static void TestParsing()
    {
      Section("Slot specification parsing");

      NNDeviceSlots s = new("GPU:0+GPU:0,1,2@HOP+GPU:0@station");
      Check(s.Slots.Length == 5 && s.TotalStreams == 5, "each listed GPU is one slot (5 slots)");
      Check(string.Join(" ", s.Slots.Select(x => x.ToString())) == "GPU:0 GPU:0@HOP GPU:1@HOP GPU:2@HOP GPU:0@station",
            "slots are local 0; HOP 0, 1, 2; station 0");
      Check(!s.Slots[0].IsRemote && s.Slots[1].IsRemote && s.Slots[1].RemoteHost == "HOP"
         && s.Slots[1].RemotePort == NNRemoteProtocol.DEFAULT_PORT, "remote host and default port are recorded");

      s = new("GPU:0+GPU:(0,1)@HOP+GPU:3@station");
      Check(s.Slots.Length == 3 && s.Slots[1].Devices.Length == 2
         && s.Slots[1].Devices[0].DeviceIndex == 0 && s.Slots[1].Devices[1].DeviceIndex == 1
         && s.Slots[2].Devices[0].DeviceIndex == 3, "parentheses bundle GPUs into one slot");
      Check(s.Slots[1].ToString() == "GPU:(0,1)@HOP", "a bundled slot is displayed with parentheses");

      s = new("GPU:0+GPU:(0,1)*2@HOP+GPU:3*3@station");
      Check(s.Slots.Length == 3 && s.TotalStreams == 6
         && s.Slots.Select(x => x.Multiplicity).SequenceEqual([1, 2, 3]), "*N gives a slot N streams (6 streams in total)");

      s = new("GPU:0,1*2,2@HOP");
      Check(s.Slots.Length == 3 && s.TotalStreams == 4
         && s.Slots.Select(x => x.Multiplicity).SequenceEqual([1, 2, 1]), "*N binds to the GPU immediately before it");

      s = new("GPU:(0,1),2,3@HOP");
      Check(s.Slots.Length == 3 && s.Slots[0].Devices.Length == 2 && s.Slots.All(x => x.RemoteHost == "HOP"),
            "bundles and single GPUs can be mixed in one group");

      s = new("GPU:(0,1)*2#TensorRTNative@HOP/50060");
      Check(s.Slots.Length == 1 && s.Slots[0].Multiplicity == 2 && s.Slots[0].RemotePort == 50060
         && s.Slots[0].Devices.All(d => d.OverrideEngineType == "TensorRTNative"), "#engine and /port apply to the slots of the group");

      s = new(" gpu:0 + GPU:1*2 ");
      Check(s.Slots.Length == 2 && s.TotalStreams == 3 && s.Slots[0].Devices[0].OverrideEngineType == null,
            "whitespace and case are tolerated; engine is unspecified unless stated");

      foreach (string bad in new[] { "", "CPU", "GPU:0:POOLED", "GPU:0=shared", "GPU:(0,1@HOP", "GPU:0)@HOP", "GPU:((0,1))",
                                     "GPU:0*0", "GPU:0*x", "GPU:0@0.5,1@0.5", "GPU:0[256]", "GPU:0,,1", "GPU:a", "0,1" })
      {
        Check(!NNDeviceSlots.IsValid(bad), $"invalid specification is rejected: \"{bad}\"");
      }
    }


    static void TestStreamOrder()
    {
      Section("Order in which streams are handed out");

      NNDeviceSlots s = new("GPU:0+GPU:(0,1)*2@HOP+GPU:3*3@station");

      string Order(int? n) => string.Join(" | ", s.StreamOrder(n).Select(x => x.ToString()));

      Check(Order(null) == "GPU:0 | GPU:(0,1)@HOP | GPU:3@station | GPU:(0,1)@HOP #1 | GPU:3@station #1 | GPU:3@station #2",
            "default concurrency is the declared total, handed out in rounds");
      Check(Order(3) == "GPU:0 | GPU:(0,1)@HOP | GPU:3@station", "concurrency 3: one stream per slot");
      Check(Order(4) == "GPU:0 | GPU:(0,1)@HOP | GPU:3@station | GPU:(0,1)@HOP #1", "concurrency 4: adds the second HOP stream");

      NNDeviceStream[] twelve = s.StreamOrder(12);
      Check(twelve.GroupBy(x => x.SlotIndex).OrderBy(g => g.Key).Select(g => g.Count()).SequenceEqual([2, 4, 6]),
            "concurrency 12: wraps around keeping the declared proportions (2/4/6)");
      Check(twelve.GroupBy(x => x.SlotIndex).All(g => g.Select(x => x.StreamIndex).Distinct().Count() == g.Count()),
            "every stream on a slot has its own stream index, also when wrapped");

      CheckThrows(() => s.StreamOrder(0), "concurrency 0 is rejected");
    }


    static void TestRetargeting()
    {
      Section("Re-targeting an evaluator definition to a slot");

      NNEvaluatorDef MakeDef()
      {
        NNEvaluatorDef def = new NNEvaluatorDef(NNEvaluatorType.Ceres, "dummy_net", NNDeviceType.GPU, 0);
        def.Devices[0].Device.OverrideEngineType = "TensorRTNative";
        def.Devices[0].Device.MaxBatchSize = 512;
        return def;
      }

      NNEvaluatorDef def = MakeDef();
      def.TrySetDevices(new NNDeviceSlots("GPU:(0,1)@HOP/50060").Slots[0], 2);
      Check(def.NumDevices == 2 && def.DeviceCombo == NNEvaluatorDeviceComboType.Split, "the slot determines the number of devices");
      Check(def.Devices.All(d => d.Device.RemoteHost == "HOP" && d.Device.RemotePort == 50060 && d.Device.StreamIndex == 2),
            "host, port and stream index are applied to every device");
      Check(def.Devices.All(d => d.Device.OverrideEngineType == "TensorRTNative" && d.Device.MaxBatchSize == 512),
            "the engine type and batch hints of the definition are kept");
      Check(def.DeviceIndices.SequenceEqual([0, 1]) && def.EqualFractions, "device indices come from the slot, equally weighted");

      string specString = NNDevicesSpecificationString.ToSpecificationString(def.DeviceCombo, def.Devices);
      Check(specString == "Device=GPU:0,1#TensorRTNative@HOP/50060/2", $"device string includes host, port and stream ({specString})");

      NNDevicesSpecificationString reparsed = new(specString.Replace("Device=", ""));
      Check(reparsed.Devices.Count == 2
         && reparsed.Devices.All(d => d.Item1.RemoteHost == "HOP" && d.Item1.RemotePort == 50060 && d.Item1.StreamIndex == 2
                                   && d.Item1.OverrideEngineType == "TensorRTNative"),
            "that device string parses back to the same placement");

      def.TrySetDevices(new NNDeviceSlots("GPU:3").Slots[0], 0);
      Check(def.NumDevices == 1 && def.DeviceCombo == NNEvaluatorDeviceComboType.Single
         && def.Devices[0].Device.RemoteHost == null && def.Devices[0].Device.DeviceIndex == 3
         && def.Devices[0].Device.StreamIndex == 0, "re-targeting to a local slot clears the remote host");

      def = MakeDef();
      def.TrySetDevices(new NNDeviceSlots("GPU:2#ONNXRuntime@HOP").Slots[0], 0);
      Check(def.Devices[0].Device.OverrideEngineType == "ONNXRuntime", "an engine type stated by the slot overrides that of the definition");

      NNEvaluatorDef pooled = new NNEvaluatorDef(new NNEvaluatorNetDef("dummy_net", NNEvaluatorType.Ceres, NNEvaluatorPrecision.FP16),
                                                 NNEvaluatorDeviceComboType.Pooled, null, new NNEvaluatorDeviceDef(NNDeviceType.GPU, 0));
      pooled.TrySetDevices(new NNDeviceSlots("GPU:5@HOP").Slots[0], 1);
      Check(pooled.Devices[0].Device.DeviceIndex == 0 && pooled.Devices[0].Device.RemoteHost == null, "a pooled definition is left untouched");
    }


    /// <summary>
    /// Stand-in for an external engine which has a neural network evaluator but cannot evaluate remotely.
    /// </summary>
    [Serializable]
    sealed class ExternalNNEngineDefForTest : GameEngineDef
    {
      readonly NNEvaluatorDef evaluatorDef;
      public ExternalNNEngineDefForTest(NNEvaluatorDef evaluatorDef) : base("ExternalNN") => this.evaluatorDef = evaluatorDef;
      public override bool SupportsNodesPerGameMode => false;
      public override GameEngine CreateEngine() => throw new NotSupportedException();
      public override NNEvaluatorDef GetEvaluatorDef() => evaluatorDef;
    }


    static void TestValidation()
    {
      Section("Validation before engines are built");

      NNEvaluatorDef evaluatorDef = new NNEvaluatorDef(NNEvaluatorType.Ceres, "dummy_net", NNDeviceType.GPU, 0);
      GameEngineDef ceres = new GameEngineDefCeresMCGS("Ceres", evaluatorDef, new ParamsSearch(), new ParamsSelect());
      GameEngineDef external = new ExternalNNEngineDefForTest(evaluatorDef);

      List<string> warnings = new();

      NNDeviceSlots remoteSlots = new("GPU:0+GPU:1@HOP");
      CheckThrows(() => remoteSlots.Validate([ceres, external], 2, warnings.Add),
                  "an external NN engine on a remote slot is rejected");

      bool ok = true;
      try
      {
        remoteSlots.Validate([ceres], 2, warnings.Add);
        new NNDeviceSlots("GPU:0+GPU:1").Validate([ceres, external], 2, warnings.Add);
      }
      catch (Exception)
      {
        ok = false;
      }
      Check(ok && warnings.Count == 0, "in-process Ceres on remote slots, and external engines on local slots, are accepted");

      warnings.Clear();
      new NNDeviceSlots("GPU:0,(0,1)@HOP").Validate([ceres], 2, warnings.Add);
      Check(warnings.Count == 1 && warnings[0].Contains("appears in two device slots"), "a GPU in two different slots draws a warning");

      warnings.Clear();
      new NNDeviceSlots("GPU:0*2@HOP+GPU:0@HOP/50060").Validate([ceres], 3, warnings.Add);
      Check(warnings.Count == 0, "explicit *N, and the same GPU index on a different server port, draw no warning");

      warnings.Clear();
      new NNDeviceSlots("GPU:0*2").Validate([ceres], 3, warnings.Add);
      Check(warnings.Count == 1 && warnings[0].Contains("exceeds"), "concurrency above the declared streams draws a warning");

      // Nothing is expected to be listening on this port.
      CheckThrows(() => new NNDeviceSlots("GPU:0@localhost/50999").VerifyRemoteHostsReachable(1, 500),
                  "an unreachable remote server is reported before anything is built");
    }

    #endregion


    #region Live tests (local server process, tournament and suites)

    /// <summary>
    /// A local "Ceres SERVER" child process whose output is captured (and echoed).
    /// </summary>
    sealed class LocalServer
    {
      public Process Process;
      public readonly ConcurrentQueue<string> OutputLines = new();
      readonly ManualResetEventSlim listening = new(false);

      public static LocalServer Start(int port)
      {
        string processPath = Environment.ProcessPath;
        string arguments = $"SERVER PORT={port}";

        // If running under the dotnet host (rather than the native launcher) pass the main assembly.
        if (Path.GetFileNameWithoutExtension(processPath).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
        {
          arguments = $"\"{System.Reflection.Assembly.GetEntryAssembly().Location}\" {arguments}";
        }

        LocalServer server = new LocalServer();
        server.Process = Process.Start(new ProcessStartInfo
        {
          FileName = processPath,
          Arguments = arguments,
          UseShellExecute = false,
          RedirectStandardOutput = true,
          RedirectStandardError = true,
          CreateNoWindow = true
        });

        void Pump(StreamReader reader)
        {
          Task.Run(() =>
          {
            string line;
            while ((line = reader.ReadLine()) != null)
            {
              server.OutputLines.Enqueue(line);
              if (line.Contains("Listening for connections"))
              {
                server.listening.Set();
              }
              if (line.Contains("[SERVER]"))
              {
                Console.WriteLine("      " + line);
              }
            }
          });
        }

        Pump(server.Process.StandardOutput);
        Pump(server.Process.StandardError);

        if (!server.listening.Wait(60_000))
        {
          throw new Exception("Server process did not start listening within 60 seconds.");
        }
        return server;
      }

      public void Kill()
      {
        try
        {
          Process.Kill(entireProcessTree: true);
          Process.WaitForExit(10_000);
        }
        catch (Exception)
        {
        }
      }
    }


    static readonly string[] OPENING_FENS =
    [
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
      "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - 0 1",
      "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - 0 1",
      "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - 1 1",
      "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
      "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
      "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
      "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 1 2",
      "rnbqkbnr/pppp1ppp/4p3/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
      "rnbqkbnr/pp1ppppp/2p5/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
      "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
      "rnbqkbnr/pppppp1p/6p1/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2",
    ];

    static readonly string[] SUITE_EPDS =
    [
      "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - bm Qxf7; id \"S1\";",
      "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - bm Rd8; id \"S2\";",
      "k7/8/1K6/8/8/8/8/7R w - - bm Rh8; id \"S3\";",
      "6k1/8/6K1/8/8/8/8/R7 w - - bm Ra8; id \"S4\";",
      "4k3/8/4K3/8/8/8/8/R7 w - - bm Ra8; id \"S5\";",
      "1k6/1P6/1K6/8/8/8/8/7R w - - bm Rh8; id \"S6\";",
      "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - bm Qxf7; id \"S7\";",
      "3k4/8/3K4/8/8/8/8/7R w - - bm Rh8; id \"S8\";",
    ];


    /// <summary>
    /// Positions (not decided at once) used for the suite which is interrupted by a server restart.
    /// </summary>
    static readonly string[] SUITE_RESTART_EPDS =
    [
      "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - bm e5; id \"R1\";",
      "rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq - bm d5; id \"R2\";",
      "rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR b KQkq - bm e5; id \"R3\";",
      "rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R b KQkq - bm d5; id \"R4\";",
      "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - bm Nf3; id \"R5\";",
      "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - bm Nf3; id \"R6\";",
      "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - bm c4; id \"R7\";",
      "rnbqkb1r/pppppppp/5n2/8/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - bm c4; id \"R8\";",
    ];


    static void RunLiveTests(string net, string device, int port, int nodes, bool basic, bool restart, string lc0Net)
    {
      Section($"Live tests: net={net} device={device} server port={port}");

      string workDir = Path.Combine(Path.GetTempPath(), "ceres_test_slots_" + Environment.ProcessId);
      Directory.CreateDirectory(workDir);
      string openingsFileName = Path.Combine(workDir, "openings.epd");
      string suiteFileName = Path.Combine(workDir, "suite.epd");
      string suiteRestartFileName = Path.Combine(workDir, "suite_restart.epd");
      File.WriteAllLines(openingsFileName, OPENING_FENS);
      File.WriteAllLines(suiteFileName, SUITE_EPDS);
      File.WriteAllLines(suiteRestartFileName, SUITE_RESTART_EPDS);

      Console.WriteLine("    Starting local server process...");
      LocalServer server = LocalServer.Start(port);

      // One local stream plus two streams on the "remote" server, all on the same device.
      string deviceIndices = device.Split(':')[1].Split('#')[0];
      string slots = $"GPU:{deviceIndices}+GPU:{deviceIndices}*2@localhost/{port}";

      try
      {
        if (basic)
        {
          TestTournament(net, device, slots, openingsFileName, SearchLimit.NodesPerMove(nodes), server);
          TestSuites(net, device, slots, suiteFileName, SearchLimit.NodesPerMove(nodes));
        }

        if (restart)
        {
          server = TestServerRestart(net, device, deviceIndices, port, openingsFileName, server);
          server = TestSuiteServerRestart(net, device, deviceIndices, port, suiteRestartFileName, server);
        }

        if (lc0Net != null)
        {
          TestLC0(lc0Net, device, port);
        }
      }
      finally
      {
        server.Kill();
        try
        {
          Directory.Delete(workDir, true);
        }
        catch (Exception)
        {
        }
      }
    }


    static TournamentDef MakeTournamentDef(string id, string net, string device, string openingsFileName,
                                           SearchLimit limit, int numGamePairs)
    {
      NNEvaluatorDef evaluatorDef = NNEvaluatorDef.FromSpecification(net, device);
      EnginePlayerDef player1 = new(new GameEngineDefCeresMCGS("Ceres1", evaluatorDef, new ParamsSearch(), new ParamsSelect()), limit);
      EnginePlayerDef player2 = new(new GameEngineDefCeresMCGS("Ceres2", evaluatorDef, new ParamsSearch(), new ParamsSelect()), limit);

      TournamentDef def = new TournamentDef(id, player1, player2);
      def.OpeningsFileName = openingsFileName;
      def.NumGamePairs = numGamePairs;
      def.EnableLiveStreaming = false;
      def.UseTablebasesForAdjudication = false;
      return def;
    }


    static void TestTournament(string net, string device, string slots, string openingsFileName, SearchLimit limit, LocalServer server)
    {
      Section($"Tournament on slots {slots}");

      const int NUM_PAIRS = 6;
      TournamentDef def = MakeTournamentDef("TEST_SLOTS", net, device, openingsFileName, limit, NUM_PAIRS);
      def.DeviceSlots = slots;

      TournamentManager manager = new TournamentManager(def);
      Check(manager.NumConcurrent == 3, "concurrency defaults to the number of streams declared (3)");

      TournamentResultStats results = manager.RunTournament(null, enableCancelVialCtrlC: false);

      Check(results.GameInfos.Count == 2 * NUM_PAIRS, $"all {2 * NUM_PAIRS} games were played ({results.GameInfos.Count})");
      Check(results.GameInfos.Select(g => g.GameSequenceNum).Distinct().Count() == results.GameInfos.Count, "no game was played twice");
      Check(manager.NumGamesPerThread.Length == 3 && manager.NumGamesPerThread.All(n => n > 0),
            $"all three streams played games ({string.Join("/", manager.NumGamesPerThread)})");
      Check(manager.NumThreadsRetired == 0, "no stream was retired");

      // Each remote stream must have its own evaluator on the server (rather than sharing one).
      string[] built = server.OutputLines.Where(l => l.Contains("building new evaluator")).ToArray();
      Check(built.Any(l => l.Contains("|s0")) && built.Any(l => l.Contains("|s1")),
            "the server built separate evaluators for the two remote streams");
    }


    static void TestSuites(string net, string device, string slots, string suiteFileName, SearchLimit limit)
    {
      Section($"Suite (two-engine mode) on slots {slots}");

      NNEvaluatorDef evaluatorDef = NNEvaluatorDef.FromSpecification(net, device);
      GameEngineDef Engine(string id) => new GameEngineDefCeresMCGS(id, evaluatorDef, new ParamsSearch(), new ParamsSelect());

      SuiteTestDef def = new SuiteTestDef("TEST_SLOTS_SUITE", suiteFileName,
                                          new EnginePlayerDef(Engine("C1"), limit), new EnginePlayerDef(Engine("C2"), limit));
      SuiteTestRunner runner = new SuiteTestRunner(def, null, slots);
      SuiteTestResult result = runner.Run(1, true, false, enableCancelVialCtrlC: false);

      Check(runner.NumConcurrent == 3, "concurrency was taken from the slots (3), overriding the thread count argument");
      Check(result != null && result.NumPositionsTested == SUITE_EPDS.Length, "all positions were tested");
      Check(runner.NumPositionsPerWorker.Sum() == SUITE_EPDS.Length,
            $"every position was evaluated exactly once ({string.Join("/", runner.NumPositionsPerWorker)} per worker)");
      Check(result != null && result.TotalNodes1 > 0 && result.TotalNodes2 > 0, "both engines searched");

      Section($"Suite (multi-engine mode) on slots {slots}");

      SuiteTestDef multiDef = new SuiteTestDef("TEST_SLOTS_MULTI", suiteFileName,
                                               ("A", new EnginePlayerDef(Engine("A"), limit), true, limit),
                                               ("B", new EnginePlayerDef(Engine("B"), limit), false, limit));
      multiDef.DeviceSlots = slots;
      SuiteTestRunner multiRunner = new SuiteTestRunner(multiDef);
      MultiEngineSuiteResult multiResult = multiRunner.RunMultiEngine(1, false, false);

      Check(multiResult.NumPositionsTested == SUITE_EPDS.Length, "all positions were tested");
      Check(multiRunner.NumPositionsPerWorker.Sum() == SUITE_EPDS.Length,
            $"every position was evaluated exactly once ({string.Join("/", multiRunner.NumPositionsPerWorker)} per worker)");
    }


    /// <summary>
    /// Kills and restarts the server in the middle of a time-limited tournament running on two remote streams.
    /// The game threads are expected to stall, reconnect, replay the games which were disrupted and finish.
    /// </summary>
    static LocalServer TestServerRestart(string net, string device, string deviceIndices, int port,
                                         string openingsFileName, LocalServer server)
    {
      string slots = $"GPU:{deviceIndices}*2@localhost/{port}";
      Section($"Server restart during a time-limited tournament on slots {slots}");

      const int NUM_PAIRS = 10;
      TournamentDef def = MakeTournamentDef("TEST_SLOTS_RESTART", net, device, openingsFileName, SearchLimit.SecondsPerMove(0.25f), NUM_PAIRS);

      // Signal when the first game has been completed (so the outage certainly falls within the tournament).
      ManualResetEventSlim firstGameDone = new(false);
      def.PerGameCallback = stats =>
      {
        firstGameDone.Set();
        return false;
      };

      TournamentManager manager = new TournamentManager(def, null, slots);

      Task<TournamentResultStats> tournamentTask = Task.Run(() => manager.RunTournament(null, enableCancelVialCtrlC: false));

      // Once the games are under way take the server away for a while.
      firstGameDone.Wait(300_000);
      Thread.Sleep(1_000);
      Console.WriteLine("    >>> Killing the server process...");
      server.Kill();
      Thread.Sleep(5_000);
      Console.WriteLine("    >>> Restarting the server process...");
      server = LocalServer.Start(port);

      TournamentResultStats results = tournamentTask.Result;

      Check(results.GameInfos.Count == 2 * NUM_PAIRS, $"all {2 * NUM_PAIRS} games were completed despite the outage ({results.GameInfos.Count})");
      Check(results.GameInfos.Select(g => g.GameSequenceNum).Distinct().Count() == results.GameInfos.Count, "no game was recorded twice");
      Check(manager.NumThreadsRetired == 0, "no stream was retired (the outage was shorter than the reconnect patience)");
      Check(manager.NumGamesReplayedAfterDisruption > 0,
            $"games disrupted by the outage were discarded and replayed ({manager.NumGamesReplayedAfterDisruption})");
      return server;
    }


    /// <summary>
    /// Kills and restarts the server in the middle of a time-limited suite running on two remote streams.
    /// The workers are expected to stall, reconnect, evaluate again the positions which were disrupted and finish.
    /// </summary>
    static LocalServer TestSuiteServerRestart(string net, string device, string deviceIndices, int port,
                                              string suiteFileName, LocalServer server)
    {
      string slots = $"GPU:{deviceIndices}*2@localhost/{port}";
      Section($"Server restart during a time-limited suite on slots {slots}");

      NNEvaluatorDef evaluatorDef = NNEvaluatorDef.FromSpecification(net, device);
      GameEngineDef Engine(string id) => new GameEngineDefCeresMCGS(id, evaluatorDef, new ParamsSearch(), new ParamsSelect());

      SearchLimit limit = SearchLimit.SecondsPerMove(1.0f);
      SuiteTestDef def = new SuiteTestDef("TEST_SLOTS_SUITE_RESTART", suiteFileName,
                                          ("A", new EnginePlayerDef(Engine("A"), limit), true, limit),
                                          ("B", new EnginePlayerDef(Engine("B"), limit), false, limit));
      SuiteTestRunner runner = new SuiteTestRunner(def, null, slots);

      Task<MultiEngineSuiteResult> suiteTask = Task.Run(() => runner.RunMultiEngine(1, false, false));

      // Wait until the first position is done, then take the server away for a while.
      Stopwatch waited = Stopwatch.StartNew();
      while (runner.NumPositionsPerWorker.Sum() == 0 && !suiteTask.IsCompleted && waited.Elapsed.TotalSeconds < 300)
      {
        Thread.Sleep(50);
      }

      Console.WriteLine("    >>> Killing the server process...");
      server.Kill();
      Thread.Sleep(4_000);
      Console.WriteLine("    >>> Restarting the server process...");
      server = LocalServer.Start(port);

      MultiEngineSuiteResult result = suiteTask.Result;

      Check(result.NumPositionsTested == SUITE_RESTART_EPDS.Length, "all positions were tested despite the outage");
      Check(runner.NumPositionsPerWorker.Sum() == SUITE_RESTART_EPDS.Length,
            $"every position was recorded exactly once ({string.Join("/", runner.NumPositionsPerWorker)} per worker)");
      Check(runner.NumPositionsReplayedAfterDisruption > 0,
            $"positions disrupted by the outage were evaluated again ({runner.NumPositionsReplayedAfterDisruption})");
      return server;
    }


    /// <summary>
    /// Runs the same search of an in-process MCGS engine using an LC0 network twice,
    /// with the network evaluated locally and then on the server, and compares the results.
    /// </summary>
    static void TestLC0(string lc0Net, string device, int port)
    {
      Section($"LC0 network {lc0Net} used by an MCGS search, evaluated locally and then remotely");

      const int NUM_NODES = 2000;

      GameEngineSearchResult RunSearch(string label, string deviceSpec)
      {
        try
        {
          NNEvaluatorDef evaluatorDef = NNEvaluatorDef.FromSpecification(lc0Net, deviceSpec);
          GameEngine engine = new GameEngineDefCeresMCGS("LC0" + label, evaluatorDef, new ParamsSearch(), new ParamsSelect()).CreateEngine();
          GameEngineSearchResult result = engine.Search(PositionWithHistory.StartPosition, SearchLimit.NodesPerMove(NUM_NODES));
          Console.WriteLine($"    {label,-7} device {deviceSpec}: best move {result.MoveString}, N={result.FinalN}, Q={result.ScoreQ:F3}");
          return result;
        }
        catch (Exception exc)
        {
          Console.WriteLine($"    {label,-7} device {deviceSpec}: FAILED with {exc.GetBaseException().GetType().Name}: {exc.GetBaseException().Message}");
          return null;
        }
      }

      GameEngineSearchResult local = RunSearch("local", device);
      Check(local != null && local.FinalN >= NUM_NODES, "the search with the LC0 network evaluated locally completed");

      GameEngineSearchResult remote = RunSearch("remote", $"{device}@localhost/{port}");
      Check(remote != null && remote.FinalN >= NUM_NODES, "the search with the LC0 network evaluated on the server completed");

      if (local != null && remote != null)
      {
        Check(MathF.Abs((float)(local.ScoreQ - remote.ScoreQ)) < 0.05f,
              $"local and remote evaluation agree on the value of the start position ({local.ScoreQ:F3} vs {remote.ScoreQ:F3})");
      }
    }

    #endregion
  }
}
