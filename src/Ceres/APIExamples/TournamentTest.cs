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
using System.Diagnostics;
using System.Threading.Tasks;
using System.Threading;
using System.Collections.Generic;

using Ceres.Base.OperatingSystem;
using Ceres.Base.Benchmarking;

using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Features.Players;
using Ceres.Features.Suites;
using Ceres.Features.Tournaments;
using Ceres.Features.GameEngines;
using Ceres.MCTS.Params;
using Ceres.Chess.NNEvaluators;
using Ceres.Features;
using Ceres.MCTS.Environment;
using Ceres.Features.EngineTests;
using Ceres.Chess.NNBackends.CUDA;
using Ceres.Chess.LC0.WeightsProtobuf;
using Ceres.Chess.LC0.NNFiles;
using Ceres.Chess.Positions;
using Ceres.Chess.NetEvaluation.Batch;
using System.Runtime.InteropServices;
using Ceres.Chess.Games.Utils;
using Ceres.Chess.Data.Nets;

#endregion

namespace Ceres.APIExamples
{
  public static class TournamentTest
  {
    const bool POOLED = false;

    static int CONCURRENCY = POOLED ? 8 : Environment.MachineName.ToUpper().Contains("DEV") ? 2 : 6;
    static int[] OVERRIDE_DEVICE_IDs = /*POOLED ? null*/
       (Environment.MachineName.ToUpper() switch
      {
        var name when name.Contains("DGX") => new int[] { 0, 1, 2, 3 },
        var name when name.Contains("HOP") => new int[] { 0, 1, 2 },
        _ => new int[] { 0 }
      });

    static string GPUS_1 = POOLED ? "GPU:0:POOLED"
                       : "GPU:0";
    static string GPUS_2 = POOLED ? "GPU:0:POOLED"
                           : "GPU:0";


    static bool RUN_DISTRIBUTED = false;


    private static void KillCERES()
    {
      foreach (Process p in Process.GetProcesses())
      {
        if (p.ProcessName.ToUpper().StartsWith("CERES") &&
          !p.ProcessName.ToUpper().StartsWith("TRAIN") &&
          p.Id != Process.GetCurrentProcess().Id)
          p.Kill();
      }
    }

    static string exeCeres() => SoftwareManager.IsLinux ? @"/raid/dev/Ceres/artifacts/release/net8.0/Ceres.dll"
                                          : @"C:\dev\ceres\artifacts\release\net8.0\ceres.exe";
    static string exeCeres93() => SoftwareManager.IsLinux ? @"/raid/dev/Ceres93/artifacts/release/5.0/Ceres.dll"
                                                : @"C:\ceres\releases\v0.93\ceres.exe";
    static string exeCeres96() => SoftwareManager.IsLinux ? @"/raid/dev/Ceres96/Ceres.dll"
                                                : @"C:\ceres\releases\v0.96\ceres.exe";
    static string exeCeresPreNC() => SoftwareManager.IsLinux ? @"/raid/dev/v0.97RC3/artifacts/release/5.0/Ceres.dll"
                                                : @"c:\ceres\releases\v0.97RC3\ceres.exe";

    const string SF11_EXE = @"\\synology\dev\chess\engines\stockfish_11_x64_bmi2.exe";
    const string SF12_EXE = @"\\synology\dev\chess\engines\stockfish_20090216_x64_avx2.exe";
    static string SF14_EXE => SoftwareManager.IsLinux ? @"/raid/dev/SF14.1/stockfish14.1"
                                                      : @"\\synology\dev\chess\engines\stockfish_15_x64_avx2.exe";
    static string SF15_EXE => SoftwareManager.IsLinux ? @"/raid/dev/Stockfish/src/stockfish"
                                                      : @"\\synology\dev\chess\engines\stockfish_15_x64_avx2.exe";


    static List<string> extraUCI = null;// new string[] {"setoption name Contempt value 5000" };
    static GameEngineDef engineDefStockfish11 = new GameEngineDefUCI("SF11", new GameEngineUCISpec("SF11", SF11_EXE, SF_NUM_THREADS, SF_HASH_SIZE_MB(),
                                                                     TB_PATH, uciSetOptionCommands: extraUCI));

    public static GameEngineDef EngineDefStockfish14(int numThreads = SF_NUM_THREADS, int hashtableSize = -1) =>
      new GameEngineDefUCI("SF14.1", new GameEngineUCISpec("SF14.1", SF14_EXE, numThreads,
                           hashtableSize == -1 ? SF_HASH_SIZE_MB() : hashtableSize, TB_PATH, uciSetOptionCommands: extraUCI));
    public static GameEngineDef EngineDefStockfish15(int numThreads = SF_NUM_THREADS, int hashtableSize = -1) =>
      new GameEngineDefUCI("SF15", new GameEngineUCISpec("SF15", SF15_EXE, numThreads,
                           hashtableSize == -1 ? SF_HASH_SIZE_MB() : hashtableSize, TB_PATH, uciSetOptionCommands: extraUCI));

    const int SF_NUM_THREADS = 24;

    static string TB_PATH => CeresUserSettingsManager.Settings.TablebaseDirectory;
    static int SF_HASH_SIZE_MB() => HardwareManager.MemorySize > (256L * 1024 * 1024 * 1024) ? 32_768 : 2_048;

    public static void PreTournamentCleanup()
    {
      if (!RUN_DISTRIBUTED)
      {
        KillCERES();

        File.Delete("Ceres1.log.txt");
        File.Delete("Ceres2.log.txt");
      }
    }



    /// <summary>
    /// Test code.
    /// </summary>
    public static void Test()
    {
      //      DisposeTest(); System.Environment.Exit(3);
      //      if (Marshal.SizeOf<MCTSNodeStruct>() != 64)
      //        throw new Exception("Wrong size " + Marshal.SizeOf<MCTSNodeStruct>().ToString());
      if (Marshal.SizeOf<MCTS.MTCSNodes.Struct.MCTSNodeStruct>() != 64)
        throw new Exception("Wrong size " + Marshal.SizeOf<MCTS.MTCSNodes.Struct.MCTSNodeStruct>().ToString());
      //      PreTournamentCleanup();
      //RunEngineComparisons(); return;

      if (false)
      {
        Parallel.Invoke(() => TestSF(0, true), () => { Thread.Sleep(7_000); TestSF(1, false); });
        System.Environment.Exit(3);
      }


      string NET1_SECONDARY1 = null;// "610024";
      string NET1 = "j94-100";
      string NET2 = "j94-100";

      NET1 = "610235";
      NET2 = "610235";

      NET1 = "760751";
      NET2 = "42767";// "753723";
      NET1 = @"ONNX_ORT:d:\weights\lczero.org\hydra_t00-attn.gz.onnx";// "apv4_t14";// apv4_t16";
      NET2 = @"ONNX_ORT:d:\weights\lczero.org\apv4_t16.onnx";

      //      NET1 = "760998";
      //      NET1 = "790734;1;0;0,753723;0;1;1"; --> 0 +/-10 (new value head)
      //NET1 = "790734;0;1;1,753723;1;0;0"; // 8 +/-8 (new policy head)
      // No obvious progress with T79, 790940 vs 790855 tests at +2 Elo (+/-7) using 1000 nodes/move

      //var pb1 = LC0ProtobufNet.LoadedNet(NET2);
      //pb1.Dump();

      //NET2 = "ap-mish-20b-swa-2000000";
      //      NET2 = "781561";
      //NET1 = "782344";
      //NET1 = @"d:\weights\lczero.org\ap-mish-20b-swa-2000000.pb.gz";
      //      NET1 = NET2 = "ap-mish-20b-swa-2000000";

      //NET2 = "781561";
      //NET1 = "803420";
      //      NET1 = NET2 = "703810";// "784072"; // late T78, 20b, absolute best July 2022
      //      NET1 = NET2 = "803420";
      //NET1 = NET2 = "610889";
      //      NET1 = "800525";
      //      NET2 = "781474"; // last 40b in the T78 series but only as good as 784063 and much slower
      //NET1 = "test-12b1024h8-noclipping-swa-2506000";
      //      NET2 = "784072";// test-12b1024h8-noclipping-swa-1796000";
      //      NET2 = "test-12b1024h8-noclipping-swa-762000";
      //NET1 = "784072";
      //      NET2 = "ap-mish-20b-swa-2000000";
      //NET2 = "20b_mish-swa-2000000";
      //NET1 = "66666";
      //      NET1 = "610889";//= NET2 = "610889";
      //NET1 = "703810"; //NET2 = "753723";

      //      NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.gz#32";
      //      NET1 = @"ONNX_TRT:d:\weights\lczero.org\BT2-768x15smolgen-3326000#16";

//      NET1 = "CUSTOM1:703810,CUSTOM1:703810";
       NET1 = "CUSTOM1:703810";
       NET2 = "CUSTOM2:703810";

//      NET2 = "~T1_DISTILL_256_10_FP16";//// "~T80";

//       NET2 = "~T1_DISTILL_512_15_FP16";

//      NET2= "~T70";

      //      NET1 = "CUSTOM1:753723;1;0;0;1,~T1_DISTILL_512_15;0;1;1;0";
      //-87     NET1 = "CUSTOM1:753723;0;1;0;1,~T1_DISTILL_512_15;1;0;1;0";
      //  ///   "LS15;0.25;0.25;0.25,66666;0.75;0.75;0.75"

      //NET1 =   "~T3";

      //NET1 = "~T3_DISTILL";
      //NET2 = "~BT3";

      //NET2 = "~T1_DISTIL_512_15_NATIVE";
      //NET2 = "~T1_DISTILL_512_15_FP16";

      //NET2 = "~T70";
      //      NET2 = "~T81";


      //NET1 = NET2 = "~T80";
      //NET1 = "ONNX_ORT:BT3_750_optimistic#32,BT3_750#32,";

      //NET1 = "ONNX_ORT:BT3_750_policy_vanilla#32,ONNX_ORT:BT3_750_policy_optimistic#32";
      //NET2 = "ONNX_ORT:BT3_750_policy_vanilla#32";

      if (false)
      {
        var evaluator = NNEvaluator.FromSpecification(NET1, "GPU:0");
        NNEvaluatorBenchmark.EstNPS(evaluator, computeBreaks: false, 64); // warmup
        NNEvaluatorBenchmark.EstNPS(evaluator, computeBreaks: false, 64); // warmup
        NNEvaluatorBenchmark.EstNPS(evaluator, computeBreaks: false, 64); // warmup
        const int BATCH_SIZE = 512;
        while (true)
        {
          (float NPSSingletons, float NPSBigBatch, int[] Breaks) bResults = NNEvaluatorBenchmark.EstNPS(evaluator, computeBreaks: false, BATCH_SIZE);
          Console.WriteLine($"{evaluator} NPS singletons: {bResults.NPSSingletons:n0} NPS big batch: {bResults.NPSBigBatch:n0} batch size {BATCH_SIZE}");
        }
        System.Environment.Exit(3);
      }
      // Won't work (old data format) NET1 = @"ONNX_ORT:d:\cnets\ckpt_NENC_768_15_16_NOS_1331200000#32";


      //NET2 = @"ONNX_ORT:d:\cnets\ckpt_NENC_768_15_16_NOS_1331200000#32";
      //NET2 = ReferenceNetIDs.BEST_T60;

      //NET1 = NET2 = ReferenceNetIDs.BEST_T75;

      //NET1 = "753723";
      //NET2 = "803420";
      //      NET1 = "753723";
      //NET2 = "CUSTOM2:703810";
      //      NET2 = "610889"; //<.................

      //      NET2 = ReferenceNetIDs.BEST_T60;
      //      NET1 = ReferenceNetIDs.BEST_T81;

      //     NET1 = NET2 = ReferenceNetIDs.BEST_T81;
      //      NNWeightsFileLC0 xnetWeightsFile = NNWeightsFileLC0.LookupOrDownload("813338");

      //      NET1 = "813514";
      //NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-2350000-rule50.gz#32";
      // bad NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-2090000#32";
      //NET2 = "753723";// "801307";610889
      //      NET2 = "784984";
      //NET2 = @"d:\weights\lczero.org\t12test6-swa-678000.pb.gz";//"610889";//  ;
     // NET1 = NET2 = "753723";// "703810";
      //NET2 = "t12test6-swa-678000";

      //      NET2 = ReferenceNetIDs.BT2;

      SearchLimit limit1 = SearchLimit.NodesForAllMoves(100_000, 1000) * 3;
      limit1 = SearchLimit.NodesPerMove(100);
      //      limit1 = SearchLimit.NodesPerMove(1000 + ((int)DateTime.Now.Millisecond % 200));
//      limit1 = SearchLimit.BestValueMove;
//      limit1 = SearchLimit.BestActionMove;

//      SearchLimit limit2 = SearchLimit.NodesPerMove(1);
      SearchLimit limit2 = limit1;
      //      limit1 = SearchLimit.SecondsForAllMoves(60, 0.6f);

      // coefficient 3, 250 nodes --> 391 (+119,=170,-102), 52.2 %
      // coefficient 3, 150 nodes -->  138 (+ 47,= 59,- 32), 55.4 %
      // coefficent 3, 750 nodes
      //    boostBase = Math.Min(0.20f, 3 * 0.01f * boostBase);

      //*** WARNING ***
      // ****** NOTE *****
      //  -  ENABLE_LC0 set from true to false
      //  - TestFlag enabled below
      //  - set to run Ceres2 not Ceres2UCI
      //  - 4 GPUS
      //NET2 = NET1;
      //**** END WARNING ***

//      SearchLimit limit2 = limit1;// SearchLimit.NodesPerMove(1);

      //      NET1 = NET2 = "753723";// "610889";
      //      NET1 = "803907";
      //      NET2 = "609966";
      //NET2 = "703810";      
      //      NET1 = "ONNX_ORT:BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.gz";

      // LOSES BY >300 ELO
      //NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16y.gz#16";
      // This is to test for lag in the network and this is to test for lag

      // this is a test
      // BOTH OK (identical, only name differs)
      //      NET1 = @"TRT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-2090000#32";
      //      NET1 = @"ONNX_TRT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-2090000#16";
      //NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-2350000-rule50.gz#32";

      // 3 has diffs of ORT vs TRT
      // for -6: -123 Elo
      // for -7:  FAIL : TensorRT input: /encoder0/smolgen/ln1 has no shape specified.
      //        NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-1260000-fp16-5#16";
      //      NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-2994000.pb.gz";
      //NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-2994";
      //        NET2 = @"ONNX_TRT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-1260000-fp16-5#16";

      //      NET2 = @"LC0:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-2350000-rule50.pb.gz";
      //
      //      NET1 = @"TRT:d:\weights\lczero.org\t1-smolgen-1024x10-swa-1625000#16";      
      //      NET2 = @"ONNX_ORT:d:\weights\lczero.org\t1-smolgen-1024x10-swa-1625000#32";

      //t1-smolgen-1024x10-swa-1625000.onnx
      //      NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1420000-rule50#16";
      //      NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1420000-rule50#16";

      //NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.gz#32";
      //NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16p.gz#16";
      //      NET1 = @"ONNX_ORT:d:\weights\lczero.org\t1-smolgen-1024x10-swa-1625000.16#32";
      //      NET2 = @"ONNX_ORT:d:\weights\lczero.org\t1-smolgen-1024x10-swa-1625000#32";

      //NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.gz#32";

      //NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.noscale.gz#32";

      //NET1 = @"LC0:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.pb.gz#32";

      //NET1 = @"ONNX_ORT:d:\weights\lczero.org\t1-smolgen-1024x10-swa-1625000#16";
      //NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.gz#16";

      // NET1 = @"ONNX_ORT:d:\weights\lczero.org\t1-smolgen-1024x10-swa-1625000.16#16";
      //NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.16.gz#16";

      //NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.gz#32";

      //NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16.gz";

      //OK
      //NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.gz#32";

      //      NET1 = @"ONNX_TRT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16.gz";
      // ***************************
      //      NET1 = "t12b1024h8-swa-2938k-nd";
      //      NET1 = NET2 = "test-12b1024h8-noclipping-swa-2740000";

      //NET1 = @"LC0:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.pb.gz#32";
      //      NET1 = @"LC0:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.pb.gz#32"; // ok

      // -100Elo  NET1 = @"ONNX_TRT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16y.gz#16";
      // -Infinite      NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16y.gz#16";

      //      NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16y.gz#16";
      // OK OK      NET1 = @"ONNX_TRT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.gz#32"; // ok, 71/sec
      // OK LC0_      NET1 = "LC0:d:\\weights\\lczero.org\\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.pb.gz";

      // TRT bad quality      NET1 = @"LC0:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.pb.gz#32"; // see NNEvaluatorFactory line 193

      //try    NET1 = @"ONNX_TRT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.gz#16";

      //      NET1 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16.gz";

      if (false)
      {
        //        var nec1 = NNEvaluator.FromSpecification("ONNX_TRT:d:\\weights\\lczero.org\\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16.gz", "GPU:1");
        var nec1 = NNEvaluator.FromSpecification(NET1, "GPU:0");
        nec1.CalcStatistics(false, 2);
        Console.WriteLine(nec1.EstNPSSingleton + " " + nec1.EstNPSBatch);

        var nec2 = NNEvaluator.FromSpecification(NET2, "GPU:0");
        nec2.CalcStatistics(false, 2);
        Console.WriteLine(nec2.EstNPSSingleton + " " + nec2.EstNPSBatch);

        //        var nec2 = NNEvaluator.FromSpecification("ONNX_TRT:d:\\weights\\lczero.org\\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.gz#32", "GPU:0");
        //        var nec2 = NNEvaluator.FromSpecification("ONNX_ORT:d:\\weights\\lczero.org\\BT2-768x15smolgen-12h-do-01-swa-onnx-1420000-rule50.16", "CPU");
        //var nec2 = nec1;
        NNEvaluatorCompare nec = new NNEvaluatorCompare(nec1, nec2);
        foreach (EPDEntry epd in EPDEntry.EPDEntriesInEPDFile(@"\\synology\dev\chess\data\openings\4mvs_+90_+99.epd"))
        {
          var necEval = nec.Evaluate(epd.PosWithHistory, true);
          Console.WriteLine(necEval.M + " " + nec2.Evaluate(epd.PosWithHistory, true).M);
          //         using (new TimingBlock("1")) Console.WriteLine(nec1.Evaluate(epd.PosWithHistory, true).M);
          //         using (new TimingBlock("2")) Console.WriteLine(nec2.Evaluate(epd.PosWithHistory, true).M);
        }
        System.Environment.Exit(3);
      }
      //NET2 = "t12test6-1600000";
      //NET2 = "CUSTOM1";

      //NET1 = "test-12b1024h8-noclipping-swa-3412000";
      //NET2 = "test-12b1024h8-noclipping-swa-3142000";

      NNEvaluatorDef evalDef1 = NET1 == "CUSTOM1" ? new NNEvaluatorDef(NNEvaluatorType.Custom1, "703810")
                                                 : NNEvaluatorDefFactory.FromSpecification(NET1, GPUS_2);
      NNEvaluatorDef evalDef2 = NET2 == "CUSTOM2" ? new NNEvaluatorDef(NNEvaluatorType.Custom2, "703810")
                                                 : NNEvaluatorDefFactory.FromSpecification(NET2, GPUS_2);

      NNEvaluatorDef? evalDefSecondary1 = null;
      if (NET1_SECONDARY1 != null)
      {
        evalDefSecondary1 = NNEvaluatorDefFactory.FromSpecification($@"LC0:{NET1_SECONDARY1}", GPUS_1);
      }

      NNEvaluatorDef? evalDefSecondary2 = null;



      //      public NNEvaluatorDynamic(NNEvaluator[] evaluators,
      //                        Func<IEncodedPositionBatchFlat, int> dynamicEvaluatorIndexPredicate = null)

      //evalDef1 = NNEvaluatorDefFactory.FromSpecification("ONNX:tfmodelc", "GPU:0");
      //evalDef1 = NNEvaluatorDef.FromSpecification("703810", "GPU:0");


      // 140 good for 203 pairs, 300 good for 100 pairs
      //limit1 = SearchLimit.NodesForAllMoves(200_000, 500);
      //      limit1 = SearchLimit.SecondsForAllMoves(12 + 0.12f);
      //      limit1 = SearchLimit.NodesPerMove(1);
      //limit1 = SearchLimit.SecondsForAllMoves(1);
      //      limit1 = SearchLimit.NodesForAllMoves(100_000, 1000) * 1.0f;
      //      limit1 = SearchLimit.SecondsForAllMoves(60, 0.6f);
      //limit1 = SearchLimit.NodesPerTree(15_000);
      //limit1 = SearchLimit.SecondsForAllMoves(30, 0.3f) * 5f;
      //ok      limit1 = SearchLimit.NodesPerMove(350_000); try test3.pgn against T75 opponent Ceres93 (in first position, 50% of time misses win near move 12

      //      SearchLimit limit2 = limit1;// * 1.18f;
      //      limit2 = SearchLimit.NodesPerMove(1);
      //limit2 = SearchLimit.NodesPerMove(5000);

      // Don't output log if very small games
      // (to avoid making very large log files or slowing down play).
      bool outputLog = true;// limitare1.EstNumNodes(500_000, false) > 10_000;
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", evalDef1, evalDefSecondary1, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres1.log.txt" : null);
      GameEngineDefCeres engineDefCeres2 = new GameEngineDefCeres("Ceres2", evalDef2, evalDefSecondary2, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres2.log.txt" : null);
      GameEngineDefCeres engineDefCeres3 = new GameEngineDefCeres("Ceres3", evalDef2, evalDefSecondary2, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres3.log.txt" : null);

//      engineDefCeres1.SearchParams.ValueTemperature = 0.85f;

//      engineDefCeres1.SearchParams.HistoryFillIn = false;

      //engineDefCeres1.OverrideLimitManager = new  Ceres.MCTS.Managers.Limits.ManagerGameLimitTest();
      if (false)
      {
        //engineDefCeres1.OverrideLimitManager = new MCTS.Managers.Limits.ManagerGameLimitCeresL();
        //engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
        //engineDefCeres1.SearchParams.EnableInstamoves = false;
      }

      //engineDefCeres1.SelectParams.CPUCT *= 0.85f;
      //engineDefCeres2.SelectParams.CPUCT *= 0.85f;

//      engineDefCeres1.SearchParams.BestMoveMode = ParamsSearch.BestMoveModeEnum.TopV;
//      engineDefCeres2.SearchParams.BestMoveMode = ParamsSearch.BestMoveModeEnum.TopV;

//engineDefCeres1.SearchParams.Execution.FlowDualSelectors = false;
//engineDefCeres2.SearchParams.Execution.FlowDualSelectors = false;

//AdjustSelectParamsNewTuneBR(engineDefCeres1.SelectParams);
//AdjustSelectParamsNewTuneBR(engineDefCeres2.SelectParams);
//engineDefCeres1.SelectParams.UCTNonRootDenominatorExponent = 0.95f;
//engineDefCeres1.SelectParams.UCTRootDenominatorExponent = 0.90f;

//engineDefCeres1.SelectParams.CPUCTFactorAtRoot *= 1.5f;
//engineDefCeres1.SelectParams.VirtualLossDefaultRelative = -0.06f;
// This was +2 Elo (+/-13) in 100 seconds games with late T60 *********************************
//engineDefCeres1.SelectParams.UCTRootNumeratorExponent = 0.52f;
//engineDefCeres1.SelectParams.UCTNonRootNumeratorExponent = 0.48f;

//      engineDefCeres1.SearchParams.MoveFutilityPruningAggressiveness *= 0.5f;
//      engineDefCeres2.SearchParams.MoveFutilityPruningAggressiveness *= 0;// 0.5f;

//      engineDefCeres1.SelectParams.CPUCTDualSelectorDiffFraction = 0.04f;

// &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

engineDefCeres1.SearchParams.ActionHeadSelectionWeight = 0.666f;
//engineDefCeres2.SearchParams.ActionHeadSelectionWeight = 0.666f;

engineDefCeres1.SearchParams.Execution.MaxBatchSize = 4;
engineDefCeres2.SearchParams.Execution.MaxBatchSize = 4;

engineDefCeres1.SearchParams.TreeReuseEnabled = false;
engineDefCeres2.SearchParams.TreeReuseEnabled = false;

engineDefCeres1.SearchParams.Execution.SelectParallelEnabled = false;
engineDefCeres2.SearchParams.Execution.SelectParallelEnabled = false;

engineDefCeres1.SearchParams.Execution.FlowDualSelectors = false;
engineDefCeres2.SearchParams.Execution.FlowDualSelectors = false;

engineDefCeres1.SearchParams.Execution.TranspositionMode = TranspositionMode.None;
engineDefCeres2.SearchParams.Execution.TranspositionMode = TranspositionMode.None;



//engineDefCeres1.SearchParams.EnableTablebases = false;
//engineDefCeres2.SearchParams.EnableTablebases = false;

      //engineDefCeres1.SearchParams.Execution.MaxBatchSize = 128;
      //      engineDefCeres1.SearchParams.BatchSizeMultiplier = 2;

      //      engineDefCeres1.SearchParams.ResamplingMoveSelectionFractionMove = 1f;
      //      engineDefCeres1.SearchParams.EnableSearchExtension = false;
      //      engineDefCeres2.SearchParams.EnableSearchExtension = false;

      //            engineDefCeres1.SearchParams.TestFlag2 = true;
      //      engineDefCeres1.SearchParams.Execution.FlowDualSelectors = false;
      //      engineDefCeres1.SearchParams.TranspositionRootPolicyBlendingFraction = 0.5f;

      //      engineDefCeres1.SearchParams.TestFlag = true;
      //      engineDefCeres1.SearchParams.EnableUncertaintyBoosting = true;
      //      engineDefCeres2.SearchParams.EnableUncertaintyBoosting = true;

      //     engineDefCeres1.SearchParams.Execution.TranspositionMode = TranspositionMode.None;
      //engineDefCeres1.SearchParams.Execution.TranspositionMode = TranspositionMode.SingleNodeCopy;
      //engineDefCeres1.SearchParams.TranspositionCloneNodeSubtreeFracs[0] = 0;
      //engineDefCeres1.SearchParams.TranspositionCloneNodeSubtreeFracs[1] = 0;
      //engineDefCeres1.SearchParams.TranspositionRootBackupSubtreeFracs[0] = 0;
      //engineDefCeres1.SearchParams.TranspositionRootBackupSubtreeFracs[1] = 0;

      //      engineDefCeres2.SearchParams.Execution.TranspositionMode = TranspositionMode.SingleNodeCopy;
      // engineDefCeres1.SearchParams.TreeReuseRetainedPositionCacheEnabled = true;
      //engineDefCeres2.SearchParams.TreeReuseRetainedPositionCacheEnabled = true;

      //      engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
      //      engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled = false;

      //engineDefCeres2.SearchParams.TestFlag2 = true;

      //      AdjustSelectParamsNewTune(engineDefCeres1.SelectParams);
      //      AdjustSelectParamsNewTune(engineDefCeres2.SelectParams);

      //engineDefCeres1.SelectParams.CPUCT *= 1.15f;

      //      engineDefCeres1.SearchParams.TestFlag2 = true;
      //engineDefCeres1.SearchParams.EnableUncertaintyBoosting = true;
      //      engineDefCeres2.SearchParams.EnableUncertaintyBoosting = true;

      //engineDefCeres1.SelectParams.CPUCT *= 0.94f;
      //engineDefCeres1.SearchParams.BestMoveMode = ParamsSearch.BestMoveModeEnum.TopN;
      //      engineDefCeres1.SearchParams.TranspositionRootMaxN = true;
      //      engineDefCeres1.SearchParams.EnableUseSiblingEvaluations = true;

      //      engineDefCeres1.SearchParams.TranspositionRootMaxN = true;

      //      engineDefCeres1.SearchParams.TestFlag = true;
      //engineDefCeres1.SearchParams.TestFlag = true;

      //      engineDefCeres1.SearchParams.GameLimitUsageAggressiveness = 1.3f;
      //      engineDefCeres2.SearchParams.TestFlag2 = true;

      //      engineDefCeres1.SearchParams.TreeReuseRetainedPositionCacheEnabled = true;

      //engineDefCeres2.SearchParams.TestFlag = true;
      //      engineDefCeres1.SelectParams.CPUCT *= 0.9f;


      //      engineDefCeres2.SearchParams.EnableUncertaintyBoosting = false;
      //      engineDefCeres1.SelectParams.CPUCT *= 0.30f;

      //engineDefCeres2.SelectParams.CPUCTAtRoot *= 1.33f;


      //engineDefCeres1.SearchParams.TestFlag = true;


      //    engineDefCeres1.SearchParams.TranspositionRootBackupSubtreeFracs = new float[] { 1,0, float.NaN };
      //      engineDefCeres2.SearchParams.TranspositionRootBackupSubtreeFracs = new float[] { 1,0, float.NaN };
      //engineDefCeres1.SearchParams.TranspositionCloneNodeSubtreeFracs = new float[] { 1, 0, float.NaN };
      //engineDefCeres1.SearchParams.TranspositionCloneNodeSubtreeFracs = new float[] { 1, 0, float.NaN };
#if NOT

#endif

      if (false)
      {
        // ************ SPECIAL *****************
        engineDefCeres1.SearchParams.ParamsSecondaryEvaluator.UpdateFrequencyMinNodesAbsolute = 200;
        engineDefCeres1.SearchParams.ParamsSecondaryEvaluator.UpdateFrequencyMinNodesRelative = 0.03f;
        engineDefCeres1.SearchParams.ParamsSecondaryEvaluator.UpdateMinNFraction = 0.03f; // was 0.01
        engineDefCeres1.SearchParams.ParamsSecondaryEvaluator.UpdateValueFraction = 0.5f;
        engineDefCeres1.SearchParams.ParamsSecondaryEvaluator.UpdatePolicyFraction = 0 * 0.5f;

        engineDefCeres1.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
        engineDefCeres2.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
        // ****************************************************
      }

      if (!limit1.IsNodesLimit)
      {
        engineDefCeres1.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
        engineDefCeres2.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
      }


      // TODO: support this in GameEngineDefCeresUCI
      bool forceDisableSmartPruning = limit1.IsNodesLimit && !limit1.IsPerGameLimit;
      if (forceDisableSmartPruning)
      {
        engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres1.SearchParams.MoveFutilityPruningAggressiveness = 0;
        engineDefCeres2.SearchParams.MoveFutilityPruningAggressiveness = 0;
      }

      //GameEngineDef engineDefCeresUCI = new GameEngineDefUCI("CeresUCI", new GameEngineUCISpec("CeresUCI", @"c:\dev\ceres\artifacts\release\net5.0\ceres.exe"));
      //      GameEngineDef engineDefCeresUCI1x = new GameEngineDefCeresUCI("CeresUCINew", evalDef1, overrideEXE: @"C:\dev\Ceres\artifacts\release\net5.0\ceres.exe");

      GameEngineDef engineDefCeresUCI1 = new GameEngineDefCeresUCI("CeresUCINew", evalDef1, overrideEXE: exeCeres(), disableFutilityStopSearch: forceDisableSmartPruning);
      GameEngineDef engineDefCeresUCI2 = new GameEngineDefCeresUCI("CeresUCINew", evalDef2, overrideEXE: exeCeres(), disableFutilityStopSearch: forceDisableSmartPruning);

      GameEngineDef engineDefCeres93 = new GameEngineDefCeresUCI("Ceres93", evalDef2, overrideEXE: exeCeres93(), disableFutilityStopSearch: forceDisableSmartPruning);
      GameEngineDef engineDefCeres96 = new GameEngineDefCeresUCI("Ceres96", evalDef2, overrideEXE: exeCeres96(), disableFutilityStopSearch: forceDisableSmartPruning);
      GameEngineDef engineDefCeresPreNC = new GameEngineDefCeresUCI("CeresPreNC", evalDef2, overrideEXE: exeCeresPreNC(), disableFutilityStopSearch: forceDisableSmartPruning);

      EnginePlayerDef playerCeres1UCI = new EnginePlayerDef(engineDefCeresUCI1, limit1);
      EnginePlayerDef playerCeres2UCI = new EnginePlayerDef(engineDefCeresUCI2, limit2);
      EnginePlayerDef playerCeres93 = new EnginePlayerDef(engineDefCeres93, limit2);
      EnginePlayerDef playerCeres96 = new EnginePlayerDef(engineDefCeres96, limit2);
      EnginePlayerDef playerCeresPreNC = new EnginePlayerDef(engineDefCeresPreNC, limit2);

      EnginePlayerDef playerCeres1 = new EnginePlayerDef(engineDefCeres1, limit1);
      EnginePlayerDef playerCeres2 = new EnginePlayerDef(engineDefCeres2, limit2);
      EnginePlayerDef playerCeres3 = new EnginePlayerDef(engineDefCeres3, limit1);

      bool ENABLE_LC0 = false;// evalDef1.Nets[0].Net.Type == NNEvaluatorType.LC0Library && (evalDef1.Nets[0].WeightValue == 1 && evalDef1.Nets[0].WeightPolicy == 1 && evalDef1.Nets[0].WeightM == 1);
      string OVERRIDE_LC0_EXE = @"c:\apps\lc0_30\lc0_PR917.exe";
      GameEngineDefLC0 engineDefLC1 = ENABLE_LC0 ? new GameEngineDefLC0("LC0_0", evalDef1, forceDisableSmartPruning, null, null, overrideEXE: OVERRIDE_LC0_EXE) : null;
      GameEngineDefLC0 engineDefLC2 = ENABLE_LC0 ? new GameEngineDefLC0("LC0_2", evalDef2, forceDisableSmartPruning, null, null, overrideEXE: OVERRIDE_LC0_EXE) : null;

      EnginePlayerDef playerStockfish14 = new EnginePlayerDef(EngineDefStockfish14(), limit2 * 0.30f);// * 350);
      EnginePlayerDef playerLC0 = ENABLE_LC0 ? new EnginePlayerDef(engineDefLC1, limit1) : null;
      EnginePlayerDef playerLC0_2 = ENABLE_LC0 ? new EnginePlayerDef(engineDefLC2, limit2) : null;


      if (false)
      {
        string BASE_NAME = "ERET";// nice_lcx Stockfish238 ERET_VESELY203 endgame2 chad_tactics-100M lichess_chad_bad.csv
        ParamsSearch paramsNoFutility = new ParamsSearch() { FutilityPruningStopSearchEnabled = false };

        // ===============================================================================
        string suiteGPU = POOLED ? "GPU:0,1,2,3:POOLED=SHARE1" : "GPU:0";
        SuiteTestDef suiteDef =
          new SuiteTestDef("Suite",
                           SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/epd/{BASE_NAME}.epd"
                                                   : @$"\\synology\dev\chess\data\epd\{BASE_NAME}.epd",
                           SearchLimit.NodesPerMove(500),
                           GameEngineDefFactory.CeresInProcess("Ceres1", NET1, suiteGPU, paramsNoFutility with { }),
                           GameEngineDefFactory.CeresInProcess("Ceres2", NET2, suiteGPU, paramsNoFutility with { }),
                           null);// engineDefCeres96);// playerLC0.EngineDef);

        suiteDef.MaxNumPositions = 500;
        suiteDef.EPDLichessPuzzleFormat = suiteDef.EPDFileName.ToUpper().Contains("LICHESS");

        suiteDef.AcceptPosPredicate = null;// p => IsKRP(p);

        SuiteTestRunner suiteRunner = new SuiteTestRunner(suiteDef);

        SuiteTestResult suiteResult = suiteRunner.Run(POOLED ? 12 : 1, true, false);
        Console.WriteLine("Max mbytes alloc: " + WindowsVirtualAllocManager.MaxBytesAllocated / (1024 * 1024));
        Console.WriteLine("Test counter 1  : " + MCTSEventSource.TestCounter1);
        Console.WriteLine("Test metric 1   : " + MCTSEventSource.TestMetric1);
        return;
      }

#if NOT
      EnginePlayerDef playerDefCSNoNN = new EnginePlayerDef(engineDefCSNoNN, limit2);
      EnginePlayerDef playerDefCSNN1 = new EnginePlayerDef(engineDefCSNN1, limit2);
      EnginePlayerDef playerDefCSNN50 = new EnginePlayerDef(engineDefCSNN50, limit2);
#endif
      // **************************************************
      EnginePlayerDef player1 = playerCeres1;// playerCeres1UCI;// new EnginePlayerDef(engineDefCSNN1, SearchLimit.NodesPerMove(30));
      EnginePlayerDef player2 = playerCeres2;// playerCeres96;// new EnginePlayerDef(EnginDefStockfish14(), SearchLimit.NodesPerMove(300 * 10_000));
      //new EnginePlayerDef(engineDefCSNoNN, SearchLimit.NodesPerMove(300 * 10_000));
      // **************************************************

      TournamentGameQueueManager queueManager = null;
      bool isDistributed = false;
      if (CommandLineWorkerSpecification.IsWorker)
      {
        queueManager = new TournamentGameQueueManager(Environment.GetCommandLineArgs()[2]);
        int gpuID = CommandLineWorkerSpecification.GPUID;
        Console.WriteLine($"\r\n***** Running in DISTRIBUTED mode as WORKER on gpu {gpuID} (queue directory {queueManager.QueueDirectory})\r\n");

        player1.EngineDef.ModifyDeviceIndexIfNotPooled(gpuID);
        player2.EngineDef.ModifyDeviceIndexIfNotPooled(gpuID);
      }
      else if (RUN_DISTRIBUTED)
      {
        isDistributed = true;
        queueManager = new TournamentGameQueueManager(null);
        Console.WriteLine($"\r\n***** Running in DISTRIBUTED mode as COORDINATOR (queue directory {queueManager.QueueDirectory})\r\n");
      }

      TournamentDef def;
      bool roundRobin = false;
      if (roundRobin)
      {
        def = new TournamentDef("RR");
        const float SF_TIME_SCALE = 0.8f;
        //        def.AddEngine(playerStockfish14.EngineDef, limit1 * SF_TIME_SCALE);

        def.AddEngines(limit1, engineDefCeres1);
        def.AddEngines(limit1, engineDefLC1);
        //def.ReferenceEngineId = def.Engines[0].ID;

      }
      else
      {
        def = new TournamentDef("TOURN", player1, player2);
        //def.CheckPlayer2Def = playerLC0;
      }

      // TODO: UCI engine should point to .NET 6 subdirectory if on .NET 6
      if (isDistributed)
      {
        def.IsDistributedCoordinator = true;
      }


      def.NumGamePairs = 10_000;// 10_000;// 2000;// 10_000;// 203;//1000;//203;//203;// 500;// 203;//203;// 102; 203
      def.ShowGameMoves = false;

      //string baseName = "tcec1819";
      //      string baseName = "4mvs_+90_+99";
      //      string baseName = "4mvs_+90_+99.epd";

      string baseName = "book-ply8-unifen-Q-0.25-0.40";
      baseName = "book-ply8-unifen-Q-0.25-0.40";
//      baseName = "single_bad";
//            baseName = "Noomen 2-move Testsuite.pgn";
//            baseName = "book-ply8-unifen-Q-0.40-1.0";
//      baseName = "book-ply8-unifen-Q-0.0-0.25.pgn";
//       baseName = "endingbook-10man-3181.pgn";
//      baseName = "book-ply8-unifen-Q-0.25-0.40";
      const bool KRP =false;
      if (KRP)
      {
        throw new NotImplementedException();
//        baseName = "endingbook-16man-9609.pgn";
//        def.AcceptPosExcludeIfContainsPieceTypeList = [PieceType.Queen, PieceType.Bishop, PieceType.Knight];
      }
//       baseName = "tcec_big";
      string postfix = (baseName.ToUpper().EndsWith(".EPD") || baseName.ToUpper().EndsWith(".PGN")) ? "" : ".pgn";
      def.OpeningsFileName = SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/openings/{baseName}{postfix}"
                                                     : @$"\\synology\dev\chess\data\openings\{baseName}{postfix}";
      // not functioning def.AcceptPosPredicate = p => IsKRP(p);  

//ConsoleUtils.WriteLineColored(ConsoleColor.Red, "WARNING TB ADJUDICATION OFF");
//def.UseTablebasesForAdjudication = false;

      if (false)
      {
        def.AdjudicateDrawThresholdCentipawns = 0;
        def.AdjudicateDrawThresholdNumMoves = 999;

        def.AdjudicateWinThresholdCentipawns = int.MaxValue;
        def.AdjudicateWinThresholdNumMovesDecisive = 3000;
        def.UseTablebasesForAdjudication = false;
      }

      TournamentManager runner = new TournamentManager(def, CONCURRENCY, OVERRIDE_DEVICE_IDs);

      TournamentResultStats results;

      //UCIEngineProcess.VERBOSE = true;

      TimingStats stats = new TimingStats();
      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        results = runner.RunTournament(queueManager);
      }

      Console.WriteLine();
      Console.WriteLine($"Tournament completed in {stats.ElapsedTimeSecs,8:F2} seconds.");
      //Console.WriteLine(results.GameOutcomesString);

      Console.WriteLine();
      Console.WriteLine("<CRLF> to continue");
      Console.ReadLine();
    }


    public static void TestSF(int index, bool gitVersion)
    {
      // Initialize settings by loading configuration file
      //CeresUserSettingsManager.LoadFromFile(@"c:\dev\ceres\artifacts\release\net5.0\ceres.json");

      // Define constants for engine parameters
      string SF14_EXE = Path.Combine(CeresUserSettingsManager.Settings.DirExternalEngines, "Stockfish14.1.exe");
      const int SF_THREADS = 8;
      const int SF_TB_SIZE_MB = 1024;

      string CERES_NETWORK = CeresUserSettingsManager.Settings.DefaultNetworkSpecString; //"LC0:703810";
      const string CERES_GPU = "GPU:0";

      string TB_DIR = CeresUserSettingsManager.Settings.DirTablebases;
      SearchLimit TIME_CONTROL = SearchLimit.SecondsForAllMoves(10, 0.5f); //* 0.15f;            
      const int NUM_GAME_PAIRS = 50;
      const string logfile = "ceres.log.txt"; //Path.Combine(CeresUserSettingsManager.Settings.DirCeresOutput, "ceres.log.txt");

      // Define Stockfish engine (via UCI) 
      GameEngineDefUCI sf14Engine = new GameEngineDefUCI("SF14", new GameEngineUCISpec("SF14", SF14_EXE, SF_THREADS, SF_TB_SIZE_MB, TB_DIR));

      // Define Ceres engine (in process) with associated neural network and GPU and parameter customizations
      NNEvaluatorDef ceresNNDef = NNEvaluatorDefFactory.FromSpecification(CERES_NETWORK, CERES_GPU);
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", ceresNNDef, null,
                                                                  new ParamsSearch() { /* FutilityPruningStopSearchEnabled = false, */ },
                                                                  new ParamsSelect(),
                                                                  logFileName: logfile);

      // Define players using these engines and specified time control
      EnginePlayerDef playerCeres = new EnginePlayerDef(engineDefCeres1, TIME_CONTROL);
      EnginePlayerDef playerSF = new EnginePlayerDef(sf14Engine, TIME_CONTROL);

      // Create a tournament definition
      TournamentDef tournDef = new TournamentDef("Ceres_vs_Stockfish", playerCeres, playerSF);
      tournDef.NumGamePairs = NUM_GAME_PAIRS;
      tournDef.OpeningsFileName = "WCEC.pgn";
      tournDef.ShowGameMoves = false;

      // Run the tournament
      TimingStats stats = new TimingStats();
      TournamentResultStats results;
      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        results = new TournamentManager(tournDef).RunTournament();
      }
      Console.WriteLine();
      Console.WriteLine($"Tournament completed in {stats.ElapsedTimeSecs,8:F2} seconds.");
      Console.ReadLine();
    }


    public static void TestSFLeela(int index, bool gitVersion)
    {
      // Define constants for engine parameters  
      string SF14_EXE = Path.Combine(CeresUserSettingsManager.Settings.DirExternalEngines, "Stockfish14.1.exe");
      //string leela_EXE = Path.Combine(CeresUserSettingsManager.Settings.DirExternalEngines, "lc0-v0.28.0-windows-gpu-nvidia-cuda", "LC0.exe");
      const int SF_THREADS = 8;
      const int SF_TB_SIZE_MB = 1024;
      string TB_DIR = CeresUserSettingsManager.Settings.DirTablebases;
      string CERES_NETWORK = CeresUserSettingsManager.Settings.DefaultNetworkSpecString; //"LC0:703810";
      const string CERES_GPU = "GPU:0";

      SearchLimit TIME_CONTROL = SearchLimit.SecondsForAllMoves(60, 1f) * 0.1f;
      const int NUM_GAME_PAIRS = 1;
      const string logfile = "CeresRR.log.txt";

      // Define Stockfish engine (via UCI) 
      GameEngineDefUCI sf14Engine = new GameEngineDefUCI("SF14.1", new GameEngineUCISpec("SF14.1", SF14_EXE, SF_THREADS, SF_TB_SIZE_MB, TB_DIR));

      // Define Ceres engine (in process) with associated neural network and GPU and parameter customizations
      NNEvaluatorDef ceresNNDef = NNEvaluatorDefFactory.FromSpecification(CERES_NETWORK, CERES_GPU);
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", ceresNNDef, null, new ParamsSearch(), new ParamsSelect(), logFileName: logfile);
      GameEngineDefCeres engineDefCeres2 = new GameEngineDefCeres("Ceres2", ceresNNDef, null, new ParamsSearch(), new ParamsSelect(), logFileName: "ceres2.log.txt");

      // Define Leela engine (in process) with associated neural network and GPU and parameter customizations
      GameEngineDefLC0 engineDefLC0 = new GameEngineDefLC0("LC0", ceresNNDef, forceDisableSmartPruning: false, null, null);

      // Define players using these engines and specified time control
      EnginePlayerDef playerCeres1 = new EnginePlayerDef(engineDefCeres1, TIME_CONTROL);
      EnginePlayerDef playerLeela = new EnginePlayerDef(engineDefLC0, TIME_CONTROL);
      EnginePlayerDef playerSf14 = new EnginePlayerDef(sf14Engine, TIME_CONTROL);
      EnginePlayerDef playerCeres2 = new EnginePlayerDef(engineDefCeres2, TIME_CONTROL);
      EnginePlayerDef playerSf14Slow = new EnginePlayerDef(sf14Engine, TIME_CONTROL * 0.5f, "SF14*0.5");

      // Create a tournament definition
      TournamentDef tournDef = new TournamentDef("Round Robin Test", playerCeres1, playerLeela);
      //tournDef.ReferenceEngineId = playerCeres1.ID;
      tournDef.NumGamePairs = NUM_GAME_PAIRS;
      tournDef.OpeningsFileName = "WCEC.pgn";
      tournDef.ShowGameMoves = false;


      // Run the tournament
      TimingStats stats = new TimingStats();
      TournamentResultStats results;
      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        results = new TournamentManager(tournDef).RunTournament();
      }
      Console.WriteLine();
      Console.WriteLine($"Tournament completed in {stats.ElapsedTimeSecs,8:F2} seconds.");
      Console.ReadLine();
    }

    public static void TestLeela(int index, bool gitVersion)
    {
      // Initialize settings by loading configuration file
      //CeresUserSettingsManager.LoadFromFile(@"c:\dev\ceres\artifacts\release\net5.0\ceres.json");

      //example code:
      // SearchLimit TIME_CONTROL = SearchLimit.NodesPerMove(10_000);
      // for Ceres, set: new ParamsSearch() { FutilityPruningStopSearchEnabled = false, },            
      // for LC0 player, set in constructor: forceDisableSmartPruning:true

      // Define constants for engine parameters           

      string CERES_NETWORK = CeresUserSettingsManager.Settings.DefaultNetworkSpecString; //"LC0:703810";
      const string CERES_GPU = "GPU:0";
      string TB_DIR = CeresUserSettingsManager.Settings.DirTablebases;
      SearchLimit TIME_CONTROL = SearchLimit.SecondsForAllMoves(30, 1f) * 0.07f;
      const string logfileCeres = "ceres.log.txt";

      // Define Ceres engine (in process) with associated neural network and GPU and parameter customizations
      NNEvaluatorDef ceresNNDef = NNEvaluatorDefFactory.FromSpecification(CERES_NETWORK, CERES_GPU);
      GameEngineDefCeres engineDefCeres = new GameEngineDefCeres("Ceres-1", ceresNNDef, null,
                                                                  new ParamsSearch() { /* FutilityPruningStopSearchEnabled = false, */ },
                                                                  new ParamsSelect(),
                                                                  logFileName: logfileCeres);

      GameEngineDefCeres engineDef1Ceres = new GameEngineDefCeres("Ceres-2", ceresNNDef, null,
                                                      new ParamsSearch() { /* FutilityPruningStopSearchEnabled = false, */ },
                                                      new ParamsSelect(),
                                                      logFileName: "ceres2.log.txt");

      // Define Leela engine (in process) with associated neural network and GPU and parameter customizations
      GameEngineDefLC0 engineDefLC0 = new GameEngineDefLC0("LC0-1", ceresNNDef, forceDisableSmartPruning: false, null, null);
      GameEngineDefLC0 engineDef1LC0 = new GameEngineDefLC0("LC0-2", ceresNNDef, forceDisableSmartPruning: false, null, null);

      //NNEvaluatorDef leelaNNDef = NNEvaluatorDefFactory.FromSpecification($"LC0:{CERES_NETWORK}", CERES_GPU);
      //GameEngineDefUCI engineDefLeela1 = new GameEngineDefUCI("Leela", new GameEngineUCISpec("LC0",leela_EXE, syzygyPath: TB_DIR));           

      // Define players using these engines and specified time control
      EnginePlayerDef playerCeres = new EnginePlayerDef(engineDefCeres, TIME_CONTROL);
      EnginePlayerDef playerCeres2 = new EnginePlayerDef(engineDef1Ceres, TIME_CONTROL);
      EnginePlayerDef playerLeela = new EnginePlayerDef(engineDefLC0, TIME_CONTROL);
      EnginePlayerDef playerLeela2 = new EnginePlayerDef(engineDef1LC0, TIME_CONTROL);


      // Create a tournament definition
      TournamentDef tournDef = new TournamentDef("Tournament A", playerCeres, playerLeela, playerLeela2);
      // Create a tournament definition
      //TournamentDef tournDef = new TournamentDef("Ceres_vs_Leela", playerCeres, playerLeela);
      tournDef.NumGamePairs = 1;
      tournDef.OpeningsFileName = "WCEC_decisive.pgn";
      tournDef.ShowGameMoves = false;

      // Run the tournament
      TimingStats stats = new TimingStats();
      TournamentResultStats results;
      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        results = new TournamentManager(tournDef).RunTournament();
      }
      Console.WriteLine();
      Console.WriteLine($"Tournament completed in {stats.ElapsedTimeSecs,8:F2} seconds.");
      Console.ReadLine();
    }

    static NNEvaluatorDef EvaluatorValueOnly(string netID1, string netID2, int gpuID, bool valueNet1)
    {
      string wtStr1 = valueNet1 ? "0.5;1.0;0.5" : "0.5;0.0;0.5";
      string wtStr2 = valueNet1 ? "0.5;0.0;0.5" : "0.5;1.0;0.5";
      NNEvaluatorDef spec = NNEvaluatorDef.FromSpecification($"LC0:{netID1}@{wtStr1},{netID2}@{wtStr2}", $"GPU:{gpuID}");
      return spec;
    }

    //      evalDef1 = EvaluatorValueOnly(NET1, NET2, 0, true);
    //      evalDef2 = EvaluatorValueOnly(NET1, NET2, 0, false);


    public static void RunEngineComparisons()
    {
      string pgnFileName = SoftwareManager.IsWindows ? @"\\synology\dev\chess\data\pgn\raw\ceres_big.pgn"
                                               : @"/mnt/syndev/chess/data/pgn/raw/ceres_big.pgn";
      string NET_ID = RegisteredNets.Aliased["T78"].NetSpecificationString;// "753723";// "610889";// "803907";
      CompareEngineParams parms = new CompareEngineParams("Resapling", pgnFileName,
                                              10_000, // number of positions
                                              s => true,//s.FinalPosition.PieceCount > 15,
                                              CompareEnginesVersusOptimal.PlayerMode.CeresCustom1, "703810", //610034
                                              CompareEnginesVersusOptimal.PlayerMode.UCI, NET_ID,
                                              CompareEnginesVersusOptimal.PlayerMode.LC0, RegisteredNets.Aliased["T80"].NetSpecificationString,
                                              SearchLimit.NodesPerMove(50), // search limit
                                              new int[] { 0 },//new int[] { 0, 1, 2, 3 },
                                              s =>
                                              {
                                                //s.TestFlag = true;
                                                //s.Execution.MaxBatchSize = 256;
                                                //s.ResamplingMoveSelectionFractionMove = 1f;
                                                //s.ResamplingMoveSelectionTemperature = 1.5f;
                                                //s.TranspositionRootPolicyBlendingFraction = 0.25f;
                                                // s.TranspositionRootPolicyBlendingFraction = 0.333f;
                                                //s.EnableUncertaintyBoosting = true;
                                              },
                                              l =>
                                              {
                                                //l.CPUCT *= 1.2f;
                                                //l.size
                                                //l.VirtualLossDefaultRelative = -0.02f;
                                                //AdjustSelectParamsNewTune(l);
                                              },
                                              null, // l => l.CPUCT = 1.1f,
                                              null,
                                              true,
                                              1,
                                              20,
                                              true, // Stockfish crosscheck
                                              null,
                                              exeCeres(),
                                              0.25f
                                             );


      CompareEngineResultSummary result = new CompareEnginesVersusOptimal(parms).Run();
    }

    static void AdjustSelectParamsNewTuneBR(ParamsSelect p)
    {
      p.CPUCT = 1.88f;
      p.CPUCTAtRoot = 1.88f;
      p.CPUCTFactor = 3.973f;
      p.CPUCTFactorAtRoot = 3.973f;
      p.CPUCTBase = 45669;
      p.CPUCTBaseAtRoot = 45669;
      p.FPUValue = 0.286f;
      p.PolicySoftmax = 1.16f;
    }



    static void AdjustSelectParamsNewTuneT60(ParamsSelect p)
    {
      p.CPUCT = 1.473f;
      p.CPUCTAtRoot = 1.473f;
      p.CPUCTFactor = 3.973f;
      p.CPUCTFactorAtRoot = 3.973f;
      p.CPUCTBase = 45669;
      p.CPUCTBaseAtRoot = 45669;
      p.FPUValue = 0.2790f;
      p.PolicySoftmax = 1.3f;
    }


    static void DisposeTest()
    {
      const string NET_ID = "703810";
      // TODO: repeated execution does not release all memory
      while (false)
      {
        using (new TimingBlock("CUDA create/dispose", TimingBlock.LoggingType.ConsoleWithMemoryTracking))
        {
          for (int i = 0; i < 10; i++)
          {
            Console.WriteLine("create " + i);
            NNWeightsFileLC0 netWeightsFile = NNWeightsFileLC0.LookupOrDownload(NET_ID);
            LC0ProtobufNet net = LC0ProtobufNet.LoadedNet(netWeightsFile.FileName);
            NNBackendLC0_CUDA backend = new NNBackendLC0_CUDA(0, netWeightsFile);
            Console.WriteLine("dispose " + i);
            backend.Dispose();
          }
          GC.Collect(3);
          GC.WaitForFullGCComplete();
        }
        Console.WriteLine("<CR> to continue....");
        Console.ReadLine();
      }

      NNEvaluatorDef nd = NNEvaluatorDefFactory.FromSpecification(NET_ID, "GPU:0");
      NNEvaluator referenceEvaluator = NNEvaluatorFactory.BuildEvaluator(nd);

      if (true)
      {
        using (new TimingBlock("GameEngineCeresInProcess create/evaluate pos", TimingBlock.LoggingType.ConsoleWithMemoryTracking))
        {
          for (int i = 0; i < 10; i++)
          {
            if (true)
            {
              GameEngineCeresInProcess engineCeres = new("Ceres", nd, null);
              GameEngineSearchResult searchResult = engineCeres.Search(PositionWithHistory.StartPosition, SearchLimit.NodesPerMove(1));
              Console.WriteLine("evaluated " + searchResult);
              engineCeres.Dispose();
            }
          }
        }
      }

      if (false)
      {
        using (new TimingBlock("NNEvaluator create/evaluate pos", TimingBlock.LoggingType.ConsoleWithMemoryTracking))
        {
          for (int i = 0; i < 10; i++)
          {
            NNEvaluator evaluator = NNEvaluatorFactory.BuildEvaluator(nd, referenceEvaluator);
            NNEvaluatorResult posEval = evaluator.Evaluate(PositionWithHistory.StartPosition.FinalPosition, true);
            Console.WriteLine(posEval);
            evaluator.Shutdown();
          }
          GC.Collect(3);
          GC.WaitForFullGCComplete();
        }
        Console.WriteLine("<CR> to continue....");
        Console.ReadLine();
      }

      if (false)
      {
        using (new TimingBlock("NNEvaluatorSet create/evaluate pos", TimingBlock.LoggingType.ConsoleWithMemoryTracking))
        {
          for (int i = 0; i < 10; i++)
          {
            Console.WriteLine("Create NNEvaluatorSet");
            NNEvaluatorSet nevaluatorSet = new NNEvaluatorSet(nd, true);
            nevaluatorSet.Warmup(false);
            nevaluatorSet.Dispose();
            Console.WriteLine("Dispose NNEvaluatorSet");
          }
        }
      }

      Console.WriteLine("final shutdown");
      referenceEvaluator.Shutdown();
    }

    static bool IsKRP(in Position position)
    {
      if (position.PieceCountOfType(new Piece(SideType.White, PieceType.Queen)) > 0) return false;
      if (position.PieceCountOfType(new Piece(SideType.Black, PieceType.Queen)) > 0) return false;
      if (position.PieceCountOfType(new Piece(SideType.White, PieceType.Bishop)) > 0) return false;
      if (position.PieceCountOfType(new Piece(SideType.Black, PieceType.Bishop)) > 0) return false;
      if (position.PieceCountOfType(new Piece(SideType.White, PieceType.Knight)) > 0) return false;
      if (position.PieceCountOfType(new Piece(SideType.Black, PieceType.Knight)) > 0) return false;

      return true;

    }

  }
}
