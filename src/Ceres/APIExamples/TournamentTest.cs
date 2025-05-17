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
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.OperatingSystem;
using Ceres.Chess;
using Ceres.Chess.Data.Nets;
using Ceres.Chess.GameEngines;
using Ceres.Chess.Games.Utils;
using Ceres.Chess.LC0.NNFiles;
using Ceres.Chess.LC0.WeightsProtobuf;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNBackends.CUDA;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;
using Ceres.Features;
using Ceres.Features.EngineTests;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;
using Ceres.Features.Suites;
using Ceres.Features.Tournaments;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Params;
using Chess.Ceres.PlayEvaluation;
using Pblczero;
using static Microsoft.FSharp.Core.ByRefKinds;

#endregion

namespace Ceres.APIExamples
{
  public static class TournamentTest
  {
    const bool POOLED = false;

    static int CONCURRENCY = POOLED ? 8 : Environment.MachineName.ToUpper().Contains("DEV") ? 3 : 3;
    static int[] OVERRIDE_DEVICE_IDs = /*POOLED ? null*/
       (Environment.MachineName.ToUpper() switch
       {
         var name when name.ToUpper().Contains("SUPER") => new int[] { 0 },
         var name when name.ToUpper().Contains("DGX") => [0, 1, 2, 3],
         var name when name.ToUpper().Contains("HOP") => [0, 1, 2, 3],
         _ => new int[] { 0 }
       });

    static string GPUS_1 = POOLED ? "GPU:0:POOLED" : "GPU:0";
    static string GPUS_2 = POOLED ? "GPU:0:POOLED" : "GPU:0";


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

    static string SFDEV_EXE => SoftwareManager.IsLinux ? @"/home/david/apps/SF/stockfish_dev"
                                                      : @"\\synology\dev\chess\engines\stockfish_dev.exe";
    static string SF17_EXE => SoftwareManager.IsLinux ? @"/home/david/apps/SF/sf16.1"
                                                      : @"\\synology\dev\chess\engines\stockfish17-windows-x86-64-avx2.exe";

    static List<string> extraUCI = null;// new string[] {"setoption name Contempt value 5000" };


    const int SF_NUM_THREADS = 8;
    static GameEngineDef MakeEngineDefStockfish(string id, string exePath, int numThreads = SF_NUM_THREADS, int hashtableSize = -1)
    {
      return new GameEngineDefUCI(id, new GameEngineUCISpec(id, exePath, numThreads,
                           hashtableSize == -1 ? SF_HASH_SIZE_MB() : hashtableSize, TB_PATH, uciSetOptionCommands: extraUCI));
    }



    static string TB_PATH => CeresUserSettingsManager.Settings.TablebaseDirectory;
    static int SF_HASH_SIZE_MB() => HardwareManager.MemorySize > (256L * 1024 * 1024 * 1024)
                                                                ? 4096
                                                                 : 512;

    public static void PreTournamentCleanup()
    {
      if (!RUN_DISTRIBUTED)
      {
        KillCERES();

        File.Delete("Ceres1.log.txt");
        File.Delete("Ceres2.log.txt");
      }
    }


    public enum HeadTestType { Default, Policy, Value };

    /// <summary>
    /// Test code.
    /// </summary>
    public static float Test(GameEngineDef overrideCeresEngine1Def = null,
                             GameEngineDef overrideCeresEngine2Def = null,
                             string overrideNET1 = null, string overrideNET2 = null,
                             HeadTestType headType = HeadTestType.Default,
                             SearchLimit overrideSearchLimit1 = null,
                             SearchLimit overrideSearchLimit2 = null)
    {
      //      PreTournamentCleanup();
      //RunEngineComparisons(); return;

      string NET1_SECONDARY1 = null;
      string NET1 = "TBD";
      string NET2 = "TBD";

      //      NET1 = "790734;1;0;0,753723;0;1;1"; --> 0 +/-10 (new value head)
      //NET1 = "790734;0;1;1,753723;1;0;0"; // 8 +/-8 (new policy head)
      // No obvious progress with T79, 790940 vs 790855 tests at +2 Elo (+/-7) using 1000 nodes/move

      //var pb1 = LC0ProtobufNet.LoadedNet(NET2);
      //pb1.Dump();

      //NET2 = "~T2_LEARNED_LOOKAHEAD_PAPER_TRT|ZeroHistory";


      NET1 = "~T3_DISTILL_512_15_FP16_TRT";
      NET2 = "~T3_DISTILL_512_15_NATIVE";
      NET1 = "~BT4_FP16_TRT";


      //      NET2 = "~T1_DISTILL_256_10_FP16";

      // First 256x10 net
      const string NET_256 = "Ceres:HOP_CL_CLEAN_256_10_FFN6_B1_4bn_fp16_4000006144.onnx";

      // First 256x10 net (but with NLA)
      const string NET_256_NLA = "Ceres:HOP_CL_CLEAN_256_10_FFN6_B1_NLATT_4bn_fp16_4000006144.onnx";

      // 256x9 net with 2x attention. Moderately strong, slow.
      //const string NET_256_NLA_2x = "Ceres:DGX_CL_256_9_FFN3_H16_B1_NLATTN_a2x_42bn_fp16_4200005632.onnx";
      //const string NET_256_NLA_2x_FIX = "Ceres:DGX_CL_256_9_FFN3_H16_B1_NLATTN_a2x_fixd_noval2_42bn_fp16_4199989248.onnx";
      //BAD      const string NET_256_NLA_2xh_FIX = "Ceres:DGX_CL_256_10_FFN4_H16_B1_NLATTN_a2xpartial_fixd_noval2_42bn_fp16_4200005632.onnx";

      // Amazing 256x10 net with AdEMAMix
      const string NET_256_NLA_EM = "Ceres:DGX_EX_256_10_16H_FFN6_NLA_EM_B1_4bn_fp16_4000006144_nc.onnx";

      // First 512x15, without NLA
      const string NET_512_4bn = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_4bn_fp16_4000002048.onnx";
      const string NET_512_NLA_4bn = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_4000006144.onnx";

      // First 512x15, but with NLA (played at TCEC)
      const string NET_512_NLA_52bn = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072.onnx";
      const string NET_512_NLA_52bn_SKINNY = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072_skinny.onnx";

      // First 384, multiboard
      const string NET_384_12_NLA_B4 = "Ceres:DGX_S_384_12_FFN4_H16_NLA_B4_6bn_fp16_6000001024_nc.onnx|4BOARD";

      // Second 384 (but B1 not B4), tried AdEMAMix but very low LR/clip to keep stable
      // (possibly fails when clipping is removed?)
      const string NET_384_12_NLA_EM_4bn = "Ceres:DGX_EX_384_11_16H_FFN4_NLA_EM_B1_4bn_fp16_4000002048.onnx";

      // Third 384, good clean SOAP run
      const string NET_384_12_NLA_SOAP_4bn = "Ceres:3ae496bee329_SP_384_11_16H_FFN4_NLA_EMSh_B1_4bn_fp16_4000006144_nc.onnx";

      // Big 768 run for TCEC (ended with SOAP)
      const string NET_768_15_NLA_9bn = "Ceres:c5f8bf3f7678_S_768_15_FFN3_H24_NLA_B1_SP_9bn_fp16_8999979008_nc.onnx";

      const string NET_512_25_INPROG = "Ceres:b6c528ec4923_SP_512_25_16H_FFN3_NLA_SMOL_SOAP_B1_6bn_fp16_last.onnx";

      string[] MODELS =
      [
        NET_256_NLA_EM,
        NET_384_12_NLA_SOAP_4bn,
        NET_512_NLA_52bn,
        NET_768_15_NLA_9bn,
        NET_512_25_INPROG
      ];



      //      NET2 = "~T1_DISTILL_256_10_FP16";


      //NET2 = "~BT4_FP16_TRT";
      //NET2 = "~T3_512_15_FP16_TRT";

      //      NNEvaluatorDef t1Distill = NNEvaluatorDef.FromSpecification("~T1_512_RL_TRT", "GPU:0#TensorRT16");
      NNEvaluatorDef t1Distill = NNEvaluatorDef.FromSpecification("~T1_DISTILL_256_10_FP16", "GPU:0#TensorRT16");
      NNEvaluatorDef bt4 = NNEvaluatorDef.FromSpecification("~BT4_FP16_TRT", "GPU:0#TensorRT16");

      NNEvaluatorDef evaluatorDefEndgameStrong = new((Position p, NNPositionEvaluationBatchMember[] batchMember) =>
      {
        return p.PieceCount <= 12 ? 1 : 0;
      },
        bt4.Devices[0].Device,
        // new NNDevicesSpecificationString("X").Devices,
        null, null,
        (t1Distill.Nets[0].Net, 1, 1, 1, 1, 1, 1),
        (bt4.Nets[0].Net, 1, 1, 1, 1, 1, 1));


      //      NNEvaluatorDef twoNets = new NNEvaluatorDef(NNEvaluatorNetComboType.WtdAverage, gpu3, null, null, (netT30, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f), (netT40, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f));

      //      NNEvaluatorFactory.CustomDelegate factoryCombo = (string netID, int gpuID, NNEvaluator referenceEvaluator, object options) =>
      //   new NNEvaluatorDynamicByPos([evaluatorCeres, evaluatorLC0], (pos, _) => PIECES.PositionMatches(in pos) ? 0 : 1);


      NET1 = "~T75";
      NET2 = "~T75";
      NET1 = overrideNET1 ?? NET1;
      NET2 = overrideNET2 ?? NET2;
      if (NET2 == "NONE") NET2 = null;

      GPUS_1 = "GPU:0#TensorRT16";// overrideNET1 != null ? "GPU:0" : "GPU:0#TensorRT16";
      GPUS_2 = "GPU:0#TensorRT16";

#if NOT
  To build Torchscript evaluator:
  const string BASE_FNAME = "b6c528ec4923_SP_512_25_16H_FFN3_NLA_SMOL_SOAP_B1_6bn_fp16_1399996416";
  string FN_BASELINE = "Ceres:" + BASE_FNAME + ".onnx";,
  string FN_TS = "Ceres:" + BASE_FNAME.Replace("_fp16", "") + ".ts";
  NET1 = FN_TS;
  GPUS_1 = "GPU:0#Torchscript";
  NET2 = FN_BASELINE;
#endif

      SearchLimit limit1 = SearchLimit.NodesPerMove(500); //with 100 game pairs 20=78sec, 50=105sec
      //limit1 = SearchLimit.BestValueMove;
      if (headType == HeadTestType.Policy)
      {
        limit1 = SearchLimit.NodesPerMove(1);
      }
      else if (headType == HeadTestType.Value)
      {
        limit1 = SearchLimit.BestValueMove;
      }

      //      limit1 = new SearchLimit(SearchLimitType.SecondsForAllMoves, 60, false, 0.1f);
      SearchLimit limit2 = limit1;
      //limit2 *= 0.75f;

      //      limit1 = limit2 = SearchLimit.SecondsForAllMoves(30, .5f);

      if (overrideSearchLimit1 != null)
      {
        limit1 = overrideSearchLimit1;
      }

      if (overrideSearchLimit2 != null)
      {
        limit2 = overrideSearchLimit2;
      }
      Console.WriteLine();

#if STOCKFISH_GOOD_LIMIT
SearchLimit limit1 = SearchLimit.NodesPerMove(500);
SearchLimit limit2 = SearchLimit.NodesPerMove(350_000);

or better (timed):

SF +46 Elo (+/9) on DEV
BT4 800 nodes vs SF17 0.20sec (6 threads)
  <Ceres1 SearchLimit=<NM  800 nodes> Def=<Ceres1 using <Nets: Single <ONNXViaTRT BT4-1024x15x32h-swa-6147500.pb.gz_fp16 FP16>  Device=GPU:0#TensorRT >>>
  <SF17 SearchLimit=<SM 0.20s> Def=<<SF17 using \\synology\dev\chess\engines\stockfish17-windows-x86-64-avx2.exe NumThreads=6 HashSizeMB=64 SyzygyPath=i:\sygyzy\5and6man;i:\sygyzy\7man >>>
---------------------------------------------------------------------------------------------
|         Player          |  Elo   | +/- | CFS(%) |    W-D-L    |    Time    |   NPS-avg    |
---------------------------------------------------------------------------------------------
|          SF17*          |  0.0   | --- |  ----  | +175=351-94 |  5576.70   |    6,332,002 |
|           BT4           |  -46   |  9  |   0%   | +94=351-175 |  7196.43   |        5,707 |
---------------------------------------------------------------------------------------------

#endif


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


      if (false)
      {
        //        var nec1 = NNEvaluator.FromSpecification("ONNX_TRT:d:\\weights\\lczero.org\\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.16.gz", "GPU:1");
        var nec1 = NNEvaluator.FromSpecification(NET1, "GPU:0");
        nec1.CalcStatistics(false, 2);
        Console.WriteLine(nec1.EstNPSSingleton + " " + nec1.EstNPSBatch);

        var nec2 = NNEvaluator.FromSpecification(NET2, "GPU:0");
        nec2.CalcStatistics(false, 2);
        Console.WriteLine(nec2.EstNPSSingleton + " " + nec2.EstNPSBatch);

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

      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(NET1, GPUS_1);
      NNEvaluatorDef evalDef2 = NET2 == null ? null : NNEvaluatorDefFactory.FromSpecification(NET2, GPUS_2);

      //      evalDef1 = evaluatorDefEndgameStrong;
      //      evalDef2 = t1Distill;

      NNEvaluatorDef? evalDefSecondary1 = null;
      if (NET1_SECONDARY1 != null)
      {
        evalDefSecondary1 = NNEvaluatorDefFactory.FromSpecification($@"LC0:{NET1_SECONDARY1}", GPUS_1);
      }

      NNEvaluatorDef? evalDefSecondary2 = null;


      // Don't output log if very small games
      // (to avoid making very large log files or slowing down play).
      bool outputLog = false;// limit1.EstNumSearchNodes(0, 20000, true) > 50_000;
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", evalDef1, evalDefSecondary1, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres1.log.txt" : null);
      GameEngineDefCeres engineDefCeres2 = null;
      GameEngineDefCeres engineDefCeres3 = null;

      if (evalDef2 != null)
      {
        engineDefCeres2 = new GameEngineDefCeres("Ceres2", evalDef2, evalDefSecondary2, new ParamsSearch(), new ParamsSelect(),
                                                 null, outputLog ? "Ceres2.log.txt" : null);
        engineDefCeres3 = new GameEngineDefCeres("Ceres3", evalDef2, evalDefSecondary2, new ParamsSearch(), new ParamsSelect(),
                                                  null, outputLog ? "Ceres3.log.txt" : null);
      }
      else
      {
        // Substitute engine1 for others so we don't get null dereferences below in steup code
        engineDefCeres2 = engineDefCeres1;
        engineDefCeres3 = engineDefCeres1;
      }
      //      engineDefCeres1.SearchParams.ValueTemperature = 0.85f;
      //      engineDefCeres1.SearchParams.HistoryFillIn = false;

      //engineDefCeres1.OverrideLimitManager = new  Ceres.MCTS.Managers.Limits.ManagerGameLimitTest();
      if (false)
      {
        //engineDefCeres1.OverrideLimitManager = new MCTS.Managers.Limits.ManagerGameLimitCeresL();
        //engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
        //engineDefCeres1.SearchParams.EnableInstamoves = false;
      }

      //engineDefCeres1.SearchParams.BestMoveMode = ParamsSearch.BestMoveModeEnum.TopN;

      //engineDefCeres1.SearchParams.ActionHeadSelectionWeight = 0.5f;
      //engineDefCeres1.SearchParams.Execution.MaxBatchSize = 3;
      //engineDefCeres2.SearchParams.Execution.MaxBatchSize = 3;

      //engineDefCeres1.SelectParams.MinimaxSurpriseMultiplier = 0.10f;
      //      engineDefCeres1.SelectParams.CPUCT *= 0.70f;
      // engineDefCeres1.SelectParams.CPUCT *= 0.65f;
      //      engineDefCeres2.SelectParams.CPUCT *= 0.55f; // .80->42, 0.70
      //engineDefCeres1.SelectParams.PolicySoftmax *= 0.85f;

      //engineDefCeres1.SelectParams.TestFlag = true; // <------------ ************************************* ---------->
      //engineDefCeres1.SelectParams.CPUCT *= 0.6f;

      engineDefCeres1.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
      engineDefCeres2.SearchParams.ReusePositionEvaluationsFromOtherTree = false;

      if (true)
      {
        engineDefCeres1.SearchParams.Execution.FlowDualSelectors = false;
        engineDefCeres2.SearchParams.Execution.FlowDualSelectors = false;
        engineDefCeres1.SearchParams.Execution.FlowDirectOverlapped = false;
        engineDefCeres2.SearchParams.Execution.FlowDirectOverlapped = false;
      }

      engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
      engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled = false;


      if (false)
      {
        engineDefCeres1.SearchParams.EnableTablebases = false;
        engineDefCeres2.SearchParams.EnableTablebases = false;

        engineDefCeres1.SearchParams.Execution.MaxBatchSize = 1;
        engineDefCeres2.SearchParams.Execution.MaxBatchSize = 1;

        engineDefCeres1.SearchParams.TreeReuseEnabled = false;
        engineDefCeres2.SearchParams.TreeReuseEnabled = false;


        engineDefCeres1.SearchParams.Execution.SelectParallelEnabled = false;
        engineDefCeres2.SearchParams.Execution.SelectParallelEnabled = false;

        engineDefCeres1.SearchParams.Execution.FlowDirectOverlapped = false;
        engineDefCeres2.SearchParams.Execution.FlowDirectOverlapped = false;

        engineDefCeres1.SearchParams.Execution.FlowDualSelectors = false;
        engineDefCeres2.SearchParams.Execution.FlowDualSelectors = false;

        engineDefCeres1.SearchParams.Execution.TranspositionMode = TranspositionMode.None;
        engineDefCeres2.SearchParams.Execution.TranspositionMode = TranspositionMode.None;
      }

#if NOT
 engineDefCeres1.SearchParams.ActionHeadSelectionWeight = 0.3333f;
//engineDefCeres2.SearchParams.ActionHeadSelectionWeight = 0.666f;

//engineDefCeres1.SelectParams.FPUValue = 0;
//engineDefCeres1.SelectParams.FPUValueAtRoot = 0;
#endif


      //engineDefCeres1.SearchParams.EnableTablebases = false;
      //engineDefCeres2.SearchParams.EnableTablebases = false;

      //engineDefCeres1.SearchParams.Execution.MaxBatchSize = 128;
      //      engineDefCeres1.SearchParams.BatchSizeMultiplier = 2;

      //      engineDefCeres1.SearchParams.EnableSearchExtension = false;
      //      engineDefCeres2.SearchParams.EnableSearchExtension = false;
      //engineDefCeres1.SearchParams.TestFlag = true;
      //      engineDefCeres1.SearchParams.Execution.FlowDualSelectors = false;
      //      engineDefCeres1.SearchParams.TranspositionRootPolicyBlendingFraction = 0.5f;

      //      engineDefCeres1.SearchParams.TestFlag = true;
      //      engineDefCeres1.SearchParams.EnableUncertaintyBoosting = true;
      //      engineDefCeres2.SearchParams.EnableUncertaintyBoosting = true;


      //      engineDefCeres2.SearchParams.Execution.TranspositionMode = TranspositionMode.SingleNodeCopy;
      // engineDefCeres1.SearchParams.TreeReuseRetainedPositionCacheEnabled = true;
      //engineDefCeres2.SearchParams.TreeReuseRetainedPositionCacheEnabled = true;

      //engineDefCeres2.SearchParams.TestFlag2 = true;

      //      AdjustSelectParamsNewTune_Lc0_TCEC_2024(engineDefCeres1.SelectParams);
      //      AdjustSelectParamsNewTune(engineDefCeres2.SelectParams);
      //      engineDefCeres1.SelectParams.PolicySoftmax *= 0.9f;
      //      engineDefCeres1.SelectParams.CPUCT *= 0.80f;
      //      engineDefCeres2.SelectParams.CPUCT *= 0.80f;
      //engineDefCeres1.SelectParams.CPUCT *= 0.75f;
      //      engineDefCeres2.SelectParams.CPUCT *= 0.75f;
      //    
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

      EnginePlayerDef playerCeres1 = new EnginePlayerDef(overrideCeresEngine1Def ?? engineDefCeres1, limit1);
      EnginePlayerDef playerCeres2 = new EnginePlayerDef(overrideCeresEngine2Def ?? engineDefCeres2, limit2);
      EnginePlayerDef playerCeres3 = new EnginePlayerDef(engineDefCeres3, limit1);

      bool ENABLE_LC0_1 = evalDef1.Nets[0].Net.Type == NNEvaluatorType.LC0;// && (evalDef1.Nets[0].WeightValue == 1 && evalDef1.Nets[0].WeightPolicy == 1 && evalDef1.Nets[0].WeightM == 1);
      bool ENABLE_LC0_2 = evalDef2 != null && evalDef2.Nets[0].Net.Type == NNEvaluatorType.LC0;// && (evalDef1.Nets[0].WeightValue == 1 && evalDef1.Nets[0].WeightPolicy == 1 && evalDef1.Nets[0].WeightM == 1);

      string OVERRIDE_LC0_EXE = null;// @"C:\apps\lc0_30_onnx_dml\lc0.exe";
      string OVERRIDE_LC0_BACKEND_STRING = "";
      GameEngineDefLC0 engineDefLC1 = ENABLE_LC0_1 ? new GameEngineDefLC0("LC0_0", evalDef1, forceDisableSmartPruning, null, null, overrideEXE: OVERRIDE_LC0_EXE, overrideBackendString: OVERRIDE_LC0_BACKEND_STRING) : null;
      GameEngineDefLC0 engineDefLC2 = ENABLE_LC0_2 ? new GameEngineDefLC0("LC0_2", evalDef2, forceDisableSmartPruning, null, null, overrideEXE: OVERRIDE_LC0_EXE, overrideBackendString: OVERRIDE_LC0_BACKEND_STRING) : null;

      EnginePlayerDef playerStockfish17 = new EnginePlayerDef(MakeEngineDefStockfish("SF17", SF17_EXE, hashtableSize: 64, numThreads: 6), limit2);// * 350);
      EnginePlayerDef playerLC0 = ENABLE_LC0_1 ? new EnginePlayerDef(engineDefLC1, limit1) : null;
      EnginePlayerDef playerLC0_2 = ENABLE_LC0_2 ? new EnginePlayerDef(engineDefLC2, limit2) : null;


      const bool RUN_SUITE = false;
      if (RUN_SUITE)
      {
        // NET2 = null;

        Task keyDetectionTask = Task.Run(() => DetectKeyPresses());

        static void DetectKeyPresses()
        {
          while (true)
          {
            if (Console.KeyAvailable)
            {
              var key = Console.ReadKey(true);
              if (key.Modifiers == ConsoleModifiers.Control && key.Key == ConsoleKey.S)
              {
                Console.WriteLine("Ctrl-S pressed!");
              }
            }
            Task.Delay(100).Wait();
          }
        }

        NNEvaluator bmEval1 = NNEvaluator.FromSpecification(NET1, GPUS_1);
        NNEvaluator bmEval2 = NET2 == null ? null : NNEvaluator.FromSpecification(NET2, GPUS_2);

        if (overrideNET1 == null)
        {
          Console.WriteLine();
          Console.WriteLine("BENCHMARK");
          Console.WriteLine("1: " + NNEvaluatorBenchmark.EstNPS(bmEval1, bigBatchSize: 256).NPSBigBatch);
          Console.WriteLine("1: " + NNEvaluatorBenchmark.EstNPS(bmEval1, bigBatchSize: 256).NPSBigBatch);
          if (bmEval2 != null)
          {
            Console.WriteLine("2: " + NNEvaluatorBenchmark.EstNPS(bmEval2, bigBatchSize: 256).NPSBigBatch);
            Console.WriteLine("2: " + NNEvaluatorBenchmark.EstNPS(bmEval2, bigBatchSize: 256).NPSBigBatch);
          }
          Console.WriteLine();
        }

        string BASE_NAME = "lichess_db_puzzle.epd"; //"hardtestpos-clean.epd";// "ERET_VESELY203.epd";//"chad_tactics-100M.epd";//"jhorthos_problem_suite.1.2_half2.epd"; // "Elo2500_10k.epd";// "hard-talkchess-2022.epd";//  "endgame2.epd";// "benchmark.epd";// "endgame2.epd";//  Stockfish238
        ParamsSearch paramsNoFutility = new ParamsSearch() { FutilityPruningStopSearchEnabled = false };

        // ===============================================================================
        string suiteGPU = POOLED ? "GPU:0:POOLED=SHARE1" : GPUS_1;
        string epdName = SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/epd/{BASE_NAME}"
                                                   : @$"\\synology\dev\chess\data\epd\{BASE_NAME}";
        const bool BIG_TEST = true;
        SuiteTestDef suiteDef;
        if (overrideCeresEngine1Def != null)
        {
          suiteDef = new("Suite", epdName, limit1, overrideCeresEngine1Def, overrideCeresEngine2Def, null);
        }
        else
        {
          suiteDef = new("Suite", epdName, limit1,
                                               GameEngineDefFactory.CeresInProcess("Ceres1", NET1, suiteGPU, engineDefCeres1.SearchParams with { FutilityPruningStopSearchEnabled = false }, engineDefCeres1.SelectParams),
                         NET2 == null ? null : GameEngineDefFactory.CeresInProcess("Ceres2", NET2, suiteGPU, engineDefCeres2.SearchParams with { FutilityPruningStopSearchEnabled = false }, engineDefCeres2.SelectParams),
                         null);
        }
        suiteDef.MaxNumPositions = BIG_TEST ? 3500 : 1000; // 38

        suiteDef.EPDLichessPuzzleFormat = suiteDef.EPDFileName.ToUpper().Contains("LICHESS");
        if (suiteDef.EPDLichessPuzzleFormat)
        {
          suiteDef.EPDFilter = epd => !epd.AsLichessDatabaseRecord.Description.Contains("ZZZ")
                                   && epd.AsLichessDatabaseRecord.Rating > (BIG_TEST ? 2600 : 2700);
        }

        suiteDef.AcceptPosPredicate = null;// p => IsKRP(p);

        SuiteTestRunner suiteRunner = new SuiteTestRunner(suiteDef);

        SuiteTestResult suiteResult = suiteRunner.Run(POOLED ? 12 : CONCURRENCY, true, false);
        Console.WriteLine("Max mbytes alloc: " + WindowsVirtualAllocManager.MaxBytesAllocated / (1024 * 1024));
        Console.WriteLine("Test counter 1  : " + MCTSEventSource.TestCounter1);
        Console.WriteLine("Test metric 1   : " + MCTSEventSource.TestMetric1);
        return suiteResult.AvgScore1;
      }

#if NOT
      EnginePlayerDef playerDefCSNoNN = new EnginePlayerDef(engineDefCSNoNN, limit2);
      EnginePlayerDef playerDefCSNN1 = new EnginePlayerDef(engineDefCSNN1, limit2);
      EnginePlayerDef playerDefCSNN50 = new EnginePlayerDef(engineDefCSNN50, limit2);
#endif
      // **************************************************
      EnginePlayerDef player1 = playerCeres1;// playerCeres1UCI;// new EnginePlayerDef(engineDefCSNN1, SearchLimit.NodesPerMove(30));
      EnginePlayerDef player2 = playerCeres2;// new EnginePlayerDef(EnginDefStockfish14(), SearchLimit.NodesPerMove(300 * 10_000));
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

        (string, string)[] nets =
        [
          ("BT4", "~BT4_FP16_TRT"),
          ("Ceres-85",  "Ceres:C1-640-25|TEST85"),
          ("Ceres-75",  "Ceres:C1-640-25|TEST75"),
//          ("SF17", "SF17"),

//          ("NLA_RPE_SP25", "Ceres:39ed759cf2c5_SP_512_10_16H_FFN4_NLA_SP25_B1_RPE_250mm_fp16_250003456.onnx"),
//          ("NLA_RPE_SP5", "Ceres:bf1db067d53b_SP_512_10_16H_FFN4_NLA_SP5_B1_RPE_250mm_fp16_250003456.onnx"),
//          ("NLA_Adam", "Ceres:bf1db067d53b_SP_512_10_16H_FFN4_NLA_Adam_B1_RPE_250mm_fp16_250003456.onnx")

//          ("D256", "~T1_DISTILL_256_10_FP16"),
//          ("SF161", "SF161"),
//          ("C52bn", "Ceres:C:\\dev\\Ceres\\artifacts\\release\\net8.0\\_ctx1.onnx"),
//          ("C52bn", @"Ceres:e:\cout\nets\HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072_skinny_A6000.onnx"),
//          ("C52bn_g", @"Ceres:c:\temp\simple_64bn.onnx"),
//          ("C52bn", "Ceres:/mnt/deve/cout/nets/HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072_skinny_A100.onnx"),
//          ("T3D", "~T3_512_15_FP16_TRT"),
//          ("LC0_T3D", "~T3_512_15_NATIVE"),
        ];

        const float TIME_SCALE = 1.0f;
        foreach (var (name, net) in nets)
        {
          if (name == "SF17")
          {
            const float SF_SCALE_TIME = 0.30f;
            def.AddEngine(playerStockfish17.EngineDef, limit1 * SF_SCALE_TIME);
            def.ReferenceEngineId = "SF17";
          }
          else if (name.StartsWith("LC0_"))
          {
            GameEngineDefLC0 edLC0 = new("LC0", NNEvaluatorDef.FromSpecification(net, "GPU:0"), false);
            def.AddEngine(edLC0, limit1 * TIME_SCALE);
          }
          else
          {
            string GPU_SPEC = "GPU:0#TensorRT16";
            GameEngineDefCeres engineCeres = new(name, NNEvaluatorDef.FromSpecification(net, GPU_SPEC));
            //           engineCeres.SelectParams.CPUCT *= 0.85f;
            def.AddEngine(engineCeres, limit1 * TIME_SCALE);
          }
        }
      }
      else
      {
        def = new TournamentDef("TOURN", player1, player2);
        //        def.ReferenceEngineId = def.Engines[1].ID;

        //        def.CheckPlayer2Def = player1;
      }

      // TODO: UCI engine should point to .NET 6 subdirectory if on .NET 6
      if (isDistributed)
      {
        def.IsDistributedCoordinator = true;
      }


      def.NumGamePairs = 250;
      def.ShowGameMoves = false;

      //string baseName = "tcec1819";
      //      string baseName = "4mvs_+90_+99";
      //      string baseName = "4mvs_+90_+99.epd";

      string baseName = "book-ply8-unifen-Q-0.25-0.40";
      //      baseName = "single_bad";
      //            baseName = "Noomen 2-move Testsuite.pgn";
      //            baseName = "book-ply8-unifen-Q-0.40-1.0";
      const bool KRP = false;
      if (KRP)
      {
        throw new NotImplementedException();
        baseName = "endingbook-16man-9609.pgn";
        //        def.AcceptPosExcludeIfContainsPieceTypeList = [PieceType.Queen, PieceType.Bishop, PieceType.Knight];
      }

      baseName = "tcec_big";
      //baseName = "UHO_Lichess_4852_v1.epd"; // recommended by Kovax
      baseName = "UHO_Lichess_4852_v1_first.epd"; // approximately first 2500 positions for faster loading
                                                  //    baseName = "UHO_Lichess_4582_v1_last_10000.epd";
      if (false)
      {
        baseName = "eco_test";
        def.AcceptGamePredicate = (Game game) => game.PlayerWhite.Contains("French");
      }

      //baseName = "endingbook-10man-3181.pgn";
      //baseName = "endingbook-16man-9609.pgn";
      //      baseName = "endingbook-12man-4624.pgn";

      string postfix = (baseName.ToUpper().EndsWith(".EPD") || baseName.ToUpper().EndsWith(".PGN")) ? "" : ".pgn";
      def.OpeningsFileName = SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/openings/{baseName}{postfix}"
                                                     : @$"\\synology\dev\chess\data\openings\{baseName}{postfix}";
      // not functioning def.AcceptPosPredicate = p => IsKRP(p);  

      //ConsoleUtils.WriteLineColored(ConsoleColor.Red, "WARNING TB ADJUDICATION OFF");
      //def.UseTablebasesForAdjudication = false;

      // Result very sensitive to adjudication, value estimates unreliable.
      // TODO: someday use a strong reference engine for adjudication
      bool DISABLE_ADJUDICATION = false;// limit1.IsHeadTestLimit || limit2.IsHeadTestLimit;
      if (DISABLE_ADJUDICATION)
      {
        def.AdjudicateDrawThresholdCentipawns = 0;
        def.AdjudicateDrawThresholdNumMoves = 999;

        def.AdjudicateWinThresholdCentipawns = int.MaxValue;
        def.AdjudicateWinThresholdNumMovesDecisive = 3000;
        //def.UseTablebasesForAdjudication = false;
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

      if (false)
      {
        Console.WriteLine();
        Console.WriteLine("<CRLF> to continue");
        Console.ReadLine();
      }

      // Compute and return ELO performance of first player
      string firstPlayerID = overrideCeresEngine1Def != null ? overrideCeresEngine1Def.ID : "Ceres1";
      PlayerStat player = results.Players.FirstOrDefault(p => p.Name == firstPlayerID);
      int numDraws = player.NumGames - player.PlayerWins - player.PlayerLosses;
      float eloPerf = EloCalculator.EloDiff(player.PlayerWins, numDraws, player.PlayerLosses);

      return eloPerf;
    }


    public static void TestSF(int index, bool gitVersion)
    {
      // Initialize settings by loading configuration file
      //CeresUserSettingsManager.LoadFromFile(@"c:\dev\ceres\artifacts\release\net5.0\ceres.json");

      // Define constants for engine parameters
      string SF14_EXE = Path.Combine(CeresUserSettingsManager.Settings.DirExternalEngines, "Stockfish14.1.exe");
      const int SF_THREADS = 4;
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
      CompareEngineParams parms = new CompareEngineParams(
            "Resapling", pgnFileName,
            10_000, // number of positions
            s => true,//s.FinalPosition.PieceCount > 15,
            CompareEnginesVersusOptimal.PlayerMode.Ceres, "703810",
            CompareEnginesVersusOptimal.PlayerMode.Ceres, NET_ID,
            CompareEnginesVersusOptimal.PlayerMode.Ceres, RegisteredNets.Aliased["T80"].NetSpecificationString,
            SearchLimit.NodesPerMove(50), // search limit
            new int[] { 0 },//new int[] { 0, 1, 2, 3 },
            s =>
            {
            },
            l =>
            {
            },
            null, // l => l.CPUCT = 1.1f,
            null,
            Verbose: true,
            Engine1LimitMultiplier: 1,
            EngineArbiterLimitMultiplier: 20,
            RunStockfishCrosscheck: true,
            PosResultCallback: null,
            EngineUCIFileNameEXE: exeCeres(),
            QDiffThresholdDumpVerboseMoveStats: 0.25f
           );


      CompareEngineResultSummary result = new CompareEnginesVersusOptimal(parms).Run();
    }

    static void AdjustSelectParamsNewTune_Lc0_TCEC_2024(ParamsSelect p)
    {
      p.CPUCT = 2.897f;
      p.CPUCTAtRoot = p.CPUCT;
      p.CPUCTFactor = 3.973f;
      p.CPUCTFactorAtRoot = p.CPUCTFactor;
      p.CPUCTBase = 45669;
      p.CPUCTBaseAtRoot = 45669;
      p.FPUValue = 0.98416f;
      p.PolicySoftmax = 1.4f;
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
