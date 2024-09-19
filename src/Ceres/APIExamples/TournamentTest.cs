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

    static int CONCURRENCY = POOLED ? 8 : Environment.MachineName.ToUpper().Contains("DEV") ? 1 : 4;
    static int[] OVERRIDE_DEVICE_IDs = /*POOLED ? null*/
       (Environment.MachineName.ToUpper() switch
      {
        var name when name.Contains("DGX") => new int[] { 0, 1, 2, 3 },
        var name when name.Contains("HOP") => new int[] { 0, 1, 2, 3 },
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
    public static void Test(GameEngineDef overrideCeresEngine1Def = null, 
                            GameEngineDef overrideCeresEngine2Def = null)
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
      string NET1 = "TBD";
      string NET2 = "TBD";

      //      NET1 = "790734;1;0;0,753723;0;1;1"; --> 0 +/-10 (new value head)
      //NET1 = "790734;0;1;1,753723;1;0;0"; // 8 +/-8 (new policy head)
      // No obvious progress with T79, 790940 vs 790855 tests at +2 Elo (+/-7) using 1000 nodes/move

      //var pb1 = LC0ProtobufNet.LoadedNet(NET2);
      //pb1.Dump();

      //      NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1675000-rule50.gz#32";
      //      NET1 = "CUSTOM1:703810,CUSTOM1:703810";

      //      NET1 = "CUSTOM1";
      //NET2 = "~BT4_TRT";
      //NET1 = "~T80";
      //NET2 = "~T60";
      //NET2 = "CUSTOM1";
      //NET1 = "~T2";
      //NET2 = "~T2_LEARNED_LOOKAHEAD_PAPER_TRT|ZeroHistory";


      NET1 = "~T3_DISTILL_512_15_FP16_TRT";
      NET2 = "~T3_DISTILL_512_15_NATIVE";
      NET1 = "~BT4_FP16_TRT";
      NET2 = "~BT4";

      //      NET1 = "CUSTOM1:ckpt_DGX_C7_B4_256_10_8_8_32bn_2024_final.ts.fp16.onnx";
//NET2 = "CUSTOM2:ckpt_HOP_C7_256_12_8_6_40bn_B1_2024_postconvert.ts.fp16.onnx"; // best so far

//NET1 = "CUSTOM2:ckpt_HOP_C6_B4_256_12_8_6_BS8_48bn_2024_final.ts"; // old 4board

//NET2 = "CUSTOM1:ckpt_DGX_C7_B4_256_10_8_8_32bn_2024_final.ts.fp16.onnx"; // prep for Daniel
//      NET2 = "CUSTOM2:ckpt_DEV_C6_B4_256_12_8_6_BS8_48bn_2024_postconvert.ts.fp16.onnx";
      //      NET2 = "CUSTOM2:ckpt_HOP_C7_256_12_8_6_40bn_B1_2024_postconvert.ts.fp16.onnx";// 2197929984.ts";
      //NET2 = "CUSTOM2:ckpt_HOP_C7_256_12_8_6_40bn_B1_2024_postconvert.ts.fp16.onnx";// 2197929984.ts";

     //      NET2 = "CUSTOM2:ckpt_HOP_C7_256_12_8_6_40bn_B1_2024_late.ts";// 2197929984.ts";

//      NET2 = "CUSTOM1:ckpt_DGX_C_512_15_16_4_32bn_B1_2024_26Jun_final.ts.fp16.onnx"; // Late June 512

//      NET1 = "CUSTOM1:ckpt_DGX_C_256_12_8_6_4bn_B1_2024_vl01_sf_c3_1bn_vl2p3_last.ts.fp16.onnx";  // extension: value2 loss 0.3 (not 0.1)
//      NET2 = "CUSTOM1:ckpt_DGX_C_256_12_8_6_4bn_B1_2024_vl01_sf_c2_1bn_last.ts.fp16.onnx";          // baseline (+1bn)
//      NET1 = "CUSTOM1:ckpt_DGX_C_256_12_8_6_4bn_B1_2024_vl01_sf_last.ts.fp16.onnx";                 // baseline
//NET2 = "CUSTOM1:HOP_C_256_12_8_6_4bn_B1_2024_wd005_blhead_4000006144.ts.fp16.onnx"; // final blunder->v2, wd 0.005
         //      NET1 = "CUSTOM1:ckpt_DEV_C6_B4_256_12_8_6_BS8_48bn_2024_postconvert.ts.fp16.onnx"; old strong B4 net
        //NET2 = "CUSTOM2:newtest.fp16.onnx";

      //NET1 = "CUSTOM2:ckpt_DGX_C_256_12_8_6_4bn_B1_2024_vl01_sf_c3_1bn_wd001_last.ts.fp16.onnx"; // extension: WD 0.001
      //      NET1 = "CUSTOM1:HOP_C_256_12_8_6_4bn_B1_2024_vl01_sf_c3_1bn_auxtenth_2199994368.ts"; // extension: new data (2024)
      //      NET1 = "CUSTOM1:ckpt_HOP_C6_B4_256_12_8_6_BS8_48bn_2024_final.ts";// old generation (May) good net

      //      NET2 = "CUSTOM1:ckpt_DGX_C_256_12_8_6_60bn_B4_2024_final.ts.fp16.onnx"; // best June2024 (4B)
//      NET1 = "CUSTOM1:HOP_C_GL_RPE_lategelu_ln_256_10_FFN4_4bn_fp16_4000006144.onnx";
//      NET2 = "CUSTOM1:ckpt_HOP_C7_256_12_8_6_40bn_B1_2024_postconvert.ts.fp16.onnx"; // best June2024 (1B)
      //NET2 = "CUSTOM1:HOP_C_GL_RPE_lategelu_ln_256_10_FFN4_4bn_fp16_last.onnx";


      //NET1 = "CUSTOM1:HOP_CL_CLEAN_256_10_FFN6_B1_NLATT_4bn_fp16_x.onnx"; // _4bn_fp16

      //            NET1 = "~T80";

      //      NET2 = "CUSTOM1:HOP_C_256_12_8_6_4bn_B1_2024_vl01_sf_c3_1bn_auxtenth_399998976.ts.fp16.onnx";
      //      NET2 = "CUSTOM1:HOP_C_GL_RPE_lategelu_ln_256_10_FFN4_4bn_fp16_399998976.onnx";

      //      NET2 = "~T1_DISTILL_256_10_FP16";

      const string NET_256        = "Ceres:HOP_CL_CLEAN_256_10_FFN6_B1_4bn_fp16_4000006144.onnx";
      const string NET_256_NLA    = "Ceres:HOP_CL_CLEAN_256_10_FFN6_B1_NLATT_4bn_fp16_4000006144.onnx";
      const string NET_256_NLA_2x = "Ceres:DGX_CL_256_9_FFN3_H16_B1_NLATTN_a2x_42bn_fp16_4200005632.onnx";
      const string NET_256_NLA_2x_FIX = "Ceres:DGX_CL_256_9_FFN3_H16_B1_NLATTN_a2x_fixd_noval2_42bn_fp16_4199989248.onnx";
//BAD      const string NET_256_NLA_2xh_FIX = "Ceres:DGX_CL_256_10_FFN4_H16_B1_NLATTN_a2xpartial_fixd_noval2_42bn_fp16_4200005632.onnx";

      const string NET_256_NLA_EM = "Ceres:DGX_EX_256_10_16H_FFN6_NLA_EM_B1_4bn_fp16_4000006144.onnx";

      const string NET_512_4bn = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_4bn_fp16_4000002048.onnx";
      const string NET_512_NLA_4bn = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_4000006144.onnx";

      const string NET_512_NLA_52bn = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072.onnx";
      const string NET_512_NLA_52bn_SKINNY = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072_skinny.onnx";

      const string NET_384_12_NLA_B4 = "Ceres:DGX_S_384_12_FFN4_H16_NLA_B4_6bn_fp16_6000001024.onnx|4BOARD";
      const string NET_384_12_NLA_B4_NC = "Ceres:DGX_S_384_12_FFN4_H16_NLA_B4_6bn_fp16_6000001024_nc.onnx|4BOARD";


      //      NET2 = "~T1_DISTILL_512_15_FP16";// _TRT";
      //NET1 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_combo_473_485_495.onnx"; // -25@100

      //      NET1 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_c6_fp16_6000001024.onnx";
      //      NET2 = "~BT3_FP16_TRT";

      //      NET2 = "~T3_DISTILL_512_15_FP16_TRT";


      //NET1 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_c6a_fp16_6500007936.onnx";
      //NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_c6a_fp16_6500007936.onnx";
      //      NET1 =  @"Ceres:E:\cout\nets\trt_engines_embed\DEV\HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_4000006144.onnx";
      //      NET2 = "~T1_256_RL_NATIVE";

      //      NET1 = "CUSTOM1:last.onnx";
      //      NET2 = "CUSTOM1:HOP_CL_CLEAN_512_15_FFN4_B1_4bn_fp16_599998464.onnx";
      //      NET2 = "~T1_256_RL_NATIVE";


      //NET2 = "CUSTOM1:ckpt_DGX_C_256_12_8_6_60bn_B4_2024_final.ts.fp16.onnx";
      //      NET2 = "~T1_DISTILL_256_10_FP16";

      //      NET1 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_4000006144.onnx";
      //NET1 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_c6a_fp16_6500007936.onnx"; // -25@100
      //NET1 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072_skinny.onnx";
      //NET2 = "~T4_RPE_512_15_FP16_TRT";


      //      NET1 = NET2 = NET_512_NLA_52bn;

      NET1 = NET_256_NLA_2x;
      NET2 = NET_256_NLA;

      NET2 = "~T1_256_RL_TRT";

      NET1 = "~T3_512_15_Q_FP16_TRT";
      NET2 = "~T3_512_15_FP16_TRT";

      // Test HOP Nadam nets
      //NET1 = "Ceres:4a7f3aec0da2_S_768_15_FFN3_H24_NLA_B1_52_68bn_fp16_last.onnx";
      NET2 = "Ceres:4c10e0e0b1ce_S_768_15_FFN3_H24_NLA_B1_52bn_fp16_5199986688.onnx";// combox_495_511_nc.onnx";
//      NET1 = "Ceres:4a7f3aec0da2_S_768_15_FFN3_H24_NLA_B1_52_68bn_fp16_653bn.onnx"; // policy -2, value 15 +/- 9, 100
//      NET1 = "Ceres:4a7f3aec0da2_S_768_15_FFN3_H24_NLA_B1_52_68bn_fp16_668bn.onnx";  // value 15+/-9 policy 2+/-9, 100 crash/unclear
      NET1 = "Ceres:4a7f3aec0da2_S_768_15_FFN3_H24_NLA_B1_52_68bn_fp16_6800003072_nc.onnx";  // 671: policy +19 +/-9, value 6 +/-9  100 +4
                                                                                             //      NET1 = "Ceres:4a7f3aec0da2_S_768_15_FFN3_H24_NLA_B1_52_68bn_fp16_651bn.onnx"; // 100 nodes +21 +/-11, policy -2 +/-10, value 15 +/- 9

      NET1 = NET_384_12_NLA_B4_NC;
      NET2 = NET_384_12_NLA_B4_NC;

      //      NET2 = NET_512_NLA_4bn;

      //      NET1 = "Ceres:ckpt_DGX_C5_B4_512_15_16_4_48bn_2024_4780638208.ts.fp16.onnx|4BOARD";
      //      NET2 = NET1;
      //      NET2 = NET_512_NLA_52bn;

      //      NET1 = NET_384_12_NLA_B4_NC;
      //      NET2 = NET_384_12_NLA_B4_NC;

      //test 512
      //      NET1 = "Ceres:d63853d8876d_S_512_15_FFN4_H16_NLA_B4_76bn_fp16_1999994880.onnx";
      //      NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_1999994880.onnx";

      // ### TEST 256
      //      NET1 = "Ceres:DGX_EX_256_10_16H_FFN6_NLA_EM_B1_4bn_fp16_last.onnx";
      // TEST 512
      NET1 = "Ceres:19bec799eec4_EX_512_15_16H_FFN4_NLA_MX1_B1_40bn_fp16_1599995904.onnx";      
      NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_1599995904.onnx";

      // TEST 384
//      NET1 = "Ceres:DGX_EX_384_11_16H_FFN4_NLA_EM_B1_4bn_fp16_1999994880.onnx";      
//      NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_1999994880.onnx";


//      NET2 = "Ceres:HOP_CL_CLEAN_256_10_FFN6_B1_NLATT_4bn_fp16_3599990784.onnx";
//NET2 = NET_256_NLA;
//NET2 = "~T1_256_RL_TRT";

      // ### TEST 512
      //NET1 = "Ceres:c58d5d5a597d_EX_512_15_16H_FFN4_NLA_MX_B1_45bn_fp16_1799995392.onnx";
      //NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_1799995392.onnx";

      //NET1 = "Ceres:B4_384_48bn_A6000.onnx";
      //      NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_2799992832.onnx";
      //      NET2 = NET_256_NLA;
      //      NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_3599990784.onnx";
      //      NET2 = "~T4_RPE_512_15_FP16_TRT";
      //      NET2 = NET_512_NLA_52bn;
      //      NET2 = "~T1_256_RL_TRT";

      GPUS_1 = "GPU:0#TensorRT16";
      GPUS_2 = "GPU:0#TensorRT16";

      //      NET2 = "Ceres:4c10e0e0b1ce_S_768_15_FFN3_H24_NLA_B1_52bn_fp16_last.onnx";
      //      NET2 = "~BT5_FP16_TRT";
      //NET1 = NET_512_NLA_4bn + "|TEST";
//      NET2 = NET_512_NLA_4bn;
//      NET2 = "~BT4_FP16_TRT";

      //NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_3399991296.onnx";

      //NET2 = "Ceres:585cb2203243_S_768_15_FFN3_H24_NLA_B1_52bn_fp16_999997440.onnx";
      //      NET2 = NET_512_NLA_4bn;
      //      NET2 = "~BT4_600k";
      //NET1 = 
      //      NET2 = "~T1_256_RL_TRT";
      //      NET2 = "~T1_DISTILL_512_15_FP16";
      //       NET1 = "~T3_512_15_FP16_TRT";
      //      NET1 = "~T4_RPE_512_15_FP16_TRT";
      //NET1 = "~BT2_FP16_TRT";
      //      NET1 = "~BT4_SPSA_FP16_TRT";
      //      NET1 = "~T3_DISTILL_512_15_FP16_TRT";

      //      NET1 = "Ceres:HOP_CL_384_11_FFN3_H32_B1_NLA_a2x_nowd_5bn_fp16_last.onnx";      // crashed
      //      NET2 = "Ceres:HOP_CL_CLEAN_256_10_FFN6_B1_4bn_fp16_199999488.onnx"; // weak
      // 384 vs 512      
      //NET1 = "Ceres:HOP_CL_384_11_FFN3_H32_B1_NLA_a2x_nowd_ndm_5bn_fp16_199999488.onnx";
      //      NET1 = "Ceres:HOP_CL_CLEAN_384_11_FFN3_H32_B1_NLATTN_a2x_42bn_fp16_last.onnx";
      //      NET2 = "Ceres:HOP_CL_384_11_FFN3_H32_B1_NLA_a2x_nowd_5bn_fp16_199999488.onnx";
      //NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_199999488.onnx";

      //      NET2 = "Ceres:HOP_CL_CLEAN_384_11_FFN3_H32_B1_NLATTN_a2x_42bn_fp16_199999488.onnx";
      
      //      NET1 = NET_512_NLA_52bn;// "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072_skinny_A6000.onnx";
      //      NET2 = "~BT4_NATIVE";

      //      NET1 = NET_512;
      //      NET2 = "~T3_DISTILL_512_15_FP16_TRT";


      //  NET1 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_4000006144";
      //  NET2 = "Ceres:HOP_CL_CLEAN_512_15_FFN4_B1_4bn_fp16_4000002048";

      //NET2 = "~T4_RPE_512_15_NATIVE";
      //      NET2 = "~T81";
      //      NET1 = "CUSTOM1:753723;1;0;0;1,~T1_DISTILL_512_15;0;1;1;0";
      //NET1 = "ONNX_ORT:BT3_750_policy_vanilla#32,ONNX_ORT:BT3_750_policy_optimistic#32";
      //      NET1 = "~BT4|1.26"; // 1.26 -13+/-13
      //NET2 = "~T1_DISTIL_512_15_NATIVE";


      SearchLimit limit1 = SearchLimit.NodesPerMove(1); // 33
      limit1 = SearchLimit.BestValueMove;
//      limit1 = new SearchLimit(SearchLimitType.SecondsForAllMoves, 60, false, 0.1f);
      //SearchLimit limit2 = SearchLimit.NodesPerMove(1);
      
      SearchLimit limit2 = limit1;
      //      limit1 = SearchLimit.BestActionMove;
      //SearchLimit limit2 = SearchLimit.NodesPerMove(350_000);
      //      limit1 = limit2 = SearchLimit.SecondsForAllMoves(30, 0.5f);

#if STOCKFISH_GOOD_LIMIT
SearchLimit limit1 = SearchLimit.NodesPerMove(500);
SearchLimit limit2 = SearchLimit.NodesPerMove(350_000);

#endif

      //NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.gz#32";
      //NET2 = @"ONNX_ORT:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.noscale.gz#32";
      //NET1 = @"LC0:d:\weights\lczero.org\BT2-768x15smolgen-12h-do-01-swa-onnx-1800000-rule50.pb.gz#32";

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

      NNEvaluatorDef evalDef1 =  NNEvaluatorDefFactory.FromSpecification(NET1, GPUS_1);
      NNEvaluatorDef evalDef2 =  NNEvaluatorDefFactory.FromSpecification(NET2, GPUS_2);

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

      //engineDefCeres1.SearchParams.BestMoveMode = ParamsSearch.BestMoveModeEnum.TopQIfSufficientN;

//engineDefCeres1.SearchParams.ActionHeadSelectionWeight = 0.5f;
//engineDefCeres1.SearchParams.Execution.MaxBatchSize = 3;
//engineDefCeres2.SearchParams.Execution.MaxBatchSize = 3;



      engineDefCeres1.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
      engineDefCeres2.SearchParams.ReusePositionEvaluationsFromOtherTree = false;

      if (true)
      {
        engineDefCeres1.SearchParams.Execution.FlowDualSelectors = false;
        engineDefCeres2.SearchParams.Execution.FlowDualSelectors = false;
        engineDefCeres1.SearchParams.Execution.FlowDirectOverlapped = false;
        engineDefCeres2.SearchParams.Execution.FlowDirectOverlapped = false;
      }
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

      //      engineDefCeres1.SearchParams.ResamplingMoveSelectionFractionMove = 1f;
      //      engineDefCeres1.SearchParams.EnableSearchExtension = false;
      //      engineDefCeres2.SearchParams.EnableSearchExtension = false;

      //engineDefCeres1.SearchParams.TestFlag2 = true; // XXX
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
      bool ENABLE_LC0_2 = evalDef2.Nets[0].Net.Type == NNEvaluatorType.LC0;// && (evalDef1.Nets[0].WeightValue == 1 && evalDef1.Nets[0].WeightPolicy == 1 && evalDef1.Nets[0].WeightM == 1);

      string OVERRIDE_LC0_EXE = null;// @"C:\apps\lc0_30_onnx_dml\lc0.exe";
string OVERRIDE_LC0_BACKEND_STRING = "";
      GameEngineDefLC0 engineDefLC1 = ENABLE_LC0_1 ? new GameEngineDefLC0("LC0_0", evalDef1, forceDisableSmartPruning, null, null, overrideEXE: OVERRIDE_LC0_EXE, overrideBackendString: OVERRIDE_LC0_BACKEND_STRING) : null;
      GameEngineDefLC0 engineDefLC2 = ENABLE_LC0_2 ? new GameEngineDefLC0("LC0_2", evalDef2, forceDisableSmartPruning, null, null, overrideEXE: OVERRIDE_LC0_EXE, overrideBackendString: OVERRIDE_LC0_BACKEND_STRING) : null;

      EnginePlayerDef playerStockfish17 = new EnginePlayerDef(MakeEngineDefStockfish("SF17", SF17_EXE), limit2);// * 350);
      EnginePlayerDef playerLC0 = ENABLE_LC0_1 ? new EnginePlayerDef(engineDefLC1, limit1) : null;
      EnginePlayerDef playerLC0_2 = ENABLE_LC0_2 ? new EnginePlayerDef(engineDefLC2, limit2) : null;


      const bool RUN_SUITE = false;
      if (RUN_SUITE)
      {
//        NET2 = null;

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

        NNEvaluator bmEval1 = NNEvaluator.FromSpecification(NET1, "GPU:0#TensorRT16");
        NNEvaluator bmEval2 = NET2 == null ? null : NNEvaluator.FromSpecification(NET2, "GPU:0#TensorRT16");

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

        string BASE_NAME = "endgame2.epd";// "chad_tactics-100M.epd";//"ERET_VESELY203.epd";//  eret nice_lcx Stockfish238 ERET_VESELY203 endgame2 chad_tactics-100M lichess_chad_bad.csv
        ParamsSearch paramsNoFutility = new ParamsSearch() { FutilityPruningStopSearchEnabled = false };

        //Z:\chess\data\epd>type lichess.csv 
//        BASE_NAME = "lichess.csv";

        // ===============================================================================
        string suiteGPU = POOLED ? "GPU:0:POOLED=SHARE1" : "GPU:0";
        SuiteTestDef suiteDef =
          new SuiteTestDef("Suite",
                           SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/epd/{BASE_NAME}"
                                                   : @$"\\synology\dev\chess\data\epd\{BASE_NAME}",
                           limit1,
                           GameEngineDefFactory.CeresInProcess("Ceres1", NET1, suiteGPU, paramsNoFutility with {  }),
                           NET2 == null ? null :  GameEngineDefFactory.CeresInProcess("Ceres2", NET2, suiteGPU, paramsNoFutility with { }),
                           null);// engineDefCeres96);// playerLC0.EngineDef);

        suiteDef.MaxNumPositions = 1000;
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
//          ("D256", "~T1_DISTILL_256_10_FP16"),
          ("SF161", "SF161"),
//          ("C52bn", "Ceres:C:\\dev\\Ceres\\artifacts\\release\\net8.0\\_ctx1.onnx"),
//          ("C52bn", @"Ceres:e:\cout\nets\HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072_skinny_A6000.onnx"),
          ("C52bn_g", @"Ceres:c:\temp\simple_64bn.onnx"),
//          ("C52bn", "Ceres:/mnt/deve/cout/nets/HOP_CL_CLEAN_512_15_FFN4_B1_NLATTN_4bn_fp16_5200003072_skinny_A100.onnx"),
//          ("T3D", "~T3_512_15_FP16_TRT"),
//          ("LC0_T3D", "~T3_512_15_NATIVE"),
        ];

        const float TIME_SCALE = 1f;
        foreach (var (name, net) in nets)
        {
          if (name == "SF161")
          {
            def.AddEngine(playerStockfish17.EngineDef, limit1 * TIME_SCALE);
            def.ReferenceEngineId = "SF161";
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


      def.NumGamePairs = 500;// 2000;// 10_000;// 203;//1000;//203;//203;// 500;// 203;//203;// 102; 203
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
       baseName = "endingbook-10man-3181.pgn";
      baseName = "book-ply8-unifen-Q-0.25-0.40";
      const bool KRP =false;
      if (KRP)
      {
        throw new NotImplementedException();
//        baseName = "endingbook-16man-9609.pgn";
//        def.AcceptPosExcludeIfContainsPieceTypeList = [PieceType.Queen, PieceType.Bishop, PieceType.Knight];
      }
//       baseName = "tcec_big";
  baseName = "UHO_Lichess_4852_v1.epd"; // recommended by Kovax
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
