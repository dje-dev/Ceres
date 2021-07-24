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
using Ceres.Chess;

using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Features.Players;
using Ceres.Features.Suites;
using Ceres.Features.Tournaments;
using Ceres.Features.GameEngines;
using Ceres.MCTS.Params;
using Ceres.Base.Benchmarking;
using System.IO;
using System.Diagnostics;
using System.Threading.Tasks;
using System.Threading;
using Ceres.Chess.NNEvaluators;
using Chess.Ceres.NNEvaluators;
using Ceres.Features;
using Ceres.MCTS.Iteration;
using Ceres.Base.Math;
using Ceres.Base.Misc;
using Ceres.Base.DataType;
using Ceres.Base.OperatingSystem;
using System.Collections.Generic;

#endregion

namespace Ceres.APIExamples
{
  public static class TournamentTest
  {
    static int CONCURRENCY = 4; // POOLED ? 16 : 4;
    static bool RUN_DISTRIBUTED = false;

    
    private static void KillCERES()
    {
      foreach (Process p in Process.GetProcesses())
      {
        if (p.ProcessName.ToUpper().StartsWith("CERES") && p.Id != Process.GetCurrentProcess().Id)
          p.Kill();
      }
    }

    const string ETHERAL_EXE = @"\\synology\dev\chess\engines\Ethereal12.75-x64-popcnt-avx2.exe";
    const string SF11_EXE = @"\\synology\dev\chess\engines\stockfish_11_x64_bmi2.exe";
    const string SF12_EXE = @"\\synology\dev\chess\engines\stockfish_20090216_x64_avx2.exe";
    static string SF14_EXE => SoftwareManager.IsLinux ? @"/raid/dev/SF14/stockfish_14_linux_x64_avx2"
                                                      : @"\\synology\dev\chess\engines\stockfish_14_x64_avx2.exe";

    static GameEngineUCISpec specEthereal = new GameEngineUCISpec("Ethereal12", ETHERAL_EXE);
    static GameEngineUCISpec specSF14 = new GameEngineUCISpec("SF14", SF14_EXE);
    static GameEngineUCISpec specLC0 = new GameEngineUCISpec("LC0", "lc0.exe");

    static List<string> extraUCI = null;// new string[] {"setoption name Contempt value 5000" };
    static GameEngineDef engineDefEthereal = new GameEngineDefUCI("Etheral", new GameEngineUCISpec("Etheral", ETHERAL_EXE, SF_NUM_THREADS, SF_HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));
    static GameEngineDef engineDefStockfish11 = new GameEngineDefUCI("SF11", new GameEngineUCISpec("SF11", SF11_EXE, SF_NUM_THREADS, SF_HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));

    public static GameEngineDef EngineDefStockfish14(int numThreads = SF_HASH_SIZE_MB, int hastableSize = SF_NUM_THREADS) =>
      new GameEngineDefUCI("SF14", new GameEngineUCISpec("SF14", SF14_EXE, numThreads,
                           hastableSize, TB_PATH, uciSetOptionCommands: extraUCI));

    const int SF_NUM_THREADS = 8;
    static string TB_PATH => CeresUserSettingsManager.Settings.TablebaseDirectory;
    const int SF_HASH_SIZE_MB = 2048;

    public static void PreTournamentCleanup()
    {
      //      KillCERES();

      File.Delete("Ceres1.log.txt");
      File.Delete("Ceres2.log.txt");
    }


    /// <summary>
    /// Test code.
    /// </summary>
    public static void Test()
    {
      PreTournamentCleanup();

      if (false)
      {
        Parallel.Invoke(() => TestSF(0, true), () => { Thread.Sleep(7_000); TestSF(1, false); });
        System.Environment.Exit(3);
      }

      const bool POOLED = false;
      string GPUS = POOLED ? "GPU:1,2,3:POOLED"
                           : "GPU:0";
      //LC0:40x512-lr015-swa-115000 20b_1-15000.pb.gz 20b_1-15000 20b_1-swa-90000
      // 460000 best? 20b_500000 teck45_190k.gz
      // /raid/train/nets/40b/40b/40b-swa-20000.pb.gz
      // 20b_500000 vs LC0:66511
      // teck45_170k 40b-swa-80000
      // test-250
      //      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification("LC0:test-swa-2500", GPUS); // j64-210 LS16 40x512-lr015-swa-167500
      //      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification("LC0:40bs-swa-212500", GPUS); // j64-210 LS16 40x512-lr015-swa-167500
      //      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification("LC0:40bs-swa-210000", GPUS); // j64-210 LS16 40x512-lr015-swa-167500

      //      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification("LC0:20b_500000", GPUS); // j104.1-30 61339 j64-210
      // badgyal-3 testFILL96s2-swa-10000 testFILL96noqs2-swa-15000 751156
      //      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(@"LC0:d:\nets\20b_128\20b_128\20b_128-swa-40000.pb.gz", GPUS); // j64-210 LS16 40x512-lr015-swa-167500
      //      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification(@"LC0:751156", GPUS); // j64-210 LS16 40x512-lr015-swa-167500
      //      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification(@"LC0:d:\temp\20b_128-swa-40000.pb.gz", GPUS); // j104.1-30 61339 j64-210

      //      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(@"LC0:d:\nets\tinkerF_biasReg2\tinkerF_biasReg2-12500.pb.gz", GPUS); // j64-210 LS16 40x512-lr015-swa-167500

      string NET1 = @"d:\weights\lczero.org\t64df-5000.pb.gz"; // direct Linux T75 192
      string NET2 = @"d:\weights\lczero.org\t64dbm-5000.pb.gz";

      NET1 = @"d:\weights\lczero.org\t60apr2k-swa-227500.pb.gz";
      NET2 = @"d:\weights\lczero.org\t60apr4k-67500.pb.gz";
      NET2 = @"703810";


      NET1 = @"d:\weights\lczero.org\t60apr2kmdb-swa-47500.pb.gz"; ///raid/train/nets/t60apr2kmdb/t60apr2kmdb-swa-47500.pb.gz
      NET2 = @"D:\weights\lczero.org\t60apr4k-67500.pb.gz";

      NET2 = @"d:\weights\lczero.org\t60apr4kMDB-swa-125000.pb.gz";
//      NET2 = @"d:\weights\lczero.org\t60apr2kmdb-swa-250000.pb.gz";
      NET2 = @"d:\weights\lczero.org\t60apr2kmdc-swa-250000.pb.gz";
      //      NET1 = @"d:\weights\lczero.org\t60apr2kmdb-swa-50000.pb.gz";


      NET1 = @"d:\weights\lczero.org\t60apr2kmdb1-swa-200000.pb.gz";
      //      NET2 = @"d:\weights\lczero.org\t60apr2kmdc1-swa-200000.pb.gz";
      //      NET2 = @"d:\weights\lczero.org\t60apr4kMDB1-swa-100000.pb.gz";
      //t60apr2kmdc-250000.pb.gz
      //NET1 = @"d:\weights\lczero.org\512x30b-swa-65000.pb.gz";
      //NET1 = "j94-100";
      NET1 = @"d:\weights\lczero.org\t60apr2kmdc1-swa-200000.pb.gz";
      NET2 = "703810";
      //      NET2 = "751267";
      //      NET2 = "badgyal-3";

      ///raid/train/nets/512x30b/512x30b-swa-112500.pb.gz 975

      ///raid/train/nets/t60apr2kmdc1/t60apr2kmdc1-swa-477500.pb.gz
      //////raid/train/nets/512x30b/512x30b-swa-407500.pb.gz
      NET1 = @"t60apr2kmdc1-swa-787500"; // was 320 t60apr2kmdc1-swa-400000.pb.gz
      NET2 = @"t60apr2kmdc1-swa-700000"; // was 320 t60apr2kmdc1-swa-400000.pb.gz
//      NET2 = @"t60apr2kmdc1-swa-450000"; // was 320 t60apr2kmdc1-swa-400000.pb.gz

      NET1 = @"d:\weights\lczero.org\512x30b-swa-407500.pb.gz"; // 352500
      NET2 = @"d:\weights\lczero.org\512x30b-swa-295000.pb.gz"; ///raid/train/nets/512x30b/512x30b-252500.pb.gz
      //NET2 = @"d:\weights\lczero.org\40bs-swa-182500.pb.gz"; // prior 512x24 effort, LR 0.002 (?)
      //      / raid / train / nets / 512x30b / 512x30b - swa - 155000.pb.gz
//NET2 = "J94-100";
      //NET1 = @"d:\weights\lczero.org\shrink_no-swa-300000.pb.gz";
      //NET2 = @"d:\weights\lczero.org\shrink_no-swa-200000.pb.gz";

//      NET1 = "703810"; // 5+/- 13
//      NET2 = "703810";// j64-210";

      //NET1 = "TK-6430 aka 128x10-BPR-64M-6430000"; // GOOD NET - better thatn 703810
      //      NET2 = "j94-100";
      //      NET1 = @"badgyal-3";
      //      NET2 = @"d:\weights\lczero.org\t60apr4_64-swa-57500.pb.gz"; // direct Linux T75 192

//      NET1 = @"d:\weights\lczero.org\test64a-swa-75000.pb.gz"; // 475
//      NET1 = "test64c_2k-swa-492500";
//      NET1 = "test64c-swa-147500";


// Good 64x6 test
//      NET1 = "test64c_2k-565000";
//      NET2 = "11258-64x6-se";// "badgyal-3";

// Good 128x10b test
//      NET1 = "t60apr2kmdc1-swa-700000";
//      NET2 = @"703810";
string NET1_SECONDARY1 = null; 
      NET1 = "J94-100";
      NET2 = "J94-100";

//      NET1 = "703810";
//      NET2 = "703810";
//      string NET1_SECONDARY1 = "j94-100";
        
      //string       NET2 = @"j64-210";
      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(@$"LC0:{NET1}", GPUS); // j64-210 LS16 40x512-lr015-swa-167500
//evalDef1.SECONDARY_NETWORK_ID = "j94-100";
      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification($@"LC0:{NET2}", GPUS); ;

      NNEvaluatorDef evalDefSecondary1 = null;
      if (NET1_SECONDARY1 != null)
      {
        evalDefSecondary1 = NNEvaluatorDefFactory.FromSpecification($@"LC0:{NET1_SECONDARY1}", GPUS);
      }

      NNEvaluatorDef evalDefSecondary2 = null;


      static NNEvaluatorDef EvaluatorValueOnly(string netID1, string netID2, int gpuID, bool valueNet1)
      {
//        string wtStr1 = valueNet1 ? "1.0;0.5;0.5" : "0.0;0.5;0.5";
//       string wtStr2 = valueNet1 ? "0.0;0.5;0.5" : "1.0;0.5;0.5";
       string wtStr1 = valueNet1 ? "0.5;1.0;0.5" : "0.5;0.0;0.5";
        string wtStr2 = valueNet1 ? "0.5;0.0;0.5" : "0.5;1.0;0.5";
        NNEvaluatorDef spec = NNEvaluatorDef.FromSpecification($"LC0:{netID1}@{wtStr1},{netID2}@{wtStr2}", $"GPU:{gpuID}");
        return spec;
      }

//      evalDef1 = EvaluatorValueOnly(NET1, NET2, 0, true);
//      evalDef2 = EvaluatorValueOnly(NET1, NET2, 0, false);

      //NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(@"ONNX:tinkerF_normal_vq-50000", GPUS); // j64-210 LS16 40x512-lr015-swa-167500
      //d:\weights\lczero.org\tinkerF_normal_vq-5000.onnx
      //evalDef1 = NNEvaluatorDefFactory.FromSpecification("ONNX:tfmodelc", "GPU:0");

      // Good progress. 68240 versus j94-100 yields: +21 (+/- 11) at 1k/move and +15 (+- 11) at 10k/move
      if (false)
      {
        NNEvaluatorNetDef nnd8 = new NNEvaluatorNetDef("59999", NNEvaluatorType.LC0TensorRT, NNEvaluatorPrecision.FP16);
        NNEvaluatorDef nd8 = new NNEvaluatorDef(nnd8, NNEvaluatorDeviceComboType.Single, new NNEvaluatorDeviceDef(NNDeviceType.GPU, 0));
        //        NNEvaluatorNetDef nnd16 = new NNEvaluatorNetDef("62242", NNEvaluatorType.LC0TensorRT, NNEvaluatorPrecision.FP16);
        //        NNEvaluatorDef nd16 = new NNEvaluatorDef(nnd16, NNEvaluatorDeviceComboType.Single, new NNEvaluatorDeviceDef(NNDeviceType.GPU, 0));
        var nd16 = NNEvaluatorDefFactory.FromSpecification("LC0:59999", GPUS); // j104.1-30 61339

        if (true)
        {
          NNEvaluator ntest = NNEvaluatorFactory.BuildEvaluator(nd8);
          ntest.CalcStatistics(false);
          for (int i = 0; i < 5; i++)
          {
            ntest.CalcStatistics(false);
            Console.WriteLine(ntest.PerformanceStats);
          }
        }


        evalDef1 = nd8;
        evalDef2 = nd16;
      }

      // was 703810 @ 50k

      if (POOLED)
      {
        evalDef1.MakePersistent();
        evalDef2.MakePersistent();
      }

      //      SearchLimit slLC0 = SearchLimit.NodesPerMove(10_000);
      //      SearchLimit slEthereal = slLC0 * 875;
      //      SearchLimit slSF = slLC0 * 875;


      SearchLimit limit1 = SearchLimit.NodesPerMove(500_000);

      //limit1 = SearchLimit.SecondsForAllMoves(900, 15) * 0.04f;
      //limit1 = SearchLimit.NodesPerMove(1);

      //limit1 = SearchLimit.NodesForAllMoves(500_000);//, 25_000);

      limit1 = SearchLimit.SecondsForAllMoves(1, 1);
      //limit1 = SearchLimit.SecondsForAllMoves(1, 0.15f) * 1f;
      //limit1 = SearchLimit.SecondsForAllMoves(900, 15) * 0.05f;
//      limit1 = SearchLimit.SecondsPerMove(1);
//      limit1 = SearchLimit.NodesPerMove(3000);

      SearchLimit limit2 = limit1;// * 0.2f;// SearchLimit.NodesPerMove(2500);
      //limit2 = limit1 * 3;
//      limit2 *= 0.5f;
//      limit2 = SearchLimit.NodesPerMove(400);


//      ParamsSearchExecutionModifier.Register("SetBatchSize", pe => pe.MaxBatchSize = 1024);


      // Don't output log if very small games
      // (to avoid making very large log files or slowing down play).
      bool outputLog = limit1.EstNumNodes(200_000, false) > 10_000;
      outputLog = false;
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", evalDef1, evalDefSecondary1, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres1.log.txt" : null);
      GameEngineDefCeres engineDefCeres2 = new GameEngineDefCeres("Ceres2", evalDef2, evalDefSecondary2, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres2.log.txt" : null);

      //engineDefCeres1.OverrideTimeManager = new ManagerGameLimitCeres();
      //      engineDefCeres1.SearchParams.EnableInstamoves = false;

//      engineDefCeres1.SearchParams.ExecutionModifierID = "SetBatchSize";

      ////////
      // THIS MIGHT BE GOOD - but only with T60 networks with quality MLH
      //      engineDefCeres1.SearchParams.MLHBonusFactor = 0.50f;
      ////////


//engineDefCeres1.SearchParams.TestFlag = true;
      //engineDefCeres1.SearchParams.MoveFutilityPruningAggressiveness = 0.4f;
      //engineDefCeres1.SearchParams.GameLimitUsageAggressiveness *= 1.2f;
      //engineDefCeres1.SearchParams.MoveFutilityPruningAggressiveness = 0.75f;
      //      engineDefCeres2.SearchParams.MoveFutilityPruningAggressiveness = 1.25f;// 0.75f;

      //engineDefCeres1.SearchParams.EnableInstamoves=false; 
      //engineDefCeres2.SearchParams.EnableInstamoves = false;

      //engineDefCeres1.SearchParams.TreeReuseEnabled = false;
      //engineDefCeres2.SearchParams.TreeReuseEnabled = false;
      //engineDefCeres1.SearchParams.MLHBonusFactor = 0.1f;

//engineDefCeres1.SearchParams.TestFlag = true;
      if (false)
      {
#if NOT
        engineDefCeres1.SelectParams.CPUCT = engineDefCeres1.SelectParams.CPUCTAtRoot = 1.745f;
        engineDefCeres1.SelectParams.CPUCTBase = engineDefCeres1.SelectParams.CPUCTBaseAtRoot = 38739;
        engineDefCeres1.SelectParams.CPUCTFactor = engineDefCeres1.SelectParams.CPUCTFactorAtRoot = 3.894f;
        engineDefCeres1.SelectParams.FPUValue = 0.330f;
        engineDefCeres1.SelectParams.PolicySoftmax = 1.359f;
#endif
      }

      if (!limit1.IsNodesLimit)
      {
        engineDefCeres1.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
        engineDefCeres2.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
      }

      //engineDefCeres1.SelectParams.PowerMeanNExponent = 0.12f;
      //engineDefCeres1.SelectParams.PolicyDecayFactor = 10f;
      //engineDefCeres1.SelectParams.PolicyDecayExponent = 0.40f;

      //engineDefCeres1.SelectParams.PolicySoftmax = 1.5f;

      //engineDefCeres1.SelectParams.EnableExplorationGuard = true;
      //engineDefCeres1.SearchParams.EnableTablebases = false;

      // TODO: support this in GameEngineDefCeresUCI
      bool forceDisableSmartPruning = false;// limit1.IsNodesLimit;
//forceDisableSmartPruning=true;
      if (forceDisableSmartPruning)
      {
        engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres1.SearchParams.MoveFutilityPruningAggressiveness = 0;
        engineDefCeres2.SearchParams.MoveFutilityPruningAggressiveness = 0;
      }

      //GameEngineDef engineDefCeresUCI = new GameEngineDefUCI("CeresUCI", new GameEngineUCISpec("CeresUCI", @"c:\dev\ceres\artifacts\release\net5.0\ceres.exe"));
      GameEngineDef engineDefCeresUCI1 = new GameEngineDefCeresUCI("CeresUCINew", evalDef1, overrideEXE: @"C:\dev\Ceres\artifacts\release\net5.0\ceres.exe");
      GameEngineDef engineDefCeres91 = new GameEngineDefCeresUCI("Ceres91", evalDef2, overrideEXE: SoftwareManager.IsLinux ? @"/raid/dev/Ceres91b/Ceres/artifacts/release/net5.0/Ceres.dll"
                                                                                                                           : @"C:\ceres\releases\v0.91b\ceres.exe");

      EnginePlayerDef playerCeres1UCI = new EnginePlayerDef(engineDefCeresUCI1, limit1);
      EnginePlayerDef playerCeres91 = new EnginePlayerDef(engineDefCeres91, limit2);

      EnginePlayerDef playerCeres1 = new EnginePlayerDef(engineDefCeres1, limit1);
      EnginePlayerDef playerCeres2 = new EnginePlayerDef(engineDefCeres2, limit2);

      //#if NOTParamsSearch
      ParamsSearch paramsSearchLC0 = new ParamsSearch();
      //      paramsSearchLC0.TestFlag = true;

      bool ENABLE_LC0 = evalDef1.Nets[0].Net.Type == NNEvaluatorType.LC0Library && (evalDef1.Nets[0].WeightValue == 1 && evalDef1.Nets[0].WeightPolicy == 1 && evalDef1.Nets[0].WeightM == 1);
      GameEngineDefLC0 engineDefLC1 = ENABLE_LC0 ? new GameEngineDefLC0("LC0_0", evalDef1, forceDisableSmartPruning, paramsSearchLC0, null) : null;
      GameEngineDefLC0 engineDefLC2 = ENABLE_LC0 ? new GameEngineDefLC0("LC0_2", evalDef1, forceDisableSmartPruning, null, null) : null;

      GameEngineDefLC0 engineDefLC2TCEC = null;// new GameEngineDefLC0("LC0_TCEC", evalDef2, forceDisableSmartPruning, null, null,
                                               //               overrideEXE: @"c:\dev\lc0\lc_270\lc0.exe",
                                               //             extraCommandLineArgs: "--max-out-of-order-evals-factor=2.4 --max-collision-events=917");

      EnginePlayerDef playerEthereal = new EnginePlayerDef(engineDefEthereal, limit1);
      EnginePlayerDef playerStockfish11 = new EnginePlayerDef(engineDefStockfish11, limit1);
      EnginePlayerDef playerStockfish14 = new EnginePlayerDef(EngineDefStockfish14(), limit1);// * 350);
      EnginePlayerDef playerLC0 = ENABLE_LC0 ? new EnginePlayerDef(engineDefLC1, limit1) : null;
      EnginePlayerDef playerLC0_2 = ENABLE_LC0 ? new EnginePlayerDef(engineDefLC2, limit2) : null;
//      EnginePlayerDef playerLC0TCEC =  new EnginePlayerDef(engineDefLC2TCEC, limit1);
                                           //#endif

      //      def.SearchLimitEngine1 = def.SearchLimitEngine2 = SearchLimit.SecondsForAllMoves(15, 0.25f);
      //      def.SearchLimitEngine2 = def.SearchLimitEngine2 = SearchLimit.SecondsForAllMoves(15, 0.25f);


      //(playerCeres1.EngineDef as GameEngineDefCeres).SearchParams.DrawByRepetitionLookbackPlies = 40;

      if (false)
      {
        // ===============================================================================
        SuiteTestDef suiteDef = new SuiteTestDef("Suite",
                                                //@"\\synology\dev\chess\data\epd\chad_tactics-100M.epd",
                                                @"\\synology\dev\chess\data\epd\ERET.epd",
                                                //@"\\synology\dev\chess\data\epd\ERET_VESELY203.epd",
//                                                   @"\\synology\dev\chess\data\epd\sts.epd",
                                                playerCeres1, null, playerCeres91);// playerLC0);
        //playerCeres1, null, playerCeres2UCI);
        suiteDef.MaxNumPositions = 100;

        SuiteTestRunner suiteRunner = new SuiteTestRunner(suiteDef);

        //        suiteRunner.Run(POOLED ? 20 : 4, true, false);
        SuiteTestResult suiteResult = suiteRunner.Run(1, true, false);
#if NOT
        SysMisc.WriteObj("FinalQ1.dat", suiteResult.FinalQ1);

        float[] priorQ1 = null;
        if (File.Exists("CompareFinalQ1.dat"))
        {
          priorQ1 = SysMisc.ReadObj<float[]>("CompareFinalQ1.dat");

          float correl1Prior = (float)StatUtils.Correlation(suiteResult.FinalQ1, priorQ1);
          float nps1 = suiteResult.TotalNodes1 / suiteResult.TotalRuntime1;
          Console.WriteLine($"Correlation Q1 and prior Q1 {correl1Prior,6:F3} scores {nps1 * correl1Prior * correl1Prior,10:F0}");
        }

        if (suiteResult.FinalQ2 != null)
        {
          float correl = (float)StatUtils.Correlation(suiteResult.FinalQ1, suiteResult.FinalQ2);
          float nps2 = suiteResult.TotalNodes2 / suiteResult.TotalRuntime2;
          Console.WriteLine();
          Console.WriteLine($"Correlation Q1 and Q2 {correl,6:F3}");
          if (priorQ1 != null)
          {
            float correl2Prior = (float)StatUtils.Correlation(suiteResult.FinalQ2, priorQ1);
            Console.WriteLine($"Correlation Q2 and prior Q1 {correl2Prior,6:F3} scores {nps2 * correl * correl,10:F0}");
          }
        }

        return;
        // ===============================================================================
#endif
      }

      //      engineDefCeres2.SearchParams.TwofoldDrawEnabled = false;
      //engineDefCeres1.SearchParams.TreeReuseEnabled = false;
      //engineDefCeres2.SearchParams.TreeReuseEnabled = false;
      //engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled= false;
      //engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled= false;
      //engineDefLC0.SearchParamsEmulate.FutilityPruningStopSearchEnabled= false;

      TournamentDef def = new TournamentDef("TOURN", playerCeres91, playerLC0);


      def.NumGamePairs = 102;// 102;
      def.ShowGameMoves = false;

      string baseName = "tcec1819";
      def.OpeningsFileName = SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/openings/{baseName}.pgn"
                                                     : @$"\\synology\dev\chess\data\openings\{baseName}.pgn";

      if (false)
      {
        def.AdjudicationThresholdCentipawns = int.MaxValue;
        def.AdjudicationThresholdNumMoves = 3000;
        def.UseTablebasesForAdjudication = false;
      }

      if (POOLED)
      {
        CONCURRENCY = 16;
      }

      TournamentManager runner = new TournamentManager(def, CONCURRENCY);
      TournamentGameQueueManager queueManager = null;

      if (CommandLineWorkerSpecification.IsWorker)
      {
        queueManager = new TournamentGameQueueManager(Environment.GetCommandLineArgs()[2]);
        int gpuID = CommandLineWorkerSpecification.GPUID;
        Console.WriteLine($"\r\n***** Running in DISTRIBUTED mode as WORKER on gpu {gpuID} (queue directory {queueManager.QueueDirectory})\r\n");

        def.Player1Def.EngineDef.ModifyDeviceIndexIfNotPooled(gpuID);
        def.Player2Def.EngineDef.ModifyDeviceIndexIfNotPooled(gpuID);
      }
      else
      {
        if (RUN_DISTRIBUTED)
        {
          queueManager = new TournamentGameQueueManager(null);
          Console.WriteLine($"\r\n***** Running in DISTRIBUTED mode as COORDINATOR (queue directory {queueManager.QueueDirectory})\r\n");
        }
      }

       TournamentResultStats results;

      //UCIEngineProcess.VERBOSE = true;

      TimingStats stats = new TimingStats();
      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        results = runner.RunTournament(queueManager);
      }

      Console.WriteLine();
      Console.WriteLine($"Tournament completed in {stats.ElapsedTimeSecs,8:F2} seconds.");
      Console.WriteLine(results.GameOutcomesString);

      Console.WriteLine();
      Console.WriteLine("<CRLF> to continue");
      Console.ReadLine();
    }


    public static void TestSF(int index, bool gitVersion)
    {
      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification("LC0:j94-100", "GPU:" + index);
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("CeresInProc", evalDef1, null,
                                                                  new ParamsSearch(), new ParamsSelect(), null,
                                                                  "CeresSF.log.txt");

      SearchLimit limitCeres = SearchLimit.SecondsForAllMoves(60, 1.25f) * 0.15f;
      SearchLimit limitSF = limitCeres * 1.5f;

      GameEngineDef engineDefCeresUCIGit = new GameEngineDefCeresUCI("CeresUCIGit", evalDef1, overrideEXE: @"C:\ceres\releases\v0.88\ceres.exe");
      EnginePlayerDef playerCeres = new EnginePlayerDef(gitVersion ? engineDefCeresUCIGit : engineDefCeres1,
                                                        limitCeres);


      EnginePlayerDef playerSF = new EnginePlayerDef(EngineDefStockfish14(), limitSF);

      TournamentDef def = new TournamentDef("TOURN", playerCeres, playerSF);
      def.OpeningsFileName = "TCEC1819.pgn";
      //def.NumGamePairs = 10;
      def.ShowGameMoves = false;

      TournamentManager runner = new TournamentManager(def, 1);
      TournamentGameQueueManager queueManager = null;

      TournamentResultStats results;
      TimingStats stats = new TimingStats();


      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {

        results = runner.RunTournament(queueManager);
      }

      Console.WriteLine();
      Console.WriteLine($"Tournament completed in {stats.ElapsedTimeSecs,8:F2} seconds.");
      Console.WriteLine(playerCeres + " " + results.GameOutcomesString);
    }

  }
}
