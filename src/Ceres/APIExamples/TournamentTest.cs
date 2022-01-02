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
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Environment;
using Ceres.Features.EngineTests;

#endregion

namespace Ceres.APIExamples
{
  public static class TournamentTest
  {
    const bool POOLED = false;

    static int CONCURRENCY = POOLED ? 16 : 1;
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


    static List<string> extraUCI = null;// new string[] {"setoption name Contempt value 5000" };
    static GameEngineDef engineDefEthereal = new GameEngineDefUCI("Etheral", new GameEngineUCISpec("Etheral", ETHERAL_EXE, SF_NUM_THREADS, SF_HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));
    static GameEngineDef engineDefStockfish11 = new GameEngineDefUCI("SF11", new GameEngineUCISpec("SF11", SF11_EXE, SF_NUM_THREADS, SF_HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));

    public static GameEngineDef EngineDefStockfish14(int numThreads = SF_NUM_THREADS, int hastableSize = SF_HASH_SIZE_MB) =>
      new GameEngineDefUCI("SF14", new GameEngineUCISpec("SF14", SF14_EXE, numThreads,
                           hastableSize, TB_PATH, uciSetOptionCommands: extraUCI));

    const int SF_NUM_THREADS = 6;
    static string TB_PATH => CeresUserSettingsManager.Settings.TablebaseDirectory;
    const int SF_HASH_SIZE_MB = 2048;

    public static void PreTournamentCleanup()
    {
      KillCERES();

      File.Delete("Ceres1.log.txt");
      File.Delete("Ceres2.log.txt");
    }



    /// <summary>
    /// Test code.
    /// </summary>
    public static void Test()
    {
      //      if (Marshal.SizeOf<MCTSNodeStruct>() != 64)
      //        throw new Exception("Wrong size " + Marshal.SizeOf<MCTSNodeStruct>().ToString());
      InstallCUSTOM1AsDynamicByPhase();
      PreTournamentCleanup();

//      RunEngineComparisons(); return;

      if (false)
      {
        Parallel.Invoke(() => TestSF(0, true), () => { Thread.Sleep(7_000); TestSF(1, false); });
        System.Environment.Exit(3);
      }

      string GPUS = POOLED ? "GPU:0,1,2,3:POOLED"
                           : "GPU:0";


      string NET1_SECONDARY1 = null;// "610024";
      string NET1 = "j94-100";
      string NET2 = "j94-100";
      //NET1 = "610024";
      //NET2 = "610024";
      NET1 = "610235";
      NET2 = "610235";

      NET1 = "760751";
      NET2 = "42767";// "753723";
      NET1 = @"ONNX_ORT:d:\weights\lczero.org\hydra_t00-attn.gz.onnx";// "apv4_t14";// apv4_t16";
      NET2 = @"ONNX_ORT:d:\weights\lczero.org\apv4_t16.onnx";

      
      NET1 = "baseline02#8";
      NET2 = "baseline02#16";

      NET1 = "ONNX_TRT:d:\\weights\\lczero.org\\baseline02.onnx#8";
      NET1 = "ONNX_ORT:d:\\weights\\lczero.org\\baseline02.onnx#16";

      NET1 = "744204_onnx";
      NET2 = "744204_onnx";

      NET1 = "apv5_t01.ORT";
      NET2 = "apv5_t01.ORT";
      //      NET1 = "apv5_t01#8";
      //      NET2 = "apv5_t01#16";

      //NET1 = @"LC0:hydra_t00-attn.onnx";// "apv4_t14";// apv4_t16";
      //      string NET1_SECONDARY1 = "j94-100";

NET1 = "attn_10b_800k";
NET2 = "base_10b_800k";


//      NET1 = "760998";
      NET1 = "753723";
      NET2 = "753723";
//      NET1 = "69637";
//      NET2 = "69637";

      NET1 = "703810";
      NET2 = "703810";
//      NET1 = "badgyal-3";
//      NET2 = "badgyal-3";

      //NET1 = "744204_onnx_int8";

      //r1kb4/3r1p2/8/7R/p3R3/P1B5/KP6/8 w - - 3 49

      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(NET1, GPUS); // j64-210 LS16 40x512-lr015-swa-167500
      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification(NET2, GPUS); ;

      NNEvaluatorDef evalDefSecondary1 = null;
      if (NET1_SECONDARY1 != null)
      {
        evalDefSecondary1 = NNEvaluatorDefFactory.FromSpecification($@"LC0:{NET1_SECONDARY1}", GPUS);
      }

      NNEvaluatorDef evalDefSecondary2 = null;


      //      public NNEvaluatorDynamic(NNEvaluator[] evaluators,
      //                        Func<IEncodedPositionBatchFlat, int> dynamicEvaluatorIndexPredicate = null)

      //evalDef1 = NNEvaluatorDefFactory.FromSpecification("ONNX:tfmodelc", "GPU:0");

      SearchLimit limit1 = SearchLimit.NodesForAllMoves(1_000_000);
      //      limit1 = SearchLimit.NodesPerTree(50_000);

      // 140 good for 203 pairs, 300 good for 100 pairs
      //      limit1 = SearchLimit.NodesForAllMoves(200_000, 500) * 0.25f;
      //      limit1 = SearchLimit.SecondsForAllMoves(5, 0.02f) * 2;
      limit1 = SearchLimit.NodesPerMove(2_000);
      //limit1 = SearchLimit.SecondsForAllMoves(60, 1) * 0.5f;
      //ok      limit1 = SearchLimit.NodesPerMove(350_000); try test3.pgn against T75 opponent Ceres93 (in first position, 50% of time misses win near move 12

      SearchLimit limit2 = limit1;

      // Don't output log if very small games
      // (to avoid making very large log files or slowing down play).
      bool outputLog = true;// limit1.EstNumNodes(500_000, false) > 10_000;
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", evalDef1, evalDefSecondary1, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres1.log.txt" : null);
      GameEngineDefCeres engineDefCeres2 = new GameEngineDefCeres("Ceres2", evalDef2, evalDefSecondary2, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres2.log.txt" : null);
      GameEngineDefCeres engineDefCeres3 = new GameEngineDefCeres("Ceres3", evalDef2, evalDefSecondary2, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres3.log.txt" : null);

      //engineDefCeres1.OverrideLimitManager = new  Ceres.MCTS.Managers.Limits.ManagerGameLimitSimple(20);
      //      engineDefCeres2.SearchParams.TestFlag = true;
      //      engineDefCeres2.SearchParams.EnableSearchExtension = false;
      // engineDefCeres1.SearchParams.TreeReuseRetainedPositionCacheEnabled = true;
      //engineDefCeres2.SearchParams.TreeReuseRetainedPositionCacheEnabled = true;

      //      engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
      //      engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled = false;

      //engineDefCeres2.SearchParams.TestFlag2 = true;
      //engineDefCeres2.SearchParams.EnableUncertaintyBoosting = false;
      //      engineDefCeres1.SearchParams.TranspositionRootMaxN = true;
      //      engineDefCeres1.SearchParams.EnableUseSiblingEvaluations = false;

      //      engineDefCeres1.SearchParams.TranspositionRootMaxN = true;

      //      engineDefCeres1.SearchParams.TestFlag = true;

      //      engineDefCeres1.SearchParams.GameLimitUsageAggressiveness = 1.5f;
      //      engineDefCeres2.SearchParams.TestFlag2 = true;

      //      engineDefCeres1.SearchParams.TreeReuseRetainedPositionCacheEnabled = true;

      //engineDefCeres2.SearchParams.TestFlag = true;
      //      engineDefCeres1.SelectParams.CPUCT *= 0.9f;

      //      engineDefCeres1.SelectParams.CPUCTFactorAtRoot *= (1.0f / 0.8f);

      //      engineDefCeres2.SearchParams.EnableUncertaintyBoosting = false;
      //      engineDefCeres1.SelectParams.CPUCT *= 0.30f;

      //engineDefCeres2.SelectParams.CPUCTAtRoot *= 1.33f;


      //engineDefCeres1.SearchParams.TestFlag = true;


#if NOT
      engineDefCeres1.SearchParams.TranspositionRootBackupSubtreeFracs = new float[] { 1,1,1 };
      engineDefCeres2.SearchParams.TranspositionRootBackupSubtreeFracs = new float[] { 1,1,1 };

      engineDefCeres1.SearchParams.TranspositionCloneNodeSubtreeFracs = new float[] { 1,0,0};
      engineDefCeres2.SearchParams.TranspositionCloneNodeSubtreeFracs = new float[] { 1,1,1 };
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
      bool forceDisableSmartPruning = limit1.IsNodesLimit;
      if (forceDisableSmartPruning)
      {
        engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled = false;
        //engineDefCeres1.SearchParams.MoveFutilityPruningAggressiveness = 0;
        //engineDefCeres2.SearchParams.MoveFutilityPruningAggressiveness = 0;
      }

      string exeCeres = SoftwareManager.IsLinux ? @"/raid/dev/Ceres/artifacts/release/net5.0/Ceres.dll"
                                                : @"C:\dev\ceres\artifacts\release\net5.0\ceres.exe";
      string exeCeres93 = SoftwareManager.IsLinux ? @"/raid/dev/Ceres93/artifacts/release/5.0/Ceres.dll"
                                                  : @"C:\ceres\releases\v0.93\ceres.exe";
      string exeCeres94 = SoftwareManager.IsLinux ? @"/raid/dev/Ceres94/Ceres.dll"
                                                  : @"\ceres\releases\v0.95rc4\ceres.exe";
      string exeCeresPreNC = SoftwareManager.IsLinux ? @"/raid/dev/Ceres_PreNC/artifacts/release/5.0/Ceres.dll"
                                                  : @"c:\ceres\releases\v0.95_PreNC\ceres.exe";

      //GameEngineDef engineDefCeresUCI = new GameEngineDefUCI("CeresUCI", new GameEngineUCISpec("CeresUCI", @"c:\dev\ceres\artifacts\release\net5.0\ceres.exe"));
      //      GameEngineDef engineDefCeresUCI1x = new GameEngineDefCeresUCI("CeresUCINew", evalDef1, overrideEXE: @"C:\dev\Ceres\artifacts\release\net5.0\ceres.exe");

      GameEngineDef engineDefCeresUCI1 = new GameEngineDefCeresUCI("CeresUCINew", evalDef1, overrideEXE: exeCeres, disableFutilityStopSearch: forceDisableSmartPruning);
      GameEngineDef engineDefCeresUCI2 = new GameEngineDefCeresUCI("CeresUCINew", evalDef2, overrideEXE: exeCeres, disableFutilityStopSearch: forceDisableSmartPruning);

      GameEngineDef engineDefCeres93 = new GameEngineDefCeresUCI("Ceres93", evalDef2, overrideEXE: exeCeres93, disableFutilityStopSearch:forceDisableSmartPruning);
      GameEngineDef engineDefCeres94 = new GameEngineDefCeresUCI("Ceres94", evalDef2, overrideEXE: exeCeres94, disableFutilityStopSearch: forceDisableSmartPruning);
      GameEngineDef engineDefCeresPreNC = new GameEngineDefCeresUCI("CeresPreNC", evalDef2, overrideEXE: exeCeresPreNC, disableFutilityStopSearch: forceDisableSmartPruning);

      EnginePlayerDef playerCeres1UCI = new EnginePlayerDef(engineDefCeresUCI1, limit1);
      EnginePlayerDef playerCeres2UCI = new EnginePlayerDef(engineDefCeresUCI2, limit2);
      EnginePlayerDef playerCeres93 = new EnginePlayerDef(engineDefCeres93, limit2);
      EnginePlayerDef playerCeres94 = new EnginePlayerDef(engineDefCeres94, limit2);
      EnginePlayerDef playerCeresPreNC = new EnginePlayerDef(engineDefCeresPreNC, limit2);

      EnginePlayerDef playerCeres1 = new EnginePlayerDef(engineDefCeres1, limit1);
      EnginePlayerDef playerCeres2 = new EnginePlayerDef(engineDefCeres2, limit1);
      EnginePlayerDef playerCeres3 = new EnginePlayerDef(engineDefCeres3, limit1);

      bool ENABLE_LC0 = evalDef1.Nets[0].Net.Type == NNEvaluatorType.LC0Library && (evalDef1.Nets[0].WeightValue == 1 && evalDef1.Nets[0].WeightPolicy == 1 && evalDef1.Nets[0].WeightM == 1);
      GameEngineDefLC0 engineDefLC1 = ENABLE_LC0 ? new GameEngineDefLC0("LC0_0", evalDef1, forceDisableSmartPruning, null, null) : null;
      GameEngineDefLC0 engineDefLC2 = ENABLE_LC0 ? new GameEngineDefLC0("LC0_2", evalDef2, forceDisableSmartPruning, null, null) : null;

      EnginePlayerDef playerEthereal = new EnginePlayerDef(engineDefEthereal, limit2);
      EnginePlayerDef playerStockfish11 = new EnginePlayerDef(engineDefStockfish11, limit2);
      EnginePlayerDef playerStockfish14 = new EnginePlayerDef(EngineDefStockfish14(), limit2 * 0.30f);// * 350);
      EnginePlayerDef playerLC0 = ENABLE_LC0 ? new EnginePlayerDef(engineDefLC1, limit2) : null;
      EnginePlayerDef playerLC0_2 = ENABLE_LC0 ? new EnginePlayerDef(engineDefLC2, limit2) : null;


      if (false)
      {        
        string BASE_NAME = "Elo2500_10k";// nice_lcx Stockfish238 ERET_VESELY203 endgame2 chad_tactics-100M lichess_chad_bad.csv
        ParamsSearch paramsNoFutility = new ParamsSearch() { FutilityPruningStopSearchEnabled = false };

        // ===============================================================================
        string suiteGPU = POOLED ? "GPU:0,1,2,3:POOLED=SHARE1" : "GPU:0,1";
        SuiteTestDef suiteDef =
          new SuiteTestDef("Suite",
                           SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/epd/{BASE_NAME}.epd"
                                                   : @$"\\synology\dev\chess\data\epd\{BASE_NAME}.epd",
                           SearchLimit.NodesPerMove(125_000),
                           GameEngineDefFactory.CeresInProcess("Ceres1", NET1, suiteGPU, paramsNoFutility with { TestFlag = true }),
                           GameEngineDefFactory.CeresInProcess("Ceres2", NET1, suiteGPU, paramsNoFutility with {  }),
                           null);// playerLC0.EngineDef);

        suiteDef.MaxNumPositions = 15_000;
        suiteDef.EPDLichessPuzzleFormat = suiteDef.EPDFileName.ToUpper().Contains("LICHESS");

        //suiteDef.EPDFilter = s => !s.Contains(".exe"); // For NICE suite, these represent positions with multiple choices

        SuiteTestRunner suiteRunner = new SuiteTestRunner(suiteDef);

        SuiteTestResult suiteResult = suiteRunner.Run(POOLED ? 12 : 1, true, false);
        Console.WriteLine("Max mbytes alloc: " + WindowsVirtualAllocManager.MaxBytesAllocated / (1024 * 1024));
        Console.WriteLine("Test counter 1  : " + MCTSEventSource.TestCounter1);
        Console.WriteLine("Test metric 1   : " + MCTSEventSource.TestMetric1);
        return;
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

      //    EnginePlayerDef playerDef = new EnginePlayerDef(engineDefLC0, searchLimit);
      GameEngineDef[] ceresEngines = new GameEngineDef[3+1];
      for (int i = 0; i < ceresEngines.Length; i++)
      {
        float value = 0.80f + 0.15f * i;
        ceresEngines[i] = GameEngineDefFactory.CeresInProcess("C_" + 100*MathF.Round(value, 1), NET1, GPUS,
                                                              engineDefCeres1.SearchParams with { GameLimitUsageAggressiveness = value}, null);
      }

      // TODO: UCI engine should point to .NET 6 subdirectory if on .NET 6
      TournamentDef def = new TournamentDef("TOURN", playerCeres1UCI, playerCeresPreNC);//, playerStockfish14/*, playerCeres3,
#if NOT
      TournamentDef def = new TournamentDef("TIME_AGG");
      def.AddEngine(playerStockfish14.EngineDef, limit1 * 0.7f);
      def.AddEngines(limit1, ceresEngines);
      def.ReferenceEngineId = def.Engines[0].ID;
#endif
      //      def.ReferenceEngineId = playerStockfish14.ID;
      //        TournamentDef def = new TournamentDef("TOURN", playerCeres1UCI, playerCeres93);
      //TournamentDef def = new TournamentDef("TOURN", playerCeres93, playerCeres1);


      def.NumGamePairs = 500;// 203;//1000;//203;//203;// 500;// 203;//203;// 102; 203
      def.ShowGameMoves = false;

//      string baseName = "tcec1819";
string      baseName = "4mvs_+90_+99";
//      baseName = "book-ply8-unifen-Q-0.25-0.40";
//      baseName = "test3";
//     baseName = "tcec_big";
//      baseName = "endgame-16-piece-book_Q-0.0-0.6_1";
      def.OpeningsFileName = SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/openings/{baseName}.pgn"
                                                     : @$"\\synology\dev\chess\data\openings\{baseName}.pgn";

      if (false)
      {
        def.AdjudicateDrawThresholdCentipawns = 0;
        def.AdjudicateDrawThresholdNumMoves = 999;

        def.AdjudicateWinThresholdCentipawns = int.MaxValue;
        def.AdjudicateWinThresholdNumMoves = 3000;
        def.UseTablebasesForAdjudication = false;
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

      string leela_EXE = Path.Combine(CeresUserSettingsManager.Settings.DirExternalEngines, "lc0-v0.28.0-windows-gpu-nvidia-cuda", "LC0.exe");
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
      GameEngineDefLC0 engineDef1LC0 = new GameEngineDefLC0("LC0-2", ceresNNDef, forceDisableSmartPruning: true, null, null);

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

      CompareEngineParams parms = new CompareEngineParams("VsLC0", pgnFileName,
                                              1000, // number of positions
                                              null,//s => s.FinalPosition.PieceCount <= 15,
                                              CompareEnginesVersusOptimal.PlayerMode.Ceres, "703810", //610034
                                              CompareEnginesVersusOptimal.PlayerMode.LC0, "703810",
                                              CompareEnginesVersusOptimal.PlayerMode.Ceres, "703810",
                                              SearchLimit.NodesPerMove(2_000), // search limit
                                              new int[] { 0, 1, 2, 3 },
                                              s =>
                                              {
//                                                s.EnableUncertaintyBoosting = true;
                                              },
                                              null, // l => l.CPUCT = 1.1f,
                                              null,
                                              null,
                                              true,
                                              1, 
                                              7,
                                              false // Stockfish crosscheck
                                             );


      CompareEngineResultSummary result = new CompareEnginesVersusOptimal(parms).Run();
    }


  

  /// <summary>
  /// Test code that installs "CUSTOM1" network type which is an NNEvaluatorDynamic
  /// and uses one network for first part of search then switches to second network.
  /// Example usage (also customized nets and fraction below):
  ///   NET1 = "CUSTOM1:66666";
  /// Was -7Elo +/10 with at 60 second games (switch point 0.666).
  /// </summary>
  public static void InstallCUSTOM1AsDynamicByPhase()
    {
      static NNEvaluator Build(string netID1, int gpuID, NNEvaluator referenceEvaluator)
      {
        // Construct a compound evaluator which does both fast and slow (potentially parallel)
        NNEvaluator[] evaluators = new NNEvaluator[] {NNEvaluator.FromSpecification("66666", $"GPU:{gpuID}"),
                                                      NNEvaluator.FromSpecification("66511", $"GPU:{gpuID}")};

        const float FRACTION_SWITCH_ALTERNATE_NET = 0.75f;
throw new NotImplementedException("COMBO_PHASED deprecated");
//        NNEvaluatorDynamic dyn = new NNEvaluatorDynamic(evaluators, (batch)
//          => MCTSManager.ThreadSearchContext != null ? (MCTSManager.ThreadSearchContext.Manager.FractionSearchCompleted < FRACTION_SWITCH_ALTERNATE_NET ? 0 : 1) : 0);
//        return dyn;
      }
      NNEvaluatorFactory.Custom1Factory = Build;
    }


  }
}
