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
using Ceres.MCTS.MTCSNodes.Struct;
using System.Runtime.InteropServices;
using Ceres.MCTS.Environment;

#endregion

namespace Ceres.APIExamples
{
  public static class TournamentTest
  {
    const bool POOLED = false;

    static int CONCURRENCY = POOLED ? 16 : 4;
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

    const int SF_NUM_THREADS = 8;
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
      NET1 = "610024";
      NET2 = "610024";

      NET1 = "753723";
      NET2 = "753723";
      NET1 = "703810";
      NET2 = "703810";
      //      string NET1_SECONDARY1 = "j94-100";

      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification(NET1, GPUS); // j64-210 LS16 40x512-lr015-swa-167500
      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification($@"LC0:{NET2}", GPUS); ;

      NNEvaluatorDef evalDefSecondary1 = null;
      if (NET1_SECONDARY1 != null)
      {
        evalDefSecondary1 = NNEvaluatorDefFactory.FromSpecification($@"LC0:{NET1_SECONDARY1}", GPUS);
      }

      NNEvaluatorDef evalDefSecondary2 = null;


//      public NNEvaluatorDynamic(NNEvaluator[] evaluators,
//                        Func<IEncodedPositionBatchFlat, int> dynamicEvaluatorIndexPredicate = null)

      //evalDef1 = NNEvaluatorDefFactory.FromSpecification("ONNX:tfmodelc", "GPU:0");

      if (POOLED)
      {
        evalDef1.MakePersistent();
        evalDef2.MakePersistent();
      }

      SearchLimit limit1 = SearchLimit.NodesPerMove(100_000);
      //limit1 = SearchLimit.NodesForAllMoves(1_000_000, 10_000);

      // 140 good for 203 pairs, 300 good for 100 pairs
      //      limit1 = SearchLimit.SecondsForAllMovess(90, 1f);
      limit1 = SearchLimit.SecondsForAllMoves(100, 0.5f) * 0.2f;
      //limit1 = SearchLimit.SecondsPerMove(1);
//limit1 = SearchLimit.SecondsForAllMoves(50, 0.1f) * 1.1f;
//ok      limit1 = SearchLimit.NodesPerMove(350_000); try test3.pgn against T75 opponent Ceres93 (in first position, 50% of time misses win near move 12
      
      SearchLimit limit2 = limit1;

      // Don't output log if very small games
      // (to avoid making very large log files or slowing down play).
      bool outputLog = false;// limit1.EstNumNodes(500_000, false) > 10_000;
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", evalDef1, evalDefSecondary1, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres1.log.txt" : null);
      GameEngineDefCeres engineDefCeres2 = new GameEngineDefCeres("Ceres2", evalDef2, evalDefSecondary2, new ParamsSearch(), new ParamsSelect(),
                                                                  null, outputLog ? "Ceres2.log.txt" : null);

//      engineDefCeres1.SearchParams.EnableUseSiblingEvaluations = true;
     engineDefCeres1.SearchParams.TestFlag = true;

//      engineDefCeres1.SelectParams.CPUCT       *= 1.075f;
//      engineDefCeres1.SelectParams.CPUCTAtRoot *= 1.075f;

      //      engineDefCeres1.SearchParams.EnableUseSiblingEvaluations = true;

      //      engineDefCeres1.SearchParams.EnableTablebases = false;
      //engineDefCeres1.SearchParams.Execution.FlowDirectOverlapped = false;
      //      engineDefCeres2.SearchParams.Execution.FlowDirectOverlapped = false;

      //      engineDefCeres1.SearchParams.Execution.TranspositionMode = TranspositionMode.None;
      //      engineDefCeres1.SearchParams.Execution.InFlightThisBatchLinkageEnabled = false;

#if NOT
      
      engineDefCeres1.SearchParams.TranspositionRootBackupSubtreeFracs = new float[] { 1,1 };
        engineDefCeres1.SearchParams.TranspositionCloneNodeSubtreeFracs = new float[] { 1,1,};
      engineDefCeres2.SearchParams.TranspositionRootBackupSubtreeFracs = new float[] { 1,1 };
      engineDefCeres2.SearchParams.TranspositionCloneNodeSubtreeFracs = new float[] { 1,1 };
#endif
      //      engineDefCeres1.SearchParams.TreeReuseSwapRootEnabled = false;
      //      engineDefCeres2.SearchParams.TreeReuseSwapRootEnabled = false;

      if (false)
      {
        //engineDefCeres1.SearchParams.TreeReuseSwapRootEnabled = true;
        //engineDefCeres2.SearchParams.TreeReuseSwapRootEnabled = false;

        //        engineDefCeres1.SearchParams.Execution.TranspositionMode = TranspositionMode.SingleNodeDeferredCopy;
        //engineDefCeres1.SearchParams.Execution.InFlightThisBatchLinkageEnabled = false;
        //engineDefCeres1.SearchParams.TranspositionRootBackupSubtreeFracs = new float[] { 1, 1, 1 };
        //engineDefCeres1.SearchParams.TranspositionCloneNodeSubtreeFracs = new float[] { 1, 1, 1 };

        //engineDefCeres2.SearchParams.EnableUseSiblingEvaluations = false;
        //        engineDefCeres2.SearchParams.Execution.TranspositionMode = TranspositionMode.SingleNodeDeferredCopy;
        //engineDefCeres2.SearchParams.TranspositionRootBackupSubtreeFracs = new float[] { 1, float.NaN, float.NaN };
        //engineDefCeres2.SearchParams.TranspositionCloneNodeSubtreeFracs = new float[] { 1, float.NaN, float.NaN };
        //engineDefCeres2.SearchParams.Execution.InFlightThisBatchLinkageEnabled = false;
      }

      //engineDefCeres1.SelectParams.UseDynamicVLoss = true;
      //engineDefCeres1.SearchParams.GameLimitUsageAggressiveness *= 1.2f;
      //engineDefCeres1.SearchParams.MoveFutilityPruningAggressiveness = 0.75f;
      //engineDefCeres1.SearchParams.EnableInstamoves=false; 
      //engineDefCeres1.SearchParams.TreeReuseEnabled = false;
      //engineDefCeres1.SearchParams.MLHBonusFactor = 0.1f;

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
      //engineDefCeres1.SelectParams.CPUCTAtRoot *=1.5f;


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

      //engineDefCeres1.SelectParams.PowerMeanNExponent = 0.12f;
      //engineDefCeres1.SelectParams.PolicyDecayFactor = 10f;
      //engineDefCeres1.SelectParams.PolicyDecayExponent = 0.40f;
      //engineDefCeres1.SelectParams.PolicySoftmax = 1.5f;

      //engineDefCeres1.SelectParams.EnableExplorationGuard = true;
      
      //      engineDefCeres1.SearchParams.Execution.FlowDualSelectors = false;
      // TODO: support this in GameEngineDefCeresUCI
      bool forceDisableSmartPruning = limit1.IsNodesLimit;
      forceDisableSmartPruning = false;
      if (forceDisableSmartPruning)
      {
        engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres1.SearchParams.MoveFutilityPruningAggressiveness = 0;
        engineDefCeres2.SearchParams.MoveFutilityPruningAggressiveness = 0;
      }

      //GameEngineDef engineDefCeresUCI = new GameEngineDefUCI("CeresUCI", new GameEngineUCISpec("CeresUCI", @"c:\dev\ceres\artifacts\release\net5.0\ceres.exe"));
      //      GameEngineDef engineDefCeresUCI1x = new GameEngineDefCeresUCI("CeresUCINew", evalDef1, overrideEXE: @"C:\dev\Ceres\artifacts\release\net5.0\ceres.exe");

      GameEngineDef engineDefCeresUCI1 = new GameEngineDefCeresUCI("CeresUCINew", evalDef1,
                                                                   overrideEXE: SoftwareManager.IsLinux ? @"/raid/dev/Ceres/artifacts/release/net5.0/Ceres.dll"
                                                                                                        : @"C:\dev\ceres\artifacts\release\net5.0\ceres.exe",
                                                                   disableFutilityStopSearch: forceDisableSmartPruning);

      GameEngineDef engineDefCeres93 = new GameEngineDefCeresUCI("Ceres93", evalDef2, overrideEXE: SoftwareManager.IsLinux ? @"/raid/dev/Ceres93/artifacts/release/5.0/Ceres.dll"
                                                                                                                           : @"C:\ceres\releases\v0.93\ceres.exe");
      GameEngineDef engineDefCeres94 = new GameEngineDefCeresUCI("Ceres94", evalDef2, overrideEXE: SoftwareManager.IsLinux ? @"/raid/dev/Ceres94/Ceres.dll"
                                                                                                                           : @"C:\ceres\releases\v0.94\ceres.exe");

      EnginePlayerDef playerCeres1UCI = new EnginePlayerDef(engineDefCeresUCI1, limit1);
      EnginePlayerDef playerCeres93 = new EnginePlayerDef(engineDefCeres93, limit2);
      EnginePlayerDef playerCeres94 = new EnginePlayerDef(engineDefCeres94, limit2);

      EnginePlayerDef playerCeres1 = new EnginePlayerDef(engineDefCeres1, limit1);
      EnginePlayerDef playerCeres2 = new EnginePlayerDef(engineDefCeres2, limit2);

      bool ENABLE_LC0 = evalDef1.Nets[0].Net.Type == NNEvaluatorType.LC0Library && (evalDef1.Nets[0].WeightValue == 1 && evalDef1.Nets[0].WeightPolicy == 1 && evalDef1.Nets[0].WeightM == 1);
      GameEngineDefLC0 engineDefLC1 = ENABLE_LC0 ? new GameEngineDefLC0("LC0_0", evalDef1, forceDisableSmartPruning, null, null) : null;
      GameEngineDefLC0 engineDefLC2 = ENABLE_LC0 ? new GameEngineDefLC0("LC0_2", evalDef2, forceDisableSmartPruning, null, null) : null;

      EnginePlayerDef playerEthereal = new EnginePlayerDef(engineDefEthereal, limit2);
      EnginePlayerDef playerStockfish11 = new EnginePlayerDef(engineDefStockfish11, limit2);
      EnginePlayerDef playerStockfish14 = new EnginePlayerDef(EngineDefStockfish14(), limit2);// * 350);
      EnginePlayerDef playerLC0 = ENABLE_LC0 ? new EnginePlayerDef(engineDefLC1, limit1) : null;
      EnginePlayerDef playerLC0_2 = ENABLE_LC0 ? new EnginePlayerDef(engineDefLC2, limit2) : null;

      //(playerCeres1.EngineDef as GameEngineDefCeres).SearchParams.DrawByRepetitionLookbackPlies = 40;

      if (false)
      {
        string BASE_NAME = "Stockfish238";// nice_lcx Stockfish238 ERET_VESELY203 endgame2

        // ===============================================================================
        SuiteTestDef suiteDef = new SuiteTestDef("Suite",
                                                 //@"\\synology\dev\chess\data\epd\chad_tactics-100M.epd",
                                                 //@"\\synology\dev\chess\data\epd\lichess_chad_bad.csv",
                                                 SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/epd/{BASE_NAME}.epd"
                                                                         : @$"\\synology\dev\chess\data\epd\{BASE_NAME}.epd",
                                                playerCeres1, playerCeres2, null);

//        suiteDef.MaxNumPositions = 200;
        suiteDef.EPDLichessPuzzleFormat = suiteDef.EPDFileName.ToUpper().Contains("LICHESS");

        //suiteDef.EPDFilter = s => !s.Contains(".exe"); // For NICE suite, these represent positions with multiple choices

        SuiteTestRunner suiteRunner = new SuiteTestRunner(suiteDef);

        SuiteTestResult suiteResult = suiteRunner.Run(POOLED ? 16 : 1, true, false);
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

      TournamentDef def = new TournamentDef("TOURN", playerCeres1, playerCeres2);
      //        TournamentDef def = new TournamentDef("TOURN", playerCeres1UCI, playerCeres93);
      //TournamentDef def = new TournamentDef("TOURN", playerCeres93, playerCeres1);


      def.NumGamePairs = 203;// 500;// 203;//203;// 102; 203
      def.ShowGameMoves = false;

//      string baseName = "tcec1819";
string      baseName = "4mvs_+90_+99";
//      baseName = "book-ply8-unifen-Q-0.25-0.40";
//      baseName = "test3";
//      baseName = "tcec_big";
//      baseName = "endgame-16-piece-book_Q-0.0-0.6_1";
      def.OpeningsFileName = SoftwareManager.IsLinux ? @$"/mnt/syndev/chess/data/openings/{baseName}.pgn"
                                                     : @$"\\synology\dev\chess\data\openings\{baseName}.pgn";

      if (false)
      {
        def.AdjudicationThresholdCentipawns = int.MaxValue;
        def.AdjudicationThresholdNumMoves = 3000;
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
      def.OpeningsFileName = "4mvs_+90_+99.pgn";// "TCEC1819.pgn";
      def.NumGamePairs = 250;
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

    static NNEvaluatorDef EvaluatorValueOnly(string netID1, string netID2, int gpuID, bool valueNet1)
    {
      string wtStr1 = valueNet1 ? "0.5;1.0;0.5" : "0.5;0.0;0.5";
      string wtStr2 = valueNet1 ? "0.5;0.0;0.5" : "0.5;1.0;0.5";
      NNEvaluatorDef spec = NNEvaluatorDef.FromSpecification($"LC0:{netID1}@{wtStr1},{netID2}@{wtStr2}", $"GPU:{gpuID}");
      return spec;
    }

    //      evalDef1 = EvaluatorValueOnly(NET1, NET2, 0, true);
    //      evalDef2 = EvaluatorValueOnly(NET1, NET2, 0, false);

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
        NNEvaluatorDynamic dyn = new NNEvaluatorDynamic(evaluators, (batch)
          => MCTSManager.ThreadSearchContext != null ? (MCTSManager.ThreadSearchContext.Manager.FractionSearchCompleted < FRACTION_SWITCH_ALTERNATE_NET ? 0 : 1) : 0);
        return dyn;
      }
      NNEvaluatorFactory.Custom1Factory = Build;
    }


  }
}
