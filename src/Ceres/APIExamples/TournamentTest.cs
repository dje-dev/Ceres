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

#endregion

namespace Ceres.APIExamples
{
  public static class TournamentTest
  {

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
    const string SF13_EXE = @"\\synology\dev\chess\engines\stockfish_13_win_x64_bmi2.exe";

    static GameEngineUCISpec specEthereal = new GameEngineUCISpec("Ethereal12", ETHERAL_EXE);
    static GameEngineUCISpec specSF13 = new GameEngineUCISpec("SF13", SF13_EXE);
    static GameEngineUCISpec specLC0 = new GameEngineUCISpec("LC0", "lc0.exe");

    static string[] extraUCI = null;// new string[] {"setoption name Contempt value 5000" };
    static GameEngineDef engineDefEthereal = new GameEngineDefUCI("Etheral", new GameEngineUCISpec("Etheral", ETHERAL_EXE, SF_NUM_THREADS, SF_HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));
    static GameEngineDef engineDefStockfish11 = new GameEngineDefUCI("SF11", new GameEngineUCISpec("SF11", SF11_EXE, SF_NUM_THREADS, SF_HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));
    static GameEngineDef engineDefStockfish13 = new GameEngineDefUCI("SF13", new GameEngineUCISpec("SF13", SF13_EXE, SF_NUM_THREADS, SF_HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));

    const int SF_NUM_THREADS = 15;
    static string TB_PATH => CeresUserSettingsManager.Settings.TablebaseDirectory;
    const int SF_HASH_SIZE_MB = 2048;

    public static void PreTournamentCleanup()
    {
      KillCERES();

      File.Delete("Ceres1.log.txt");
      File.Delete("Ceres2.log.txt");
    }

    public static void TestSF(int index, bool gitVersion)
    {
      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification("LC0:j94-100", "GPU:" + index);
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("CeresInProc", evalDef1, 
                                                                  new ParamsSearch(), null, new ParamsSelect(),
                                                                  null, "CeresSF.log.txt");

      SearchLimit limitCeres = SearchLimit.SecondsForAllMoves(60, 1.25f) * 0.15f;
      SearchLimit limitSF = limitCeres * 1.5f;

      GameEngineDef engineDefCeresUCIGit = new GameEngineDefCeresUCI("CeresUCIGit", evalDef1, overrideEXE: @"C:\ceres\releases\v0.88\ceres.exe");
      EnginePlayerDef playerCeres = new EnginePlayerDef(gitVersion ? engineDefCeresUCIGit : engineDefCeres1, 
                                                        limitCeres);


      EnginePlayerDef playerSF = new EnginePlayerDef(engineDefStockfish13, limitSF);

      TournamentDef def = new TournamentDef("TOURN", playerCeres, playerSF);
      def.OpeningsFileName = "TCEC1819.pgn";
      //def.NumGamePairs = 10;
      def.ShowGameMoves = false;

      TournamentManager runner = new TournamentManager(def, 1);

      TournamentResultStats results;
      TimingStats stats = new TimingStats();
      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        results = runner.RunTournament();
      }

      Console.WriteLine();
      Console.WriteLine($"Tournament completed in {stats.ElapsedTimeSecs,8:F2} seconds.");
      Console.WriteLine(playerCeres + " " + results.GameOutcomesString);
    }

    /// <summary>
    /// Test code.
    /// </summary>
    public static void Test()
    {
      PreTournamentCleanup();

      if (true)
      {
        Parallel.Invoke(() => TestSF(0, true), () => { Thread.Sleep(7_000); TestSF(1, false); });
        System.Environment.Exit(3);
      }

      const bool POOLED = false;
      string GPUS = POOLED ? "GPU:0,1,2,3:POOLED"
                           : "GPU:0";
      //703810
      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification("LC0:j94-100", GPUS); // j64-210 LS16
      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification("LC0:j94-100", GPUS); // j104.1-30 61339

      // was 703810 @ 50k

      if (POOLED)
      {
        evalDef1.MakePersistent();
        evalDef2.MakePersistent();
      }

      //      SearchLimit slLC0 = SearchLimit.NodesPerMove(10_000);
      //      SearchLimit slEthereal = slLC0 * 875;
      //      SearchLimit slSF = slLC0 * 875;


      SearchLimit limit1 = SearchLimit.NodesPerMove(15_000);

      //limit1 = SearchLimit.SecondsForAllMoves(900, 15) * 0.04f;
      //limit1 = SearchLimit.SecondsPerMove(5);

      //limit1 = SearchLimit.NodesForAllMoves(500_000);//, 25_000);

//      limit1 = SearchLimit.SecondsForAllMoves(60, 1) *0.15f;
      //limit1 = SearchLimit.SecondsForAllMoves(40, 0.5f);
      //limit1 = SearchLimit.SecondsForAllMoves(900, 15) * 0.05f;

      //limit1 = SearchLimit.NodesPerMove(100_000);

      // Don't output log if very small games
      // (to avoid making very large log files or slowing down play).
      bool outputLog = limit1.EstNumNodes(30_000, false) > 10_000;
outputLog=false;
      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", evalDef1, new ParamsSearch(), null, new ParamsSelect(),
                                                                  null, outputLog ? "Ceres1.log.txt" : null);
      GameEngineDefCeres engineDefCeres2 = new GameEngineDefCeres("Ceres2", evalDef2, new ParamsSearch(), null, new ParamsSelect(),
                                                                  null, outputLog ? "Ceres2.log.txt" : null);

      //engineDefCeres1.OverrideTimeManager = new ManagerGameLimitCeres();
      //      engineDefCeres1.SearchParams.EnableInstamoves = false;

      ////////
      // THIS MIGHT BE GOOD - but only with T60 networks with quality MLH
      //      engineDefCeres1.SearchParams.MLHBonusFactor = 0.50f;
      ////////

      //engineDefCeres1.SelectParams.CPUCTAtRoot *= 1.5f;
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

      if (true)
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
      bool forceDisableSmartPruning = limit1.IsNodesLimit;
      if (forceDisableSmartPruning)
      {
        engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres1.SearchParams.MoveFutilityPruningAggressiveness = 0;
        engineDefCeres2.SearchParams.MoveFutilityPruningAggressiveness = 0;
      }

      //GameEngineDef engineDefCeresUCI = new GameEngineDefUCI("CeresUCI", new GameEngineUCISpec("CeresUCI", @"c:\dev\ceres\artifacts\release\net5.0\ceres.exe"));
      GameEngineDef engineDefCeresUCI1 = new GameEngineDefCeresUCI("CeresUCINew", evalDef1, overrideEXE: @"C:\dev\Ceres\artifacts\release\net5.0\ceres.exe");
      GameEngineDef engineDefCeresUCI2 = new GameEngineDefCeresUCI("CeresUCIGit", evalDef2, overrideEXE: @"C:\ceres\releases\v0.88\ceres.exe");


      GameEngineDefLC0 engineDefLC1 = new GameEngineDefLC0("LC0_0", evalDef1, forceDisableSmartPruning, null, null);
      GameEngineDefLC0 engineDefLC2TCEC = new GameEngineDefLC0("LC0_TCEC", evalDef2, forceDisableSmartPruning, null, null,
                                                                overrideEXE: @"c:\dev\lc0\lc_270rc2\lc0.exe",
                                                                extraCommandLineArgs: "--max-out-of-order-evals-factor=2.4 --max-collision-events=917");

      EnginePlayerDef playerCeres1UCI = new EnginePlayerDef(engineDefCeresUCI1, limit1);
      EnginePlayerDef playerCeres2UCI = new EnginePlayerDef(engineDefCeresUCI2, limit1);

      EnginePlayerDef playerCeres1 = new EnginePlayerDef(engineDefCeres1, limit1);
      EnginePlayerDef playerCeres2 = new EnginePlayerDef(engineDefCeres2, limit1);
      EnginePlayerDef playerEthereal = new EnginePlayerDef(engineDefEthereal, limit1);
      EnginePlayerDef playerStockfish11 = new EnginePlayerDef(engineDefStockfish11, limit1);
      EnginePlayerDef playerStockfish12 = new EnginePlayerDef(engineDefStockfish13, limit1);// * 350);
      EnginePlayerDef playerLC0 = new EnginePlayerDef(engineDefLC1, limit1);
      EnginePlayerDef playerLC0TCEC = new EnginePlayerDef(engineDefLC2TCEC, limit1);

      //      def.SearchLimitEngine1 = def.SearchLimitEngine2 = SearchLimit.SecondsForAllMoves(15, 0.25f);
      //      def.SearchLimitEngine2 = def.SearchLimitEngine2 = SearchLimit.SecondsForAllMoves(15, 0.25f);


      //(playerCeres1.EngineDef as GameEngineDefCeres).SearchParams.DrawByRepetitionLookbackPlies = 40;

      if (true)
      {
        // ===============================================================================
        SuiteTestDef suiteDef = new SuiteTestDef("Suite",
                                                //@"\\synology\dev\chess\data\epd\chad_tactics-100M.epd",
                                                //@"\\synology\dev\chess\data\epd\ERET.epd",
                                                @"\\synology\dev\chess\data\epd\ERET_VESELY203.epd",
                                                //   @"\\synology\dev\chess\data\epd\sts.epd",
                                                playerCeres1, null, playerCeres2UCI);
//        suiteDef.MaxNumPositions = 50;

        SuiteTestRunner suiteRunner = new SuiteTestRunner(suiteDef);

//        suiteRunner.Run(POOLED ? 20 : 4, true, false);
        suiteRunner.Run(1, true, false);
        return;
        // ===============================================================================
      }

      //      engineDefCeres2.SearchParams.TwofoldDrawEnabled = false;
      //engineDefCeres1.SearchParams.TreeReuseEnabled = false;
      //engineDefCeres2.SearchParams.TreeReuseEnabled = false;
      //engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled= false;
      //engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled= false;
      //engineDefLC0.SearchParamsEmulate.FutilityPruningStopSearchEnabled= false;

//      TournamentDef def = new TournamentDef("TOURN", playerCeres1UCI, playerCeres2UCI);// playerCeres2UCI);// playerLC0TCEC);
//      TournamentDef def = new TournamentDef("TOURN", playerCeres1, playerCeres2);// playerCeres2UCI);// playerLC0TCEC);
    TournamentDef def = new TournamentDef("TOURN", playerCeres1, playerLC0);

      //TournamentDef def = new TournamentDef("TOURN", playerLC0Tilps, playerLC0);

      //def.NumGamePairs = 10;
      //      def.ShowGameMoves = false;

      //      def.OpeningsFileName = @"HERT_2017\Hert500.pgn";

      //      def.StartingFEN = "1q6/2n4k/1r1p1pp1/RP1P2p1/2Q1P1P1/2N4P/3K4/8 b - - 8 71";
      //      def.OpeningsFileName = @"\\synology\dev\chess\data\openings\Drawkiller_500pos_reordered.pgn";//                                                                                                 
      //def.OpeningsFileName = "TCEC19_NoomenSelect.pgn";
      def.OpeningsFileName = "TCEC1819.pgn";
      // broken      def.OpeningsFileName = "TCEC_9-20.pgn";
//      def.OpeningsFileName = "4mvs_+90_+99.pgn";
      //      def.OpeningsFileName = "startpos.pgn";

      if (false)
      {
        def.AdjudicationThresholdCentipawns = 500;
        def.AdjudicationThresholdNumMoves = 3000;
        def.UseTablebasesForAdjudication = false;
      }

      const int CONCURRENCY = POOLED ? 16 : 4;
      TournamentManager runner = new TournamentManager(def, CONCURRENCY);

      TournamentResultStats results;

      //UCIEngineProcess.VERBOSE = true;

      TimingStats stats = new TimingStats();
      using (new TimingBlock(stats, TimingBlock.LoggingType.None))
      {
        results = runner.RunTournament();
      }

      Console.WriteLine();
      Console.WriteLine($"Tournament completed in {stats.ElapsedTimeSecs,8:F2} seconds.");
      Console.WriteLine(results.GameOutcomesString);

      Console.WriteLine();
      Console.WriteLine("<CRLF> to continue");
      Console.ReadLine();
    }

  }
}
