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

#endregion

namespace Ceres.APIExamples
{
  public static class TournamentTest
  {
    /// <summary>
    /// Test code. Currently configured for 703810 using 2A100 versus LC0.
    /// </summary>
    public static void Test()
    {
      string ETHERAL_EXE = @"\\synology\dev\chess\engines\Ethereal12.75-x64-popcnt-avx2.exe";
      string SF11_EXE = @"\\synology\dev\chess\engines\stockfish_11_x64_bmi2.exe";
      string SF12_EXE = @"\\synology\dev\chess\engines\stockfish_20090216_x64_avx2.exe";

      GameEngineUCISpec specEthereal = new GameEngineUCISpec("Ethereal12", ETHERAL_EXE);
      GameEngineUCISpec specSF = new GameEngineUCISpec("SF12", SF12_EXE);
      GameEngineUCISpec specLC0 = new GameEngineUCISpec("LC0", "lc0.exe");
      // 66511
      //NNEvaluatorDef def1 = NNEvaluatorDefFactory.SingleNet("j92-280", NNEvaluatorType.LC0Dll,1);

      string GPU = System.Environment.GetCommandLineArgs().Length > 2 ? "GPU:1" : "GPU:0";
      //      NNEvaluatorDef def0 = NNEvaluatorDefFactory.FromSpecification("LC0:j92-280", "GPU:1:POOLED");// POOLED");//:POOLED");
      //      NNEvaluatorDef def1 = NNEvaluatorDefFactory.FromSpecification("LC0:66666", "GPU:1:POOLED");// POOLED");//:POOLED");
      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification("LC0:66740", GPU);// POOLED");//:POOLED");
      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification("LC0:66740", GPU);// POOLED");//:POOLED");
                                                                                          // sv5300 j104.0-10000
                                                                                          //      def1.MakePersistent();
                                                                                          //def1.PersistentID = "PERSIST";

      //      NNEvaluatorDef def2 = NNEvaluatorDefFactory.FromSpecification("LC0:66581", "GPU:3");//:POOLED");


      //      SearchLimit slLC0 = SearchLimit.NodesPerMove(10_000);
      //      SearchLimit slEthereal = slLC0 * 875;
      //      SearchLimit slSF = slLC0 * 875;

      //specEthereal = specSF;

      string[] extraUCI = null;// new string[] {"setoption name Contempt value 5000" };

      const int NUM_THREADS = 14;
      const int HASH_SIZE_MB = 2048;
      string TB_PATH = CeresUserSettingsManager.Settings.DirTablebases;

      SearchLimit limit1 = SearchLimit.NodesPerMove(1_000);
      //limit = SearchLimit.SecondsForAllMoves(60);
      limit1 = SearchLimit.SecondsForAllMoves(5 * 60, 5 * 1);
      //limit = SearchLimit.SecondsForAllMoves(120, 0.5f);

      SearchLimit limit2 = SearchLimit.SecondsForAllMoves(3 * 60, 3 * 1);

      limit1 = SearchLimit.SecondsPerMove(1.5f);
      limit2 = SearchLimit.SecondsPerMove(1);

      GameEngineDefCeres engineDefCeres1 = new GameEngineDefCeres("Ceres1", evalDef1, new ParamsSearch() { TestFlag = true }, null, new ParamsSelect(), null);
      GameEngineDefCeres engineDefCeres2 = new GameEngineDefCeres("Ceres2", evalDef2, new ParamsSearch(), null, new ParamsSelect(), null);

      bool forceDisableSmartPruning = limit1.IsNodesLimit;
forceDisableSmartPruning = true;
      if (forceDisableSmartPruning)
      {
        engineDefCeres1.SearchParams.FutilityPruningStopSearchEnabled = false;
        engineDefCeres2.SearchParams.FutilityPruningStopSearchEnabled = false;
      }
      GameEngineDef engineDefEthereal = new GameEngineDefUCI("Etheral", new GameEngineUCISpec("Etheral", ETHERAL_EXE, NUM_THREADS, HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));
      GameEngineDef engineDefStockfish11 = new GameEngineDefUCI("SF11", new GameEngineUCISpec("SF11", SF11_EXE, NUM_THREADS, HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));
      GameEngineDef engineDefStockfish12 = new GameEngineDefUCI("SF12", new GameEngineUCISpec("SF12", SF12_EXE, NUM_THREADS, HASH_SIZE_MB, TB_PATH, uciSetOptionCommands: extraUCI));

      //GameEngineDef engineDefCeresUCI = new GameEngineDefUCI("CeresUCI", new GameEngineUCISpec("CeresUCI", @"c:\dev\ceres\artifacts\release\net5.0\ceres.exe"));
      GameEngineDef engineDefCeresUCI = new GameEngineDefCeresUCI("CeresUCI", evalDef1, overrideEXE: @"c:\dev\ceres\artifacts\release\net5.0\ceres.exe");


      GameEngineDefLC0 engineDefLC1 = new GameEngineDefLC0("LC0_0", evalDef1, forceDisableSmartPruning, null, null);
      GameEngineDefLC0 engineDefLC2 = new GameEngineDefLC0("LC0_1", evalDef2, forceDisableSmartPruning, null, null);


      EnginePlayerDef playerCeres1UCI = new EnginePlayerDef(engineDefCeresUCI, limit1);
      EnginePlayerDef playerCeres2UCI = new EnginePlayerDef(engineDefCeresUCI, limit2);

      EnginePlayerDef playerCeres1 = new EnginePlayerDef(engineDefCeres1, limit1);
      EnginePlayerDef playerCeres2 = new EnginePlayerDef(engineDefCeres2, limit2);
      EnginePlayerDef playerEthereal = new EnginePlayerDef(engineDefEthereal, limit1);
      EnginePlayerDef playerStockfish11 = new EnginePlayerDef(engineDefStockfish11, limit1);
      EnginePlayerDef playerStockfish12 = new EnginePlayerDef(engineDefStockfish12, limit1);
      EnginePlayerDef playerLC1 = new EnginePlayerDef(engineDefLC1, limit1);
      EnginePlayerDef playerLC2 = new EnginePlayerDef(engineDefLC2, limit2);

      //      def.SearchLimitEngine1 = def.SearchLimitEngine2 = SearchLimit.SecondsForAllMoves(15, 0.25f);
      //      def.SearchLimitEngine2 = def.SearchLimitEngine2 = SearchLimit.SecondsForAllMoves(15, 0.25f);


      //(playerCeres1.EngineDef as GameEngineDefCeres).SearchParams.DrawByRepetitionLookbackPlies = 40;

      if (false)
      {
        // ===============================================================================
        SuiteTestDef suiteDef = new SuiteTestDef("Suite", @"\\synology\dev\chess\data\epd\endgame2.epd", playerCeres1, playerCeres2, null);
        //        suiteDef.MaxNumPositions = 100;
        SuiteTestRunner suiteRunner = new SuiteTestRunner(suiteDef);
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

      //TournamentDef def = new TournamentDef("TOURN", playerCeres1, playerLC0);
      TournamentDef def = new TournamentDef("TOURN",
        System.Environment.GetCommandLineArgs().Length > 3 ? playerCeres2 : playerLC2, playerStockfish12);

      def.NumGamePairs = 50;
      //      def.ShowGameMoves = false;

      //      def.OpeningsFileName = @"HERT_2017\Hert500.pgn";

      //      def.StartingFEN = "1q6/2n4k/1r1p1pp1/RP1P2p1/2Q1P1P1/2N4P/3K4/8 b - - 8 71";
      //      def.OpeningsFileName = @"\\synology\dev\chess\data\openings\Drawkiller_500pos_reordered.pgn";//                                                                                                 
            def.OpeningsFileName = "TCEC19_NoomenSelect.pgn";
      //def.OpeningsFileName = "TCEC1819.pgn";

      //      def.AdjudicationThresholdCentipawns = 500;
      //      def.AdjudicationThresholdNumMoves = 3;

      const int CONCURRENCY = 1;// 15;
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
    }

  }
}
