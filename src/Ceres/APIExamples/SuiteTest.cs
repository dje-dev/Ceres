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

using System.Collections.Generic;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.NNFiles;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;
using Ceres.Features.Suites;


#endregion

namespace Ceres.APIExamples
{
  public static class SuiteTest
  {
    /// <summary>
    /// Experimental sample code of runnings suites via the API.
    /// </summary>
    public static void RunSuiteTest()
    {
      const int PARALLELISM = 1;

      string deviceSuffix = PARALLELISM > 1 ? ":POOLED" : "";

      NNEvaluatorDef evalDef1 = NNEvaluatorDefFactory.FromSpecification("LC0:j92-280", $"GPU:1{deviceSuffix}");
      NNEvaluatorDef evalDef2 = NNEvaluatorDefFactory.FromSpecification("LC0:66733", $"GPU:1{deviceSuffix}");

      SearchLimit limit = SearchLimit.NodesPerMove(10_000);

      List<string> extraUCI = null; // new string[] { "setoption name Contempt value 5000" };

      GameEngineDef ged1 = new GameEngineDefCeres("Ceres1", evalDef1, null);
      GameEngineDef ged2 = new GameEngineDefCeres("Ceres1", evalDef2, null);
      GameEngineUCISpec geSF = new GameEngineUCISpec("SF12", @"\\synology\dev\chess\engines\stockfish_20090216_x64_avx2.exe",
                                                     32, 2048, CeresUserSettingsManager.Settings.TablebaseDirectory, 
                                                     uciSetOptionCommands: extraUCI);

      EnginePlayerDef ceresEngineDef1 = new EnginePlayerDef(ged1, limit);
      EnginePlayerDef ceresEngineDef2 = new EnginePlayerDef(ged2, limit);

      GameEngineDefUCI sf12EngineDef = new GameEngineDefUCI("SF12", geSF);
      EnginePlayerDef sfEngine = new EnginePlayerDef(sf12EngineDef, limit * 875);

      SuiteTestDef def = new SuiteTestDef("Test1", @"\\synology\dev\chess\data\epd\ERET_VESELY203.epd",
                                          ceresEngineDef1, ceresEngineDef2, sfEngine);
      def.MaxNumPositions = 1500;

      SuiteTestRunner ser = new SuiteTestRunner(def);
      ser.Run(PARALLELISM, true);
    }

  }
}
