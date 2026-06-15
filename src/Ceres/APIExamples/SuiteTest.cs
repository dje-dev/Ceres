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
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.LC0.NNFiles;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Features.GameEngines;
using Ceres.Features.Suites;
using Ceres.MCTS.GameEngines;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Search.Params;


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


    /// <summary>
    /// Sample code demonstrating "multiengine mode": comparing an arbitrary number of engines
    /// (a mix of standard in-process Ceres engines and, optionally, external engines) on the
    /// same set of positions, each with its own short id and search limit and exactly one marked
    /// as the baseline. Results are shown in a single statistics block which is refreshed in
    /// place as the suite runs (best value per row green, worst red).
    ///
    /// Optionally spread concurrent workers over multiple GPUs by passing numConcurrent &gt; 1
    /// together with a flat pool of deviceIDs (e.g. numConcurrent 2, deviceIDs [0,1,2,3] with a
    /// "GPU:0,1" specification puts one worker on GPUs [0,1] and the other on [2,3]).
    /// </summary>
    public static void RunMultiEngineSuiteTest(string epdFileName = @"\\synology\dev\chess\data\epd\ERET_VESELY203.epd",
                                               string netSpec = "LC0:j92-280",
                                               int numConcurrent = 1,
                                               int[] deviceIDs = null,
                                               int maxPositions = 50)
    {
      string deviceSuffix = (numConcurrent > 1 && deviceIDs == null) ? ":POOLED" : "";
      string device = "GPU:0" + deviceSuffix;

      NNEvaluatorDef evalDef = NNEvaluatorDefFactory.FromSpecification(netSpec, device);

      // The engines to compare. Any number is allowed; each may be a standard in-process Ceres
      // engine (MCGS or MCTS) or an external engine. Here: an MCGS baseline, an MCGS variant,
      // and an MCTS engine sharing the same network.
      // MCGS engines require (non-null) search/selection parameter objects; give each engine its
      // own instances (DisableTreeReuse mutates them, so they must not be shared).
      GameEngineDef mcgsBaseline = new GameEngineDefCeresMCGS("MCGS_base", evalDef, new ParamsSearch(), new ParamsSelect());
      GameEngineDef mcgsVariant  = new GameEngineDefCeresMCGS("MCGS_var", evalDef, new ParamsSearch(), new ParamsSelect());
      GameEngineDef mctsEngine   = new GameEngineDefCeres("MCTS", evalDef);

      // Each engine can use its own search limit.
      SearchLimit limitSmall  = SearchLimit.NodesPerMove(5_000);
      SearchLimit limitMedium = SearchLimit.NodesPerMove(10_000);
      SearchLimit limitLarge  = SearchLimit.NodesPerMove(20_000);

      EnginePlayerDef pBase = new EnginePlayerDef(mcgsBaseline, limitMedium);
      EnginePlayerDef pVar  = new EnginePlayerDef(mcgsVariant, limitMedium);
      EnginePlayerDef pMCTS = new EnginePlayerDef(mctsEngine, limitMedium);

      // The new multiengine constructor: (short id, engine player def, is-baseline, search limit).
      // The middle engine (ID2) is the baseline here, so its difference cells render blank.
      SuiteTestDef def = new SuiteTestDef("MultiTest", epdFileName,
          ("ID1", pBase, false, limitSmall),
          ("ID2", pVar,  true,  limitMedium),
          ("ID3", pMCTS, false, limitLarge));
      def.MaxNumPositions = maxPositions;

      SuiteTestRunner runner = deviceIDs == null
                             ? new SuiteTestRunner(def)
                             : new SuiteTestRunner(def, numConcurrent, deviceIDs);

      MultiEngineSuiteResult result = runner.RunMultiEngine(numConcurrent);

      Console.WriteLine();
      Console.WriteLine($"Multiengine suite complete: {result.NumPositionsTested} positions, baseline = {result.Baseline?.ID}");
    }


    /// <summary>
    /// Sample code demonstrating the parameter-sweep convenience factory: a single MCGS engine
    /// definition is swept across a range of values for one parameter (one engine per value, the
    /// middle value taken as the baseline), and compared in the multiengine live statistics block.
    /// Only the modifier action needs to be written to choose which parameter is swept.
    /// </summary>
    public static void RunParameterSweepSuiteTest(string epdFileName = @"\\synology\dev\chess\data\epd\ERET_VESELY203.epd",
                                                  string netSpec = "LC0:j92-280",
                                                  int numConcurrent = 1,
                                                  int[] deviceIDs = null,
                                                  int maxPositions = 50)
    {
      string deviceSuffix = (numConcurrent > 1 && deviceIDs == null) ? ":POOLED" : "";
      NNEvaluatorDef evalDef = NNEvaluatorDefFactory.FromSpecification(netSpec, "GPU:0" + deviceSuffix);

      GameEngineDefCeresMCGS engineDef = new GameEngineDefCeresMCGS("Sweep", evalDef, new ParamsSearch(), new ParamsSelect());
      SearchLimit limit = SearchLimit.NodesPerMove(10_000);

      // Sweep a single parameter across several values (the middle value becomes the baseline).
      float[] values = { 0.01f, 0.02f, 0.04f, 0.08f, 0.12f };

      SuiteTestDef def = SuiteTestDef.CreateParameterSweep("ParamSweep", epdFileName, engineDef, limit, values,
          (search, value) => search.VisitSuboptimalityRejectThreshold = value);
      def.MaxNumPositions = maxPositions;

      SuiteTestRunner runner = deviceIDs == null
                             ? new SuiteTestRunner(def)
                             : new SuiteTestRunner(def, numConcurrent, deviceIDs);

      MultiEngineSuiteResult result = runner.RunMultiEngine(numConcurrent);

      Console.WriteLine();
      Console.WriteLine($"Parameter sweep complete: {result.NumPositionsTested} positions, baseline = {result.Baseline?.ID}");
    }

  }
}
