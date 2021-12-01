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

using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.MCTS.Params;
using System.Collections.Generic;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Set of static factor methods to facilitate creating various types of GameEngineDef objects.
  /// </summary>
  public static class GameEngineDefFactory
  {
    /// <summary>
    /// Returns a GameEngineDef for the Ceres engine running in this process.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="networkSpecification"></param>
    /// <param name="deviceSpecification"></param>
    /// <param name="paramsSearch"></param>
    /// <param name="paramsSelect"></param>
    /// <param name="logFileName"></param>
    /// <param name="secondaryNetworkSpecification"></param>
    /// <returns></returns>
    public static GameEngineDef CeresInProcess(string id, string networkSpecification, string deviceSpecification,
                                               ParamsSearch paramsSearch = null, ParamsSelect paramsSelect = null,
                                               string logFileName = null, string secondaryNetworkSpecification = null)
    {
      paramsSearch ??= new();
      paramsSelect ??= new();

      NNEvaluatorDef netDef = NNEvaluatorDefFactory.FromSpecification(networkSpecification, deviceSpecification);
      NNEvaluatorDef netDefSecondary = secondaryNetworkSpecification == null ?
                                                                        null : NNEvaluatorDefFactory.FromSpecification(secondaryNetworkSpecification, deviceSpecification);

      return new GameEngineDefCeres(id, netDef, netDefSecondary, paramsSearch, paramsSelect, null, logFileName);
    }


    /// <summary>
    /// Returns a GameDefEngine for Ceres engine running in an external EXE (via UCI protocol).
    /// </summary>
    /// <param name="id"></param>
    /// <param name="networkSpecification"></param>
    /// <param name="deviceSpecification"></param>
    /// <param name="exePath">path to Ceres EXE, or the executing assembly if null</param>
    /// <returns></returns>
    public static GameEngineDef CeresUCI(string id, string networkSpecification, string deviceSpecification, string exePath = null)
    {
      NNEvaluatorDef netDef = NNEvaluatorDefFactory.FromSpecification(networkSpecification, deviceSpecification);
      return new GameEngineDefCeresUCI(id, netDef, overrideEXE: exePath);
    }


    /// <summary>
    /// Returns a GameDefEngine for a generic engine running in an external EXE (via UCI protocol).
    /// </summary>
    /// <param name="id"></param>
    /// <param name="exePath"></param>
    /// <param name="numThreads"></param>
    /// <param name="hashSizeMB"></param>
    /// <param name="tbPath"></param>
    /// <param name="extraUCICommands"></param>
    /// <returns></returns>
    public static GameEngineDef UCI(string id, string exePath,
                                    int numThreads = 4, int hashSizeMB = 1024,
                                    string tbPath = null, List<string> extraUCICommands = null)
    {
      GameEngineUCISpec uciSpec = new GameEngineUCISpec(id, exePath, numThreads, hashSizeMB, tbPath, uciSetOptionCommands: extraUCICommands);
      return new GameEngineDefUCI(id, uciSpec);
    }


    /// <summary>
    /// Returns a GameDefEngine for a LC0 (Leela Chess Zero) chess engine running in an extrenal EXE (via UCI protocol).
    /// </summary>
    /// <param name="id"></param>
    /// <param name="networkSpecification"></param>
    /// <param name="deviceSpecification"></param>
    /// <param name="forceDisableSmartPruning"></param>
    /// <param name="extraCommandLineArgs"></param>
    /// <returns></returns>
    public static GameEngineDef LC0(string id, string networkSpecification, string deviceSpecification,
                                    bool forceDisableSmartPruning = false, string extraCommandLineArgs = null)
    {
      NNEvaluatorDef netDef = NNEvaluatorDefFactory.FromSpecification(networkSpecification, deviceSpecification);
      return new GameEngineDefLC0(id, netDef, forceDisableSmartPruning, null, extraCommandLineArgs: extraCommandLineArgs);
    }

  }
}
