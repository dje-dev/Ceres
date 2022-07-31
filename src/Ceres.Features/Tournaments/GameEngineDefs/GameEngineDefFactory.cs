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
    /// <param name="networkSpec"></param>
    /// <param name="deviceSpec"></param>
    /// <param name="paramsSearch"></param>
    /// <param name="paramsSelect"></param>
    /// <param name="logFileName"></param>
    /// <param name="secondaryNetworkSpec"></param>
    /// <returns></returns>
    public static GameEngineDefCeres CeresInProcess(string id, string networkSpec, string deviceSpec,
                                                    ParamsSearch paramsSearch = null, ParamsSelect paramsSelect = null,
                                                    string logFileName = null, string secondaryNetworkSpec = null)
    {
      paramsSearch ??= new();
      paramsSelect ??= new();

      NNEvaluatorDef netDef = NNEvaluatorDefFactory.FromSpecification(networkSpec, deviceSpec);
      NNEvaluatorDef netDefSecondary = secondaryNetworkSpec == null ? null : NNEvaluatorDefFactory.FromSpecification(secondaryNetworkSpec, deviceSpec);

      return new GameEngineDefCeres(id, netDef, netDefSecondary, paramsSearch, paramsSelect, null, logFileName);
    }


    /// <summary>
    /// Returns a GameDefEngine for Ceres engine running in an external EXE (via UCI protocol).
    /// </summary>
    /// <param name="id"></param>
    /// <param name="networkSpec"></param>
    /// <param name="deviceSpec"></param>
    /// <param name="exePath">path to Ceres EXE, or the executing assembly if null</param>
    /// <returns></returns>
    public static GameEngineDefCeresUCI CeresUCI(string id, string networkSpec, string deviceSpec, string exePath = null)
    {
      NNEvaluatorDef netDef = NNEvaluatorDefFactory.FromSpecification(networkSpec, deviceSpec);
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
    public static GameEngineDefUCI UCI(string id, string exePath,
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
    /// <param name="networkSpec"></param>
    /// <param name="deviceSpec"></param>
    /// <param name="forceDisableSmartPruning"></param>
    /// <param name="extraCommandLineArgs"></param>
    /// <returns></returns>
    public static GameEngineDefLC0 LC0(string id, string networkSpec, string deviceSpec,
                                       bool forceDisableSmartPruning = false, string extraCommandLineArgs = null)
    {
      NNEvaluatorDef netDef = NNEvaluatorDefFactory.FromSpecification(networkSpec, deviceSpec);
      return new GameEngineDefLC0(id, netDef, forceDisableSmartPruning, null, extraCommandLineArgs: extraCommandLineArgs);
    }

  }
}
