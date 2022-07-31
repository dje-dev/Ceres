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
using Ceres.MCTS.Params;
using System.Collections.Generic;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Set of static factor methods to facilitate creating various types of GameEngine objects.
  /// </summary>
  public static class GameEngineFactory
  {
    /// <summary>
    /// Returns a GameEngine for the Ceres engine running in this process.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="networkSpec"></param>
    /// <param name="deviceSpec"></param>
    /// <param name="paramsSearch"></param>
    /// <param name="paramsSelect"></param>
    /// <param name="logFileName"></param>
    /// <param name="secondaryNetworkSpec"></param>
    /// <returns></returns>
    public static GameEngineCeresInProcess CeresInProcess(string id, string networkSpec, string deviceSpec,
                                                          ParamsSearch paramsSearch = null, ParamsSelect paramsSelect = null,
                                                          string logFileName = null, string secondaryNetworkSpec = null)
    {

      GameEngineDef def = GameEngineDefFactory.CeresInProcess(id, networkSpec, deviceSpec, paramsSearch, paramsSelect, 
                                                              logFileName, secondaryNetworkSpec); ;
      return def.CreateEngine() as GameEngineCeresInProcess;
    }


    /// <summary>
    /// Returns a GameEngine for Ceres engine running in an external EXE (via UCI protocol).
    /// </summary>
    /// <param name="id"></param>
    /// <param name="networkSpec"></param>
    /// <param name="deviceSpec"></param>
    /// <param name="exePath">path to Ceres EXE, or the executing assembly if null</param>
    /// <returns></returns>
    public static GameEngineUCI CeresUCI(string id, string networkSpec, string deviceSpec, string exePath = null)
    {
      GameEngineDefCeresUCI def = GameEngineDefFactory.CeresUCI(id, networkSpec, deviceSpec, exePath);
      return def.CreateEngine() as GameEngineUCI;
    }


    /// <summary>
    /// Returns a GameEngine for a generic engine running in an external EXE (via UCI protocol).
    /// </summary>
    /// <param name="id"></param>
    /// <param name="exePath"></param>
    /// <param name="numThreads"></param>
    /// <param name="hashSizeMB"></param>
    /// <param name="tbPath"></param>
    /// <param name="extraUCICommands"></param>
    /// <returns></returns>
    public static GameEngineUCI UCI(string id, string exePath,
                                    int numThreads = 4, int hashSizeMB = 1024,
                                    string tbPath = null, List<string> extraUCICommands = null)
    {
      GameEngineDefUCI def = GameEngineDefFactory.UCI(id, exePath, numThreads, hashSizeMB, tbPath, extraUCICommands);
      return def.CreateEngine() as GameEngineUCI;
    }


    /// <summary>
    /// Returns a GameEngine for a LC0 (Leela Chess Zero) chess engine running in an extrenal EXE (via UCI protocol).
    /// </summary>
    /// <param name="id"></param>
    /// <param name="networkSpec"></param>
    /// <param name="deviceSpec"></param>
    /// <param name="forceDisableSmartPruning"></param>
    /// <param name="extraCommandLineArgs"></param>
    /// <returns></returns>
    public static GameEngineLC0 LC0(string id, string networkSpec, string deviceSpec,
                                    bool forceDisableSmartPruning = false, string extraCommandLineArgs = null)
    {
      GameEngineDefLC0 def = GameEngineDefFactory.LC0(id, networkSpec, deviceSpec, forceDisableSmartPruning, extraCommandLineArgs);
      return def.CreateEngine() as GameEngineLC0;
    }

  }

}
