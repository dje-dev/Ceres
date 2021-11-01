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
using System.Reflection;
using System.IO;
using System.Collections.Generic;

using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.MCTS.Params;
using Ceres.Base.OperatingSystem;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Subclass of GameEngine for Ceres chess engine accessed via UCI protocol.
  /// 
  /// Typically it will be more flexible and performant to 
  /// instead use GameEngineCeresInProcess, but nevertheless this class
  /// can be helpful for testing purposes or possibly eventually for distributed engines.
  /// </summary>
  public class GameEngineCeresUCI : GameEngineUCI
  {
    /// <summary>
    /// Constructor. 
    /// </summary>
    /// <param name="id">descriptive identifier</param>
    /// <param name="evaluatorDef">specification of the neural network to be used</param>
    /// <param name="forceDisableSmartPruning"></param>
    /// <param name="emulateCeresSettings"></param>
    /// <param name="searchParams"></param>
    /// <param name="selectParams"></param>
    /// <param name="uciSetOptionCommands"></param>
    /// <param name="callback"></param>
    /// <param name="overrideEXE">optinally the full name of the executable file(otherwise looks for Ceres executable in working directory)</param>
    public GameEngineCeresUCI(string id,
                              NNEvaluatorDef evaluatorDef,
                              bool forceDisableSmartPruning = false,
                              bool emulateCeresSettings = false,
                              ParamsSearch searchParams = null, 
                              ParamsSelect selectParams = null,
                              List<string> uciSetOptionCommands = null,
                              ProgressCallback callback = null,
                              string overrideEXE = null)
      : base(id, GetExecutableFN(overrideEXE), null, null, null, 
             forceDisableSmartPruning ? AddedDisableSetSmartPruning(uciSetOptionCommands) : uciSetOptionCommands,
             callback, false, ExtraArgsForEvaluator(evaluatorDef))
    {
      // TODO: support some limited emulation of options
      if (searchParams != null || selectParams != null) throw new NotSupportedException("Customized searchParams and selectParams not yet supported");
      if (evaluatorDef == null) throw new ArgumentNullException(nameof(evaluatorDef));
      if (emulateCeresSettings) throw new NotImplementedException(nameof(emulateCeresSettings));
      if (evaluatorDef.DeviceCombo == NNEvaluatorDeviceComboType.Pooled)
        throw new Exception("Evaluators for GameEngineDefCeresUCI should not be created as Pooled.");
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public override bool SupportsNodesPerGameMode => true;


    /// <summary>
    /// Optional name of program used as launcher for executable.
    /// </summary>
    public override string OverrideLaunchExecutable => SoftwareManager.IsLinux ? "dotnet" : null;



    #region Internal helper methods

    private static List<string> AddedDisableSetSmartPruning(List<string> options)
    {
      if (options == null)
      {
        options = new List<string>();
        options.Add("setoption name smartpruningfactor value 0");
      }
      return options;
    }

    /// <summary>
    /// Returns the network and device arguments string matching specified NNEvaluatorDef.
    /// </summary>
    /// <param name="evaluatorDef"></param>
    /// <returns></returns>
    static string ExtraArgsForEvaluator(NNEvaluatorDef evaluatorDef)
    {
      if (evaluatorDef.Nets[0].Net.Type != NNEvaluatorType.LC0Library)
      {
        throw new Exception("GameEngineCeresUCI cannot be used with network definition of type " + evaluatorDef.Nets[0].Net.Type);
      }
      else
      {
        string netSpecString = NNNetSpecificationString.ToSpecificationString(evaluatorDef.NetCombo, evaluatorDef.Nets);
        if (!netSpecString.Contains("Network=")) throw new Exception("Unsupported network specification");

        string deviceSpecString = NNDevicesSpecificationString.ToSpecificationString(evaluatorDef.DeviceCombo, evaluatorDef.Devices);
        if (!deviceSpecString.Contains("Device=")) throw new Exception("Unsupported device specification");
        return "UCI " + netSpecString + " " + deviceSpecString;
      }
    }


    /// <summary>
    /// Determines (and validates existence) the name of the executable to be used.
    /// </summary>
    /// <param name="overrideEXE"></param>
    /// <returns></returns>
    static string GetExecutableFN(string overrideEXE)
    {
      string executableFN = overrideEXE ?? Assembly.GetEntryAssembly().FullName;
      if (!File.Exists(executableFN)) throw new ArgumentException(nameof(overrideEXE), $"specified executable not found: {executableFN}");
      return executableFN;
    }

    #endregion


    public override void Dispose()
    {
      UCIRunner.Shutdown();
    }


    /// <summary>
    /// Returns string summary of object.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<GameEngineCeresUCI>";
    }

  }

}
