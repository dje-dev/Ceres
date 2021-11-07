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

using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.GameEngines;
using System;
using System.Collections.Generic;
using System.Reflection;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Subclass of GameEngineDef for Ceres engines which are
  /// controlled via UCI protocol and live in a separate process.
  /// 
  /// More commonly GameEmgineCeresInProcess is used due to greater
  /// efficiency and flexibility.
  /// </summary>
  [Serializable]
  public class GameEngineDefCeresUCI : GameEngineDef
  {
    /// <summary>
    /// Definition of underlying neural netowrk.
    /// </summary>
    public NNEvaluatorDef EvaluatorDef;

    /// <summary>
    /// Optional array of set option commands to be issued to engine at startup.
    /// </summary>
    public readonly List<string> UCISetOptionCommands;

    /// <summary>
    /// Optional callback to be called with progress updates.
    /// 
    /// Note: this will not serialize, 
    /// for example when tournament definitions are cloned via deep serialization.
    /// </summary>
    public readonly GameEngine.ProgressCallback Callback;

    /// <summary>
    /// Optional override of path to executable file (otherwise defaults to the running executable).
    /// </summary>
    public readonly string OverrideEXE;

    /// <summary>
    /// If every searche should be forced to run the full allotted amount of nodes/time.
    /// </summary>
    public readonly bool DisableFutilityStopSearch;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="paramsNN"></param>
    /// <param name="uciSetOptionCommands"></param>
    /// <param name="callback"></param>
    /// <param name="overrideEXE"></param>
    public GameEngineDefCeresUCI(string id, 
                                 NNEvaluatorDef evaluatorDef,
                                 List<string> uciSetOptionCommands = null,
                                 GameEngine.ProgressCallback callback = null,
                                 string overrideEXE = null,
                                 bool disableFutilityStopSearch = false) 
      : base(id)
    {
      EvaluatorDef = evaluatorDef;
      UCISetOptionCommands = uciSetOptionCommands;
      Callback = callback;
      OverrideEXE = overrideEXE ?? Assembly.GetExecutingAssembly().Location;
      DisableFutilityStopSearch = disableFutilityStopSearch;
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public override bool SupportsNodesPerGameMode => true;


    /// <summary>
    /// Abstract virtual mthod to create underlying engine.
    /// </summary>
    /// <returns></returns>
    public override GameEngine CreateEngine()
    {
      return new GameEngineCeresUCI(ID, EvaluatorDef, DisableFutilityStopSearch, 
                                    false, null, null, UCISetOptionCommands, Callback, OverrideEXE);
    }


    /// <summary>
    /// If applicable, modifies the device index associated with the underlying evaluator.
    /// The new index will be the current index plus a specified increment.
    /// </summary>
    /// <param name="deviceIndexIncrement"></param>
    public override void ModifyDeviceIndexIfNotPooled(int deviceIndexIncrement)
    {
      // Note that the GameEngineCeresUCI constructor
      // will verify the evaluator is not Pooled so no need to check here.
      EvaluatorDef.TryModifyDeviceID(EvaluatorDef.DeviceIndices[0] + deviceIndexIncrement);
    }

  }

}
