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

using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.MCTS.LeafExpansion;
using Ceres.Features.GameEngines;
using Ceres.MCTS.Params;
using Ceres.Base.Misc;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Defines a game engine based on running an 
  /// Leela Chess Zero executable via standard UCI interface.
  /// </summary>
  [Serializable]
  public class GameEngineDefLC0 : GameEngineDef
  {
    /// <summary>
    /// Identifying string of engine.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Underlying network and devices to used.
    /// 
    /// Although the NNEvaluatorDef structure is used usually for 
    /// specifying Ceres evaluators, a subset of 
    /// evaluator options are 
    /// </summary>
    public readonly NNEvaluatorDef EvaluatorDef;
    
    /// <summary>
    /// If not null, LC0 is configured to emulate Ceres settings where comparable.
    /// </summary>
    public readonly ParamsSearch SearchParamsEmulate;

    /// <summary>
    /// If not null, LC0 is configured to emulate Ceres settings where comparable.
    /// </summary>
    public readonly ParamsSelect SelectParamsEmulate;

    /// <summary>
    /// If smart pruning should be disabled.
    /// </summary>
    public readonly bool ForceDisableSmartPruning;

    /// <summary>
    /// If a non-default executable file should be used 
    /// (otherwise LC0 binary in directory CeresUserSettings.DirLC0Binaries is used).
    /// </summary>
    public readonly string OverrideEXE;

    /// <summary>
    /// Optional additional string that is appended to the end of the LC0 arguments.
    /// </summary>
    public readonly string ExtraCommandLineArgs;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="evaluatorDef"></param>
    /// <param name="forceDisableSmartPruning"></param>
    /// <param name="searchParamsEmulate"></param>
    /// <param name="selectParamsEmulate"></param>
    /// <param name="overrideEXE"></param>
    /// <param name="extraCommandLineArgs"></param>
    public GameEngineDefLC0(string id,
                            NNEvaluatorDef evaluatorDef,
                            bool forceDisableSmartPruning,
                            ParamsSearch searchParamsEmulate = null, 
                            ParamsSelect selectParamsEmulate = null, 
                            string overrideEXE = null,
                            string extraCommandLineArgs = null)
      : base(id)
    {
      if ((SearchParamsEmulate == null) != (SelectParamsEmulate == null))
        throw new ArgumentException("SearchParamsEmulate and SelectParamsEmulate must be both provided or not");

      // Verify compatability of evaluator for LC0
      if (evaluatorDef == null) throw new ArgumentNullException(nameof(evaluatorDef));
      if (evaluatorDef.Nets.Length != 1) throw new Exception("Exactly one network must be specified for use with LC0.");
      if (evaluatorDef.Nets[0].Net.Type != NNEvaluatorType.LC0Library) throw new Exception("Network Type must be LC0Library");
      if (evaluatorDef.NetCombo != NNEvaluatorNetComboType.Single) throw new Exception("Network Type must be Single");

      ID = id;

      // Make a defensive clone of the EvaluatorDef so it will definitely not be shared.
      EvaluatorDef = ObjUtils.DeepClone(evaluatorDef);

      ForceDisableSmartPruning = forceDisableSmartPruning;

      SearchParamsEmulate = searchParamsEmulate;
      SelectParamsEmulate = selectParamsEmulate;
      OverrideEXE = overrideEXE;
      ExtraCommandLineArgs = extraCommandLineArgs;
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public override bool SupportsNodesPerGameMode => false;


    /// <summary>
    /// Implementation of virtual method to create underlying engine.
    /// </summary>
    /// <returns></returns>
    public override GameEngine CreateEngine()
    {
      bool emulate = SearchParamsEmulate != null;
        return new GameEngineLC0(ID, EvaluatorDef.Nets[0].Net.NetworkID, 
                                 ForceDisableSmartPruning, emulate,
                                 SearchParamsEmulate, SelectParamsEmulate, EvaluatorDef,                               
                                 null, OverrideEXE, extraCommandLineArgs:ExtraCommandLineArgs);   
    }


    /// <summary>
    /// If applicable, modifies the device index associated with the underlying evaluator.
    /// The new index will be the current index plus a specified increment.
    /// </summary>
    /// <param name="deviceIndexIncrement"></param>
    public override void ModifyDeviceIndexIfNotPooled(int deviceIndexIncrement)
    {
      if (EvaluatorDef.DeviceCombo != NNEvaluatorDeviceComboType.Pooled)
      {
        EvaluatorDef.TryModifyDeviceID(EvaluatorDef.DeviceIndices[0] + deviceIndexIncrement);
      }
    }

    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<GameEngineDefLC0 {ID} Evaluator={EvaluatorDef} "
        + $" {(OverrideEXE ?? "")}"
        + $"{(ForceDisableSmartPruning ? " with smart pruning disabled" : "")}>";
    }

  }
}

