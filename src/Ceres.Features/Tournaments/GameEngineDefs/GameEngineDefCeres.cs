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
using System.IO;
using Ceres.Base.Misc;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.MCTS.LeafExpansion;
using Ceres.Features.GameEngines;
using Ceres.MCTS.Managers.Limits;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.Features.GameEngines
{
  [Serializable]
  public class GameEngineDefCeres : GameEngineDef
  {
    /// <summary>
    /// Definition of NN evaluator.
    /// </summary>
    public NNEvaluatorDef EvaluatorDef;

    /// <summary>
    /// Definition of secondary NN evaluator (optional).
    /// </summary>
    public NNEvaluatorDef EvaluatorDefSecondary;

    /// <summary>
    /// Parameters applied to MCTS search.
    /// </summary>
    public ParamsSearch SearchParams;

    /// <summary>
    /// Select parameters to be used by MCTS engine.
    /// </summary>
    public ParamsSelect SelectParams;

    /// <summary>
    /// Optional override limits manager.
    /// </summary>
    public IManagerGameLimit OverrideLimitManager;

    /// <summary>
    /// Optional name of log file to which detailed diagnostics information is written.
    /// </summary>
    public string LogFileName;


    /// <summary>
    /// Constructor.
    /// </summary>
    public GameEngineDefCeres(string id, NNEvaluatorDef evaluatorDef,
                              NNEvaluatorDef evaluatorDefSecondary,
                              ParamsSearch searchParams= null,
                              ParamsSelect selectParams = null, 
                              IManagerGameLimit overrideLimitManager = null,
                              string logFileName = null)
      : base(id)
    {
      // Make a defensive clone of the EvaluatorDef so it will definitely not be shared.
      EvaluatorDef = ObjUtils.DeepClone(evaluatorDef);
      EvaluatorDefSecondary = evaluatorDefSecondary == null ? null : ObjUtils.DeepClone(evaluatorDefSecondary);
      SearchParams = searchParams ?? new ParamsSearch();
      SelectParams = selectParams ?? new ParamsSelect();
      OverrideLimitManager = overrideLimitManager;
      LogFileName = logFileName;
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public override bool SupportsNodesPerGameMode => true;


    /// <summary>
    /// Implementation of virtual method to create underlying engine.
    /// </summary>
    /// <returns></returns>
    public override GameEngine CreateEngine()
    {
      return new GameEngineCeresInProcess(ID, EvaluatorDef, EvaluatorDefSecondary, SearchParams, SelectParams, 
                                          OverrideLimitManager, LogFileName);
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

      if (EvaluatorDefSecondary  != null && EvaluatorDef.DeviceCombo != NNEvaluatorDeviceComboType.Pooled)
      {
        EvaluatorDefSecondary.TryModifyDeviceID(EvaluatorDefSecondary.DeviceIndices[0] + deviceIndexIncrement);
      }
    }

    /// <summary>
    /// Returns string description.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<GameEngineDefCeres {ID} using {EvaluatorDef}>";
    }


    public void DumpComparison(TextWriter writer, GameEngineDefCeres other, bool differentOnly = true)
    {
      writer.WriteLine("\r\n-----------------------------------------------------------------------");
      writer.WriteLine("Evaluator 1     : " + EvaluatorDef);
      writer.WriteLine("Evaluator 2     : " + other.EvaluatorDef);
      if (EvaluatorDefSecondary != null) writer.WriteLine("Evaluator Secnd : " + other.EvaluatorDefSecondary);
      writer.WriteLine("Time Manager 1  : " + (OverrideLimitManager == null ? "(default)" : OverrideLimitManager));
      writer.WriteLine("Time Manager 2  : " + (other.OverrideLimitManager == null ? "(default)" : other.OverrideLimitManager));
      writer.WriteLine();

      writer.WriteLine("ENGINE 1 Options Modifications from Default");

      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSelect>(SelectParams, new ParamsSelect(), differentOnly));
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearch>(SearchParams, new ParamsSearch(), differentOnly));
      //      ParamsDump.DumpTimeManagerDifference(false, null, OverrideTimeManager);
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(SearchParams.Execution, new ParamsSearchExecution(), differentOnly));


      writer.WriteLine("\r\n-----------------------------------------------------------------------");
      writer.WriteLine("ENGINE 2 Options Modifications from Engine 1");

      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSelect>(other.SelectParams, SelectParams, differentOnly));
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearch>(other.SearchParams, SearchParams, differentOnly));
      //      ParamsDump.DumpTimeManagerDifference(differentOnly, OverrideTimeManager, other.OverrideTimeManager);
      writer.Write(ObjUtils.FieldValuesDumpString<ParamsSearchExecution>(other.SearchParams.Execution, SearchParams.Execution, differentOnly));
      writer.WriteLine();
    }

  }
}
