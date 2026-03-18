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
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.GameEngines;

using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.GameEngines;

/// <summary>
/// Game definition for an MCGS engine.
/// </summary>
[Serializable]
public class GameEngineDefCeresMCGS : GameEngineDef
{
  public override bool SupportsNodesPerGameMode => true;

  public readonly NNEvaluatorDef EvaluatorDef;

  public readonly ParamsSearch SearchParams;
  public readonly ParamsSelect SelectParams;
  public readonly bool DisposeGraphAfterSearch;

  /// <summary>
  /// Optional fixed search limit known at engine creation time.
  /// When specified, allows optimizations for small searches.
  /// </summary>
  public readonly SearchLimit FixedSearchLimit;



  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="id"></param>
  /// <param name="evaluatorDef"></param>
  /// <param name="searchParams"></param>
  /// <param name="selectParams"></param>
  /// <param name="disposeGraphAfterSearch"></param>
  /// <param name="fixedSearchLimit"></param>
  /// <exception cref="ArgumentNullException"></exception>
  public GameEngineDefCeresMCGS(string id, NNEvaluatorDef evaluatorDef,
                                  ParamsSearch searchParams,
                                  ParamsSelect selectParams,
                                  bool disposeGraphAfterSearch = true,
                                  SearchLimit fixedSearchLimit = null) : base(id)
  {
    EvaluatorDef = evaluatorDef ?? throw new ArgumentNullException(nameof(evaluatorDef));

    SearchParams = searchParams;
    SelectParams = selectParams;
    DisposeGraphAfterSearch = disposeGraphAfterSearch;
    FixedSearchLimit = fixedSearchLimit;
  }


  public override GameEngine CreateEngine()
  {
    GameEngineCeresMCGSInProcess ret = new(ID, EvaluatorDef, SearchParams, SelectParams, 
                                           disposeGraphAfterSearch: DisposeGraphAfterSearch,
                                           fixedSearchLimit: FixedSearchLimit);
    ret.Warmup();

    return ret;
  }


  public override void ModifyDeviceIndexIfNotPooled(int deviceIndexIncrement)
  {
    if (EvaluatorDef.DeviceCombo != NNEvaluatorDeviceComboType.Pooled)
    {
      EvaluatorDef.TryModifyDeviceID(EvaluatorDef.DeviceIndices[0] + deviceIndexIncrement);
    }
  }

}
