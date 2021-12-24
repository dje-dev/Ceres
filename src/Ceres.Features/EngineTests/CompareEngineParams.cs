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
using Ceres.Chess.Positions;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.Features.EngineTests
{
  /// <summary>
  /// Parameters which control CompareEnginesVersusOptimal runs.
  /// </summary>
  /// <param name="Description"></param>
  /// <param name="PGNFileName"></param>
  /// <param name="NumPositions"></param>
  /// <param name="PosFilter"></param>
  /// <param name="Player1"></param>
  /// <param name="NetworkID1"></param>
  /// <param name="Player2"></param>
  /// <param name="NetworkID2"></param>
  /// <param name="PlayerArbiter"></param>
  /// <param name="NetworkArbiterID"></param>
  /// <param name="Limit"></param>
  /// <param name="GPUIDs"></param>
  /// <param name="SearchModifier1"></param>
  /// <param name="SelectModifier1"></param>
  /// <param name="SearchModifier2"></param>
  /// <param name="SelectModifier2"></param>
  /// <param name="Verbose"></param>
  /// <param name="Engine1LimitMultiplier"></param>
  /// <param name="RunStockfishCrosscheck"></param>
  /// <param name="PosResultCallback"></param>
  public record CompareEngineParams(string Description, string PGNFileName, int NumPositions,
                                    Predicate<PositionWithHistory> PosFilter,
                                    CompareEnginesVersusOptimal.PlayerMode Player1, string NetworkID1,
                                    CompareEnginesVersusOptimal.PlayerMode Player2, string NetworkID2,
                                    CompareEnginesVersusOptimal.PlayerMode PlayerArbiter, string NetworkArbiterID,
                                    SearchLimit Limit, int[] GPUIDs = null,
                                    Action<ParamsSearch> SearchModifier1 = null, Action<ParamsSelect> SelectModifier1 = null,
                                    Action<ParamsSearch> SearchModifier2 = null, Action<ParamsSelect> SelectModifier2 = null,
                                    bool Verbose = true,
                                    float Engine1LimitMultiplier = 1.0f,
                                    bool RunStockfishCrosscheck = false,
                                    Action<CompareEnginePosResult> PosResultCallback = null)
  {
  }
}
