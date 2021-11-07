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
using Ceres.Chess.GameEngines;
using Ceres.Features.GameEngines;

#endregion

namespace Ceres.Features.Players
{
  /// <summary>
  /// Definition of an engine player (e.g. participating in a tournament or suite)
  /// including the underlying game engine and time limit.
  /// </summary>
  [Serializable]
  public class EnginePlayerDef
  {
    /// <summary>
    /// Identifying string of player.
    /// </summary>
    public string ID;

    /// <summary>
    /// Definition of underlying game engine.
    /// </summary>
    public readonly GameEngineDef EngineDef;

    /// <summary>
    /// Search limit for this player.
    /// </summary>
    public readonly SearchLimit SearchLimit;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="engineDef"></param>
    /// <param name="searchLimit"></param>
    public EnginePlayerDef(GameEngineDef engineDef, SearchLimit searchLimit, string id = null)
    {
      if (engineDef == null)
      {
        throw new NullReferenceException(nameof(engineDef));
      }

      ID = id ?? engineDef.ID;
      EngineDef = engineDef;
      SearchLimit = searchLimit;
    }


    /// <summary>
    /// Returns string description.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<EnginePlayerDef {ID} SearchLimit={SearchLimit} Def={EngineDef}>";
    }

  }
}
