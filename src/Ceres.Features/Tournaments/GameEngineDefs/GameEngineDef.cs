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

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Abstract base class for definitions of game engines,
  /// which could be internal Ceres engine, or external LC0 or UCI engines.
  /// </summary>
  [Serializable]
  public abstract class GameEngineDef
  {
    /// <summary>
    /// Identifying string of the engine.
    /// </summary>
    public readonly string ID;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    public GameEngineDef(string id)
    {
      ID = id;
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public abstract bool SupportsNodesPerGameMode { get; }

    
    /// <summary>
    /// Abstract virtual mthod to create underlying engine.
    /// </summary>
    /// <returns></returns>
    public abstract GameEngine CreateEngine();


    /// <summary>
    /// If applicable, modifies the device index associated with the underlying evaluator.
    /// The new index will be the current index plus a specified increment.
    /// </summary>
    /// <param name="deviceIndexIncrement"></param>
    public abstract void ModifyDeviceIndexIfNotPooled(int deviceIndexIncrement);
  }
}

