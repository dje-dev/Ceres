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
using System.Collections;
using System.Security.Cryptography.X509Certificates;
using Ceres.Chess.GameEngines;

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Definition of a game engine which is provided via
  /// an UCI-compliant executable.
  /// </summary>
  [Serializable]
  public class GameEngineDefUCI : GameEngineDef
  {
    /// <summary>
    /// Definition of the UCI engine.
    /// </summary>
    public readonly GameEngineUCISpec UCIEngineSpec;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="uciEngineSpec"></param>
    public GameEngineDefUCI(string id, GameEngineUCISpec uciEngineSpec)
      : base(id)
    {
      UCIEngineSpec = uciEngineSpec;
    }


    /// <summary>
    /// If the NodesPerGame time control mode is supported.
    /// </summary>
    public override bool SupportsNodesPerGameMode => false;


    /// <summary>
    /// Implementation of virtual method to create underlying engine.
    /// </summary>
    /// <returns></returns>
    public override GameEngine CreateEngine() => UCIEngineSpec.CreateEngine();


    /// <summary>
    /// If applicable, modifies the device index associated with the underlying evaluator.
    /// The new index will be the current index plus a specified increment.
    /// </summary>
    /// <param name="deviceIndexIncrement"></param>
    public override void ModifyDeviceIndexIfNotPooled(int deviceIndexIncrement)
    {
      // Nothing to do (not supported).
    }


    /// <summary>
    /// Returns string description.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<GameEngineDefUCI {UCIEngineSpec}>";
    }
  }
}
