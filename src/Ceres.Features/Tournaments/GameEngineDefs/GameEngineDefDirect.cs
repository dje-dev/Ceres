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

#endregion

namespace Ceres.Features.GameEngines
{
  /// <summary>
  /// Subclass of GameEngineDef that always returns the same engine instance from CreateEngine().
  /// Intended mainly for test/debugging purposes.
  /// </summary>
  public class GameEngineDefDirect : GameEngineDef
  {
    /// <summary>
    /// The engine instance to be returned from CreateEngine().
    /// </summary>
    public readonly GameEngine Engine;

    public override bool SupportsNodesPerGameMode => Engine.SupportsNodesPerGameMode;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="id"></param>
    /// <param name="gameEngine"></param>
    public GameEngineDefDirect(string id, GameEngine gameEngine) : base(id)
    {
      Engine = gameEngine;
    } 

    public override GameEngine CreateEngine() => Engine;


    public override void ModifyDeviceIndexIfNotPooled(int deviceIndexIncrement)
    {
      // Nothing to do.
      // TODO: Possibly log a warning?
    }
  }
}

