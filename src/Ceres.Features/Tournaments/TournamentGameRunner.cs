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

using System.Threading.Tasks;

using Ceres.Chess.GameEngines;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Manages execution of a tournament (match between two players).
  /// </summary>
  internal class TournamentGameRunner
  {
    /// <summary>
    /// Parent tournament definition.
    /// </summary>
    public readonly TournamentDef Def;

    /// <summary>
    /// Instance of first engine that is currently active.
    /// </summary>
    public GameEngine Engine1;

    /// <summary>
    /// Instance of second engine that is currently active.
    /// </summary>
    public GameEngine Engine2;

    /// <summary>
    /// Instance of optional "check" engine against which
    /// the moves of engine 2 are compared.
    /// </summary>
    public GameEngine Engine2CheckEngine;

    /// <summary>
    /// List of game engines in tournament.
    /// During any individual game being run, the two currently
    /// active engines will be copied into Engine1 and Engine2 properties 
    /// </summary>
    public GameEngine[] Engines { get; set; }


    /// <summary>
    /// Constructor from a given tournament defintion.
    /// </summary>
    /// <param name="def"></param>
    public TournamentGameRunner(TournamentDef def)
    {
      Def = def;

      // Create and warmup all engines (in parallel).
      Engines = new GameEngine[Def.Engines.Length];
      Parallel.For(0, Def.Engines.Length,
        delegate (int i)
        {
          Engines[i] = Def.Engines[i].EngineDef.CreateEngine();
          Engines[i].Warmup();
        });
    }


    /// <summary>
    /// Pairing engines based on index in Engines list. Useful for Round Robin tournaments
    /// </summary>
    /// <param name="gameEngine1Index"></param>
    /// <param name="gameEngine2Index"></param>
    public void SetEnginePair(int gameEngine1Index, int gameEngine2Index)
    {
      Engine1 = Engines[gameEngine1Index];
      Engine2 = Engines[gameEngine2Index];

      if (Def.CheckPlayer2Def != null && Engine2CheckEngine == null)
      {
        Engine2CheckEngine = Def.CheckPlayer2Def.EngineDef.CreateEngine();
        Engine2CheckEngine.Warmup(Def.CheckPlayer2Def.SearchLimit.KnownMaxNumNodes);
      }

      Engine1.OpponentEngine = Engine2;
      Engine2.OpponentEngine = Engine1;
      Def.Player1Def = Def.Engines[gameEngine1Index];
      Def.Player2Def = Def.Engines[gameEngine2Index];
    }

  }
}
