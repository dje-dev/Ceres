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

using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

using Ceres.Chess.GameEngines;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Manages execution of a tournament (match between two players).
  /// </summary>
  public class TournamentGameRunner
  {
    /// <summary>
    /// Parent tournament definition.
    /// </summary>
    public readonly TournamentDef Def;

    /// <summary>
    /// Instance of first engine.
    /// </summary>
    public GameEngine Engine1;

    /// <summary>
    /// Instance of second engine.
    /// </summary>
    public GameEngine Engine2;

    /// <summary>
    /// Instance of optional "check" engine against which
    /// the moves of engine 2 are compared.
    /// </summary>
    public GameEngine Engine2CheckEngine;

    /// <summary>
    /// List of game engines in tournament
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
      Parallel.For(0, Def.Engines.Length, i => Engines[i] = Def.Engines[i].EngineDef.CreateEngine());
      Parallel.ForEach(Engines, e => e.Warmup());
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

      if (Def.CheckPlayer2Def != null)
      {
        Engine2CheckEngine = Def.CheckPlayer2Def.EngineDef.CreateEngine();
        Engine2CheckEngine.Warmup(Def.CheckPlayer2Def.SearchLimit.KnownMaxNumNodes);
      }

      Engine1.OpponentEngine = Engine2;
      Engine2.OpponentEngine = Engine1;
      Def.Player1Def = Def.Engines[gameEngine1Index];
      Def.Player2Def = Def.Engines[gameEngine2Index];

    }

#if NOT_USED
    static GameEngine GetEngine(GameEngineUCISpec engineSpec, string suffix, 
                                NNEvaluatorDef evaluatorDef,
                                ParamsSearch paramsSearch, ParamsSelect paramsSelect, IManagerGameLimit timeManager)
    {
      bool resetMovesBetweenMoves = !paramsSearch.TreeReuseEnabled;
      bool enableTranpsositions = paramsSearch.Execution.TranspositionMode != TranspositionMode.None;

      // Create requested type of engine
      if (engineSpec == null)
      {
        return new GameEngineCeresInProcess("Ceres_" + suffix, evaluatorDef, paramsSearch, paramsSelect, timeManager, null);
      }
      else if (engineSpec.Name == "LC0")
      {
        if (evaluatorDef == null) 
          throw new Exception("EvaluatorDef must be specified when running LC0 engine");

        // TODO: do we really want to emulate always here? probably not
        // WARNING: above.
        bool forceDisableSmartPruning = false;
        return new GameEngineLC0("LZ0_" + suffix, evaluatorDef.Nets[0].Net.NetworkID, 
                                 forceDisableSmartPruning, false, 
                                 paramsSearch, paramsSelect, evaluatorDef,                             
                                 null, CeresUserSettingsManager.GetLC0ExecutableFileName());
      }
      else
      {
        return engineSpec.CreateEngine();
      }
    }
#endif

  }
}
