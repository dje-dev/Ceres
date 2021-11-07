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
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Defines the parameters of a tournament between chess engines.
  /// </summary>
  [Serializable]
  public class TournamentDef
  {
    /// <summary>
    /// Descriptive identifying string of tournament.
    /// </summary>
    public string ID;

    /// <summary>
    /// The reference engine used in tournaments
    /// </summary>
    public string ReferenceEngineId = null;

    /// <summary>
    /// Optional number of game pairs to run (opponents play both sides in a pair).
    /// If not specified, one game pair is run for each specified opening position.
    /// </summary>
    public int? NumGamePairs;

    /// <summary>
    /// Target for logging messages.
    /// </summary>
    [NonSerialized] public TextWriter Logger = Console.Out;

    /// <summary>
    /// List of engines in the tournament
    /// </summary>
    public EnginePlayerDef[] Engines { get; set; }

    /// <summary>
    /// Definition of the first player engine.
    /// </summary>
    public EnginePlayerDef Player1Def;

    /// <summary>
    /// Definition of the second player engine.
    /// </summary>
    public EnginePlayerDef Player2Def;

    /// <summary>
    /// Optinal definition of engine used to check moves of second player.
    /// </summary>
    public EnginePlayerDef CheckPlayer2Def;

    /// <summary>
    /// If each move in each game should be output to the log/console.
    /// </summary>
    public bool ShowGameMoves = true;

    /// <summary>
    /// Name PGN or EPD file containing set of starting positions to be used.
    /// If null then chess start position is used for every game.
    /// The user setting "DirPGN" is consulted to determine the source directory.
    /// </summary>
    public string OpeningsFileName;

    /// <summary>
    /// Starting position for the games (unless OpeningsFileName is specified).
    /// Defaults to the chess start position is used for every game.
    /// </summary>
    public string StartingFEN = null;

    /// <summary>
    /// If the order of the openings from the book should be randomized.
    /// </summary>
    public bool RandomizeOpenings = false;

    /// <summary>
    /// If tablebases should be consulted for adjudication purposes.
    /// </summary>
    public bool UseTablebasesForAdjudication = true;

    /// <summary>
    /// Minimum absolute evaluation (in centipawns)
    /// which must be exceeded by both engines
    /// for a win to be declared by adjudication.
    /// </summary>
    public int AdjudicationThresholdCentipawns = 350;


    /// <summary>
    /// Minimum number of moves for which both engines evaluations
    /// must be more extreme than AdjudicationThresholdCentipawns.
    /// </summary>
    public int AdjudicationThresholdNumMoves = 2;

    /// <summary>
    /// Creation time of tournament (use as a unique ID for generating PGN)
    /// </summary>
    public readonly DateTime StartTime;

    /// <summary>
    /// If the touranment has been instructed to shut down (e.g. Ctrl-C pressed).
    /// </summary>
    public bool ShouldShutDown = false;

    /// <summary>
    /// The parent object (if this object was created by Clone method, otherwise this);
    /// </summary>
    internal TournamentDef parentDef;


    public TournamentDef(string id, params EnginePlayerDef[] engines)
    {
      ID = id;
      Engines = engines;
      StartTime = DateTime.Now;

      foreach (EnginePlayerDef engine in Engines)
      {
        if (engines.Where(e => e != engine).Any(e => object.ReferenceEquals(engine, e)))
        {
          throw new Exception("playerDef must be different from each other");
        }
      }

      parentDef = this;
    }

    public TournamentDef Clone()
    {
      TournamentDef clone = ObjUtils.DeepClone<TournamentDef>(this);
      clone.Logger = this.Logger;
      clone.parentDef = this;
      return clone;
    }

    public void DumpParams()
    {
      string refEngine = String.IsNullOrEmpty(ReferenceEngineId) ? "None" : ReferenceEngineId;
      Console.WriteLine($"TOURNAMENT:  {ID}");
      Console.WriteLine($"  Game Pairs : {NumGamePairs} ");
      Console.WriteLine($"  Openings   : {OpeningsDescription()}");
      Console.WriteLine($"  Ref engine : {refEngine}");
      Console.WriteLine($"  Adjudicate : {AdjudicationThresholdNumMoves} moves at {AdjudicationThresholdCentipawns}cp"
                     + $"{(UseTablebasesForAdjudication ? " or via tablebases" : "")}");

      for (int i = 0; i < Engines.Length; i++)
      {
        Console.WriteLine($"  Player {i + 1} : {Engines[i]}");
      }

      //Check if we need to dump comparison info too
      IEnumerable<GameEngineDefCeres> ceresEngines =
          Engines
          .Where(e => e.EngineDef is GameEngineDefCeres)
          .Select(e => e.EngineDef as GameEngineDefCeres);

      if (ceresEngines.Count() > 1)
      {
        GameEngineDefCeres toCompare = ceresEngines.First();

        //dump pairs here
        foreach (GameEngineDefCeres engine in ceresEngines.Skip(1))
        {
          toCompare.DumpComparison(Console.Out, engine, true);
        }
      }

      Console.WriteLine();
#if NOT
      ParamsDump.DumpParams(Console.Out, true,
                 UCIEngine1Spec, UCIEngine2Spec,
                 EvaluatorDef1, EvaluatorDef2,
                 SearchLimitEngine1, SearchLimitEngine2,
                 SelectParams1, SelectParams2,
                 SearchParams1, SearchParams2,
                 OverrideTimeManager1, OverrideTimeManager2,
                 SearchParams1.Execution, SearchParams2.Execution
                 );
#endif
    }


    #region Helper methods

    /// <summary>
    /// Adds a specified engine (with search limit) to set of tournament players.
    /// </summary>
    /// <param name="gameEngineDef"></param>
    /// <param name="searchLimit"></param>  
    public void AddEngine(GameEngineDef gameEngineDef, SearchLimit searchLimit)
    {
      EnginePlayerDef enginePlayerDef = new(gameEngineDef, searchLimit);
      EnginePlayerDef[] newEngines = new EnginePlayerDef[Engines.Length + 1];
      Array.Copy(Engines, newEngines, Engines.Length);
      newEngines[Engines.Length] = enginePlayerDef;
      Engines = newEngines;
    }


    /// <summary>
    /// Adds a set of specified engines (with search limit) to set of tournament players.
    /// </summary>
    /// <param name="gameEngineDef"></param>
    /// <param name="searchLimit"></param>
    public void AddEngines(SearchLimit searchLimit, params GameEngineDef[] gameEngineDefs)
    {
      foreach (GameEngineDef ged in gameEngineDefs)
      {
        AddEngine(ged, searchLimit);
      }
    }

#endregion

    /// <summary>
    /// Returns string summary.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      string openingsInfo = OpeningsDescription();

      return $"<TournamentDef {ID} with {NumGamePairs} game pairs from "
           + $"{openingsInfo} {(RandomizeOpenings ? " Randomized" : "")} "
           + $" adjudicate {AdjudicationThresholdNumMoves} moves at {AdjudicationThresholdCentipawns}cp "
           + $"{(UseTablebasesForAdjudication ? " or via tablebases" : "")}"
           + $"{Player1Def} vs {Player2Def}"
           + ">";
    }

    private string OpeningsDescription()
    {
      string openingsInfo = StartingFEN != null ? StartingFEN : OpeningsFileName;
      return openingsInfo;
    }
  }
}
