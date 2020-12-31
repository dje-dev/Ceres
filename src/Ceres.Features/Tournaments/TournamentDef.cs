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
    /// Optional number of game pairs to run (opponents play both sides in a pair).
    /// If not specified, one game pair is run for each specified opening position.
    /// </summary>
    public int? NumGamePairs;

    /// <summary>
    /// Target for logging messages.
    /// </summary>
    [NonSerialized] public TextWriter Logger = Console.Out;

    /// <summary>
    /// Definition of the first player engine.
    /// </summary>
    public EnginePlayerDef Player1Def;

    /// <summary>
    /// Definition of the second player engine.
    /// </summary>
    public EnginePlayerDef Player2Def;

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


    public TournamentDef(string id, EnginePlayerDef player1Def, EnginePlayerDef player2Def)
    {
      ID = id;
      Player1Def = player1Def;
      Player2Def = player2Def;
      StartTime = DateTime.Now;
    }


    public TournamentDef Clone()
    {
      TournamentDef clone = ObjUtils.DeepClone<TournamentDef>(this);
      clone.Logger = this.Logger;
      return clone;
    }


    public void DumpParams()
    {
      Console.WriteLine($"TOURNAMENT    {ID}");
      Console.WriteLine($"  Game Pairs: {NumGamePairs} ");
      Console.WriteLine($"  Openings  : {OpeningsDescription()}");
      Console.WriteLine($"  Adjudicate: {AdjudicationThresholdNumMoves} moves at {AdjudicationThresholdCentipawns}cp"
                     + $"{(UseTablebasesForAdjudication ? " or via tablebases" : "")}");
      Console.WriteLine($"  Player 1  : {Player1Def} ");
      Console.WriteLine($"  Player 2  : {Player2Def} ");

//      Console.WriteLine("ID       : " + ID);
//      Console.WriteLine("Player 1 : " + Player1Def.ID + " with search limit " + Player1Def.SearchLimit);
//      Console.WriteLine("Player 2 : " + Player2Def.ID + " with search limit " + Player2Def.SearchLimit);

      if (Player1Def.EngineDef is GameEngineDefCeres &&
        Player2Def.EngineDef is GameEngineDefCeres)
        (Player1Def.EngineDef as GameEngineDefCeres).DumpComparison(Console.Out, Player2Def.EngineDef as GameEngineDefCeres, true);
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
      string openingsInfo;
      if (StartingFEN != null)
        openingsInfo = StartingFEN;
      else
        openingsInfo = OpeningsFileName;
      return openingsInfo;
    }
  }
}
