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
using AutoMapper;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Features.GameEngines;
using Ceres.Features.Players;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Defines the parameters of a tournament between chess engines.
  /// </summary>
  [Serializable]
  public sealed class TournamentDef // sealed because the Clone method might not work if derived from
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
    /// If moves played by reference engine are forced to be same (for same position)
    /// across all games. This reduces test variance by removing non-determinism of
    /// some engines (e.g. Stockfish when multithreaded), insuring that each opponent
    /// encounters the same moves by Stockfish.
    /// </summary>
    public bool ForceReferenceEngineDeterministic = true;

    /// <summary>
    /// Name PGN or EPD file containing set of starting positions to be used.
    /// If null then chess start position is used for every game.
    /// The user setting "DirPGN" is consulted to determine the source directory.
    /// </summary>
    public string OpeningsFileName;

    /// <summary>
    /// Optional predicate which can be used to specify filter on which positions are accepted.
    /// </summary>    
    public Predicate<Position> AcceptPosPredicate;

    /// <summary>
    /// Optional list of PieceType which define positions that should be excluded if they contain any of those pieces.
    /// </summary>
    public List<PieceType> AcceptPosExcludeIfContainsPieceTypeList;

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
    /// Minimum number of moves before any adjudication (except tablebase) 
    /// will be allowed.
    /// </summary>
    public int AdjudicateMinNumMoves = 5;

    /// <summary>
    /// Minimum absolute evaluation (in centipawns)
    /// which must be exceeded by both engines
    /// for a win to be declared by adjudication.
    /// </summary>
    public int AdjudicateWinThresholdCentipawns = 350;


    /// <summary>
    /// Minimum number of moves for which both engines evaluations
    /// must be more extreme than AdjudicateWinThresholdCentipawns.
    /// </summary>
    public int AdjudicateWinThresholdNumMovesDecisive = 2;

    /// <summary>
    /// Minimum absolute absolute evaluation (in centipawns)
    /// which must not be exceeded by both engines
    /// for a draw to be declared by adjudication.
    /// </summary>
    public int AdjudicateDrawThresholdCentipawns = 10;

    /// <summary>
    /// Minimum number of moves for which both engines evaluations
    /// must be more less extreme than AdjudicateDrawThresholdCentipawns.
    /// </summary>
    public int AdjudicateDrawThresholdNumMoves = 10;

    /// <summary>
    /// If positions having a nonzero repetition count should immediately be scored as a draw.
    /// </summary>
    public bool AdjudicateDrawByRepetitionImmediately = true;

    /// <summary>
    /// The index of the processor group to which the engines should be affinitized. 
    /// </summary>
    public int ProcessGroupIndex = 0;

    /// <summary>
    /// Creation time of tournament (use as a unique ID for generating PGN)
    /// </summary>
    public readonly DateTime StartTime;

    /// <summary>
    /// If the tournament has been instructed to shut down (e.g. Ctrl-C pressed).
    /// </summary>
    public bool ShouldShutDown = false;

    /// <summary>
    /// If this instance is the coordinator in a distributed tournament.
    /// </summary>
    public bool IsDistributedCoordinator = false;

    /// <summary>
    /// The parent object (if this object was created by Clone method, otherwise this);
    /// </summary>
    internal TournamentDef parentDef;


    public TournamentDef(string id, params EnginePlayerDef[] engines)
    {
      ID = id;
      Engines = engines;
      StartTime = DateTime.Now;
      
      HashSet<string> engineIDs = new();

      foreach (EnginePlayerDef engine in Engines)
      {
        if (engines.Where(e => e != engine).Any(e => object.ReferenceEquals(engine, e)))
        {
          throw new Exception("playerDef must be different from each other");
        }

        if (engineIDs.Contains(engine.ID))
        {
          throw new Exception("Engine ID appears more than once: " + engine.ID);
        }

        engineIDs.Add(engine.ID); 
      }

      parentDef = this;
    }


    public void DumpParams(TextWriter logger)
    {
      logger.WriteLine($"TOURNAMENT:  {ID}");
      logger.WriteLine($"  Machine Name        : {Environment.MachineName} ");
      logger.WriteLine($"  Date/Time           : {DateTime.Now} ");
      logger.WriteLine($"  Game Pairs          : {NumGamePairs} ");
      logger.WriteLine($"  Openings            : {OpeningsDescription()}");
      if (ReferenceEngineId != null)
      {
        logger.WriteLine($"  Ref engine          : {ReferenceEngineId}");
        logger.WriteLine($"  Force deterministic : {ForceReferenceEngineDeterministic}");
      }
      logger.WriteLine($"  Adjudicate draw     : {AdjudicateDrawThresholdNumMoves} moves < {AdjudicateDrawThresholdCentipawns}cp");
      logger.WriteLine($"  Adjudicate win      : {AdjudicateWinThresholdNumMovesDecisive} moves at {AdjudicateWinThresholdCentipawns}cp");
      logger.WriteLine($"  Adjudicate via TB?  : {UseTablebasesForAdjudication}");

      for (int i = 0; i < Engines.Length; i++)
      {
        logger.WriteLine($"  Player {i + 1} : {Engines[i]}");
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
          toCompare.DumpComparison(logger, engine, true);
        }
      }

      logger.WriteLine();
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

    #region Cloning

    static IMapper cloneMapper = null;

    /// <summary>
    /// Returns a shallow clone of the TournamentDef.
    /// 
    /// This is necessary because an underlying NNEvaluatorDef may be replicated multiple times, 
    /// each with a different target GPU ID.
    /// 
    /// Previously ObjUtils.DeepClone was used but this relied upon the now-deprecated BinarySerialization.
    /// </summary>
    /// <returns></returns>
    public TournamentDef Clone()
    {
      if (cloneMapper == null)
      {
        MapperConfiguration config = new MapperConfiguration(cfg =>
        {
          cfg.CreateMap<TournamentDef, TournamentDef>();
          cfg.CreateMap<NNEvaluatorDef, NNEvaluatorDef>();
          cfg.CreateMap<GameEngineDefCeres, GameEngineDefCeres>();
          cfg.CreateMap<EnginePlayerDef, EnginePlayerDef>();
        });


        cloneMapper = config.CreateMapper();
      }

      TournamentDef clone = cloneMapper.Map<TournamentDef>(this);      
      clone.parentDef = this;
      clone.AcceptPosPredicate = this.AcceptPosPredicate;
      return clone;
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
           + $" adjudicate {AdjudicateWinThresholdNumMovesDecisive} moves at {AdjudicateWinThresholdCentipawns}cp "
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
