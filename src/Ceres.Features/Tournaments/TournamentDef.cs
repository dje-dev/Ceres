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
using Ceres.Features.Tournaments.Streaming;
using Ceres.MCTS.GameEngines;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Defines the randomization method for openings in a tournament.
  /// </summary>
  [Serializable]
  public enum OpeningRandomizationEnum
  {
    /// <summary>
    /// Openings are used in the order they appear in the file.
    /// </summary>
    None,

    /// <summary>
    /// Openings are shuffled once at the beginning of the tournament and then used in that order.
    /// This is deterministic.
    /// </summary>
    ShuffleDeterministic,

    /// <summary>
    /// Openings are drawn randomly without replacement for each game.
    /// </summary>
    Randomize
  }


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
    /// Optional predicate which can be used to specify filter on which games are accepted 
    /// </summary>
    public Predicate<Game> AcceptGamePredicate;


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
    public OpeningRandomizationEnum OpeningRandomization = OpeningRandomizationEnum.None;

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
    /// If nonzero, enables blunder diagnostic dumps: after each move, if an engine's evaluation
    /// improved by more than BlunderDumpThresholdQ compared to its prior move (implying
    /// the opponent just blundered), and the node count (N) of that move, the engine's prior move,
    /// and the intervening opponent move are all at least this value, and the opponent is an
    /// in-process Ceres MCGS engine, then the opponent engine's search graph is dumped to a
    /// "blunder_info_NNN.txt" file in the current working directory for post-hoc analysis.
    /// </summary>
    public int BlunderDumpThresholdN = 5000;

    /// <summary>
    /// Minimum improvement (in units of Q, the win probability in [-1, 1]) in the moving (reference)
    /// engine's evaluation (versus its prior move) required to flag a candidate blunder, and also the
    /// minimum amount by which the blundering engine's own Q evaluation must subsequently fall (on its
    /// next move) to confirm the blunder before a dump is written (see BlunderDumpThresholdN and
    /// BlunderDumpMaxPriorAbsQ).
    /// </summary>
    public float BlunderDumpThresholdQ = 0.12f;

    /// <summary>
    /// Maximum absolute value (in units of Q, the win probability in [-1, 1]) of the moving (reference)
    /// engine's evaluation BEFORE the opponent's move for that move to be eligible as a blunder. If the
    /// position was already more decisive than this (i.e. already clearly won or lost), the evaluation
    /// swing is treated as "piling on" in an already-decided game and is ignored. Set to 1.0 (or larger)
    /// to disable this filter.
    /// </summary>
    public float BlunderDumpMaxPriorAbsQ = 0.70f;

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
    /// Optional controller coordinating a cooperative pause/resume of the tournament's worker
    /// threads (driven by the Ctrl-P console command). Non-null only for local interactive
    /// tournaments; worker threads reach it via parentDef and null-check at each call site,
    /// exactly like ShouldShutDown. NonSerialized because TournamentDef is deep-cloned per
    /// worker thread (the shared controller lives only on parentDef).
    /// </summary>
    [NonSerialized] public TournamentPauseController PauseController;

    /// <summary>
    /// Optional callback invoked after each game is processed by any tournament thread.
    ///
    /// The callback is passed the TournamentResultStats accumulated so far (across all threads),
    /// allowing it to monitor the progress and partial results of the tournament as it runs.
    ///
    /// If the callback returns true then an orderly (early) shutdown of the tournament is
    /// requested: no further games are started and the tournament concludes once the games
    /// already in progress have finished (equivalent to setting ShouldShutDown, the same
    /// mechanism used by the Ctrl-C handler).
    ///
    /// The callback may be invoked from any of the worker threads. Invocations are serialized
    /// (and the supplied statistics are stable for the duration of the call) because the callback
    /// runs while holding the statistics lock; consequently it should return promptly and must
    /// not block or perform expensive work, as doing so would stall the other tournament threads.
    ///
    /// Marked NonSerialized because TournamentDef is deep-cloned (via BinaryFormatter) per worker
    /// thread; the callback is always resolved via parentDef so the per-thread clones need not
    /// carry it (mirroring how ShouldShutDown is coordinated through parentDef).
    /// </summary>
    [NonSerialized] public Func<TournamentResultStats, bool> PerGameCallback;

    /// <summary>
    /// Optional, transport-agnostic observer that receives live tournament/game/move events.
    /// Any consumer may attach (one example is the live streaming publisher used by the
    /// EngineBattle GUI in REMOTE mode). Resolved via parentDef and null-checked at each call
    /// site, exactly like PerGameCallback. When null there is zero behavioral change.
    /// Marked NonSerialized because TournamentDef is deep-cloned per worker thread.
    /// </summary>
    [NonSerialized] public ITournamentObserver Observer;

    /// <summary>
    /// If true (the default), a live streaming TCP listener is started automatically when the
    /// tournament runs, so any remote consumer can connect and watch games. Set false to
    /// disable. NonSerialized (resolved via parentDef).
    /// </summary>
    [NonSerialized] public bool EnableLiveStreaming = true;

    /// <summary>
    /// TCP port on which the live streaming listener accepts subscriber connections.
    /// NonSerialized (resolved via parentDef).
    /// </summary>
    [NonSerialized] public int LiveStreamPort = 7440;

    /// <summary>
    /// Steady-state interval (milliseconds) at which transient mid-search interim snapshots are
    /// streamed while an engine is thinking, so the live viewer updates instead of freezing between
    /// moves. The actual cadence is graduated around this value (faster at the start of a move,
    /// easing off for very long thinks). Set to 0 to disable interim updates (only completed-move
    /// frames are sent). NonSerialized (resolved via parentDef).
    /// </summary>
    [NonSerialized] public int LiveStreamInterimIntervalMs = 1000;

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

    public TournamentDef Clone()
    {
      TournamentDef clone = ObjUtils.DeepClone(this);
      clone.Logger = Logger;
      clone.parentDef = this;
      clone.AcceptPosPredicate = AcceptPosPredicate;
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

      string randomStr = OpeningRandomization switch
      {
        OpeningRandomizationEnum.Randomize => " Randomized",
        OpeningRandomizationEnum.ShuffleDeterministic => " Shuffled",
        _ => ""
      };

      return $"<TournamentDef {ID} with {NumGamePairs} game pairs from "
           + $"{openingsInfo}{randomStr} "
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
