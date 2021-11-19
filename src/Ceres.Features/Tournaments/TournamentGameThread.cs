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

using Ceres.Base.Benchmarking;
using Ceres.Base.Environment;
using Ceres.Chess.Positions;
using static Ceres.Base.Misc.StringUtils;
using Ceres.Chess;
using Ceres.Chess.Games;
using Chess.Ceres.PlayEvaluation;
using Ceres.Chess.GameEngines;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Iteration;
using System.Linq;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Manage execution of a single thread of a tournament,
  /// running the main move loop which alternates between players.
  /// </summary>
  public class TournamentGameThread
  {
    public delegate TournamentGameInfo GamePairRunnerDelegate(string pgnFileName, int gameSequenceNum, int openingIndex, int roundNumber, bool engine2White);

    /// <summary>
    /// Definition of associated parent tournament definition.
    /// </summary>
    public readonly TournamentDef Def;

    /// <summary>
    /// Collection of tournament result statistics.
    /// </summary>
    public readonly TournamentResultStats ParentStats;

    // TODO: This is not correct to be static in the case that 
    //       multiple tournaments are being run simultaneously.
    static bool havePrintedHeaders = false;



    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="def"></param>
    /// <param name="parentTestResults"></param>
    public TournamentGameThread(TournamentDef def, TournamentResultStats parentTestResults)
    {
      Def = def;
      ParentStats = parentTestResults;
    }


    static PositionsWithHistory openings = new PositionsWithHistory();

    static object writePGNLock = new();

    public long TotalNodesEngine1 = 0;
    public long TotalNodesEngine2 = 0;
    public int TotalMovesEngine1 = 0;
    public int TotalMovesEngine2 = 0;
    public float TotalTimeEngine1 = 0;
    public float TotalTimeEngine2 = 0;
    public float NumGames;


    public TournamentGameRunner Run;

    int numGamePairsLaunched = 0;
    readonly object lockObj = new();

    /// <summary>
    /// Method called by threads to get the next available game to be played.
    /// </summary>
    /// <returns></returns>
    int GetNextOpeningIndexForLocalThread(int maxOpenings)
    {
      if (Def.RandomizeOpenings) throw new NotImplementedException();

      lock (lockObj)
      {
        if (numGamePairsLaunched < maxOpenings)
        {
          return numGamePairsLaunched++;
        }
        else
          return -1;
      }
    }

    public void RunGameTests(int runnerIndex, Func<int> getGamePairToProcess,
                             Action<TournamentGameInfo, TournamentGameInfo> doneGamePairCallback = null)
    {
      Run = new TournamentGameRunner(Def);

      // Create a file name that will be common to all threads in tournament.
      string pgnFileName;
      if (Run.Engines.Length > 2)
      {
        pgnFileName = Path.Combine(CeresUserSettingsManager.Settings.DirCeresOutput, "match_" + Def.ID + "_"
                                    + "MultiEngine" + "_" + Def.StartTime.Ticks + ".pgn");
      }
      else
      {
        pgnFileName = Path.Combine(CeresUserSettingsManager.Settings.DirCeresOutput, "match_" + Def.ID + "_"
                                    + Run.Engines[0].ID + "_" + Run.Engines[1].ID + "_" + Def.StartTime.Ticks + ".pgn");
      }
      lock (writePGNLock) File.AppendAllText(pgnFileName, "");

      havePrintedHeaders = false;

      SetOpeningsSource();

      Random rand = new Random();

      // Def.Logger.WriteLine($"Begin {def.NumGames} game test with limit {def.SearchLimitEngine1} {def.SearchLimitEngine2} ID { gameSequenceNum } ");
      int numPairsProcessed = 0;

      int maxOpenings = Math.Min(Def.NumGamePairs ?? int.MaxValue, openings.Count);

      while (!Def.parentDef.ShouldShutDown)
      {
        int openingIndex = getGamePairToProcess();

        // Look for sentinel indicating end
        if (openingIndex < 0)
        {
          break;
        }
#if NOT
        // Some engines such as LZ0 doesn't seem to support "setoption name Clear Hash"
        // Therefore we don't clear cache every time, instead let it stick around unless first game in a set
        bool clearHashTable = isFirstGameOfPossiblePair;
#endif

        int gameSequenceNum = openingIndex * 2;

        // The engine going second could possibly benefit 
        // if REUSE_POSITION_EVALUATIONS_FROM_OTHER_TREE is true.
        // Therefore we alternate each pair which one goes first.
        int roundNumber = 1 + gameSequenceNum / 2;

        // Alternate which side gets white to avoid any clustered imbalance
        // which could happen if multiple threads are active
        // (possibly otherwise all first giving white to one particular side).
        bool engine2White = (numPairsProcessed + runnerIndex) % 2 == 0;

        if (string.IsNullOrEmpty(Def.ReferenceEngineId))
        {
          List<GameEngine> list = new List<GameEngine>(Run.Engines);
          int engine1Index = 0;
          while (list.Count > 1)
          {
            for (int i = 1; i < list.Count; i++)
            {
              Run.SetEnginePair(engine1Index, i + engine1Index);

              TournamentGameInfo gameInfo = RunGame(pgnFileName, engine2White, openingIndex, gameSequenceNum, roundNumber);
              TournamentGameInfo gameReverseInfo = RunGame(pgnFileName, !engine2White, openingIndex, gameSequenceNum + 1, roundNumber);
              doneGamePairCallback?.Invoke(gameInfo, gameReverseInfo);
            }
            list.RemoveAt(0);
            engine1Index++;
          }
          numPairsProcessed++;
        }

        else
        {
          GameEngine refEngine = Run.Engines.FirstOrDefault(e => e.ID == Def.ReferenceEngineId);
          if (refEngine == null)
            throw new Exception("Error in loading reference engine");
          int index =  Array.IndexOf(Run.Engines, refEngine);
          for (int i = 0; i < Run.Engines.Length; i++)
          {
            if (index == i)
            {
              continue;
            }
            GameEngine engineToPair = Run.Engines[i];
            Run.SetEnginePair(index, i);
            TournamentGameInfo gameInfo = RunGame(pgnFileName, engine2White, openingIndex, gameSequenceNum, roundNumber);
            TournamentGameInfo gameReverseInfo = RunGame(pgnFileName, !engine2White, openingIndex, gameSequenceNum + 1, roundNumber);
            doneGamePairCallback?.Invoke(gameInfo, gameReverseInfo);
          }
          numPairsProcessed++;
        }
      }

      foreach (GameEngine engine in Run.Engines)
      {
        engine.Dispose();
        Run.Engine2CheckEngine?.Dispose();
      }
    }

    private void SetOpeningsSource()
    {
      if (Def.OpeningsFileName != null)
      {
        openings = PositionsWithHistory.FromEPDOrPGNFile(Def.OpeningsFileName);
      }
      else if (Def.StartingFEN != null)
      {
        openings = PositionsWithHistory.FromFEN(Def.StartingFEN, Def.NumGamePairs ?? 1);
      }
      else
      {
        openings = PositionsWithHistory.FromFEN(Position.StartPosition.FEN, Def.NumGamePairs ?? 1);
      }
    }


    /// <summary>
    /// Returns list of supplemental name/value tags to be 
    /// included in the PGN header with helpful ancillary metadata.
    /// </summary>
    /// <param name="engine2White"></param>
    /// <returns></returns>
    List<(string name, string value)> GetSupplementalTags(bool engine2White)
    {
      List<(string name, string value)> tags = new List<(string name, string value)>();

      tags.Add((engine2White ? "BlackEngine" : "WhiteEngine", Def.Player1Def.EngineDef.ToString()));
      tags.Add((engine2White ? "WhiteEngine" : "BlackEngine", Def.Player2Def.EngineDef.ToString()));
      tags.Add(("CeresVersion", CeresVersion.VersionString));
      tags.Add(("CeresGIT", GitInfo.VersionString));

      return tags;
    }


    HashSet<int> openingsFinishedAtLeastOnce = new();

#if NOT
    public record TournamentGameRunRequest
    {
      public string PGNFileName;
      public bool Engine2White;
      public int OpeningIndex;
      public int GameSequenceNum;
      public int RoundNumber;
    }
#endif

    private TournamentGameInfo RunGame(string pgnFileName, bool engine2White, int openingIndex, int gameSequenceNum, int roundNumber)
    {
      TournamentGameInfo thisResult;

      // Add some supplemental tags with round number and also
      // information about Ceres engine configuration.
      List<(string name, string value)> extraTags = GetSupplementalTags(engine2White);
      extraTags.Add(("Round", roundNumber.ToString()));

      PGNWriter pgnWriter = new PGNWriter(null, engine2White ? Run.Engine2.ID : Run.Engine1.ID,
                                  engine2White ? Run.Engine1.ID : Run.Engine2.ID,
                                  extraTags.ToArray());


      TimingStats gameTimingStats = new TimingStats();
      using (new TimingBlock(gameTimingStats, TimingBlock.LoggingType.None))
      {
        // Start new game
        string gameID = $"{Def.ID}_GAME_SEQ{gameSequenceNum}_OPENING_{openingIndex}";
        Run.Engine1.ResetGame(gameID);
        Run.Engine2.ResetGame(gameID);
        Run.Engine2CheckEngine?.ResetGame(gameID);

        bool checkTablebases = Def.UseTablebasesForAdjudication && CeresUserSettingsManager.Settings.TablebaseDirectory != null;
        thisResult = DoGameTest(Def.Logger, pgnWriter, gameSequenceNum, openingIndex,
                                Run.Engine1, Run.Engine2,
                                Run.Engine2CheckEngine, engine2White,
                                Def.Player1Def.SearchLimit, Def.Player2Def.SearchLimit,
                                checkTablebases, Def.ShowGameMoves);

        if (MCTSDiagnostics.TournamentDumpEngine2EndOfGameSummary)
        {
          Run.Engine2.DumpFullMoveHistory(thisResult.GameMoveHistory, engine2White);
        }

        WritePGNResult(engine2White, thisResult, pgnWriter);

        if (pgnFileName != null)
        {
          lock (writePGNLock) File.AppendAllText(pgnFileName, pgnWriter.GetText());
        }

      }
      
      thisResult.PlayerWhite = engine2White ? Run.Engine2.ID : Run.Engine1.ID;
      thisResult.PlayerBlack = engine2White ? Run.Engine1.ID : Run.Engine2.ID;
      UpdateStatsAndOutputSummaryFromGameResult(pgnFileName, engine2White, openingIndex, gameSequenceNum, thisResult);

      return thisResult;
    }

    internal void UpdateStatsAndOutputSummaryFromGameResult(string pgnFileName, bool engine2White, int openingIndex, int gameSequenceNum, TournamentGameInfo thisResult)
    {
      lock (ParentStats)
      {
        ParentStats.UpdateTournamentStats(thisResult, Run.Engine1);
      }

      // Only show headers first time for first thread
      if (!havePrintedHeaders)
      {
        OutputHeaders(pgnFileName);
      }

      OutputGameResultInfo(engine2White, openingIndex, gameSequenceNum, thisResult);
    }

    private void OutputGameResultInfo(bool engine2White, int openingIndex, int gameSequenceNum, TournamentGameInfo thisResult)
    {
      ParentStats.GameInfos.Add(thisResult);

      PlayerStat player = engine2White ?
        ParentStats.GetPlayer(Run.Engine2.ID, Run.Engine1.ID): 
        ParentStats.GetPlayer(Run.Engine1.ID, Run.Engine2.ID);
      
      float gNumber = NumGames + 1;
      (float eloMin, float eloAvg, float eloMax) = EloCalculator.EloConfidenceInterval(player.PlayerWins, player.Draws, player.PlayerLosses);
      float eloSD = eloMax - eloAvg;
      float los = EloCalculator.LikelihoodSuperiority(player.PlayerWins, player.Draws, player.PlayerLosses);

      string wdlStr = $"{player.PlayerWins,3} {player.Draws,3} {player.PlayerLosses,3}";

      // Show a "." after the opening index if this was the second of the pair of games played.
      string openingPlayedBothWaysStr = gameSequenceNum % 2 == 1 && openingsFinishedAtLeastOnce.Contains(openingIndex) ? "." : " ";

      string player1ForfeitChar = thisResult.ShouldHaveForfeitedOnLimitsEngine1 ? "f" : " ";
      string player2ForfeitChar = thisResult.ShouldHaveForfeitedOnLimitsEngine2 ? "f" : " ";

      const string TournamentGameResultReasonCodes = "CSTMERVL";

      char resultReasonChar = TournamentGameResultReasonCodes[(int)thisResult.ResultReason];

      static bool EvalInconsistent(TournamentGameResult outcome, float scoreCP, int numMovesBack)
      {
        float mult = numMovesBack % 2 == 1 ? 1 : -1; // adjust for side to move perspective
        float adjustedCP = mult * scoreCP;
        return (outcome == TournamentGameResult.Win && adjustedCP < 100) ||
               (outcome == TournamentGameResult.Draw && MathF.Abs(adjustedCP) > 50) ||
               (outcome == TournamentGameResult.Loss && adjustedCP > -100);
      }

      // Possibly show one or two "?" characters if the evaluations of the final two positions
      // are very inconsisten with the actual game outcome (as a diagnostic).
      string questionableFinalCP = "  ";
      float endingCP = 0;
      if (thisResult.GameMoveHistory.Count > 0)
      {
        bool finalMoveIsPlayer1 = (thisResult.GameMoveHistory[^1].Side == SideType.White) != engine2White;
        float reverseMult = finalMoveIsPlayer1 ? 1 : -1;
        endingCP = reverseMult * thisResult.GameMoveHistory[^1].ScoreCentipawns;

        if (EvalInconsistent(thisResult.Result, reverseMult * thisResult.GameMoveHistory[^1].ScoreCentipawns, 1))
        {
          questionableFinalCP = "? ";
        }

        if (thisResult.GameMoveHistory.Count > 1)
        {
          if (EvalInconsistent(thisResult.Result, reverseMult * thisResult.GameMoveHistory[^2].ScoreCentipawns, 2))
          {
            questionableFinalCP = questionableFinalCP[0] + "?";
          }

        }
      }

      lock (outputLockObj)
      {
        string checkEnginePlyDifferent = thisResult.NumEngine2MovesDifferentFromCheckEngine == 0 ? "   " : $"{thisResult.NumEngine2MovesDifferentFromCheckEngine,3:N0}";
        if (Def.ShowGameMoves) Def.Logger.WriteLine();
        if (engine2White)
          Def.Logger.Write($" {TrimmedIfNeeded(Def.Player2Def.ID, 10),-10} {TrimmedIfNeeded(Def.Player1Def.ID, 10),-10}");
        else
          Def.Logger.Write($" {TrimmedIfNeeded(Def.Player1Def.ID, 10),-10} {TrimmedIfNeeded(Def.Player2Def.ID, 10),-10}");
        Def.Logger.Write($"{eloAvg,4:0} {eloSD,4:0} {100.0f * los,5:0}  ");
        Def.Logger.Write($"{gNumber,5} {DateTime.Now.ToString().Split(" ")[1],10}  {gameSequenceNum,4:F0}  {openingIndex,4:F0}{openingPlayedBothWaysStr}  ");

        if (engine2White)
        {
          Def.Logger.Write($"{thisResult.TotalTimeEngine2,8:F2}{player2ForfeitChar}{thisResult.RemainingTimeEngine2,7:F2} ");
          Def.Logger.Write($"{thisResult.TotalTimeEngine1,8:F2}{player1ForfeitChar}{thisResult.RemainingTimeEngine1,7:F2}  ");
          // Def.Logger.Write($"{thisResult.TotalTimeEngine2,8:F2}{player2ForfeitChar}{thisResult.RemainingTimeEngine2,7:F2}  {thisResult.TimeAggressivenessRatio(true),5:F2} ");
          // Def.Logger.Write($"{thisResult.TotalTimeEngine1,8:F2}{player1ForfeitChar}{thisResult.RemainingTimeEngine1,7:F2}  {thisResult.TimeAggressivenessRatio(false),5:F2}  ");
          Def.Logger.Write($"{thisResult.TotalNodesEngine2,16:N0} {thisResult.TotalNodesEngine1,16:N0}   ");
        }
        else
        {
          Def.Logger.Write($"{thisResult.TotalTimeEngine1,8:F2}{player1ForfeitChar}{thisResult.RemainingTimeEngine1,7:F2} ");
          Def.Logger.Write($"{thisResult.TotalTimeEngine2,8:F2}{player2ForfeitChar}{thisResult.RemainingTimeEngine2,7:F2}  ");
          // Def.Logger.Write($"{thisResult.TotalTimeEngine1,8:F2}{player1ForfeitChar}{thisResult.RemainingTimeEngine1,7:F2}  {thisResult.TimeAggressivenessRatio(true),5:F2} ");
          // Def.Logger.Write($"{thisResult.TotalTimeEngine2,8:F2}{player2ForfeitChar}{thisResult.RemainingTimeEngine2,7:F2}  {thisResult.TimeAggressivenessRatio(false),5:F2}  ");
          Def.Logger.Write($"{thisResult.TotalNodesEngine1,16:N0} {thisResult.TotalNodesEngine2,16:N0}   ");
        }


        if (Def.CheckPlayer2Def != null)
        {
          Def.Logger.Write($"{thisResult.PlyCount,4:F0}  {checkEnginePlyDifferent}  {TournamentUtils.ResultStr(thisResult.Result, !engine2White),4}  ");
        }
        else
        {
          Def.Logger.Write($"{thisResult.PlyCount,4:F0}  {TournamentUtils.ResultStr(thisResult.Result, !engine2White),4}  ");
        }

        Def.Logger.Write($" {resultReasonChar}   {endingCP,5:F0}{questionableFinalCP}");
        Def.Logger.Write($" {wdlStr}   {thisResult.FEN} ");
        Def.Logger.WriteLine();
      }

      UpdateAggregateGameStats(thisResult);

      openingsFinishedAtLeastOnce.Add(openingIndex);
      NumGames++;
    }

    private void UpdateAggregateGameStats(TournamentGameInfo thisResult)
    {
      TotalNodesEngine1 += thisResult.TotalNodesEngine1;
      TotalNodesEngine2 += thisResult.TotalNodesEngine2;
      TotalMovesEngine1 += thisResult.PlyCount;
      TotalMovesEngine2 += thisResult.PlyCount;
      TotalTimeEngine1 += thisResult.TotalTimeEngine1;
      TotalTimeEngine2 += thisResult.TotalTimeEngine2;
    }

    private static void WritePGNResult(bool engine2White, TournamentGameInfo thisResult, PGNWriter pgnWriter)
    {
      switch (thisResult.Result)
      {
        case TournamentGameResult.Win:
          if (engine2White)
          {
            pgnWriter.WriteResultBlackWins();
          }
          else
          {
            pgnWriter.WriteResultWhiteWins();
          }
          break;

        case TournamentGameResult.Loss:
          if (engine2White)
          {
            pgnWriter.WriteResultWhiteWins();
          }
          else
          {
            pgnWriter.WriteResultBlackWins();
          }
          break;

        default:
          pgnWriter.WriteResultDraw();
          break;
      }
    }


    private void OutputHeaders(string pgnFileName)
    {
      Def.Logger.WriteLine();
      Def.Logger.WriteLine($"Games will be incrementally written to file: {pgnFileName}");
      Def.Logger.WriteLine("Result codes: C=checkmate S=stalemate T=tablebase M=insufficient material E=excessive moves R=draw by repetition V=evaluation agreement F=time forfeit");
      Def.Logger.WriteLine();

      if (Def.CheckPlayer2Def != null)
      {
        Def.Logger.WriteLine("  Player1    Player2   ELO   +/-  LOS   GAME#     TIME    TH#   OP#      TIME1    REM1    TIME2    REM2        NODES 2       PLY  DIF    RES  R   ENDCP     W   D   L   FEN");
        Def.Logger.WriteLine(" ---------  ---------  ---   ---  ---   -----  --------   ---   ---     ------  ------   ------  ------   --------------   ----  ---    ---  -   -----     -   -   -   ---------------------------------------------------");
      }
      else
      {
        Def.Logger.WriteLine("  Player1    Player2   ELO   +/-  LOS   GAME#     TIME    TH#   OP#      TIME1    REM1    TIME2    REM2       NODES 1           NODES 2       PLY    RES  R   ENDCP     W   D   L   FEN");
        Def.Logger.WriteLine(" ---------  ---------  ---   ---  ---   -----  --------   ---   ---     ------  ------   ------  ------    --------------   --------------   ----    ---  -   -----     -   -   -   ---------------------------------------------------");
      }
      havePrintedHeaders = true;
    }


    static readonly object outputLockObj = new();

    /// <summary>
    /// 
    /// </summary>
    /// <param name="openingIndex"></param>
    /// <param name="engine1"></param>
    /// <param name="engine2"></param>
    /// <param name="engine2IsWhite"></param>
    /// <param name="searchLimit"></param>
    /// <returns>GameResult (from the perspective of engine2)</returns>
    TournamentGameInfo DoGameTest(TextWriter logger, PGNWriter pgnWriter, int gameSequenceNum, int openingIndex,
                                 GameEngine engine1, GameEngine engine2, GameEngine engineCheckAgainstEngine2,
                                 bool engine2IsWhite,
                                 SearchLimit searchLimitEngine1, SearchLimit searchLimitEngine2,
                                 bool useTablebasesForAdjudication, bool showMoves)
    {
      if (openingIndex >= openings.Count)
      {
        throw new Exception($"Reached end of openings of size {openings.Count}.");
      }

      PositionWithHistory curPositionAndMoves = openings.GetAtIndex(openingIndex);
      string startFEN = curPositionAndMoves.FinalPosition.FEN;
      bool engine2ToMove = engine2IsWhite == (curPositionAndMoves.FinalPosition.MiscInfo.SideToMove == SideType.White);

      if (showMoves)
      {
        logger.WriteLine();
        logger.WriteLine("NEW GAME from " + curPositionAndMoves);
        if (engine2ToMove)
        {
          logger.WriteLine((engine2IsWhite ? "White: " : "Black: ") + engine2.ID + " [" + searchLimitEngine2 + "]");
          logger.WriteLine((engine2IsWhite ? "Black: " : "White: ") + engine1.ID + " [" + searchLimitEngine1 + "]");
        }
        else
        {
          logger.WriteLine((engine2IsWhite ? "Black: " : "White: ") + engine1.ID + " [" + searchLimitEngine1 + "]");
          logger.WriteLine((engine2IsWhite ? "White: " : "Black: ") + engine2.ID + " [" + searchLimitEngine2 + "]");
        }
      }

      pgnWriter.WriteStartPosition(startFEN);


      GameMoveConsoleInfo info = new GameMoveConsoleInfo();

      int plyCount = 1;
      float timeEngine1Tot = 0;
      float timeEngine2Tot = 0;
      long visitsEngine1Tot = 0;
      long visitsEngine2Tot = 0;
      long nodesEngine1Tot = 0;
      long nodesEngine2Tot = 0;
      int movesEngine1 = 0;
      int movesEngine2 = 0;
      bool engine1ShouldHaveForfieted = false;
      bool engine2ShouldHaveForfieted = false;

      int numEngine2MovesDifferentFromCheckEngine = 0;

      List<float> scoresEngine1 = new List<float>();
      List<float> scoresEngine2 = new List<float>();

      List<GameMoveStat> gameMoveHistory = new List<GameMoveStat>();

      static float RemainingTime(SearchLimit limit, int numMoves, float timeUsed)
      {
        if (limit.Type == SearchLimitType.SecondsForAllMoves)
        {
          return limit.MaxValueAfterMoves(numMoves) - timeUsed;
        }
        else
        {
          // TODO: in principle we could do this also for nodes per game limits
          return 0;
        }
      }

      TournamentGameInfo MakeGameInfo(TournamentGameResult result, TournamentGameResultReason reason)
      {
        return new TournamentGameInfo()
        {
          Engine2IsWhite = engine2IsWhite,
          GameSequenceNum = gameSequenceNum,
          OpeningIndex = openingIndex,
          FEN = startFEN,
          Result = result, //TournamentGameInfo.InvertedResult(result),
          ResultReason = reason,
          PlyCount = plyCount,
          TotalTimeEngine1 = timeEngine1Tot,
          TotalTimeEngine2 = timeEngine2Tot,
          TotalNodesEngine1 = nodesEngine1Tot,
          TotalNodesEngine2 = nodesEngine2Tot,
          RemainingTimeEngine1 = RemainingTime(searchLimitEngine1, movesEngine1, timeEngine1Tot),
          RemainingTimeEngine2 = RemainingTime(searchLimitEngine2, movesEngine2, timeEngine2Tot),
          ShouldHaveForfeitedOnLimitsEngine1 = engine1ShouldHaveForfieted,
          ShouldHaveForfeitedOnLimitsEngine2 = engine2ShouldHaveForfieted,
          NumEngine2MovesDifferentFromCheckEngine = numEngine2MovesDifferentFromCheckEngine,
          GameMoveHistory = gameMoveHistory
        };

      }

      // Reset the game state before we begin making moves
      engine1.CumulativeSearchTimeSeconds = 0;
      engine1.CumulativeNodes = 0;
      engine2.CumulativeSearchTimeSeconds = 0;
      engine2.CumulativeNodes = 0;
      TournamentGameResult result = TournamentGameResult.None;

      SearchLimit searchLimitWithIncrementsEngine1 = searchLimitEngine1 with { };
      SearchLimit searchLimitWithIncrementsEngine2 = searchLimitEngine2 with { };


      // Keep a list of all positions seen so far in game      
      List<Position> encounteredPositions = new List<Position>();

      // Repeatedly make moves
      while (true)
      {
        // Add this position to the set of positions encountered
        Position currentPosition = curPositionAndMoves.FinalPosition;
        encounteredPositions.Add(currentPosition);

        // If time or node increment was specified for game-level search limit,
        // adjust the search limit to give credit for this increment
        if (!engine2ToMove && searchLimitEngine1.ValueIncrement != 0)
        {
          searchLimitWithIncrementsEngine1 = searchLimitWithIncrementsEngine1.WithIncrementApplied();
        }

        if (engine2ToMove && searchLimitEngine2.ValueIncrement != 0)
        {
          searchLimitWithIncrementsEngine2 = searchLimitWithIncrementsEngine2.WithIncrementApplied();
        }

        // Check for very bad evaluation for one side (agreed by both sides)
        if (scoresEngine1.Count > 5)
        {
          result = CheckResultAgreedBothEngines(scoresEngine1, scoresEngine2, result);
          if (result != TournamentGameResult.None)
          {
            return MakeGameInfo(result, TournamentGameResultReason.AdjudicatedEvaluation);
          }
        }

        info.FEN = currentPosition.FEN;
        info.MoveNum = plyCount / 2;

        // Check for draw by repetition of position (3 times)
        int countRepetitions = 0;
        foreach (Position pos in encounteredPositions)
        {
          if (pos.EqualAsRepetition(in currentPosition))
          {
            countRepetitions++;
          }
        }

        if (countRepetitions >= 3)
        {
          return MakeGameInfo(TournamentGameResult.Draw, TournamentGameResultReason.Repetition);
        }


        // Check for draw by insufficient material
        if (curPositionAndMoves.FinalPosition.CheckDrawBasedOnMaterial == Position.PositionDrawStatus.DrawByInsufficientMaterial)
        {
          return MakeGameInfo(TournamentGameResult.Draw, TournamentGameResultReason.AdjudicateMaterial);
        }
        else if (curPositionAndMoves.FinalPosition.CheckDrawCanBeClaimed == Position.PositionDrawStatus.DrawCanBeClaimed)
        {
          return MakeGameInfo(TournamentGameResult.Draw, TournamentGameResultReason.Repetition);
        }
        else if (plyCount >= 500)
        {
          return MakeGameInfo(TournamentGameResult.Draw, TournamentGameResultReason.ExcessiveMoves);
        }

        // Check for terminal result (tablebase or intrinsically terminal)
        TournamentGameResultReason reason;
        result = TournamentUtils.TryGetGameResultIfTerminal(curPositionAndMoves, !engine2IsWhite, useTablebasesForAdjudication, out reason);
        if (result != TournamentGameResult.None)
        {
          return MakeGameInfo(result, reason);
        }

        plyCount++;

        int numPieces = Position.FromFEN(info.FEN).PieceCount;

        // Make player's move
        GameMoveStat moveStat;
        if (engine2ToMove)
        {
          info = DoMove(engine2, engineCheckAgainstEngine2,
                        gameMoveHistory, searchLimitWithIncrementsEngine2, scoresEngine2,
                        ref nodesEngine2Tot, ref visitsEngine2Tot, ref timeEngine2Tot);
          movesEngine2++;
          
          if (engine2IsWhite)
          {
            engine2ShouldHaveForfieted |= info.WhiteShouldHaveForfeitedOnLimit;
            moveStat = new GameMoveStat(plyCount, SideType.White, info.WhiteScoreQ, info.WhiteScoreCentipawns, engine2.CumulativeSearchTimeSeconds, numPieces, info.WhiteMAvg, info.WhiteFinalN, info.WhiteNumNodesComputed, info.WhiteSearchLimitPre, info.WhiteMoveTimeUsed);
          }
          else
          {
            engine2ShouldHaveForfieted |= info.BlackShouldHaveForfeitedOnLimit;
            moveStat = new GameMoveStat(plyCount, SideType.Black, info.BlackScoreQ, info.BlackScoreCentipawns, engine2.CumulativeSearchTimeSeconds, numPieces, info.BlackMAvg, info.BlackFinalN, info.BlackNumNodesComputed, info.BlackSearchLimitPre, info.BlackMoveTimeUsed);
          }
        }
        else
        {
          info = DoMove(engine1, null,
                        gameMoveHistory, searchLimitWithIncrementsEngine1, scoresEngine1,
                        ref nodesEngine1Tot, ref visitsEngine1Tot, ref timeEngine1Tot);
          movesEngine1++;
          if (engine2IsWhite)
          {
            engine1ShouldHaveForfieted |= info.BlackShouldHaveForfeitedOnLimit;
            moveStat = new GameMoveStat(plyCount, SideType.Black, info.BlackScoreQ, info.BlackScoreCentipawns, engine1.CumulativeSearchTimeSeconds, numPieces, info.BlackMAvg, info.BlackFinalN, info.BlackNumNodesComputed, info.BlackSearchLimitPre, info.BlackMoveTimeUsed);
          }
          else
          {
            engine1ShouldHaveForfieted |= info.WhiteShouldHaveForfeitedOnLimit;
            moveStat = new GameMoveStat(plyCount, SideType.White, info.WhiteScoreQ, info.WhiteScoreCentipawns, engine1.CumulativeSearchTimeSeconds, numPieces, info.WhiteMAvg, info.WhiteFinalN, info.WhiteNumNodesComputed, info.WhiteSearchLimitPre, info.WhiteMoveTimeUsed);
          }
        }
                
        moveStat.Id = engine2ToMove ? engine2.ID : engine1.ID;
        gameMoveHistory.Add(moveStat);

        engine2ToMove = !engine2ToMove;
        if (plyCount % 2 == 1)
        {
          info = new GameMoveConsoleInfo();
          if (showMoves) logger.WriteLine();
        }
      }


      GameMoveConsoleInfo DoMove(GameEngine engine, GameEngine checkMoveEngine,
                                 List<GameMoveStat> gameMoveHistory,
                                 SearchLimit searchLimit, List<float> scoresCP,
                                  ref long totalNodesUsed, ref long totalVisitsUsed, ref float totalTimeUsed)
      {
        SearchLimit thisMoveSearchLimit = searchLimit.Type switch
        {
          SearchLimitType.SecondsPerMove => searchLimit,
          SearchLimitType.NodesPerMove => searchLimit,
          SearchLimitType.NodesPerTree => searchLimit,
          SearchLimitType.NodesForAllMoves => new SearchLimit(SearchLimitType.NodesForAllMoves,
                                                              Math.Max(0, searchLimit.Value - totalVisitsUsed),
                                                              searchLimit.SearchCanBeExpanded,
                                                              searchLimit.ValueIncrement,
                                                              searchLimit.MaxMovesToGo),
          SearchLimitType.SecondsForAllMoves => new SearchLimit(SearchLimitType.SecondsForAllMoves,
                                                                MathF.Max(0, searchLimit.Value - totalTimeUsed),
                                                                searchLimit.SearchCanBeExpanded,
                                                                searchLimit.ValueIncrement,
                                                                searchLimit.MaxMovesToGo),
          _ => throw new Exception($"Internal error, unknown SearchLimit.LimitType {searchLimit.Type}")
        };

        GameEngineSearchResult engineMove = engine.Search(curPositionAndMoves, thisMoveSearchLimit, gameMoveHistory);
        float engineTime = (float)engineMove.TimingStats.ElapsedTimeSecs;

        // Check for time forfeit
        bool shouldHaveForfeited = false;
        if (thisMoveSearchLimit.IsTimeLimit)
        {
          const float GRACE_FRACTION = 0.02f; // Allow up to 2% over
          float GRACE_SECONDS = searchLimit.Value * GRACE_FRACTION;
          float timeExcess = engineTime - thisMoveSearchLimit.Value;
          if (gameMoveHistory.Count >= 2 && timeExcess > GRACE_SECONDS)
          {
            shouldHaveForfeited = true;

            // TODO: (a) remove the "Count > 2" restriction above,
            //       (b) reconsider the GRACE_FRACTION above
            //       (b) log this
            //throw new Exception($"Time forfeit, allotted {thisMoveSearchLimit.Value} used {engineTime} for engine {engine}");
          }
        }

        GameEngineSearchResult checkSearch = default;
        string checkMove = "";
        if (checkMoveEngine != null)
        {
          checkSearch = checkMoveEngine.Search(curPositionAndMoves, searchLimit);
          checkMove = checkSearch.MoveString;
          if (engineMove.MoveString != checkMove)
          {
            numEngine2MovesDifferentFromCheckEngine++;
          }
        }

        // Verify the engine's move was legal by trying to make it.
        try
        {
          Position positionAfterMoveTest = curPositionAndMoves.FinalPosition.AfterMove(Move.FromUCI(engineMove.MoveString));
        }
        catch (Exception exc)
        {
          throw new Exception($"Engine {engine.ID} made illegal move {engineMove.MoveString} in position {curPositionAndMoves.FinalPosition.FEN}");
        }

        PositionWithHistory newPosition = new PositionWithHistory(curPositionAndMoves);
        newPosition.AppendMove(engineMove.MoveString);

        // Output position to PGN
        pgnWriter.WriteMove(newPosition.Moves[^1], curPositionAndMoves.FinalPosition, engineTime, engineMove.Depth, engineMove.ScoreCentipawns);

        scoresCP.Add(engineMove.ScoreCentipawns);
    
        totalNodesUsed += engineMove.FinalN;
        totalVisitsUsed += engineMove.Visits;
        totalTimeUsed += engineTime;
        if (curPositionAndMoves.FinalPosition.MiscInfo.SideToMove == SideType.White)
        {
          info.WhiteMoveStr = engineMove.MoveString;
          info.WhiteScoreQ = engineMove.ScoreQ;
          info.WhiteScoreCentipawns = engineMove.ScoreCentipawns;
          info.WhiteSearchLimitPre = thisMoveSearchLimit;
          info.WhiteSearchLimitPost = engineMove.Limit;
          info.WhiteMoveTimeUsed = engineTime;
          info.WhiteTimeAllMoves = totalTimeUsed;
          info.WhiteNodesAllMoves = totalNodesUsed;
          info.WhiteStartN = engineMove.StartingN;
          info.WhiteFinalN = engineMove.FinalN;
          info.WhiteMAvg = engineMove.MAvg;
          info.WhiteCheckMoveStr = checkMove;
          info.WhiteShouldHaveForfeitedOnLimit = shouldHaveForfeited;
        }
        else
        {
          info.BlackMoveStr = engineMove.MoveString;
          info.BlackScoreQ = engineMove.ScoreQ;
          info.BlackScoreCentipawns = engineMove.ScoreCentipawns;
          info.BlackSearchLimitPre = thisMoveSearchLimit;
          info.BlackSearchLimitPost = engineMove.Limit;
          info.BlackMoveTimeUsed = engineTime;
          info.BlackTimeAllMoves = totalTimeUsed;
          info.BlackNodesAllMoves = totalNodesUsed;
          info.BlackStartN = engineMove.StartingN;
          info.BlackFinalN = engineMove.FinalN;
          info.BlackMAvg = engineMove.MAvg;
          info.BlackCheckMoveStr = checkMove;
          info.BlackShouldHaveForfeitedOnLimit = shouldHaveForfeited;
        }

        if (showMoves) info.PutStr();

        // Advance to next move
        curPositionAndMoves = newPosition;

        return info;
      }
    }


    static float MinLastN(List<float> vals, int count)
    {
      float min = float.MaxValue;
      for (int i = 1; i <= count; i++)
        if (vals[^i] < min)
          min = vals[^i];
      return min;
    }

    static float MaxLastN(List<float> vals, int count)
    {
      float max = float.MinValue;
      for (int i = 1; i <= count; i++)
        if (vals[^i] > max)
          max = vals[^i];
      return max;
    }

    TournamentGameResult CheckResultAgreedBothEngines(List<float> scoresCPEngine1, List<float> scoresCPEngine2, TournamentGameResult result)
    {
      int WIN_THRESHOLD = Run.Def.AdjudicationThresholdCentipawns;
      int NUM_MOVES = Run.Def.AdjudicationThresholdNumMoves;

      // Return if insufficient moves in history to make determination.
      if (scoresCPEngine2.Count < NUM_MOVES || scoresCPEngine1.Count < NUM_MOVES)
        return result;

      if (MinLastN(scoresCPEngine2, NUM_MOVES) > WIN_THRESHOLD && MaxLastN(scoresCPEngine1, NUM_MOVES) < -WIN_THRESHOLD)
      {
        return TournamentGameResult.Loss;
      }
      else if (MinLastN(scoresCPEngine1, NUM_MOVES) > WIN_THRESHOLD && MaxLastN(scoresCPEngine2, NUM_MOVES) < -WIN_THRESHOLD)
      {
        return TournamentGameResult.Win;
      }
      else
      {
        return result;
      }
    }

  }
}
