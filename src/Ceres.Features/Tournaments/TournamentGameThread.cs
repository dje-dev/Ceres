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
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCTS.GameEngines;
using Ceres.Base.Misc;
using Ceres.Features.Tournaments.Streaming;

#endregion

namespace Ceres.Features.Tournaments
{
  /// <summary>
  /// Manage execution of a single thread of a tournament,
  /// running the main move loop which alternates between players.
  /// </summary>
  internal class TournamentGameThread
  {
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
      for (int i = 0; i < def.Engines.Length; i++)
      {
        def.Engines[i].EngineDef.ProcessorGroupID = def.ProcessGroupIndex;
      }

      Run = new TournamentGameRunner(Def);

      // Register with the (per-tournament) shared stats object so that all sibling threads
      // can be polled for their pair-completion state when deciding whether to emit "*".
      lock (parentTestResults.GameThreads) parentTestResults.GameThreads.Add(this);
    }


    /// <summary>
    /// True if this thread's most recently output game was the second (completing) game of a
    /// pair. Written and read only while holding outputLockObj, which serializes all game-result
    /// output across threads. Used to flag the moments when every thread is simultaneously at a
    /// pair boundary (see OutputGameResultInfo).
    /// </summary>
    bool lastOutputCompletedPair = false;


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

    /// <summary>
    /// Stable 0-based index identifying this game thread within the tournament (used to tag
    /// live streaming events so subscribers can watch a specific concurrent game thread).
    /// </summary>
    public int ThreadIndex { get; private set; }

    public void RunGameTests(int runnerIndex, Func<int, int> getGamePairToProcess,
                             Action<TournamentGameInfo, TournamentGameInfo> doneGamePairCallback = null)
    {
      ThreadIndex = runnerIndex;

      // Verbose move-stat gathering (needed for per-move WDL and top-move/TopQ data) carries some
      // overhead, so only enable it once a client is actually watching: register a callback that the
      // streaming publisher invokes on the first client connection (and never, if no one connects).
      // When streaming is disabled the Observer is null and this is a no-op (zero overhead).
      Def.parentDef.Observer?.RegisterOnFirstClient(() =>
      {
        foreach (GameEngine engine in Run.Engines)
        {
          if (engine is GameEngineCeresInProcess ceresEngine)
          {
            ceresEngine.GatherVerboseMoveStats = true;
          }
          else if (engine is Ceres.MCGS.GameEngines.GameEngineCeresMCGSInProcess mcgsEngine)
          {
            mcgsEngine.GatherVerboseMoveStats = true;
          }
        }
      });

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

      LoadOpenings();

      Random rand = new Random();

      // Def.Logger.WriteLine($"Begin {def.NumGames} game test with limit {def.SearchLimitEngine1} {def.SearchLimitEngine2} ID { gameSequenceNum } ");
      int numPairsProcessed = 0;

      // Register as an active participant in cooperative pause coordination (Ctrl-P). The matching
      // DeregisterActive in the finally below removes this thread from the "all threads parked"
      // accounting whether it exits the game loop normally or via an exception. No-op (null
      // controller) for non-interactive or distributed tournaments.
      Def.parentDef.PauseController?.RegisterActive();
      try
      {
      while (!Def.parentDef.ShouldShutDown)
      {
        int openingIndex = getGamePairToProcess(openings.Count);

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

        int numEngines = Run.Engines.Length;
        int pairsPerOpening = string.IsNullOrEmpty(Def.ReferenceEngineId)
          ? (numEngines * (numEngines - 1)) / 2
          : numEngines - 1;
        int baseGameSequenceNum = openingIndex * pairsPerOpening * 2;
        int roundNumber = 1 + openingIndex;

        if (string.IsNullOrEmpty(Def.ReferenceEngineId))
        {
          List<GameEngine> list = new List<GameEngine>(Run.Engines);
          int engine1Index = 0;
          int pairCount = 0;
          while (list.Count > 1)
          {
            for (int i = 1; i < list.Count; i++)
            {
              Run.SetEnginePair(engine1Index, i + engine1Index);

              bool engine2White = (numPairsProcessed + runnerIndex + pairCount) % 2 == 0;
              int pairSeqNum = baseGameSequenceNum + pairCount * 2;
              TournamentGameInfo gameInfo = RunGame(pgnFileName, engine2White, openingIndex, pairSeqNum, roundNumber);
              TournamentGameInfo gameReverseInfo = RunGame(pgnFileName, !engine2White, openingIndex, pairSeqNum + 1, roundNumber);
              doneGamePairCallback?.Invoke(gameInfo, gameReverseInfo);
              pairCount++;
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
          {
            throw new Exception("Error in loading reference engine");
          }

          int index = Array.IndexOf(Run.Engines, refEngine);
          int pairCount = 0;
          for (int i = 0; i < Run.Engines.Length; i++)
          {
            if (index == i)
            {
              continue;
            }
            Run.SetEnginePair(index, i);
            bool engine2White = (numPairsProcessed + runnerIndex + pairCount) % 2 == 0;
            int pairSeqNum = baseGameSequenceNum + pairCount * 2;
            TournamentGameInfo gameInfo = RunGame(pgnFileName, engine2White, openingIndex, pairSeqNum, roundNumber);
            TournamentGameInfo gameReverseInfo = RunGame(pgnFileName, !engine2White, openingIndex, pairSeqNum + 1, roundNumber);
            doneGamePairCallback?.Invoke(gameInfo, gameReverseInfo);
            pairCount++;
          }
          numPairsProcessed++;
        }
      }
      }
      finally
      {
        Def.parentDef.PauseController?.DeregisterActive();
      }

      // Dispose of all engines.
      foreach (GameEngine engine in Run.Engines)
      {
        engine.Dispose();
        Run.Engine2CheckEngine?.Dispose();
      }
    }


    /// <summary>
    /// Populates the openings collection with the set of positions to be used in the tournament.
    /// </summary>
    private void LoadOpenings()
    {
      if (Def.OpeningsFileName != null)
      {
        openings = PositionsWithHistory.FromEPDOrPGNFile(Def.OpeningsFileName, int.MaxValue, Def.AcceptGamePredicate, Def.AcceptPosPredicate);
      }
      else if (Def.StartingFEN != null)
      {
        openings = PositionsWithHistory.FromFEN(Def.StartingFEN, Def.NumGamePairs ?? 1);
      }
      else
      {
        openings = PositionsWithHistory.FromFEN(Position.StartPosition.FEN, Def.NumGamePairs ?? 1);
      }

      if (Def.AcceptPosExcludeIfContainsPieceTypeList != null)
      {
        openings = PositionsWithHistory.FromMoveSequences(
          openings.Where(pos => !Def.AcceptPosExcludeIfContainsPieceTypeList
                                       .Exists(piece => pos.FinalPosition.PieceExists(new Piece(SideType.White, piece))
                                                     || pos.FinalPosition.PieceExists(new Piece(SideType.Black, piece)))).ToArray());
      }

      if (Def.AcceptPosPredicate != null)
      {
        openings = PositionsWithHistory.FromMoveSequences(openings.Where(s => Def.AcceptPosPredicate(s.FinalPosition)).ToArray());
      }

      // Finally, remove any possible duplicates.
      openings = PositionsWithHistory.FromMoveSequences(openings.ToArray(), true);
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


    Dictionary<int, TournamentGameInfo> GameInfoFirstFinishedForByOpening = new();

    /// <summary>
    /// Returns true if the given position has Chess960 characteristics
    /// (non-standard rook placement or non-standard king file with castling rights).
    /// </summary>
    static bool PositionIsChess960(in Position pos)
    {
      if (!pos.MiscInfo.CastlingRightsAny)
      {
        return false;
      }

      // Non-standard rook placement detected by FENParser.
      if (pos.MiscInfo.RookInfo.RawValue != 0)
      {
        return true;
      }

      // Standard rook placement but check if king is not on standard e-file.
      // E-file = file index 4. Check both white and black kings.
      if (pos.MiscInfo.WhiteCanOO || pos.MiscInfo.WhiteCanOOO)
      {
        Piece whiteKingPiece = pos.PieceOnSquare(new Square("e1"));
        if (whiteKingPiece.Type != PieceType.King || whiteKingPiece.Side != SideType.White)
        {
          return true;
        }
      }

      if (pos.MiscInfo.BlackCanOO || pos.MiscInfo.BlackCanOOO)
      {
        Piece blackKingPiece = pos.PieceOnSquare(new Square("e8"));
        if (blackKingPiece.Type != PieceType.King || blackKingPiece.Side != SideType.Black)
        {
          return true;
        }
      }

      return false;
    }


    bool? lastChess960Mode = null;

    /// <summary>
    /// Sets Chess960 mode on all engines if it has changed since last game.
    /// For UCI engines, sends the UCI_Chess960 option.
    /// For Ceres in-process engines, sets the IsChess960 property.
    /// </summary>
    void SetEnginesChess960Mode(bool isChess960)
    {
      if (lastChess960Mode == isChess960)
      {
        return;
      }

      lastChess960Mode = isChess960;
      MGPositionConstants.IsChess960 = isChess960;

      string valueStr = isChess960 ? "true" : "false";
      foreach (GameEngine engine in Run.Engines)
      {
        if (engine is GameEngineUCI uciEngine)
        {
          uciEngine.IsChess960 = isChess960;
          uciEngine.UCIRunner.SendCommand($"setoption name UCI_Chess960 value {valueStr}");
        }
        else if (engine is GameEngineCeresInProcess ceresEngine)
        {
          ceresEngine.IsChess960 = isChess960;
        }
      }
    }


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

      // Auto-detect Chess960 from current opening's initial position.
      // Uses RookPlacementInfo populated by FENParser (RawValue == 0 for standard rook placement)
      // and king square check for the case where rooks are on standard files but king is not on e-file.
      PositionWithHistory opening = openings.GetAtIndex(openingIndex);
      bool gameIsChess960 = PositionIsChess960(opening.InitialPosMG.ToPosition);
      SetEnginesChess960Mode(gameIsChess960);
      pgnWriter.IsChess960 = gameIsChess960;

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
      thisResult.SearchLimitWhite = engine2White ? Def.Player2Def.SearchLimit : Def.Player1Def.SearchLimit;
      thisResult.SearchLimitBlack = engine2White ? Def.Player1Def.SearchLimit : Def.Player2Def.SearchLimit;

      UpdateStatsAndOutputSummaryFromGameResult(pgnFileName, engine2White, openingIndex, gameSequenceNum, thisResult);

      // Cooperative pause checkpoint (Ctrl-P): park here after each completed game if a pause is
      // in effect, so all worker threads quiesce at an end-of-game boundary. No-op when not paused.
      Def.parentDef.PauseController?.WaitIfPaused();

      return thisResult;
    }

    internal void UpdateStatsAndOutputSummaryFromGameResult(string pgnFileName, bool engine2White, int openingIndex, int gameSequenceNum, TournamentGameInfo thisResult)
    {
      string engine1ID = engine2White ? thisResult.PlayerBlack : thisResult.PlayerWhite;
      string engine2ID = engine2White ? thisResult.PlayerWhite : thisResult.PlayerBlack;

      bool shutDownRequested = false;

      // Pentanomial (paired-game) result for the running display, computed under the statistics
      // lock since the underlying accumulation arrays are mutated by other threads under that lock.
      // Perspective matches OutputGameResultInfo's 'player' (the engine playing White this game).
      PentanomialResult penta;
      lock (ParentStats)
      {
        ParentStats.UpdateTournamentStats(thisResult, engine1ID, engine2ID);

        // Record this game in the overall list (performed under the lock since
        // multiple threads may be recording game results concurrently).
        ParentStats.GameInfos.Add(thisResult);

        penta = ParentStats.PentanomialFor(thisResult.PlayerWhite, thisResult.PlayerBlack);

        // Notify any registered per-game callback that a game has been processed,
        // passing the results accumulated so far. The callback runs while holding the
        // statistics lock so the stats it observes remain stable for the duration of the call.
        // A return value of true requests that the tournament begin an orderly shutdown.
        Func<TournamentResultStats, bool> perGameCallback = Def.parentDef.PerGameCallback;
        if (perGameCallback != null)
        {
          shutDownRequested = perGameCallback(ParentStats);
        }
      }

      // Notify any live-streaming observer that this game has finished (drives the per-thread
      // end frame and the tournament-global result used for standings/crosstable).
      Def.parentDef.Observer?.OnGameEnd(ThreadIndex, DtoMappers.ToGameEnd(thisResult));

      if (shutDownRequested)
      {
        Def.parentDef.ShouldShutDown = true;
      }

      // Only show headers first time for first thread
      if (!havePrintedHeaders)
      {
        OutputHeaders(pgnFileName);
      }

      OutputGameResultInfo(engine2White, openingIndex, gameSequenceNum, engine1ID, engine2ID, thisResult, penta);
    }


    /// <summary>
    /// Returns true if the three most recent moves indicate the opponent played a candidate blunder:
    /// the moving engine's evaluation improved by more than cpThreshold since its prior move, the
    /// position was not already decided before that move (the moving engine's prior evaluation was
    /// within +/- maxPriorAbsCp of equality), and the node counts (N) of the moving engine's current
    /// and prior moves and the intervening opponent move are all at least thresholdN.
    /// Note: this is only the immediate candidate test; a candidate is not dumped until the blundering
    /// engine confirms the deterioration on its own following move (see CheckForBlunderDump).
    /// </summary>
    /// <param name="cur">Moving engine's current move.</param>
    /// <param name="opp">Opponent's intervening move.</param>
    /// <param name="prev">Moving engine's prior move.</param>
    /// <param name="thresholdN">Minimum N required for all three moves.</param>
    /// <param name="cpThreshold">Minimum centipawn improvement required to trigger.</param>
    /// <param name="maxPriorAbsCp">Maximum absolute value (cp) of the moving engine's prior evaluation;
    /// if the position was already more decisive than this it is considered already won/lost and ignored.</param>
    /// <param name="improvement">Centipawn improvement since the moving engine's prior move.</param>
    internal static bool IsBlunderCondition(GameMoveStat cur, GameMoveStat opp, GameMoveStat prev,
                                            int thresholdN, float cpThreshold, float maxPriorAbsCp, out float improvement)
    {
      // cur and prev are both by the moving engine (same color), so their evaluations
      // (each from the moving engine's perspective) are directly comparable.
      improvement = cur.ScoreCentipawns - prev.ScoreCentipawns;
      return improvement > cpThreshold
          && Math.Abs(prev.ScoreCentipawns) <= maxPriorAbsCp   // ignore positions already won/lost ("piling on")
          && cur.FinalN >= thresholdN
          && prev.FinalN >= thresholdN
          && opp.FinalN >= thresholdN;
    }

    /// <summary>
    /// A detected-but-unconfirmed blunder candidate. The diagnostic dump is captured at detection
    /// time (while the blundering engine's most recent completed search is still the suspect move)
    /// and only written to disk once the blundering engine confirms the deterioration on its own
    /// next move. One instance is carried across plies within a single game by the caller.
    /// </summary>
    private sealed class BlunderCandidate
    {
      public GameEngine BlundererEngine;    // engine that played the suspect move
      public int BlunderPlyNum;             // ply number of the suspect move
      public float BlunderMoveScoreCp;      // blunderer's own evaluation (cp) of the suspect move
      public float ReferenceImprovementCp;  // reference engine's evaluation swing across the suspect move
      public string Header;                 // pre-built (self-locating) header text
      public string DumpBody;               // captured search-graph diagnostics
    }

    /// <summary>
    /// Detects a likely opponent blunder (a large favorable evaluation swing for the moving/reference
    /// engine across the opponent's last move, from a position that was not already decided) and, once
    /// the blundering engine confirms the deterioration on its OWN following move, dumps that engine's
    /// search graph diagnostics to a "blunder_info_*.txt" file in the current working directory.
    /// Called once per ply; <paramref name="pendingBlunder"/> carries a detected-but-unconfirmed
    /// candidate from one ply to the next (it is reset to null per game by the caller).
    /// Only in-process Ceres MCGS engines can be dumped; other opponents are skipped.
    /// </summary>
    private void CheckForBlunderDump(int gameSequenceNum, int roundNumber, List<GameMoveStat> gameMoveHistory,
                                     GameEngine movingEngine, GameEngine opponentEngine,
                                     string opponentBlunderMoveStr, ref BlunderCandidate pendingBlunder)
    {
      // (1) Resolve any pending candidate. The blunderer plays again exactly one ply after detection;
      //     write the dump only if the blunderer's OWN evaluation has now fallen by at least the
      //     threshold (i.e. it agrees, on reflection, that the move was a blunder).
      if (pendingBlunder != null && ReferenceEquals(movingEngine, pendingBlunder.BlundererEngine))
      {
        GameMoveStat confirmMove = gameMoveHistory[^1];   // blunderer's next move (same color as the suspect move)
        float blundererDrop = pendingBlunder.BlunderMoveScoreCp - confirmMove.ScoreCentipawns;
        if (blundererDrop > Def.BlunderDumpThresholdCentipawns)
        {
          string confirmLine =
              $"Blunderer confirmation: {pendingBlunder.BlundererEngine.ID} own evaluation fell {blundererDrop:F0}cp on its next "
            + $"move ({pendingBlunder.BlunderMoveScoreCp:F0}cp -> {confirmMove.ScoreCentipawns:F0}cp), confirming the blunder.";

          string fileName = $"blunder_info_g{gameSequenceNum + 1}_ply{pendingBlunder.BlunderPlyNum}_{SanitizeForFileName(pendingBlunder.BlundererEngine.ID)}.txt";
          string fullPath = Path.Combine(Directory.GetCurrentDirectory(), fileName);
          File.WriteAllText(fullPath, pendingBlunder.Header + Environment.NewLine
                                    + confirmLine + Environment.NewLine + Environment.NewLine
                                    + pendingBlunder.DumpBody);

          ConsoleUtils.WriteLineColored(ConsoleColor.Red,
              $"BLUNDER: engine {pendingBlunder.BlundererEngine.ID} position worse by {pendingBlunder.ReferenceImprovementCp:F0}cp "
            + $"(self-confirmed {blundererDrop:F0}cp), dumped to {fullPath}");
        }
        pendingBlunder = null;   // candidate resolved (whether confirmed or rejected)
      }

      // (2) Detect a new candidate from the three most recent plies.
      if (gameMoveHistory.Count < 3)
      {
        return;
      }

      GameMoveStat cur = gameMoveHistory[^1];   // moving engine's current move
      GameMoveStat opp = gameMoveHistory[^2];   // opponent's intervening (suspect) move
      GameMoveStat prev = gameMoveHistory[^3];  // moving engine's prior move

      if (!IsBlunderCondition(cur, opp, prev, Def.BlunderDumpThresholdN, Def.BlunderDumpThresholdCentipawns,
                              Def.BlunderDumpMaxPriorAbsCentipawns, out float improvement))
      {
        return;
      }

      // Buffer the dump now, while the opponent's most recent completed search is still the suspect move.
      StringWriter dump = new StringWriter();
      if (!opponentEngine.TryDumpLastSearchDiagnostics(dump, "UCI"))
      {
        return;
      }

      // Build a self-locating header from the AUTHORITATIVE game coordinates (the tournament board),
      // not the engine's internal search-graph counters (which can disagree under transposition / tree
      // reuse - e.g. a move counter several plies off). 'disagreement' is how much more optimistic the
      // blunderer is about its own position than the reference engine; a large value is a strong
      // "engine did not see the loss" bug signal.
      float refViewOfBlunderer = -cur.ScoreCentipawns;                 // reference eval, in the blunderer's perspective
      float disagreement = opp.ScoreCentipawns - refViewOfBlunderer;   // blunderer optimism over reference

      string header =
          $"Engine {movingEngine.ID} detected {improvement:F0}cp improvement since its previous move, "
        + $"diagnostic dump of opponent engine {opponentEngine.ID} follows (move actually played was {opponentBlunderMoveStr})." + Environment.NewLine
        + $"  Game        : {gameSequenceNum + 1} (Round {roundNumber}; {movingEngine.ID} vs {opponentEngine.ID})" + Environment.NewLine
        + $"  Blunderer   : {opponentEngine.ID} ({opp.Side}), played {opponentBlunderMoveStr} at ply {opp.PlyNum}" + Environment.NewLine
        + $"  FEN (before): {opp.Position.FEN}" + Environment.NewLine
        + $"  Reference {movingEngine.ID} eval: {prev.ScoreCentipawns:F0}cp (Q {prev.ScoreQ:F3}) before -> "
        + $"{cur.ScoreCentipawns:F0}cp (Q {cur.ScoreQ:F3}) after   [swing {improvement:F0}cp]" + Environment.NewLine
        + $"  Blunderer self-eval of the move : {opp.ScoreCentipawns:F0}cp (Q {opp.ScoreQ:F3})" + Environment.NewLine
        + $"  Disagreement (blunderer optimism vs reference): {disagreement:F0}cp   (large => likely engine vision bug)" + Environment.NewLine
        + $"  Nodes (N): reference prev {prev.FinalN:N0}, reference cur {cur.FinalN:N0}, blunder move {opp.FinalN:N0}";

      pendingBlunder = new BlunderCandidate
      {
        BlundererEngine = opponentEngine,
        BlunderPlyNum = opp.PlyNum,
        BlunderMoveScoreCp = opp.ScoreCentipawns,
        ReferenceImprovementCp = improvement,
        Header = header,
        DumpBody = dump.ToString()
      };
    }

    /// <summary>
    /// Replaces any character that is not a letter, digit, '-' or '_' with '_' so an engine ID
    /// can be embedded safely in a dump file name.
    /// </summary>
    private static string SanitizeForFileName(string s)
    {
      if (string.IsNullOrEmpty(s))
      {
        return "engine";
      }

      char[] cs = s.ToCharArray();
      for (int i = 0; i < cs.Length; i++)
      {
        if (!char.IsLetterOrDigit(cs[i]) && cs[i] != '-' && cs[i] != '_')
        {
          cs[i] = '_';
        }
      }
      return new string(cs);
    }


    private void OutputGameResultInfo(bool engine2White, int openingIndex, int gameSequenceNum,
                                      string engineID, string opponentID,
                                      TournamentGameInfo thisResult, PentanomialResult penta)
    {
      string engine1ID = engine2White ? thisResult.PlayerBlack : thisResult.PlayerWhite;
      string engine2ID = engine2White ? thisResult.PlayerWhite : thisResult.PlayerBlack;

      // Note: thisResult was already added to ParentStats.GameInfos (under the statistics lock)
      // in UpdateStatsAndOutputSummaryFromGameResult before this method was called.

      PlayerStat player = engine2White ?
        ParentStats.GetPlayer(opponentID, engineID) :
        ParentStats.GetPlayer(engineID, opponentID);

      float gNumber = NumGames + 1;
      // Elo point estimate from the trinomial mean (identical to the pentanomial mean);
      // the +/- error bar and LOS shown below use pentanomial (paired-game) analysis.
      (_, float eloAvg, _) = EloCalculator.EloConfidenceInterval(player.PlayerWins, player.Draws, player.PlayerLosses);

      // Pentanomial +/- and LOS are undefined until the first pair completes (they advance
      // on the second game of each pair); show a placeholder until then.
      string pentaErrStr = penta.NumPairs == 0 ? "----" : penta.EloErrorMargin.ToString("0");
      string pentaLOSStr = penta.NumPairs == 0 ? "----" : (100.0f * penta.LOS).ToString("0");

      string wdlStr = $"{player.PlayerWins,3} {player.Draws,3} {player.PlayerLosses,3}";

      // Show either = or ! (same game or differing moves) after the opening index
      // if this was the second of the pair of games played.
      string openingPlayedBothWaysStr = " ";
      bool wasSecondOfPair = gameSequenceNum % 2 == 1
                          && GameInfoFirstFinishedForByOpening.ContainsKey(openingIndex);
      if (wasSecondOfPair)
      {
        TournamentGameInfo firstGameInfo = GameInfoFirstFinishedForByOpening.GetValueOrDefault(openingIndex);
        bool wasSameGame = thisResult.HasSameMovesAs(firstGameInfo);
        openingPlayedBothWaysStr = wasSameGame ? "=" : "!";
      }
      else
      {
        GameInfoFirstFinishedForByOpening[openingIndex] = thisResult;
      }

      string player1ForfeitChar = thisResult.ShouldHaveForfeitedOnLimitsEngine1 ? "f" : " ";
      string player2ForfeitChar = thisResult.ShouldHaveForfeitedOnLimitsEngine2 ? "f" : " ";

      const string TournamentGameResultReasonCodes = "CSTMERAL";

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
      // are very inconsistent with the actual game outcome (as a diagnostic).
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
        // Update this thread's pair-completion state, then (if this game completed a pair) check
        // whether every participating thread's most recently output game also completed a pair.
        // If so, all threads are simultaneously at a pair boundary -- a point at which the running
        // Elo reflects only fully completed pairs -- so flag it with "*" in the dedicated column
        // immediately to the right of the ELO column (the "="/"!" OP# marker is left untouched).
        // All reads/writes of lastOutputCompletedPair are serialized by outputLockObj (held here).
        lastOutputCompletedPair = wasSecondOfPair;
        string allThreadsPairBoundaryStr = " ";
        if (wasSecondOfPair)
        {
          bool allThreadsAtPairBoundary;
          lock (ParentStats.GameThreads)
          {
            allThreadsAtPairBoundary = ParentStats.GameThreads.TrueForAll(t => t.lastOutputCompletedPair);
          }
          if (allThreadsAtPairBoundary)
          {
            allThreadsPairBoundaryStr = "*";
          }
        }

        string checkEnginePlyDifferent = thisResult.NumEngine2MovesDifferentFromCheckEngine == 0 ? "   " : $"{thisResult.NumEngine2MovesDifferentFromCheckEngine,3:N0}";
        if (Def.ShowGameMoves) Def.Logger.WriteLine();
        if (engine2White)
        {
          Def.Logger.Write($" {TrimmedIfNeeded(engine2ID, 10),-10} {TrimmedIfNeeded(engine1ID, 10),-10}");
        }
        else
        {
          Def.Logger.Write($" {TrimmedIfNeeded(engine1ID, 10),-10} {TrimmedIfNeeded(engine2ID, 10),-10}");
        }
        Def.Logger.Write($"{eloAvg,4:0} {allThreadsPairBoundaryStr,1} {pentaErrStr,4} {pentaLOSStr,5}  ");
        Def.Logger.Write($"{gNumber,5} {DateTime.Now.ToString().Split(" ")[1],10}  {gameSequenceNum,7:F0}  {openingIndex,7:F0} {openingPlayedBothWaysStr} ");

        // TODO: these averages are inexact, one player may have 1 ply more than thisResult.PlyCount/2
        int avgNodesPerMovePlayer1 = (int)MathF.Round(thisResult.TotalNodesEngine1 / (thisResult.PlyCount / 2), 0);
        int avgNodesPerMovePlayer2 = (int)MathF.Round(thisResult.TotalNodesEngine2 / (thisResult.PlyCount / 2), 0);

        // Evaluations per second for each player this game (blank when the engine did not report EPS).
        string epsPlayer1Str = (thisResult.TotalEvaluationsEngine1 > 0 && thisResult.TotalTimeEngine1 > 0)
                             ? MathF.Round(thisResult.TotalEvaluationsEngine1 / thisResult.TotalTimeEngine1).ToString("N0") : "";
        string epsPlayer2Str = (thisResult.TotalEvaluationsEngine2 > 0 && thisResult.TotalTimeEngine2 > 0)
                             ? MathF.Round(thisResult.TotalEvaluationsEngine2 / thisResult.TotalTimeEngine2).ToString("N0") : "";
        if (engine2White)
        {
          Def.Logger.Write($"{thisResult.TotalTimeEngine2,8:F2}{player2ForfeitChar}{thisResult.RemainingTimeEngine2,7:F2} ");
          Def.Logger.Write($"{thisResult.TotalTimeEngine1,8:F2}{player1ForfeitChar}{thisResult.RemainingTimeEngine1,7:F2}  ");
          // Def.Logger.Write($"{thisResult.TotalTimeEngine2,8:F2}{player2ForfeitChar}{thisResult.RemainingTimeEngine2,7:F2}  {thisResult.TimeAggressivenessRatio(true),5:F2} ");
          // Def.Logger.Write($"{thisResult.TotalTimeEngine1,8:F2}{player1ForfeitChar}{thisResult.RemainingTimeEngine1,7:F2}  {thisResult.TimeAggressivenessRatio(false),5:F2}  ");
          Def.Logger.Write($"{avgNodesPerMovePlayer2,12:N0} {avgNodesPerMovePlayer1,12:N0}  {epsPlayer2Str,7} {epsPlayer1Str,7}   ");
        }
        else
        {
          Def.Logger.Write($"{thisResult.TotalTimeEngine1,8:F2}{player1ForfeitChar}{thisResult.RemainingTimeEngine1,7:F2} ");
          Def.Logger.Write($"{thisResult.TotalTimeEngine2,8:F2}{player2ForfeitChar}{thisResult.RemainingTimeEngine2,7:F2}  ");
          // Def.Logger.Write($"{thisResult.TotalTimeEngine1,8:F2}{player1ForfeitChar}{thisResult.RemainingTimeEngine1,7:F2}  {thisResult.TimeAggressivenessRatio(true),5:F2} ");
          // Def.Logger.Write($"{thisResult.TotalTimeEngine2,8:F2}{player2ForfeitChar}{thisResult.RemainingTimeEngine2,7:F2}  {thisResult.TimeAggressivenessRatio(false),5:F2}  ");

          Def.Logger.Write($"{avgNodesPerMovePlayer1,12:N0} {avgNodesPerMovePlayer2,12:N0}  {epsPlayer1Str,7} {epsPlayer2Str,7}   ");
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
      Def.Logger.WriteLine("Result codes: C=checkmate S=stalemate T=tablebase M=insufficient material E=excessive moves R=draw by repetition A=adjudicate eval agreement F=time forfeit");
      Def.Logger.WriteLine("Note: the +/- and LOS columns use pentanomial (paired-game) analysis (shown once each game pair completes).");
      Def.Logger.WriteLine("Pair marker (after OP#): = pair completed with identical moves, ! pair completed with differing moves.");
      Def.Logger.WriteLine("The * column (immediately right of ELO) marks games output at the instant every thread is simultaneously at a completed-pair boundary (a fair Elo evaluation point).");
      Def.Logger.WriteLine();

      // Build the header (and dashed underline) so each label sits over its data column
      // (the field widths/separators here mirror exactly those emitted in OutputGameResultInfo).
      // The prefix through the EPS2 column is identical for both variants; only the PLY/DIF/RES
      // portion of the tail differs (the check variant adds the DIF column).
      static string Dsh(int n) => new string('-', n);

      string headerPrefix =
          $" {"Player1",-10} {"Player2",-10}" +
          $"{"ELO",4} {"*",1} {"+/-",4} {"LOS",5}  " +
          $"{"GAME#",5} {"TIME",10}  {"TH#",7}  {"OP#",7} {"",1} " +
          $"{"TIME1",8} {"REM1",7} " +
          $"{"TIME2",8} {"REM2",7}  " +
          $"{"AVG NODE1",12} {"AVG NODE2",12}  {"EPS1",7} {"EPS2",7}   ";
      string headerTail =
          $" {"R",1}   {"ENDCP",5}  " +
          $" {"W",3} {"D",3} {"L",3}   FEN";

      string dashPrefix =
          $" {Dsh(10),-10} {Dsh(10),-10}" +
          $"{Dsh(4),4} {Dsh(1),1} {Dsh(4),4} {Dsh(5),5}  " +
          $"{Dsh(5),5} {Dsh(10),10}  {Dsh(7),7}  {Dsh(7),7} {Dsh(1),1} " +
          $"{Dsh(8),8} {Dsh(7),7} " +
          $"{Dsh(8),8} {Dsh(7),7}  " +
          $"{Dsh(12),12} {Dsh(12),12}  {Dsh(7),7} {Dsh(7),7}   ";
      string dashTail =
          $" {Dsh(1),1}   {Dsh(5),5}  " +
          $" {Dsh(3),3} {Dsh(3),3} {Dsh(3),3}   {Dsh(51)}";

      if (Def.CheckPlayer2Def != null)
      {
        Def.Logger.WriteLine(headerPrefix + $"{"PLY",4}  {"DIF",3}  {"RES",4}  " + headerTail);
        Def.Logger.WriteLine(dashPrefix + $"{Dsh(4),4}  {Dsh(3),3}  {Dsh(4),4}  " + dashTail);
      }
      else
      {
        Def.Logger.WriteLine(headerPrefix + $"{"PLY",4}  {"RES",4}  " + headerTail);
        Def.Logger.WriteLine(dashPrefix + $"{Dsh(4),4}  {Dsh(4),4}  " + dashTail);
      }
      havePrintedHeaders = true;
    }


    static readonly object outputLockObj = new();

    Dictionary<Position, GameMoveConsoleInfo> referenceEngineMoveHistory = new();


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

      pgnWriter.WriteStartPosition(curPositionAndMoves.InitialPosition.FEN);
      foreach (PositionWithMove move in curPositionAndMoves.PositionsWithMoves)
      {
        pgnWriter.WriteMove(MGMoveConverter.MGMoveFromPosAndMove(in move.Position, move.Move), in move.Position);
      }

      // Notify any live-streaming observer that a new game has started on this thread.
      Def.parentDef.Observer?.OnGameStart(ThreadIndex,
          DtoMappers.ToGameStart(gameSequenceNum, openingIndex, engine1, engine2, engine2IsWhite,
                                 searchLimitEngine1, searchLimitEngine2, startFEN, curPositionAndMoves, null));

      GameMoveConsoleInfo info = new GameMoveConsoleInfo();

      int plyCount = 1;
      float timeEngine1Tot = 0;
      float timeEngine2Tot = 0;
      long visitsEngine1Tot = 0;
      long visitsEngine2Tot = 0;
      long nodesEngine1Tot = 0;
      long nodesEngine2Tot = 0;
      long evalsEngine1Tot = 0;
      long evalsEngine2Tot = 0;
      double backendWaitEngine1Tot = 0;
      double backendWaitEngine2Tot = 0;
      double backendSearchEngine1Tot = 0;
      double backendSearchEngine2Tot = 0;
      int movesEngine1 = 0;
      int movesEngine2 = 0;
      bool engine1ShouldHaveForfieted = false;
      bool engine2ShouldHaveForfieted = false;
      int numNodesForcedDeterministic = 0;

      // Most recent move string played (used by blunder detection to report the opponent's prior move).
      string lastPlayedMoveStr = null;
      // Carries a detected-but-unconfirmed blunder candidate from one ply to the next within this game.
      BlunderCandidate pendingBlunder = null;
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
          TotalEvaluationsEngine1 = evalsEngine1Tot,
          TotalEvaluationsEngine2 = evalsEngine2Tot,
          BackendWaitSecondsEngine1 = backendWaitEngine1Tot,
          BackendWaitSecondsEngine2 = backendWaitEngine2Tot,
          BackendSearchSecondsEngine1 = backendSearchEngine1Tot,
          BackendSearchSecondsEngine2 = backendSearchEngine2Tot,
          NumMovesForcedDeterministic = numNodesForcedDeterministic,
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
        if (scoresEngine1.Count >= Def.AdjudicateMinNumMoves)
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
        result = TournamentUtils.TryGetGameResultIfTerminal(curPositionAndMoves, !engine2IsWhite, useTablebasesForAdjudication,
                                                            Def.AdjudicateDrawByRepetitionImmediately, out reason);
        if (result != TournamentGameResult.None)
        {
          return MakeGameInfo(result, reason);
        }

        plyCount++;

        Position position = Position.FromFEN(info.FEN);
        int numPieces = position.PieceCount;

        // Make player's move
        GameMoveStat moveStat;
        if (engine2ToMove)
        {
          info = DoMove(engine2, engineCheckAgainstEngine2,
                        gameMoveHistory, searchLimitWithIncrementsEngine2, scoresEngine2,
                        ref nodesEngine2Tot, ref visitsEngine2Tot, ref timeEngine2Tot, ref evalsEngine2Tot,
                        ref backendWaitEngine2Tot, ref backendSearchEngine2Tot, ref numNodesForcedDeterministic);
          movesEngine2++;

          if (engine2IsWhite)
          {
            engine2ShouldHaveForfieted |= info.WhiteShouldHaveForfeitedOnLimit;
            moveStat = new GameMoveStat(plyCount, SideType.White, position, info.WhiteScoreQ, info.WhiteScoreCentipawns, engine2.CumulativeSearchTimeSeconds, numPieces, info.WhiteMAvg, info.WhiteFinalN, info.WhiteNumNodesComputed, info.WhiteSearchLimitPre, info.WhiteMoveTimeUsed);
          }
          else
          {
            engine2ShouldHaveForfieted |= info.BlackShouldHaveForfeitedOnLimit;
            moveStat = new GameMoveStat(plyCount, SideType.Black, position, info.BlackScoreQ, info.BlackScoreCentipawns, engine2.CumulativeSearchTimeSeconds, numPieces, info.BlackMAvg, info.BlackFinalN, info.BlackNumNodesComputed, info.BlackSearchLimitPre, info.BlackMoveTimeUsed);
          }
        }
        else
        {
          info = DoMove(engine1, null,
                        gameMoveHistory, searchLimitWithIncrementsEngine1, scoresEngine1,
                        ref nodesEngine1Tot, ref visitsEngine1Tot, ref timeEngine1Tot, ref evalsEngine1Tot,
                        ref backendWaitEngine1Tot, ref backendSearchEngine1Tot, ref numNodesForcedDeterministic);
          movesEngine1++;
          if (engine2IsWhite)
          {
            engine1ShouldHaveForfieted |= info.BlackShouldHaveForfeitedOnLimit;
            moveStat = new GameMoveStat(plyCount, SideType.Black, position, info.BlackScoreQ, info.BlackScoreCentipawns, engine1.CumulativeSearchTimeSeconds, numPieces, info.BlackMAvg, info.BlackFinalN, info.BlackNumNodesComputed, info.BlackSearchLimitPre, info.BlackMoveTimeUsed);
          }
          else
          {
            engine1ShouldHaveForfieted |= info.WhiteShouldHaveForfeitedOnLimit;
            moveStat = new GameMoveStat(plyCount, SideType.White, position, info.WhiteScoreQ, info.WhiteScoreCentipawns, engine1.CumulativeSearchTimeSeconds, numPieces, info.WhiteMAvg, info.WhiteFinalN, info.WhiteNumNodesComputed, info.WhiteSearchLimitPre, info.WhiteMoveTimeUsed);
          }
        }

        moveStat.Id = engine2ToMove ? engine2.ID : engine1.ID;
        gameMoveHistory.Add(moveStat);

        // Possibly detect a blunder by the opponent (whose move was the prior ply) and dump its
        // search graph for diagnosis (the dump is written only after the blundering engine confirms
        // it on its own next move). lastPlayedMoveStr currently holds the opponent's prior move.
        if (Def.BlunderDumpThresholdN != 0)
        {
          GameEngine movingEngine = engine2ToMove ? engine2 : engine1;
          GameEngine opponentEngine = engine2ToMove ? engine1 : engine2;
          CheckForBlunderDump(gameSequenceNum, 1 + openingIndex, gameMoveHistory, movingEngine, opponentEngine,
                              lastPlayedMoveStr, ref pendingBlunder);
        }
        lastPlayedMoveStr = moveStat.Side == SideType.White ? info.WhiteMoveStr : info.BlackMoveStr;

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
                                 ref long totalNodesUsed, ref long totalVisitsUsed, ref float totalTimeUsed,
                                 ref long totalEvalsUsed,
                                 ref double totalBackendWaitUsed, ref double totalBackendSearchUsed,
                                 ref int numNodesForcedDeterministic)
      {
        SearchLimit thisMoveSearchLimit = searchLimit.Type switch
        {
          SearchLimitType.SecondsPerMove => searchLimit,
          SearchLimitType.NodesPerMove => searchLimit,
          SearchLimitType.NodesPerTree => searchLimit,
          SearchLimitType.BestValueMove => searchLimit,
          SearchLimitType.BestActionMove => searchLimit,
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

        // Build a throttled progress callback that streams transient interim (mid-search) snapshots
        // to any live observer while the engine is still thinking, so the viewer's stats/eval/TopQ
        // update ~1s instead of freezing until the move completes. Costs nothing when nobody streams
        // (gated by a null observer, interval <= 0, and the cheap WantsInterim check).
        ITournamentObserver interimObserver = Def.parentDef?.Observer;
        int interimIntervalMs = Def.parentDef?.LiveStreamInterimIntervalMs ?? 0;
        GameEngine.ProgressCallback progressCb = null;
        if (interimObserver != null && interimIntervalMs > 0)
        {
          bool interimSideIsWhite = curPositionAndMoves.FinalPosition.MiscInfo.SideToMove == SideType.White;
          int interimPly = plyCount;
          DateTime interimSearchStart = DateTime.UtcNow;
          DateTime interimLastEmit = DateTime.MinValue;
          progressCb = (object ctx) =>
          {
            try
            {
              DateTime now = DateTime.UtcNow;
              double elapsedSec = (now - interimSearchStart).TotalSeconds;
              if (interimLastEmit == DateTime.MinValue)
              {
                // Quick first update so the panel populates shortly after the move begins.
                if (elapsedSec * 1000.0 < Math.Min(300, interimIntervalMs))
                {
                  return;
                }
              }
              else
              {
                // Graduated cadence: faster early, easing off for very long thinks.
                double reqMs = elapsedSec < 5 ? interimIntervalMs * 0.5
                             : elapsedSec < 30 ? interimIntervalMs
                             : interimIntervalMs * 3.0;
                if ((now - interimLastEmit).TotalMilliseconds < reqMs)
                {
                  return;
                }
              }

              interimLastEmit = now;
              if (!interimObserver.WantsInterim(ThreadIndex))
              {
                return;
              }

              InterimDTO interimDto = DtoMappers.ToInterim(ctx, interimPly, interimSideIsWhite);
              if (interimDto != null)
              {
                interimObserver.OnInterim(ThreadIndex, interimDto);
              }
            }
            catch { }
          };
        }

        GameEngineSearchResult engineMove = engine.Search(curPositionAndMoves, thisMoveSearchLimit, gameMoveHistory, progressCb);
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
          Position priorFinalPosition = curPositionAndMoves.FinalPosition;
          Position positionAfterMoveTest = priorFinalPosition.AfterMove(Move.FromUCI(in priorFinalPosition, engineMove.MoveString));
        }
        catch (Exception exc)
        {
          throw new Exception($"Engine {engine.ID} made illegal move {engineMove.MoveString} "
                           + $"in position {curPositionAndMoves.FinalPosition.FEN} "
                           + $"with limit {thisMoveSearchLimit}");
        }

        bool isWhite = curPositionAndMoves.FinalPosition.MiscInfo.SideToMove == SideType.White;

        // Possibly discard this move and replace with prior moved played by engine in this same position.
        GameMoveConsoleInfo overrideMoveToUse = null;
        if (Def.ForceReferenceEngineDeterministic && engine.ID == Def.ReferenceEngineId)
        {
          if (referenceEngineMoveHistory.ContainsKey(curPositionAndMoves.FinalPosition))
          {
            GameMoveConsoleInfo priorMove = referenceEngineMoveHistory[curPositionAndMoves.FinalPosition];
            string priorMoveStr = isWhite ? priorMove.WhiteMoveStr : priorMove.BlackMoveStr;
            if (priorMoveStr != engineMove.MoveString)
            {
              overrideMoveToUse = priorMove;
              numNodesForcedDeterministic++;
              //Console.WriteLine("See different move, was " + priorMoveStr + " now " + engineMove.MoveString + " in " + curPositionAndMoves.FinalPosition + "nps " + engineMove.NPS);
            }
          }
          else
          {
            // Save this move from reference engine to
            // make sure same moved always played in future games.
            referenceEngineMoveHistory[curPositionAndMoves.FinalPosition] = info;
          }
        }

        // Not currently trying to replace the statistics realting to N and time
        // so it doesn't look like the engine falsely violated the limits (at game level).
        string moveStr = overrideMoveToUse != null ? (isWhite ? overrideMoveToUse.WhiteMoveStr : overrideMoveToUse.BlackMoveStr) : engineMove.MoveString;
        float moveScoreQ = overrideMoveToUse != null ? (isWhite ? overrideMoveToUse.WhiteScoreQ : overrideMoveToUse.BlackScoreQ) : engineMove.ScoreQ;
        float moveScoreCentipawns = overrideMoveToUse != null ? (isWhite ? overrideMoveToUse.WhiteScoreCentipawns : overrideMoveToUse.BlackScoreCentipawns) : engineMove.ScoreCentipawns;
        float moveMAvg = overrideMoveToUse != null ? (isWhite ? overrideMoveToUse.WhiteMAvg : overrideMoveToUse.BlackMAvg) : engineMove.MAvg;
        SearchLimit searchLimitPost = overrideMoveToUse != null ? (isWhite ? overrideMoveToUse.WhiteSearchLimitPost : overrideMoveToUse.BlackSearchLimitPost) : engineMove.Limit;
        int moveDepth = overrideMoveToUse != null ? (isWhite ? overrideMoveToUse.WhiteDepth : overrideMoveToUse.BlackDepth) : engineMove.Depth;

        PositionWithHistory newPosition = new PositionWithHistory(curPositionAndMoves);
        newPosition.AppendMove(moveStr);

        // Output position to PGN
        pgnWriter.WriteMove(newPosition.Moves[^1], curPositionAndMoves.FinalPosition, engineTime, moveDepth, moveScoreCentipawns);

        scoresCP.Add(moveScoreCentipawns);

        totalNodesUsed += engineMove.FinalN;
        totalVisitsUsed += engineMove.Visits;
        totalTimeUsed += engineTime;

        // Accumulate neural network evaluations for this move (only where the engine reports EPS).
        // EPS is a per-move rate, so total evaluations are reconstructed as rate * elapsed time.
        if (engineMove.EPS > 0)
        {
          totalEvalsUsed += (long)MathF.Round(engineMove.EPS * engineTime);
        }

        // Accumulate device backend ("in C++ interop") busy time for this move (only where the
        // backend supports the metric, i.e. NNEvaluatorTensorRT). The search-loop elapsed time is
        // accumulated alongside as the matching denominator so the aggregate fraction is well-defined.
        if (!double.IsNaN(engineMove.TimeDeviceBackendWaitSeconds))
        {
          totalBackendWaitUsed += engineMove.TimeDeviceBackendWaitSeconds;
          totalBackendSearchUsed += engineMove.TimeElapsedTotalSeconds;
        }
        if (isWhite)
        {
          info.WhiteMoveStr = moveStr;
          info.WhiteScoreQ = moveScoreQ;
          info.WhiteScoreCentipawns = moveScoreCentipawns;
          info.WhiteSearchLimitPre = thisMoveSearchLimit;
          info.WhiteSearchLimitPost = searchLimitPost;
          info.WhiteMoveTimeUsed = engineTime;
          info.WhiteTimeAllMoves = totalTimeUsed;
          info.WhiteNodesAllMoves = totalNodesUsed;
          info.WhiteStartN = engineMove.StartingN;
          info.WhiteFinalN = engineMove.FinalN;
          info.WhiteMAvg = moveMAvg;
          info.WhiteCheckMoveStr = checkMove;
          info.WhiteShouldHaveForfeitedOnLimit = shouldHaveForfeited;
          info.WhiteDepth = moveDepth;
        }
        else
        {
          info.BlackMoveStr = moveStr;
          info.BlackScoreQ = moveScoreQ;
          info.BlackScoreCentipawns = moveScoreCentipawns;
          info.BlackSearchLimitPre = thisMoveSearchLimit;
          info.BlackSearchLimitPost = searchLimitPost;
          info.BlackMoveTimeUsed = engineTime;
          info.BlackTimeAllMoves = totalTimeUsed;
          info.BlackNodesAllMoves = totalNodesUsed;
          info.BlackStartN = engineMove.StartingN;
          info.BlackFinalN = engineMove.FinalN;
          info.BlackMAvg = moveMAvg;
          info.BlackCheckMoveStr = checkMove;
          info.BlackShouldHaveForfeitedOnLimit = shouldHaveForfeited;
          info.BlackDepth = moveDepth;
        }

        // Notify any live-streaming observer of this completed half-move.
        Def.parentDef.Observer?.OnMove(ThreadIndex,
            DtoMappers.ToMove(plyCount, isWhite, moveStr, newPosition.FinalPosition.FEN,
                              moveScoreCentipawns, moveScoreQ, moveMAvg, moveDepth,
                              engineMove, engineTime, thisMoveSearchLimit, newPosition.FinalPosition.PieceCount));

        if (showMoves)
        {
          info.PutStr();
        }

        // Advance to next move
        curPositionAndMoves = newPosition;

        return info;
      }
    }


    static float MaxAbsLastN(List<float> vals, int count)
    {
      float max = float.MinValue;
      for (int i = 1; i <= count; i++)
      {
        if (MathF.Abs(vals[^i]) > max)
        {
          max = MathF.Abs(vals[^i]);
        }
      }
      return max;
    }

    static float MinLastN(List<float> vals, int count)
    {
      float min = float.MaxValue;

      for (int i = 1; i <= count; i++)
      {
        if (vals[^i] < min)
        {
          min = vals[^i];
        }
      }

      return min;
    }

    static float MaxLastN(List<float> vals, int count)
    {
      float max = float.MinValue;

      for (int i = 1; i <= count; i++)
      {
        if (vals[^i] > max)
        {
          max = vals[^i];
        }
      }
      return max;
    }


    TournamentGameResult CheckResultAgreedBothEngines(List<float> scoresCPEngine1, List<float> scoresCPEngine2, TournamentGameResult result)
    {
      int WIN_THRESHOLD = Run.Def.AdjudicateWinThresholdCentipawns;
      int NUM_MOVES = Run.Def.AdjudicateWinThresholdNumMovesDecisive;

      // Return if insufficient moves in history to make determination.
      if (scoresCPEngine2.Count < NUM_MOVES || scoresCPEngine1.Count < NUM_MOVES)
      {
        return result;
      }

      if (MinLastN(scoresCPEngine2, NUM_MOVES) > WIN_THRESHOLD && MaxLastN(scoresCPEngine1, NUM_MOVES) < -WIN_THRESHOLD)
      {
        return TournamentGameResult.Loss;
      }
      else if (MinLastN(scoresCPEngine1, NUM_MOVES) > WIN_THRESHOLD && MaxLastN(scoresCPEngine2, NUM_MOVES) < -WIN_THRESHOLD)
      {
        return TournamentGameResult.Win;
      }
      else if (scoresCPEngine1.Count > Run.Def.AdjudicateDrawThresholdNumMoves
            && scoresCPEngine2.Count > Run.Def.AdjudicateDrawThresholdNumMoves
            && MaxAbsLastN(scoresCPEngine1, Run.Def.AdjudicateDrawThresholdNumMoves) < Run.Def.AdjudicateDrawThresholdCentipawns
            && MaxAbsLastN(scoresCPEngine2, Run.Def.AdjudicateDrawThresholdNumMoves) < Run.Def.AdjudicateDrawThresholdCentipawns)
      {
        return TournamentGameResult.Draw;
      }

      return result;
    }

  }
}
