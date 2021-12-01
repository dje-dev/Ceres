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
using System.Text;
using System.Threading.Tasks;

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.ExternalPrograms.UCI;
using Ceres.Chess.GameEngines;
using Ceres.Chess.Games.Utils;
using Ceres.Chess.LC0.Engine;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNFiles;
using Ceres.MCTS.Utils;
using Ceres.Features.GameEngines;
using Ceres.MCTS.Iteration;
using Ceres.Chess.Positions;
using Ceres.MCTS.Params;
using Ceres.Base.Misc;
using Ceres.Base.Benchmarking;
using System.Collections.Concurrent;
using Ceres.Features.Players;
using Ceres.Base.Math;
using Ceres.Chess.NNEvaluators.Internals;

#endregion

namespace Ceres.Features.Suites
{
  public class SuiteTestRunner
  {
    /// <summary>
    /// Definition of tournament used.
    /// </summary>
    public SuiteTestDef Def;

    /// <summary>
    /// Number of suite positions to evaluate concurrently.
    /// </summary>
    public int NumConcurrent;

    /// <summary>
    /// Engine for first Ceres engine definition.
    /// </summary>
    public ConcurrentBag<GameEngineCeresInProcess> EnginesCeres1 = new();

    /// <summary>
    /// Engine for optional second optional Ceres engine definition.
    /// </summary>
    public ConcurrentBag<GameEngineCeresInProcess> EnginesCeres2 = new();

    /// <summary>
    /// Engine for optional external UCI engine.
    /// </summary>
    public GameEngine EngineExternal { get; private set; }


    int numConcurrentSuiteThreads;


    static void PopulateCeresEngines(GameEngineDefCeres engineDefCeres, EnginePlayerDef engineDef, ConcurrentBag<GameEngineCeresInProcess> engines, int count)
    {
      if (engineDefCeres != null)
      {
        for (int i = 0; i < count; i++)
        {
          GameEngineCeresInProcess engine = engineDefCeres.CreateEngine() as GameEngineCeresInProcess;
          engine.Warmup(engineDef.SearchLimit.KnownMaxNumNodes);
          engines.Add(engine);
        }
      }
    }


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="def"></param>
    public SuiteTestRunner(SuiteTestDef def)
    {
      Def = def;
    }

    void Init()
    {
      // Create and warmup both engines (in parallel)
      Parallel.Invoke(() => { PopulateCeresEngines(Def.Engine1Def, Def.CeresEngine1Def, EnginesCeres1, numConcurrentSuiteThreads); },
                      () => { PopulateCeresEngines(Def.Engine2Def, Def.CeresEngine2Def, EnginesCeres2, numConcurrentSuiteThreads); },
                      () => { EngineExternal = Def.ExternalEngineDef?.EngineDef.CreateEngine(); EngineExternal?.Warmup(Def.ExternalEngineDef.SearchLimit.KnownMaxNumNodes); });
    }

    int numSearches = 0;
    int numSearchesBothFound = 0;
    int accCeres1 = 0, accCeres2 = 0, accWCeres1 = 0, accWCeres2 = 0, avgOther = 0;

    List<float> solvedPct1MinusPct2Samples = new();

    float totalTimeOther = 0;
    float totalTimeCeres1 = 0;
    float totalTimeCeres2 = 0;

    long totalNodesOther = 0;
    long totalNodes1 = 0;
    long totalNodes2 = 0;

    int sumEvalNumPosOther;
    int sumEvalNumBatches1;
    int sumEvalNumPos1;
    int sumEvalNumBatches2;
    int sumEvalNumPos2;


    void DumpParams(TextWriter writer, bool differentOnly)
    {
      // Consider instead emulating/consolidating code in used by TournamentDef dumping
      ParamsDump.DumpParams(writer, differentOnly,
                  null, null,
                  Def.Engine1Def.EvaluatorDef, Def.Engine2Def?.EvaluatorDef,
                  Def.CeresEngine1Def.SearchLimit, Def.CeresEngine2Def?.SearchLimit,
                 Def.Engine1Def.SelectParams, Def.Engine2Def?.SelectParams,
                 Def.Engine1Def.SearchParams, Def.Engine2Def?.SearchParams,
                 null, null,
                 Def.Engine1Def.SearchParams.Execution, Def.Engine2Def?.SearchParams.Execution);
    }


    public SuiteTestResult Run(int numConcurrentSuiteThreads = 1, bool outputDetail = true, bool saveCacheWhenDone = true)
    {
      // Tree reuse is no help, indicate that we won't need it
      Def.Engine1Def.SearchParams.TreeReuseEnabled = false;
      if (Def.Engine2Def != null) Def.Engine2Def.SearchParams.TreeReuseEnabled = false;

      // Disable dump for now, the execution parameters are modified 
      // for the warmup which is confusing because different parameters
      // will be chosen for the actual search.
      //DumpParams(Def.Output, true);
      this.numConcurrentSuiteThreads = numConcurrentSuiteThreads;

      Init();

      int timerFiredCount = 0;

      // TODO: add path automatically
      List<EPDEntry> epds = EPDEntry.EPDEntriesInEPDFile(Def.EPDFileName, Def.MaxNumPositions, 
                                                         Def.EPDLichessPuzzleFormat, Def.EPDFilter);

      // Possiblyskip some of positions at beginning of file.
      if (Def.SkipNumPositions > 0)
      {
        if (Def.SkipNumPositions <= epds.Count)
        {
          throw new Exception("Insufficient positions in " + Def.EPDFileName + " to skip " + Def.SkipNumPositions);
        }
        epds = epds.GetRange(Def.SkipNumPositions, epds.Count - Def.SkipNumPositions);
      }

      if (Def.MaxNumPositions == 0)
      {
        Def.MaxNumPositions = epds.Count;
      }

      Def.Output.WriteLine();
      Def.Output.WriteLine("C1 = " + Def.Engine1Def.EvaluatorDef);
      if (Def.RunCeres2Engine)
      {
        Def.Output.WriteLine("C2 = " + Def.Engine2Def.EvaluatorDef);
      }

      if (Def.ExternalEngineDef != null)
      {
        Def.Output.WriteLine("EX = " + Def.ExternalEngineDef.EngineDef);
      }

#if NOT
      // To make up for the fact that LZ0 "cheats" by sometimes running over specified number of nodes
      // (she seems to always fill the batch even if reached limit), add half a batch extra for Ceres as compensation 
      if (searchLimitCeres1.Type == SearchLimit.LimitType.NodesPerMove)
      {
        searchLimitCeres1 = new SearchLimit(searchLimit.Type, searchLimit.Value + paramsSearch1.BATCH_SIZE_PRIMARY / 2);
        searchLimitCeres2 = new SearchLimit(searchLimit.Type, searchLimit.Value + paramsSearch2.BATCH_SIZE_PRIMARY / 2);
      }
#endif

      //Def.Output.WriteLine($"MAX_CERES_GAME_THREADS {numConcurrentCeresGames} MAX_LEELA_GAME_THREADS {MAX_LEELA_GAME_THREADS}");

      // Turn of position reuse if evaluators produce different results
      if (Def.RunCeres2Engine && !Def.Engine1Def.EvaluatorDef.NetEvaluationsIdentical(Def.Engine2Def.EvaluatorDef))
      {
        Def.Engine1Def.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
        Def.Engine2Def.SearchParams.ReusePositionEvaluationsFromOtherTree = false;
      }

      if (Def.RunCeres2Engine && (Def.Engine1Def.SearchParams.ReusePositionEvaluationsFromOtherTree ||
          Def.Engine2Def.SearchParams.ReusePositionEvaluationsFromOtherTree))
      {
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("\r\nWARNING: REUSE_POSITION_EVALUATIONS_FROM_OTHER_TREE is turned on for one or both evaluators\r\n"
                         + "(alternating between the two evaluators). This may cause slight differences in search behavior and speed.\r\n");
        Console.ForegroundColor = ConsoleColor.White;
      }

      Def.Output.WriteLine();

      if (Def.MaxNumPositions > epds.Count) Def.MaxNumPositions = epds.Count;
      epds = epds.GetRange(Def.FirstTestPosition, Def.MaxNumPositions);

      int numExternalGameProcesses = 1;

      numConcurrentSuiteThreads = Math.Min(Def.MaxNumPositions, numConcurrentSuiteThreads);

      if (numConcurrentSuiteThreads > 1)
      {
        bool evaluator1NonPooled = Def.Engine1Def?.EvaluatorDef != null && Def.Engine1Def.EvaluatorDef.DeviceCombo != Chess.NNEvaluators.Defs.NNEvaluatorDeviceComboType.Pooled;
        bool evaluator2NonPooled = Def.Engine2Def?.EvaluatorDef != null && Def.Engine2Def.EvaluatorDef.DeviceCombo != Chess.NNEvaluators.Defs.NNEvaluatorDeviceComboType.Pooled;

        if (evaluator1NonPooled || evaluator2NonPooled)
        {
          throw new Exception("Must use POOLED neural network evaluator when running suites with parallelism"); ;
        }

        if (Def.ExternalEngineDef != null)
        {
          // For safety (to not overflow main or GPU memory) we limit number of LC0 processes.
          const int MAX_LC0_PROCESSES = 4;
          numExternalGameProcesses = Math.Min(MAX_LC0_PROCESSES, numConcurrentSuiteThreads);
        }
      }

      bool leelaVerboseMovesStats = true;//xxx Def.NumTestPos == 1;
      Func<object> makeExternalEngine = null;

      if (Def.ExternalEngineDef != null)
      {
        if (Def.ExternalEngineDef.EngineDef is GameEngineDefLC0)
        {
          GameEngineDefLC0 lc0EngineDef = Def.ExternalEngineDef.EngineDef as GameEngineDefLC0;
          bool forceDisableSmartPruning = lc0EngineDef.ForceDisableSmartPruning;
          const bool FILL_HISTORY = true;
          makeExternalEngine = () =>
          {
            LC0Engine engine = LC0EngineConfigured.GetLC0Engine(null, null, Def.Engine1Def.EvaluatorDef,
                                                                NNWeightsFiles.LookupNetworkFile(Def.Engine1Def.EvaluatorDef.Nets[0].Net.NetworkID),
                                                                true,
                                                                false, leelaVerboseMovesStats, forceDisableSmartPruning,
                                                                lc0EngineDef.OverrideEXE, FILL_HISTORY, lc0EngineDef.ExtraCommandLineArgs);
            // WARMUP
            engine.AnalyzePositionFromFEN(Position.StartPosition.FEN, SearchLimit.NodesPerMove(1));
            return engine;
          };
        }
        else
        {
          //bool resetMovesBetweenMoves = !Def.Engine2Def.SearchParams.TreeReuseEnabled;
          //bool enableTranpsositions = Def.Engine2Def.SearchParams.Execution.TranspositionMode != TranspositionMode.None;
          //bool enableTablebases = Def.Engine2Def.SearchParams.EnableTablebases;

          makeExternalEngine = () => Def.ExternalEngineDef.EngineDef.CreateEngine();
        }
      }

      // Don't create too many non_Ceres threads since each one will consume seaprate GPU memory or threads
      int maxLeelaThreads = Math.Min(numExternalGameProcesses, numConcurrentSuiteThreads);
      ObjectPool<object> externalEnginePool = new ObjectPool<object>(makeExternalEngine, maxLeelaThreads);

      using (new TimingBlock("EPDS"))
      {
        Parallel.For(0, epds.Count,
                     new ParallelOptions() { MaxDegreeOfParallelism = numConcurrentSuiteThreads },
                     delegate (int gameNum)
                     {
                       try
                       {
                         EPDEntry epd = epds[gameNum];

                         // Skip positions which are already draws
                         if (epd.Position.CheckDrawBasedOnMaterial == Position.PositionDrawStatus.DrawByInsufficientMaterial)
                         {
                           return;
                         }
                         // TODO: also do this for checkmate?

                         ProcessEPD(gameNum, epds[gameNum], outputDetail, externalEnginePool);

                       }
                       catch (Exception exc)
                       {
                         Def.Output.WriteLine("Error in ProcessEPD " + exc);
                         throw exc;
                       }
                     });
      }

      WriteSummaries();

      Shutdown(externalEnginePool);

      return new SuiteTestResult(Def)
      {
        AvgScore1 = (float)accCeres1 / numSearches,
        AvgScore2 = (float)accCeres2 / numSearches,
        AvgWScore1 = (float)accWCeres1 / numSearches,
        AvgWScore2 = (float)accWCeres2 / numSearches,
        AvgScoreLC0 = (float)avgOther / numSearches,

        TotalRuntimeLC0 = totalTimeOther,
        TotalRuntime1 = totalTimeCeres1,
        TotalRuntime2 = totalTimeCeres2,

        FinalQ1 = finalQ1.ToArray(),
        FinalQ2 = finalQ2?.ToArray(),

        TotalNodesLC0 = totalNodesOther,
        TotalNodes1 = totalNodes1,
        TotalNodes2 = totalNodes2
      };

    }

    private void Shutdown(ObjectPool<object> externalEnginePool)
    {
      return;

      // TODO: restore this, currently buggy (stack overflow)
      foreach (var engine in EnginesCeres1)
      {
        engine.Dispose();
      }
      EnginesCeres1.Clear();

      foreach (var engine in EnginesCeres2)
      {
        engine.Dispose();
      }
      EnginesCeres2.Clear();

      EngineExternal?.Dispose();

      externalEnginePool.Shutdown(engineObj => (engineObj as IDisposable)?.Dispose());
    }

    private void WriteSummaries()
    {
      Def.Output.WriteLine();

      Def.Output.WriteLine();
      if (Def.ExternalEngineDef != null)
      {
        Def.Output.WriteLine($"Total {Def.ExternalEngineDef.ID} Time {totalTimeOther,6:F2}");
      }

      Def.Output.WriteLine($"Total C1 Time {totalTimeCeres1,6:F2}");
      if (Def.CeresEngine2Def != null)
      {
        Def.Output.WriteLine($"Total C2 Time {totalTimeCeres2,6:F2}");
      }

      Def.Output.WriteLine();
      if (Def.ExternalEngineDef != null)
      {
        Def.Output.WriteLine($"Avg {Def.ExternalEngineDef.ID} pos/sec    {totalNodesOther / totalTimeOther,8:F2}");
      }

      Def.Output.WriteLine($"Avg Ceres    pos/sec    {totalNodes1 / totalTimeCeres1,8:F2}");
      if (Def.CeresEngine2Def != null)
      {
        Def.Output.WriteLine($"Avg Ceres2    pos/sec    {totalNodes2 / totalTimeCeres2,8:F2}");
      }

      Def.Output.WriteLine();

      Def.Output.WriteLine($"Num all evaluations      :   {NNEvaluatorStats.TotalPosEvaluations,12:N0}");
      Def.Output.WriteLine($"Num secondary batches    :   {MCTSManager.NumSecondaryBatches,12:N0}");
      Def.Output.WriteLine($"Num secondary evaluations:   {MCTSManager.NumSecondaryEvaluations,12:N0}");

      Def.Output.WriteLine();

      float avgFaster = (float) StatUtils.Average(solvedPct1MinusPct2Samples);
      float stdFaster = (float)StatUtils.StdDev(solvedPct1MinusPct2Samples) / MathF.Sqrt(solvedPct1MinusPct2Samples.Count);
      Def.Output.WriteLine($"Ceres1 time required to solve vs. Ceres2 (%) {(100* avgFaster),5:F2} +/-{(100 * stdFaster),5:F2}");

      Def.Output.WriteLine();
    }



    List<float> finalQ1 = new();
    List<float> finalQ2 = new();

    readonly object lockObj = new();

    void ProcessEPD(int epdNum, EPDEntry epd, bool outputDetail, ObjectPool<object> otherEngines)
    {
      GameEngineCeresInProcess EngineCeres1 = null;
      GameEngineCeresInProcess EngineCeres2 = null;

      if (!EnginesCeres1.TryTake(out EngineCeres1))
      {
        throw new Exception("No engine available");
      }

      if (Def.Engine2Def != null)
      {
        if (!EnginesCeres2.TryTake(out EngineCeres2))
        {
          throw new Exception("No engine available");
        }
      }

      EngineCeres1.ResetGame();
      EngineCeres2?.ResetGame();
      EngineExternal?.ResetGame();

      UCISearchInfo otherEngineAnalysis2 = default;

      EPDEntry epdToUse = epd;

      Task RunNonCeres()
      {
        if (Def.ExternalEngineDef != null)
        {
          object engineObj = otherEngines.GetFromPool();

          SearchLimit adjustedLimit = Def.ExternalEngineDef.SearchLimit.ConvertedGameToMoveLimit;
          if (engineObj is LC0Engine)
          {
            LC0Engine le = (LC0Engine)engineObj;

            // Run test 2 first since that's the one we dump in detail, to avoid any possible caching effect from a prior run
            otherEngineAnalysis2 = le.AnalyzePositionFromFEN(epdToUse.FENAndMoves, adjustedLimit);
            //            leelaAnalysis2 = le.AnalyzePositionFromFEN(epdToUse.FEN, new SearchLimit(SearchLimit.LimitType.NodesPerMove, 2)); // **** TEMP
            otherEngines.RestoreToPool(le);
          }
          else
          {
            UCIGameRunner runner = (engineObj is UCIGameRunner) ? (engineObj as UCIGameRunner) 
            : (engineObj as GameEngineUCI).UCIRunner;
            string moveType = Def.ExternalEngineDef.SearchLimit.Type == SearchLimitType.NodesPerMove ? "nodes" : "movetime";
            int moveValue = moveType == "nodes" ? (int)Def.ExternalEngineDef.SearchLimit.Value : (int)adjustedLimit.Value * 1000;
            runner.EvalPositionPrepare();
            otherEngineAnalysis2 = runner.EvalPosition(epdToUse.FEN, epdToUse.StartMoves, moveType, moveValue, null);
            otherEngines.RestoreToPool(runner);
            //          public UCISearchInfo EvalPosition(int engineNum, string fenOrPositionCommand, string moveType, int moveMetric, bool shouldCache = false)
          }
        }
        return Task.CompletedTask;
      }

      bool EXTERNAL_CONCURRENT = numConcurrentSuiteThreads > 1;

      Task lzTask = EXTERNAL_CONCURRENT ? Task.Run(RunNonCeres) : RunNonCeres();

      // Compute search limit
      // If possible, adjust for the fact that LC0 "cheats" by going slightly over node budget
      SearchLimit ceresSearchLimit1 = Def.CeresEngine1Def.SearchLimit.ConvertedGameToMoveLimit;
      SearchLimit ceresSearchLimit2 = Def.CeresEngine2Def?.SearchLimit.ConvertedGameToMoveLimit;

      if (Def.CeresEngine1Def.SearchLimit.Type == SearchLimitType.NodesPerMove
       && otherEngineAnalysis2 != null
       && !Def.Engine1Def.SearchParams.FutilityPruningStopSearchEnabled)
      {
        if (Def.CeresEngine1Def.SearchLimit.Type == SearchLimitType.NodesPerMove)
        {
          ceresSearchLimit1 = new SearchLimit(SearchLimitType.NodesPerMove, otherEngineAnalysis2.Nodes);
        }
        if (Def.CeresEngine1Def.SearchLimit.Type == SearchLimitType.NodesPerMove)
        {
          ceresSearchLimit2 = new SearchLimit(SearchLimitType.NodesPerMove, otherEngineAnalysis2.Nodes);
        }
      }

      PositionWithHistory pos = epd.PosWithHistory;

      // TODO: should this be switched to GameEngineCeresInProcess?

      // Note that if we are running both Ceres1 and Ceres2 we alternate which search goes first.
      // This prevents any systematic difference/benefit that might come from order
      // (for example if we reuse position evaluations from the other tree, which can benefit only one of the two searches).
      GameEngineSearchResultCeres search1 = null;
      GameEngineSearchResultCeres search2 = null;

      if (epdNum % 2 == 0 || Def.CeresEngine2Def == null)
      {
        EngineCeres1.ResetGame();
        search1 = EngineCeres1.SearchCeres(pos, ceresSearchLimit1);

        MCTSIterator shareContext = null;
        if (Def.RunCeres2Engine)
        {
          if (Def.Engine2Def.SearchParams.ReusePositionEvaluationsFromOtherTree)
          {
            shareContext = search1.Search.Manager.Context;
          }

          search2 = EngineCeres2.SearchCeres(pos, ceresSearchLimit2);
        }
        
      }
      else
      {
        EngineCeres2.ResetGame();
        search2 = EngineCeres2.SearchCeres(pos, ceresSearchLimit2);

        MCTSIterator shareContext = null;
        if (Def.Engine1Def.SearchParams.ReusePositionEvaluationsFromOtherTree)
        {
          shareContext = search2.Search.Manager.Context;
        }

        search1 = EngineCeres1.SearchCeres(pos, ceresSearchLimit1);

      }

      // Restore engines to pool
      EnginesCeres1.Add(EngineCeres1);
      EnginesCeres2?.Add(EngineCeres2);

      lock (lockObj)
      {
        while (finalQ1.Count <= epdNum)
        {
          finalQ1.Add(float.NaN);
        }

        finalQ1[epdNum] = (float)search1.ScoreQ;

        if (search2 != null)
        {
          while (finalQ2.Count <= epdNum)
          {
            finalQ2.Add(float.NaN);
          }

          finalQ2[epdNum] = (float)search2.ScoreQ;
        }
      }

      // Wait for LZ analysis
      if (EXTERNAL_CONCURRENT) lzTask.Wait();

      Move bestMoveOtherEngine = default;

      if (Def.ExternalEngineDef != null)
      {
        MGPosition thisPosX = PositionWithHistory.FromFENAndMovesUCI(epdToUse.FEN, epdToUse.StartMoves).FinalPosMG;

        MGMove lzMoveMG1 = MGMoveFromString.ParseMove(thisPosX, otherEngineAnalysis2.BestMove);
        bestMoveOtherEngine = MGMoveConverter.ToMove(lzMoveMG1);
      }

      Move bestMoveCeres1 = MGMoveConverter.ToMove(search1.BestMove.BestMove);
      Move bestMoveCeres2 = search2 == null ? default : MGMoveConverter.ToMove(search2.BestMove.BestMove);

      char CorrectStr(Move move) => epdToUse.CorrectnessScore(move, 10) == 10 ? '+' : '.';

      int scoreCeres1 = epdToUse.CorrectnessScore(bestMoveCeres1, 10);
      int scoreCeres2 = epdToUse.CorrectnessScore(bestMoveCeres2, 10);
      int scoreOtherEngine = epdToUse.CorrectnessScore(bestMoveOtherEngine, 10);

      SearchResultInfo result1 = new SearchResultInfo(search1.Search.Manager, search1.BestMove);
      SearchResultInfo result2 = search2 == null ? null : new SearchResultInfo(search2.Search.Manager, search2.BestMove);

      float otherEngineTime = otherEngineAnalysis2 == null ? 0 : (float)otherEngineAnalysis2.EngineReportedSearchTime / 1000.0f;

      lock (lockObj)
      {
        accCeres1 += scoreCeres1;
        accCeres2 += scoreCeres2;

        // Accumulate how many nodes were required to find one of the correct moves
        // (in the cases where both succeeded)
        if (scoreCeres1 > 0 && (search2 == null || scoreCeres2 > 0))
        {
          accWCeres1 += (scoreCeres1 == 0) ? result1.N : result1.NumNodesWhenChoseTopNNode;
          if (search2 != null)
          {
            accWCeres2 += (scoreCeres2 == 0) ? result2.N : result2.NumNodesWhenChoseTopNNode;
            solvedPct1MinusPct2Samples.Add(result1.FractionNumNodesWhenChoseTopNNode - result2.FractionNumNodesWhenChoseTopNNode);
          }
          numSearchesBothFound++;
        }
        this.avgOther += scoreOtherEngine;

        numSearches++;
       }

        float avgCeres1 = (float)accCeres1 / numSearches;
        float avgCeres2 = (float)accCeres2 / numSearches;
        float avgWCeres1 = (float)accWCeres1 / numSearchesBothFound;
        float avgWCeres2 = (float)accWCeres2 / numSearchesBothFound;

        float avgOther = (float)this.avgOther / numSearches;

        string MoveIfWrong(Move m) => m.IsNull || epdToUse.CorrectnessScore(m, 10) == 10 ? "    " : m.ToString().ToLower();

        int diff1 = scoreCeres1 - scoreOtherEngine;

        //NodeEvaluatorNeuralNetwork
        int evalNumBatches1 = result1.NumNNBatches;
        int evalNumPos1 = result1.NumNNNodes;
        int evalNumBatches2 = search2 == null ? 0 : result2.NumNNBatches;
        int evalNumPos2 = search2 == null ? 0 : result2.NumNNNodes;

        string correctMove = null;
        if (epdToUse.AMMoves != null)
        {
          correctMove = "-" + epdToUse.AMMoves[0];
        }
        else if (epdToUse.BMMoves != null)
        {
          correctMove = epdToUse.BMMoves[0];
        }

      lock (lockObj)
      {
        totalTimeOther += otherEngineTime;
        totalTimeCeres1 += (float)search1.TimingStats.ElapsedTimeSecs;

        totalNodesOther += otherEngineAnalysis2 == null ? 0 : (int)otherEngineAnalysis2.Nodes;
        totalNodes1 += (int)result1.N;

        sumEvalNumPosOther += otherEngineAnalysis2 == null ? 0 : (int)otherEngineAnalysis2.Nodes;
        sumEvalNumBatches1 += evalNumBatches1;
        sumEvalNumPos1 += evalNumPos1;

        if (Def.RunCeres2Engine)
        {
          totalTimeCeres2 += (float)search2.TimingStats.ElapsedTimeSecs;
          totalNodes2 += (int)result2.N;
          sumEvalNumBatches2 += evalNumBatches2;
          sumEvalNumPos2 += evalNumPos2;
        }
      }

      float Adjust(int score, float frac) => score == 0 ? 0 : Math.Max(1.0f, MathF.Round(frac * 100.0f, 0));

      string worker1PickedNonTopNMoveStr = result1.PickedNonTopNMoveStr;
      string worker2PickedNonTopNMoveStr = result2?.PickedNonTopNMoveStr;

      bool ex = otherEngineAnalysis2 != null;
      bool c2 = search2 != null;

      Writer writer = new Writer(epdNum == 0);
      writer.Add("#", $"{epdNum,4}", 6);

      if (ex)
      {
        writer.Add("CEx", $"{avgOther,5:F2}", 7);
      }

      writer.Add("CC", $"{avgCeres1,5:F2}", 7);
      if (c2)
      {
        writer.Add("CC2", $"{avgCeres2,5:F2}", 7);
      }

      writer.Add("P", $"{0.001f * avgWCeres1,6:f2}", 8);
      if (c2)
      {
        writer.Add("P2", $"{0.001f * avgWCeres2,6:f2}", 8);
      }

      if (ex)
      {
        writer.Add("SEx", $" {scoreOtherEngine,3}", 5);
      }

      writer.Add("SC", $" {scoreCeres1,3}", 5);
      if (c2)
      {
        writer.Add("SC2", $" {scoreCeres2,3}", 5);
      }

      if (ex)
      {
        writer.Add("MEx", $"{otherEngineAnalysis2.BestMove,7}", 9);
      }

      writer.Add("MC", $"{search1.BestMove.BestMove,7}", 9);
      if (c2)
      {
        writer.Add("MC2", $"{search2.BestMove.BestMove,7}", 9);
      }

      writer.Add("Fr", $"{worker1PickedNonTopNMoveStr}{ 100.0f * result1.TopNNodeN / result1.N,3:F0}%", 8);
      if (c2)
      {
        writer.Add("Fr2", $"{worker2PickedNonTopNMoveStr}{ 100.0f * result2?.TopNNodeN / result2?.N,3:F0}%", 8);
      }

      writer.Add("Yld", $"{result1.NodeSelectionYieldFrac,6:f3}", 9);
      if (c2)
      {
        writer.Add("Yld2", $"{result2.NodeSelectionYieldFrac,6:f3}", 9);
      }

      // Search time
      if (ex)
      {
        writer.Add("TimeEx", $"{otherEngineTime,7:F2}", 9);
      }

      writer.Add("TimeC", $"{search1.TimingStats.ElapsedTimeSecs,7:F2}", 9);
      if (c2) {
        writer.Add("TimeC2", $"{search2.TimingStats.ElapsedTimeSecs,7:F2}", 9);
      }

      writer.Add("Dep", $"{result1.AvgDepth,5:f1}", 7);
      if (c2)
      {
        writer.Add("Dep2", $"{result2.AvgDepth,5:f1}", 7);
      }

      // Nodes
      if (ex) writer.Add("NEx", $"{otherEngineAnalysis2.Nodes,12:N0}", 14);
      writer.Add("Nodes", $"{result1.N,12:N0}", 14);
      if (c2)
      {
        writer.Add("Nodes2", $"{result2.N,12:N0}", 14);
      }

      // Fraction when chose top N
      writer.Add("Frac", $"{Adjust(scoreCeres1, result1.FractionNumNodesWhenChoseTopNNode),4:F0}", 6);
      if (c2)
      {
        writer.Add("Frac2", $"{Adjust(scoreCeres2, result2.FractionNumNodesWhenChoseTopNNode),4:F0}", 6);
      }

      // Score (Q)
      if (ex)
      {
        writer.Add("QEx", $"{otherEngineAnalysis2.ScoreLogistic,6:F3}", 8);
      }

      writer.Add("QC", $"{result1.Q,6:F3}", 8);
      if (c2)
      {
        writer.Add("QC2", $"{result2.Q,6:F3}", 8);
      }

      // Num batches&positions
      writer.Add("Batches", $"{evalNumBatches1,8:N0}", 10);
      writer.Add("NNEvals", $"{evalNumPos1,11:N0}", 13);
      if (c2)
      {
        writer.Add("Batches2", $"{evalNumBatches2,8:N0}", 10);
        writer.Add("NNEvals2", $"{evalNumPos2,11:N0}", 13);
      }

      // Tablebase hits
      writer.Add("TBase", $"{(search1.Search.CountSearchContinuations > 0 ? 0 : search1.Search.Manager.CountTablebaseHits),8:N0}", 10);
      if (c2)
      {
        writer.Add("TBase2", $"{(search2.Search.CountSearchContinuations > 0 ? 0 : search2.Search.Manager.CountTablebaseHits),8:N0}", 10);
      }

      //      writer.Add("EPD", $"{epdToUse.ID,-30}", 32);

      if (outputDetail)
      {
        if (epdNum == 0)
        {
          Def.Output.WriteLine(writer.ids.ToString());
          Def.Output.WriteLine(writer.dividers.ToString());
        }
        Def.Output.WriteLine(writer.text.ToString());
      }

      //      MCTSNodeStorageSerialize.Save(worker1.Context.Store, @"c:\temp", "TESTSORE");

      // TODO: seems not safe to release here, is it ok just to leave to finalization for dispose?
      //search1?.Search.Manager?.Dispose();
      //if (!object.ReferenceEquals(search1?.Search.Manager, search2?.Search.Manager)) search2?.Search.Manager?.Dispose();
    }


  }

  internal class Writer
  {
    public readonly bool WithHeader;

    public StringBuilder text = new StringBuilder();
    public StringBuilder ids;
    public StringBuilder dividers;

    public Writer(bool withHeader)
    {
      WithHeader = withHeader;

      if (withHeader)
      {
        ids = new StringBuilder();
        dividers = new StringBuilder();
      }
    }

    public void Add(string id, string value, int width)
    {
      if (WithHeader)
      {
        if (id.Length > width)
        {
          id = id.Substring(width);
        }

        ids.Append(Center(id, width));

        for (int i = 0; i < width - 2; i++)
        {
          dividers.Append("-");
        }

        dividers.Append("  ");
      }

      text.Append(StringUtils.Sized(value, width));
    }

    static string Center(string str, int width)
    {
      int pad = width - str.Length;
      int padLeft = pad / 2 + str.Length;
      return str.PadLeft(padLeft, ' ').PadRight(width, ' ');
    }
  }

}
