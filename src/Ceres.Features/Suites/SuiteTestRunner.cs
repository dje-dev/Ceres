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
using Ceres.Features.GameEngines;
using Ceres.Chess.Positions;
using Ceres.Base.Misc;
using Ceres.Base.Benchmarking;
using System.Collections.Concurrent;
using Ceres.Base.Math;
using Ceres.Chess.NNEvaluators.Internals;
using System.Linq;

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
    public ConcurrentBag<GameEngine> EnginesCeres1 = new();

    /// <summary>
    /// Engine for optional second optional Ceres engine definition.
    /// </summary>
    public ConcurrentBag<GameEngine> EnginesCeres2 = new();

    /// <summary>
    /// Engine for optional external UCI engine.
    /// </summary>
    public GameEngine EngineExternal { get; private set; }


    int numConcurrentSuiteThreads;


    static void PopulateCeresEngines(GameEngineDef engineDef, EnginePlayerDef playerDef, ConcurrentBag<GameEngine> engines, int count)
    {
      if (engineDef != null)
      {
        for (int i = 0; i < count; i++)
        {
          GameEngine engine = engineDef.CreateEngine();
          engine.Warmup(playerDef.SearchLimit.KnownMaxNumNodes);
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
    int sumCeres1NumNodesWhenChoseTopNode, sumCeres2NumNodesWhenChoseTopNode;

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


#if NOT
    void DumpParams(TextWriter writer, bool differentOnly)
    {
      // Consider instead emulating/consolidating code in used by TournamentDef dumping
      ParamsDump.DumpParams(writer, differentOnly,
                  null, null,
                  Def.Engine1Def.GetEvaluatorDef(), Def.Engine2Def?.GetEvaluatorDef(),
                  Def.CeresEngine1Def.SearchLimit, Def.CeresEngine2Def?.SearchLimit,
                  null, null,
                  null, null,
                  null, null);
    }
#endif


    public SuiteTestResult Run(int numConcurrentSuiteThreads = 1,
                               bool outputDetail = true,
                               bool saveCacheWhenDone = true,
                               bool enableCancelVialCtrlC = true)
    {
      // Tree reuse is no help, indicate that we won't need it
      Def.Engine1Def.DisableTreeReuse();
      Def.Engine2Def?.DisableTreeReuse();

      // Disable dump for now, the execution parameters are modified
      // for the warmup which is confusing because different parameters
      // will be chosen for the actual search.
      //DumpParams(Def.Output, true);
      this.numConcurrentSuiteThreads = numConcurrentSuiteThreads;

      Init();

      // Install Ctrl-C handler to allow ad hoc clean termination (with stats).
      bool stopRequested = false;
      if (enableCancelVialCtrlC)
      {
        ConsoleCancelEventHandler ctrlCHandler = new ConsoleCancelEventHandler((object sender, ConsoleCancelEventArgs args) =>
        {
          Console.WriteLine("Suite pending shutdown....");
          stopRequested = true;
          args.Cancel = true;
        });
        Console.CancelKeyPress += ctrlCHandler;
      }

      int timerFiredCount = 0;

      // TODO: add path automatically
      List<EPDEntry> epds = EPDEntry.EPDEntriesInEPDFile(Def.EPDFileName, Def.MaxNumPositions,
                                                         Def.EPDLichessPuzzleFormat,
                                                         Def.EPDRawLineFilter, Def.EPDFilter);

      // Possibly skip some of positions at beginning of file.
      if (Def.SkipNumPositions > 0)
      {
        if (Def.SkipNumPositions <= epds.Count)
        {
          throw new Exception("Insufficient positions in " + Def.EPDFileName + " to skip " + Def.SkipNumPositions);
        }
        epds = epds.GetRange(Def.SkipNumPositions, epds.Count - Def.SkipNumPositions);
      }

      // Position filter EPDs based on the starting position.
      if (Def.AcceptPosPredicate != null)
      {
        epds = epds.Where(s => Def.AcceptPosPredicate(s.Position)).ToList();
      }

      if (Def.MaxNumPositions == 0)
      {
        Def.MaxNumPositions = epds.Count;
      }

      Def.Output.WriteLine();
      Def.Output.WriteLine("C1 = " + Def.CeresEngine1Def.EngineDef);
      if (Def.RunCeres2Engine)
      {
        Def.Output.WriteLine("C2 = " + Def.CeresEngine2Def.EngineDef);
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

      // Turn off position reuse if evaluators produce different results
      if (Def.RunCeres2Engine)
      {
        Chess.NNEvaluators.Defs.NNEvaluatorDef evalDef1 = Def.Engine1Def.GetEvaluatorDef();
        Chess.NNEvaluators.Defs.NNEvaluatorDef evalDef2 = Def.Engine2Def.GetEvaluatorDef();
        if (evalDef1 != null && evalDef2 != null && !evalDef1.NetEvaluationsIdentical(evalDef2))
        {
          Def.Engine1Def.SetReusePositionEvaluationsFromOther(false);
          Def.Engine2Def.SetReusePositionEvaluationsFromOther(false);
        }
      }

      if (Def.RunCeres2Engine && (Def.Engine1Def.GetReusePositionEvaluationsFromOther() ||
          Def.Engine2Def.GetReusePositionEvaluationsFromOther()))
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
        if (Def.ExternalEngineDef != null)
        {
          // For safety (to not overflow main or GPU memory) we limit number of LC0 processes.
          const int MAX_LC0_PROCESSES = 4;
          numExternalGameProcesses = Math.Min(MAX_LC0_PROCESSES, numConcurrentSuiteThreads);
        }
      }

      bool leelaVerboseMovesStats = true;//xxx Def.NumTestPos == 1;
      Func<int, object> makeExternalEngine = null;

      // TODO: eventually create clones of engine definitions
      //       then set each one to a different processorGroupID
      const int processorGroupID = 0;

      if (Def.ExternalEngineDef != null)
      {
        if (Def.ExternalEngineDef.EngineDef is GameEngineDefLC0)
        {
          GameEngineDefLC0 lc0EngineDef = Def.ExternalEngineDef.EngineDef as GameEngineDefLC0;
          bool forceDisableSmartPruning = lc0EngineDef.ForceDisableSmartPruning;
          const bool FILL_HISTORY = true;
          Chess.NNEvaluators.Defs.NNEvaluatorDef engine1EvalDef = Def.Engine1Def.GetEvaluatorDef();
          makeExternalEngine = (int processorGroupID) =>
          {
            LC0Engine engine = LC0EngineConfigured.GetLC0Engine(null, null, engine1EvalDef,
                                                                NNWeightsFiles.LookupNetworkFile(engine1EvalDef.Nets[0].Net.NetworkID),
                                                                processorGroupID, true,
                                                                false, leelaVerboseMovesStats, forceDisableSmartPruning,
                                                                lc0EngineDef.OverrideEXE, FILL_HISTORY, lc0EngineDef.ExtraCommandLineArgs);
            // WARMUP
            engine.AnalyzePositionFromFEN(Position.StartPosition.FEN, SearchLimit.NodesPerMove(1));
            return engine;
          };
        }
        else
        {
          makeExternalEngine = delegate (int processorGroupID)
          {
            GameEngine engine = Def.ExternalEngineDef.EngineDef.CreateEngine();
            engine.Warmup(Def.ExternalEngineDef.SearchLimit.KnownMaxNumNodes);
            return engine;
          };
        }
      }

      // Don't create too many non_Ceres threads since each one will consume separate GPU memory or threads.
      int maxLeelaThreads = Math.Min(numExternalGameProcesses, numConcurrentSuiteThreads);
      ObjectPool<object> externalEnginePool = new ObjectPool<object>(() => makeExternalEngine(processorGroupID), maxLeelaThreads);

      using (new TimingBlock("EPDS", Def.Output == Console.Out ? TimingBlock.LoggingType.Console : TimingBlock.LoggingType.None))
      {
        Parallel.For(0, epds.Count,
                     new ParallelOptions() { MaxDegreeOfParallelism = numConcurrentSuiteThreads },
                     delegate (int gameNum)
                     {
                       if (stopRequested)
                       {
                         return;
                       }

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
        TotalNodes2 = totalNodes2,

        AvgAbsQDifference = finalQ2.Count == 0 ? 0 : StatUtils.Average(StatUtils.AbsDiff(finalQ1.ToArray(), finalQ2.ToArray()))
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

      Def.Output.WriteLine();
      Def.Output.WriteLine($"Ceres1 total nodes to solve {sumCeres1NumNodesWhenChoseTopNode,12:N0}");
      Def.Output.WriteLine($"Ceres2 total nodes to solve {sumCeres2NumNodesWhenChoseTopNode,12:N0}");

      if (Def.CeresEngine2Def != null)
      {
        Def.Output.WriteLine();
        Def.Output.WriteLine($"Average absolute difference in final Q {StatUtils.Average(StatUtils.AbsDiff(finalQ1.ToArray(), finalQ2.ToArray())),5:F3}");
      }

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
      GameEngine EngineCeres1 = null;
      GameEngine EngineCeres2 = null;

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


      GameEngineSearchResult search1 = null;
      GameEngineSearchResult search2 = null;
      int scoreCeres1 = 0, scoreCeres2 = 0, scoreOtherEngine = 0;
      float otherEngineTime = 0;

      // The case of Lichess puzzles has to be handled specially.
      // We need to sequentially play out all the moves in the puzzle.
      // The engine gets all or no credit depending if it finds every one of the correct moves.
      LichessDatabaseRecord lichessPuzzle = epd.IsLichessPuzzle ? new LichessDatabaseRecord(epd.LichessRawLine) : default;
      int count = epd.IsLichessPuzzle ? lichessPuzzle.NumPuzzleMoves : 1;
      int countFailuresCeres1 = 0;
      int countFailuresCeres2 = 0;
      int countFailuresOtherEngine = 0;
      for (int i = 0; i < count; i++)
      {
        EPDEntry epdInSequence = epd.IsLichessPuzzle ? lichessPuzzle.EPDForPuzzleMoveAtIndex(i) : epd;

        bool isLastEntry = i == count - 1;
        otherEngineAnalysis2 = RunSearch(epdNum, epdInSequence,
                                         otherEngines, EngineCeres1, EngineCeres2,
                                         otherEngineAnalysis2,
                                         isLastEntry,
                                         out search1, out search2,
                                         out scoreCeres1, out scoreCeres2,
                                         out scoreOtherEngine,
                                         out otherEngineTime);

        if (scoreCeres1 < 10)
        {
          countFailuresCeres1++;
        }
        if (scoreCeres2 < 10)
        {
          countFailuresCeres2++;
        }
        if (scoreOtherEngine < 10)
        {
          countFailuresOtherEngine++;
        }
      }

      if (epd.IsLichessPuzzle && lichessPuzzle.NumPuzzleMoves > 1)
      {
        if (countFailuresCeres1 > 0)
        {
          scoreCeres1 = 0;
        }
        if (countFailuresCeres2 > 0)
        {
          scoreCeres2 = 0;
        }
        if (countFailuresOtherEngine > 0)
        {
          scoreOtherEngine = 0;
        }
      }

      float avgCeres1 = (float)accCeres1 / numSearches;
      float avgCeres2 = (float)accCeres2 / numSearches;
      float avgWCeres1 = (float)accWCeres1 / numSearchesBothFound;
      float avgWCeres2 = (float)accWCeres2 / numSearchesBothFound;

      float avgOther = (float)this.avgOther / numSearches;

      string MoveIfWrong(Move m) => m.IsNull || epd.CorrectnessScore(m, 10) == 10 ? "    " : m.ToString().ToLower();

      int diff1 = scoreCeres1 - scoreOtherEngine;

      //NodeEvaluatorNeuralNetwork
      int evalNumBatches1 = search1.NumNNBatches;
      int evalNumPos1 = search1.NumNNNodes;
      int evalNumBatches2 = search2 == null ? 0 : search2.NumNNBatches;
      int evalNumPos2 = search2 == null ? 0 : search2.NumNNNodes;

      string correctMove = null;
      if (epd.AMMoves != null)
      {
        correctMove = "-" + epd.AMMoves[0];
      }
      else if (epd.BMMoves != null)
      {
        correctMove = epd.BMMoves[0];
      }

      lock (lockObj)
      {
        totalTimeOther += otherEngineTime;
        totalTimeCeres1 += (float)search1.TimingStats.ElapsedTimeSecs;

        totalNodesOther += otherEngineAnalysis2 == null ? 0 : (int)otherEngineAnalysis2.Nodes;
        totalNodes1 += search1.FinalN;

        sumEvalNumPosOther += otherEngineAnalysis2 == null ? 0 : (int)otherEngineAnalysis2.Nodes;
        sumEvalNumBatches1 += evalNumBatches1;
        sumEvalNumPos1 += evalNumPos1;

        if (Def.RunCeres2Engine)
        {
          totalTimeCeres2 += (float)search2.TimingStats.ElapsedTimeSecs;
          totalNodes2 += search2.FinalN;
          sumEvalNumBatches2 += evalNumBatches2;
          sumEvalNumPos2 += evalNumPos2;
        }
      }

      float Adjust(int score, float frac) => score == 0 ? 0 : Math.Max(1.0f, MathF.Round(frac * 100.0f, 0));

      string worker1PickedNonTopNMoveStr = search1.PickedNonTopNMoveStr;
      string worker2PickedNonTopNMoveStr = search2?.PickedNonTopNMoveStr;

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

      writer.Add("MC", $"{search1.BestMoveMG,7}", 9);
      if (c2)
      {
        writer.Add("MC2", $"{search2.BestMoveMG,7}", 9);
      }

      writer.Add("Fr", $"{worker1PickedNonTopNMoveStr}{100.0f * search1.TopNNodeN / search1.FinalN,3:F0}%", 8);
      if (c2)
      {
        writer.Add("Fr2", $"{worker2PickedNonTopNMoveStr}{100.0f * search2?.TopNNodeN / search2?.FinalN,3:F0}%", 8);
      }

      writer.Add("Yld", $"{search1.NodeSelectionYieldFrac,6:f3}", 9);
      if (c2)
      {
        writer.Add("Yld2", $"{search2.NodeSelectionYieldFrac,6:f3}", 9);
      }

      // Search time
      if (ex)
      {
        writer.Add("TimeEx", $"{otherEngineTime,7:F2}", 9);
      }

      writer.Add("TimeC", $"{search1.TimingStats.ElapsedTimeSecs,7:F2}", 9);
      if (c2)
      {
        writer.Add("TimeC2", $"{search2.TimingStats.ElapsedTimeSecs,7:F2}", 9);
      }

      writer.Add("ADep", $"{search1.AvgDepth,5:f1}", 7);
      if (c2)
      {
        writer.Add("ADep2", $"{search2.AvgDepth,5:f1}", 7);
      }

      writer.Add("MDep", $"{search1.MaxDepth,5:f1}", 7);
      if (c2)
      {
        writer.Add("MDep2", $"{search2.MaxDepth,5:f1}", 7);
      }

      writer.Add("VEnt", $"{search1.VisitEntropy,5:f2}", 7);
      if (c2)
      {
        writer.Add("VEnt2", $"{search2.VisitEntropy,5:f2}", 7);
      }

      // Nodes
      if (ex) writer.Add("NEx", $"{otherEngineAnalysis2.Nodes,12:N0}", 14);
      writer.Add("Nodes", $"{search1.FinalN,12:N0}", 14);
      if (c2)
      {
        writer.Add("Nodes2", $"{search2.FinalN,12:N0}", 14);
      }

      // Fraction when chose top N
      writer.Add("Frac", $"{Adjust(scoreCeres1, search1.FractionNumNodesWhenChoseTopNNode),4:F0}", 6);
      if (c2)
      {
        writer.Add("Frac2", $"{Adjust(scoreCeres2, search2.FractionNumNodesWhenChoseTopNNode),4:F0}", 6);
      }

      // Score (Q)
      if (ex)
      {
        writer.Add("QEx", $"{otherEngineAnalysis2.ScoreLogistic,6:F3}", 8);
      }

      writer.Add("QC", $"{search1.ScoreQRoot,6:F3}", 8);
      if (c2)
      {
        writer.Add("QC2", $"{search2.ScoreQRoot,6:F3}", 8);
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
      writer.Add("TBase", $"{(search1.CountSearchContinuations > 0 ? 0 : search1.CountTablebaseHits),8:N0}", 10);
      if (c2)
      {
        writer.Add("TBase2", $"{(search2.CountSearchContinuations > 0 ? 0 : search2.CountTablebaseHits),8:N0}", 10);
      }

      if (Def.DumpEPDInfo)
      {
        writer.Add("FEN", epd.FEN, -1);
      }

      if (outputDetail)
      {
        if (epdNum == 0)
        {
          Def.Output.WriteLine(writer.ids.ToString());
          Def.Output.WriteLine(writer.dividers.ToString());
        }
        Def.Output.WriteLine(writer.text.ToString());
      }
    }


    private UCISearchInfo RunSearch(int epdNum, EPDEntry epd,
                                    ObjectPool<object> otherEngines,
                                    GameEngine engineCeres1,
                                    GameEngine engineCeres2,
                                    UCISearchInfo otherEngineAnalysis2,
                                    bool restoreEnginesToPoolWhenDone,
                                    out GameEngineSearchResult search1,
                                    out GameEngineSearchResult search2,
                                    out int scoreCeres1, out int scoreCeres2, out int scoreOtherEngine,
                                    out float otherEngineTime)
    {
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
            otherEngineAnalysis2 = le.AnalyzePositionFromFEN(epd.FENAndMoves, adjustedLimit);

            if (restoreEnginesToPoolWhenDone)
            {
              otherEngines.RestoreToPool(le);
            }
          }
          else if (engineObj is GameEngine gameEngine)
          {
            GameEngineSearchResult result = gameEngine.Search(epd.PosWithHistory, adjustedLimit);
            UCISearchInfo uciInfo = new UCISearchInfo(null, result.MoveString, null);
            uciInfo.Nodes = (ulong)result.FinalN;
            uciInfo.EngineReportedSearchTime = (int)(1000.0f * result.TimingStats.ElapsedTimeSecs);
            uciInfo.ScoreCentipawns = (int)MathF.Round(result.ScoreCentipawns);
            uciInfo.BestMove = result.MoveString;
            otherEngineAnalysis2 = uciInfo;
            if (restoreEnginesToPoolWhenDone)
            {
              otherEngines.RestoreToPool(engineObj);
            }
          }
          else if (engineObj is UCIGameRunner runner)
          {
            string moveType = Def.ExternalEngineDef.SearchLimit.Type == SearchLimitType.NodesPerMove ? "nodes" : "movetime";
            int moveValue = moveType == "nodes" ? (int)Def.ExternalEngineDef.SearchLimit.Value : (int)adjustedLimit.Value * 1000;
            runner.EvalPositionPrepare();
            otherEngineAnalysis2 = runner.EvalPosition(epd.FEN, epd.StartMoves, moveType, moveValue, null);
            if (restoreEnginesToPoolWhenDone)
            {
              otherEngines.RestoreToPool(runner);
            }
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
       && !Def.Engine1Def.GetFutilityPruningStopSearchEnabled())
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

      // Note that if we are running both Ceres1 and Ceres2 we alternate which search goes first.
      // This prevents any systematic difference/benefit that might come from order
      // (for example if we reuse position evaluations from the other tree, which can benefit only one of the two searches).
      search1 = null;
      search2 = null;
      if (epdNum % 2 == 0 || Def.CeresEngine2Def == null)
      {
        engineCeres1.ResetGame();
        search1 = engineCeres1.Search(pos, ceresSearchLimit1);

        if (Def.RunCeres2Engine)
        {
          search2 = engineCeres2.Search(pos, ceresSearchLimit2);
        }

      }
      else
      {
        engineCeres2.ResetGame();
        search2 = engineCeres2.Search(pos, ceresSearchLimit2);

        search1 = engineCeres1.Search(pos, ceresSearchLimit1);

      }

      if (restoreEnginesToPoolWhenDone)
      {
        // Restore engines to pool
        EnginesCeres1.Add(engineCeres1);
        EnginesCeres2?.Add(engineCeres2);
      }

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
        MGPosition thisPosX = PositionWithHistory.FromFENAndMovesUCI(epd.FEN, epd.StartMoves).FinalPosMG;

        MGMove lzMoveMG1 = MGMoveFromString.ParseMove(thisPosX, otherEngineAnalysis2.BestMove);
        bestMoveOtherEngine = MGMoveConverter.ToMove(lzMoveMG1);
      }

      Move bestMoveCeres1 = MGMoveConverter.ToMove(search1.BestMoveMG);
      Move bestMoveCeres2 = search2 == null ? default : MGMoveConverter.ToMove(search2.BestMoveMG);

      char CorrectStr(Move move) => epd.CorrectnessScore(move, 10) == 10 ? '+' : '.';

      scoreCeres1 = epd.CorrectnessScore(bestMoveCeres1, 10);
      scoreCeres2 = epd.CorrectnessScore(bestMoveCeres2, 10);
      scoreOtherEngine = epd.CorrectnessScore(bestMoveOtherEngine, 10);

      // Possibly invoke callback method.
      Def.Callback?.Invoke(epd, scoreCeres1, search1);

      otherEngineTime = otherEngineAnalysis2 == null ? 0 : (float)otherEngineAnalysis2.EngineReportedSearchTime / 1000.0f;
      lock (lockObj)
      {
        accCeres1 += scoreCeres1;
        accCeres2 += scoreCeres2;

        // Accumulate how many nodes were required to find one of the correct moves
        // (only in cases where both engines found correct move at some point).
        if (scoreCeres1 > 0 && (search2 == null || scoreCeres2 > 0))
        {
          sumCeres1NumNodesWhenChoseTopNode += search1.NumNodesWhenChoseTopNNode;
          if (search2 != null)
          {
            sumCeres2NumNodesWhenChoseTopNode += search2.NumNodesWhenChoseTopNNode;
          }
        }

        // Accumulate how many nodes were required to find one of the correct moves
        // (in the cases where both succeeded)
        if (scoreCeres1 > 0 && (search2 == null || scoreCeres2 > 0))
        {
          accWCeres1 += (scoreCeres1 == 0) ? search1.FinalN : search1.NumNodesWhenChoseTopNNode;
          if (search2 != null)
          {
            accWCeres2 += (scoreCeres2 == 0) ? search2.FinalN : search2.NumNodesWhenChoseTopNNode;
            solvedPct1MinusPct2Samples.Add(search1.FractionNumNodesWhenChoseTopNNode - search2.FractionNumNodesWhenChoseTopNNode);
          }
          numSearchesBothFound++;
        }
        this.avgOther += scoreOtherEngine;

        numSearches++;
      }

      return otherEngineAnalysis2;
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
        if (width != -1 && id.Length > width)
        {
          id = id.Substring(width);
        }

        ids.Append(width == -1 ? id : Center(id, width));

        if (width != -1)
        {
          for (int i = 0; i < width - 2; i++)
          {
            dividers.Append("-");
          }
        }
        dividers.Append("  ");
      }

      text.Append(width == -1 ? value : StringUtils.Sized(value, width));
    }

    static string Center(string str, int width)
    {
      int pad = width - str.Length;
      int padLeft = pad / 2 + str.Length;
      return str.PadLeft(padLeft, ' ').PadRight(width, ' ');
    }
  }

}
