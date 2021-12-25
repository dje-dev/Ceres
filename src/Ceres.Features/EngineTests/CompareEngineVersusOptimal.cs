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
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataType.Trees;
using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.GameEngines;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Chess.SearchResultVerboseMoveInfo;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;
using Ceres.Features.GameEngines;
using Ceres.Base.OperatingSystem;

#endregion

namespace Ceres.Features.EngineTests
{


  /// <summary>
  /// Runs many searches using two engines, one baseline vs one with specified modifications
  /// and compares best move against best move according to the baseline engine 
  /// run for much longer search (presumably seeing something closer to the true best move).
  /// </summary>
  public class CompareEnginesVersusOptimal
  {
    public enum PlayerMode
    {
      Ceres,
      LC0,
      Stockfish14_1
    };

    /// <summary>
    /// If smart pruning should be disabled on engines
    /// so that searches by number of nodes are truly equivalent.
    /// </summary>
    const bool DISABLE_PRUNING = true;

    public CompareEngineParams Params;

    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="parms"></param>
    public CompareEnginesVersusOptimal(CompareEngineParams parms)
    {
      Params = parms;
    }


    List<float> qDiffs = new();

    int countMuchBetter = 0;
    int countMuchWorse = 0;
    int countScored = 0;
    float accOverlapDepth6 = 0;
    int countDifferentMoves = 0;

    ConcurrentDictionary<ulong, bool> seenPositions = new ();

    volatile bool shutdownRequested;

    float timeAccumulatorEngine1 = 0;
    float timeAccumulatorEngine2 = 0;

    private string ShortID1 => Params.Player1.ToString()[0] + "_" + Params.NetworkID1;
    private string ShortID2 => Params.Player2.ToString()[0] + "_" + Params.NetworkID2;
    private string ShortIDArbiter => Params.PlayerArbiter.ToString()[0] + "_" + Params.NetworkArbiterID;


    ParamsSearch pArbiter;
    ParamsSelect sArbiter;
    TimingStats timingStats = new TimingStats();

    public int[] GPUIDs => Params.GPUIDs ?? new int[] { 0 };
    public SearchLimit Limit => Params.Limit with { SearchCanBeExpanded = false };

    public void RequestCancellation() => shutdownRequested = true;
    

    public CompareEngineResultSummary Run()
    {
      WriteIntroBanner();

      // Install Ctrl-C handler to allow ad hoc clean termination of tournament (with stats).
      ConsoleCancelEventHandler ctrlCHandler = new ConsoleCancelEventHandler((object sender,
        ConsoleCancelEventArgs args) =>
      {
        Console.WriteLine("Pending shutdown....");
        shutdownRequested = true;
      }); ;
      Console.CancelKeyPress += ctrlCHandler;

      // Create default parameters, with smart pruning tuned off.
      ParamsSearch p1 = new ParamsSearch()
      {
        FutilityPruningStopSearchEnabled = !DISABLE_PRUNING,
        MoveOverheadSeconds = 0
      };
      ParamsSearch p2 = new ParamsSearch()
      {
        FutilityPruningStopSearchEnabled = !DISABLE_PRUNING,
        MoveOverheadSeconds = 0
      };

      ParamsSelect s1 = new ParamsSelect();
      ParamsSelect s2 = new ParamsSelect();

      // A higher CPUCTAtRoot is used with arbiter to encourage to
      // get more accurate Q values across all moves
      // (including possibly inferior ones chosen by other engines).
      pArbiter = new ParamsSearch();
      sArbiter = new ParamsSelect() { CPUCTAtRoot = new ParamsSelect().CPUCTAtRoot * 3 };

      Params.SearchModifier1?.Invoke(p1);
      Params.SelectModifier1?.Invoke(s1);

      Params.SearchModifier2?.Invoke(p2);
      Params.SelectModifier2?.Invoke(s2);

      using (new TimingBlock(timingStats, TimingBlock.LoggingType.None))
      {
        Parallel.ForEach(GPUIDs, i=> RunCompareThread(i, p1, p2, s1, s2, pArbiter, sArbiter));
      }

      return ProcessSummaryInfo();
    }


    private void RunCompareThread(int gpuID,
                                  ParamsSearch p1, ParamsSearch p2,
                                  ParamsSelect s1, ParamsSelect s2,
                                  ParamsSearch pOptimal, ParamsSelect sOptimal)
    {
      try
      {
        DoRunCompareThread(gpuID, p1, p2, s1, s2, pArbiter, sArbiter);
      }
      catch (Exception ex)
      {
        Console.WriteLine($"Exception in DoCompareThread on GPU {gpuID}, shutting thread down.");
        Console.WriteLine(ex.ToString()); 
      }
    }


    private void DoRunCompareThread(int gpuID,
                                    ParamsSearch p1, ParamsSearch p2,
                                    ParamsSelect s1, ParamsSelect s2,
                                    ParamsSearch pOptimal, ParamsSelect sOptimal)
    {
      NNEvaluatorDef evaluatorDef1 = Params.NetworkID1 != null ? NNEvaluatorDef.FromSpecification(Params.NetworkID1, $"GPU:{gpuID}") : null;
      NNEvaluatorDef evaluatorDef2 = Params.NetworkID2 != null ? NNEvaluatorDef.FromSpecification(Params.NetworkID2, $"GPU:{gpuID}") : null;
      NNEvaluatorDef evaluatorDefOptimal = Params.NetworkArbiterID != null ? NNEvaluatorDef.FromSpecification(Params.NetworkArbiterID, $"GPU:{gpuID}") : null;

      static GameEngine MakeEngine(PlayerMode playerMode, string networkID, NNEvaluatorDef evaluatorDef,
                                   ParamsSearch paramsSearch, ParamsSelect paramsSelect)
      {
        if (playerMode == PlayerMode.Ceres)
        {
          GameEngineCeresInProcess ge = new ("Ceres", evaluatorDef, null, paramsSearch, paramsSelect);
          ge.GatherVerboseMoveStats = true;
          return ge;
        }
        else if (playerMode == PlayerMode.LC0)
        {
          return new GameEngineLC0("LC0", networkID, DISABLE_PRUNING, false, paramsSearch, paramsSelect, evaluatorDef, 
                                   verbose:true, extraCommandLineArgs:"--move-overhead=0");
        }
        else if (playerMode == PlayerMode.Stockfish14_1)
        {
          string sf14FN = Path.Combine(CeresUserSettingsManager.Settings.DirExternalEngines, SoftwareManager.IsLinux ? "stockfish14.1" : "stockfish14.1.exe");
          GameEngineDef engineDef = new GameEngineDefUCI("SF14", new GameEngineUCISpec("SF14", sf14FN, 16,
                                                         2048, CeresUserSettingsManager.Settings.TablebaseDirectory));
          return engineDef.CreateEngine();
        }
        else
        {
          throw new NotImplementedException("Unsupported opponent mode " + playerMode);
        }
      }


      GameEngine engine1 = null;
      GameEngine engine2 = null;
      GameEngine engineOptimal = null;
      GameEngine engineSF = null;

      // An engine that returns Q values for all moves is required.
      if (Params.PlayerArbiter != PlayerMode.Ceres && Params.PlayerArbiter != PlayerMode.LC0)
      {
        throw new NotImplementedException("Arbiter engine must be Ceres or LC0");
      };

      // Initialize all engine (in parallel for speed).
      Parallel.Invoke(
        () => engine1 = MakeEngine(Params.Player1, Params.NetworkID1, evaluatorDef1, p1, s1),
        () => engine2 = MakeEngine(Params.Player2, Params.NetworkID2, evaluatorDef2, p2, s2),
        () => engineOptimal = MakeEngine(Params.PlayerArbiter, Params.NetworkArbiterID, evaluatorDefOptimal, pOptimal, sOptimal),
        () => engineSF = Params.RunStockfishCrosscheck ? MakeEngine(PlayerMode.Stockfish14_1, null, null, default, default) : null);

      int threadCount = 0;
      foreach (Game game in Game.FromPGN(Params.PGNFileName))
      {
        foreach (PositionWithHistory pos in game.PositionsWithHistory)
        {
          if (shutdownRequested || countScored > Params.NumPositions)
          {
            return;
          }

          if (Params.PosFilter != null && !Params.PosFilter(pos))
          {
            continue;
          }

          // Skip some positions to make more varied/independent, and also based on gpu ID to vary across threads.
          const int SKIP_COUNT = 15;
          if ((threadCount++ % GPUIDs.Length != gpuID) || (pos.FinalPosition.FEN.GetHashCode() % SKIP_COUNT != 0))
          {
            continue;
          }

          // Do not allow repeate positions to be processed.
          ulong posHash = pos.FinalPosition.CalcZobristHash(PositionMiscInfo.HashMove50Mode.ValueBoolIfAbove98);
          if (seenPositions.ContainsKey(posHash))
          {
            continue;
          }
          seenPositions[posHash] = true;

          if (pos.FinalPosition.CalcTerminalStatus() != GameResult.Unknown
            || pos.FinalPosition.CheckDrawCanBeClaimed == Position.PositionDrawStatus.DrawCanBeClaimed) continue;

          engine1.ResetGame();
          engine2.ResetGame();

          // Search with first engine.
          GameEngineSearchResult search1 = engine1.Search(pos, Limit * Params.Engine1LimitMultiplier);

          // Skip comparison if position is totally won/lost
          // (rarely are mistakes found here, and differences may be spurious
          // due to distance to mate encoding).
          if (MathF.Abs(search1.ScoreQ) > 0.85f)
          {
            continue;
          }

          GameEngineSearchResult search2 = engine2.Search(pos, Limit);

          if (search1.FinalN < 1 || search2.FinalN < 1) continue;

          MCTSNode root1;
          MGMove move1;
          GetBestMoveAndNode(pos, search1, out root1, out move1);

          MCTSNode root2;
          MGMove move2;
          GetBestMoveAndNode(pos, search2, out root2, out move2);

          countScored++;
          timeAccumulatorEngine1 += (float)search1.TimingStats.ElapsedTimeSecs;
          timeAccumulatorEngine2 += (float)search2.TimingStats.ElapsedTimeSecs;
          if (move1 == move2)
          {
            // Move agreement, no need to compare against long search.
            continue;
          }

          countDifferentMoves++;

          // Run a long search using arbiter to determine Q values associated with each possible move.
          engineOptimal.ResetGame();
          SearchLimit arbiterLimit = Limit * Params.EngineArbiterLimitMultiplier;
          if (arbiterLimit.IsNodesLimit && arbiterLimit.Value < 1000)
          {
            throw new Exception("Arbiter engine limit must be at least 1000 nodes");
          }
          GameEngineSearchResult searchBaselineLong = engineOptimal.Search(pos, arbiterLimit);
          if (searchBaselineLong.FinalN < 1)
          {
            continue;
          }

          VerboseMoveStat FindMove(MGMove moveMG)
          {
            Move move = MGMoveConverter.ToMove(moveMG);
            foreach (VerboseMoveStat ve in searchBaselineLong.VerboseMoveStats)
            {
              if (ve.MoveString != "node" && ve.Move == move)
              {
                return ve;
              }
            }
            return default;
          }

          float scoreBestMove1 = default;
          float scoreBestMove2 = default;
          if (searchBaselineLong is GameEngineSearchResultCeres)
          {
            GameEngineSearchResultCeres searchBaselineLongCeres = searchBaselineLong as GameEngineSearchResultCeres;

            // Determine how much better engine1 was versus engine2 according to the long search
            using (new SearchContextExecutionBlock(searchBaselineLongCeres.Search.Manager.Context))
            {
              MCTSNode bestMoveFrom1 = searchBaselineLongCeres.Search.SearchRootNode.FollowMovesToNode(new MGMove[] { move1 });
              scoreBestMove1 = (float)-bestMoveFrom1.Q;

              MCTSNode bestMoveFrom2 = searchBaselineLongCeres.Search.SearchRootNode.FollowMovesToNode(new MGMove[] { move2 });
              scoreBestMove2 = (float)-bestMoveFrom2.Q;
            }
          }
          else
          {
            VerboseMoveStat statMove1 = FindMove(move1);
            VerboseMoveStat statMove2 = FindMove(move2);
            if (statMove1 == default || statMove2 == default)
            {
              continue;
            }
            scoreBestMove1 = (float)statMove1.Q.LogisticValue;
            scoreBestMove2 = (float)statMove2.Q.LogisticValue;
          }

          float[] overlaps = new float[7];
          if (root1 != default && root2 != default)
          {
            for (int i = 1; i < overlaps.Length; i++)
            {
              overlaps[i] = PctOverlapLevel(((GameEngineSearchResultCeres)search1).Search.Manager,
                                            ((GameEngineSearchResultCeres)search2).Search.Manager, root1, root2, i);
            }
          }


          // Determine how much better (worse) engine 1 move was compared to engine2.
          float diffFromBest = scoreBestMove1 - scoreBestMove2;
          qDiffs.Add(diffFromBest);

          // Suppress showing/counting difference if extremely small.
          const float THRESHOLD_DIFF = 0.02f;
          string diffStrfromBest = MathF.Abs(diffFromBest) < THRESHOLD_DIFF ? "      " : $"{diffFromBest,6:F2}";
          if (diffFromBest > THRESHOLD_DIFF)
          {
            countMuchBetter++;
          }
          else if (diffFromBest < -THRESHOLD_DIFF)
          {
            countMuchWorse++;
          }

          GameEngineSearchResult resultSF = null;
          if (Params.RunStockfishCrosscheck && MathF.Abs(diffFromBest) > THRESHOLD_DIFF)
          {
            const int SF_NODES_MULTIPLIER = 750;
            SearchLimit sfLimit = Limit * Params.EngineArbiterLimitMultiplier * (Limit.IsNodesLimit ? SF_NODES_MULTIPLIER : 1); 
            resultSF = engineSF.Search(pos, sfLimit);
          }

          accOverlapDepth6 += overlaps[6];

          if (Params.Verbose)
          {
            WriteColumnHeaders();

            Move moveSF = resultSF == null ? default : Move.FromUCI(resultSF.MoveString);
            string sfMoveStr = "";
            if (resultSF != null)
            {
              sfMoveStr = moveSF.ToSAN(pos.FinalPosition);
            }
            Move bestMove = diffFromBest > 0 ? MGMoveConverter.ToMove(move1) : MGMoveConverter.ToMove(move2);
            string overlapst(int i) => MathF.Abs(overlaps[i]) < 0.99 ? $"{overlaps[i],6:F2}" : "      ";
            string moveStr1 = MGMoveConverter.ToMove(move1).ToSAN(pos.FinalPosition);
            string moveStr2 = MGMoveConverter.ToMove(move2).ToSAN(pos.FinalPosition);
            bool sfAgrees = sfMoveStr == "" || moveSF == bestMove;
            string sfDisagreeChar = sfAgrees ? " " : "?";

            CompareEnginePosResult posResult = new(gpuID, countScored, (float)countDifferentMoves / countScored,
                                                   (float)search1.TimingStats.ElapsedTimeSecs, (float)search2.TimingStats.ElapsedTimeSecs, 
                                                   search1.FinalN, search2.FinalN,
                                                   countMuchBetter, countMuchWorse, scoreBestMove1, diffFromBest,
                                                   sfAgrees, moveStr1, moveStr2, sfMoveStr, pos, 
                                                   search1.VerboseMoveStats, search2.VerboseMoveStats, searchBaselineLong.VerboseMoveStats);
            Params.PosResultCallback?.Invoke(posResult);

            Console.WriteLine($" {gpuID,4}  {countScored,6:N0}    {100.0f * (float)countDifferentMoves / countScored,6:F2}%   "
                            + $"{ search1.TimingStats.ElapsedTimeSecs,5:F2}   { search2.TimingStats.ElapsedTimeSecs,5:F2}    "
                            + $"{ search1.FinalN,12:N0}  {search2.FinalN,12:N0}  "
                            + $"  {countMuchBetter,5:N0} {countMuchWorse,5:N0}    {scoreBestMove1,5:F2}   {diffStrfromBest} {sfDisagreeChar}  "
                            + $"  {moveStr1,7}  {moveStr2,7}  {sfMoveStr,7} "
                            + $"  {pos.FinalPosition.FEN}");
          }

        }
      }
    }


    private static void GetBestMoveAndNode(PositionWithHistory pos, GameEngineSearchResult search1, out MCTSNode root1, out MGMove move1)
    {
      root1 = default;
      if (search1 is GameEngineSearchResultCeres)
      {
        GameEngineSearchResultCeres searchCeres = (GameEngineSearchResultCeres)search1;
        using (new SearchContextExecutionBlock(searchCeres.Search.Manager.Context))
        {
          root1 = searchCeres.Search.SearchRootNode;
          move1 = root1.BestMoveInfo(false).BestMove;
        }
      }
      else
      {
        Move move = Move.FromUCI(search1.MoveString);
        move1 = MGMoveConverter.MGMoveFromPosAndMove(pos.FinalPosition, move);
      }
    }

    static float PctOverlapLevel(MCTSManager manager1, MCTSManager manager2, MCTSNode node1, MCTSNode node2, int depth)
    {
      MCTSNode largerNode = node1.N > node2.N ? node1 : node2;
      MCTSNode smallerNode = node1.N > node2.N ? node2 : node1;

      HashSet<ulong> indices = new();

      int startDepth;

      using (new SearchContextExecutionBlock(node1.N > node2.N ? manager1.Context : manager2.Context))
      {
        startDepth = largerNode.Depth;
        largerNode.StructRef.Traverse(largerNode.Context.Tree.Store,
        (ref MCTSNodeStruct node) =>
        {
          if (node.DepthInTree > startDepth + depth)
            return false;

          indices.Add(node.ZobristHash);
          return true;
        }, TreeTraversalType.DepthFirst);
      }

      int countFound = 0;
      int countNotFound = 0;

      using (new SearchContextExecutionBlock(node1.N > node2.N ? manager2.Context : manager1.Context))
      {
        smallerNode.StructRef.Traverse(smallerNode.Context.Tree.Store,
      (ref MCTSNodeStruct node) =>
      {
        if (node.DepthInTree > startDepth + depth)
          return false;

        if (indices.Contains(node.ZobristHash))
          countFound++;
        else
          countNotFound++;

        return true;
      }, TreeTraversalType.DepthFirst);
      }

      float fracOverlap = (float)countFound / (countNotFound + countFound);
      return fracOverlap;
    }

    #region Header/summary output

    bool columnHeadersWritten = false;

    void WriteColumnHeaders()
    {
      if (!columnHeadersWritten)
      {
        columnHeadersWritten = true;
        Console.WriteLine();
        Console.WriteLine("  GPU    Pos#      %diff    Time1   Time2      Nodes1        Nodes2        Cnt1  Cnt2      Q1     QDiff        Move1    Move2  MoveSF    FEN");
        Console.WriteLine("  ---   -----     ------    -----  ------   ------------  ------------     ----  ----    -----    -----       ------   ------  -------   ------------------------------------------------------------------");
      }
    }

    void WriteIntroBanner()
    {
      Console.WriteLine();
      Console.WriteLine("Engine Comparision Tool - Compare two engines versus optimal engine with deeper search.");
      Console.WriteLine($"  Description     { Params.Description}");
      Console.WriteLine($"  Engine 1        { Params.Player1 } { Params.NetworkID1}");
      Console.WriteLine($"  Engine 2        { Params.Player2 } { Params.NetworkID2}");
      Console.WriteLine($"  Arbiter Engine  { Params.PlayerArbiter } { Params.NetworkArbiterID} ({Params.EngineArbiterLimitMultiplier}x)");
      Console.WriteLine($"  Num Positions   { Params.NumPositions}");
      Console.WriteLine($"  Limit           { Limit}");

      Console.WriteLine();
    }

    CompareEngineResultSummary ProcessSummaryInfo()
    {
      float avg = StatUtils.Average(qDiffs.ToArray());
      float sd = (float)StatUtils.StdDev(qDiffs.ToArray()) / MathF.Sqrt(qDiffs.Count);
      float z = avg / sd;

      CompareEngineResultSummary summary = new((float)timingStats.ElapsedTimeSecs, timeAccumulatorEngine1, timeAccumulatorEngine2,
                                               countScored, countDifferentMoves, avg, sd, z, countMuchBetter, countMuchWorse);

      Console.WriteLine($"CompareEngine done in {timingStats.ElapsedTimeSecs,7:F2}seconds");
      Console.WriteLine($"{Params.Description,20} {Params.NumPositions,6:N0} {ShortID1,12}  {ShortID2,12} {ShortIDArbiter,12}  {Limit.ToString(),10}  "
                      + $"{timeAccumulatorEngine1 / countScored,6:F3}s  {timeAccumulatorEngine2 / countScored,6:F3}s  "
                      + $" {100.0f * (float)countDifferentMoves / countScored,6:F2}% diff  {avg,6:F3} +/-{sd,5:F3} z= {z,5:F2}  "
                      + $" {100.0f * accOverlapDepth6 / countScored,6:F2}%  {countMuchBetter,6:N0} {countMuchWorse,6:N0}");
      return summary;
    }


    #endregion
  }

  public static class DeepEvalResults
  {
    static Dictionary<ulong, DeepEvalResult> resultsDict;
    static List<DeepEvalResult> results;
    public static void Load()
    {

    }

    public static void Save()
    {
      //SysMisc.WriteSpanToFile("x", results.ToArray().AsSpan());
      //SysMisc.ReadFileIntoSpan("x", )
      string FN = "poscache";
      File.Delete(FN);
      using (FileStream ms = new FileStream(FN, FileMode.CreateNew))
      {
        //        ms.Write(SerializationUtils.Serialize(results.ToArray()));

      }
    }

    public static void Add(MCTSNode node)
    {

    }

    public static float Lookup(ulong hash, MGMove move)
    {
      return default;

    }
  }
  [Serializable]
  public unsafe struct DeepEvalResult
  {
    public readonly long Hash;
    public fixed short Moves[64];
    public fixed float Q[64];

  }
}
