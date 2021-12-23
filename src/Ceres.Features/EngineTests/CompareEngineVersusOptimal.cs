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
using Ceres.Features.GameEngines;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading.Tasks;

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

    const bool DISABLE_PRUNING = true;
    const int LONG_SEARCH_MULTIPLIER = 7;

    public CompareEnginesVersusOptimal(string desc, string pgnFileName, int numPositions,
                                       PlayerMode player1, string networkID1,
                                       PlayerMode player2, string networkID2,
                                       PlayerMode playerArbiter, string networkArbiterID,
                                       SearchLimit searchLimit, int[] gpuIDs = null,
                                       Action<ParamsSearch> searchModifier1 = null, Action<ParamsSelect> selectModifier1 = null,
                                       Action<ParamsSearch> searchModifier2 = null, Action<ParamsSelect> selectModifier2 = null,
                                       bool verbose = true,
                                       float engine1LimitMultiplier = 1.0f)
    {
      Desc = desc;
      PGNFileName = pgnFileName;
      NumPositions = numPositions;
      Player1 = player1;
      NetworkID1 = networkID1;
      Player2 = player2;
      NetworkID2 = networkID2;
      PlayerArbiter = playerArbiter;
      NetworkArbiterID = networkArbiterID;
      GPUIDs = gpuIDs ?? new int[] { 0 };
      Limit = searchLimit with { SearchCanBeExpanded = false };
      SearchModifier1 = searchModifier1;
      SelectModifier1 = selectModifier1;
      SearchModifier2 = searchModifier2;
      SelectModifier2 = selectModifier2;
      Verbose = verbose;
      Engine1LimitMultiplier = engine1LimitMultiplier;
    }

    public readonly string Desc;
    public readonly string PGNFileName;
    public readonly int NumPositions;
    public readonly PlayerMode Player1;
    public readonly PlayerMode Player2;
    public readonly PlayerMode PlayerArbiter;
    public readonly string NetworkID1;
    public readonly string NetworkID2;
    public readonly string NetworkArbiterID;
    public readonly int[] GPUIDs;
    public readonly SearchLimit Limit;
    public readonly Action<ParamsSearch> SearchModifier1;
    public readonly Action<ParamsSelect> SelectModifier1;
    public readonly Action<ParamsSearch> SearchModifier2;
    public readonly Action<ParamsSelect> SelectModifier2;
    public bool Verbose;
    public float Engine1LimitMultiplier;

    List<float> qDiffs = new();

    int count = 0;
    int countMuchBetter = 0;
    int countMuchWorse = 0;
    int countScored = 0;
    float accOverlapDepth6 = 0;
    int countDifferentMoves = 0;

    ConcurrentDictionary<ulong, bool> seenPositions = new ();

    volatile bool shutdownRequested;

    float timeAccumulatorEngine1 = 0;
    float timeAccumulatorEngine2 = 0;

    public void Summarize()
    {
      float avg = StatUtils.Average(qDiffs.ToArray());
      float sd = (float)StatUtils.StdDev(qDiffs.ToArray()) / MathF.Sqrt(qDiffs.Count);
      float z = avg / sd;

      Console.WriteLine($"CompareEngine done in {timingStats.ElapsedTimeSecs,7:F2}seconds");
      Console.WriteLine($"{Desc,20} {NumPositions,6:N0} {ShortID1,12}  {ShortID2,12} {ShortIDArbiter,12}  {Limit.ToString(),10}  "
                      + $"{timeAccumulatorEngine1 / countScored,6:F3}s  {timeAccumulatorEngine2 / countScored,6:F3}s  "
                      + $" {100.0f * (float)countDifferentMoves / countScored,6:F2}% diff  {avg,6:F3} +/-{sd,5:F3} z= {z,5:F2}  "
                      + $" {100.0f * accOverlapDepth6 / countScored,6:F2}%  {countMuchBetter,6:N0} {countMuchWorse,6:N0}");
    }

    private string ShortID1 => Player1.ToString()[0] + "_" + NetworkID1;
    private string ShortID2 => Player2.ToString()[0] + "_" + NetworkID2;
    private string ShortIDArbiter => PlayerArbiter.ToString()[0] + "_" + NetworkArbiterID;


    void WriteIntroBanner()
    {
      Console.WriteLine();
      Console.WriteLine("Engine Comparision Tool - Compare two engines versus optimal engine with deeper search.");
      Console.WriteLine($"  Description     { Desc}");
      Console.WriteLine($"  Engine 1        { Player1 } { NetworkID1}");
      Console.WriteLine($"  Engine 2        { Player2 } { NetworkID2}");
      Console.WriteLine($"  Arbiter Engine  { PlayerArbiter } { NetworkArbiterID} ({LONG_SEARCH_MULTIPLIER}x)");
      Console.WriteLine($"  Num Positions   { NumPositions}");
      Console.WriteLine($"  Limit           { Limit}");

      Console.WriteLine();
    }

    ParamsSearch p1;
    ParamsSearch p2;
    ParamsSearch pArbiter;
    ParamsSelect s1;
    ParamsSelect s2;
    ParamsSelect sArbiter;
    TimingStats timingStats = new TimingStats();

    public void Run()
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
        FutilityPruningStopSearchEnabled = !DISABLE_PRUNING
      };
      ParamsSearch p2 = new ParamsSearch()
      {
        FutilityPruningStopSearchEnabled = !DISABLE_PRUNING
      };

      ParamsSelect s1 = new ParamsSelect();
      ParamsSelect s2 = new ParamsSelect();

      // A higher CPUCTAtRoot is used with arbiter to encourage to
      // get more accurate Q values across all moves
      // (including possibly inferior ones chosen by other engines).
      pArbiter = new ParamsSearch();
      sArbiter = new ParamsSelect() { CPUCTAtRoot = new ParamsSelect().CPUCTAtRoot * 3 };

      SearchModifier1?.Invoke(p1);
      SelectModifier1?.Invoke(s1);

      SearchModifier2?.Invoke(p2);
      SelectModifier2?.Invoke(s2);

      using (new TimingBlock(timingStats, TimingBlock.LoggingType.None))
      {
        Parallel.ForEach(GPUIDs,
          delegate (int i)
          {
            RunCompareThread(i, p1, p2, s1, s2, pArbiter, sArbiter);
          });
      }

      Summarize();
    }


    private void RunCompareThread(int gpuID,
                                  ParamsSearch p1, ParamsSearch p2,
                                  ParamsSelect s1, ParamsSelect s2,
                                  ParamsSearch pOptimal, ParamsSelect sOptimal)
    {
      NNEvaluatorDef evaluatorDef1 = NetworkID1 != null ? NNEvaluatorDef.FromSpecification(NetworkID1, $"GPU:{gpuID}") : null;
      NNEvaluatorDef evaluatorDef2 = NetworkID2 != null ? NNEvaluatorDef.FromSpecification(NetworkID2, $"GPU:{gpuID}") : null;
      NNEvaluatorDef evaluatorDefOptimal = NetworkArbiterID != null ? NNEvaluatorDef.FromSpecification(NetworkArbiterID, $"GPU:{gpuID}") : null;

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
          return new GameEngineLC0("LC0", networkID, DISABLE_PRUNING, false, paramsSearch, paramsSelect, evaluatorDef, verbose:true);
        }
        else if (playerMode == PlayerMode.Stockfish14_1)
        {
          string sf14FN = Path.Combine(CeresUserSettingsManager.Settings.DirExternalEngines, "stockfish14.1.exe");
          GameEngineDef engineDef = new GameEngineDefUCI("SF14", new GameEngineUCISpec("SF14", sf14FN, 8,
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

      if (PlayerArbiter != PlayerMode.Ceres && PlayerArbiter != PlayerMode.LC0)
      {
        throw new NotImplementedException("Currently arbiter engine must be Ceres or LC0");
      };

      Parallel.Invoke(
        () => engine1 = MakeEngine(Player1, NetworkID1, evaluatorDef1, p1, s1),
        () => engine2 = MakeEngine(Player2, NetworkID2, evaluatorDef2, p2, s2),
        () => engineOptimal = MakeEngine(PlayerArbiter, NetworkArbiterID, evaluatorDefOptimal, pOptimal, sOptimal));

      int threadCount = 0;
      foreach (Game game in Game.FromPGN(PGNFileName))
      {
        foreach (PositionWithHistory pos in game.PositionsWithHistory)
        {
          if (shutdownRequested || countScored > NumPositions)
          {
            return;
          }

          // Skip some positions to make more varied/independent, and also based on gpu ID to vary across threads.
          const int SKIP_COUNT = 17;
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

          count++;
          if (pos.FinalPosition.CalcTerminalStatus() != GameResult.Unknown
            || pos.FinalPosition.CheckDrawCanBeClaimed == Position.PositionDrawStatus.DrawCanBeClaimed) continue;

          countScored++;
          engine1.ResetGame();
          engine2.ResetGame();

          GameEngineSearchResult search1 = engine1.Search(pos, Limit * Engine1LimitMultiplier);
          GameEngineSearchResult search2 = engine2.Search(pos, Limit);

          timeAccumulatorEngine1 += (float)search1.TimingStats.ElapsedTimeSecs;
          timeAccumulatorEngine2 += (float)search2.TimingStats.ElapsedTimeSecs;

          if (search1.FinalN <= 1 || search2.FinalN <= 1) continue;

          MCTSNode root1;
          MGMove move1;
          GetBestMoveAndNode(pos, search1, out root1, out move1);

          MCTSNode root2;
          MGMove move2;
          GetBestMoveAndNode(pos, search2, out root2, out move2);

          if (move1 == move2)
          {
            // Move agreement, no need to compare against long search.
            continue;
          }

          countDifferentMoves++;

          // Run a long search
          engineOptimal.ResetGame();
          GameEngineSearchResult searchBaselineLong = engineOptimal.Search(pos, Limit * LONG_SEARCH_MULTIPLIER);
          if (searchBaselineLong.FinalN <= 1)
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
              var bestMoveFrom1 = searchBaselineLongCeres.Search.SearchRootNode.FollowMovesToNode(new MGMove[] { move1 });
              var bestMoveFrom2 = searchBaselineLongCeres.Search.SearchRootNode.FollowMovesToNode(new MGMove[] { move2 });
              scoreBestMove1 = (float)-bestMoveFrom1.Q;
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


          // Check if engine1 had much better move than engine2
          bool sameMove = move1 == move2;
          float diffFromBest = scoreBestMove1 - scoreBestMove2;
          qDiffs.Add(diffFromBest);

          const float THREASHOLD_DIFF = 0.015f;
          string diffStrfromBest = MathF.Abs(diffFromBest) < THREASHOLD_DIFF ? "      " : $"{diffFromBest,6:F2}";
          if (diffFromBest > THREASHOLD_DIFF)
          {
            countMuchBetter++;
          }
          else if (diffFromBest < -THREASHOLD_DIFF)
          {
            countMuchWorse++;
          }

          accOverlapDepth6 += overlaps[6];

          if (Verbose)
          {
            string overlapst(int i) => MathF.Abs(overlaps[i]) < 0.99 ? $"{overlaps[i],6:F2}" : "      ";
            Console.WriteLine($" {gpuID,4}  {countScored,6:N0}    {100.0f * (float)countDifferentMoves / countScored,6:F2}% diff   "
                            + $"{ search1.TimingStats.ElapsedTimeSecs,5:F2}s  { search2.TimingStats.ElapsedTimeSecs,5:F2}s  "
                            + $"  {countMuchBetter,5:N0} {countMuchWorse,5:N0}   {diffStrfromBest}  "
                            + $"  {move1,7}  {move2,7}  "
                            //                            + $" {overlapst(1)}  {overlapst(2)}  {overlapst(3)}  {overlapst(4)}  {overlapst(5)}  {overlapst(6)}" 
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
