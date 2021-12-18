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
using System.Threading.Tasks;


using Ceres.Base.DataType.Trees;
using Ceres.Base.Math;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;
using Ceres.Chess.UserSettings;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.Params;
using Ceres.Features.GameEngines;
using System.IO;

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
    static string PGN_PATH = CeresUserSettingsManager.Settings.DirPGN;
    readonly static string[] TEST_PGN = new string[]
    {
      Path.Combine(PGN_PATH, "players", "Korchnoi.pgn"),
      Path.Combine(PGN_PATH, "players","Karpov.pgn"),
      Path.Combine(PGN_PATH, "players","Anand.pgn"),
      Path.Combine(PGN_PATH, "players","Kasparov.pgn")
    };


    const int LONG_SEARCH_MULTIPLIER = 5;

    public CompareEnginesVersusOptimal(string desc, int numPositions, string networkID, SearchLimit searchLimit,
                                       Action<ParamsSearch> searchModifier, Action<ParamsSelect> selectModifier,
                                       bool verbose = false)
    {
      Desc = desc;
      NumPositions = numPositions;
      NetworkID = networkID;
      Limit = searchLimit;
      SearchModifier = searchModifier;
      SelectModifier = selectModifier;
      Verbose = verbose;
    }

    public readonly string Desc;
    public readonly int NumPositions;
    public readonly string NetworkID;
    public readonly SearchLimit Limit;
    public readonly Action<ParamsSearch> SearchModifier;
    public readonly Action<ParamsSelect> SelectModifier;
    public bool Verbose;


    List<float> bigDiffs = new();

    int count = 0;
    int countBetter = 0;
    int countWorse = 0;
    int countScored = 0;
    float accOverlapDepth6 = 0;

    public void Summarize()
    {
      float avg = StatUtils.Average(bigDiffs.ToArray());
      float sd = (float)StatUtils.StdDev(bigDiffs.ToArray());
      float z = (avg / (sd / MathF.Sqrt(bigDiffs.Count)));

      Console.WriteLine($"{Desc,20} {NumPositions,6:N0} {NetworkID,12}  {Limit.ToString(),10}  {avg,6:F3} +/-{sd,5:F3} z= {z,5:F2}  "
                      + $" {100.0f * accOverlapDepth6 / countScored,6:F2}%  {countBetter,6:N0} {countWorse,6:N0}");
    }

    public void Run()
    {
      // Create default parameters, with smart pruning tuned off.
      ParamsSearch p1 = new ParamsSearch()
      {
        FutilityPruningStopSearchEnabled = false,
      };
      ParamsSearch p2 = new ParamsSearch()
      {
        FutilityPruningStopSearchEnabled = false,
      };

      ParamsSelect s1 = new ParamsSelect()
      {
      };
      ParamsSelect s2 = new ParamsSelect()
      {

      };

      SearchModifier?.Invoke(p1);
      SelectModifier?.Invoke(s1);


      Parallel.ForEach(new int[] { 0, 1, 2, 3 },
        delegate (int i)
        {
          RunCompareThread(TEST_PGN[i], NetworkID, i, p1, p2, s1, s2);
        });

      Summarize();
    }


    private void RunCompareThread(string pgn, string networkID, int gpuID,
                                  ParamsSearch p1, ParamsSearch p2, ParamsSelect s1, ParamsSelect s2)
    {
      NNEvaluatorDef evaluatorDef = NNEvaluatorDef.FromSpecification(networkID, $"GPU:{gpuID}");

      GameEngineCeresInProcess engine1 = null;
      GameEngineCeresInProcess engine2 = null;

      Parallel.Invoke(
        () => engine1 = new GameEngineCeresInProcess("Ceres", evaluatorDef, null, p1, s1),
        () => engine2 = new GameEngineCeresInProcess("Ceres", evaluatorDef, null, p2, s2));

      foreach (Game game in Game.FromPGN(pgn))
      {
        foreach (PositionWithHistory pos in game.PositionsWithHistory)
        {
          if (countScored > NumPositions)
          {
            return;
          }

          // Skip some positions to make more varied/independent
          const int SKIP_PLY_COUNT = 5;
          if (count++ % SKIP_PLY_COUNT != SKIP_PLY_COUNT - 1)
          {
            continue;
          }

          if (pos.FinalPosition.CalcTerminalStatus() != GameResult.Unknown) continue;

          countScored++;
          engine1.ResetGame();
          engine2.ResetGame();

          GameEngineSearchResultCeres search1 = engine1.SearchCeres(pos, Limit);
          GameEngineSearchResultCeres search2 = engine2.SearchCeres(pos, Limit);

          if (search1.FinalN <= 1 || search2.FinalN <= 1) continue;

          MCTSNode root1;
          MGMove move1;
          MCTSNode root2;
          MGMove move2;
          using (new SearchContextExecutionBlock(search1.Search.Manager.Context))
          {
            root1 = search1.Search.SearchRootNode;
            move1 = root1.BestMoveInfo(false).BestMove;
          }
          using (new SearchContextExecutionBlock(search2.Search.Manager.Context))
          {
            root2 = search2.Search.SearchRootNode;
            move2 = root2.BestMoveInfo(false).BestMove;
          }

          // Run a long search
          engine2.ResetGame();
          GameEngineSearchResultCeres searchBaselineLong = engine2.SearchCeres(pos, Limit * LONG_SEARCH_MULTIPLIER);
          if (searchBaselineLong.FinalN <= 1) continue;

          // Determine how much better engine1 was versus engine2 according to the long search
          float scoreBestMove1;
          float scoreBestMove2;
          using (new SearchContextExecutionBlock(searchBaselineLong.Search.Manager.Context))
          {
            var bestMoveFrom1 = searchBaselineLong.Search.SearchRootNode.FollowMovesToNode(new MGMove[] { move1 });
            var bestMoveFrom2 = searchBaselineLong.Search.SearchRootNode.FollowMovesToNode(new MGMove[] { move2 });
            scoreBestMove1 = (float)-bestMoveFrom1.Q;
            scoreBestMove2 = (float)-bestMoveFrom2.Q;
          }

          float[] overlaps = new float[7];
          for (int i = 1; i < overlaps.Length; i++)
          {
            overlaps[i] = PctOverlapLevel(search1.Search.Manager, search2.Search.Manager, root1, root2, i);
          }

          float diff = (float)(search1.ScoreQ - search2.ScoreQ);
          bool sameMove = search1.BestMove.BestMove == search2.BestMove.BestMove;
          string diffStr = MathF.Abs(diff) < 0.005f ? "      " : $"{diff,6:F2}";

          // Check if engine1 had much better move than engine2
          float diffFromBest = scoreBestMove1 - scoreBestMove2;
          const float THREASHOLD_DIFF = 0.02f;
          string diffStrfromBest = MathF.Abs(diffFromBest) < THREASHOLD_DIFF ? "      " : $"{diffFromBest,6:F2}";
          if (diffFromBest > THREASHOLD_DIFF)
          {
            countBetter++;
            bigDiffs.Add(diffFromBest);
          }
          else if (diffFromBest < -THREASHOLD_DIFF)
          {
            countWorse++;
            bigDiffs.Add(diffFromBest);
          }

          accOverlapDepth6 += overlaps[6];

          if (Verbose)
          {
            string overlapst(int i) => MathF.Abs(overlaps[i]) < 0.99 ? $"{overlaps[i],6:F2}" : "      ";
            Console.WriteLine($" {gpuID,4}  {countScored,6:N0}   {100.0f * accOverlapDepth6 / countScored,6:F2}%  {countBetter,6:N0} {countWorse,6:N0}  {diffStrfromBest}  {(sameMove ? " " : "x")} {diffStr} "
                            + $" {move1,7}  {move2,7}  "
                            + $" {overlapst(1)}  {overlapst(2)}  {overlapst(3)}  {overlapst(4)}  {overlapst(5)}  {overlapst(6)}   {pos.FinalPosition.FEN}");
          }

          if (false && Math.Abs(diff) > 0.02 /*|| search1.BestMove.BestMove.ToString() != search2.BestMove.BestMove.ToString()*/)
          {
            Console.WriteLine(search1.ScoreQ + " " + search1.FinalN + " " + search1.BestMove);
            Console.WriteLine(search2.ScoreQ + " " + search2.FinalN + " " + search2.BestMove);
            Console.WriteLine(pos.ToString());
            Console.WriteLine();
            //  search1.Search.Manager.DumpRootMoveStatistics();
            // search2.Search.Manager.DumpRootMoveStatistics();

            Dictionary<(ulong, ulong), int> dict = new();

            if (false)
            {
              using (new SearchContextExecutionBlock(search1.Search.Manager.Context))
              {
                if (true)
                {
                  search1.Search.Manager.Context.Tree.Store.Dump(true);
                  search1.Search.Manager.Context.Tree.Root.MaterializeAllTranspositionLinks();
                  search1.Search.Manager.Context.Tree.Store.Dump(true);
                }
                search1.Search.Manager.Root.StructRef.Traverse(search1.Search.Manager.Context.Tree.Store,
                 (ref MCTSNodeStruct nodeRef) =>
                 {
                   if (!nodeRef.IsOldGeneration)
                   {
                     if (!nodeRef.IsRoot)
                     {
                       dict[(nodeRef.ZobristHash, nodeRef.ParentRef.ZobristHash)] = nodeRef.Index.Index;
                     }
                   }
                   return true;
                 }, TreeTraversalType.Sequential);

                //            search1.Search.Manager.Context.Tree.Store.Dump(true);
                //            search1.Search.Manager.Context.Tree.Store.Validate(null, false);
              }

              using (new SearchContextExecutionBlock(search2.Search.Manager.Context))
              {
                search2.Search.Manager.Context.Tree.Root.MaterializeAllTranspositionLinks();
                search2.Search.Manager.Root.StructRef.Traverse(search1.Search.Manager.Context.Tree.Store,
                 (ref MCTSNodeStruct nodeRef) =>
                 {
                   if (!nodeRef.IsOldGeneration)
                   {
                     if (!nodeRef.IsRoot)
                     {
                       bool found = dict.TryGetValue((nodeRef.ZobristHash, nodeRef.ParentRef.ZobristHash), out int otherIndex);
                       ref readonly MCTSNodeStruct otherNode = ref search1.Search.Manager.Context.Tree.Store.Nodes.nodes[otherIndex];
                       //                   if (found && Math.Abs(otherNode.Q - (float)nodeRef.Q) > 0.03f)
                       if (found && otherNode.N != nodeRef.N)
                       {
                         Console.WriteLine(nodeRef.DepthInTree + " " + nodeRef.V + " " + nodeRef.N + "   --> " + otherNode.V + " " + otherNode.N + " " +
                         nodeRef.NumVisitsPendingTranspositionRootExtraction + " " + otherNode.NumVisitsPendingTranspositionRootExtraction);
                       }
                     }
                   }
                   return true;
                 }, TreeTraversalType.Sequential);

                //            search2.Search.Manager.Context.Tree.Store.Dump(true);
                search2.Search.Manager.Context.Tree.Store.Validate(null, false);
              }
            }
          }

        }
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
}
