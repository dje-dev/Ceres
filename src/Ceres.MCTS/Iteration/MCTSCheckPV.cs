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

using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.Positions;
using Ceres.MCTS.Environment;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.MCTS.NodeCache;
using Ceres.MCTS.Utils;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Checks the principal variation for sibling moves which
  /// were little visited for possible superiority.
  /// 
  /// </summary>
  public class MCTSCheckPV
  {
    const bool VERBOSE = true;

    public static PVCheckResult Check(MCTSearch Search, IMCTSNodeCache reuseNodeCache, int pvDepth)
    {
      SearchPrincipalVariation pv = new SearchPrincipalVariation(Search.Manager.Root);
      if (pv.Nodes.Count <= pvDepth)
      {
        return null;
      }

      MCTSNode pvRoot = pv.Nodes[pvDepth];

      if (pvRoot.NumChildrenExpanded == 0
       || pvRoot.Terminal.IsTerminal())
      {
        return null;
      }

      // Build the position history leading down the PV to this node.
      PositionWithHistory pwh = new PositionWithHistory(Search.Manager.Context.StartPosAndPriorMoves);
      for (int history = 1; history <= pvDepth; history++)
      {
        pwh.AppendMove(pv.Nodes[history].Annotation.PriorMoveMG);
      }

      if (pwh.FinalPosition.CheckDrawCanBeClaimed != Position.PositionDrawStatus.NotDraw)
      {
        return null;
      }

      int topRootN;
      MCTSNode pvRootBestMove;
      int pvRootN;

      topRootN = Search.Manager.Context.Root.N;
      pvRootN = pvRoot.N;
      pvRoot.Annotate();
      pvRootBestMove = pvRoot.ChildrenSorted(s => (float)s.Q)[0];

      if (pvRootN < 10 || pvRootN < Search.Manager.Root.N / 1000)
      {
        //Console.WriteLine($"stop {pvRoot.N} vs {Search.Manager.Root.N}");
        return null;
      }


      var newSearchMoves = new List<Move>();
      int totalNInSearchable = 0;
      foreach (var child in pvRoot.ChildrenSorted(n => n.N))
      {
        child.Annotate();

        // Add to search moves if has received few visits so far.
        const float MAX_N_FRAC = 0.03f;
        if (child.N < pvRoot.N * MAX_N_FRAC)
        {
          newSearchMoves.Add(MGMoveConverter.ToMove(child.Annotation.PriorMoveMG));
          totalNInSearchable += child.N;
        }
      }

      if (totalNInSearchable == 0)
      {
        return null;
      }

      MCTSearch search1 = new MCTSearch();

      // Determine how many additional nodes to use in this search.
      const float N_MULTIPLIER = 2f;
      int maxNewNodes = (int)(MathF.Max(30, totalNInSearchable * N_MULTIPLIER));

      MCTSEventSource.TestCounter1 += maxNewNodes;
      SearchLimit newLimit = Search.Manager.SearchLimit with
      {
        Type = SearchLimitType.NodesPerMove,
        Value = maxNewNodes,
        SearchMoves = newSearchMoves,
        MaxTreeNodes = maxNewNodes + 3000
      };

      MCTSNode priorBestMove;
      float priorBestQ;
      int priorBestN;
      MGMove priorBestMoveMG;
      int priorRootDepth;
      priorRootDepth = pvRoot.Depth;
      priorBestMove = pvRoot.ChildrenSorted(s => (float)s.Q)[0];
      priorBestQ = (float)priorBestMove.Q;
      priorBestN = priorBestMove.N;
      priorBestMoveMG = priorBestMove.Annotation.PriorMoveMG;


      // Possible callback to abort search if no moves look
      // promising relative to best Q already at hand.
      void Callback(MCTSManager manager)
      {
#if NOT
          if (manager.Context.Root.N > 100)
          {
            MCTSNode bestMoveSoFar = manager.Context.Root.ChildrenSorted(s => (float)s.Q)[0];
            float fracDone = (float)manager.Context.Root.N / newLimit.Value;
            float distFromBest = (float)priorBestQ - (float)bestMoveSoFar.Q;
            if (fracDone > 0.5f && distFromBest > 0.05)
            {
              Console.WriteLine(fracDone + "  abort " + bestMoveSoFar.Q + " vs " + priorBestQ);
              manager.ExternalStopRequested = true;
            }
          }
#endif
      }

      TimingStats timingStats = new TimingStats();
      using (new TimingBlock(timingStats, TimingBlock.LoggingType.None))
      {
        search1.Search(Search.Manager.Context.NNEvaluators,
                     Search.Manager.Context.ParamsSelect with { CPUCTAtRoot = Search.Manager.Context.ParamsSelect.CPUCTAtRoot * 100 },
                     Search.Manager.Context.ParamsSearch with { FutilityPruningStopSearchEnabled = false, MoveFutilityPruningAggressiveness = 0 }, null,
                     Search.Manager.Context, /*reuseOtherContextForEvaluatedNodes,*/
                     pwh, newLimit, false, DateTime.Now,
                     null, Callback, null, reuseNodeCache, false, false,
                     true);
      }


      PVCheckResult checkResult = new(Search, pvRoot, priorBestMoveMG, (int)newLimit.Value, (float)timingStats.ElapsedTimeSecs);

      MCTSNode newBestMove = search1.Manager.Root.ChildrenSorted(s => search1.Manager.Context.RootMovesPruningStatus[s.IndexInParentsChildren] == MCTSFutilityPruningStatus.PrunedDueToSearchMoves ? float.MaxValue : (float)s.Q)[0];
      priorBestQ *= -1f;

      // Only report moves which appear much better than prior best Q.
      const float THRESHOLD = 0.05f;
      float newBestQ = -(float)newBestMove.Q;
      float pessimisticNewBestQ = newBestQ - THRESHOLD;
      if (pessimisticNewBestQ > priorBestQ)
      {
        if (VERBOSE)
        {
          //            pvRoot.Dump(pvRoot.Depth + 1, 100);
          Console.WriteLine();

          //            search1.Manager.DumpRootMoveStatistics();

          Console.WriteLine($"\r\nPV DEVIATION at depth { priorRootDepth} with ExtraN= {search1.Manager.Root.N} vs original {totalNInSearchable}/{pvRoot.N} in a tree of total size {topRootN} "
                            + " from root position ");// + Search.SearchRootNode.Annotation.Pos.FEN);
          Console.WriteLine("  Position: " + pwh);
          Console.WriteLine("  Prior best: " + priorBestQ + " " + priorBestN + " " + priorBestMoveMG);
          Console.WriteLine("  New best: " + newBestQ + " " + newBestMove.N + " " + newBestMove.Annotation.PriorMoveMG + " with policy " + newBestMove.P);
        }

        //newBestMove.Parent.InfoRef.BoostNodeIndex = new MCTSNodeStructIndex(newBestMove.Parent.Index);
        //newBestMove.Parent.InfoRef.BoostNodeChildIndex = (byte)newBestMove.IndexInParentsChildren;
        //newBestMove.Parent.InfoRef.BoostNodeChildStopN = newBestMove.N;

        checkResult.AddBetterMove(newBestMove.Annotation.PriorMoveMG, newBestQ, newBestMove.N);
      }


      return checkResult;
    }


    public class PVCheckResult
    {
      public MCTSearch RootSearch;
      public MCTSNode PVNode;
      public MGMove PriorBestMove;

      public int ExtraN;
      public float CalcTimeSeconds;
      public List<(MGMove betterMove, float betterQ, int betterN)> BetterMoves;

      public void AddBetterMove(MGMove betterMove, float betterQ, int betterN)
      {
        if (BetterMoves == null)
        {
          BetterMoves = new();
        }

        BetterMoves.Add((betterMove, betterQ, betterN));
      }

      public void Dump()
      {
        // Eventually implement this, see code above which does Console.WriteLine for sample format.
      }

      public PVCheckResult(MCTSearch rootSearch, MCTSNode pVNode, MGMove priorBestMove, int extraN, float calcTimeSeconds)
      {
        RootSearch = rootSearch ?? throw new ArgumentNullException(nameof(rootSearch));
        PVNode = pVNode;
        PriorBestMove = priorBestMove;
        ExtraN = extraN;
        CalcTimeSeconds = calcTimeSeconds;
      }
    }
  }
}
