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
using System.IO;
using System.Collections.Generic;

using Ceres.Chess;
using Ceres.Base.DataType.Trees;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes.Struct;
using Ceres.Chess.NetEvaluation.Batch;
using static Tensorflow.CostGraphDef;

#endregion

namespace Ceres.MCTS.MTCSNodes
{
  /// <summary>
  /// Utility methods for dumping information relating to the MCTS node,
  /// including ToString() for a single node and Dump to dump statistics
  /// relating to moves from root.
  /// </summary>
  public partial struct MCTSNode
  {
    public override string ToString()
    {
      string pendingWDL = "";
      if (!EvalResult.IsNull)
      {
        pendingWDL = $" PendingWDL={EvalResult.WinP:F3}/{EvalResult.DrawP:F3}/{EvalResult.LossP:F3}";
      }

      string priorMoveStr = Annotation.PriorMoveMG.ToString();

      return $"<MCTSNode [#{Index}] Depth {Depth} {priorMoveStr} [{ActionType}]  ({N},{NInFlight},{NInFlight2})  [{P * 100:F3}%] {Q:F3} "
           + $"Parent={(ParentIndex.IsNull ? "none" : ParentIndex.Index.ToString())}"
           + $" V={V:F3} " + (VSecondary == 0 ? "" : $"VSecondary={VSecondary:F3} ")
           + pendingWDL
           + $" MPos={MPosition:F3} MAvg={MAvg:F3} "
           //           + $" Score0={(IsRoot ? float.NaN : Parent.ChildScore(0, Depth, IndexInParentsChildren)),6:F3} "
           //           + $" Score1={(IsRoot ? float.NaN : Parent.ChildScore(1, Depth, IndexInParentsChildren)),6:F3} "
           + $" with {NumPolicyMoves} policy moves>";
    }


    /// <summary>
    /// Dumps detailed information about possible moves from this node
    /// and optionally from descendent nodes up to a specified depth.
    /// </summary>
    /// <param name="lastLevel"></param>
    /// <param name="firstLevelStartPVOnly"></param>
    /// <param name="minNodes"></param>
    /// <param name="prefixString"></param>
    /// <param name="shouldAbort"></param>
    public void Dump(int lastLevel = int.MaxValue,
                     int firstLevelStartPVOnly = int.MaxValue,
                     int minNodes = int.MinValue,
                     string prefixString = null,
                     Predicate<MCTSNode> shouldAbort = null,
                     int maxMoves = int.MaxValue,
                     TextWriter writer = null,
                     MCTSNode dumpRoot = default,
                     bool shortForm = false)
    {
      if (IsNull)
      {
        Console.WriteLine("<MCTSNode NULL>");
      }

      if (N < minNodes)
      {
        return;
      }

      if (shouldAbort != null && shouldAbort(this))
      {
        return;
      }

      Annotate();

      dumpRoot = dumpRoot.IsNull ? Tree.Root : dumpRoot;
      writer = writer ?? Console.Out;

      Position cPos = MGChessPositionConverter.PositionFromMGChessPosition(in Annotation.PosMG);

      float multiplerOurPerspective = dumpRoot.IsNull || (dumpRoot.Depth % 2 == Depth % 2) ? 1.0f : -1.0f;

      bool minimize = true;
      float bestChildValue = minimize ? int.MaxValue : int.MinValue;
      if (Parent.IsNotNull)
      {
        MCTSNode[] sortedChildren = Parent.ChildrenSorted(n => n.V);
        bestChildValue = minimize ? sortedChildren[0].V : sortedChildren[^1].V;
      }

      // Print extra characters for nodes with special characteristics
      char extraFlag = ' ';
      if (!IsRoot && Parent.IsRoot
        && Context.RootMovesPruningStatus != null
        && Context.RootMovesPruningStatus[IndexInParentsChildren] != MCTSFutilityPruningStatus.NotPruned)
      {
        MCTSFutilityPruningStatus status = Context.RootMovesPruningStatus[IndexInParentsChildren];
        extraFlag = status switch
        {
          MCTSFutilityPruningStatus.PrunedDueToFutility => 'F',
          MCTSFutilityPruningStatus.PrunedDueToTablebaseNotWinning => 'T',
          MCTSFutilityPruningStatus.PrunedDueToSearchMoves => 'S',
        };
      }

      if (Terminal == GameResult.Draw)
      {
        extraFlag = 'D';
      }
      else if (Terminal == GameResult.Checkmate)
      {
        extraFlag = 'C';
      }

      string extraInfo = prefixString;

      string fracVisitStr = "   ";
      if (Depth == 1 && Context?.RootMoveTracker.RunningFractionVisits != null)
      {
        float runningAvg = Context.RootMoveTracker.RunningFractionVisits[IndexInParentsChildren];
        fracVisitStr = $"{Math.Round(runningAvg * 100, 0),3:F0}";
      }

      string recentQAvgStr = "     ";
      if (Depth == 1 && Context?.RootMoveTracker.RunningVValues != null)
      {
        float runningQ = Context.RootMoveTracker.RunningVValues[IndexInParentsChildren];
        recentQAvgStr = $"{multiplerOurPerspective * runningQ,5:F2}";

      }

      float resampleAvg = float.NaN;
      float resampleSD = float.NaN;
      if (Context.ParamsSearch.ResamplingMoveSelectionFractionMove > 0)
      { 
        const int NUM_SAMPLES = 1024;
        (resampleAvg, resampleSD) = MCTSNodeResampling.GetResampledStats(this, NUM_SAMPLES, 1, Context.ParamsSearch.ResamplingMoveSelectionTemperature);
      }

      int lastN = 0;
      if (Depth == 1 && Context?.RootMoveTracker.LastRootN != null)
      {
        lastN = Context.RootMoveTracker.LastRootN[IndexInParentsChildren];
      }

      float[] childScores0 = Depth == 1 ? Parent.InfoRef.CalcChildScores(0, 0, 0, 1.0f) : null;
      float[] childScores1 = Depth == 1 && Tree.Context.ParamsSearch.Execution.FlowDualSelectors ? Parent.InfoRef.CalcChildScores(1, 0, 0, 1.0f) : null;

#if LEGACY_UNCERTAINTY
      float qUncertainty = StructRef.Uncertainty.UncertaintyAtN(N, 0);
#else
      float qUncertainty = StructRef.UncertaintyVPosition;
#endif
      bool invert = multiplerOurPerspective == -1;
      float u0 = childScores0 != null ? childScores0[IndexInParentsChildren] : 0;
      float u1 = childScores1 != null ? childScores1[IndexInParentsChildren] : 0;
      string uStr = "";
      if (!shortForm)
      {
        uStr = $"  U0={u0,5:F4}   U1={u1,5:F4} ";
      }

      extraInfo = $" N={N,11:F0}";
      if (!shortForm)
      {
        extraInfo += $"({fracVisitStr}% {recentQAvgStr}{lastN,11:F0})";
      }

      float actionV = 0;
      if (!IsRoot 
        && Context.NNEvaluators.Evaluator1.HasAction 
        && Context.Tree.Store.AllActionVectors != null)
      {
        CompressedActionVector actions = Context.Tree.Store.AllActionVectors[Parent.Index];
        int index = IndexInParentsChildren;
        actionV = actions[index].W - actions[index].L;
      }


      extraInfo += $"  A= {actionV,6:F3}  ";
      extraInfo += $" Q= {multiplerOurPerspective * Q,6:F3}  ";

      //      extraInfo += $" RSA= {-resampleAvg,6:F3}  ";
      //      extraInfo += $" RSS= {resampleSD,6:F3}  ";
      extraInfo += $"U= {qUncertainty,3:N0}  V= {  multiplerOurPerspective * V,6:F3} {uStr} ";


      extraInfo += $" WDL= {(invert ? LossP : WinP),4:F2} {DrawP,4:F2} {(invert ? WinP : LossP),4:F2} ";
      extraInfo += $" WDLA= {(invert ? LAvg : WAvg),4:F2} {DAvg,4:F2} {(invert ? WAvg : LAvg),4:F2}  ";
      extraInfo += $"M= {MPosition,4:F0} {MAvg,4:F0}  ";

      MGMove move = Annotation.PriorMoveMG;
      writer.WriteLine($"{extraFlag} {Depth,4:F0} {move.MoveStr(MGMoveNotationStyle.ShortAlgebraic),-6} {100 * P,8:F2}% " +
                       extraInfo);

      if (Depth < lastLevel && ExpandedChildrenList.Count > 0)
      {
        if (Depth >= firstLevelStartPVOnly)
        {
          MCTSNode[] sortedChildren2 = ChildrenSorted(n => -n.N);
          sortedChildren2[0].Dump(firstLevelStartPVOnly, lastLevel, minNodes, maxMoves: maxMoves,
                                  writer: writer, dumpRoot: dumpRoot, shortForm:shortForm);
        }
        else
        {
          // Make a copy of children and sort as desired 
          MCTSNode[] sortedChildren1 = ChildrenSorted(n => -n.N);
          string otherNodesMessage = null;
          if (sortedChildren1.Length > maxMoves)
          {
            otherNodesMessage = $"  (followed by {sortedChildren1.Length - maxMoves} additional moves not shown...)";
            Array.Resize(ref sortedChildren1, maxMoves);
          }

          foreach (MCTSNode child in sortedChildren1)
          {
            child.Dump(lastLevel, firstLevelStartPVOnly, minNodes, maxMoves: maxMoves, writer: writer, dumpRoot: dumpRoot, shortForm: shortForm);
          }

          if (otherNodesMessage != null)
          {
            writer.WriteLine(otherNodesMessage);
          }

          writer.WriteLine();
        }
      }
    }
  }

}
