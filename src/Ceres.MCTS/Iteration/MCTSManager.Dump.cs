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
using System.Threading;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Analysis;
using Ceres.MCTS.Utils;

#endregion

namespace Ceres.MCTS.Iteration
{
  public partial class MCTSManager
  {
    /// <summary>
    /// Writes extensive descriptive information to a specified TextWriter,
    /// including verbose move statistics, principal variation, and move timing information.
    /// </summary>
    /// <param name="searchRootNode"></param>
    /// <param name="writer"></param>
    /// <param name="description"></param>
    public void DumpFullInfo(MGMove bestMove, MCTSNode searchRootNode = null, TextWriter writer = null, string description = null)
    {
      using (new SearchContextExecutionBlock(this.Context))
      {
        searchRootNode = searchRootNode ?? Root;
        writer = writer ?? Console.Out;

        int moveIndex = searchRootNode.Tree.Store.Nodes.PriorMoves.Moves.Count;

        writer.WriteLine();
        writer.WriteLine("=================================================================================");
        writer.Write(DateTime.Now + " SEARCH RESULT INFORMATION,  Move = " + ((1 + moveIndex / 2)));
        writer.WriteLine($" Thread = {Thread.CurrentThread.ManagedThreadId}");
        if (description != null)
        {
          writer.WriteLine(description);
        }

        writer.WriteLine();

        writer.WriteLine("Tree root           : " + Context.Root);
        if (searchRootNode != Root)
        {
          writer.WriteLine("Search root         : " + searchRootNode);
        }
        writer.WriteLine();

        MCTSNode[] nodesSortedN = null;
        MCTSNode[] nodesSortedQ = null;

        string bestMoveInfo = "";
        if (searchRootNode.NumChildrenExpanded > 0
         && StopStatus != SearchStopStatus.TablebaseImmediateMove
         && StopStatus != SearchStopStatus.OnlyOneLegalMove)
        {
          MCTSNode[] childrenSortedN = searchRootNode.ChildrenSorted(node => -node.N);
          MCTSNode[] childrenSortedQ = searchRootNode.ChildrenSorted(node => (float)node.Q);

          childrenSortedQ[0].Annotate();
          childrenSortedN[0].Annotate();

          bool isTopN = childrenSortedN[0].Annotation.PriorMoveMG == bestMove;
          bool isTopQ = childrenSortedQ[0].Annotation.PriorMoveMG == bestMove;
          if (isTopN && isTopQ)
          {
            bestMoveInfo = "(TopN and TopQ)";
          }
          else if (isTopN)
          {
            bestMoveInfo = "(TopN)";
          }
          else if (isTopQ)
          {
            bestMoveInfo = "(TopQ)";
          }
        }

        // Output position (with history) information.
        writer.WriteLine("Position            : " + searchRootNode.Annotation.Pos.FEN);
        writer.WriteLine("Tree root position  : " + Context.Tree.Store.Nodes.PriorMoves);
        writer.WriteLine("Search stop status  : " + StopStatus);
        writer.WriteLine("Best move selected  : " + bestMove.MoveStr(MGMoveNotationStyle.LC0Coordinate) + " " + bestMoveInfo);
        writer.WriteLine();

        string infoUpdate = UCIInfo.UCIInfoString(this, searchRootNode);
        writer.WriteLine(infoUpdate);

        writer.WriteLine();
        DumpTimeInfo(writer);

        writer.WriteLine();
        searchRootNode.Info.Dump(1, 1, writer: writer);

        writer.WriteLine();
        MCTSPosTreeNodeDumper.DumpPV(searchRootNode, true, writer);
      }
    }


    /// <summary>
    /// Dumps information relating to the timing 
    /// and best moves selection from the last move made.
    /// </summary>
    public void DumpTimeInfo(TextWriter writer = null)
    {
      writer = writer ?? Console.Out;

      if (StopStatus != SearchStopStatus.Instamove
       && StopStatus != SearchStopStatus.TablebaseImmediateMove)
      {
        writer.WriteLine();
        writer.WriteLine($"StartTimeFirstVisit        {StartTimeFirstVisit}");
        writer.WriteLine($"StartTimeThisSearch        {StartTimeThisSearch}");
      }

      writer.WriteLine($"Root N                     {Root.N,14:N0}");
      writer.WriteLine($"RootNWhenSearchStarted     {RootNWhenSearchStarted,14:N0}");

      writer.WriteLine($"SearchLimit.Type           {SearchLimit.Type}");
      writer.WriteLine($"SearchLimit.Value          {SearchLimit.Value,14:N2}");
      writer.WriteLine($"Elapsed Search Time        {(DateTime.Now - StartTimeThisSearch).TotalSeconds,14:F2}");

      writer.WriteLine($"FractionSearchRemaining    {FractionSearchRemaining,14:F3}");
      writer.WriteLine($"Estimated NPS              {EstimatedNPS,14:N0}");

      writer.WriteLine($"EstimatedNumStepsRemaining {EstimatedNumVisitsRemaining(),14:N0}");
    }

  }

}
