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
using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.Managers;
using Ceres.MCTS.Managers.Limits;
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
    public void DumpFullInfo(MGMove bestMove, MCTSNode searchRootNode, 
                             ManagerTreeReuse.ReuseDecision reuseDecision,
                             TimingStats makeNewRootTimingStats,
                             ManagerGameLimitInputs limitInputs,
                             TextWriter writer, string description)
    {
      try
      {
        using (new SearchContextExecutionBlock(this.Context))
        {
          searchRootNode = searchRootNode.IsNotNull ? searchRootNode : Root;
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

          if (reuseDecision != null)
          {
            writer.WriteLine("\r\nTREE REUSE DECISION");
            reuseDecision.Dump(writer);
            if (makeNewRootTimingStats != default)
            {
              writer.WriteLine($"  Tree Reuse Prep Time    { makeNewRootTimingStats.ElapsedTimeSecs,12:F4}s");
            }
            writer.WriteLine();
          }


          writer.WriteLine("\r\nLIMITS MANAGER DECISION");
          limitInputs?.Dump(writer);
          DumpTimeInfo(writer, searchRootNode);

          writer.WriteLine();
          searchRootNode.Dump(1, 1, writer: writer);

          writer.WriteLine();
          MCTSPosTreeNodeDumper.DumpPV(searchRootNode, true, writer);
        }
      }
      catch (Exception e)
      {
        Console.WriteLine($"Dump failed with message: {e.Message}");
        writer.Write($"Dump failed with message: {e.Message}");
        Console.WriteLine($"Stacktrace: {e.StackTrace}");
        writer.Write($"Stacktrace: {e.StackTrace}");
        //throw;
      }

    }


    /// <summary>
    /// Dumps information relating to the timing 
    /// and best moves selection from the last move made.
    /// </summary>    
    public void DumpTimeInfo(TextWriter writer = null, MCTSNode searchRootNode = default)
    {
      writer = writer ?? Console.Out;
      searchRootNode = searchRootNode.IsNotNull ? searchRootNode : Root;

      using (new SearchContextExecutionBlock(this.Context))
      {

        if (StopStatus != SearchStopStatus.Instamove
         && StopStatus != SearchStopStatus.TablebaseImmediateMove)
        {
          writer.WriteLine();
          writer.WriteLine($"StartTimeFirstVisit        {StartTimeFirstVisit}");
          writer.WriteLine($"StartTimeThisSearch        {StartTimeThisSearch}");
        }

        writer.WriteLine($"Root N                     {searchRootNode.N,14:N0}");
        writer.WriteLine($"RootNWhenSearchStarted     {RootNWhenSearchStarted,14:N0}");
        writer.WriteLine($"Store N                    {searchRootNode.Store.Nodes.NumUsedNodes,14:N0}");

        writer.WriteLine($"SearchLimitInitial.Type    {SearchLimitInitial.Type}");
        writer.WriteLine($"SearchLimitInitial.Value   {SearchLimitInitial.Value,14:N2}");
        writer.WriteLine($"SearchLimit.FracExtensible {SearchLimit.FractionExtensibleIfNeeded,14:N2}");
        writer.WriteLine($"FractionExtended           {searchRootNode.Context.Manager.FractionExtendedSoFar,14:F2}");
        writer.WriteLine($"Elapsed Search Time        {(DateTime.Now - StartTimeThisSearch).TotalSeconds,14:F2}");

        writer.WriteLine($"FractionSearchRemaining    {FractionSearchRemaining,14:F3}");
        writer.WriteLine($"Estimated NPS              {EstimatedNPS,14:N0}");

        writer.WriteLine($"EstimatedNumStepsRemaining {EstimatedNumVisitsRemaining(),14:N0}");
      }
    }
  }

}
