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
using System.Diagnostics;
using System.IO;
using System.Threading;

using Ceres.Base.Benchmarking;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Managers.Limits;
using Ceres.MCGS.Search;
using Ceres.MCGS.Utils;


#endregion

namespace Ceres.MCGS.Search.Coordination;

public partial class MCGSManager
{
  /// <summary>
  /// Writes extensive descriptive information to a specified TextWriter,
  /// including verbose move statistics, principal variation, and move timing information.
  /// </summary>
  /// <param name="writer"></param>
  /// <param name="description"></param>
  public void DumpFullInfo(GameEngineSearchResultCeresMCGS searchResult, TextWriter writer, string description)
  {
    DumpFullInfo(searchResult.BestMoveInfo, searchResult.Search.SearchRootNode,
                 // TODO: fill this in Search.LastGameLimitInputs, 
                 default, writer, description);
  }


  /// <summary>
  /// <summary>
  /// Writes extensive descriptive information to a specified TextWriter,
  /// including verbose move statistics, principal variation, and move timing information.
  /// </summary>
  /// </summary>
  /// <param name="bestMove"></param>
  /// <param name="searchRootNode"></param>
  /// <param name="reuseDecision"></param>
  /// <param name="makeNewRootTimingStats"></param>
  /// <param name="limitInputs"></param>
  /// <param name="writer"></param>
  /// <param name="description"></param>
  public void DumpFullInfo(BestMoveInfoMCGS bestMoveInfo,
                           GNode searchRootNode,
                           ManagerGameLimitInputs limitInputs,
                           TextWriter writer, string description)
  {
    try
    {
      MGMove bestMove = bestMoveInfo.BestMove;

      searchRootNode = searchRootNode.IsNull ? searchRootNode : Engine.SearchRootNode;
      writer ??= Console.Out;

      int moveIndex = searchRootNode.Graph.Store.HistoryHashes.PriorPositionsMG.Length;

      writer.WriteLine();
      writer.WriteLine("=================================================================================");
      writer.Write(DateTime.Now + " MCGS SEARCH RESULT INFORMATION,  Move = " + ((1 + moveIndex / 2)));
      writer.WriteLine($" Thread = {Thread.CurrentThread.ManagedThreadId}");
      if (description != null)
      {
        writer.WriteLine(description);
      }

      writer.WriteLine();

      writer.WriteLine("Graph root           : " + Engine.SearchRootNode);
      if (searchRootNode != Engine.SearchRootNode)
      {
        writer.WriteLine("Search root         : " + searchRootNode);
      }
      writer.WriteLine();

      GEdge[] nodesSortedN = null;
      GEdge[] nodesSortedQ = null;

      string bestMoveInfoStr = "";
      if (searchRootNode.NumEdgesExpanded > 0
       && StopStatus != SearchStopStatus.TablebaseImmediateMove
       && StopStatus != SearchStopStatus.OnlyOneLegalMove)
      {
        GEdge[] childrenSortedN = searchRootNode.EdgesSorted(node => -node.N);
        GEdge[] childrenSortedQ = searchRootNode.EdgesSorted(node => (float)node.Q);

        GEdge bestMoveEdge = searchRootNode.EdgeForMove(bestMove);
        GNode bestMoveNode = bestMoveEdge.ChildNode;

        bool isTopN = childrenSortedN[0].N == bestMoveNode.N; // could be ties
        bool isTopQ = childrenSortedQ[0].MoveMG == bestMove;
        if (isTopN && isTopQ)
        {
          bestMoveInfoStr = "(TopN and TopQ)";
        }
        else if (isTopN)
        {
          bestMoveInfoStr = "(TopN)";
        }
        else if (isTopQ)
        {
          bestMoveInfoStr = "(TopQ)";
        }
      }

      // Output position (with history) information.
      writer.WriteLine("Position            : " + searchRootNode.CalcPosition().ToPosition.FEN);
      writer.WriteLine("Tree root position  : " + this.Engine.Graph.Store.NodesStore.PositionHistory);
      writer.WriteLine("Search stop status  : " + StopStatus);
      writer.WriteLine("Best move selected  : " + bestMove.MoveStr(MGMoveNotationStyle.Coordinates) + " " + bestMoveInfoStr);
      writer.WriteLine();

      string infoUpdate = UCIInfoMCGS.UCIInfoString(this);
      writer.WriteLine(infoUpdate);

      writer.WriteLine("\r\nLIMITS MANAGER DECISION");
      limitInputs?.Dump(writer);
      DumpTimeInfo(writer, searchRootNode);

      // TODO: next block
      writer.WriteLine();
      DumpNodeEdges(0, Engine.SearchRootNode, writer);

      writer.WriteLine();
      MCGSPosGraphNodeDumper.DumpPV(this, searchRootNode, true, writer);
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


  public void DumpRootTopMoves() => DumpNodeEdges(0, Engine.SearchRootNode, Console.Out);

  public void DumpNodeEdges(int depth, GNode node, TextWriter writer)
  {
    MCGSPosGraphNodeDumper.WriteHeaders(true, writer);

    MCGSPosGraphNodeDumper.DumpNodeStr(this, depth, Engine.SearchRootNode, default, node, default, 0, true, writer);
    foreach (GEdge edge in node.EdgesSorted(e=>-e.N))
    {
      MCGSPosGraphNodeDumper.DumpNodeStr(this,depth,  Engine.SearchRootNode, node, edge.ChildNode, edge, 0, true, writer);
    }
  }



  /// <summary>
  /// Dumps information relating to the timing 
  /// and best moves selection from the last move made.
  /// </summary>    
  public void DumpTimeInfo(TextWriter writer = null, GNode searchRootNode = default)
  {
    writer ??= Console.Out;
    searchRootNode = !searchRootNode.IsNull ? searchRootNode : Engine.SearchRootNode;

    if (StopStatus != SearchStopStatus.Instamove
     && StopStatus != SearchStopStatus.TablebaseImmediateMove)
    {
      writer.WriteLine();
      writer.WriteLine($"StartTimeFirstVisit        {StartTimeFirstVisit}");
      writer.WriteLine($"StartTimeThisSearch        {StartTimeThisSearch}");
    }

    /// <summary>
    /// Returns the number of transpositions of various types being tracked in the graph.
    /// </summary>
    (int countTranspositionAndSequence, int countTranspositionStandlone) = searchRootNode.Graph.TranspositionCounts;


    writer.WriteLine($"Root N                      {searchRootNode.N,14:N0}");
    writer.WriteLine($"RootNWhenSearchStarted      {RootNWhenSearchStarted,14:N0}");
    writer.WriteLine($"Store N                     {searchRootNode.GraphStore.NodesStore.NumUsedNodes,14:N0}");
    writer.WriteLine($"Transpositions (Sequence)   {countTranspositionAndSequence,14:N0}");
    writer.WriteLine($"Transpositions (Standalone) {countTranspositionStandlone,14:N0}");
    writer.WriteLine($"NN evaluations              {searchRootNode.Graph.NNPositionEvaluationsCount,14:N0}");
    writer.WriteLine($"NN batches                  {searchRootNode.Graph.NNBatchesCount,14:N0}");
    writer.WriteLine($"NN batch size max           {searchRootNode.Graph.NNBatchSizeMax,14:N0}");
    writer.WriteLine();

    writer.WriteLine($"SearchLimitInitial.Type     {SearchLimitInitial.Type}");
    writer.WriteLine($"SearchLimitInitial.Value    {SearchLimitInitial.Value,14:N2}");
    writer.WriteLine($"SearchLimit.FracExtensible  {SearchLimit.FractionExtensibleIfNeeded,14:N2}");
    writer.WriteLine($"FractionExtended            {FractionExtendedSoFar,14:F2}");
    writer.WriteLine($"Elapsed Search Time         {(DateTime.Now - StartTimeThisSearch).TotalSeconds,14:F2}");

    writer.WriteLine($"FractionSearchRemaining     {FractionSearchRemaining,14:F3}");
    writer.WriteLine($"Estimated NPS               {EstimatedNPS,14:N0}");

    writer.WriteLine($"EstimatedNumStepsRemaining  {EstimatedNumVisitsRemaining(),14:N0}");
  }
}
