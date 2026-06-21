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
using Ceres.Chess.NNEvaluators;
using Ceres.MCGS.GameEngines;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Managers.Limits;
using Ceres.MCGS.Search;
using Ceres.MCGS.Search.Phases;
using Ceres.MCGS.Utils;


#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Selects which section(s) of DumpFullInfo output are emitted. Used (for example) by the live
/// TCEC monitor to continuously update only one section at a time.
/// </summary>
[Flags]
public enum DumpFullInfoSections
{
  None = 0,
  Info = 1,
  Moves = 2,
  PrincipalVariation = 4,
  All = Info | Moves | PrincipalVariation
}

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
                           TextWriter writer, string description,
                           DumpFullInfoSections sections = DumpFullInfoSections.All)
  {
    try
    {
      MGMove bestMove = bestMoveInfo.BestMove;

      searchRootNode = searchRootNode.IsNull ? searchRootNode : Engine.SearchRootNode;
      writer ??= Console.Out;

      if ((sections & DumpFullInfoSections.Info) != 0)
      {
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

      // Print the actual graph root and the search root as distinct nodes. Under graph reuse the
      // search root descends below the graph root, so these are frequently different nodes.
      writer.WriteLine("Graph root           : " + Engine.Graph.GraphRootNode);
      writer.WriteLine("Search root          : " + Engine.SearchRootNode);
      if (!searchRootNode.IsNull && searchRootNode != Engine.SearchRootNode)
      {
        writer.WriteLine("Search root (arg)    : " + searchRootNode);
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
      // NOTE: Under graph reuse (especially Position / PositionEquivalence mode) the search root
      // descends BELOW the graph root by design, so "Position (search root)" and "Graph root
      // position" (an ancestor) legitimately differ. The "Search root path" line shows the moves
      // from the graph root down to the search root; its final FEN matches the "Position" line.
      int pliesBelowGraphRoot = Engine.SearchRootPathFromGraphRoot?.Length ?? 0;
      writer.WriteLine("Position (search root): " + searchRootNode.CalcPosition().ToPosition.FEN);
      writer.WriteLine($"Graph root position   : {this.Engine.Graph.Store.NodesStore.PositionHistory} "
                     + $"(ancestor, +{pliesBelowGraphRoot} plies above search root)");

      // Move sequence from the graph root down to the search root (final FEN matches "Position" above).
      System.Text.StringBuilder sbPath = new();
      sbPath.Append(Engine.Graph.Store.NodesStore.PositionHistory.FinalPosition.FEN);
      foreach (var pathInfo in Engine.SearchRootPathFromGraphRoot ?? [])
      {
        sbPath.Append("  " + pathInfo.MoveToChild.MoveStr(MGMoveNotationStyle.Coordinates)
                    + " -> " + pathInfo.ChildPosMG.ToPosition.FEN);
      }
      writer.WriteLine("Search root path      : " + sbPath.ToString());

      // Internal consistency of the root state (also logs a red diagnostic block above on mismatch).
      bool rootConsistent = MCGSRootConsistencyCheck.Validate(Engine, Engine.SearchRootPosMG.ToPosition, "dump", out _);
      writer.WriteLine("Root consistency      : " + (rootConsistent ? "OK" : "*** MISMATCH — see red diagnostic above ***"));
      writer.WriteLine("Search stop status  : " + StopStatus);
      writer.WriteLine("Best move selected  : " + bestMove.MoveStr(MGMoveNotationStyle.Coordinates) + " " + bestMoveInfoStr);
      writer.WriteLine();

      string infoUpdate = UCIInfoMCGS.UCIInfoString(this);
      writer.WriteLine(infoUpdate);

      writer.WriteLine("\r\nLIMITS MANAGER DECISION");
      limitInputs?.Dump(writer);
      DumpTimeInfo(writer, searchRootNode);
      }

      if ((sections & DumpFullInfoSections.Moves) != 0)
      {
        writer.WriteLine();
        DumpNodeEdges(0, Engine.SearchRootNode, writer);
      }

      if ((sections & DumpFullInfoSections.PrincipalVariation) != 0)
      {
        writer.WriteLine();
        MCGSPosGraphNodeDumper.DumpPV(this, searchRootNode, true, writer);
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


  public void DumpRootTopMoves() => DumpNodeEdges(0, Engine.SearchRootNode, Console.Out);

  public void DumpNodeEdges(int depth, GNode node, TextWriter writer)
  {
    MCGSPosGraphNodeDumper.WriteHeaders(true, writer);

    MCGSPosGraphNodeDumper.DumpNodeStr(this, depth, Engine.SearchRootNode, default, node, default, 0, true, writer);
    foreach (GEdge edge in node.EdgesSorted(e=>-e.N))
    {
      if (edge.ChildNode.IsNull)
      {
        // Terminal edge (checkmate/tablebase/drawn): no child node to dump.
        writer.WriteLine("  TERMINAL " + edge);
        continue;
      }

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
    if (EvaluatorNN0 != null && EvaluatorNN0.Stats.NumBatches > 0)
    {
      writer.WriteLine($"NN evaluator 0              {EvaluatorNN0.Stats}");
    }
    if (EvaluatorNN1 != null && EvaluatorNN1.Stats.NumBatches > 0)
    {
      writer.WriteLine($"NN evaluator 1              {EvaluatorNN1.Stats}");
    }
    writer.WriteLine();

    writer.WriteLine($"SearchLimitInitial.Type     {SearchLimitInitial.Type}");
    writer.WriteLine($"SearchLimitInitial.Value    {SearchLimitInitial.Value,14:N2}");
    writer.WriteLine($"SearchLimit.FracExtensible  {SearchLimit.FractionExtensibleIfNeeded,14:N2}");
    writer.WriteLine($"FractionExtended            {FractionExtendedSoFar,14:F2}");

    {
      // Use the true captured search duration once the search has completed (TimeElapsedTotalSeconds),
      // falling back to a live measurement only while a search is still in progress. This keeps the
      // elapsed time and the backend-busy fraction consistent and free of any idle wall-clock that
      // would otherwise accrue between the search finishing and this dump being requested.
      bool searchCompleted = !double.IsNaN(TimeElapsedTotalSeconds);
      double searchSecs = searchCompleted ? TimeElapsedTotalSeconds
                                          : (DateTime.Now - StartTimeThisSearch).TotalSeconds;
      writer.WriteLine($"Elapsed Search Time         {searchSecs,14:F2}");

      // Device backend utilization: fraction of the search wall-clock during which at least one
      // evaluator was inside the backend (C++ interop). Target is 1.0 (whole move GPU-bound).
      BackendTimeTracker backendTracker = EvaluatorsSet?.BackendTimeTracker;
      double backendBusy = (backendTracker?.EverUsed ?? false) ? backendTracker.BusySeconds : double.NaN;
      double busyFrac = (!double.IsNaN(backendBusy) && searchSecs > 0) ? backendBusy / searchSecs : double.NaN;
      writer.WriteLine($"Backend Busy Time           {backendBusy,14:F2}");
      writer.WriteLine($"Backend Busy Fraction       {busyFrac,14:F3}");
      writer.WriteLine($"Phase Timing                {MCGSIterator.PhaseTimingSummary()}");
      PhaseCoordinator coord = Engine.Coordinator;
      writer.WriteLine($"Backup Out-Of-Order         {coord.BackupOutOfOrderFraction,14:F4}  ({coord.BackupOutOfOrderCount:N0} of {coord.BackupTotalCount:N0} backups; inorderEnforced={coord.EnforceInOrderBackup})");
    }

    writer.WriteLine($"FractionSearchRemaining     {FractionSearchRemaining,14:F3}");
    writer.WriteLine($"Estimated NPS               {EstimatedNPS,14:N0}");

    writer.WriteLine($"EstimatedNumStepsRemaining  {EstimatedNumVisitsRemaining(),14:N0}");
  }
}
