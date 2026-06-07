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
using System.Text;
using System.Threading;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GNodes;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Diagnostic consistency checks verifying that the MCGS search root node
/// genuinely represents the current position being searched.
///
/// Background: under graph reuse (especially PositionEquivalence / "Position" mode) the
/// search root descends BELOW the graph root by design, so the search-root position and the
/// graph-root position legitimately differ. These checks therefore do NOT compare the search
/// root against the graph root. Instead they verify the invariants that must always hold:
///   (1) the search root reconstructs to the actual current position,
///   (2) the two independent routes used to derive the search-root position agree
///       (tree-parent walk via <see cref="GNode.CalcPosition"/> vs. the graph-reuse path walk
///        captured in <see cref="MCGSEngine.SearchRootPosMG"/>), and
///   (3) the graph root is internally self-consistent (its reconstructed position equals the
///       stored prior-moves history).
///
/// A failure indicates genuine corruption (e.g. a position hash collision in
/// PositionEquivalence mode, a broken tree-parent invariant, or an invalid graph-reuse path),
/// not the benign search-root-below-graph-root situation.
///
/// Comparisons use <see cref="Position.EqualAsRepetition"/> (piece placement, side to move,
/// castling and en-passant rights) and intentionally ignore the half-move (50-move) clock and
/// full-move number, which can legitimately differ between transposing paths in Position mode.
/// </summary>
public static class MCGSRootConsistencyCheck
{
  /// <summary>
  /// Validates search-root / graph-root position consistency.
  /// Returns true if all invariants hold; otherwise writes a detailed red diagnostic
  /// block to the console and returns false.
  /// </summary>
  /// <param name="engine">The engine whose search root is being validated.</param>
  /// <param name="expectedCurrentPos">
  /// The position the search root is expected to represent. At search start this is the actual
  /// current game position (priorMoves.FinalPosition); within the post-search dump the engine's
  /// own <see cref="MCGSEngine.SearchRootPosMG"/> (derived during reuse) is used as the reference.
  /// </param>
  /// <param name="context">Short label describing the call site (shown in diagnostics).</param>
  /// <param name="detail">Populated with the diagnostic text if a mismatch is found (else empty).</param>
  /// <returns>True if consistent; false if any invariant is violated.</returns>
  public static bool Validate(MCGSEngine engine, in Position expectedCurrentPos, string context, out string detail)
  {
    detail = "";

    if (engine == null || engine.SearchRootNode.IsNull)
    {
      return true; // Nothing to validate.
    }

    try
    {
      // Position the search root reconstructs to, following tree-parent edges to the graph root.
      Position searchRootCalc = engine.SearchRootNode.CalcPosition().ToPosition;

      // (1) Search root must represent the actual current position (board-level equality;
      //     ignores move-count / 50-move clock which can legitimately differ in Position mode).
      bool okCurrent = searchRootCalc.EqualAsRepetition(in expectedCurrentPos);

      // (2) The graph-reuse path walk (SearchRootPosMG) and the tree-parent walk (CalcPosition)
      //     must agree on the search-root position.
      Position searchRootPath = engine.SearchRootPosMG.ToPosition;
      bool okInternal = searchRootCalc.EqualAsRepetition(in searchRootPath);

      // (3) Graph-root self-consistency: the node flagged as graph root must reconstruct to the
      //     stored prior-moves history's final position.
      Position graphRootCalc = engine.Graph.GraphRootNode.CalcPosition().ToPosition;
      Position graphRootHistory = engine.Graph.Store.NodesStore.PositionHistory.FinalPosition;
      bool okGraphRoot = graphRootCalc.EqualAsRepetition(in graphRootHistory);

      if (okCurrent && okInternal && okGraphRoot)
      {
        return true;
      }

      detail = BuildDiagnostic(engine, in expectedCurrentPos, context,
                               in searchRootCalc, in searchRootPath, in graphRootCalc, in graphRootHistory,
                               okCurrent, okInternal, okGraphRoot);
      ConsoleUtils.WriteLineColored(ConsoleColor.Red, detail);
      return false;
    }
    catch (Exception e)
    {
      // A diagnostic must never destabilize the engine; report softly and treat as consistent.
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow,
        $"MCGSRootConsistencyCheck.Validate threw ({context}): {e.Message}");
      return true;
    }
  }


  /// <summary>
  /// Builds the multi-line diagnostic block describing a detected inconsistency.
  /// </summary>
  static string BuildDiagnostic(MCGSEngine engine, in Position expectedCurrentPos, string context,
                                in Position searchRootCalc, in Position searchRootPath,
                                in Position graphRootCalc, in Position graphRootHistory,
                                bool okCurrent, bool okInternal, bool okGraphRoot)
  {
    GraphRootToSearchRootNodeInfo[] path = engine.SearchRootPathFromGraphRoot ?? [];
    string graphRootFEN = engine.Graph.Store.NodesStore.PositionHistory.FinalPosition.FEN;

    StringBuilder sb = new();
    sb.AppendLine();
    sb.AppendLine("**********************************************************************************");
    sb.AppendLine($"*** MCGS ROOT POSITION CONSISTENCY MISMATCH ({context})  Thread={Thread.CurrentThread.ManagedThreadId}");
    sb.AppendLine($"*** PathTranspositionMode = {engine.Manager.ParamsSearch.PathTranspositionMode}");
    sb.AppendLine("*** Failed checks: "
                + (okCurrent ? "" : "[searchRoot != currentPos] ")
                + (okInternal ? "" : "[treeWalk != reusePathWalk] ")
                + (okGraphRoot ? "" : "[graphRoot != history] "));
    sb.AppendLine($"*** Search root CalcPosition : {searchRootCalc.FEN}");
    sb.AppendLine($"*** Expected current pos     : {expectedCurrentPos.FEN}");
    sb.AppendLine($"*** Search root reuse-path   : {searchRootPath.FEN}");
    sb.AppendLine($"*** Graph root CalcPosition  : {graphRootCalc.FEN}");
    sb.AppendLine($"*** Graph root history pos   : {graphRootHistory.FEN}");
    sb.AppendLine($"*** Search root node         : {engine.SearchRootNode}");
    sb.AppendLine($"*** Graph root -> search root path ({path.Length} plies):");
    sb.Append($"***   {graphRootFEN}");
    foreach (GraphRootToSearchRootNodeInfo info in path)
    {
      sb.Append($"  {info.MoveToChild.MoveStr(MGMoveNotationStyle.Coordinates)} -> {info.ChildPosMG.ToPosition.FEN}");
    }
    sb.AppendLine();
    sb.AppendLine("**********************************************************************************");
    return sb.ToString();
  }
}
