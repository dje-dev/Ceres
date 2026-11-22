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

using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;

#endregion

namespace Ceres.MCGS.Search.ProbeGraft;

/// <summary>
/// Directed visits: committed, NN-evaluated visits forced through a chosen edge of a chosen
/// node (with stock PUCT selection below the forced child and stock backup above it), and
/// lines of such visits applied ply by ply along a given move sequence.
///
/// This is the "repair" primitive of the LOCUS experiments: once an external analysis has
/// located the node whose evaluation MCGS got wrong (and the move it never looked at), the
/// search is led there many times so that its OWN net re-founds the value of that line. No
/// probe-derived value ever enters the graph; the only effect is where the visits go.
///
/// Built entirely on MCGSIterator.RunProbeSpecs in commit mode (the same primitive the M1
/// graft scheduler uses), so it must only run against a quiescent graph, on a dedicated
/// iterator (see MCGSEngine.CreateInnerIterator / RunWithInnerIterator).
/// </summary>
internal static class DirectedVisits
{
  /// <summary>Upper bound on visits forced per RunProbeSpecs call (keeps batches moderate).</summary>
  const int MAX_VISITS_PER_CALL = 512;

  /// <summary>Reliability gate for a child to be "best by Q" in ReweightSpine.</summary>
  const int MIN_RELIABLE_N = 50;
  const double RELIABLE_FRACTION_OF_MAX_N = 0.10;

  /// <summary>
  /// Outcome of a directed-visit operation.
  /// </summary>
  /// <param name="VisitsApplied">growth of the forced edge's visit count (sum over plies for a line)</param>
  /// <param name="PliesApplied">plies of a line successfully crossed</param>
  /// <param name="RootNGrowth">growth of the search root's N</param>
  /// <param name="Abort">null on success, otherwise a short reason the operation stopped early</param>
  public readonly record struct Outcome(int VisitsApplied, int PliesApplied, int RootNGrowth, string Abort)
  {
    public override string ToString()
      => $"visits={VisitsApplied} plies={PliesApplied} rootN+={RootNGrowth}" + (Abort == null ? "" : $" abort={Abort}");
  }


  /// <summary>
  /// Forces the given number of visits through parent -> childIndex (stock selection below),
  /// chunked so that each RunProbeSpecs call stays within the maximum batch size. Expansion
  /// holes below the target slot are backfilled with single visits first (the edge store
  /// expands strictly in order). Stops early if the edge becomes terminal.
  /// </summary>
  /// <param name="iterator"></param>
  /// <param name="parent"></param>
  /// <param name="childIndex">index within the node's policy move list (expanded or not)</param>
  /// <param name="numVisits"></param>
  /// <param name="markStale">if true the edge is flagged IsStale afterwards (for StaleDrain propagation)</param>
  /// <returns></returns>
  internal static Outcome ForceChildVisits(MCGSIterator iterator, GNode parent, int childIndex, int numVisits, bool markStale = false)
  {
    int maxBatchSize = Math.Min(MAX_VISITS_PER_CALL, iterator.Manager.ParamsSearch.Execution.MaxBatchSize);
    int rootNBefore = iterator.Engine.SearchRootNode.N;

    if (parent.IsNull || !parent.IsEvaluated)
    {
      return new Outcome(0, 0, 0, "ParentNotEvaluated");
    }
    if (parent.Terminal.IsTerminal())
    {
      return new Outcome(0, 0, 0, "ParentTerminal");
    }
    if (childIndex < 0 || childIndex >= parent.NumPolicyMoves)
    {
      return new Outcome(0, 0, 0, "ChildIndexOutOfRange");
    }

    int edgeNBefore = childIndex < parent.NumEdgesExpanded ? parent.ChildEdgeAtIndex(childIndex).N : 0;

    List<ProbeSpec> specs = new();
    int numExpanded = parent.NumEdgesExpanded;
    int numBackfill = Math.Max(0, childIndex - numExpanded);
    if (numBackfill + 1 > maxBatchSize)
    {
      return new Outcome(0, 0, 0, "TooManyBackfills");
    }
    for (int slot = numExpanded; slot < childIndex; slot++)
    {
      specs.Add(new ProbeSpec(parent, slot, 1));
    }

    string abort = null;
    int remaining = numVisits;
    while (remaining > 0)
    {
      int chunk = Math.Min(remaining, maxBatchSize - specs.Count);
      specs.Add(new ProbeSpec(parent, childIndex, chunk));
      iterator.RunProbeSpecs(specs, null, commitInsteadOfDrop: true);
      specs.Clear();
      remaining -= chunk;

      if (childIndex >= parent.NumEdgesExpanded)
      {
        abort = "SlotNotExpanded";
        break;
      }

      GEdge edge = parent.ChildEdgeAtIndex(childIndex);
      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge)
      {
        // Terminal edge: the value is exact and already backed up; nothing more to learn.
        break;
      }
    }

    int edgeNAfter = childIndex < parent.NumEdgesExpanded ? parent.ChildEdgeAtIndex(childIndex).N : 0;
    if (markStale && childIndex < parent.NumEdgesExpanded)
    {
      GEdge edge = parent.ChildEdgeAtIndex(childIndex);
      if (edge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        edge.IsStale = true;
      }
    }

    return new Outcome(edgeNAfter - edgeNBefore, 0, iterator.Engine.SearchRootNode.N - rootNBefore, abort);
  }


  /// <summary>
  /// Applies a line of moves from an anchor node: at each ply the next move is resolved to a
  /// child slot (position-aware), visitsPerPly visits are forced through it, the crossing is
  /// re-validated and the cursor advances to the child. Returns the number of plies crossed.
  /// </summary>
  /// <param name="iterator"></param>
  /// <param name="anchor"></param>
  /// <param name="moves">moves from the anchor's position</param>
  /// <param name="visitsPerPly"></param>
  /// <param name="maxPlies"></param>
  /// <param name="markStale"></param>
  /// <returns></returns>
  internal static Outcome ApplyLine(MCGSIterator iterator, GNode anchor, IReadOnlyList<MGMove> moves,
                                    int visitsPerPly, int maxPlies, bool markStale = false)
  {
    int rootNBefore = iterator.Engine.SearchRootNode.N;
    int visitsTotal = 0;
    int plies = 0;
    string abort = null;

    GNode cursor = anchor;
    int numPlies = Math.Min(moves.Count, maxPlies);
    for (int ply = 0; ply < numPlies; ply++)
    {
      if (cursor.IsNull || !cursor.IsEvaluated)
      {
        abort = "CursorGone";
        break;
      }
      if (cursor.Terminal.IsTerminal())
      {
        break; // line ran into a terminal node: exact value already backed up
      }

      if (cursor.IsPendingPolicyCopy)
      {
        cursor.AcquireLock();
        bool materialized;
        try
        {
          materialized = cursor.TryDoDeferredPolicyCopyIfNeeded();
        }
        finally
        {
          cursor.ReleaseLock();
        }
        if (!materialized)
        {
          abort = "PolicyCopyStalled";
          break;
        }
      }

      if (cursor.NumPolicyMoves == 0)
      {
        abort = "NoPolicyMoves";
        break;
      }

      MGPosition cursorPos = cursor.CalcPosition();
      int childIndex = ProbeGraftScheduler.ResolveMoveToChildIndex(cursor, in cursorPos, moves[ply]);
      if (childIndex < 0)
      {
        abort = "MoveNotFound";
        break;
      }

      Outcome step = ForceChildVisits(iterator, cursor, childIndex, visitsPerPly, markStale);
      visitsTotal += step.VisitsApplied;
      if (step.Abort != null)
      {
        abort = step.Abort;
        break;
      }

      if (childIndex >= cursor.NumEdgesExpanded)
      {
        abort = "SlotNotExpanded";
        break;
      }
      GEdge edge = cursor.ChildEdgeAtIndex(childIndex);
      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge)
      {
        plies++;
        break; // terminal: success
      }

      GNode child = edge.ChildNode;
      if (child.IsNull || !child.IsEvaluated || child.N == 0)
      {
        abort = "ChildUnvisited";
        break;
      }

      plies++;
      cursor = child;
    }

    return new Outcome(visitsTotal, plies, iterator.Engine.SearchRootNode.N - rootNBefore, abort);
  }


  /// <summary>
  /// Flags every edge on the tree-parent chain from the node up to the search root as stale,
  /// so that a StaleDrain post-backup pass refreshes the cached child values along the spine
  /// (only meaningful when ParamsSearch.PostBackupQMode == StaleDrain).
  /// </summary>
  /// <param name="node"></param>
  /// <returns>number of edges flagged</returns>
  internal static int MarkSpineStale(GNode node)
  {
    int count = 0;
    GNode cursor = node;
    int guard = 0;
    while (!cursor.IsNull && !cursor.IsSearchRoot && !cursor.IsGraphRoot && guard++ < 512)
    {
      GEdge parentEdge = cursor.TreeParentEdge;
      if (parentEdge.IsNull)
      {
        break;
      }
      if (parentEdge.Type == GEdgeStruct.EdgeType.ChildEdge)
      {
        parentEdge.IsStale = true;
        count++;
      }
      cursor = parentEdge.ParentNode;
    }
    return count;
  }

  /// <summary>
  /// One-shot "forgetting" reweight of the spine from a node up to the search root, applied
  /// AFTER the node's line has been re-founded by directed visits.
  ///
  /// Averaging backup cannot express "the mover would no longer choose that child" without
  /// visit mass: once a locus has been re-evaluated, every ancestor's Q is still dominated by
  /// visits allocated under the old belief (inertia of order the ancestor's N). This pass
  /// re-derives the spine under MCGS's OWN current child values: at each spine node the child
  /// that is best for the node's mover (minimum child-perspective edge Q) keeps its visit mass
  /// and every other child's edge mass is scaled down to keepFraction. Child Q values are never
  /// altered (no value is injected); only the weights change, node N and Q are updated to stay
  /// consistent with the select-phase reset (Q = sum over edges N*q / N), the removed mass is
  /// propagated up the tree-parent chain, and the spine edges' cached child Q is refreshed.
  /// A false alarm is recoverable: the shrunk siblings keep their Q, so PUCT re-grows them.
  /// </summary>
  /// <param name="locus">node whose line was re-founded (processing starts here and walks up)</param>
  /// <param name="keepFraction">fraction of each non-best child's edge visits retained</param>
  /// <param name="processed">node indices already reweighted this round (their siblings are not
  /// shrunk again; removed mass still propagates through them); may be null</param>
  /// <param name="reweightRoot">if false the search root's children are left untouched (the
  /// root move decision stays with the stock N-based rule; only the mass removed below is propagated)</param>
  /// <returns>visits removed at the search root (equivalently the total mass forgotten)</returns>
  internal static int ReweightSpine(GNode locus, double keepFraction, HashSet<int> processed = null, bool reweightRoot = false)
  {
    if (locus.IsNull || keepFraction >= 1.0)
    {
      return 0;
    }

    GNode cursor = locus;
    int removedBelow = 0;          // visits removed at the level below, already propagated into cursor's edge
    double removedBelowValue = 0;  // their average value seen from the level below
    int guard = 0;
    int removedAtRoot = 0;

    while (!cursor.IsNull && guard++ < 512)
    {
      int numExpanded = cursor.NumEdgesExpanded;
      bool shrinkHere = (reweightRoot || !cursor.IsSearchRoot)
                     && (processed == null || processed.Add(cursor.Index.Index));

      // Best child for the mover at cursor: minimum child-perspective Q over visited edges whose
      // visit count is reliable enough to trust its Q (mirrors the TopQIfSufficientN idea: a
      // low-N sibling with a noisy optimistic Q must not dethrone a well-founded favorite).
      int bestIndex = -1;
      double bestQ = double.PositiveInfinity;
      if (shrinkHere)
      {
        int maxChildN = 0;
        for (int i = 0; i < numExpanded; i++)
        {
          GEdge e = cursor.ChildEdgeAtIndex(i);
          if (e.Type != GEdgeStruct.EdgeType.Uninitialized && e.N > maxChildN)
          {
            maxChildN = e.N;
          }
        }
        int minReliableN = Math.Max(MIN_RELIABLE_N, (int)(RELIABLE_FRACTION_OF_MAX_N * maxChildN));
        for (int i = 0; i < numExpanded; i++)
        {
          GEdge e = cursor.ChildEdgeAtIndex(i);
          if (e.Type == GEdgeStruct.EdgeType.Uninitialized || e.N < minReliableN)
          {
            continue;
          }
          if (e.Q < bestQ)
          {
            bestQ = e.Q;
            bestIndex = i;
          }
        }
      }

      int removedHere = 0;
      double removedW = 0;   // total value of removed visits seen from cursor
      if (bestIndex >= 0)
      {
        for (int i = 0; i < numExpanded; i++)
        {
          if (i == bestIndex)
          {
            continue;
          }
          GEdge e = cursor.ChildEdgeAtIndex(i);
          if (e.Type == GEdgeStruct.EdgeType.Uninitialized || e.N <= 0)
          {
            continue;
          }
          int keep = (int)Math.Round(e.N * keepFraction);
          keep = Math.Max(keep, e.NDrawByRepetition);     // never below the draw-by-repetition share
          int removed = e.N - keep;
          if (removed <= 0)
          {
            continue;
          }
          removedW += removed * (-e.Q);                     // child-perspective -> cursor perspective
          e.N = keep;
          removedHere += removed;
        }
      }

      // Visits removed below passed through cursor with the opposite sign.
      int totalRemoved = removedHere + removedBelow;
      double totalW = removedW + removedBelow * (-removedBelowValue);

      int n = cursor.N;
      if (totalRemoved > 0 && n - totalRemoved >= 1)
      {
        double newQ = (cursor.Q * n - totalW) / (n - totalRemoved);
        newQ = Math.Clamp(newQ, -1.0, 1.0);
        cursor.NodeRef.Q = newQ;
        cursor.NodeRef.N = n - totalRemoved;
      }
      else
      {
        totalRemoved = 0;
        totalW = 0;
      }

      if (cursor.IsSearchRoot || cursor.IsGraphRoot)
      {
        removedAtRoot = totalRemoved;
        break;
      }

      // Walk up the highest-N parent edge whose parent is strictly closer to the search root
      // (under transpositions the creating tree-parent edge may carry few of the visits).
      GEdge parentEdge = HighestNRootwardParentEdge(cursor);
      if (parentEdge.IsNull || parentEdge.Type != GEdgeStruct.EdgeType.ChildEdge)
      {
        break;
      }

      // Propagate: the parent's edge into cursor loses the removed mass (bounded by what it
      // carried - under transpositions some of it arrived via other parents) and its cached
      // child value is refreshed.
      int removedViaEdge = Math.Min(totalRemoved, Math.Max(0, parentEdge.N - parentEdge.NDrawByRepetition));
      parentEdge.N = parentEdge.N - removedViaEdge;
      parentEdge.QChild = cursor.Q;
      parentEdge.IsStale = true;

      removedBelow = removedViaEdge;
      removedBelowValue = totalRemoved > 0 ? totalW / totalRemoved : 0;
      cursor = parentEdge.ParentNode;
    }

    return removedAtRoot;
  }

  /// <summary>
  /// Forces visits through parent -> childIndex in chunks until the child's value (parent-mover
  /// perspective) exceeds the best reliable sibling's by the margin ("the search sees the
  /// light": PUCT will take over by Q alone), or maxVisits have been applied. Directed visits are
  /// double-edged: they consume the edge's exploration credit (its U term), so a line forced to a
  /// Q still below the favorite's is frozen out rather than helped. Callers should roll such
  /// visits back (ForgetEdgeVisits) when flipped is false.
  /// </summary>
  internal static Outcome ForceChildVisitsUntilBest(MCGSIterator iterator, GNode parent, int childIndex, int maxVisits,
                                                    double margin, out bool flipped, int chunk = 512)
  {
    flipped = false;
    int applied = 0, rootGrowth = 0;
    string abort = null;
    while (applied < maxVisits)
    {
      Outcome step = ForceChildVisits(iterator, parent, childIndex, Math.Min(chunk, maxVisits - applied));
      applied += step.VisitsApplied;
      rootGrowth += step.RootNGrowth;
      if (step.Abort != null)
      {
        abort = step.Abort;
        break;
      }
      if (step.VisitsApplied == 0)
      {
        abort = "NoProgress";
        break;
      }
      if (childIndex >= parent.NumEdgesExpanded)
      {
        abort = "SlotNotExpanded";
        break;
      }
      GEdge edge = parent.ChildEdgeAtIndex(childIndex);
      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge)
      {
        flipped = -edge.Q >= BestSiblingValue(parent, childIndex) + margin;
        break;   // terminal edge: exact
      }
      if (-edge.Q >= BestSiblingValue(parent, childIndex) + margin)
      {
        flipped = true;
        break;
      }
    }
    return new Outcome(applied, 0, rootGrowth, abort);
  }


  /// <summary>
  /// Value (parent-mover perspective) of the best sibling of childIndex among edges with a
  /// reliable visit count; -1 when there is none.
  /// </summary>
  internal static double BestSiblingValue(GNode parent, int childIndex)
  {
    int maxN = 0;
    for (int i = 0; i < parent.NumEdgesExpanded; i++)
    {
      if (i == childIndex) continue;
      GEdge e = parent.ChildEdgeAtIndex(i);
      if (e.Type != GEdgeStruct.EdgeType.Uninitialized && e.N > maxN) maxN = e.N;
    }
    int minReliableN = Math.Max(MIN_RELIABLE_N, (int)(RELIABLE_FRACTION_OF_MAX_N * maxN));
    double best = -1.0;
    for (int i = 0; i < parent.NumEdgesExpanded; i++)
    {
      if (i == childIndex) continue;
      GEdge e = parent.ChildEdgeAtIndex(i);
      if (e.Type == GEdgeStruct.EdgeType.Uninitialized || e.N < minReliableN) continue;
      best = Math.Max(best, -e.Q);
    }
    return best;
  }


  /// <summary>
  /// Rolls back `count` visits of the edge parent -> childIndex in N only: the edge's visit
  /// count, the parent's N/Q and every rootward spine ancestor's N/Q are reduced as if those
  /// visits had not happened, while the child subtree keeps everything it learned (its Q, its
  /// nodes). Restores the edge's exploration credit after a directed-visit campaign that did not
  /// flip the parent's preference. Returns the visits removed at the search root.
  /// </summary>
  internal static int ForgetEdgeVisits(GNode parent, int childIndex, int count)
  {
    if (parent.IsNull || count <= 0 || childIndex < 0 || childIndex >= parent.NumEdgesExpanded)
    {
      return 0;
    }
    GEdge edge = parent.ChildEdgeAtIndex(childIndex);
    if (edge.Type == GEdgeStruct.EdgeType.Uninitialized)
    {
      return 0;
    }
    int removable = Math.Max(0, edge.N - edge.NDrawByRepetition);
    int removed = Math.Min(count, removable);
    if (removed <= 0)
    {
      return 0;
    }
    double valueFromParent = -edge.Q;
    edge.N = edge.N - removed;

    GNode cursor = parent;
    int removedBelow = removed;
    double removedBelowValue = valueFromParent;   // seen from cursor
    int guard = 0;
    int removedAtRoot = 0;
    while (!cursor.IsNull && guard++ < 512)
    {
      int n = cursor.N;
      if (n - removedBelow < 1)
      {
        break;
      }
      double newQ = Math.Clamp((cursor.Q * n - removedBelow * removedBelowValue) / (n - removedBelow), -1.0, 1.0);
      cursor.NodeRef.Q = newQ;
      cursor.NodeRef.N = n - removedBelow;

      if (cursor.IsSearchRoot || cursor.IsGraphRoot)
      {
        removedAtRoot = removedBelow;
        break;
      }
      GEdge parentEdge = HighestNRootwardParentEdge(cursor);
      if (parentEdge.IsNull || parentEdge.Type != GEdgeStruct.EdgeType.ChildEdge)
      {
        break;
      }
      int viaEdge = Math.Min(removedBelow, Math.Max(0, parentEdge.N - parentEdge.NDrawByRepetition));
      parentEdge.N = parentEdge.N - viaEdge;
      parentEdge.QChild = cursor.Q;
      parentEdge.IsStale = true;
      removedBelow = viaEdge;
      removedBelowValue = -removedBelowValue;   // perspective flips each ply
      cursor = parentEdge.ParentNode;
    }
    return removedAtRoot;
  }


  /// <summary>
  /// Returns the parent edge carrying the most visits among parents strictly closer to the
  /// search root (the edge most of the node's visits arrived by), or the tree-parent edge as a
  /// fallback; a null edge at the search/graph root.
  /// </summary>
  static GEdge HighestNRootwardParentEdge(GNode node)
  {
    if (node.IsSearchRoot || node.IsGraphRoot)
    {
      return default;
    }
    int depth = node.DepthFromSearchRoot();
    GEdge chosen = default;
    int chosenN = -1;
    foreach (GEdge parentEdge in node.ParentEdges)
    {
      GNode candidate = parentEdge.ParentNode;
      if (candidate.IsNull)
      {
        continue;
      }
      int candidateDepth;
      try
      {
        candidateDepth = candidate.DepthFromSearchRoot();
      }
      catch
      {
        continue;
      }
      if (candidateDepth < depth && parentEdge.N > chosenN)
      {
        chosen = parentEdge;
        chosenN = parentEdge.N;
      }
    }
    return chosenN >= 0 ? chosen : node.TreeParentEdge;
  }
}
