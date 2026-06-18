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
using System.Diagnostics;

using Ceres.Base.Misc;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Selective, amortized upward Q-propagation (an alternative to the full-graph BottomUpQRecalculator).
///
/// Q normally flows leaf->root only along each batch's backed-up visit path. For transposition
/// (multi-parent) nodes, the value is updated through the one parent on the path, but the cached
/// edge.QChild on the node's OTHER (off-path) parent edges goes stale, so Q drifts on those off-path
/// ancestors. This class actively drains that staleness using the engine's existing per-edge IsStale
/// flag, doing only a small bounded amount of work per batch so that - over time - changes propagate
/// up everywhere without a full-graph sweep.
///
/// Mechanism (asynchronous worklist relaxation), run in the post-backup quiescent region of
/// MCGSIterator.RunOnce (backup lock held; graph quiescent; exactly one thread per engine):
///   - SEED: walk this batch's paths; for each multi-parent node touched, mark its genuinely-stale
///     parent edges IsStale and enqueue it on a persistent per-engine queue.
///   - DRAIN: pop nodes up to a per-batch recompute budget; for each popped node X, recompute every
///     parent P whose edge to X is stale (idempotent full recompute via QRecomputeHelper), and if P
///     changed by more than epsilon, mark P's stale parent edges and enqueue P (cascade upward). The
///     tree-parent-index < child-index invariant guarantees cascades reach the root.
///   - Anything beyond budget stays queued and drains over later batches; the existing lazy consumer
///     (Graph.GatherChildInfoViaChildren) keeps draining IsStale on re-visits as a second drainer.
///
/// IsStale is the single source of truth for edge staleness; the queue is only a locating hint
/// (avoids an O(N) scan). The active drainer and the lazy consumer both do the same idempotent
/// refresh, so whichever fires first wins and the other safely skips/repeats.
///
/// Stats are aggregated into a STATIC process-wide 60 second window (so it spans the many short
/// per-move searches) and dumped in yellow as a single "[SelectiveQ]" line.
/// </summary>
public sealed class SelectiveQPropagator
{
  /// <summary>
  /// Duration of the statistics aggregation window (seconds).
  /// </summary>
  const double WINDOW_SECONDS = 60.0;

  /// <summary>
  /// Threshold (absolute Q delta) at/above which an edge is considered stale, a recompute is counted
  /// as a "large" change, and a changed node cascades to its parents. Shared with the existing
  /// off-path machinery so there is a single notion of "significant".
  /// </summary>
  const double EPSILON = MCGSParamsFixed.PROPAGATE_OFF_VISIT_PARENTS_MIN_Q_DELTA;

  /// <summary>
  /// Threshold (absolute Q delta) above which a recompute is counted as "large" (diagnostics only).
  /// </summary>
  const double LARGE_DELTA_THRESHOLD = 0.03;

  /// <summary>
  /// Hard cap on queue size (bounds memory; excess enqueues are dropped and recovered by re-seeding
  /// and the lazy consumer).
  /// </summary>
  const int MAX_QUEUE_SIZE = 1 << 18;

  /// <summary>
  /// Associated engine (source of the graph and parameters).
  /// </summary>
  readonly MCGSEngine Engine;

  /// <summary>
  /// FIFO of node indices whose parent edges may need refreshing, and its membership set for dedup.
  /// Instance-level: this propagator is created once per search and shared by both iterators, so the
  /// storage is reused across all batches of a search (carrying the backlog forward), while a new
  /// search gets a fresh propagator (no cross-search sharing). Only ever touched by the single
  /// backup-lock holder at a time, so no synchronization is needed.
  /// </summary>
  readonly Queue<int> queue = new Queue<int>();
  readonly HashSet<int> queued = new HashSet<int>();

  // Shared aggregation state (static so it persists across the many short searches that occur over a
  // game/session, and so the 60 second window actually fills). All access serialized via statsLock.
  static readonly object statsLock = new object();
  static readonly Stopwatch windowTimer = Stopwatch.StartNew();
  static long windowNumPasses;
  static long windowSeeds;
  static long windowPops;
  static long windowRecomputes;
  static long windowCascades;
  static long windowNumLargeDeltas;
  static long windowOverflowDrops;
  static double windowSumAbsDelta;
  static double windowMaxAbsDelta;
  static double windowSumPassMilliseconds;
  static double windowSumQueueDepth;
  static double cumulativePassMilliseconds; // never reset: cumulative pass time over the session


  /// <summary>
  /// Per-pass accumulator for diagnostics.
  /// </summary>
  struct PassStats
  {
    public long Seeds;
    public long Pops;
    public long Recomputes;
    public long Cascades;
    public long NumLargeDeltas;
    public long OverflowDrops;
    public double SumAbsDelta;
    public double MaxAbsDelta;
  }


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="engine"></param>
  public SelectiveQPropagator(MCGSEngine engine)
  {
    Engine = engine;
  }


  /// <summary>
  /// Seeds from this batch's paths then drains up to the configured budget, propagating stale child Q
  /// values upward. Must be called while holding the backup lock (graph quiescent).
  /// </summary>
  /// <param name="paths">This batch's backed-up paths (the event source for what changed).</param>
  public void RunPass(MCGSPathsSet paths)
  {
    int budget = Engine.Manager.ParamsSearch.SelectiveQDrainBudgetPerBatch;
    if (budget <= 0)
    {
      return;
    }

    long startTicks = Stopwatch.GetTimestamp();

    ParamsSelect paramsSelect = Engine.Manager.ParamsSelect;
    bool cbgPUCTBackupActive = paramsSelect.CBGPUCTBackupActive;

    PassStats stats = default;
    Seed(paths, ref stats);
    Drain(budget, paramsSelect, cbgPUCTBackupActive, ref stats);

    double passMilliseconds = (Stopwatch.GetTimestamp() - startTicks) * 1000.0 / Stopwatch.Frequency;

    lock (statsLock)
    {
      windowNumPasses++;
      windowSeeds += stats.Seeds;
      windowPops += stats.Pops;
      windowRecomputes += stats.Recomputes;
      windowCascades += stats.Cascades;
      windowNumLargeDeltas += stats.NumLargeDeltas;
      windowOverflowDrops += stats.OverflowDrops;
      windowSumAbsDelta += stats.SumAbsDelta;
      windowSumPassMilliseconds += passMilliseconds;
      cumulativePassMilliseconds += passMilliseconds;
      windowSumQueueDepth += queue.Count;
      if (stats.MaxAbsDelta > windowMaxAbsDelta)
      {
        windowMaxAbsDelta = stats.MaxAbsDelta;
      }

      if (windowTimer.Elapsed.TotalSeconds >= WINDOW_SECONDS)
      {
        DumpWindowStatsAndReset();
      }
    }
  }


  /// <summary>
  /// Walks this batch's paths and enqueues each multi-parent node touched (the only nodes with
  /// off-path parents, where transposition drift lives), marking its stale parent edges.
  /// </summary>
  void Seed(MCGSPathsSet paths, ref PassStats stats)
  {
    foreach (MCGSPath path in paths.Paths)
    {
      foreach (MCGSPathVisitMember member in path.PathVisitsLeafToRoot)
      {
        GNode node = member.PathVisitRef.ChildNode;
        if (node.IsNull || node.IsOldGeneration || !node.IsEvaluated || !node.NumParentsMoreThanOne)
        {
          continue;
        }

        if (MarkStaleParentsAndEnqueue(node, ref stats))
        {
          stats.Seeds++;
        }
      }
    }
  }


  /// <summary>
  /// Drains up to <paramref name="budget"/> parent recomputes, cascading upward.
  /// </summary>
  void Drain(int budget, ParamsSelect paramsSelect, bool cbgPUCTBackupActive, ref PassStats stats)
  {
    Graph graph = Engine.Graph;

    // Budget bounds the number of (expensive) parent recomputes and is the termination guarantee in
    // the presence of transposition cycles. Checked between pops; a single pop is processed fully.
    while (stats.Recomputes < budget && queue.Count > 0)
    {
      int nodeIndex = queue.Dequeue();
      queued.Remove(nodeIndex);

      GNode node = graph[nodeIndex];
      if (node.IsNull || node.IsOldGeneration || !node.IsEvaluated)
      {
        continue;
      }

      stats.Pops++;
      ProcessParents(node, paramsSelect, cbgPUCTBackupActive, ref stats);
    }
  }


  /// <summary>
  /// For each parent P of <paramref name="node"/> whose edge to it is stale, refreshes that edge and
  /// recomputes P (idempotent), cascading to P's parents if P's Q changed by more than epsilon.
  /// </summary>
  void ProcessParents(GNode node, ParamsSelect paramsSelect, bool cbgPUCTBackupActive, ref PassStats stats)
  {
    double nodeQ = node.Q;

    foreach (GEdge edge in node.ParentEdges)
    {
      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNodeHasDrawKnownToExist)
      {
        continue;
      }

      // Edge is stale if flagged, or if its cached child Q has drifted from the node's current Q.
      bool stale = edge.IsStale || Math.Abs(edge.QChild - nodeQ) > EPSILON;
      if (!stale)
      {
        continue;
      }

      GNode parent = edge.ParentNode;
      if (!QRecomputeHelper.IsEligibleForRecompute(parent))
      {
        edge.IsStale = false; // nothing to recompute here; stop revisiting this edge
        continue;
      }

      double oldQ = parent.Q;

      // Full live recompute refreshes ALL of parent's child edges (incl. this one) and clears their
      // IsStale; idempotent, so it never double-counts with the incremental/lazy paths.
      double newQ = QRecomputeHelper.RecomputeNodeQ(parent, ReadOnlySpan<double>.Empty, paramsSelect, cbgPUCTBackupActive);

      // Keep the (display-only) draw probability D fresh alongside Q. Live reads are safe here:
      // this pass runs in the quiescent post-backup region (no concurrent in-place D writes).
      QRecomputeHelper.RecomputeNodeD(parent, ReadOnlySpan<double>.Empty);

      double absDelta = Math.Abs(newQ - oldQ);
      stats.Recomputes++;
      stats.SumAbsDelta += absDelta;
      if (absDelta > stats.MaxAbsDelta)
      {
        stats.MaxAbsDelta = absDelta;
      }
      if (absDelta > LARGE_DELTA_THRESHOLD)
      {
        stats.NumLargeDeltas++;
      }

      if (absDelta > EPSILON && MarkStaleParentsAndEnqueue(parent, ref stats))
      {
        stats.Cascades++;
      }
    }
  }


  /// <summary>
  /// Marks the genuinely-stale parent edges of <paramref name="node"/> as IsStale and, if any were
  /// found, enqueues the node for draining (deduped, capped). Returns whether it was enqueued; counts
  /// a queue-overflow drop in <paramref name="stats"/> when the cap is hit for a not-yet-queued node.
  /// </summary>
  bool MarkStaleParentsAndEnqueue(GNode node, ref PassStats stats)
  {
    double nodeQ = node.Q;
    bool anyStale = false;

    foreach (GEdge edge in node.ParentEdges)
    {
      if (edge.Type != GEdgeStruct.EdgeType.ChildEdge || edge.ChildNodeHasDrawKnownToExist)
      {
        continue;
      }

      if (edge.IsStale)
      {
        anyStale = true;
      }
      else if (Math.Abs(edge.QChild - nodeQ) > EPSILON)
      {
        edge.IsStale = true;
        anyStale = true;
      }
    }

    if (!anyStale)
    {
      return false;
    }

    int idx = node.Index.Index;
    if (queued.Contains(idx))
    {
      return false; // already queued: a dedup skip, not an overflow drop
    }
    if (queued.Count >= MAX_QUEUE_SIZE)
    {
      // Queue full: drop this update (it is recovered later by re-seeding and the lazy consumer).
      stats.OverflowDrops++;
      return false;
    }

    queued.Add(idx);
    queue.Enqueue(idx);
    return true;
  }


  /// <summary>
  /// Writes a single yellow summary line for the elapsed window and resets the window.
  /// Caller must hold statsLock.
  /// </summary>
  static void DumpWindowStatsAndReset()
  {
    double seconds = windowTimer.Elapsed.TotalSeconds;
    double passes = windowNumPasses;

    double seedsPer = passes == 0 ? 0 : windowSeeds / passes;
    double popsPer = passes == 0 ? 0 : windowPops / passes;
    double recompPer = passes == 0 ? 0 : windowRecomputes / passes;
    double cascadePer = passes == 0 ? 0 : windowCascades / passes;
    double queueDepthAvg = passes == 0 ? 0 : windowSumQueueDepth / passes;
    double deltaAbsAvg = windowRecomputes == 0 ? 0 : windowSumAbsDelta / windowRecomputes;
    double pctOverLarge = windowRecomputes == 0 ? 0 : 100.0 * windowNumLargeDeltas / windowRecomputes;
    double avgPassMilliseconds = passes == 0 ? 0 : windowSumPassMilliseconds / passes;

    // Overflow drops as a percentage of distinct-node enqueue attempts (enqueued + dropped).
    double enqueueAttempts = windowSeeds + windowCascades + windowOverflowDrops;
    double pctOverflow = enqueueAttempts == 0 ? 0 : 100.0 * windowOverflowDrops / enqueueAttempts;

    string line = $"[SelectiveQ] window={seconds:F1}s passes={windowNumPasses:N0} "
                + $"seeds/pass={seedsPer:N0} pops/pass={popsPer:N0} recomp/pass={recompPer:N0} "
                + $"cascade/pass={cascadePer:N0} queueDepth={queueDepthAvg:N0} overflow={pctOverflow:F2}% "
                + $"deltaAbsAvg={deltaAbsAvg:F6} deltaAbsMax={windowMaxAbsDelta:F6} "
                + $"pctOver{LARGE_DELTA_THRESHOLD:F2}={pctOverLarge:F2}% "
                + $"avgPassMs={avgPassMilliseconds:F3} cumPassSec={cumulativePassMilliseconds / 1000.0:F1}";
    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, line);

    windowNumPasses = 0;
    windowSeeds = 0;
    windowPops = 0;
    windowRecomputes = 0;
    windowCascades = 0;
    windowNumLargeDeltas = 0;
    windowOverflowDrops = 0;
    windowSumAbsDelta = 0;
    windowMaxAbsDelta = 0;
    windowSumPassMilliseconds = 0;
    windowSumQueueDepth = 0;
    windowTimer.Restart();
  }
}
