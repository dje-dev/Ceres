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
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Threading.Tasks;

using Ceres.Base.Misc;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Graphs.GraphStores;
using Ceres.MCGS.Search.Params;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Diagnostic helper that performs a full recomputation of the Q value of every eligible
/// node in the search graph.
///
/// This is an experimental facility (ParamsSearch.PostBackupQMode == FullRecompute) intended to be
/// invoked once per batch immediately after the backup phase completes, while the invoking
/// iterator still holds the backup lock. At that point the graph is quiescent: no other
/// iterator can be in its select or backup phase (they are mutually exclusive), so the
/// whole-graph traversal performed here observes and mutates a stable graph.
///
/// PARALLELISM / SNAPSHOT:
/// Each node's Q is updated exactly once per pass. A strict leaves-to-root recompute is
/// inherently sequential (a parent depends on its children), and the Q fields are unaligned
/// doubles in a packed struct, so naively partitioning the node range across threads would
/// risk torn reads. To parallelize safely the pass uses a SNAPSHOT: every node's Q is copied
/// into a buffer first (read-only), then nodes are recomputed in parallel - each node reads
/// its children's Q exclusively from the snapshot while writing its own Q in place. Because
/// no thread ever reads another node's live Q, the in-place writes cannot race. This makes a
/// single pass a Jacobi-style sweep that propagates one graph level (the sequential version
/// propagated the full depth in one pass); since the recompute runs every batch the values
/// stay refreshed, and the per-node delta cleanly measures the consistency of each stored Q
/// with what its children currently imply. Degree of parallelism is Min(1 + proc/2, 32).
///
/// For each node the cached child-Q on every outgoing edge is first refreshed from the
/// snapshot child Q, then Q is recomputed using exactly the backup rule the engine applies,
/// so the recomputed values remain self-consistent with subsequent search:
///   - standard mode : pure Q = (sum over children of -edge.Q * edge.N + V) / N
///   - TPS backup    : the tempered posterior (TPSScoreCalc.ComputeVBar), which is the
///                     rule used whenever ParamsSelect.TPSBackupActive is true.
///
/// The magnitude of the change (delta) applied to each node's Q is tracked, and once per
/// (approximately) 60 second interval a single yellow summary line is written to the Console.
///
/// IMPORTANT: a new MCGSEngine (and therefore a new BottomUpQRecalculator) is created for
/// each search (i.e. each move). The aggregation state below is therefore STATIC so that the
/// 60 second window spans the lifetime of the process (across all moves) rather than being
/// reset on every short search - otherwise a window would essentially never fill for normal
/// few-second moves. Folding of per-pass results into the shared state is guarded by a lock
/// (the expensive recomputation itself runs outside the lock on each engine's own graph).
/// </summary>
public sealed class BottomUpQRecalculator
{
  /// <summary>
  /// Duration of the statistics aggregation window (seconds).
  /// </summary>
  const double WINDOW_SECONDS = 60.0;

  /// <summary>
  /// Threshold (absolute Q delta) above which a node update is counted as "large".
  /// </summary>
  const double LARGE_DELTA_THRESHOLD = 0.03;

  /// <summary>
  /// Associated engine (source of the graph and parameters).
  /// </summary>
  readonly MCGSEngine Engine;

  /// <summary>
  /// Reusable per-pass snapshot of every node's Q (indexed by node index), grown as the graph
  /// grows. Instance-level (not static): each engine has its own graph and its RunPass is only
  /// ever entered by one thread at a time (the backup-lock holder), so no cross-engine sharing.
  /// </summary>
  double[] qSnapshot = Array.Empty<double>();

  /// <summary>
  /// Companion snapshot of every node's draw probability D (indexed by node index), used the same
  /// Jacobi way as qSnapshot so the parallel D recompute reads children's D race-free. Same
  /// lifetime/threading guarantees as qSnapshot.
  /// </summary>
  double[] dSnapshot = Array.Empty<double>();

  // Shared aggregation state (static so it persists across the many short searches that
  // occur over a game/session, and so the 60 second window actually fills). All access is
  // serialized via statsLock.
  static readonly object statsLock = new object();
  static readonly Stopwatch windowTimer = Stopwatch.StartNew();
  static long windowNumPasses;
  static long windowNumNodes;
  static double windowSumDelta;
  static double windowSumAbsDelta;
  static double windowSumDeltaSquared;
  static double windowSumPassMilliseconds;
  static double windowMaxAbsDelta;
  static long windowNumLargeDeltas;
  static double cumulativePassMilliseconds; // never reset: cumulative pass time over the session


  /// <summary>
  /// Per-thread accumulator for the statistics gathered during a recomputation pass.
  /// </summary>
  struct PassAccumulator
  {
    public long Count;
    public double SumDelta;
    public double SumAbsDelta;
    public double SumDeltaSquared;
    public double MaxAbsDelta;
    public long NumLargeDeltas;
  }


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="engine"></param>
  public BottomUpQRecalculator(MCGSEngine engine)
  {
    Engine = engine;
  }


  /// <summary>
  /// Performs one full recomputation pass over the graph and folds the resulting per-node
  /// delta statistics into the current (process-wide) aggregation window, emitting a summary
  /// line once the window duration has elapsed.
  ///
  /// Must be called while holding the backup lock (graph quiescent).
  /// </summary>
  public void RunPass()
  {
    long startTicks = Stopwatch.GetTimestamp();
    PassAccumulator pass = RecomputeAllNodes();
    double passMilliseconds = (Stopwatch.GetTimestamp() - startTicks) * 1000.0 / Stopwatch.Frequency;

    // Fold this pass into the shared window and possibly emit (cheap; the heavy work above
    // ran outside the lock on this engine's own graph).
    lock (statsLock)
    {
      windowNumPasses++;
      windowNumNodes += pass.Count;
      windowSumDelta += pass.SumDelta;
      windowSumAbsDelta += pass.SumAbsDelta;
      windowSumDeltaSquared += pass.SumDeltaSquared;
      windowSumPassMilliseconds += passMilliseconds;
      cumulativePassMilliseconds += passMilliseconds;
      windowNumLargeDeltas += pass.NumLargeDeltas;
      if (pass.MaxAbsDelta > windowMaxAbsDelta)
      {
        windowMaxAbsDelta = pass.MaxAbsDelta;
      }

      if (windowTimer.Elapsed.TotalSeconds >= WINDOW_SECONDS)
      {
        DumpWindowStatsAndReset();
      }
    }
  }


  /// <summary>
  /// Recomputes the Q value of every eligible node in the graph in parallel, returning the
  /// aggregated per-node delta statistics for the pass.
  /// </summary>
  PassAccumulator RecomputeAllNodes()
  {
    Graph graph = Engine.Graph;
    ParamsSelect paramsSelect = Engine.Manager.ParamsSelect;
    bool regularizedBackupActive = paramsSelect.RegularizedBackupActive;
    int numTotalNodes = graph.NodesStore.NumTotalNodes;

    // Ensure snapshot buffer capacity (grown geometrically; never shrunk). Safe without
    // locking: RunPass is single-threaded per engine instance.
    if (qSnapshot.Length < numTotalNodes)
    {
      qSnapshot = new double[Math.Max(numTotalNodes, qSnapshot.Length * 2)];
    }
    if (dSnapshot.Length < numTotalNodes)
    {
      dSnapshot = new double[Math.Max(numTotalNodes, dSnapshot.Length * 2)];
    }
    double[] snapshot = qSnapshot;
    double[] snapshotD = dSnapshot;

    // Phase 1: snapshot every node's current Q (and D) (read-only -> race free). Child Q/D values
    // are read exclusively from these snapshots during phase 2 so that the in-place writes to node
    // Q/D below cannot race with concurrent reads. Valid node indices are [ROOT_NODE_INDEX,
    // NumTotalNodes); index 0 is the null node.
    OrderablePartitioner<Tuple<int, int>> partitioner = Partitioner.Create(GraphStore.ROOT_NODE_INDEX, numTotalNodes);
    Parallel.ForEach(partitioner, range =>
    {
      for (int i = range.Item1; i < range.Item2; i++)
      {
        GNode node = graph[i];
        snapshot[i] = node.Q;
        snapshotD[i] = node.D;
      }
    });

    // Phase 2: recompute every eligible node in parallel, accumulating delta statistics in
    // per-thread accumulators that are merged at the end of each task.
    object combineLock = new object();
    PassAccumulator total = default;

    Parallel.ForEach(partitioner,
      () => default(PassAccumulator),
      (range, loopState, local) =>
      {
        for (int nodeIndex = range.Item1; nodeIndex < range.Item2; nodeIndex++)
        {
          GNode node = graph[nodeIndex];

          // Skip nodes whose Q is undefined, fixed, or trivial (see QRecomputeHelper).
          if (!QRecomputeHelper.IsEligibleForRecompute(node))
          {
            continue;
          }

          double oldQ = node.Q;
          double newQ = QRecomputeHelper.RecomputeNodeQ(node, snapshot, paramsSelect, regularizedBackupActive);

          // Recompute the (display-only) draw probability D from the D snapshot, in tandem with Q.
          // Delta stats below track Q only; D is maintained for free off the same sweep.
          QRecomputeHelper.RecomputeNodeD(node, snapshotD);

          double delta = newQ - oldQ;
          double absDelta = Math.Abs(delta);
          local.Count++;
          local.SumDelta += delta;
          local.SumAbsDelta += absDelta;
          local.SumDeltaSquared += delta * delta;
          if (absDelta > local.MaxAbsDelta)
          {
            local.MaxAbsDelta = absDelta;
          }
          if (absDelta > LARGE_DELTA_THRESHOLD)
          {
            local.NumLargeDeltas++;
          }
        }

        return local;
      },
      local =>
      {
        lock (combineLock)
        {
          total.Count += local.Count;
          total.SumDelta += local.SumDelta;
          total.SumAbsDelta += local.SumAbsDelta;
          total.SumDeltaSquared += local.SumDeltaSquared;
          total.NumLargeDeltas += local.NumLargeDeltas;
          if (local.MaxAbsDelta > total.MaxAbsDelta)
          {
            total.MaxAbsDelta = local.MaxAbsDelta;
          }
        }
      });

    return total;
  }


  /// <summary>
  /// Writes a single yellow summary line for the elapsed window and resets the window.
  /// Caller must hold statsLock.
  /// </summary>
  static void DumpWindowStatsAndReset()
  {
    double seconds = windowTimer.Elapsed.TotalSeconds;

    double avgDelta = windowNumNodes == 0 ? 0 : windowSumDelta / windowNumNodes;
    double avgAbsDelta = windowNumNodes == 0 ? 0 : windowSumAbsDelta / windowNumNodes;
    double meanSquare = windowNumNodes == 0 ? 0 : windowSumDeltaSquared / windowNumNodes;

    // Population standard deviation of the per-node delta (guard against tiny negative
    // values arising from floating point cancellation).
    double variance = meanSquare - (avgDelta * avgDelta);
    double stdDevDelta = variance <= 0 ? 0 : Math.Sqrt(variance);

    double avgNodesPerPass = windowNumPasses == 0 ? 0 : (double)windowNumNodes / windowNumPasses;
    double avgPassMilliseconds = windowNumPasses == 0 ? 0 : windowSumPassMilliseconds / windowNumPasses;
    double pctOverLarge = windowNumNodes == 0 ? 0 : 100.0 * windowNumLargeDeltas / windowNumNodes;

    string line = $"[BottomUpQ] window={seconds:F1}s passes={windowNumPasses:N0} "
                + $"nodes/pass={avgNodesPerPass:N0} "
                + $"deltaAvg={avgDelta:F6} deltaAbsAvg={avgAbsDelta:F6} deltaStdDev={stdDevDelta:F6} "
                + $"deltaAbsMax={windowMaxAbsDelta:F6} pctOver{LARGE_DELTA_THRESHOLD:F2}={pctOverLarge:F2}% "
                + $"avgPassMs={avgPassMilliseconds:F3} cumPassSec={cumulativePassMilliseconds / 1000.0:F1}";
    ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, line);

    windowNumPasses = 0;
    windowNumNodes = 0;
    windowSumDelta = 0;
    windowSumAbsDelta = 0;
    windowSumDeltaSquared = 0;
    windowSumPassMilliseconds = 0;
    windowMaxAbsDelta = 0;
    windowNumLargeDeltas = 0;
    windowTimer.Restart();
  }
}
