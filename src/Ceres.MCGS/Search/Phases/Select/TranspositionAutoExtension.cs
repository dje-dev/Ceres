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
using System.Threading;

using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Strategies;

#endregion

namespace Ceres.MCGS.Search.Phases;

/// <summary>
/// Transposition auto-extension (PositionAndHistoryEquivalence mode):
/// when a new node N1 has just been created with its value/policy copied from an
/// evaluated pseudo-twin (TranspositionCopyValues), synchronously also perform the
/// deterministic next visit from N1 - always its top-policy child 0, exactly what the
/// select fast path would choose on N1's next (second) visit - creating or linking
/// node N2 (value-copied from N2's own transposition source, linked to an exact
/// existing node, or created as a terminal edge), and installing N1 with N=2 and the
/// exact two-visit Q via the standard BackupToNode/BackupToEdge primitives.
///
/// The path's single accepted visit then backs up a value informed two plies deeper
/// (the path's termination reason becomes TranspositionCopyValuesAutoExtended whose
/// leaf pre-processing is a no-op, since N1 is fully installed here).
///
/// Motivation: in history mode, pseudo-duplicated regions must be re-expanded one
/// visit at a time, so depth is achieved much more slowly than in Position mode.
/// The extension gains depth 2 (instead of 1) per pseudo-twin expansion with no
/// additional neural network evaluations and exact bookkeeping throughout (every
/// invariant - node.N == sum(edge.N) + 1, QPure identity, child.N >= edge.N - holds
/// exactly as if a real second visit had descended through N1).
///
/// CONCURRENCY: runs while the expansion parent's lock is held, so every additional
/// lock acquisition here is NON-BLOCKING (TryAcquire) with graceful fallback to the
/// plain TranspositionCopyValues behavior; this excludes any possibility of deadlock
/// with concurrent select threads (whose parent->child and twin-copy lock orders are
/// otherwise unconstrained relative to ours).
/// </summary>
internal static class TranspositionAutoExtension
{
  /// <summary>
  /// Diagnostic counters, kept in per-thread cells with plain (non-atomic) increments on
  /// thread-private cache lines and aggregated only when the stats string is built.
  /// Shared Interlocked counters here would cost a cross-socket locked operation per
  /// extension attempt at high transposition rates.
  /// </summary>
  private sealed class StatsCell
  {
    public long NumExtendedNewNode;
    public long NumExtendedLinked;
    public long NumExtendedTerminal;
    public long NumExtendedDegenerate;
    public long NumSkipped;
  }

  [ThreadStatic]
  static StatsCell statsCellForThread;

  static readonly ConcurrentQueue<StatsCell> allStatsCells = new();

  static StatsCell StatsCellThisThread
  {
    get
    {
      StatsCell cell = statsCellForThread;
      if (cell == null)
      {
        cell = new StatsCell();
        statsCellForThread = cell;
        allStatsCells.Enqueue(cell);
      }
      return cell;
    }
  }


  internal static string StatsString
  {
    get
    {
      // Aggregate over per-thread cells (reads of aligned longs are atomic on x64;
      // tearing-free but possibly slightly stale, acceptable for diagnostics).
      long newNode = 0, linked = 0, terminal = 0, degenerate = 0, skipped = 0;
      foreach (StatsCell cell in allStatsCells)
      {
        newNode += cell.NumExtendedNewNode;
        linked += cell.NumExtendedLinked;
        terminal += cell.NumExtendedTerminal;
        degenerate += cell.NumExtendedDegenerate;
        skipped += cell.NumSkipped;
      }

      return $"AutoExtend newNode={newNode:N0} linked={linked:N0} "
           + $"terminal={terminal:N0} degenerate={degenerate:N0} skipped={skipped:N0}";
    }
  }


  /// <summary>
  /// Interval between periodic console dumps of StatsString (while the feature is active).
  /// </summary>
  const long STATS_DUMP_INTERVAL_MS = 3 * 60 * 1000;

  // Initialized at class load so the first dump arrives one full interval into use.
  static long lastStatsDumpTimeMS = System.Environment.TickCount64;

  /// <summary>
  /// Dumps StatsString to the console (in yellow) if at least STATS_DUMP_INTERVAL_MS has
  /// elapsed since the last dump. Called from TryExtend, so dumps occur only while the
  /// feature is enabled and being exercised. The CompareExchange guarantees a single
  /// dump per interval under concurrent callers.
  /// (The hot-path read is a plain Volatile.Read: an Interlocked.Read would be a bus-locked
  /// RMW taking the cache line exclusive on every call from every thread.)
  /// </summary>
  static void PossiblyDumpStats()
  {
    long nowMS = System.Environment.TickCount64;
    long last = Volatile.Read(ref lastStatsDumpTimeMS);
    if (nowMS - last >= STATS_DUMP_INTERVAL_MS
     && Interlocked.CompareExchange(ref lastStatsDumpTimeMS, nowMS, last) == last)
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, StatsString);
    }
  }


  // Reusable per-thread move list for n2 move generation in TryExtend (reset at each use).
  [ThreadStatic]
  static MGMoveList n2MovesScratch;


  /// <summary>
  /// Attempts the auto-extension of newly created node n1 (just value-copied from twin).
  /// Returns true if n1 was fully installed here (N>=1 with all backups applied), in which
  /// case the caller must set the path termination reason to
  /// TranspositionCopyValuesAutoExtended so the backup phase does not double-apply the
  /// leaf node update. Returns false if nothing was changed (plain TranspositionCopyValues
  /// processing should continue unchanged).
  /// </summary>
  /// <param name="engine">The engine.</param>
  /// <param name="path">Current path; its leaf visit is n1 (already added).</param>
  /// <param name="n1">The newly created node (N=0, value-copied, policy possibly deferred).</param>
  /// <param name="twin">The pseudo-twin used as n1's value source.</param>
  /// <param name="n1Pos">n1's position.</param>
  /// <param name="n1Hash96">n1's standalone 96-bit position hash.</param>
  /// <param name="moveToN1Irreversible">If the move from the parent into n1 was irreversible.</param>
  internal static unsafe bool TryExtend(MCGSEngine engine, MCGSPath path,
                                        GNode n1, GNode twin,
                                        in MGPosition n1Pos, PosHash96 n1Hash96,
                                        bool moveToN1Irreversible)
  {
    ParamsSearch paramsSearch = engine.Manager.ParamsSearch;
    Debug.Assert(paramsSearch.EnableTranspositionAutoExtension);
    Debug.Assert(paramsSearch.PathTranspositionMode == PathMode.PositionAndHistoryEquivalence);

    PossiblyDumpStats();

    // The extension mimics the deterministic next visit (always child 0 for a node with
    // no expanded edges); features which would reorder unvisited children before that
    // real visit would make the choice diverge, so the extension defers to them.
    // N.B. the action-head resort is only operative when FPUMode == ActionHead
    //      (PossiblyActionResortUsingAction early-returns otherwise), so only that
    //      combination disables the extension.
    bool actionResortActive = engine.Manager.ParamsSelect.ActionResortUnvisitedChildren
                           && engine.Manager.ParamsSelect.FPUMode == ParamsSelect.FPUType.ActionHead;
    // N.B. check policy-move count on the TWIN: n1's own NumPolicyMoves is still 0 when its
    //      policy copy was deferred (it is set only when the edge headers materialize);
    //      the twin's count is exactly what n1 will receive.
    if (twin.N < paramsSearch.TranspositionAutoExtensionMinTwinN
     || twin.NumPolicyMoves == 0
     || paramsSearch.MoveOrderingPhase != ParamsSearch.MoveOrderingPhaseEnum.None
     || actionResortActive)
    {
      StatsCellThisThread.NumSkipped++;
      return false;
    }

    if (!n1.TryAcquireLock())
    {
      StatsCellThisThread.NumSkipped++;
      return false;
    }

    try
    {
      // Defensive: n1 was created moments ago by this thread and must still be untouched.
      if (n1.NodeRef.N != 0 || n1.NumEdgesExpanded != 0)
      {
        StatsCellThisThread.NumSkipped++;
        return false;
      }

      // Materialize n1's (possibly deferred) policy so edge header 0 is available.
      // Mirrors GNode.DoPolicyCopyFromDeferredNodeSource but acquires the source twin's
      // lock NON-blockingly (a blocking acquire here would invert the standard
      // source->destination order used elsewhere and create a deadlock surface).
      if (n1.IsPendingPolicyCopy)
      {
        GNode policySource = engine.Graph[n1.NodeRef.edgeHeaderBlockIndexOrDeferredNode.NodeIndex];
        if (!policySource.TryAcquireLock())
        {
          StatsCellThisThread.NumSkipped++;
          return false;
        }
        try
        {
          n1.NodeRef.edgeHeaderBlockIndexOrDeferredNode.Clear();
          Graph.AllocateAndCopyPolicyValues(policySource, n1);
        }
        finally
        {
          policySource.ReleaseLock();
        }
      }

      if (n1.NumPolicyMoves == 0)
      {
        StatsCellThisThread.NumSkipped++;
        return false;
      }

      // Child 0 (top policy; headers are P-sorted after the policy copy).
      GEdgeHeaderStruct header0 = n1.EdgeHeadersSpan[0];
      MGMove moveMG = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(header0.Move, in n1Pos);
      MGPosition n2Pos = n1Pos;
      n2Pos.MakeMove(moveMG);
      bool moveToN2Irreversible = n1Pos.IsIrreversibleMove(moveMG, in n2Pos);

      PosHash64 n2Hash64 = MGPositionHashing.Hash64(in n2Pos);
      PosHash96 n2Hash96 = MGPositionHashing.Hash96(in n2Pos);

      // N2's position+sequence key, mirroring ComputeChildPositionInfo exactly.
      // path.RunningHash covers positions through n1's PARENT at this point
      // (PossiblyBranched has not yet run), so extend it locally with n1.
      PosHash96MultisetRunning runningWithN1 = moveToN1Irreversible ? default : path.RunningHash;
      runningWithN1.Add(n1Hash96);
      PosHash96MultisetFinalized n2Key = moveToN2Irreversible
        ? PosHash96MultisetRunning.EpochStartFinalized(n2Hash96)
        : runningWithN1.Finalized(n2Hash96);

      // Repetition check against our full history; the path's leaf visit IS n1,
      // so this covers n1 and all ancestors plus spine/prehistory.
      bool positionDuplicate = path.HashFoundInHistoryOrPrehistory(n2Hash64);
      n2Pos.RepetitionCount = (byte)(positionDuplicate ? 1 : 0);

      // Generate into this class's own reusable scratch list (no allocation): n2Moves is
      // fully consumed before return (terminality check and node initialization only read it,
      // never retain it). A scratch distinct from the select phase's is used because the
      // caller's childMoves may share that one.
      n2MovesScratch ??= new MGMoveList();
      n2MovesScratch.NumMovesUsed = 0;
      MGMoveGen.GenerateMoves(in n2Pos, n2MovesScratch);
      MGMoveList n2Moves = n2MovesScratch;
      int minRepetitionCountForDraw = paramsSearch.TwofoldDrawEnabled ? 1 : 2;
      (GameResult result, float v, float d, bool wasDrawByRepetition) resultInfo =
        path.CalcPathTerminationFromUnexpandedLeaf(minRepetitionCountForDraw, in n2Pos, n2Moves, possiblyUseTablebase: true);

      MCGSSelectBackupStrategyBase strategy = engine.Strategy;
      // N.B. n1's SiblingsQ/SiblingsQFrac may have been pre-installed at creation by the
      //      sibling-blend-at-creation feature; the BackupToNode calls below compose that
      //      blend with the (extension-improved) pure Q automatically, since
      //      ResetNodeQUsingNewQPure reads the stored sibling fields with exact-inversion
      //      bookkeeping.
      double v1 = n1.V;
      double d1 = n1.DrawP;

      if (resultInfo.result != GameResult.Unknown)
      {
        // Terminal extension: create the terminal edge and install n1 with the exact
        // two-visit statistics. ORDER MATTERS: n1's first BackupToNode must precede the
        // edge backup so the node.N == sum(edge.N) + 1 invariant holds at each step.
        bool propagateAsDraw = resultInfo.v == 0;
        GEdge terminalEdge = engine.Graph.AddNewTerminalEdge(n1, 0, resultInfo.v, resultInfo.d, 1, propagateAsDraw);

        strategy.BackupToNode(n1, 1, v1, d1);
        strategy.BackupToEdge(terminalEdge, 1, resultInfo.v, resultInfo.d, false);
        strategy.BackupToNode(n1, 1, -resultInfo.v, resultInfo.d);

        StatsCellThisThread.NumExtendedTerminal++;
        return true;
      }

      if (positionDuplicate)
      {
        // Nonterminal duplicate (possible in 3-fold configuration): the normal flow
        // excludes duplicates from transposition value copying; skip the extension.
        StatsCellThisThread.NumSkipped++;
        return false;
      }

      // Probe for an exact (position+sequence) existing node.
      GNode existing = engine.Graph.TryGetNodeByPositionAndSequence(n2Key);
      if (!existing.IsNull && (!existing.IsEvaluated || existing.N < 1))
      {
        // Exists but still in flight (unevaluated/unvisited): linking it now would add an
        // edge we cannot value yet; skip entirely (no edge created).
        StatsCellThisThread.NumSkipped++;
        return false;
      }

      GNode twin2 = default;
      if (existing.IsNull)
      {
        // Will need to CREATE n2: verify its evaluation source exists BEFORE creating.
        // Creating first and bailing would leave a permanently unevaluated registered
        // node (never scheduled for NN eval - every future visit would piggyback/abort).
        PosHash64WithMove50AndReps n2StandaloneKey = MGPositionHashing.Hash64WithMove50AndRepsAdded(
          n2Hash64, n2Pos.RepetitionCount, n2Pos.Move50Category);
        twin2 = engine.Graph.TryLookupNode(n2StandaloneKey);
        if (twin2.IsNull || !twin2.IsEvaluated || twin2.Graph != engine.Graph)
        {
          // No same-graph evaluation source (foreign-graph sources excluded: deferred
          // policy copy does not support cross-graph and their lifecycle is unmanaged here).
          StatsCellThisThread.NumSkipped++;
          return false;
        }
      }

      // Create n2 or link the existing node (existing-node lock acquired non-blockingly).
      (GEdge edge, bool wasCollision) = engine.Graph.AddEdgeToNewOrExistingNode(
        n1, 0, in n2Pos, n2Hash64, n2Key, n2Moves,
        out bool wasCreated, out GNode _,
        okToCreate: true, nonBlockingExistingNodeLock: true);

      if (edge.edgeStructPtr == null)
      {
        // Existing node's lock unavailable; nothing was created or changed.
        StatsCellThisThread.NumSkipped++;
        return false;
      }

      GNode n2 = edge.ChildNode;

      if (wasCreated && !wasCollision)
      {
        // Fresh n2: populate values from its transposition source (policy deferred,
        // exactly as a normal TranspositionCopyValues creation), then install the
        // visit statistics through the standard primitives.
        engine.Graph.CopyNodeValues(0, twin2, n2, copyPolicy: false);

        strategy.BackupToNode(n1, 1, v1, d1);
        strategy.BackupToNode(n2, 1, n2.V, n2.DrawP);
        strategy.BackupToEdge(edge, 1, n2.Q, n2.D, false);
        strategy.BackupToNode(n1, 1, -n2.Q, n2.D);

        StatsCellThisThread.NumExtendedNewNode++;
        return true;
      }

      if (!n2.IsNull && n2.IsEvaluated && n2.N >= 1)
      {
        // Linked to an existing evaluated node (possibly the winner of a creation race):
        // use its full statistics directly - the richest extension outcome.
        strategy.BackupToNode(n1, 1, v1, d1);
        strategy.BackupToEdge(edge, 1, n2.Q, n2.D, false);
        strategy.BackupToNode(n1, 1, -n2.Q, n2.D);

        StatsCellThisThread.NumExtendedLinked++;
        return true;
      }

      // Degenerate (rare race): the edge now exists but points to a node still in flight
      // elsewhere. Install only n1's own evaluation visit (N=1, satisfying
      // N == sum(edge.N) + 1 with edge.N = 0); its creator will complete the node, and
      // ordinary selection will visit the edge later.
      strategy.BackupToNode(n1, 1, v1, d1);
      StatsCellThisThread.NumExtendedDegenerate++;
      return true;
    }
    finally
    {
      n1.ReleaseLock();
    }
  }
}
