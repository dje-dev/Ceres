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
using System.Threading;

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.Coordination;


/// <summary>
/// Specification of one probe: force the next NumVisits visits through ParentNode
/// to all go to the child at ChildIndex (with stock batched selection below the child).
/// </summary>
public readonly record struct ProbeSpec(GNode ParentNode, int ChildIndex, int NumVisits);


/// <summary>
/// Snapshot of one probe path (built before the path's ledgers are dropped), sufficient
/// for an external "shadow backup" to replay the path's leaf value through the backup
/// arithmetic without any graph mutation.
/// </summary>
public sealed class ProbePathRecord
{
  /// <summary>Node index of the probed parent (the spec's ParentNode).</summary>
  public NodeIndex SpecParentNodeIndex;

  /// <summary>Forced child index at the probed parent (the spec's ChildIndex).</summary>
  public int SpecChildIndex;

  /// <summary>Termination reason of this probe path.</summary>
  public MCGSPathTerminationReason TerminationReason;

  /// <summary>NN value (leaf perspective) for PendingNeuralNetEval terminations; NaN otherwise.</summary>
  public float LeafV;

  /// <summary>Draw probability companion of LeafV (NaN when LeafV is NaN).</summary>
  public float LeafDrawP;

  /// <summary>Visits accepted at the leaf (the visit mass this path deposits).</summary>
  public int NumVisitsAcceptedAtLeaf;

  /// <summary>Visits attempted at the leaf (>= accepted; difference evaporated).</summary>
  public int NumVisitsAttemptedAtLeaf;

  /// <summary>Leaf node index (default for terminal-edge terminations which have no node).</summary>
  public NodeIndex LeafNodeIndex;

  /// <summary>If the leaf node was created by this probe batch (vs pre-existing).</summary>
  public bool LeafIsNewlyCreatedNode;

  /// <summary>
  /// All hops of the path from the search root down to the leaf:
  /// (parent node index, child index within that parent's edge list).
  /// Hops[SpecStartHopIndex] is the forced parent->child crossing; hops before it are the
  /// forced spine (needed by the shadow replay because backup also updates those ancestors,
  /// and repetition-draw edges can reference them).
  /// </summary>
  public (NodeIndex ParentNodeIndex, int ChildIndexInParent)[] Hops;

  /// <summary>Index within Hops of the forced parent->child crossing.</summary>
  public int SpecStartHopIndex;
}


public partial class MCGSIterator
{
  /// <summary>
  /// When true, selection suppresses all writes to pre-existing graph state that the stock
  /// select phase would otherwise perform (select-phase Q reset, stale-edge refresh,
  /// unvisited-children resort, move-order rearrangement). Set by RunProbeSpecs for the
  /// duration of a probe batch; must be false during ordinary search.
  /// </summary>
  internal bool ProbeSuppressGraphWrites;


  /// <summary>
  /// Runs a batch of probe specs: for each spec, builds forced path(s) sending the requested
  /// visits through the spec's (parent, child) with stock selection below the child, evaluates
  /// all resulting leaves in a single aggregated NN batch (results are installed on the new
  /// leaf nodes as usual), then - instead of backing the paths up - drops every path through
  /// the standard dropped-visits ledger so that no pre-existing N/Q/D/edge statistic anywhere
  /// in the graph changes. New nodes/edges created by probe selection remain permanently as
  /// evaluated structure (new leaf nodes are installed to N=1; see remarks below).
  ///
  /// Returns one ProbePathRecord per completed path (snapshot taken before the drop) from
  /// which the caller computes hypothetical Q values via shadow-backup arithmetic.
  ///
  /// When commitInsteadOfDrop is true the batch is instead backed up normally (stock
  /// RunBackupPhase) - used by tests to validate shadow-backup arithmetic against the
  /// real engine backup on identical paths.
  ///
  /// preBackupCallback (if given) is invoked with the completed records after evaluation but
  /// BEFORE any backup/install/drop, so shadow computations always observe the same graph
  /// state in both modes: all pre-existing statistics unchanged, new leaf nodes evaluated
  /// (WinP/LossP/policy written) but not yet installed (N=0).
  ///
  /// Must be called only when the graph is quiescent (between batches of the main search,
  /// e.g. from MCGSEngine.PostBatchHook, or after the search completed), on a dedicated
  /// iterator constructed like MCGSEngine.RunFromInnerNodes does. Single-iterator harnesses
  /// only (no DualOverlappedIterators). The caller must chunk specs so that the sum of
  /// requested visits does not exceed Execution.MaxBatchSize.
  /// </summary>
  internal List<ProbePathRecord> RunProbeSpecs(IReadOnlyList<ProbeSpec> specs,
                                               Action<IReadOnlyList<ProbePathRecord>> preBackupCallback = null,
                                               bool commitInsteadOfDrop = false)
  {
    Debug.Assert(!Manager.ParamsSearch.Execution.DualOverlappedIterators);

    List<ProbePathRecord> records = new();
    if (specs.Count == 0)
    {
      return records;
    }

    // Probes reuse iterator lane 0 for in-flight accounting, which is only correct when no
    // other select/backup is in flight. Verify quiescence at the search root's edges.
    foreach (GEdge rootEdge in Engine.SearchRootNode.ChildEdgesExpanded)
    {
      if (rootEdge.NInFlightForIterator(IteratorID) != 0)
      {
        throw new InvalidOperationException("RunProbeSpecs requires a quiescent graph (found in-flight visits at root)");
      }
    }

    int thisBatchID = Interlocked.Add(ref Engine.nextBatchID, 1) - 1;

    PathsSet.Reset();
    ResetPaths();
    pathVisitPool.Clear(false);

    BackupMode = Engine.Backup.BackupModeToUse();

    // Nodes with index >= this value were created by this probe batch.
    int firstNewNodeIndex = Engine.Graph.Store.NodesStore.NumUsedNodes + GNodeStore.FIRST_ALLOCATED_INDEX;

    bool savedSuppress = ProbeSuppressGraphWrites;
    ProbeSuppressGraphWrites = !commitInsteadOfDrop;
    try
    {
      // STEP 1: build the forced probe paths (one spec at a time; within-batch in-flight
      // repulsion spreads each spec's visits over its subtree with stock batch semantics).
      Engine.Coordinator.EnterSelect(IteratorID, thisBatchID);
      foreach (ProbeSpec spec in specs)
      {
        Engine.Select.ExtendPathForcedChild(this, spec.ParentNode, spec.ChildIndex, spec.NumVisits);
      }
      Engine.SelectWorkerPools[IteratorID]?.WaitAll();
      Engine.Coordinator.ExitSelect(IteratorID, thisBatchID);

      if (PathsSet.Paths.Count == 0)
      {
        // Apply any deferred ledger backouts for visits dropped during select. Still pass
        // through the in-order-backup gate so the turn counter stays contiguous.
        Engine.Coordinator.EnterBackupOrder(thisBatchID);
        Engine.Coordinator.RecordBackupOrder(thisBatchID, didBackup: false);
        ApplyPendingDroppedVisits();
        Engine.Coordinator.ExitBackupOrder(thisBatchID);
        return records;
      }

      // STEP 2: evaluate all pending leaves in one aggregated NN batch
      // (results are written onto the new leaf nodes before any backup would run).
      Engine.Coordinator.EnterEvaluate(IteratorID, thisBatchID);
      RunNNEvaluationPhase(deferRetrieveResults: false);
      Engine.Coordinator.ExitEvaluate(IteratorID, thisBatchID);

      Engine.Coordinator.EnterBackup(IteratorID, thisBatchID);

      MCGSPath[] completedPaths = PathsSet.Paths.ToArray();

      // Snapshot path records BEFORE dropping (ApplyDroppedVisits zeroes the per-slot
      // attempted counters that the records capture).
      foreach (MCGSPath path in completedPaths)
      {
        ProbePathRecord rec = BuildProbePathRecord(path, firstNewNodeIndex);
        if (rec != null)
        {
          records.Add(rec);
        }
      }

      // Shadow computations run here, observing identical graph state in both modes
      // (pre-existing statistics unchanged; new leaves evaluated but uninstalled).
      preBackupCallback?.Invoke(records);

      if (commitInsteadOfDrop)
      {
        RunBackupPhase();
      }
      else
      {
        // Install newly created leaf nodes to N=1 before dropping. Without this, a dropped
        // path would leave behind an evaluated node with N=0 and Q=0: stock search treats a
        // later arrival there as AlreadyNNEvaluated (no leaf update) and would back up the
        // bogus Q=0 forever. The installed state (evaluated, N=1, no expanded edges,
        // incoming edge N=0) is exactly the legal graph-mode transposition-target state.
        foreach (MCGSPath path in completedPaths)
        {
          InstallProbeCreatedLeafNodeIfNeeded(path, firstNewNodeIndex);
        }

        // STEP 3 (drop mode): back out every path's ledgers instead of backing up. This
        // reconciles the ancestor visit counters and edge in-flight counts to exactly their
        // pre-probe values (visits already dropped during select carry their own records).
        foreach (MCGSPath path in completedPaths)
        {
          if (path.NumVisitsInPath > 0)
          {
            PathsSet.RecordDroppedVisits(path, path.LeafVisitRef.NumVisitsAttempted);
          }
        }
        ApplyPendingDroppedVisits();

        // RunBackupPhase (skipped here) ordinarily releases the evaluator's buffers lock;
        // failing to release it would deadlock the next NN evaluation.
        if (PathsSet.NNPaths.Count > 0)
        {
          EvaluatorNN.Evaluator.BuffersLock?.Release();
        }
      }

      Engine.Coordinator.ExitBackup(IteratorID, thisBatchID);
      batchSequenceNum++;

      return records;
    }
    finally
    {
      ProbeSuppressGraphWrites = savedSuppress;
    }
  }


  /// <summary>
  /// Installs a leaf node newly created by this probe batch (index >= firstNewNodeIndex) to
  /// its post-leaf-update state (N=1, Q/D from its evaluation), mirroring the leaf-node
  /// portion of MCGSBackup.ApplyLeafNodeUpdates for the termination reasons that create
  /// nodes. Pre-existing leaf nodes are never touched.
  /// </summary>
  private void InstallProbeCreatedLeafNodeIfNeeded(MCGSPath path, int firstNewNodeIndex)
  {
    if (path.NumVisitsInPath == 0)
    {
      return;
    }

    ref MCGSPathVisit leafVisit = ref path.LeafVisitRef;
    if (leafVisit.ParentChildEdge.Type != GEdgeStruct.EdgeType.ChildEdge)
    {
      return; // terminal edges have no child node
    }

    GNode leafNode = path.LeafNode;
    if (leafNode.Index.Index < firstNewNodeIndex || leafNode.N > 0)
    {
      return; // pre-existing node, or already installed (e.g. auto-extension applied at select)
    }

    switch (path.TerminationReason)
    {
      case MCGSPathTerminationReason.PendingNeuralNetEval:
        Debug.Assert(leafVisit.NumVisitsAccepted == 1);
        Engine.Strategy.BackupToNode(leafNode, 1, path.TerminationInfo.V, path.TerminationInfo.DrawP);
        break;

      case MCGSPathTerminationReason.TranspositionCopyValues:
        if (leafNode.IsEvaluated)
        {
          Engine.Strategy.BackupToNode(leafNode, 1, leafNode.V, leafNode.DrawP);
        }
        break;

      default:
        // Other reasons never require installation of a probe-created leaf node
        // (rep-draw placeholder nodes legitimately remain unevaluated at N=0, matching stock).
        break;
    }
  }


  /// <summary>
  /// Builds the shadow-backup snapshot record for one completed probe path
  /// (see ProbePathRecord). Returns null for paths that never reached the forced child
  /// (e.g. aborted on the spine).
  /// </summary>
  private ProbePathRecord BuildProbePathRecord(MCGSPath path, int firstNewNodeIndex)
  {
    if (path.NumVisitsInPath == 0 || path.InnerSearchStartNode.IsNull)
    {
      return null;
    }

    if (path.NumVisitsInPath - path.InnerSearchStartDepth <= 0)
    {
      return null; // aborted before the forced child hop
    }

    int numHops = path.NumVisitsInPath;
    (NodeIndex ParentNodeIndex, int ChildIndexInParent)[] hops = new (NodeIndex, int)[numHops];
    for (int i = 0; i < numHops; i++)
    {
      ref MCGSPathVisit visit = ref path[i];
      hops[i] = (visit.ParentChildEdge.ParentNode.Index, visit.IndexOfChildInParent);
    }

    ref MCGSPathVisit leafVisit = ref path.LeafVisitRef;
    bool leafHasNode = leafVisit.ParentChildEdge.Type == GEdgeStruct.EdgeType.ChildEdge;
    GNode leafNode = leafHasNode ? path.LeafNode : default;
    bool isNNEval = path.TerminationReason == MCGSPathTerminationReason.PendingNeuralNetEval;

    return new ProbePathRecord()
    {
      SpecParentNodeIndex = path.InnerSearchStartNode.Index,
      SpecChildIndex = hops[path.InnerSearchStartDepth].ChildIndexInParent,
      SpecStartHopIndex = path.InnerSearchStartDepth,
      TerminationReason = path.TerminationReason,
      LeafV = isNNEval ? path.TerminationInfo.V : float.NaN,
      LeafDrawP = isNNEval ? path.TerminationInfo.DrawP : float.NaN,
      NumVisitsAcceptedAtLeaf = leafVisit.NumVisitsAccepted ?? 0,
      NumVisitsAttemptedAtLeaf = leafVisit.NumVisitsAttempted,
      LeafNodeIndex = leafHasNode ? leafNode.Index : default,
      LeafIsNewlyCreatedNode = leafHasNode && leafNode.Index.Index >= firstNewNodeIndex,
      Hops = hops,
    };
  }
}
