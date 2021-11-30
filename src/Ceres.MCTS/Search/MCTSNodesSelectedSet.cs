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

using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Params;

#endregion

namespace Ceres.MCTS.Search
{
  /// <summary>
  /// Manages a set of nodes that have been selected as part of a batch.
  /// 
  /// Among other functions, we preprocess the nodes to partition them into:
  //   - those that can be immediately processed (e.g. a transposition hit or checkmate position)
  //   - those that are transpositions of other nodes already in flight (in this batch or the overlapped batch)
  //   - otherwise those that will have to be sent to neural network for evaluation
  /// </summary>
  public class MCTSNodesSelectedSet
  {
    #region Static statistics

#if DEBUG
    public static long TotalNumDualSelectorDuplicates = 0;
#endif

    public static long TotalNumNodesSelectedIntoTreeCache;
    public static long TotalNumNodesAppliedFromTreeCache;

#endregion

    int defaultMaxNodesNN;
    int maxNodesNN;
    public int MaxNodesNN 
    {
      get => maxNodesNN;
      set => maxNodesNN = Math.Max(value, NodesNN == null ? 0 : NodesNN.Count);
    }

    public MCTSNodesSelectedSet NodesOtherBatch { get; private set; }

#region Lists

    /// <summary>
    /// Holds nodes which can be evaluated immediately, but have not yet been applied
    /// </summary>
    public readonly ListBounded<MCTSNode> NodesImmediateNotYetApplied;

    /// <summary>
    /// Nodes which are transpositions of other nodes already in this batch (in the NodesNN)
    /// </summary>
    public readonly ListBounded<MCTSNode> NodesTranspositionInFlightThisBatchLinked;

    /// <summary>
    /// Nodes which are transpositions of other nodes in the other (overlapping) batch
    /// </summary>
    public ListBounded<MCTSNode> NodesTranspositionInFlightOtherBatchLinked;

    /// <summary>
    /// Nodes destined for the neural network
    /// </summary>
    public readonly ListBounded<MCTSNode> NodesNN;


#endregion

    // TODO: it seems the counts below are not quite correct, we should aggregate if repeat visits
    static int Count(ListBounded<MCTSNode> nodes) => nodes == null ? 0 : nodes.Count;
    public int TotalNodes => NumNodesImmediatelyApplied + Count(NodesImmediateNotYetApplied) + Count(NodesNN) 
                           + Count(NodesTranspositionInFlightThisBatchLinked) + Count(NodesTranspositionInFlightOtherBatchLinked);

#if NOT
    int CountNonCollisions(ListBounded<MCTSNode> nodes) => nodes == null ? 0 : CountLeafsFromSelector(SelectorID, nodes);
    public int TotalNumNonCollisions => NumLeafsFromNodesImmediatelyApplied
                                      + CountNonCollisions(NodesImmediateNotYetApplied) 
                                      + CountNonCollisions(NodesNN) 
                                      + CountNonCollisions(NodesTranspositionInFlightThisBatchLinked) 
                                      + CountNonCollisions(NodesTranspositionInFlightOtherBatchLinked);
#endif
    public readonly MCTSIterator Context;
    public readonly MCTSApply BlockApply;
    public readonly LeafSelectorMulti Selector;

    bool IN_FLIGHT_THIS_BATCH_LINKAGE_ENABLED;
    bool IN_FLIGHT_OTHER_BATCH_LINKAGE_ENABLED;


    public int NumNewLeafsAddedNonDuplicates = 0;

    public int NumNodesImmediatelyApplied = 0;
    public int NumLeafsFromNodesImmediatelyApplied = 0;

    public int NumCacheOnly;
    public int NumNotApply;

    public readonly int MaxNodes;

    HashSet<int> indicesOtherNodesAlreadyInFlight;

    Dictionary<ulong, MCTSNode> transpositionRootsThisBatch;
    Dictionary<ulong, MCTSNode> transpositionRootsOtherBatch;

    bool haveAppliedThisBatch = false;
    bool haveAppliedOtherBatch = false;

    public int SelectorID => Selector.SelectorID;


    /// <summary>
    /// Constructor.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="selector"></param>
    /// <param name="maxNodes"></param>
    /// <param name="maxNodesNN"></param>
    /// <param name="blockApply"></param>
    /// <param name="IN_FLIGHT_THIS_BATCH_LINKAGE_ENABLED"></param>
    /// <param name="IN_FLIGHT_OTHER_BATCH_LINKAGE_ENABLED"></param>
    public MCTSNodesSelectedSet(MCTSIterator context, LeafSelectorMulti selector, int maxNodes, int maxNodesNN,
                                    MCTSApply blockApply,
                                    bool IN_FLIGHT_THIS_BATCH_LINKAGE_ENABLED,
                                    bool IN_FLIGHT_OTHER_BATCH_LINKAGE_ENABLED)
    {
      Debug.Assert((selector.SelectorID == 0 ? context.Root.NInFlight : context.Root.NInFlight2) == 0);

      Context = context;
      Selector = selector;
      MaxNodes = maxNodes;
      MaxNodesNN = maxNodesNN;
      defaultMaxNodesNN = maxNodesNN;

      BlockApply = blockApply;

      if (Context.ParamsSearch.Execution.NNEvaluatorBatchSizeBreakHints != null)
      {
        // The process of trimming the set of NN nodes involves aborting leaf nodes already created.
        Debug.Assert(MCTSParamsFixed.UNINITIALIZED_TREE_NODES_ALLOWED);

        // Not compatible with in flight linkage, since the node might be aborted
        // after having already been pointed to by another node.
        Debug.Assert(!Context.ParamsSearch.Execution.InFlightThisBatchLinkageEnabled);
      }


      this.IN_FLIGHT_THIS_BATCH_LINKAGE_ENABLED = IN_FLIGHT_THIS_BATCH_LINKAGE_ENABLED;
      this.IN_FLIGHT_OTHER_BATCH_LINKAGE_ENABLED = IN_FLIGHT_OTHER_BATCH_LINKAGE_ENABLED;

      NodesImmediateNotYetApplied = new ListBounded<MCTSNode>(maxNodes);
      NodesNN = new ListBounded<MCTSNode>(maxNodes);
      NodesTranspositionInFlightThisBatchLinked = new ListBounded<MCTSNode>(maxNodes);
      transpositionRootsOtherBatch = new Dictionary<ulong, MCTSNode>(maxNodes);

      if (IN_FLIGHT_THIS_BATCH_LINKAGE_ENABLED)
      {
        transpositionRootsThisBatch = new Dictionary<ulong, MCTSNode>(32);
      }
    }

    /// <summary>
    /// Resets the state of the set for another batch.
    /// </summary>
    /// <param name="nodesOtherBatch"></param>
    public void Reset(MCTSNodesSelectedSet nodesOtherBatch)
    {
      MaxNodesNN = defaultMaxNodesNN;
      NodesOtherBatch = nodesOtherBatch;

      NodesImmediateNotYetApplied.Clear(false);
      NodesNN.Clear(false);
      NodesTranspositionInFlightThisBatchLinked.Clear(false);
      transpositionRootsThisBatch?.Clear();
      indicesOtherNodesAlreadyInFlight?.Clear();
      NodesTranspositionInFlightOtherBatchLinked?.Clear(false);
      transpositionRootsOtherBatch?.Clear();

      NumNewLeafsAddedNonDuplicates = 0;
      NumNodesImmediatelyApplied = 0;
      NumLeafsFromNodesImmediatelyApplied = 0;
      NumCacheOnly = 0;
      NumNotApply = 0;

      haveAppliedThisBatch = false;
      haveAppliedOtherBatch = false;

      if (nodesOtherBatch != null && indicesOtherNodesAlreadyInFlight == null)
      {
        indicesOtherNodesAlreadyInFlight = new HashSet<int>(MaxNodes);
        if (IN_FLIGHT_OTHER_BATCH_LINKAGE_ENABLED)
        {
          NodesTranspositionInFlightOtherBatchLinked = new ListBounded<MCTSNode>(MaxNodes);
          transpositionRootsOtherBatch = new Dictionary<ulong, MCTSNode>(nodesOtherBatch.NodesNN.Count);
        }
      }

      if (nodesOtherBatch != null)
      {
        // Initialize indicesOtherNodesAlreadyInFlight with those nodes
        // that are already in flight from an overlapped selector 
        // (so we can ignore them as duplicates)
        AddOtherNodesInFlight(nodesOtherBatch.NodesNN, true);
        AddOtherNodesInFlight(nodesOtherBatch.NodesTranspositionInFlightThisBatchLinked, false);
        AddOtherNodesInFlight(nodesOtherBatch.NodesTranspositionInFlightOtherBatchLinked, false);
        AddOtherNodesInFlight(nodesOtherBatch.NodesImmediateNotYetApplied, false);
      }
    }

    private void AddOtherNodesInFlight(ListBounded<MCTSNode> otherNodesInFlight, bool eligibleForTranspositionLinkage)
    {
      if (otherNodesInFlight != null)
      {
        foreach (MCTSNode nodeOther in otherNodesInFlight)
        {
          // Definitely remember index of node to check for duplication
          indicesOtherNodesAlreadyInFlight.Add(nodeOther.Index);

          // Possibly use as an in-flight tranposition source
          if (IN_FLIGHT_OTHER_BATCH_LINKAGE_ENABLED && eligibleForTranspositionLinkage)
          {
            transpositionRootsOtherBatch[nodeOther.StructRef.ZobristHash] = nodeOther;
          }
        }
      }
    }

    public void AddSelectedNodes(ListBounded<MCTSNode> nodes, bool processImmediate)
    {
      if (nodes.Count != 0)
      {
        foreach (MCTSNode node in nodes)
        {
          ProcessNode(node);
        }

        if (processImmediate)
        {
          BlockApply.Apply(SelectorID, NodesImmediateNotYetApplied);
          NumNodesImmediatelyApplied += NodesImmediateNotYetApplied.Count;
          NodesImmediateNotYetApplied.Clear();
        }
      }
    }

    public void ProcessNode(MCTSNode node)
    {
      if (node.ActionType == MCTSNodeInfo.NodeActionType.CacheOnly)
      {
        NumCacheOnly++;
      }
      else if (node.ActionType == MCTSNodeInfo.NodeActionType.None)
      {
        NumNotApply++;
      }

      // ....................... NOT USED ........................
      if (node.StructRef.IsTranspositionLinked && node.N > 0
       && node.Context.ParamsSearch.Execution.TranspositionMode == TranspositionMode.MultiNodeBuffered)
      {
        throw new NotImplementedException();
        //ExtractTransposition(node);
      }
      // ........................................................

      // Case 1 - this is a duplicate of an a node already in flight in the other batch - will be ignored
      //          (when the other batch applies its nodes it will also backout the NInFlight from this batch)
      if (NodesOtherBatch != null)
      {
        if (indicesOtherNodesAlreadyInFlight.Contains(node.Index))
        {
          DropNode(node);
          return;
        }
      }

      NumNewLeafsAddedNonDuplicates++;

      // Case 2 - this is a node that was evaluated immediately (or terminal)
      bool canProcessImmediate = !node.EvalResult.IsNull 
                               || node.Terminal.IsTerminal() 
                               || node.NumVisitsPendingTranspositionRootExtraction > 0;
      if (canProcessImmediate)
      {
        if (node.ActionType == MCTSNodeInfo.NodeActionType.CacheOnly)
        {
          DropNode(node); // no point in adding or caching since can be resolved immediately
        }
        else
        {
          NodesImmediateNotYetApplied.Add(node);
        }

        return;
      }

      // Case 3  - already in flight for evaluation in the other batch
      MCTSNode inFlightLinkedNode;
      ulong hash = node.StructRef.ZobristHash;
      if (IN_FLIGHT_OTHER_BATCH_LINKAGE_ENABLED &&
          NodesOtherBatch != null &&
          transpositionRootsOtherBatch.TryGetValue(hash, out inFlightLinkedNode))
      {
        Debug.Assert(inFlightLinkedNode.IsNotNull);
        node.InFlightLinkedNode = inFlightLinkedNode;
        NodesTranspositionInFlightOtherBatchLinked.Add(node);
        return;
      }

      // Case 4 - already in flight for evaluation within this same batch
      if (IN_FLIGHT_THIS_BATCH_LINKAGE_ENABLED
           && transpositionRootsThisBatch.TryGetValue(hash, out inFlightLinkedNode)
           && node != inFlightLinkedNode)
      {
        Debug.Assert(inFlightLinkedNode.IsNotNull);
        node.InFlightLinkedNode = inFlightLinkedNode;
        NodesTranspositionInFlightThisBatchLinked.Add(node);
        return;
      }

      // Case 5 - not handled by any of the above cases, need to send to the neural network
      Debug.Assert(node.Terminal == Chess.GameResult.NotInitialized);
      if (NodesNN.Count >= MaxNodesNN)
      {
        // We already full. Abort immediately.
        if (SelectorID == 0)
        {
          node.StructRef.BackupAbort0(node.NInFlight);
        }
        else
        {
          node.StructRef.BackupAbort1(node.NInFlight2);
        }
      }
      else
      {
        NodesNN.Add(node);

        // Add this node to the "this batch" transpositions dictionary
        if (IN_FLIGHT_THIS_BATCH_LINKAGE_ENABLED)
        {
          transpositionRootsThisBatch[hash] = node;
        }
      }
    }

    /// <summary>
    /// Applies any NNEvaluatorBatchSizeBreakHints which may exist.
    /// </summary>
    public void ApplyBatchSizeBreakHints()
    {
      if (Context.ParamsSearch.Execution.NNEvaluatorBatchSizeBreakHints  != null)
      {
        foreach (int batchSize in Context.ParamsSearch.Execution.NNEvaluatorBatchSizeBreakHints)
        {
          ApplyBatchSizeBreakHint(batchSize);
        }
      }
    }

    /// <summary>
    /// Trims the size of NodesNN if it is just slightly larger than the break hint.
    /// </summary>
    /// <param name="batchSizeBreakHint"></param>
    void ApplyBatchSizeBreakHint(int batchSizeBreakHint)
    {
      // Truncate batch if just slightly larger than the hint
      int maxOverage = batchSizeBreakHint / 10;
      if (NodesNN.Count > batchSizeBreakHint && NodesNN.Count < batchSizeBreakHint + maxOverage)
      {
//MCTSEventSource.TestCounter1++;
        int startCount = NodesNN.Count;
        for (int i=startCount-1;i>= batchSizeBreakHint; i--)
        {
          MCTSNode node = NodesNN[i];
          NodesNN.RemoveAt(i);

          if (SelectorID == 0)
          {
            node.StructRef.BackupAbort0(node.NInFlight);
          }
          else
          {
            node.StructRef.BackupAbort1(node.NInFlight2);
          }
        }
      }
    }

    private void DropNode(MCTSNode node)
    {
      // Duplicate with other batch. Abort.
      // TODO: NOTE: if this is terminal, then we could probably still keep this, multivisits allowed
      if (SelectorID == 0)
      {
        node.StructRef.BackupAbort0(node.NInFlight);
      }
      else
      {
        node.StructRef.BackupAbort1(node.NInFlight2);
      }

#if DEBUG
      TotalNumDualSelectorDuplicates++;
#endif
    }

    public void ApplyImmeditateNotYetApplied()
    {
      BlockApply.Apply(SelectorID, NodesImmediateNotYetApplied);
      NodesImmediateNotYetApplied.Clear();
    }

    public void ApplyAll()
    {
      // Note that the in flight nodes need to transfer
      // their values from the ordinary nodes which are processed below.
      // Putting these first insures the policies
      // copied out of the evaluation results before
      // they are released in the apply process.
      PossiblyApplyInFlightThisBatchLinked();
      PossiblyApplyInFlightOtherBatchLinked();

      ApplyImmeditateNotYetApplied();

      BlockApply.Apply(SelectorID, NodesNN);


      // To prevent memory usage of chaining of every set to every prior set
      // we truncate prior batch now that it has been applied
      NodesOtherBatch = null;

      Debug.Assert((SelectorID == 0 ? Context.Root.NInFlight : Context.Root.NInFlight2) == 0);
    }


    void PossiblyApplyInFlightThisBatchLinked()
    {
      Debug.Assert(!haveAppliedThisBatch);
      if (IN_FLIGHT_THIS_BATCH_LINKAGE_ENABLED)
      {
        // Transfer the Eval results from the (now evaluated) nodes in this batch
        foreach (MCTSNode nodeInFlight in NodesTranspositionInFlightThisBatchLinked)
        {
          Debug.Assert(nodeInFlight.InFlightLinkedNode.IsNotNull);
          Debug.Assert(!nodeInFlight.InFlightLinkedNode.EvalResult.IsNull);
          nodeInFlight.EvalResult = nodeInFlight.InFlightLinkedNode.EvalResult;
          // no need nodeInFlight.OverrideVToApplyFromTransposition = nodeInFlight.InFlightLinkedNode.OverrideVToApplyFromTransposition;
        }

        BlockApply.Apply(SelectorID, NodesTranspositionInFlightThisBatchLinked);
        haveAppliedThisBatch = true;
      }
    }

    void PossiblyApplyInFlightOtherBatchLinked()
    {
      Debug.Assert(!haveAppliedOtherBatch);
      if (IN_FLIGHT_OTHER_BATCH_LINKAGE_ENABLED)
      {
        // Transfer the Eval results from the (now evaluated) nodes in the other batch
        foreach (MCTSNode nodeInFlight in NodesTranspositionInFlightOtherBatchLinked)
        {
          Debug.Assert(!nodeInFlight.InFlightLinkedNode.EvalResult.IsNull);
          nodeInFlight.EvalResult = nodeInFlight.InFlightLinkedNode.EvalResult;
          // no need nodeInFlight.OverrideVToApplyFromTransposition = nodeInFlight.InFlightLinkedNode.OverrideVToApplyFromTransposition;
        }
        BlockApply.Apply(SelectorID, NodesTranspositionInFlightOtherBatchLinked);
        haveAppliedOtherBatch = true;
      }
    }

  }
}
