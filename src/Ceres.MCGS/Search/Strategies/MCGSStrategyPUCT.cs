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
using System.Linq;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;

using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Managers;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PUCT;

#endregion

namespace Ceres.MCGS.Search.Strategies;

/// <summary>
/// Implements PUCT-based child selection and backup strategy.
/// </summary>
public sealed class MCGSStrategyPUCT : MCGSSelectBackupStrategyBase
{
  internal const bool VERBOSE_ACTION_HEAD = false;

  internal const bool ACTION_HEAD_CONSERVATIVE_RESORT_MODE = false;
  // if the top policy move is conservatively ineligible for rearrangement
  const bool ACTION_REARRANGE_PIN_TOP_POLICY_MOVE = ACTION_HEAD_CONSERVATIVE_RESORT_MODE;
  const int MAX_SWAPS = ACTION_HEAD_CONSERVATIVE_RESORT_MODE ? 2 : 5;
  internal const float ACTION_HEAD_FPU_VALUE = 0.10f;

  public ParamsSearch ParamsSearch => Engine.Manager.ParamsSearch;
  public ParamsSelect ParamsSelect => Engine.Manager.ParamsSelect;


  /// <summary>
  /// Allocate buffer to hold child visit counts (one for each possible path depth).
  /// // TODO: why do we need one for each possible depth?
  /// </summary>
  [ThreadStatic]
  static short[][] childVisitCountsArray;

  /// <summary>
  /// Allocate buffer to hold child visit scores (one for each possible path depth).
  /// </summary>
  [ThreadStatic]
  static double[][] childScoresArray;


  /// <summary>
  /// Local copy of variable for efficient access.
  /// </summary>
  private int backupOffPathNumAdditionalLevelsToPropagate;

  bool refreshSiblingDuringBackupToNode;


  /// <summary>
  /// Constructor.
  /// </summary>
  public MCGSStrategyPUCT(MCGSEngine engine) : base(engine)
  {
    backupOffPathNumAdditionalLevelsToPropagate = engine.Manager.ParamsSearch.OffPathBackupNumAdditionalLevelsToPropagate;
    refreshSiblingDuringBackupToNode = MCGSParamsFixed.REFRESH_SIBLING_DURING_BACKUP_PHASE && Engine.Manager.ParamsSearch.EnablePseudoTranspositionBlending;
  }

  private static void CheckThreadStaticsInitialized()
  {
    if (childVisitCountsArray == null)
    {
      const int MAX_PATH_DEPTH = 255;
      childVisitCountsArray = new short[MAX_PATH_DEPTH][];
      childScoresArray = new double[MAX_PATH_DEPTH][];
    }
  }

  /// <summary>
  /// Maximum number of children to consider when selecting child to visit.
  /// 
  /// All of the children visited continue to be eligible,
  /// plus enough additional children to satisfy the numTargetVisits
  /// under the pessimistic (unlikely) assumption that PUCT chose
  /// consecutive children starting at the beginning of the unexpanded children.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="numTargetVisits"></param>
  /// <returns></returns>
  internal override int NumChildrenToConsider(GNode node, int numTargetVisits) 
    => node.NumEdgesExpanded + numTargetVisits;




  /// <summary>
  /// Reorders children (originally sorted by prior, index 0 = best prior)
  /// by <paramref name="scores"/> descending, subject to the constraint that
  /// no child moves more than <paramref name="d"/> positions from its prior rank.
  /// Writes the resulting permutation (original indices) into <paramref name="result"/>.
  /// </summary>
  public static void SortDisplacementCapped(ReadOnlySpan<double> scores, int d, Span<int> result)
  {
    int n = scores.Length;
    if (result.Length != n)
      throw new ArgumentException("result must have same length as scores", nameof(result));
    if (d < 0)
      throw new ArgumentOutOfRangeException(nameof(d));

    // For small N (typical MCTS branching), a stack-allocated bool array
    // avoids any heap allocation.
    Span<bool> placed = n <= 256 ? stackalloc bool[n] : new bool[n];
    placed.Clear();

    for (int slot = 0; slot < n; slot++)
    {
      int lo = Math.Max(0, slot - d);
      int hi = Math.Min(n - 1, slot + d);

      // Deadline rule: any unplaced child i with i + d == slot MUST go now,
      // otherwise it would violate its upper displacement bound next iteration.
      int pick = -1;
      int forcedIdx = slot - d; // the only i where i + d == slot
      if (forcedIdx >= 0 && forcedIdx <= hi && !placed[forcedIdx])
      {
        pick = forcedIdx;
      }
      else
      {
        // Otherwise: pick the eligible unplaced child with the highest score.
        // Ties break toward the lower index (earlier prior rank).
        double bestScore = double.NegativeInfinity;
        for (int i = lo; i <= hi; i++)
        {
          if (placed[i]) continue;
          double s = scores[i];
          if (s > bestScore)
          {
            bestScore = s;
            pick = i;
          }
        }
      }

      placed[pick] = true;
      result[slot] = pick;
    }
  }



  /// <inheritdoc/>
  internal override void PossiblyActionResortUnvisitedChildren(GNode node, Graph graph)
  {
    if (ParamsSelect.ActionResortUnvisitedChildren)
    {
      PossiblyActionResortUsingAction(node, graph, ParamsSearch, ParamsSelect);
    }
  }


  void PossiblyActionResortUsingAction(GNode node, Graph graph, in ParamsSearch paramsSearch, in ParamsSelect paramsSelect)
  {
    const float MIN_PROBABILITY_FOR_ACTION_BOOST = 0;

    Debug.Assert(node.N == 1);
    if (paramsSelect.FPUMode != ParamsSelect.FPUType.ActionHead
      || node.NumPolicyMoves < 2)
    {
      return;
    }

    ref readonly GNodeStruct nodeRef = ref node.NodeRef;

    // If using action head and this is the first visit,
    // resort the children using the scores computed inclusive of the action head influence.

    // Compute scores for all children so we can sort based on that.
    Span<double> scores = stackalloc double[nodeRef.NumPolicyMoves];
    PUCTSelector.ComputeTopChildScores(graph, node,
                                       paramsSearch,
                                       paramsSelect,
                                       0,
                                       false,
                                       rootMovePruningStatus: null,
                                       dualCollisionFraction: ParamsSearch.Execution.DualIteratorAlternateCollisionFraction,
                                       minChildIndex: 0,
                                       maxChildIndex: nodeRef.NumPolicyMoves - 1,
                                       numTargetVisits: 0, // to indicate all
                                       scores: scores,
                                       childVisitCounts: default,// not needed
                                       cpuctMultiplier: 1.0f,
                                       temperatureMultiplier: 1.0f);

    // Compute baseline scores (without ActionHead, using default FPU settings)
    double[] baselineScores = null;
    if (VERBOSE_ACTION_HEAD)
    {
      baselineScores = new double[nodeRef.NumPolicyMoves];
      ParamsSelect defaultParams = new ParamsSelect();
      PUCTSelector.ComputeTopChildScores(graph, node,
                                         paramsSearch,
                                         defaultParams,
                                         0,
                                         false,
                                         rootMovePruningStatus: null,
                                         dualCollisionFraction: ParamsSearch.Execution.DualIteratorAlternateCollisionFraction,
                                         minChildIndex: 0,
                                         maxChildIndex: nodeRef.NumPolicyMoves - 1,
                                         numTargetVisits: 0,
                                         scores: baselineScores,
                                         childVisitCounts: default,
                                         cpuctMultiplier: 1.0f,
                                         temperatureMultiplier: 1.0f);
    }

    Span<GEdgeHeaderStruct> moveInfosSpan = node.EdgeHeadersSpan;

    // Compute FEN for verbose output
    string fen = VERBOSE_ACTION_HEAD ? node.CalcPosition().ToPosition.FEN : null;

    if (MAX_SWAPS > 0)
    {
      if (ACTION_REARRANGE_PIN_TOP_POLICY_MOVE)
      {
        ApplyDisplacementCappedSort(scores[1..], moveInfosSpan[1..], MAX_SWAPS, MIN_PROBABILITY_FOR_ACTION_BOOST,
                                    baselineScores != null ? baselineScores.AsSpan(1) : default,
                                    nodeRef.V, nodeRef.UncertaintyValue, nodeRef.UncertaintyPolicy, fen);
      }
      else
      {
        ApplyDisplacementCappedSort(scores, moveInfosSpan, MAX_SWAPS, MIN_PROBABILITY_FOR_ACTION_BOOST,
                                    baselineScores,
                                    nodeRef.V, nodeRef.UncertaintyValue, nodeRef.UncertaintyPolicy, fen);
      }
    }
    else
    {
      const double DIFF_THRESHOLD = 0.02;

      // Bubble sort to get items in same order as the scores.
      int numSwapped;
      do
      {
        // First slot is fixed, consider rearranging thereafter.
        numSwapped = 0;
        for (int i = 1; i < nodeRef.NumPolicyMoves; i++)
        {
          // Do not swap for low probability nodes.
          if (moveInfosSpan[i].P < MIN_PROBABILITY_FOR_ACTION_BOOST)
          {
            break;
          }

          double scoreDiff = scores[i] - scores[i - 1];
          if (scoreDiff > DIFF_THRESHOLD)
          {
            numSwapped++;

            // Swap scores, children and actions.
            (scores[i - 1], scores[i]) = (scores[i], scores[i - 1]);
            (moveInfosSpan[i - 1], moveInfosSpan[i]) = (moveInfosSpan[i], moveInfosSpan[i - 1]);
          }
        }
      } while (numSwapped > 0);
    }
  }


  /// <summary>
  /// Applies a displacement-capped sort to reorder moveInfosSpan based on scores,
  /// ensuring no element moves more than maxDisplacement positions from its original index.
  /// </summary>
  private static void ApplyDisplacementCappedSort(Span<double> scores, Span<GEdgeHeaderStruct> moveInfosSpan,
                                                  int maxDisplacement, float minProbabilityForBoost,
                                                  ReadOnlySpan<double> baselineScores,
                                                  float nodeV, float nodeUV, float nodeUP,
                                                  string fen)
  {
    int n = scores.Length;

    // Get the permutation from SortDisplacementCapped.
    Span<int> permutation = stackalloc int[n];
    SortDisplacementCapped(scores, maxDisplacement, permutation);

    if (VERBOSE_ACTION_HEAD)
    {
      DumpActionHeadResortInfo(scores, moveInfosSpan, permutation, baselineScores, nodeV, nodeUV, nodeUP, fen);
    }

    // Apply the permutation to moveInfosSpan.
    // Use a temporary buffer to hold the reordered elements.
    Span<GEdgeHeaderStruct> temp = stackalloc GEdgeHeaderStruct[n];

    for (int i = 0; i < n; i++)
    {
      temp[i] = moveInfosSpan[permutation[i]];
    }

    // Copy back to original span (only for elements with sufficient probability).
    for (int i = 0; i < n; i++)
    {
      // Do not reorder low probability nodes.
      if (temp[i].P < minProbabilityForBoost)
      {
        break;
      }
      moveInfosSpan[i] = temp[i];
    }
  }


  /// <summary>
  /// Dumps verbose information about action head resorting including:
  /// - Baseline scores (default FPU mode, as from default ParamsSelect())
  /// - Action head V values (raw neural network action head output)
  /// - Action head scores (used for sorting)
  /// - New positions after displacement-capped sort
  /// </summary>
  private static void DumpActionHeadResortInfo(ReadOnlySpan<double> scores,
                                               ReadOnlySpan<GEdgeHeaderStruct> moveInfosSpan,
                                               ReadOnlySpan<int> permutation,
                                               ReadOnlySpan<double> baselineScores,
                                               float nodeV, float nodeUV, float nodeUP,
                                               string fen)
  {
    int n = scores.Length;

    // Build reverse mapping: for each original index, what's its new position?
    Span<int> newPositions = stackalloc int[n];
    for (int newPos = 0; newPos < n; newPos++)
    {
      int origIdx = permutation[newPos];
      newPositions[origIdx] = newPos;
    }

    // Copy data to arrays to avoid span-in-lambda issues
    string[] moveStrs = new string[n];
    double[] policies = new double[n];
    double[] actionVValues = new double[n];
    double[] actionHeadScores = new double[n];
    double[] baselineScoresArray = new double[n];
    int[] newPosArray = new int[n];

    for (int i = 0; i < n; i++)
    {
      moveStrs[i] = moveInfosSpan[i].Move.ToString();
      policies[i] = moveInfosSpan[i].P;
#if ACTION_ENABLED
      actionVValues[i] = moveInfosSpan[i].ActionV;
#endif
      actionHeadScores[i] = scores[i];
      baselineScoresArray[i] = baselineScores.IsEmpty ? double.NaN : baselineScores[i];
      newPosArray[i] = newPositions[i];
    }

    string FormatRow(string label, Func<int, string> valueFunc)
    {
      var values = Enumerable.Range(0, n).Select(i => valueFunc(i));
      return $"{label,-25} " + string.Join(" ", values);
    }

    Console.WriteLine();
    Console.WriteLine($"=== Action Head Resort (n={n}, V={nodeV:0.00}, UV={nodeUV:0.00}, UP={nodeUP:0.00}) {fen} ===");
    Console.WriteLine(FormatRow("Moves:", i => moveStrs[i].PadLeft(6)));
    Console.WriteLine(FormatRow("Policy:", i => (100 * policies[i]).ToString("0.0").PadLeft(5) + "%"));
#if ACTION_ENABLED
    Console.WriteLine(FormatRow("Action V:", i => actionVValues[i].ToString("0.00").PadLeft(6)));
#endif
    if (!baselineScores.IsEmpty)
    {
      Console.WriteLine(FormatRow("Scores (baseline):", i => baselineScoresArray[i].ToString("0.00").PadLeft(6)));
    }
    Console.WriteLine(FormatRow("Scores (ActionHead):", i => actionHeadScores[i].ToString("0.00").PadLeft(6)));
//    Console.WriteLine(FormatRow("Original position:", i => i.ToString().PadLeft(6)));
    Console.WriteLine(FormatRow("New position:", i =>
    {
      int delta = newPosArray[i] - i;
      if (delta == 0) return "      ";
      return (delta > 0 ? $"+{delta}" : delta.ToString()).PadLeft(6);
    }));
    Console.WriteLine();
  }


  public static void RearrangeForQOrder(Span<double> scores, double maxDelta, Action<int, int> swapFunc)
  {
    int length = scores.Length;
    bool changed;

    do
    {
      changed = false;

      // i1 is the later index
      for (int i1 = 0; i1 < length; i1++)
      {
        double s1 = scores[i1];

        // i2 is the earlier index
        for (int i2 = 0; i2 < i1; i2++)
        {
          double s2 = scores[i2];

          // Violation: earlier score is more than maxDelta greater
          if (s2 > s1 + maxDelta)
          {
            // Swap the two indices via callback (and also locally).
            swapFunc(i2, i1);
            (scores[i2], scores[i1]) = (scores[i1], scores[i2]);

            // scores has changed; refresh s1 from its new position.
            s1 = scores[i1];

            changed = true;
          }
        }
      }

    } while (changed);
  }


  public static void RearrangeForContiguity(Span<short> counts, Span<double> associatedScores, int indexFirstElementMustFillContiguously)
  {
    if (counts.Length <= indexFirstElementMustFillContiguously)
    {
      return;
    }

    int writePos = indexFirstElementMustFillContiguously;

    for (int readPos = indexFirstElementMustFillContiguously; readPos < counts.Length; readPos++)
    {
      if (counts[readPos] != 0)
      {
        if (readPos != writePos)
        {
          counts[writePos] = counts[readPos];
          counts[readPos] = 0;

          if (!associatedScores.IsEmpty)
          {
            associatedScores[writePos] = associatedScores[readPos];
            associatedScores[readPos] = 0;
          }
        }
        writePos++;
      }
    }
  }

  public override NodeSelectAccumulator SelectChildren(GNode parentNode,
                                                       int selectorID,
                                                       int depth,
                                                       int numChildrenToConsider,
                                                       int numTargetVisits,
                                                       bool alsoComputeScores,
                                                       float cpuctMultiplier,
                                                       float temperatureMultiplier,
                                                       bool refreshStaleEdges,
                                                       MCGSFutilityPruningStatus[] rootMovePruningStatus,
                                                       out Span<short> childVisitCounts,
                                                       out Span<double> childScores)
  {
    CheckThreadStaticsInitialized();

    // Allocate space to hold visit counts/scores at this level (if not already allocated).
    if (childVisitCountsArray[depth] == null)
    {
      childVisitCountsArray[depth] = new short[CompressedPolicyVector.NUM_MOVE_SLOTS];
      if (alsoComputeScores)
      {
        childScoresArray[depth] = new double[CompressedPolicyVector.NUM_MOVE_SLOTS];
      }
    }

    // Create properly sized and cleared spans for visit counts and scores.
    childVisitCounts = new Span<short>(childVisitCountsArray[depth], 0, numChildrenToConsider);
    childVisitCounts.Clear();
    if (alsoComputeScores)
    {
      childScores = new Span<double>(childScoresArray[depth], 0, numChildrenToConsider);
      childScores.Clear();
    }
    else
    {
      childScores = default;
    }

    NodeSelectAccumulator childStats =
      PUCTSelector.ComputeTopChildScores(Engine.Graph, parentNode,
                                         ParamsSearch,
                                         ParamsSelect,
                                         selectorID,
                                         refreshStaleEdges,
                                         rootMovePruningStatus,
                                         dualCollisionFraction: ParamsSearch.Execution.DualIteratorAlternateCollisionFraction,
                                         minChildIndex: 0,
                                         maxChildIndex: numChildrenToConsider - 1,
                                         numTargetVisits,
                                         childScores,
                                         childVisitCounts,
                                         cpuctMultiplier,
                                         temperatureMultiplier);

    if (Engine.Manager.ParamsSearch.MoveOrderingPhase != ParamsSearch.MoveOrderingPhaseEnum.None)
    {
      RearrangeForContiguity(childVisitCounts, childScores, parentNode.NumEdgesExpanded);
    }
#if DEBUG
    // Verify in-order
    bool haveSeenZero = false;
    for (int i = parentNode.NumEdgesExpanded; i < numChildrenToConsider; i++)
    {
      if (childVisitCounts[i] != 0)
      {
        Debug.Assert(!haveSeenZero);
      }
      else
      {
        haveSeenZero = true;
      }
    }
#endif

    return childStats;
  }



#if NOT
  Debug.Assert(Math.Abs(newQ) < 1.2f);
    node.NodeRef.Q = newQ;

    if (false && node.Graph.TestFlag && node.N > 1 && node.N <= 100 && !node.Terminal.IsTerminal())
    {
      GEdge rpoResult = RPOTests.BestMove(node, 
                                          qWhenNoChildren: (float)node.Q - 0.32f, // not actualy used
                                          numChildrenToConsider: node.NumEdgesExpanded,
                                          lambda: 0.5f,
                                          lambdaPower: 0.5f,
                                          weighted: false);

      // For total W, using value of node itself plus the Q coming from the RPO calculation
      double totalW = (-rpoResult.Q * (node.N - 1) )+ (node.V * 1);

      newQ = totalW / node.N;
      node.NodeRef.Q = newQ;
      node.NodeRef.D = 0; // ??

      if (includePseudoTranspositions)
      {
        throw new NotImplementedException("Need to disable this for comparability (not yet implemented)");
      }
    }

  }
#endif


  /// <summary>
  /// Backs up the value of a newly evaluated leaf node to a specified node.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="deltaN"></param>
  /// <param name="deltaW"></param>
  /// <param name="deltaD"></param>
  /// <param name="refreshSiblingContribution"></param>
  public override void BackupToNode(GNode node, int deltaN, double deltaW, double deltaD)
  {
    int startN = node.N;

    // Increment N.
    node.NodeRef.N += deltaN;

    if (node.CheckmateKnownToExistAmongChildren)
    {
      // Q should have already been set and will not change.
      Debug.Assert(node.Q >= 0.995);
    }
    else
    {
      double oldWPure = node.ComputeQPure() * startN;
      double newWPure = oldWPure + deltaW;
      double newQPure = newWPure / (startN + deltaN);

      node.ResetNodeQUsingNewQPure(newQPure, refreshSiblingDuringBackupToNode);

      // Update D using running average.
      double oldDSum = node.D * startN;
      double newDSum = oldDSum + deltaD;
      node.NodeRef.D = newDSum / (startN + deltaN);
    }

#if DEBUG
    int count = 0;
    for (int i=0;i<node.NumEdgesExpanded;i++)
    {
      count += node.ChildEdgeAtIndex(i).N;
    }
    Debug.Assert(node.N == count + (node.NodeRef.Terminal.IsTerminal() ? node.N : 1));
#endif
  }


  /// <summary>
  /// Backs up the value of a newly evaluated leaf node to a specified edge.
  /// </summary>
  /// <param name="edge"></param>
  /// <param name="deltaN"></param>
  /// <param name="deltaNDrawByRepetition"></param>
  /// <param name="newChildQ"></param>
  /// <param name="newD"></param>
  /// <param name="drawKnownToExistAtChild"></param>
  public override void BackupToEdge(GEdge edge, int deltaN,  double newQChild, double newD, bool drawKnownToExistAtChild)
  {
    Debug.Assert(Math.Abs(newQChild) < 1.2f);

    // If this backed up value looks to us better than a draw but
    // we know that the opponent has an available draw,
    // reset value to 0.
    if (MCGSParamsFixed.ENABLE_DRAW_KNOWN_TO_EXIST && drawKnownToExistAtChild && newQChild < 0)
    {
      edge.ChildNodeHasDrawKnownToExist = true;
      newQChild = 0;
      // TODO: deltaD adjust, should it be -deltaW?
    }

    double deltaQ = newQChild - edge.Q;

    if (MCGSParamsFixed.TRACK_NODE_EDGE_UNCERTAINTY)
    {
      double priorMean = edge.N <= 1 ? newQChild : edge.Q; // first point is just the new sample
      edge.AddUpdateSample(priorMean, newQChild);
    }

    // The N on edges represents the number of visits along this path to the child node,
    // independent of total visits to the child (some of which may have arisen via other parents).
    // (see: https://github.com/lightvector/KataGo/blob/master/docs/GraphSearch.md).
    edge.N += deltaN;
    edge.QChild = newQChild;

    // Note that the assertion on edge N and child N
    // does not apply in coalesce mode because in that case one node appears twice in the path
    // (once as the leaf) and that BackupToNode at the leaf is omitted
    // to allow the node to be updated exactly once (at the earlier visit).
//    Debug.Assert(Engine.Manager.ParamsSearch.PathTranspositionMode == Paths.MCGSPathMode.Coalesce
//            || !(edge.Type == GEdgeStruct.EdgeType.ChildEdge && edge.N > edge.ChildNode.N));

    // Potentially propagate Q changes upward to other parents of the child node.
    if (backupOffPathNumAdditionalLevelsToPropagate > 0
      && Math.Abs(deltaQ) > MCGSParamsFixed.PROPAGATE_OFF_VISIT_PARENTS_MIN_Q_DELTA
      && edge.Type == GEdgeStruct.EdgeType.ChildEdge
      && edge.ChildNode.NumParentsMoreThanOne)
    {
      Debug.Assert(edge.ParentNode.GraphStore.GraphEnabled); // stale flag only to be used in graph mode when desychronization possible

      PropagateQChangesUpward(edge.ChildNode, edge, backupOffPathNumAdditionalLevelsToPropagate);
    }
  }


  /// <summary>
  /// Recursively propagates Q value changes to other parent edges of a node,
  /// continuing upward through the graph as needed.
  /// </summary>
  /// <param name="childNode">The edge that was just updated</param>
  /// <param name="edgeToNotUpdate">Optionally an edge that should be ignored</param>
  /// <param name="levelsRemaining">Number of levels remaining before stopping propagation</param>
  private void PropagateQChangesUpward(GNode childNode, GEdge edgeToNotUpdate, int levelsRemaining)
  {
    if (levelsRemaining == 0)
    {
      return;
    }

    double newChildQ = childNode.Q;
    bool shouldContinueMoreLevels = levelsRemaining > 1;

    foreach (GEdge edge in childNode.ParentEdges)
    {
      if (edge != edgeToNotUpdate
        && edge.N > 0
        && !edge.ChildNodeHasDrawKnownToExist)
      {
        // Only mark stale if the delta of Q on the edge is also significant.
        // (it could happen that it was already stale in a way that cancelled with this staleness).
        double edgeDeltaQ = newChildQ - edge.Q;

        if (Math.Abs(edgeDeltaQ) > MCGSParamsFixed.PROPAGATE_OFF_VISIT_PARENTS_MIN_Q_DELTA)
        {
          if (!shouldContinueMoreLevels)
          {
            // Don't continue upward, but do mark this edge as stale so that future
            // visits to it (during select phase) will notice and update during gather of children.
            edge.IsStale = true;
          }
          else
          {
            // Update the parent node with the delta
            double deltaW = edgeDeltaQ * edge.N;
            double deltaD = 0; // TODO: implement this.
            BackupToNode(edge.ParentNode, 0, deltaW, deltaD);

            // Update the edge Q value
            edge.QChild = newChildQ;

            // Recursively propagate to this edge's other parents
            PropagateQChangesUpward(edge.ParentNode, default, levelsRemaining - 1);
          }
        }
      }
    }
  }
}

