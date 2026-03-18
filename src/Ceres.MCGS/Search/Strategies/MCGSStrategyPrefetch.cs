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
using Ceres.Chess;

using Ceres.MCGS.Graphs.GEdges;
using Ceres.MCGS.Graphs.GNodes;

using Ceres.MCGS.Managers;
using Ceres.MCGS.Search.Coordination;

#endregion

namespace Ceres.MCGS.Search.Strategies;

/// <summary>
/// Select/backup strategy subclass intended for prefetching of nodes 
/// prior to commencment of actual search as a speed optimization.
/// 
/// The intent is to gather larges batches of almost-certainly needed NN evluations
/// as a preprocessing step before actual search begins.
/// Nodes in the graph are created and populated with NN evaluations (but visit counts of 0).
/// 
/// However the actual speed improvement may be very limited in practice.
/// Prefetching may succeed in supplying many/most of the needed NN evlauations
/// early in search, but (except for the first handful of visits)
/// will not succeed in supplying all of the needed NN evaluations.
/// Therefore a batch will still need to be sent to the evaluator, incurring latency.
/// </summary>
public sealed class MCGSStrategyPrefetch : MCGSSelectBackupStrategyBase
{
  #region Helper Accept Delegates

  public static bool AcceptMinProbabilityPct(GNode pos, int childIndex, float minProbabilityPct)
   => 100 * pos.ChildEdgeAtIndex(childIndex).P > minProbabilityPct;

  public static bool AcceptMinRelativeProbabilityPct(GNode pos, int childIndex, float minRelativeProbabilityPct)
    => 100 * (pos.ChildEdgeAtIndex(0).P - pos.ChildEdgeAtIndex(childIndex).P) <= minRelativeProbabilityPct;

  public static bool AcceptMinProbabilityPct(GNode pos, int childIndex, float minProbabilityPct, float maxProbabilityPctGapFromBest)
  {
    float edgeP = 100f * pos.ChildEdgeAtIndex(childIndex).P;
    //      Console.WriteLine(edgeP + " vs " + minProbabilityPct);
    if (edgeP < minProbabilityPct)
    {
      return false;
    }

    float bestEdgeP = 100f * pos.ChildEdgeAtIndex(0).P; // Assume first child is the best.
    float gapP = bestEdgeP - edgeP;
    //      Console.WriteLine(gapP + " vs gap " + maxProbabilityPctGapFromBest);
    if (gapP > maxProbabilityPctGapFromBest)
    {
      return false;
    }

    return true;
  }

  #endregion

  /// <summary>
  /// Depth below which prefetch will not continue.
  /// </summary>
  public int MaxDepth { init; get; }

  /// <summary>
  /// Maximum number of children evaluated per node.
  /// </summary>
  public int MaxWidth { init; get; }


  /// <summary>
  /// Predicate that determines if prefetch should include child
  /// with given parent and give index in the parent.
  /// </summary>
  public readonly Func<GNode, int, bool> MoveAccessorPredicate;

  /// <summary>
  /// If debugging information should be logged to Console.
  /// </summary>
  public readonly bool Verbose;



  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="engine"></param>
  /// <param name="maxDepth"></param>
  /// <param name="maxWidth"></param>
  /// <param name="moveAccessorPredicate"></param>
  /// <param name="verbose"></param>
  public MCGSStrategyPrefetch(MCGSEngine engine,
                              int maxDepth,
                              int maxWidth,
                              Func<GNode, int, bool> moveAccessorPredicate,
                              bool verbose = false)
    : base(engine)
  {
    MaxDepth = maxDepth;
    MaxWidth = maxWidth;
    MoveAccessorPredicate = moveAccessorPredicate;
    Verbose = verbose;
  }


  /// <summary>
  /// Maximum number of children to consider when selecting child to visit.
  /// 
  /// We consider children even if already expanded, so do not limit.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="numTargetVisits"></param>
  /// <returns></returns>
  internal override int NumChildrenToConsider(GNode node, int numTargetVisits) => int.MaxValue;

  public const int MAX_CHILDREN = 64; // TODO: figure out where this is already defined

  [ThreadStatic]
  static short[] childVisitCountsBuffer;

  public override NodeSelectAccumulator SelectChildren(GNode parentNode,
                                                       int iteratorID,
                                                       int depth,
                                                       int numChildrenToConsider,
                                                       int numTargetVisits,
                                                       bool alsoComputeScores,
                                                       float explorationMultiplier,
                                                       float temperatureMultiplier,
                                                       bool refreshStaleEdges,
                                                       MCGSFutilityPruningStatus[] rootMovePruningStatus,
                                                       out Span<short> childVisitCounts,
                                                       out Span<double> childScores)
  {
    if (numTargetVisits != 1)
    {
      throw new Exception("Prefetch requires selection only one at a time so best child can be chosen.");
    }

    childScores = Span<double>.Empty;
    childVisitCountsBuffer ??= new short[MAX_CHILDREN];
    childVisitCounts = childVisitCountsBuffer;

    if (parentNode.NodeRef.Terminal.IsTerminal() || depth > MaxDepth + 1)
    {
      return new NodeSelectAccumulator(int.MinValue, double.NaN, double.NaN, 0); // Reached depth limit. This may case an aborted select.
    }

    int? indexBestChild = null;
    double scoreBestChlid = double.MinValue;

    // Assign scores to all children within the max width:
    //   - 0 if rejected by delegate
    //   - 0 if we are at max depth and already expanded
    //   - otherwise 1 / (1 + N + NInFlight)
    for (int i = 0; i < MaxWidth && i < parentNode.NodeRef.NumPolicyMoves; i++)
    {
      bool isExpanded = i < parentNode.NumEdgesExpanded;
      if (isExpanded && depth >= MaxDepth)
      {
        // Already expanded and we are at max depth, do not consider.
        if (alsoComputeScores)
        {
          childScores[i] = 0;
        }
      }
      else if (MoveAccessorPredicate != null && !MoveAccessorPredicate(parentNode, i))
      {
        // Rejected.
        if (alsoComputeScores)
        {
          childScores[i] = 0;
        }
      }
      else
      {
        int numVirtualChildren = isExpanded ? parentNode.ChildEdgeAtIndex(i).N
                                            + parentNode.ChildEdgeAtIndex(i).NumInFlight0 : 0;
        double thisScore = 1.0f / (1.0f + numVirtualChildren);
        if (alsoComputeScores)
        {
          childScores[i] = thisScore;
        }

        if (thisScore > scoreBestChlid)
        {
          scoreBestChlid = thisScore;
          indexBestChild = i;
        }
      }
    }

    if (indexBestChild is null)
    {
      return new NodeSelectAccumulator(int.MinValue, double.NaN, double.NaN, 0);
    }
    else
    {
      childVisitCounts[indexBestChild.Value] = 1;
      return new NodeSelectAccumulator(int.MinValue, double.NaN, double.NaN, 1);
    }
  }


  /// <summary>
  /// Backs up the value of a newly evaluated leaf node to a specified node.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="deltaN"></param>
  /// <param name="deltaW"></param>
  public override void BackupToNode(GNode node, int deltaN, double deltaW, double deltaD)
  {
  }


  /// <summary>
  /// Backs up the value of a newly evaluated leaf node to a specified edge.
  /// </summary>
  /// <param name="edge"></param>
  /// <param name="deltaN"></param>
  /// <param name="newChildQ"></param>
  /// <param name="newD"></param>
  /// <param name="drawKnownToExistAtChild"></param>
  public override void BackupToEdge(GEdge edge, int deltaN, double newChildQ, double newD, bool drawKnownToExistAtChild)
  {

  }

}
