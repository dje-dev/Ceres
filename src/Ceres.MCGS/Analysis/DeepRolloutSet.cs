#region Using directives

using System;
using System.Collections.Generic;
using System.Linq;

using Ceres.Base.Benchmarking;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;

#endregion

#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

namespace Ceres.MCGS.Analysis;

/// <summary>
/// Detailed results of the deep rollouts launched from a single start node
/// (typically a principal position, see PrincipalPosSet).
///
/// All Q values are from the start node's side-to-move perspective.
/// Summary statistics (median depth, average/dispersion of leaf Q) are computed
/// over the maximal lines - the distinct explored lines that are not a strict
/// prefix of any other (the "tips" of the exploration below the node).
/// </summary>
public sealed class DeepRolloutNodeStats
{
  /// <summary>
  /// The start node from which the rollouts were launched.
  /// </summary>
  public NodeIndex Node { get; }

  /// <summary>
  /// Number of rollouts actually applied to (and backed up through) this node.
  /// </summary>
  public int NumVisits { get; }

  /// <summary>
  /// Maximum depth descended below this node across all rollouts.
  /// </summary>
  public int MaxDepthBelowNode { get; }

  /// <summary>
  /// Rollouts reaching a terminal that is a win for the side to move at this node.
  /// </summary>
  public int NumTerminalWin { get; }

  /// <summary>
  /// Rollouts reaching a terminal draw.
  /// </summary>
  public int NumTerminalDraw { get; }

  /// <summary>
  /// Rollouts reaching a terminal that is a loss for the side to move at this node.
  /// </summary>
  public int NumTerminalLoss { get; }

  /// <summary>
  /// If the node was dropped before completing all requested rounds because consecutive
  /// rounds produced no new distinct line (dry-up detection).
  /// </summary>
  public bool StoppedByDryUp { get; }

  /// <summary>
  /// If the node was dropped before completing all requested rounds because a rollout
  /// reached a terminal leaf (when stopNodeVisitsIfTerminalReached was set).
  /// </summary>
  public bool StoppedByTerminal { get; }

  /// <summary>
  /// Average leaf Q over all rollouts (weighted by visit frequency, including repeats of the same line).
  /// </summary>
  public double AvgLeafQAllPaths { get; }

  /// <summary>
  /// The maximal lines explored below this node: each is a node-index sequence starting
  /// at the node itself (index 0) and descending to its deepest reached leaf.
  /// </summary>
  public IReadOnlyList<NodeIndex[]> MaximalPaths { get; }

  /// <summary>
  /// Leaf Q of each maximal line (parallel to MaximalPaths).
  /// </summary>
  public IReadOnlyList<double> MaximalPathLeafQ { get; }

  /// <summary>
  /// Depth below this node of each maximal line's leaf (parallel to MaximalPaths).
  /// </summary>
  public IReadOnlyList<int> MaximalPathDepths { get; }


  /// <summary>
  /// Number of distinct maximal lines explored below this node.
  /// </summary>
  public int NumDistinctPaths => MaximalPaths.Count;

  /// <summary>
  /// Median depth below this node over the maximal lines (NaN if none).
  /// </summary>
  public double MedianDepthBelowNode { get; }

  /// <summary>
  /// Average leaf Q over the maximal lines (NaN if none).
  /// </summary>
  public double AvgLeafQ { get; }

  /// <summary>
  /// Population standard deviation of leaf Q over the maximal lines (NaN if none).
  /// </summary>
  public double StdDevLeafQ { get; }

  /// <summary>
  /// Minimum leaf Q over the maximal lines (NaN if none).
  /// </summary>
  public double MinLeafQ { get; }

  /// <summary>
  /// Maximum leaf Q over the maximal lines (NaN if none).
  /// </summary>
  public double MaxLeafQ { get; }


  internal DeepRolloutNodeStats(NodeIndex node, int numVisits, int maxDepthBelowNode,
                                int numTerminalWin, int numTerminalDraw, int numTerminalLoss,
                                double avgLeafQAllPaths,
                                List<NodeIndex[]> maximalPaths, List<double> maximalPathLeafQ,
                                bool stoppedByDryUp = false, bool stoppedByTerminal = false)
  {
    Node = node;
    NumVisits = numVisits;
    MaxDepthBelowNode = maxDepthBelowNode;
    NumTerminalWin = numTerminalWin;
    NumTerminalDraw = numTerminalDraw;
    NumTerminalLoss = numTerminalLoss;
    AvgLeafQAllPaths = avgLeafQAllPaths;
    MaximalPaths = maximalPaths;
    MaximalPathLeafQ = maximalPathLeafQ;
    StoppedByDryUp = stoppedByDryUp;
    StoppedByTerminal = stoppedByTerminal;

    int[] depths = new int[maximalPaths.Count];
    for (int i = 0; i < maximalPaths.Count; i++)
    {
      depths[i] = maximalPaths[i].Length - 1;
    }
    MaximalPathDepths = depths;

    if (maximalPaths.Count == 0)
    {
      MedianDepthBelowNode = double.NaN;
      AvgLeafQ = double.NaN;
      StdDevLeafQ = double.NaN;
      MinLeafQ = double.NaN;
      MaxLeafQ = double.NaN;
    }
    else
    {
      int[] sortedDepths = (int[])depths.Clone();
      Array.Sort(sortedDepths);
      int mid = sortedDepths.Length / 2;
      MedianDepthBelowNode = sortedDepths.Length % 2 == 1
                               ? sortedDepths[mid]
                               : (sortedDepths[mid - 1] + sortedDepths[mid]) / 2.0;

      AvgLeafQ = maximalPathLeafQ.Average();
      MinLeafQ = maximalPathLeafQ.Min();
      MaxLeafQ = maximalPathLeafQ.Max();

      double sumSquaredDeviations = 0;
      foreach (double q in maximalPathLeafQ)
      {
        sumSquaredDeviations += (q - AvgLeafQ) * (q - AvgLeafQ);
      }
      StdDevLeafQ = Math.Sqrt(sumSquaredDeviations / maximalPathLeafQ.Count);
    }
  }
}


/// <summary>
/// Runs a set of deep rollouts - repeated low (or zero) exploration descents driven to
/// great depth - from a set of inner nodes of an existing MCGS search graph (typically the
/// principal positions collected by PrincipalPosSet) and retains the detailed per-node
/// results, making summary statistics available (see DeepRolloutNodeStats).
/// </summary>
public sealed class DeepRolloutSet
{
  /// <summary>
  /// The manager whose graph was rolled out.
  /// </summary>
  public MCGSManager Manager { get; }

  /// <summary>
  /// Maximum number of rollouts applied to each start node.
  /// </summary>
  public int NumVisitsPerNode { get; }

  /// <summary>
  /// Multiplier applied to the selection exploration term during the rollouts
  /// (0 yields greedy descents).
  /// </summary>
  public float ExplorationMultiplier { get; }

  /// <summary>
  /// Timing statistics for the rollout run.
  /// </summary>
  public TimingStats TimingStats { get; }

  /// <summary>
  /// Per-start-node results, in input node order
  /// (nodes that could not be rolled out are omitted).
  /// </summary>
  public IReadOnlyList<DeepRolloutNodeStats> Results { get; }

  readonly Dictionary<NodeIndex, DeepRolloutNodeStats> resultsByNode;


  private DeepRolloutSet(MCGSManager manager, int numVisitsPerNode, float explorationMultiplier,
                         TimingStats timingStats, List<DeepRolloutNodeStats> results)
  {
    Manager = manager;
    NumVisitsPerNode = numVisitsPerNode;
    ExplorationMultiplier = explorationMultiplier;
    TimingStats = timingStats;
    Results = results;

    resultsByNode = new Dictionary<NodeIndex, DeepRolloutNodeStats>(results.Count);
    foreach (DeepRolloutNodeStats result in results)
    {
      resultsByNode[result.Node] = result;
    }
  }


  /// <summary>
  /// Returns the rollout statistics for the specified start node, if it was rolled out.
  /// </summary>
  /// <param name="node"></param>
  /// <param name="stats"></param>
  /// <returns></returns>
  public bool TryGetStats(NodeIndex node, out DeepRolloutNodeStats stats)
    => resultsByNode.TryGetValue(node, out stats);


  /// <summary>
  /// Runs deep rollouts from the principal positions of the specified set.
  /// </summary>
  /// <param name="manager">Manager whose graph contains the nodes (search must not be running).</param>
  /// <param name="principalPositions">Principal positions whose nodes are used as rollout start nodes.</param>
  /// <param name="numVisitsPerNode">Maximum number of rollouts to apply to each node.</param>
  /// <param name="explorationMultiplier">Multiplier applied to the selection exploration term (0 = greedy).</param>
  /// <param name="stopNodeVisitsIfTerminalReached">If true, stop rolling out a node once one of its rollouts reaches a terminal.</param>
  /// <returns></returns>
  public static DeepRolloutSet Run(MCGSManager manager, PrincipalPosSet principalPositions,
                                   int numVisitsPerNode, float explorationMultiplier,
                                   bool stopNodeVisitsIfTerminalReached = true)
  {
    ArgumentNullException.ThrowIfNull(principalPositions);

    NodeIndex[] startNodes = new NodeIndex[principalPositions.Members.Count];
    for (int i = 0; i < startNodes.Length; i++)
    {
      startNodes[i] = principalPositions.Members[i].LeafNode.Index;
    }

    return Run(manager, startNodes, numVisitsPerNode, explorationMultiplier, stopNodeVisitsIfTerminalReached);
  }


  /// <summary>
  /// Runs deep rollouts from the specified inner nodes.
  /// Nodes that cannot be rolled out (terminal, unevaluated, the search root, etc.)
  /// are silently skipped and omitted from the results.
  /// </summary>
  /// <param name="manager">Manager whose graph contains the nodes (search must not be running).</param>
  /// <param name="startNodes">The inner nodes from which to launch rollouts.</param>
  /// <param name="numVisitsPerNode">Maximum number of rollouts to apply to each node.</param>
  /// <param name="explorationMultiplier">Multiplier applied to the selection exploration term (0 = greedy).</param>
  /// <param name="stopNodeVisitsIfTerminalReached">If true, stop rolling out a node once one of its rollouts reaches a terminal.</param>
  /// <param name="deadline">Optional wall-clock deadline; remaining rollout rounds are abandoned once passed.</param>
  /// <param name="dryUpRounds">If positive, stop rolling out a node once this many consecutive rounds yielded no new distinct line.</param>
  /// <returns></returns>
  public static DeepRolloutSet Run(MCGSManager manager, NodeIndex[] startNodes,
                                   int numVisitsPerNode, float explorationMultiplier,
                                   bool stopNodeVisitsIfTerminalReached = true,
                                   DateTime? deadline = null,
                                   int dryUpRounds = 0)
  {
    ArgumentNullException.ThrowIfNull(manager);
    ArgumentNullException.ThrowIfNull(startNodes);

    TimingStats timingStats = manager.DoSearchInnerNodes(startNodes, numVisitsPerNode, out var nodeStats,
                                                         explorationMultiplier, stopNodeVisitsIfTerminalReached,
                                                         deepRollout: true,
                                                         deadline: deadline, dryUpRounds: dryUpRounds);

    List<DeepRolloutNodeStats> results = new(nodeStats.Count);
    foreach ((NodeIndex node, int numVisits, int maxDepthBelowNode,
              int numTerminalWin, int numTerminalDraw, int numTerminalLoss,
              double avgLeafQAllPaths, double _, List<NodeIndex[]> maximalPathsNodes,
              List<double> maximalPathsLeafQ, bool droppedDryUp, bool droppedTerminal) in nodeStats)
    {
      results.Add(new DeepRolloutNodeStats(node, numVisits, maxDepthBelowNode,
                                           numTerminalWin, numTerminalDraw, numTerminalLoss,
                                           avgLeafQAllPaths, maximalPathsNodes, maximalPathsLeafQ,
                                           droppedDryUp, droppedTerminal));
    }

    return new DeepRolloutSet(manager, numVisitsPerNode, explorationMultiplier, timingStats, results);
  }
}
