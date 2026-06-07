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

using Ceres.Base.Benchmarking;
using Ceres.Chess;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.Coordination;

/// <summary>
/// Mutable accumulator of per-start-node statistics gathered across all rollout rounds of a single
/// MCGSManager.DoSearchInnerNodes call (see MCGSEngine.RunFromInnerNodes).
/// </summary>
internal sealed class InnerNodeRolloutStats
{
  /// <summary>Number of rollouts launched from (and backed up through) this node.</summary>
  public int NumVisits;

  /// <summary>Maximum depth descended below this node across all rollouts.</summary>
  public int MaxDepthBelowNode;

  /// <summary>Rollouts reaching a terminal that is a win for the side to move at this node.</summary>
  public int NumTerminalWin;

  /// <summary>Rollouts reaching a terminal draw.</summary>
  public int NumTerminalDraw;

  /// <summary>Rollouts reaching a terminal that is a loss for the side to move at this node.</summary>
  public int NumTerminalLoss;

  /// <summary>Sum of leaf Q (start-node perspective) over all rollouts (averaged to avgLeafQAllPaths).</summary>
  public double SumLeafQAllPaths;

  /// <summary>
  /// All distinct rollout lines (full node-index sequences; fully-overlapping duplicates dropped).
  /// Each starts at this node (index 0) and descends to the leaf; the root prefix is excluded. The
  /// maximal (tip) subset - lines that are not a strict prefix of any other - is derived from this
  /// when the result tuple is built.
  /// </summary>
  public readonly List<NodeIndex[]> DistinctSequences = new();

  /// <summary>Leaf Q (start-node perspective) for each entry in DistinctSequences (parallel list).</summary>
  public readonly List<double> DistinctLeafQ = new();
}


public partial class MCGSManager
{
  /// <summary>
  /// Grows the existing search graph by running additional visits that begin not at the search
  /// root but at the specified inner nodes ("deep rollouts").
  ///
  /// Processing proceeds in rounds (up to numVisitsEachNode of them). Each round sends exactly one
  /// visit to each still-active node: a single MCGSPath is formed per node spanning
  /// searchRoot -> node -> ... -> leaf, all such rollouts are aggregated and any leaves needing the
  /// neural network are evaluated together in one batch, and the rollouts are then backed up
  /// (propagating each leaf value up through the node and its ancestors to the search root).
  ///
  /// explorationMultiplier scales the exploration term used during the descent below each start
  /// node: for PUCT it multiplies CPUCT, for CB-GPUCT it multiplies the selection LambdaC. A value
  /// of 0 yields (effectively) greedy rollouts, which - combined with
  /// stopNodeVisitsIfTerminalReached = true - can be used to drive nodes to a terminal state.
  ///
  /// When stopNodeVisitsIfTerminalReached is true, any node whose rollout in a round reaches a
  /// terminal leaf is removed from subsequent rounds.
  ///
  /// Nodes that cannot be rolled out are silently skipped: a node must be an already-evaluated,
  /// already-visited, non-terminal strict descendant of the search root with at least one policy
  /// move (the search root itself is excluded - use DoSearch for that).
  /// </summary>
  /// <param name="nodes">The inner nodes from which to launch rollouts.</param>
  /// <param name="numVisitsEachNode">Maximum number of visits (rounds) to apply to each node.</param>
  /// <param name="nodeStats">
  /// Receives one tuple per rollable start node (in input order): the node; the number of visits
  /// actually applied; the maximum depth descended below it; the counts of rollouts reaching a
  /// terminal win / draw / loss; the average leaf Q over all rollouts and over the maximal lines;
  /// and the maximal node-index sequences explored from this node - the distinct lines that are not
  /// a strict prefix of any other explored line (the "tips" of the exploration tree), each starting
  /// at the node itself at index 0 and descending to its leaf (root prefix excluded). Win/draw/loss
  /// and the leaf Q values are all from that node's side-to-move perspective; averages are NaN when
  /// there are no contributing rollouts. Skipped nodes are omitted.
  /// </param>
  /// <param name="explorationMultiplier">Multiplier applied to the selection exploration term.</param>
  /// <param name="stopNodeVisitsIfTerminalReached">If true, stop visiting a node once its rollout reaches a terminal leaf.</param>
  /// <param name="progressCallback">Optional callback invoked periodically during the rollouts.</param>
  /// <param name="preBackupCallback">If not null, invoked with each round's constructed PathsSet just before it is backed up.</param>
  /// <param name="postBackupCallback">If not null, invoked with each round's PathsSet just after it is backed up.</param>
  /// <param name="deepRollout">
  /// If true, configures the rollouts for greedy depth extension to a true frontier / terminal:
  /// the transposition-sufficiency stop is bypassed (descents continue through well-visited
  /// transposition nodes) and a pessimistic FPU (FPUType.Absolute, FPUValue 1.0) is applied for the
  /// duration and then restored. Most useful together with explorationMultiplier = 0.
  /// </param>
  /// <returns>Timing statistics for the operation.</returns>
  internal TimingStats DoSearchInnerNodes(NodeIndex[] nodes, int numVisitsEachNode,
                                          out List<(NodeIndex node, int numVisits, int maxDepthBelowNode,
                                                    int numTerminalWin, int numTerminalDraw, int numTerminalLoss,
                                                    double avgLeafQAllPaths, double avgLeafQMaximalPaths,
                                                    List<NodeIndex[]> maximalPathsNodes)> nodeStats,
                                          float explorationMultiplier = 1,
                                          bool stopNodeVisitsIfTerminalReached = false,
                                          MCGSProgressCallback progressCallback = null,
                                          Action<MCGSPathsSet> preBackupCallback = null,
                                          Action<MCGSPathsSet> postBackupCallback = null,
                                          bool deepRollout = false)
  {
    nodeStats = new List<(NodeIndex node, int numVisits, int maxDepthBelowNode,
                          int numTerminalWin, int numTerminalDraw, int numTerminalLoss,
                          double avgLeafQAllPaths, double avgLeafQMaximalPaths,
                          List<NodeIndex[]> maximalPathsNodes)>();

    if (nodes == null)
    {
      throw new ArgumentNullException(nameof(nodes));
    }
    if (numVisitsEachNode <= 0)
    {
      throw new ArgumentOutOfRangeException(nameof(numVisitsEachNode), "numVisitsEachNode must be positive");
    }

    ProgressCallback = progressCallback;

    TimingStats stats = new();

    using (new TimingBlock($"MCGS SEARCH INNER NODES ({nodes.Length} nodes x {numVisitsEachNode})", stats, TimingBlock.LoggingType.None))
    {
      int numUsedNodes = Engine.Graph.Store.NodesStore.NumUsedNodes;

      // Map node indices to nodes, retaining only those that can actually be rolled out.
      List<GNode> startNodes = new(nodes.Length);
      int numSkipped = 0;
      foreach (NodeIndex nodeIndex in nodes)
      {
        if (nodeIndex.IsNull || nodeIndex.Index < 1 || nodeIndex.Index >= numUsedNodes)
        {
          numSkipped++;
          continue;
        }

        GNode node = Engine.Graph[nodeIndex];

        bool rollable = node.IsEvaluated
                     && node.N > 0
                     && !node.Terminal.IsTerminal()
                     && node.NumPolicyMoves > 0
                     && !node.IsSearchRoot;
        if (!rollable)
        {
          numSkipped++;
          continue;
        }

        startNodes.Add(node);
      }

      if (numSkipped > 0)
      {
        Console.WriteLine($"DoSearchInnerNodes: skipped {numSkipped} of {nodes.Length} node(s) "
                        + "(null, out of range, unevaluated, unvisited, terminal, no policy moves, or the search root).");
      }

      if (startNodes.Count > 0)
      {
        Dictionary<NodeIndex, InnerNodeRolloutStats> perNode =
          Engine.RunFromInnerNodes(startNodes.ToArray(), numVisitsEachNode,
                                   explorationMultiplier, stopNodeVisitsIfTerminalReached,
                                   deepRollout, preBackupCallback, postBackupCallback);

        // Emit one tuple per rollable start node, preserving input order.
        foreach (GNode node in startNodes)
        {
          if (perNode.TryGetValue(node.Index, out InnerNodeRolloutStats s))
          {
            // Maximal (tip) lines: distinct sequences that are not a strict prefix of any other.
            List<NodeIndex[]> maximalPathsNodes = new();
            double sumLeafQMaximal = 0;
            for (int i = 0; i < s.DistinctSequences.Count; i++)
            {
              if (!IsStrictPrefixOfAny(s.DistinctSequences, i))
              {
                maximalPathsNodes.Add(s.DistinctSequences[i]);
                sumLeafQMaximal += s.DistinctLeafQ[i];
              }
            }

            double avgLeafQAllPaths = s.NumVisits > 0 ? s.SumLeafQAllPaths / s.NumVisits : double.NaN;
            double avgLeafQMaximalPaths = maximalPathsNodes.Count > 0 ? sumLeafQMaximal / maximalPathsNodes.Count : double.NaN;
            nodeStats.Add((node.Index, s.NumVisits, s.MaxDepthBelowNode,
                           s.NumTerminalWin, s.NumTerminalDraw, s.NumTerminalLoss,
                           avgLeafQAllPaths, avgLeafQMaximalPaths, maximalPathsNodes));
          }
          else
          {
            nodeStats.Add((node.Index, 0, 0, 0, 0, 0, double.NaN, double.NaN, new List<NodeIndex[]>()));
          }
        }
      }
    }

    return stats;
  }


  /// <summary>
  /// Returns whether the sequence at index i is a strict prefix of some other sequence in the list
  /// (matches another line's start for its whole length, but is strictly shorter). Such a line is
  /// not maximal - it is subsumed by the longer line that extends it.
  /// </summary>
  /// <param name="sequences"></param>
  /// <param name="i"></param>
  /// <returns></returns>
  private static bool IsStrictPrefixOfAny(List<NodeIndex[]> sequences, int i)
  {
    NodeIndex[] candidate = sequences[i];
    for (int j = 0; j < sequences.Count; j++)
    {
      if (j == i)
      {
        continue;
      }

      NodeIndex[] other = sequences[j];
      if (other.Length <= candidate.Length)
      {
        continue;
      }

      bool isPrefix = true;
      for (int k = 0; k < candidate.Length; k++)
      {
        if (other[k].Index != candidate[k].Index)
        {
          isPrefix = false;
          break;
        }
      }

      if (isPrefix)
      {
        return true;
      }
    }

    return false;
  }
}
