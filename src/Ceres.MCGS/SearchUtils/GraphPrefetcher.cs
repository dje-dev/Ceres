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
using System.Threading;

using Ceres.Base.Benchmarking;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Strategies;

#endregion

namespace Ceres.MCGS.Search;

/// <summary>
/// Class to manage bulk prefetching of nodes in the graph.
/// 
/// High policy probability nodes near the root of the tree are initialized in the tree
/// (with neural network evals but with zero visits) using batches where possible.
/// </summary>
public class GraphPrefetcher
{
  /// <summary>
  /// Only rearrange (thereby overriding policy in determining selection order)
  /// if the value head disagreed by some significant margin.
  /// TODO: tune this value. Preliminary findings:
  ///  Threshold Games @250 nodes with C1-512-15
  ///  0.00 	               		 most differ,     --> -17 +/- 20 Elo	
  ///  0.10                      60% games differ --> 0 +/- 20 Elo
  ///  0.25  20% trees impacted  50% games differ --> 0 +/- 20 Elo
  ///  0.35  10% trees impacted, 35% games differ --> 7 +/- 20 Elo
  /// </summary>
  const float THRESHOLD_Q_REARRANGE = 0.35f;

  public readonly MCGSManager Manager;

  public TimingStats TimingStats { get; private set; }

  public int NumPrefetchPasses { get; private set; }

  public int NumNodesPrefetched { get; private set; }

  public int NumRearranged;

  public static int TotalNumRearranged;
  public static int TotalNumPrefetched;

  public GraphPrefetcher(MCGSManager manager)
  {
    Manager = manager;
  }


  /// <summary>
  /// Rearranges children to insure they are ordered by their V value.
  /// </summary>
  public void PossiblyResortChildrenUsingV()
  {
    DoPossiblyResortChildrenUsingV(Manager.Engine.Graph.GraphRootNode);  
  }


  /// <summary>
  /// 
  /// </summary>
  /// <param name="node"></param>
  void DoPossiblyResortChildrenUsingV(GNode node)
  {
    // Repeatedly make sure the ith child is the best one relative to those above it.
    for (int i = 0; i < node.NumEdgesExpanded - 1; i++)
    {
      // If the ith child is not the best, swap it with the best.
      DoRearrangeStartingAt(node, i);

      // Recursively rearrange.
      DoPossiblyResortChildrenUsingV(new GNode(node.Graph, node.ChildEdgeAtIndex(i).ChildNodeIndex));
    }

    //Coordinator.Graph.Validate(true);
  }


  void DoRearrangeStartingAt(GNode node, int firstChildIndex)
  {
    // Determine the child having the minimal win-loss value
    // (worst for opponent).
    double minQ = double.MaxValue;
    int minQIndex = firstChildIndex;
    for (int i = firstChildIndex; i < node.NumEdgesExpanded; i++)
    {
      float childWL = node.ChildEdgeAtIndex(i).ChildNode.NodeRef.V;
      if (childWL < minQ - THRESHOLD_Q_REARRANGE)
      {
        minQ = childWL;
        minQIndex = i;
      }
    }

    // Swap positions of the best child with the child at first index.
    if (minQIndex != firstChildIndex)
    {
      NumRearranged++;
      Interlocked.Increment(ref TotalNumRearranged);
      //Console.WriteLine($"Swap {firstChildIndex} and {minQIndex}");

      (node.EdgeHeadersSpan[firstChildIndex], node.EdgeHeadersSpan[minQIndex])
      = (node.EdgeHeadersSpan[minQIndex], node.EdgeHeadersSpan[firstChildIndex]);

      if (node.NumEdgesExpanded > 0)
      {
        // Careful, need to maintain contiguity
        throw new NotImplementedException();
      }
//        (node.EdgeStructsSpan[firstChildIndex], node.EdgeStructsSpan[minQIndex])
//        = (pnodeos.EdgeStructsSpan[minQIndex], node.EdgeStructsSpan[firstChildIndex]);
    }

  }


  public void DoPrefetchLevel3(bool debugMode)
  {

  }

  

 
  public void DoPrefetch(ParamsPrefetch prefetchParams, int maxBatchSize, bool debugMode = false)
  {
    static int GetAtIndexOrMaxValue(int[] values, int index) =>  (values == null || index >= values.Length) ? int.MaxValue : values[index];
    static float GetAtIndexOrValue(float[] values, int index, int fillInValue) => (values == null || index >= values.Length) ? fillInValue : values[index];

    using (new TimingBlock(TimingStats, TimingBlock.LoggingType.None))
    {
      for (int depth = 0; depth < prefetchParams.NumDepthLevels; depth++)
      {
        int maxNodesThisDepth = depth == 0 ? 1 : GetAtIndexOrMaxValue(prefetchParams.MaxNodesPerDepth, depth);

        const int MAX_PASSES = 5; // Why need more than 1??
        int passNum = 0;
        int numPrefetchedThisDepth = 0;
        bool lastYieldedNone = false;
        while (passNum < MAX_PASSES && !lastYieldedNone)
        {
          float minAbsProb = GetAtIndexOrValue(prefetchParams.MinAbsolutePolicyPctPerDepth, depth, 0);
          float minRelProb = GetAtIndexOrValue(prefetchParams.MaxProbabilityPctGapFromBestPerDepth, depth, 100);

          int numPrefetchTargetVisitsThisDepth = Math.Min(maxBatchSize, maxNodesThisDepth);
          int maxWidth = prefetchParams.MaxWidth;

          int numPrefetched = Manager.Engine.ProcessBatchPrefetch(numPrefetchTargetVisitsThisDepth,
                                           maxWidth, depth,
                                          (pos, childIndex) => MCGSStrategyPrefetch.AcceptMinProbabilityPct(pos, childIndex, minAbsProb, minRelProb),
                                           false/*debugMode*/);

          NumPrefetchPasses++;
          numPrefetchedThisDepth += numPrefetched;

          if (numPrefetched == 0)
          {
            lastYieldedNone = true;
          }
          else
          {
            NumNodesPrefetched += numPrefetched;
            Interlocked.Add(ref TotalNumPrefetched, numPrefetched);
          }

          if (debugMode)
          {
            // graph.DumpNodesStructure();
            Console.WriteLine($"Preload level {depth} try {passNum}, processed {numPrefetched} nodes, tree size now {Manager.Engine.Graph.Store.NodesStore.NumUsedNodes} width: {maxWidth} depth:{depth}");
            Manager.Engine.Graph.Validate();
          }

          if (numPrefetchedThisDepth >= maxNodesThisDepth)
          {
            break;
          }

          passNum++;
        }

      }
    }
  }
}
