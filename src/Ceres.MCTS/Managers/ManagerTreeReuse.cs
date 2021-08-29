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
using Ceres.MCTS.MTCSNodes;

#endregion

namespace Ceres.MCTS.Managers
{
  /// <summary>
  /// Manages logic relating to tree reuse, specifically 
  /// computes the optimal reuse strategy (new tree, rebuild, swap root)
  /// at beginning of a search.
  /// </summary>
  public static class ManagerTreeReuse
  {
    public enum Method
    {
      ForceInstamove,
      NewStore,
      KeepStoreRebuildTree,
      KeepStoreSwapRoot
    }

    static int[] counts = new int[4];
    static int maxAllocated = 0;

    const bool VERBOSE = false;


    /// <summary>
    /// Returns the optimal strategy given specified parameters.
    /// </summary>
    /// <param name="currentRootNode"></param>
    /// <param name="newRootNode"></param>
    /// <param name="maxStoreNodes"></param>
    /// <returns></returns>
    public static Method ChooseMethod(MCTSNode currentRootNode, MCTSNode newRootNode, int? maxStoreNodes)
    {
      Method decision = DoChooseMethod(currentRootNode, newRootNode, maxStoreNodes, VERBOSE);

      maxAllocated = Math.Max(maxAllocated, currentRootNode.Context.Tree.Store.Nodes.NumTotalNodes);
      counts[(int)decision]++;

      if (VERBOSE)
      {
        Console.WriteLine($"{decision}");

        Console.WriteLine("  Max                  " + maxAllocated / 1000 + "k");
        Console.WriteLine("  ForceInstamove       " + counts[0]);
        Console.WriteLine("  NewStore             " + counts[1]);
        Console.WriteLine("  KeepStoreRebuildTree " + counts[2]);
        Console.WriteLine("  KeepStoreSwapRoot    " + counts[3]);
        Console.WriteLine();
      }

      return decision;
    }


    static Method DoChooseMethod(MCTSNode currentRootNode, MCTSNode newRootNode, int? maxStoreNodes, bool dump)
    {
      if (!maxStoreNodes.HasValue)
      {
        throw new Exception("Implementation restriction: MaxTreeSize must be set");
      }

      if (newRootNode == null)
      {
        return Method.NewStore;
      }

      int curStoreAllocated = currentRootNode.Tree.Store.Nodes.NumTotalNodes;
      int curStoreWasted = currentRootNode.Tree.Store.Nodes.NumOldGeneration;
      int curStoreUsed = curStoreAllocated - curStoreWasted;

      int availNodesOfAllocatable = maxStoreNodes.Value - curStoreAllocated;
      float fracAvailableOfAllocatable = (float)availNodesOfAllocatable / maxStoreNodes.Value;

      int priorNodes = currentRootNode.N;
      int newNodes = newRootNode.N;

      float newNodesFractionOfOld = (float)newNodes / priorNodes;

      float fracWastedOfAllocated = (float)curStoreWasted / curStoreAllocated;

      //      float fracAllocatedOfAllocatable = (float)curStoreAllocated / maxStoreNodes.Value;
      //      float maxWastedOfAllocated = 0.8f - fracAllocatedOfAllocatable;


      if (dump)
      {
        float allocatedMultipleOfCurrent = (float)curStoreAllocated / curStoreUsed;
        Console.Write($"\r\n[{allocatedMultipleOfCurrent,4:F2}x {curStoreAllocated / 1000}k/{curStoreUsed / 1000}k ALLOC/USED]."
                    + $" N: {currentRootNode.N / 1000}k --> {newRootNode.N / 1000}k, "
                    + $"max_store={maxStoreNodes.Value / 1000}k (frac_wasted/allocated={fracWastedOfAllocated,4:F2} "
                    + $"frac_available/allocatable={fracAvailableOfAllocatable,4:F4}), --> ");
      }


      // Force instamove if new tree already huge.
      if (ShouldForceInstamove(currentRootNode, newRootNode, maxStoreNodes))
      {
        return Method.ForceInstamove;
      }

      // If the fraction of nodes being retained is extremely small,
      // restart search with new store. This has two advantages:
      //   - the tree rebuild/swap overhead is completely avoided (some overhead is linear in node store size)
      //   - releasing the prior store frees up its possibly accumuated large allocation
      //     thereby reducing future memory requirements and making future rebuilds/swap faster.
      //
      // However don't do this if the new tree is small in absolute sense
      // because the rebuild time will be relatively high due to NN evaluator latency
      // and memory requirements are insignificant at this level.
      if (newNodes > 10_000 && newNodesFractionOfOld < 0.05f)
      {
        return Method.NewStore;
      }

      if (!currentRootNode.Context.ParamsSearch.TreeReuseSwapRootEnabled)
      {
        return Method.KeepStoreRebuildTree;
      }

      // If tree size is small and shrinking greatly, rebuild
      // since time required will be small and tree will be kept compact.
      if (newNodes < 10_000 && newNodesFractionOfOld < 0.40f)
      {
        return Method.KeepStoreRebuildTree;
      }

      if (fracWastedOfAllocated > 0.60f) // was 0.8
      {
        // Rebuild since tree is dominated by waste,
        // which bloats memory usage and
        // also creates more fixed scanning overhead upon each rebuild/swap-root operation.
        return Method.KeepStoreRebuildTree;
      }
      else if (fracAvailableOfAllocatable < 0.25f)
      {
        // Rebuild since new search tree would have little additional room to grow.
        return Method.KeepStoreRebuildTree;
      }

      // Otherwise preferred method of swapping root can be used, 
      // which is at least 5x faster per used node than rebuild.
      return Method.KeepStoreSwapRoot;

      // TODO: shrink committed in VirtualFree
    }


    /// <summary>
    /// If new search tree is already very near maximum allowable size of a search tree,
    /// then instamove because little additional search progress is possible.
    /// </summary>
    static bool ShouldForceInstamove(MCTSNode currentRootNode, MCTSNode newRootNode, int? maxStoreNodes)
    {
      if (maxStoreNodes.HasValue)
      {
        return false;

      }

      float fracNewTreeOfMaxNodes = (float)newRootNode.N / maxStoreNodes.Value;

      const float THRESHOLD_TREE_FULL_FRACTION_INSTAMOVE = 0.90f;
      return fracNewTreeOfMaxNodes > THRESHOLD_TREE_FULL_FRACTION_INSTAMOVE;
    }
  }
}
