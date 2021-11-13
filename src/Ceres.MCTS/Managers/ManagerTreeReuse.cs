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

      maxAllocated = Math.Max(maxAllocated, currentRootNode.Store.Nodes.NumTotalNodes);
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

      if (newRootNode.IsNull || newRootNode.N <= 1)
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

      bool testMode = currentRootNode.Context.ParamsSearch.TestFlag;

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

      float thresholdWasteRebuild = 0.65f;
      if (testMode)
      {
        if (fracAvailableOfAllocatable > 0.80f)
          thresholdWasteRebuild = 0.90f;
        else if (fracAvailableOfAllocatable > 0.70f)
          thresholdWasteRebuild = 0.80f;
      }

      //Console.WriteLine(priorNodes + " --> " + newNodes + "  " + testMode + " " + fracAvailableOfAllocatable + " " + fracWastedOfAllocated);
      bool rebuildDueToWaste = fracWastedOfAllocated > thresholdWasteRebuild;

      if (rebuildDueToWaste)
      {
        // Rebuild since tree is dominated by waste,
        // which bloats memory usage and
        // also creates more fixed scanning overhead upon each rebuild/swap-root operation.
        return Method.KeepStoreRebuildTree;
      }

      if (fracAvailableOfAllocatable < 0.15f)
      {
        // Rebuild since new search tree would have little additional room to grow.
        return Method.KeepStoreRebuildTree;
      }

      // Otherwise preferred method of swapping root can be used, 
      // which is almost instantaneous and keeps a large cache of nodes in store.
      return Method.KeepStoreSwapRoot;

      // TODO: shrink committed in VirtualFree
    }


    /// <summary>
    /// Instamove if available storage is almost all full with because little additional search progress is possible.
    /// </summary>
    static bool ShouldForceInstamove(MCTSNode currentRootNode, MCTSNode newRootNode, int? maxStoreNodes)
    {
      if (maxStoreNodes == null)
      {
        return false;
      }

      int numNodesThisGeneration = currentRootNode.Tree.Store.Nodes.NumTotalNodes - currentRootNode.Tree.Store.Nodes.NumOldGeneration;
      float fracUsedOfAllocatable = (float)numNodesThisGeneration / maxStoreNodes.Value;
      float treeShrinkagesFactor = ((float)newRootNode.N / currentRootNode.N);
      float estFracNewTreeOfAllocatable = fracUsedOfAllocatable * treeShrinkagesFactor;


      // If new tree is estimated to use almost all of allocatable space 
      // then prefer to instamove since not much additional search progress is possible,
      // thereby avoiding all the overhead that would be required to do the swap/rebuild
      // before continuing search.
      const float MIN_FRAC_NEW_TREE_OF_ALLOCATABLE = 0.80f;
      return estFracNewTreeOfAllocatable > MIN_FRAC_NEW_TREE_OF_ALLOCATABLE;
    }
  }
}
