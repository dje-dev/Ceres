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
using System.IO;
using Ceres.Base.Benchmarking;
using Ceres.Chess.MoveGen;
using Ceres.MCTS.Environment;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.MTCSNodes.Storage;
using Ceres.MCTS.MTCSNodes.Struct;

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
      /// <summary>
      /// Tree will not be rebuilt, instead an instamove from existing tree will be made.
      /// </summary>
      ForceInstamove,

      /// <summary>
      /// Tree will not change (search root unchanged), search continues.
      /// </summary>
      UnchangedStore,

      /// <summary>
      /// New empty node store created.
      /// </summary>
      NewStore,

      /// <summary>
      /// Tree is fully rebuilt, putting new root at beginning and clearing unreachable nodes.
      /// </summary>
      KeepStoreRebuildTree,

      /// <summary>
      /// All tree nodes retained, but new root is swapped into first position.
      /// </summary>
      KeepStoreSwapRoot
    }


    static int[] counts = new int[5];
    static int maxAllocated = 0;

    const bool VERBOSE = false;


    /// <summary>
    /// Returns the optimal strategy given specified parameters.
    /// </summary>
    /// <param name="currentRootNode"></param>
    /// <param name="newRootNode"></param>
    /// <param name="maxStoreNodes"></param>
    /// <returns></returns>
    public static ReuseDecision ChooseMethod(MCTSNode currentRootNode, MCTSNode newRootNode, int? maxStoreNodes)
    {
      ReuseDecision decision = DoChooseMethod(currentRootNode, newRootNode, maxStoreNodes, VERBOSE);

      maxAllocated = Math.Max(maxAllocated, currentRootNode.Store.Nodes.NumTotalNodes);
      counts[(int)decision.ChosenMethod]++;

      if (VERBOSE)
      {
        Console.WriteLine($"{decision}");

        Console.WriteLine("  Max                  " + maxAllocated / 1000 + "k");
        Console.WriteLine("  ForceInstamove       " + counts[0]);
        Console.WriteLine("  UnchangedStore       " + counts[1]);
        Console.WriteLine("  NewStore             " + counts[2]);
        Console.WriteLine("  KeepStoreRebuildTree " + counts[3]);
        Console.WriteLine("  KeepStoreSwapRoot    " + counts[4]);
        Console.WriteLine();
      }

      return decision;
    }

    static ReuseDecision DoChooseMethod(MCTSNode currentRootNode, MCTSNode newRootNode, int? maxStoreNodes, bool dump)
    {
      if (!maxStoreNodes.HasValue)
      {
        throw new Exception("Implementation restriction: MaxTreeSize must be set");
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


      (float fracReachable, float fractionSecondary) = FractionReachableAndSecondaryEstimate(currentRootNode.Context, in newRootNode.StructRef);
//      Console.WriteLine(newRootNode.N + " " +  fracReachable + " " + fractionSecondary);
      ReuseDecision Decision(Method method)
      {
        ReuseDecision dd = new ReuseDecision(method, maxStoreNodes.Value, curStoreAllocated, curStoreWasted,
                                             currentRootNode.N, newRootNode.N,
                                             currentRootNode.Context.Tree.TranspositionRoots == null ?0 : currentRootNode.Context.Tree.TranspositionRoots.Count,
                                             fracReachable, fractionSecondary);
        return dd;
      }

      if (currentRootNode == newRootNode)
      {
        return Decision(Method.UnchangedStore);
      }

      if (newRootNode.IsNull || newRootNode.N <= 1)
      {
        return Decision(Method.NewStore);
      }

      if (newRootNode.Context.ParamsSearch.TestFlag && fractionSecondary < 0.05f)
      {
        MCTSEventSource.TestCounter1++;
        //Console.WriteLine(newRootNode.N + " new store forced " + fractionSecondary);
        return Decision(Method.NewStore);
      }

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
        return Decision(Method.ForceInstamove);
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
        // TODO: use MCTSNodeStorePositionExtractorToCache to fill the position cache,
        //       extracting useful nodes
        return Decision(Method.NewStore);
      }

      if (!currentRootNode.Context.ParamsSearch.TreeReuseSwapRootEnabled)
      {
        return Decision(Method.KeepStoreRebuildTree);
      }

      // Be less inclined to rebulid if little memory is being used so far.
      float thresholdWasteRebuild = 0.50f;
      if (fracAvailableOfAllocatable > 0.80f)
      {
        thresholdWasteRebuild = 0.80f;
      }
      else if (fracAvailableOfAllocatable > 0.70f)
      {
        thresholdWasteRebuild = 0.70f;
      }
      

      //Console.WriteLine(priorNodes + " --> " + newNodes + "  " + testMode + " " + fracAvailableOfAllocatable + " " + fracWastedOfAllocated);
      bool rebuildDueToWaste = fracWastedOfAllocated > thresholdWasteRebuild;

      if (rebuildDueToWaste)
      {
        // Rebuild since tree is dominated by waste,
        // which bloats memory usage and
        // also creates more fixed scanning overhead upon each rebuild/swap-root operation.
        return Decision(Method.KeepStoreRebuildTree);
      }

      if (fracAvailableOfAllocatable < 0.15f)
      {
        // Rebuild since new search tree would have little additional room to grow.
        return Decision(Method.KeepStoreRebuildTree);
      }

      // Otherwise preferred method of swapping root can be used, 
      // which is almost instantaneous and keeps a large cache of nodes in store.
      return Decision(Method.KeepStoreSwapRoot);

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
      bool shouldInstamove = estFracNewTreeOfAllocatable > MIN_FRAC_NEW_TREE_OF_ALLOCATABLE;
      return shouldInstamove;
    }

    static (float, float) FractionReachableAndSecondaryEstimate(MCTSIterator context, in MCTSNodeStruct newRootNodeRef)
    {
      int reachable = 0;
      int secondary = 0;
      int totalStoreNodes = context.Tree.Store.Nodes.NumTotalNodes;

      // Only an approximation is needed, so just sample randomly.
      const int NUM_SAMPLES_TARGET = 300;
      int NUM_SAMPLES = (int)MathF.Min(NUM_SAMPLES_TARGET, totalStoreNodes / 2);
      Random rand = new Random();

      Base.OperatingSystem.MemoryBufferOS<MCTSNodeStruct> nodes = context.Tree.Store.Nodes.nodes;

      using (new SearchContextExecutionBlock(context))
      {
        for (int i = 0; i < NUM_SAMPLES; i++)
        {
          MCTSNodeStructIndex index = new MCTSNodeStructIndex(1 + rand.Next(totalStoreNodes - 2));
          ref readonly MCTSNodeStruct scanNodeRef = ref nodes[index.Index];

          if (scanNodeRef.IsPossiblyReachableFrom(newRootNodeRef))
          {
            reachable++;
          }

          if (scanNodeRef.SecondaryNN)
          {
            secondary++;
          }
#if NOT
            // Disabled: trying to generate the position corresopnding to this node
            // and determine reachability from that is too expensive and pollutes the node cache.

          MCTSNode rootNode = context.Tree.GetNode(context.Tree.Store.RootNode.Index);
          MGPosition newRootPos = rootNode.Annotation.PosMG;

            MCTSNode nodeNewRoot = context.Tree.GetNode(newRootNodeRef.Index);
            nodeNewRoot.Annotate();

            var node = context.Tree.GetNode(index);
            MGPosition scanPos = node.Annotation.PosMG;
            if (MGPositionReachability.IsProbablyReachable(newRootPos, scanPos))
            {
              if (!scanNodeRef.IsPossiblyReachableFrom(newRootNodeRef))
              {
                Console.WriteLine(nodeNewRoot.Annotation.Pos.FEN);
                Console.WriteLine(node.Annotation.Pos.FEN);
                throw new NotImplementedException();
              }
              else
                Console.WriteLine("ok32");
              reachable++;
            }
#endif
        }
      }

      return ((float)reachable / NUM_SAMPLES, 
              (float)secondary / NUM_SAMPLES);
    }

#region ReuseDecision record

    public record ReuseDecision
    {
      public readonly Method ChosenMethod;
      public readonly int StoreMaxNodes;
      public readonly int StoreCurNodes;
      public readonly int TranspositionRootsCount;
      public readonly int StoreOldGenerationNodes;
      public float StoreFracFull => (float)StoreCurNodes / StoreMaxNodes;
      public float StoreFracOldGeneration => (float)StoreOldGenerationNodes / StoreCurNodes;

      public readonly int CurrentRootN;
      public readonly int NewRootN;
      public readonly float EstFracReachable;
      public readonly float EstFracSecondary;

      public ReuseDecision(Method chosenMethod, int storeMaxNodes, int storeCurNodes, int storeOldGenerationNodes,
                           int currentRootN, int newRootN, int transpositionRootsCount, 
                           float estFracReachable, float estFracSecondary)
      {
        ChosenMethod = chosenMethod;
        StoreMaxNodes = storeMaxNodes;
        StoreCurNodes = storeCurNodes;
        StoreOldGenerationNodes = storeOldGenerationNodes;
        CurrentRootN = currentRootN;
        NewRootN = newRootN;
        TranspositionRootsCount = transpositionRootsCount;
        EstFracReachable = estFracReachable;
        EstFracSecondary = estFracSecondary;
      }

      public void Dump(TextWriter writer)
      {
        string rebuildTreeStr = (ChosenMethod == Method.KeepStoreRebuildTree || ChosenMethod == Method.NewStore) ? "***" : "";
        writer.WriteLine($"  ChosenMethod            {ChosenMethod} {rebuildTreeStr}");
        writer.WriteLine($"  CurrentRootN            {CurrentRootN,12:N0}");
        writer.WriteLine($"  NewRootN                {NewRootN,12:N0}");

        writer.WriteLine($"  StoreCurNodes           {StoreCurNodes,12:N0}  ({100 * StoreFracFull,6:F0}%)");
        writer.WriteLine($"  StoreMaxNodes           {StoreMaxNodes,12:N0}");
        writer.WriteLine($"  TranspositionRootsCount {TranspositionRootsCount,12:N0}");
        writer.WriteLine($"  StoreOldGenerationNodes {StoreOldGenerationNodes,12:N0}  ({100 * StoreFracOldGeneration,6:F0}%)");
        writer.WriteLine($"  Reachable Frac (est.)   {100.0 * EstFracReachable,6:F0}%");
        writer.WriteLine($"  Secondary Frac (est.)   {100.0 * EstFracSecondary,6:F0}%");
      }

    }

#endregion

  }

}
