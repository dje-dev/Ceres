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
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using Ceres.Base.Math;
using Ceres.Base.Misc;
using Ceres.MCGS.Graphs.GEdgeHeaders;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.Paths;
using Ceres.MCGS.Search.Strategies;
using Ceres.MCGS.Utils;

#endregion

namespace Ceres.MCGS.Search.Phases.Backup;

public partial class MCGSBackup
{
  /// <summary>
  /// Parent MCGS engine instance.
  /// </summary>
  public readonly MCGSEngine Engine;

  /// <summary>
  /// Cached array of tasks for parallel backup to avoid allocations.
  /// </summary>
  private Task[] cachedBackupTasks;


  const int MAX_BACKUP_THREADS = 24;

  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="mcgsCoordinator"></param>
  public MCGSBackup(MCGSEngine mcgsCoordinator)
  {
    Engine = mcgsCoordinator;

    if (mcgsCoordinator.Manager.ParamsSearch.Execution.BackupMode == BackupMethodEnum.ReductionMultiThread)
    {
      // Allocate cached task array for multi-threaded backup.
      cachedBackupTasks = new Task[MAX_BACKUP_THREADS];
    } 
  }


  /// <summary>
  /// Runs backup of all paths in paths using single-threaded reduction strategy.
  /// </summary>
  /// <param name="paths"></param>
  /// <param name="strategy"></param>
  /// <param name="iterator"></param>
  internal void BackupAllSingleReduction(ConcurrentQueue<MCGSPath> paths,
                                         MCGSSelectBackupStrategyBase strategy,
                                         MCGSIterator iterator)
  {
    // TODO: is it possible that the processing in PreparePathsForReduction
    //       could be move into the loop in this, saving one loop over all paths?
    //       However, this may not be possible, the backup of some paths may depend
    //       on others having had ther leaves initialized.
    PreparePathsForBackup(paths);

    foreach (MCGSPath path in paths)
    {
      BackupReduced(strategy, path, iterator.IteratorID);
    }
  }


  /// <summary>
  /// Runs backup of all paths in paths using multi-threaded reduction strategy.
  /// </summary>
  /// <param name="paths"></param>
  /// <param name="strategy"></param>
  /// <param name="iterator"></param>
  internal void BackupAllParallelReduction(ConcurrentQueue<MCGSPath> paths,
                                           MCGSSelectBackupStrategyBase strategy,
                                           MCGSIterator iterator)
  {
    PreparePathsForBackup(paths);

#if DEBUG
    MCGSPath[] copyPaths = paths.ToArray();
#endif

    int numBackupThreads = MaxConcurrentThreadsForPaths(iterator.numAllocatedPaths);

    for (int i = 0; i < numBackupThreads; i++)
    {
      cachedBackupTasks[i] = Task.Run(() => BackupWorkerLoop(paths, strategy, iterator));
    }

    Task.WaitAll(cachedBackupTasks.AsSpan(0, numBackupThreads));

#if DEBUG
    foreach (MCGSPath path in copyPaths)
    {
      foreach (MCGSPathVisitMember visit in path.PathVisitsLeafToRoot)
      {
        if (!visit.PathVisitRef.ParentChildEdge.ParentNode.IsSearchRoot) // accumulation not done at root
        {
          Debug.Assert(visit.PathVisitRef.NumVisitsAttemptedPendingBackup == 0);
          Debug.Assert(visit.PathVisitRef.NumVisitsAttempted > 0);

        }
      }
    }
#endif

#if NOT
      foreach (MCGSPath path in paths)
      {
        path.ValidateAllPendingReleased();
      }
#endif
  }

  private int MaxConcurrentThreadsForPaths(int numPaths)
  {
    // Bigger trees have more depth, more memory access latency, need more threads.
    int PATHS_PER_THREAD = Engine.SearchRootNode.N > 1_000_000 ? 30 : 50;
    return (int)MathHelpers.Bounded(numPaths / PATHS_PER_THREAD, 1, MAX_BACKUP_THREADS);
  }


  /// <summary>
  /// Worker loop for multi-threaded reduction backup.
  /// </summary>
  /// <param name="workerID"></param>
  /// <param name="pendingBackupPaths"></param>
  /// <param name="strategy"></param>
  /// <param name="iterator"></param>
  private void BackupWorkerLoop(ConcurrentQueue<MCGSPath> pendingBackupPaths,
                                MCGSSelectBackupStrategyBase strategy,
                                MCGSIterator iterator)
  {
    while (pendingBackupPaths.TryDequeue(out MCGSPath path))
    {
      BackupReduced(strategy, path, iterator.IteratorID);
    }
  }


  /// <summary>
  /// Issues processor prefetch memory hint for child edge of parentNode at indexInParent.
  /// </summary>
  /// <param name="parentNode"></param>
  /// <param name="indexInParent"></param>
  private static void PrefetchChild(GNode parentNode, int indexInParent)
  {
    // Possibly start memory prefetch of parent edge.
    if (MCGSParamsFixed.PrefetchCacheLevel != Prefetcher.CacheLevel.None)
    {
      Span<GEdgeHeaderStruct> parentsChildEdgeHeaders = parentNode.EdgeHeadersSpan;
      unsafe
      {
        void* nodePtr = Unsafe.AsPointer(ref parentNode.EdgeStructAtIndexRef(parentsChildEdgeHeaders[indexInParent].EdgeStoreBlockIndex, indexInParent));
        Prefetcher.PrefetchLevel1(nodePtr);

        Debug.Assert(parentNode.ChildEdgeAtIndex(indexInParent).edgeStructPtr == nodePtr);
      }
    }
  }

  static bool haveWarnedBackup = false;


  /// <summary>
  /// Determines the best/most appropriate backup mode to use
  /// given current search state.
  /// </summary>
  /// <returns></returns>
  public BackupMethodEnum BackupModeToUse()
  {
    switch (Engine.Manager.ParamsSearch.Execution.BackupMode)
    {
      case BackupMethodEnum.ReductionSingleThread:
        return BackupMethodEnum.ReductionSingleThread;

      case BackupMethodEnum.ReductionMultiThread:
        // Determine actual backup mode to use.
        // Possibly downgrade for small trees, where the overhead makes it not worthwhile.
        // TODO: TUNE THIS! should perhaps be higher
        const int THRESHOLD_NO_PARALLEL = 20_000; 
        bool parallelOK = Engine.SearchRootNode.N > THRESHOLD_NO_PARALLEL;
        return parallelOK ? BackupMethodEnum.ReductionMultiThread 
                          : BackupMethodEnum.ReductionSingleThread; 

      default:
        throw new ArgumentOutOfRangeException("Unknown backup mode");
         
    } 
  }

}
