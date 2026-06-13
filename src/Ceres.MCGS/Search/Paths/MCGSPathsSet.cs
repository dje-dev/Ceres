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
using Ceres.Base.DataTypes;
using Ceres.MCGS.Search.Coordination;

#endregion

namespace Ceres.MCGS.Search.Paths;

/// <summary>
/// Contains a set of paths collected during MCGS iteration.
/// </summary>
public partial class MCGSPathsSet : IDisposable
{
  /// <summary>
  /// The MCGSIterator that owns this set of paths.
  /// </summary>
  public readonly MCGSIterator ParentIterator;

  /// <summary>
  /// The set of paths to be processed.
  /// </summary>
  public readonly ConcurrentQueue<MCGSPath> Paths;

  /// <summary>
  /// Total number of paths processed (including aborted).
  /// </summary>
  public long CountTotalPathsAttempted;

  /// <summary>
  /// Total number of non-aborted paths processed.
  /// </summary>
  public long CountNonAbortedPathVisits;

  /// <summary>
  /// Total number of path visits over all paths.
  /// </summary>
  public long SumNonAbortedPathVisits;

  /// <summary>
  /// Number of visits in the longest past encountered so far.
  /// </summary>
  public int MaxNonAbortedPathDepth;

  /// <summary>
  /// Distribution of path lengths (indexed by path length, value is count of paths with that length).
  /// </summary>
  public readonly long[] PathLengthDistribution = new long[256];

  /// <summary>
  /// Subset of PathsSet which require neural network evaluation.
  /// </summary>
  public readonly ListBounded<MCGSPath> NNPaths;

  /// <summary>
  /// Maximum number of NN evaluation paths allowed in this batch (int.MaxValue when unarmed).
  /// Armed only during the second "fill to evaluator capacity" selection pass so that the
  /// fill cannot (materially) overshoot the evaluator's padded batch capacity, which would
  /// spill into an additional engine launch. Checked via NNEvalBudgetExhausted in
  /// MCGSSelect.CapacityAbortNeeded (further descents are aborted once the budget is reached).
  /// </summary>
  internal int NNEvalSlotLimit = int.MaxValue;

  /// <summary>
  /// If the number of NN evaluation paths has reached the armed limit (see NNEvalSlotLimit).
  /// </summary>
  internal bool NNEvalBudgetExhausted => NNPaths.Count >= NNEvalSlotLimit;


  private bool disposedValue;


  /// <summary>
  /// Constructor
  /// </summary>
  /// <param name="parentIterator"></param>
  /// <param name="maxBatchSize"></param>
  public MCGSPathsSet(MCGSIterator parentIterator, int maxBatchSize)
  {
    ParentIterator = parentIterator;
    Paths = new ConcurrentQueue<MCGSPath>();
    NNPaths = new ListBounded<MCGSPath>(maxBatchSize);
 }


  /// <summary>
  /// Adds a path to the set.
  /// </summary>
  /// <param name="path"></param>
  /// <param name="numVisitsAcceptedLeafVisit"></param>
  public void AddPath(MCGSPath path, int numVisitsAcceptedLeafVisit)
  {
    Debug.Assert(path.TerminationReason != MCGSPathTerminationReason.NotYetTerminated);
    path.LeafVisitRef.NumVisitsAccepted = (short)numVisitsAcceptedLeafVisit;

    // Track total paths attempted (including aborted).
    CountTotalPathsAttempted++;

    // Update running statistics for non-aborted paths.
    if (path.TerminationReason != MCGSPathTerminationReason.Abort)
    {
      CountNonAbortedPathVisits++;

      int numVisits = path.NumVisitsInPath;
      SumNonAbortedPathVisits += numVisits;
      MaxNonAbortedPathDepth = Math.Max(MaxNonAbortedPathDepth, numVisits);

      // Update path length distribution.
      if (numVisits < PathLengthDistribution.Length)
      {
        PathLengthDistribution[numVisits]++;
      }

      if (path.TerminationReason == MCGSPathTerminationReason.PendingNeuralNetEval)
      {
        // Lock-free: NNPaths is preallocated to maxBatchSize and all readers run only
        // after the select-phase barrier (worker pool WaitAll), so slot reservation via
        // Interlocked suffices and the prior Monitor lock (a contention point under
        // parallel select) is unnecessary.
        NNPaths.AddConcurrent(path);
      }
    }

    Paths.Enqueue(path);
  }



  /// <summary>
  /// Resets the set to empty, ready for reuse.
  /// </summary>
  public void Reset()
  {
    Paths.Clear();
    NNPaths.Clear();
    NNEvalSlotLimit = int.MaxValue;
    Array.Clear(PathLengthDistribution, 0, PathLengthDistribution.Length);
  }


  /// <summary>
  /// Returns a string representation of the object.
  /// </summary>
  /// <returns></returns>
  public override string ToString() =>$"<MCGSPathSet of size {Paths.Count}>";  


  protected virtual void Dispose(bool disposing)
  {
    if (!disposedValue)
    {
      if (disposing)
      {
      }

      disposedValue = true;
    }
  }


  public void Dispose()
  {
    // Do not change this code. Put cleanup code in 'Dispose(bool disposing)' method
    Dispose(disposing: true);
    GC.SuppressFinalize(this);
  }
}
