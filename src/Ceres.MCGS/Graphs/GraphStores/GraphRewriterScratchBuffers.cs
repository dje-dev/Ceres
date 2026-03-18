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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Ceres.Chess.MoveGen;

#endregion

namespace Ceres.MCGS.Graphs.GraphStores;

/// <summary>
/// Manages reusable native memory buffers for GraphRewriter, eliminating
/// managed array allocations that cause LOH fragmentation.
/// Buffers grow on demand and are never shrunk. Each Graph holds one instance.
/// </summary>
public sealed unsafe class GraphRewriterScratchBuffers : IDisposable
{
  // --- Capacities ---
  int nodeCapacity;
  int retainedCapacity;

  // --- Buffer pointers (node-sized) ---
  int* oldToNewPtr;

  // --- Buffer pointers (retained-sized) ---
  int* generalAPtr;
  int* visitedPtr;
  int* frontierAPtr;
  int* frontierBPtr;
  MGPosition* positionsPtr;
  PosHash96MultisetRunning* runningHashesPtr;

  // --- Atomic counter for BFS next-frontier (replaces int[1]) ---
  int nextCount;

  bool disposed;

  /// <summary>
  /// Ensures all buffers have sufficient capacity for the given sizes.
  /// Grows buffers if needed. Zeroes buffers that require zero-initialization.
  /// </summary>
  public void EnsureCapacity(int numNodes, int numTotalRetained)
  {
    if (numNodes > nodeCapacity)
    {
      GrowNodeBuffers(numNodes);
    }

    if (numTotalRetained > retainedCapacity)
    {
      GrowRetainedBuffers(numTotalRetained);
    }

    // Zero buffers that need zero-init:
    // OldToNew: 0 = pruned/unmapped
    NativeMemory.Clear(oldToNewPtr, (nuint)numNodes * sizeof(int));
    // GeneralA (used as incomingN): accumulated via +=, needs 0 baseline
    NativeMemory.Clear(generalAPtr, (nuint)numTotalRetained * sizeof(int));
    // Visited: 0 = unvisited
    NativeMemory.Clear(visitedPtr, (nuint)numTotalRetained * sizeof(int));
    // RunningHashes: default(PosHash96MultisetRunning) = {0,0} = empty hash state
    NativeMemory.Clear(runningHashesPtr, (nuint)numTotalRetained * (nuint)sizeof(PosHash96MultisetRunning));

    // FrontierA, FrontierB, Positions: written before read, no zeroing needed
  }

  /// <summary>
  /// Clears the Visited buffer for reuse between phases.
  /// </summary>
  public void ClearVisited(int numTotalRetained)
  {
    Debug.Assert(numTotalRetained <= retainedCapacity);
    NativeMemory.Clear(visitedPtr, (nuint)numTotalRetained * sizeof(int));
  }

  // --- Span accessors ---

  public Span<int> OldToNew(int length)
  {
    Debug.Assert(length <= nodeCapacity);
    return new Span<int>(oldToNewPtr, length);
  }

  public Span<int> GeneralA(int length)
  {
    Debug.Assert(length <= retainedCapacity);
    return new Span<int>(generalAPtr, length);
  }

  public Span<int> Visited(int length)
  {
    Debug.Assert(length <= retainedCapacity);
    return new Span<int>(visitedPtr, length);
  }

  public Span<int> FrontierA(int length)
  {
    Debug.Assert(length <= retainedCapacity);
    return new Span<int>(frontierAPtr, length);
  }

  public Span<int> FrontierB(int length)
  {
    Debug.Assert(length <= retainedCapacity);
    return new Span<int>(frontierBPtr, length);
  }

  public Span<MGPosition> Positions(int length)
  {
    Debug.Assert(length <= retainedCapacity);
    return new Span<MGPosition>(positionsPtr, length);
  }

  public Span<PosHash96MultisetRunning> RunningHashes(int length)
  {
    Debug.Assert(length <= retainedCapacity);
    return new Span<PosHash96MultisetRunning>(runningHashesPtr, length);
  }

  // --- Raw pointer accessors (for Parallel.For lambdas that can't capture Spans) ---

  public int* OldToNewPtr => oldToNewPtr;
  public int* GeneralAPtr => generalAPtr;
  public int* VisitedPtr => visitedPtr;
  public int* FrontierAPtr => frontierAPtr;
  public int* FrontierBPtr => frontierBPtr;
  public MGPosition* PositionsPtr => positionsPtr;
  public PosHash96MultisetRunning* RunningHashesPtr => runningHashesPtr;

  /// <summary>
  /// Reference to the atomic next-count field (replaces int[1] boxing pattern).
  /// </summary>
  public ref int NextCount => ref nextCount;

  // --- Growth helpers ---

  void GrowNodeBuffers(int requiredCapacity)
  {
    if (oldToNewPtr != null)
    {
      NativeMemory.Free(oldToNewPtr);
    }

    nodeCapacity = requiredCapacity;
    oldToNewPtr = (int*)NativeMemory.Alloc((nuint)requiredCapacity, (nuint)sizeof(int));
  }

  void GrowRetainedBuffers(int requiredCapacity)
  {
    if (generalAPtr != null)
    {
      NativeMemory.Free(generalAPtr);
      NativeMemory.Free(visitedPtr);
      NativeMemory.Free(frontierAPtr);
      NativeMemory.Free(frontierBPtr);
      NativeMemory.Free(positionsPtr);
      NativeMemory.Free(runningHashesPtr);
    }

    retainedCapacity = requiredCapacity;
    generalAPtr = (int*)NativeMemory.Alloc((nuint)requiredCapacity, (nuint)sizeof(int));
    visitedPtr = (int*)NativeMemory.Alloc((nuint)requiredCapacity, (nuint)sizeof(int));
    frontierAPtr = (int*)NativeMemory.Alloc((nuint)requiredCapacity, (nuint)sizeof(int));
    frontierBPtr = (int*)NativeMemory.Alloc((nuint)requiredCapacity, (nuint)sizeof(int));
    positionsPtr = (MGPosition*)NativeMemory.Alloc((nuint)requiredCapacity, (nuint)sizeof(MGPosition));
    runningHashesPtr = (PosHash96MultisetRunning*)NativeMemory.Alloc((nuint)requiredCapacity, (nuint)sizeof(PosHash96MultisetRunning));
  }

  // --- IDisposable ---

  public void Dispose()
  {
    if (disposed)
    {
      return;
    }

    disposed = true;

    if (oldToNewPtr != null) { NativeMemory.Free(oldToNewPtr); oldToNewPtr = null; }
    if (generalAPtr != null) { NativeMemory.Free(generalAPtr); generalAPtr = null; }
    if (visitedPtr != null) { NativeMemory.Free(visitedPtr); visitedPtr = null; }
    if (frontierAPtr != null) { NativeMemory.Free(frontierAPtr); frontierAPtr = null; }
    if (frontierBPtr != null) { NativeMemory.Free(frontierBPtr); frontierBPtr = null; }
    if (positionsPtr != null) { NativeMemory.Free(positionsPtr); positionsPtr = null; }
    if (runningHashesPtr != null) { NativeMemory.Free(runningHashesPtr); runningHashesPtr = null; }

    nodeCapacity = 0;
    retainedCapacity = 0;
  }

  ~GraphRewriterScratchBuffers() => Dispose();
}
