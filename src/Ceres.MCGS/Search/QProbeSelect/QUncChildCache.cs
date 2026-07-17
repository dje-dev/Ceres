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

using System.Numerics;
using System.Threading;

#endregion

namespace Ceres.MCGS.Search.QProbeSelect;

/// <summary>
/// Direct-mapped, allocation-free per-CHILD-NODE cache of Q-uncertainty forecasts
/// (mu, sigma) at the short and (optionally) long query horizons. The forecast
/// conditions on the child's visit count N0, so an entry is fresh only while the
/// child's N stays in the same power-of-two band AND has grown less than 25% since
/// evaluation; parent-context features (sibling gap, parent N) drift within that
/// window - accepted staleness per the Phase 2 design.
///
/// Concurrency: per-slot seqlock (version counter, odd while a writer is active).
/// A collision or torn read is treated as a miss; correctness never depends on a
/// hit. Entries are invalidated implicitly by the generation stamp (new search).
/// Total footprint ~1.3 MB.
/// </summary>
internal sealed class QUncChildCache
{
  const int LOG2_SLOTS = 15;
  const int NUM_SLOTS = 1 << LOG2_SLOTS;

  const byte FLAG_HAS_LONG = 1;

  readonly int[] version = new int[NUM_SLOTS];
  readonly int[] nodeIndex = new int[NUM_SLOTS];
  readonly int[] nAtEval = new int[NUM_SLOTS];
  readonly int[] generation = new int[NUM_SLOTS];
  readonly byte[] flags = new byte[NUM_SLOTS];
  readonly float[] muShort = new float[NUM_SLOTS];
  readonly float[] sigmaShort = new float[NUM_SLOTS];
  readonly float[] muLong = new float[NUM_SLOTS];
  readonly float[] sigmaLong = new float[NUM_SLOTS];


  static int SlotOfNodeIndex(int nodeIdx) => (int)((uint)(nodeIdx * -1640531527) >> (32 - LOG2_SLOTS));


  /// <summary>
  /// Attempts to retrieve a fresh forecast for a child node. A hit requires: same
  /// node index and generation, N in the same power-of-two band and grown less than
  /// 25% since evaluation, and (when needLong) the long horizon present.
  /// </summary>
  internal bool TryGet(int nodeIdx, int currentN, int gen, bool needLong,
                       out float muS, out float sigS, out float muL, out float sigL)
  {
    muS = 0;
    sigS = 0;
    muL = 0;
    sigL = 0;

    int slot = SlotOfNodeIndex(nodeIdx);

    int versionBefore = Volatile.Read(ref version[slot]);
    if ((versionBefore & 1) != 0)
    {
      return false;
    }

    int storedN = nAtEval[slot];
    if (nodeIndex[slot] != nodeIdx
     || generation[slot] != gen
     || (needLong && (flags[slot] & FLAG_HAS_LONG) == 0)
     || storedN <= 0
     || BitOperations.Log2((uint)currentN) != BitOperations.Log2((uint)storedN)
     || currentN * 4L >= storedN * 5L)  // growth >= 25% => stale
    {
      return false;
    }

    muS = muShort[slot];
    sigS = sigmaShort[slot];
    muL = muLong[slot];
    sigL = sigmaLong[slot];

    // Full fence so the data reads above cannot drift past the validating
    // version re-read (load-load reordering is real on ARM).
    Interlocked.MemoryBarrier();
    return Volatile.Read(ref version[slot]) == versionBefore;
  }


  /// <summary>
  /// Stores a forecast for a child node (overwrites any colliding entry).
  /// </summary>
  internal void Store(int nodeIdx, int childN, int gen, bool hasLong,
                      float muS, float sigS, float muL, float sigL)
  {
    int slot = SlotOfNodeIndex(nodeIdx);

    // Seqlock write: claim (odd), fill, release (even). A concurrent writer on a
    // colliding slot simply loses the race; readers see the version change and miss.
    int claimed = Interlocked.Increment(ref version[slot]);
    if ((claimed & 1) == 0)
    {
      // Another writer was mid-flight; back out (leave version even).
      Interlocked.Increment(ref version[slot]);
      return;
    }

    nodeIndex[slot] = nodeIdx;
    nAtEval[slot] = childN;
    generation[slot] = gen;
    flags[slot] = hasLong ? FLAG_HAS_LONG : (byte)0;
    muShort[slot] = muS;
    sigmaShort[slot] = sigS;
    muLong[slot] = muL;
    sigmaLong[slot] = sigL;

    Interlocked.Increment(ref version[slot]);
  }
}
