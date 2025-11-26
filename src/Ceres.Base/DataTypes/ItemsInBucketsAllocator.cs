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
using System.Linq;
using System.Threading;

#endregion

namespace Ceres.Base.DataTypes;

/// <summary>
/// Manages allocation of items into a fixed number of buckets.
/// Buckets may be shared by multiple clients, but allocation
/// preferentially uses buckets that currently have zero active users.
/// </summary>
public class ItemsInBucketsAllocator
{
  /// <summary>
  /// Per-bucket usage count: how many clients are currently using this bucket.
  /// 0 means "not in use by any client".
  /// </summary>
  private readonly int[] bucketUseCounts;

  /// <summary>
  /// Lock to protect access to the bucketUseCounts array and inUseCount.
  /// </summary>
  private readonly object itemsLock;

  /// <summary>
  /// Total number of buckets.
  /// </summary>
  private readonly int numBuckets;

  /// <summary>
  /// Number of buckets that currently have usage &gt; 0.
  /// </summary>
  private int inUseCount;

  /// <summary>
  /// User-supplied per-bucket weights.
  /// </summary>
  private readonly float[] preferredFractions;

  /// <summary>
  /// Number of buckets managed by this allocator.
  /// </summary>
  public int BucketCount => numBuckets;


  /// <summary>
  /// Count of buckets that currently have at least one active client.
  /// </summary>
  public int AllocatedBucketCount => inUseCount;


  /// <summary>
  /// Gets the number of buckets that are currently unused (usage == 0).
  /// </summary>
  public int FreeBucketCount => BucketCount - inUseCount;


  /// <summary>
  /// Constructor.
  /// </summary>
  /// <param name="preferredFractions">Non-negative per-bucket weights.</param>
  public ItemsInBucketsAllocator(float[] preferredFractions)
  {
    if (preferredFractions == null)
    {
      throw new ArgumentNullException(nameof(preferredFractions));
    }

    if (preferredFractions.Length < 0)
    {
      throw new ArgumentOutOfRangeException(nameof(preferredFractions), "Array length must be non-negative.");
    }

    // Negative weights are not allowed.
    for (int i = 0; i < preferredFractions.Length; i++)
    {
      if (preferredFractions[i] < 0f)
      {
        throw new ArgumentOutOfRangeException(nameof(preferredFractions), "Fractions must be non-negative.");
      }
    }

    this.preferredFractions = (float[])preferredFractions.Clone();

    bucketUseCounts = new int[this.preferredFractions.Length];
    itemsLock = new object();
    numBuckets = this.preferredFractions.Length;
    inUseCount = 0;
  }


  /// <summary>
  /// Spreads totalItemCount across bucketsToUse distinct buckets.
  /// Buckets may already be in use by other clients; this method
  /// *prefers* buckets that currently have zero active clients,
  /// and only reuses in-use buckets if necessary.
  ///
  /// Returns an array of length numBuckets giving the number of
  /// items assigned to each bucket for this allocation.
  /// </summary>
  public int[] Allocate(int bucketsToUse, int totalItemCount)
  {
    if (bucketsToUse < 0)
    {
      throw new ArgumentOutOfRangeException(nameof(bucketsToUse), "bucketsToUse must be non-negative.");
    }

    if (totalItemCount < 0)
    {
      throw new ArgumentOutOfRangeException(nameof(totalItemCount), "totalItemCount must be non-negative.");
    }

    if (bucketsToUse == 0)
    {
      if (totalItemCount != 0)
      {
        throw new InvalidOperationException("Cannot allocate items to zero buckets.");
      }

      return new int[numBuckets];
    }

    if (bucketsToUse > numBuckets)
    {
      throw new InvalidOperationException("bucketsToUse exceeds total bucket capacity");
    }

    if (totalItemCount < bucketsToUse)
    {
      throw new InvalidOperationException("totalItemCount must be >= bucketsToUse");
    }

    lock (itemsLock)
    {
      // Build lists of unused (usage == 0) and in-use (usage > 0) buckets.
      List<int> unused = new(numBuckets);
      List<int> inUse = new(numBuckets);

      for (int i = 0; i < numBuckets; i++)
      {
        if (bucketUseCounts[i] == 0)
        {
          unused.Add(i);
        }
        else
        {
          inUse.Add(i);
        }
      }

      // Shuffle both lists so we randomize choice within each class.
      FisherYatesShuffleInPlace(unused);
      FisherYatesShuffleInPlace(inUse);

      // Select bucketsToUse distinct buckets:
      // 1) take as many as possible from unused (usage == 0),
      // 2) if needed, fill remainder from inUse (usage > 0).
      int[] selected = new int[bucketsToUse];
      int selectedCount = 0;

      int fromUnused = System.Math.Min(bucketsToUse, unused.Count);
      for (int i = 0; i < fromUnused; i++)
      {
        selected[selectedCount++] = unused[i];
      }

      int remainingToSelect = bucketsToUse - selectedCount;
      if (remainingToSelect > 0)
      {
        if (inUse.Count < remainingToSelect)
        {
          // Defensive check; should not occur because bucketsToUse <= numBuckets.
          throw new InvalidOperationException("Internal error: not enough buckets to allocate.");
        }

        for (int i = 0; i < remainingToSelect; i++)
        {
          selected[selectedCount++] = inUse[i];
        }
      }

      // Begin with the "at least one per bucket" guarantee.
      int[] result = new int[numBuckets];
      for (int i = 0; i < bucketsToUse; i++)
      {
        result[selected[i]] = 1;
      }

      int remaining = totalItemCount - bucketsToUse;
      if (remaining > 0)
      {
        // Gather weights for the selected buckets.
        double sumW = 0.0;
        double[] weights = new double[bucketsToUse];
        for (int i = 0; i < bucketsToUse; i++)
        {
          int idx = selected[i];
          double w = preferredFractions[idx];
          if (w < 0.0) { w = 0.0; } // safety (should not happen due to constructor validation)
          weights[i] = w;
          sumW += w;
        }

        if (sumW <= 0.0)
        {
          // All weights are zero; distribute remaining evenly with randomized tie-breaking.
          int baseAdd = remaining / bucketsToUse;
          int rem = remaining % bucketsToUse;

          // Randomize order for fairness when handing out the remainders.
          ShuffleArrayInPlace(selected);

          for (int i = 0; i < bucketsToUse; i++)
          {
            result[selected[i]] += baseAdd + (i < rem ? 1 : 0);
          }
        }
        else
        {
          // Largest Remainder (Hamilton) apportionment on the remaining items.
          int[] extra = new int[bucketsToUse];
          double[] rema = new double[bucketsToUse];

          int distributed = 0;
          for (int i = 0; i < bucketsToUse; i++)
          {
            double ideal = remaining * (weights[i] / sumW);
            int floor = (int)System.Math.Floor(ideal);
            extra[i] = floor;
            rema[i] = ideal - floor;
            distributed += floor;
          }

          int leftover = remaining - distributed;

          // Create an index list 0..bucketsToUse-1 and shuffle to randomize tie-breaking.
          int[] order = Enumerable.Range(0, bucketsToUse).ToArray();
          ShuffleArrayInPlace(order);

          // Sort by remainder descending; shuffle order provides stable random tie-breaks.
          Array.Sort(order, (a, b) =>
          {
            int cmp = rema[b].CompareTo(rema[a]);
            if (cmp != 0)
            {
              return cmp;
            }

            // Random tie-break already injected via shuffle "order".
            return 0;
          });

          for (int k = 0; k < leftover; k++)
          {
            int iSel = order[k];
            extra[iSel] += 1;
          }

          // Apply extras on top of the baseline 1 each.
          for (int i = 0; i < bucketsToUse; i++)
          {
            int idx = selected[i];
            result[idx] += extra[i];
          }
        }
      }

      // Update per-bucket usage counts and global inUseCount.
      for (int i = 0; i < bucketsToUse; i++)
      {
        int idx = selected[i];
        if (bucketUseCounts[idx] == 0)
        {
          inUseCount++;
        }
        bucketUseCounts[idx]++;
      }

      return result;
    }
  }

  /// <summary>
  /// Deallocates the buckets indicated by the nonzero entries in <paramref name="allocatedBucketItems"/>.
  /// The array must be exactly the length of the bucket set. Entries &gt; 0 mean:
  /// "this client was using that bucket and is now releasing it".
  ///
  /// Each such bucket's usage count is decremented; if it reaches zero, the bucket becomes "free"
  /// (no active clients) again.
  /// </summary>
  public void Deallocate(int[] allocatedBucketItems)
  {
    if (allocatedBucketItems == null)
    {
      throw new ArgumentNullException(nameof(allocatedBucketItems));
    }

    if (allocatedBucketItems.Length != numBuckets)
    {
      throw new ArgumentException("Deallocation array length must match the bucket count.", nameof(allocatedBucketItems));
    }

    lock (itemsLock)
    {
      int bucketsFreed = 0;

      // Validate first.
      for (int i = 0; i < numBuckets; i++)
      {
        int count = allocatedBucketItems[i];
        if (count < 0)
        {
          throw new ArgumentOutOfRangeException(nameof(allocatedBucketItems), "Counts must be non-negative.");
        }

        if (count > 0)
        {
          if (bucketUseCounts[i] <= 0)
          {
            throw new InvalidOperationException("Attempted to deallocate a bucket that has no active clients: " + i.ToString());
          }
        }
      }

      // Perform deallocation.
      for (int i = 0; i < numBuckets; i++)
      {
        if (allocatedBucketItems[i] > 0)
        {
          int before = bucketUseCounts[i];
          bucketUseCounts[i] = before - 1;

          if (bucketUseCounts[i] < 0)
          {
            throw new InvalidOperationException("Bucket usage count became negative for bucket: " + i.ToString());
          }

          if (before > 0 && bucketUseCounts[i] == 0)
          {
            bucketsFreed++;
          }
        }
      }

      if (bucketsFreed > 0)
      {
        inUseCount -= bucketsFreed;
      }
    }
  }


  #region Internal randomization utilities

  private static void FisherYatesShuffleInPlace(List<int> list)
  {
    int n = list.Count;
    for (int i = n - 1; i > 0; i--)
    {
      int j = Random.Shared.Next(i + 1);
      if (j != i)
      {
        int tmp = list[i];
        list[i] = list[j];
        list[j] = tmp;
      }
    }
  }

  private static void ShuffleArrayInPlace(int[] array)
  {
    int n = array.Length;
    for (int i = n - 1; i > 0; i--)
    {
      int j = Random.Shared.Next(i + 1);
      if (j != i)
      {
        int tmp = array[i];
        array[i] = array[j];
        array[j] = tmp;
      }
    }
  }

  #endregion
}
