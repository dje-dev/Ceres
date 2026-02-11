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
using System.Collections;
using System.Collections.Generic;
using System.Runtime.CompilerServices;
using System.Threading;

#endregion

namespace Ceres.Base.DataTypes;

/// <summary>
/// A concurrent hash map using extendible hashing.
/// Eliminates stop-the-world resize pauses: only one bucket splits at a time.
/// Per-bucket locking provides fine-grained concurrency.
/// </summary>
public class ConcurrentDictionaryExtendible<TKey, TValue> : IConcurrentDictionary<TKey, TValue> where TKey : IEquatable<TKey>
{
  /// <summary>
  /// Maximum number of entries per bucket before a split is required.
  /// </summary>
  const int BUCKET_CAPACITY = 128;

  /// <summary>
  /// Starting array size for buckets. 
  /// </summary>
  const int INITIAL_BUCKET_CAPACITY = 8;

  /// <summary>
  /// Number of lock stripes for synchronizing writers. 
  /// </summary>
  const int NUM_LOCK_STRIPES = 1024;


  struct Entry
  {
    public int HashCode;
    public TKey Key;
    public TValue Value;
  }

  
  sealed class Bucket
  {
    public int LocalDepth;
    public int Count;
    public Entry[] Entries;
  }


  /// <summary>
  /// Returns the smallest power-of-2 capacity that can hold <paramref name="count"/> entries,
  /// clamped to [INITIAL_BUCKET_CAPACITY .. BUCKET_CAPACITY].
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  static int RightSizeCapacity(int count)
  {
    int cap = RoundUpPowerOf2(System.Math.Max(count, INITIAL_BUCKET_CAPACITY));
    return System.Math.Min(cap, BUCKET_CAPACITY);
  }


  readonly Lock[] lockStripes = InitLockStripes();

  static Lock[] InitLockStripes()
  {
    Lock[] stripes = new Lock[NUM_LOCK_STRIPES];
    for (int i = 0; i < NUM_LOCK_STRIPES; i++)
    {
      stripes[i] = new Lock();
    }
    return stripes;
  }


  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  Lock GetStripeLock(int hashCode) => lockStripes[(uint)hashCode % NUM_LOCK_STRIPES];

  Bucket[] directory;
  int globalDepth;
  int totalCount;
  readonly Lock directoryLock = new();


  /// <summary>
  /// Creates a new ExtendibleConcurrentHashMap with an initial capacity hint.
  /// </summary>
  /// <param name="concurrencyLevel">Stored but unused (per-bucket locking provides fine-grained concurrency)</param>
  /// <param name="capacity">Hint for initial number of entries to accommodate</param>
  public ConcurrentDictionaryExtendible(int concurrencyLevel, int capacity)
  {
    int numBuckets = System.Math.Max(16, RoundUpPowerOf2(capacity / BUCKET_CAPACITY));
    globalDepth = Log2(numBuckets);

    directory = new Bucket[numBuckets];
    for (int i = 0; i < numBuckets; i++)
    {
      directory[i] = new Bucket { LocalDepth = globalDepth, Entries = new Entry[INITIAL_BUCKET_CAPACITY] };
    }
  }


  /// <summary>
  /// Returns the number of entries in the map.
  /// </summary>
  public int Count => Volatile.Read(ref totalCount);


  /// <summary>
  /// Attempts to get the value associated with the specified key.
  /// Lock-free: uses acquire/release ordering with copy-on-split
  /// so the read path never enters a Monitor.
  /// </summary>
  public bool TryGetValue(TKey key, out TValue value)
  {
    int hashCode = key.GetHashCode();

    while (true)
    {
      Bucket bucket = GetBucket(hashCode);

      // Acquire-read of Count guarantees that all entry writes committed
      // before the corresponding count increment are visible.
      int count = Volatile.Read(ref bucket.Count);
      Entry[] entries = bucket.Entries;

      for (int i = 0; i < count; i++)
      {
        ref Entry entry = ref entries[i];
        if (entry.HashCode == hashCode && entry.Key.Equals(key))
        {
          value = entry.Value;
          return true;
        }
      }

      // Key not found — verify this bucket is still the correct one.
      // A concurrent split may have moved the key to a sibling bucket
      // and redirected the directory entry.
      if (GetBucket(hashCode) != bucket)
      {
        continue; // Bucket was split; retry with the new bucket.
      }

      value = default;
      return false;
    }
  }


  /// <summary>
  /// Attempts to add a key/value pair. Returns true if added, false if key already exists.
  /// </summary>
  public bool TryAdd(TKey key, TValue value)
  {
    int hashCode = key.GetHashCode();

    while (true)
    {
      lock (GetStripeLock(hashCode))
      {
        Bucket bucket = GetBucket(hashCode);

        // Check for existing key.
        for (int i = 0; i < bucket.Count; i++)
        {
          ref Entry entry = ref bucket.Entries[i];
          if (entry.HashCode == hashCode && entry.Key.Equals(key))
          {
            return false; // Key already exists.
          }
        }

        // Room in bucket's current array - add directly.
        if (bucket.Count < bucket.Entries.Length)
        {
          ref Entry newEntry = ref bucket.Entries[bucket.Count];
          newEntry.HashCode = hashCode;
          newEntry.Key = key;
          newEntry.Value = value;
          Volatile.Write(ref bucket.Count, bucket.Count + 1);
          Interlocked.Increment(ref totalCount);
          return true;
        }

        // Array full but under max capacity - grow and append.
        if (bucket.Entries.Length < BUCKET_CAPACITY)
        {
          int newCap = System.Math.Min(bucket.Entries.Length * 2, BUCKET_CAPACITY);
          Entry[] grown = new Entry[newCap];
          Array.Copy(bucket.Entries, grown, bucket.Count);
          ref Entry newEntry = ref grown[bucket.Count];
          newEntry.HashCode = hashCode;
          newEntry.Key = key;
          newEntry.Value = value;
          bucket.Entries = grown;
          Volatile.Write(ref bucket.Count, bucket.Count + 1);
          Interlocked.Increment(ref totalCount);
          return true;
        }

        // Bucket full at max capacity - split it, then retry.
        SplitBucket(bucket, hashCode);
      }
      // Loop will retry with the (now-split) bucket.
    }
  }


  /// <summary>
  /// Sets the value for the specified key (upsert semantics).
  /// </summary>
  public TValue this[TKey key]
  {
    set
    {
      int hashCode = key.GetHashCode();

      while (true)
      {
        lock (GetStripeLock(hashCode))
        {
          Bucket bucket = GetBucket(hashCode);

          // Check for existing key - update in place.
          for (int i = 0; i < bucket.Count; i++)
          {
            ref Entry entry = ref bucket.Entries[i];
            if (entry.HashCode == hashCode && entry.Key.Equals(key))
            {
              entry.Value = value;
              return;
            }
          }

          // Not found - insert if room in current array.
          if (bucket.Count < bucket.Entries.Length)
          {
            ref Entry newEntry = ref bucket.Entries[bucket.Count];
            newEntry.HashCode = hashCode;
            newEntry.Key = key;
            newEntry.Value = value;
            Volatile.Write(ref bucket.Count, bucket.Count + 1);
            Interlocked.Increment(ref totalCount);
            return;
          }

          // Array full but under max capacity - grow and insert.
          if (bucket.Entries.Length < BUCKET_CAPACITY)
          {
            int newCap = System.Math.Min(bucket.Entries.Length * 2, BUCKET_CAPACITY);
            Entry[] grown = new Entry[newCap];
            Array.Copy(bucket.Entries, grown, bucket.Count);
            ref Entry newEntry = ref grown[bucket.Count];
            newEntry.HashCode = hashCode;
            newEntry.Key = key;
            newEntry.Value = value;
            bucket.Entries = grown;
            Volatile.Write(ref bucket.Count, bucket.Count + 1);
            Interlocked.Increment(ref totalCount);
            return;
          }

          // Bucket full at max capacity - split and retry.
          SplitBucket(bucket, hashCode);
        }
      }
    }
  }


  /// <summary>
  /// Gets the bucket for a given hash code by masking with current directory size.
  /// </summary>
  [MethodImpl(MethodImplOptions.AggressiveInlining)]
  Bucket GetBucket(int hashCode)
  {
    Bucket[] dir = Volatile.Read(ref directory);
    int index = hashCode & (dir.Length - 1);
    return dir[index];
  }


  /// <summary>
  /// Splits an overflowing bucket. Caller must hold the stripe lock for hashCode.
  /// May double the directory if localDepth == globalDepth.
  /// </summary>
  void SplitBucket(Bucket bucket, int hashCode)
  {
    int oldLocalDepth = bucket.LocalDepth;

    if (oldLocalDepth == globalDepth)
    {
      // Must double the directory first.
      lock (directoryLock)
      {
        // Re-check under directory lock (another thread may have doubled already).
        if (oldLocalDepth == globalDepth)
        {
          int oldLen = directory.Length;
          int newLen = oldLen * 2;
          Bucket[] newDir = new Bucket[newLen];

          for (int i = 0; i < oldLen; i++)
          {
            newDir[2 * i] = directory[i];
            newDir[2 * i + 1] = directory[i];
          }

          globalDepth++;
          Interlocked.Exchange(ref directory, newDir);
        }
      }
    }

    // The bit that distinguishes the two halves.
    int newLocalDepth = oldLocalDepth + 1;
    int splitBit = 1 << oldLocalDepth;

    // Two-pass redistribution with right-sized arrays.
    // First pass: count entries for each side.
    int oldCount = bucket.Count;
    int keepCount = 0;

    for (int i = 0; i < oldCount; i++)
    {
      if ((bucket.Entries[i].HashCode & splitBit) == 0)
      {
        keepCount++;
      }
    }

    int sibCount = oldCount - keepCount;

    // Allocate right-sized arrays for both halves.
    Entry[] keepEntries = new Entry[RightSizeCapacity(keepCount)];
    Entry[] sibEntries = new Entry[RightSizeCapacity(sibCount)];

    // Second pass: fill both arrays.
    int ki = 0, si = 0;
    for (int i = 0; i < oldCount; i++)
    {
      ref Entry entry = ref bucket.Entries[i];
      if ((entry.HashCode & splitBit) != 0)
      {
        sibEntries[si++] = entry;
      }
      else
      {
        keepEntries[ki++] = entry;
      }
    }

    Bucket sibling = new Bucket { LocalDepth = newLocalDepth, Entries = sibEntries, Count = sibCount };

    bucket.LocalDepth = newLocalDepth;

    // Publish the new entries array before the count.
    // Lock-free readers do Volatile.Read(Count) then read Entries;
    // the acquire/release pair guarantees a reader that sees the new Count
    // also sees the new Entries reference.
    bucket.Entries = keepEntries;
    Volatile.Write(ref bucket.Count, keepCount);

    // Update directory entries that should now point to the sibling.
    // These are entries where bit 'oldLocalDepth' is set in the index,
    // and the lower 'oldLocalDepth' bits match this bucket.
    Bucket[] dir = Volatile.Read(ref directory);
    int dirLen = dir.Length;
    int lowMask = (1 << oldLocalDepth) - 1;
    int bucketLowBits = hashCode & lowMask;

    // The sibling's low bits have the split bit set.
    int siblingLowBits = bucketLowBits | splitBit;

    // Step through all directory entries matching the sibling pattern.
    int step = 1 << newLocalDepth;
    for (int i = siblingLowBits; i < dirLen; i += step)
    {
      dir[i] = sibling;
    }
  }


  /// <summary>
  /// Rounds up to the next power of 2.
  /// </summary>
  static int RoundUpPowerOf2(int value)
  {
    if (value <= 1)
    {
      return 1;
    }

    value--;
    value |= value >> 1;
    value |= value >> 2;
    value |= value >> 4;
    value |= value >> 8;
    value |= value >> 16;
    return value + 1;
  }


  /// <summary>
  /// Returns floor(log2(value)) for a power of 2.
  /// </summary>
  static int Log2(int value)
  {
    int result = 0;
    while ((1 << result) < value)
    {
      result++;
    }
    return result;
  }


  /// <summary>
  /// Enumerates all key/value pairs by visiting each distinct bucket exactly once.
  /// This is intended for diagnostics; concurrent modifications may cause
  /// entries to be skipped or returned more than once.
  /// </summary>
  public IEnumerator<KeyValuePair<TKey, TValue>> GetEnumerator()
  {
    Bucket[] dir = Volatile.Read(ref directory);
    HashSet<Bucket> visited = new(ReferenceEqualityComparer.Instance);

    for (int i = 0; i < dir.Length; i++)
    {
      Bucket bucket = dir[i];
      if (!visited.Add(bucket))
      {
        continue;
      }

      for (int j = 0; j < bucket.Count; j++)
      {
        ref Entry entry = ref bucket.Entries[j];
        yield return new KeyValuePair<TKey, TValue>(entry.Key, entry.Value);
      }

    }
  }


  IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
