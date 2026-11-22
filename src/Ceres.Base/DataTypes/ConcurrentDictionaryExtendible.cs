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
using System.Numerics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
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


  struct Entry
  {
    public TKey Key;
    public TValue Value;
  }


  sealed class Bucket
  {
    public int LocalDepth;
    public int Count;

    /// <summary>
    /// Hash codes of the entries, parallel to Entries.
    ///
    /// Held apart from the key/value pairs so that probing touches 4 bytes per candidate
    /// instead of striding across whole entries. For a full bucket that is 8 cache lines
    /// rather than dozens, which is what dominates lookup cost once the table outgrows L2.
    /// </summary>
    public int[] Hashes;

    public Entry[] Entries;
    public readonly Lock SyncRoot = new();
  }


  /// <summary>
  /// Returns the index within a bucket of the entry matching hashCode and key, or -1 if absent.
  /// Scans the hash array (vectorized where available) and compares keys only where a hash matched.
  /// </summary>
  static int FindIndex(int[] hashes, Entry[] entries, int count, int hashCode, TKey key)
  {
    int i = 0;

    if (Vector256.IsHardwareAccelerated && count >= Vector256<int>.Count)
    {
      ref int hashesRef = ref MemoryMarshal.GetArrayDataReference(hashes);
      Vector256<int> target = Vector256.Create(hashCode);
      int lastBlockStart = count - Vector256<int>.Count;

      for (; i <= lastBlockStart; i += Vector256<int>.Count)
      {
        uint matches = Vector256.Equals(Vector256.LoadUnsafe(ref hashesRef, (nuint)i), target)
                                .ExtractMostSignificantBits();

        // Usually zero; when set, verify each candidate lane against the real key.
        while (matches != 0)
        {
          int index = i + BitOperations.TrailingZeroCount(matches);
          if (entries[index].Key.Equals(key))
          {
            return index;
          }

          matches &= matches - 1;   // clear the lowest set bit, keep checking the rest
        }
      }
    }

    for (; i < count; i++)
    {
      if (hashes[i] == hashCode && entries[i].Key.Equals(key))
      {
        return i;
      }
    }

    return -1;
  }


  /// <summary>
  /// Returns the smallest power-of-2 capacity that can hold <paramref name="count"/> entries,
  /// clamped to [INITIAL_BUCKET_CAPACITY .. BUCKET_CAPACITY].
  /// </summary>
  static int RightSizeCapacity(int count)
  {
    int cap = RoundUpPowerOf2(System.Math.Max(count, INITIAL_BUCKET_CAPACITY));
    return System.Math.Min(cap, BUCKET_CAPACITY);
  }


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
      directory[i] = new Bucket
      {
        LocalDepth = globalDepth,
        Hashes = new int[INITIAL_BUCKET_CAPACITY],
        Entries = new Entry[INITIAL_BUCKET_CAPACITY]
      };
    }
  }


  /// <summary>
  /// Returns the number of entries in the map.
  /// </summary>
  public int Count => Volatile.Read(ref totalCount);


  /// <summary>
  /// Attempts to get the value associated with the specified key.
  /// Uses per-bucket locking to avoid races with concurrent modifications.
  /// </summary>
  public bool TryGetValue(TKey key, out TValue value)
  {
    int hashCode = key.GetHashCode();

    while (true)
    {
      Bucket bucket = GetBucket(hashCode);

      lock (bucket.SyncRoot)
      {
        // Re-check bucket after acquiring lock (split may have redirected).
        Bucket currentBucket = GetBucket(hashCode);
        if (currentBucket != bucket)
        {
          continue; // Retry with correct bucket.
        }

        int index = FindIndex(bucket.Hashes, bucket.Entries, bucket.Count, hashCode, key);
        if (index >= 0)
        {
          value = bucket.Entries[index].Value;
          return true;
        }

        value = default;
        return false;
      }
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
      Bucket bucket = GetBucket(hashCode);

      lock (bucket.SyncRoot)
      {
        // Re-check bucket after acquiring lock (split may have redirected).
        Bucket currentBucket = GetBucket(hashCode);
        if (currentBucket != bucket)
        {
          continue; // Retry with correct bucket.
        }

        // Check for existing key.
        if (FindIndex(bucket.Hashes, bucket.Entries, bucket.Count, hashCode, key) >= 0)
        {
          return false; // Key already exists.
        }

        // Room in bucket's current array - add directly.
        if (bucket.Count < bucket.Entries.Length)
        {
          int at = bucket.Count;
          bucket.Hashes[at] = hashCode;
          ref Entry newEntry = ref bucket.Entries[at];
          newEntry.Key = key;
          newEntry.Value = value;
          bucket.Count++;
          Interlocked.Increment(ref totalCount);
          return true;
        }

        // Array full but under max capacity - grow and append.
        if (bucket.Entries.Length < BUCKET_CAPACITY)
        {
          int newCap = System.Math.Min(bucket.Entries.Length * 2, BUCKET_CAPACITY);
          int[] grownHashes = new int[newCap];
          Entry[] grown = new Entry[newCap];
          Array.Copy(bucket.Hashes, grownHashes, bucket.Count);
          Array.Copy(bucket.Entries, grown, bucket.Count);
          grownHashes[bucket.Count] = hashCode;
          ref Entry newEntry = ref grown[bucket.Count];
          newEntry.Key = key;
          newEntry.Value = value;
          bucket.Hashes = grownHashes;
          bucket.Entries = grown;
          bucket.Count++;
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
        Bucket bucket = GetBucket(hashCode);

        lock (bucket.SyncRoot)
        {
          // Re-check bucket after acquiring lock (split may have redirected).
          Bucket currentBucket = GetBucket(hashCode);
          if (currentBucket != bucket)
          {
            continue; // Retry with correct bucket.
          }

          // Check for existing key - update in place.
          int existing = FindIndex(bucket.Hashes, bucket.Entries, bucket.Count, hashCode, key);
          if (existing >= 0)
          {
            bucket.Entries[existing].Value = value;
            return;
          }

          // Not found - insert if room in current array.
          if (bucket.Count < bucket.Entries.Length)
          {
            int at = bucket.Count;
            bucket.Hashes[at] = hashCode;
            ref Entry newEntry = ref bucket.Entries[at];
            newEntry.Key = key;
            newEntry.Value = value;
            bucket.Count++;
            Interlocked.Increment(ref totalCount);
            return;
          }

          // Array full but under max capacity - grow and insert.
          if (bucket.Entries.Length < BUCKET_CAPACITY)
          {
            int newCap = System.Math.Min(bucket.Entries.Length * 2, BUCKET_CAPACITY);
            int[] grownHashes = new int[newCap];
            Entry[] grown = new Entry[newCap];
            Array.Copy(bucket.Hashes, grownHashes, bucket.Count);
            Array.Copy(bucket.Entries, grown, bucket.Count);
            grownHashes[bucket.Count] = hashCode;
            ref Entry newEntry = ref grown[bucket.Count];
            newEntry.Key = key;
            newEntry.Value = value;
            bucket.Hashes = grownHashes;
            bucket.Entries = grown;
            bucket.Count++;
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
  Bucket GetBucket(int hashCode)
  {
    Bucket[] dir = Volatile.Read(ref directory);
    int index = hashCode & (dir.Length - 1);
    return dir[index];
  }


  /// <summary>
  /// Splits an overflowing bucket. Caller must hold bucket.SyncRoot.
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

          // GetBucket indexes by lowest globalDepth bits: hashCode & (dir.Length - 1).
          // Doubling adds one new high bit (bit D). For hash h with lowest D bits = b:
          //   new index = b (bit D=0) or b+oldLen (bit D=1).
          // Both must initially point to the same bucket as old index b.
          for (int i = 0; i < oldLen; i++)
          {
            newDir[i] = directory[i];
            newDir[i + oldLen] = directory[i];
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
      if ((bucket.Hashes[i] & splitBit) == 0)
      {
        keepCount++;
      }
    }

    int sibCount = oldCount - keepCount;

    // Allocate right-sized arrays for both halves.
    int keepCap = RightSizeCapacity(keepCount);
    int sibCap = RightSizeCapacity(sibCount);
    int[] keepHashes = new int[keepCap];
    int[] sibHashes = new int[sibCap];
    Entry[] keepEntries = new Entry[keepCap];
    Entry[] sibEntries = new Entry[sibCap];

    // Second pass: fill both arrays.
    int ki = 0, si = 0;
    for (int i = 0; i < oldCount; i++)
    {
      int entryHash = bucket.Hashes[i];
      if ((entryHash & splitBit) != 0)
      {
        sibHashes[si] = entryHash;
        sibEntries[si++] = bucket.Entries[i];
      }
      else
      {
        keepHashes[ki] = entryHash;
        keepEntries[ki++] = bucket.Entries[i];
      }
    }

    Bucket sibling = new Bucket
    {
      LocalDepth = newLocalDepth,
      Hashes = sibHashes,
      Entries = sibEntries,
      Count = sibCount
    };

    bucket.LocalDepth = newLocalDepth;

    // Publish the new arrays before the count, so that any future lock-free reader which
    // does Volatile.Read(Count) and then reads Hashes/Entries is guaranteed by the
    // acquire/release pair to see arrays at least as new as the count it observed.
    // (All readers currently take the bucket lock, so this ordering is not yet relied upon.)
    bucket.Hashes = keepHashes;
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
  /// Removes all entries while RETAINING the directory and bucket allocations: the directory,
  /// globalDepth, and each bucket's Entries array and LocalDepth are preserved. Only each bucket's
  /// Count (and totalCount) is reset to zero; Entry slots are intentionally not cleared because all
  /// reads are gated by Count, so a zero-Count bucket reads as empty regardless of stale slot data.
  ///
  /// This retention is the point: a subsequent refill reuses the already-grown buckets and so incurs
  /// no (or far fewer) splits / directory doublings.
  ///
  /// PRECONDITION: the caller must guarantee EXCLUSIVE (single-threaded, quiescent) access — no other
  /// operation may be in flight concurrently (as during graph rewrite, which is the only caller). It
  /// is fast precisely because it exploits that: it takes no locks and uses no volatile/interlocked
  /// accesses, and it walks the directory directly without tracking distinct buckets. The latter is
  /// safe because zeroing a shared bucket's Count more than once is idempotent — so we avoid the
  /// per-slot reference hashing + set insert (and the HashSet allocation) that a distinct-bucket walk
  /// would otherwise cost on a directory with hundreds of thousands of slots. Cost is a single tight
  /// pass over the directory.
  /// </summary>
  public void Clear()
  {
    Bucket[] dir = directory;
    for (int i = 0; i < dir.Length; i++)
    {
      dir[i].Count = 0;
    }
    totalCount = 0;
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
