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

using Ceres.Base;
using Ceres.Base.Benchmarking;
using Ceres.Base.DataType;
using Ceres.Base.DataTypes;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Chess.PositionEvalCaching
{
  /// <summary>
  /// Maintains a memory cache of NNEvaluator results
  /// (also with serialization/deserialization capability to disk).
  /// </summary>
  [Serializable]
  public class PositionEvalCache
  {
    /// <summary>
    /// 
    /// </summary>
    public enum CacheMode 
    {  
      /// <summary>
      /// Evaluations not cached.
      /// </summary>
      None, 

      /// <summary>
      /// Evaluations cached in memory.
      /// </summary>
      MemoryOnly, 

      /// <summary>
      /// Evaluations cached in memory and persisted to disk at end of search.
      /// </summary>
      MemoryAndDisk 
    };


    /// <summary>
    /// If cache is not to be updated (calls to Store are no-ops),
    /// for example in cases where a cache is just ancillary
    /// (such as when using cache from nodes of a reused tree that were not retained).
    /// </summary>
    public bool ReadOnly { get; set; } = false;

    /// <summary>
    /// If concurrent updates are supported.
    /// </summary>
    public bool SupportsConcurrency { get; private set; }

    /// <summary>
    /// Number of entries currently in cache.
    /// </summary>
    public int Count => positionCache.Count;


    static string PositionEvalCacheBaseFN = System.IO.Path.GetTempPath(); // TODO: make this adjustable
  

    static string CACHE_FN(string id) => Path.Combine(PositionEvalCacheBaseFN, id + ".cache.dat");

    const int EST_CONCURRENT_THREADS = 16;


    IDictionary<ulong, PositionEvalCacheEntry> positionCache;

    public PositionEvalCache()
    {
    }

    public void InitializeWithSize(bool supportConcurrency, int size)
    {
      positionCache = supportConcurrency ? new ConcurrentDictionary<ulong, PositionEvalCacheEntry>(EST_CONCURRENT_THREADS, size)
                                        :  new Dictionary<ulong, PositionEvalCacheEntry>(/*EST_CONCURRENT_THREADS,*/ size);
      SupportsConcurrency = supportConcurrency;
    }


    public bool TryLookupFromHash(ulong hash, ref PositionEvalCacheEntry entry)
    {
      entry = default;
      entry.WinP = entry.LossP = FP16.NaN;

      bool found = positionCache.TryGetValue(hash, out entry);
      return found;
    }


    /// <summary>
    /// 
    /// Note that it is possible and acceptable that the specified entry might already exist.
    /// For example, when processing a large batch there might be tranpositions resulting in 
    /// multiple nodes having same position. It is harmless to store two or more times.
    /// </summary>
    /// <param name="hash"></param>
    /// <param name="value"></param>
    /// <param name="policy"></param>
    public void Store(ulong hash, GameResult terminalStatus, FP16 winP, FP16 lossP, FP16 m, in CompressedPolicyVector policy)
    {
      if (!ReadOnly)
      {
        Debug.Assert(!float.IsNaN(winP + lossP));

        positionCache[hash] = new PositionEvalCacheEntry(terminalStatus, winP, lossP, m, in policy);
      }
    }


    #region NEW

    public void LoadFromDisk(string id)
    {
      if (!System.IO.File.Exists(CACHE_FN(id)))
      {
        Console.WriteLine($"{id} cache file did not exist and was not loaded.");
        return;
      }

      using (new TimingBlock("LoadFromDisk " + CACHE_FN(id)))
      {
        using (FileStream ms = new FileStream(CACHE_FN(id), FileMode.Open, FileAccess.Read))
        {
          int maxEntries = int.MaxValue / Marshal.SizeOf<PositionEvalCacheEntry>();

          byte[] bytesInt = new byte[4];
          ms.Read(bytesInt);
          uint sizeEntries = (uint)SerializationUtils.Deserialize<int>(bytesInt);

          if (sizeEntries > maxEntries)
          {
            Console.WriteLine("cache file overflow, truncating");
            sizeEntries = (uint)(maxEntries - 1);
          }

          numEntriesUponLoad = sizeEntries;

          throw new NotImplementedException();
          //positionCache = new ConcurrentDictionary<ulong, PositionEvalCacheEntry>(); // no hint available (int)(numEntriesUponLoad * 1.1));

          byte[] keysRaw    = new byte[sizeEntries * sizeof(ulong)];
          ms.Read(keysRaw, 0, keysRaw.Length);
          ulong[] keys = SerializationUtils.DeSerializeArray<ulong>(keysRaw, keysRaw.Length);

          byte[] entriesRaw = new byte[sizeEntries * (uint)Marshal.SizeOf<PositionEvalCacheEntry>()];
          ms.Read(entriesRaw, 0, entriesRaw.Length);
          PositionEvalCacheEntry[]  entries = SerializationUtils.DeSerializeArray<PositionEvalCacheEntry>(entriesRaw, entriesRaw.Length);

          for (int i = 0; i < sizeEntries; i++)
            positionCache[keys[i]] = entries[i];
        }
      }
    }


    static uint numEntriesUponLoad;

    [Serializable]
    [StructLayout(LayoutKind.Sequential, Pack = 2)]
    readonly struct Entry
    {
      public readonly ulong Hash;
      public readonly PositionEvalCacheEntry Value;
    }


    public void SaveToDisk(string id)
    {
      // Log.Info($"Saving cache to disk file {CACHE_FN(id)}");

      ulong[] keys = new ulong[positionCache.Count];
      PositionEvalCacheEntry[] entries = new PositionEvalCacheEntry[positionCache.Count];

      if (numEntriesUponLoad < positionCache.Count)
      {
        //using (new TimingBlock("SaveToDisk new"))
        {
          int index = 0;
          foreach (KeyValuePair<ulong, PositionEvalCacheEntry> entry in positionCache)
          {
            keys[index] = entry.Key;
            entries[index] = entry.Value;
            index++;
          }

          using (FileStream ms = new FileStream(CACHE_FN(id), FileMode.OpenOrCreate))
          {
            ms.Write(SerializationUtils.Serialize(positionCache.Count));
            ms.Write(SerializationUtils.SerializeArray<ulong>(keys));
            ms.Write(SerializationUtils.SerializeArray<PositionEvalCacheEntry>(entries));

          }
        }
      }
    }
   
    #endregion


#if NOT

    public static unsafe void SaveToDiskOLD()
    {
      if (positionCache.Count > numEntriesUponLoad)
      {
        using (new TimingBlock("SaveToDisk"))
        {
          // TO DO: could be faster by preallocating buffers
          //        byte[] bytesULong = new byte[sizeof(ulong)];
          //        byte[] bytesEntry = new byte[Marshal.SizeOf<PositionEvalCacheEntry>()];

          using (FileStream ms = new FileStream(CACHE_FN, FileMode.OpenOrCreate))
          {
            ms.Write(Serialize(positionCache.Count));
            foreach (KeyValuePair<ulong, PositionEvalCacheEntry> entry in positionCache)
            {
              ms.Write(Serialize(entry.Key));
              ms.Write(Serialize(entry.Value));
            }
          }
        }
      }
    }



    public static void LoadFromDiskOLD()
    {
      if (!System.IO.File.Exists(CACHE_FN)) return;

      using (new TimingBlock("LoadFromDisk"))
      {
        using (FileStream ms = new FileStream(CACHE_FN, FileMode.Open, FileAccess.Read))
        {
          byte[] bytesInt = new byte[4];
          ms.Read(bytesInt);
          int size = Deserialize<int>(bytesInt);
          numEntriesUponLoad = size;

          byte[] bytesULong = new byte[sizeof(ulong)];
          byte[] bytesEntry = new byte[Marshal.SizeOf<PositionEvalCacheEntry>()];

          for (int i = 0; i < size; i++)
          {
            ms.Read(bytesULong);
            ulong key = Deserialize<ulong>(bytesULong);

            ms.Read(bytesEntry);
            PositionEvalCacheEntry entry = Deserialize<PositionEvalCacheEntry>(bytesEntry);

            positionCache[key] = entry;
          }
        }
      }
    }

#endif
#if OLD

    public static bool LoadFromDisk()
    {
      if (File.Exists(CACHE_FN))
      {
        IFormatter formatter = new BinaryFormatter();
        Stream stream = new FileStream(CACHE_FN, FileMode.Open, FileAccess.Read, FileShare.Read);
        using (new TimingBlock("Deserialize PositionEvalCache"))
          positionCache = (Dictionary<ulong, PositionEvalCacheEntry>)formatter.Deserialize(stream);
        stream.Close();

        numEntriesUponLoad = positionCache.Count;

        Console.WriteLine($"PositionEvalCache read {positionCache.Count}");
        return true;
      }
      else
        return false;
    }
  }

    public static void SaveToDisk()
    {
      if (numEntriesUponLoad < positionCache.Count)
      {
        IFormatter formatter = new BinaryFormatter();
        Stream stream = new FileStream(CACHE_FN, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None);
        using (new TimingBlock($"Serialize PositionEvalCache {positionCache.Count}"))
          formatter.Serialize(stream, positionCache);
        stream.Close();
      }
#endif

  }


}

#if NOT
      // Warning: When we are extracting position from LZTrainingPositionRaw
      // we don't have a LZPositionMiscInfo available, thus we can't possibly match the hash from other methods that take a List<Position>

    public static bool TryLookup(in LZTrainingPositionRaw pos, ref PositionEvalCacheEntry entry)
    {
      ulong hash = PositionCacheHelpers.Hash(in pos);
      return TryLookupFromHash(hash, ref entry);
    }


    public static bool TryLookup(Span<Position> positions, ref PositionEvalCacheEntry entry)
    {
      if (MCTSParams.CACHE_MODE == CacheMode.None) return false;

      entry = default;
      entry.Value = float.NaN;
      ulong hash = PositionCacheHelpers.Hash(positions, MCTSParams.HISTORY_FILL_IN);

      bool found = positionCache.TryGetValue(hash, out entry);
      return found;
    }

    public static bool TryLookup(in Position pos, ref PositionEvalCacheEntry entry)
    {
      if (MCTSParams.CACHE_MODE == CacheMode.None) return false;

      entry = default;
      entry.Value = float.NaN;
      ulong hash = PositionCacheHelpers.Hash(in pos);

      return positionCache.TryGetValue(hash, out entry);
    }


    /// <summary>
    /// 
    /// Note that it is possible and acceptable that the specified entry might already exist.
    /// For example, when processing a large batch there might be tranpositions resulting in 
    /// multiple nodes having same position. It is harmless to store two or more times.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="value"></param>
    /// <param name="policy"></param>
    public static void Store(in Position pos, SpeculativeTreeNode.TerminalStatus terminalStatus, float value, in ChessPolicyVectorCompressed policy)
    {
      Debug.Assert(!float.IsNaN(value));

      if (MCTSParams.CACHE_MODE == CacheMode.None) return;

      positionCache[PositionCacheHelpers.Hash(in pos)] = new PositionEvalCacheEntry(terminalStatus, value, in policy);
    }


    /// <summary>
    /// 
    /// Note that it is possible and acceptable that the specified entry might already exist.
    /// For example, when processing a large batch there might be tranpositions resulting in 
    /// multiple nodes having same position. It is harmless to store two or more times.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="value"></param>
    /// <param name="policy"></param>
    public static void Store(in LZTrainingPositionRaw pos, SpeculativeTreeNode.TerminalStatus terminalStatus, float value, in ChessPolicyVectorCompressed policy) 
    {
      Debug.Assert(!float.IsNaN(value));

      if (MCTSParams.CACHE_MODE == CacheMode.None) return;
      
      ulong hash = PositionCacheHelpers.Hash(in pos);
      positionCache[hash] = new PositionEvalCacheEntry(terminalStatus, value, in policy);
    }

#endif
