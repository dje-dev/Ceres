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
using System.IO;
using System.IO.MemoryMappedFiles;
using System.Threading;

using SysMath = global::System.Math;

#endregion

namespace Ceres.Base.Environment
{
  /// <summary>
  /// Provides protection against runaway recursive process spawning by tracking
  /// the number of concurrently running instances using shared memory.
  /// 
  /// This guard uses OS-level shared memory and a named mutex to coordinate
  /// across multiple process instances, preventing scenarios where a bug could
  /// cause infinite recursive process creation.
  /// </summary>
  public static class ProcessRecursiveOverflowGuard
  {
    /// <summary>
    /// Maximum number of concurrent process instances allowed before
    /// triggering a protective shutdown.
    /// </summary>
    private const int MAX_CONCURRENT_INSTANCES = 8;

    /// <summary>
    /// Prefix for the global shared memory name.
    /// </summary>
    private const string SHARED_MEMORY_PREFIX = "Global\\";

    /// <summary>
    /// Suffix appended to the process ID string for the shared memory counter.
    /// </summary>
    private const string COUNTER_SUFFIX = "Counter";

    /// <summary>
    /// Suffix appended to the process ID string for the mutex name.
    /// </summary>
    private const string MUTEX_SUFFIX = "CounterMutex";


    /// <summary>
    /// Checks if too many instances of the process are running and increments the counter.
    /// Should be called once at process startup.
    /// </summary>
    /// <param name="processIDString">Unique identifier string for this process type (e.g., "CeresProcess")</param>
    /// <exception cref="ArgumentNullException">Thrown when processIDString is null or empty</exception>
    /// <remarks>
    /// If the number of running instances exceeds MAX_CONCURRENT_INSTANCES,
    /// the process will be terminated with exit code 3 to prevent potential
    /// infinite recursion scenarios.
    /// </remarks>
    public static void CheckRecursiveOverflow(string processIDString)
    {
      ArgumentException.ThrowIfNullOrWhiteSpace(processIDString, nameof(processIDString));

      string sharedMemoryName = SHARED_MEMORY_PREFIX + processIDString + COUNTER_SUFFIX;
      string mutexName = SHARED_MEMORY_PREFIX + processIDString + MUTEX_SUFFIX;

      using Mutex mutex = new(false, mutexName, out _);
      try
      {
        mutex.WaitOne();

        using MemoryMappedFile mmf = MemoryMappedFile.CreateOrOpen(sharedMemoryName, sizeof(int));
        using MemoryMappedViewAccessor accessor = mmf.CreateViewAccessor();

        accessor.Read(0, out int count);
        count++;

        if (count > MAX_CONCURRENT_INSTANCES)
        {
          Console.WriteLine($"Shutting down '{processIDString}': possible infinite process recursion detected. " +
                            $"Maximum of {MAX_CONCURRENT_INSTANCES} concurrent instances exceeded.");
          System.Environment.Exit(3);
        }

        accessor.Write(0, count);
      }
      catch (AbandonedMutexException)
      {
        // Mutex was abandoned by another process (likely crashed).
        // The mutex is still acquired, so it's safe to proceed.
        Console.WriteLine($"Warning: Mutex for '{processIDString}' was abandoned. Continuing execution.");
      }
      finally
      {
        mutex.ReleaseMutex();
      }
    }


    /// <summary>
    /// Decrements the process instance counter by the specified amount.
    /// Should be called once when the process is exiting.
    /// </summary>
    /// <param name="processIDString">Unique identifier string for this process type (must match the value used in CheckRecursiveOverflow)</param>
    /// <param name="decrementCount">Number to decrement the counter by (typically 1)</param>
    /// <exception cref="ArgumentNullException">Thrown when processIDString is null or empty</exception>
    /// <exception cref="ArgumentOutOfRangeException">Thrown when decrementCount is less than 1</exception>
    public static void DecrementCounter(string processIDString, int decrementCount = 1)
    {
      ArgumentException.ThrowIfNullOrWhiteSpace(processIDString, nameof(processIDString));
      ArgumentOutOfRangeException.ThrowIfLessThan(decrementCount, 1, nameof(decrementCount));

      string sharedMemoryName = SHARED_MEMORY_PREFIX + processIDString + COUNTER_SUFFIX;
      string mutexName = SHARED_MEMORY_PREFIX + processIDString + MUTEX_SUFFIX;

      using Mutex mutex = new(false, mutexName, out _);
      try
      {
        mutex.WaitOne();

        using MemoryMappedFile mmf = MemoryMappedFile.OpenExisting(sharedMemoryName);
        using MemoryMappedViewAccessor accessor = mmf.CreateViewAccessor();

        accessor.Read(0, out int count);
        count = SysMath.Max(0, count - decrementCount);
        accessor.Write(0, count);
      }
      catch (AbandonedMutexException)
      {
        // Mutex was abandoned, but still acquired. Proceed with decrement.
        Console.WriteLine($"Warning: Mutex for '{processIDString}' was abandoned during decrement.");
      }
      catch (FileNotFoundException)
      {
        // Shared memory does not exist; nothing to decrement.
        // This can happen if the process crashes before CheckRecursiveOverflow completes.
      }
      finally
      {
        mutex.ReleaseMutex();
      }
    }
  }
}
