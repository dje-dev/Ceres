#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using System;
using System.Diagnostics;
using System.Numerics;
using System.Runtime.InteropServices;

using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem.Linux;
using Ceres.Base.OperatingSystem.Windows;


#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Static helper methods for interfacing with the hardware.
  /// </summary>
  public static class HardwareManager
  {
    public readonly record struct ProcessMemoryInfo(long ManagedBytes,
                                                    long WorkingSetBytes,
                                                    long VirtualBytes,
                                                    long PrivateBytes,
                                                    long UnmanagedEstimateBytes);
    /// <summary>
    /// Maximum expected pages size across all supported OS.
    /// </summary>
    public const int PAGE_SIZE_MAX = 2048 * 1024;

    /// <summary>
    /// Maximum number of processors which are active for this process.
    /// </summary>
    public static int MaxAvailableProcessors { private set; get; } = System.Environment.ProcessorCount;

    /// <summary>
    /// Cached Process object.
    /// </summary>
    static Process process = Process.GetCurrentProcess();


    public static void VerifyHardwareSoftwareCompatability()
    {
      string errorString = null;
      bool isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
      bool isLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);
      if (RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
      {
        errorString = "Mac OSX not currently supported.";
        errorString += "\r\nExperimental support with random NN backend is possible by";
        errorString += "\r\n  (1) set project to .NET 7, (2) change LinuxAPI DllImport";
        errorString += "\r\n  (3) disable prefetch, (4) switch IsLinux to also include MacOS";
        errorString += "\r\n  (5) disable CUDA checks at initialization.";
      }
      else if (!isWindows && !isLinux)
      {
        errorString = "Currently only Windows or Linux operating systems is supported.";
      }
      else if (isWindows && System.Environment.OSVersion.Version.Major < 6) // Note that Windows 7 is version 6
      {
        errorString = "Windows Version 7 or above required.";
      }

      if (errorString != null)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"Fatal Error. {errorString}");
        System.Environment.Exit(-1);
      }
    }

    /// <summary>
    /// Write diagnostic information to console relating to processor configuration.
    /// </summary>
    public static void DumpProcessorInfo()
    {
      if (!SoftwareManager.IsLinux) Console.WriteLine("NumCPUSockets: " + WindowsHardware.NumCPUSockets);
      Console.WriteLine("NumProcessors: " + System.Environment.ProcessorCount);
      Console.WriteLine("Affinity Mask: " + Process.GetCurrentProcess().ProcessorAffinity);
      Console.WriteLine("Memory Size  : " + MemorySize);
    }

    /// <summary>
    /// Returns amount physical memory visible (in bytes).
    /// </summary>
    public static long MemorySize => SoftwareManager.IsLinux ? LinuxAPI.PhysicalMemorySize
                                                             : (long)Win32.MemorySize;


    /// <summary>
    /// Returns estimted managed and native memory usage for the current process.
    /// </summary>
    /// <param name="forceFullGC">if a full garbage collection should be processed</param>
    /// <returns></returns>
    public static ProcessMemoryInfo GetProcessMemoryInfo(bool forceFullGC = false)
    {
      long managed = GC.GetTotalMemory(forceFullGC);

      process.Refresh();
      long workingSet = process.WorkingSet64;           // resident (physical) bytes - RSS on Linux
      long virtualMemory = process.VirtualMemorySize64; // VMSIZE
      long privateBytes = process.PrivateMemorySize64;  // "private bytes" (may be 0 or unreliable on some Unix builds)
      long unmanagedEstimate = System.Math.Max(0L, workingSet - managed);

      return new ProcessMemoryInfo(managed, workingSet, virtualMemory, privateBytes, unmanagedEstimate);
    }


#if NOT
    /// <summary>
    /// Description of the system processor(s) including NUMA domains and virtuaul processors.
    /// </summary>
    public class HardwareNUMAInfo
    {
      public int NumNUMADomains { get; private set; }
      public int? OffsetFirstVirtualProcessor { get; private set; }
      public int NumProcessorsPerNUMADomain { get; private set; }

      public static HardwareNUMAInfo SystemInfo
      {
        get
        {
          // TODO: make dynamic
          string computerName = System.Environment.GetEnvironmentVariable("COMPUTERNAME");
          if (computerName == "S1")
          {
            return new HardwareNUMAInfo() { NumNUMADomains = 2, NumProcessorsPerNUMADomain = 16, OffsetFirstVirtualProcessor = null };

          }
          else if (computerName == "S2")
          {
            return new HardwareNUMAInfo() { NumNUMADomains = 4, NumProcessorsPerNUMADomain = 16, OffsetFirstVirtualProcessor = 64 };

          }
          else
          {
            return null;
          }
        }
      }
    }

#endif
    /// <summary>
    /// Restricts this process to run only on the logical processors belonging to the
    /// specified NUMA node, reducing cross-node memory traffic. On NUMA hardware this
    /// can yield meaningful performance improvements (historically up to ~10%).
    ///
    /// Unlike the previous implementation (which capped usage at the lowest 32 logical
    /// processors), this pins to ALL logical processors on the node, so large single-node
    /// CPUs are fully utilized.
    ///
    /// No-op on non-Windows platforms and on Windows systems with more than one processor
    /// group (where a single ProcessorAffinity mask cannot represent the node).
    /// </summary>
    /// <param name="numaNode">zero-based NUMA node to which the process should be pinned</param>
    public static void AffinitizeSingleNUMANode(int numaNode)
    {
      // The coreinfo.exe utility from Sysinternals is useful for dumping topology under Windows.
      //
      // TODO: Someday it would be desirable to allow different logical calculations
      //       (e.g. distinct chess tree searches) to be placed on distinct sockets,
      //       rather than restricting all computation to a single node. That would
      //       require abandoning the TPL / .NET thread pool in favor of our own
      //       processor-constrained scheduler.
      // TODO: Extend NUMA-node-aware pinning to Linux (e.g. via libnuma or
      //       /sys/devices/system/node/nodeN/cpulist) and to multi-processor-group
      //       Windows systems (via SetThreadGroupAffinity).

      if (!SoftwareManager.IsWindows)
      {
        // Leave the process unaffinitized so the OS scheduler can place threads.
        return;
      }

      try
      {
        if (WindowsHardware.NumProcessorGroups > 1)
        {
          // A single ProcessorAffinity mask cannot span processor groups (>64 logical
          // processors); skip rather than pin to an arbitrary subset.
          Console.WriteLine($"Note: {WindowsHardware.NumProcessorGroups} processor groups detected; "
                          + "skipping NUMA affinity (group-aware pinning not yet implemented).");
          return;
        }

        if (!WindowsHardware.TryGetNumaNodeProcessorMask(numaNode, out ulong mask))
        {
          Console.WriteLine($"Note: could not determine processor mask for NUMA node {numaNode}; "
                          + "leaving processor affinity unchanged.");
          return;
        }

        // Pin to every logical processor on the requested NUMA node.
        Process.GetCurrentProcess().ProcessorAffinity = unchecked((nint)mask);
        MaxAvailableProcessors = BitOperations.PopCount(mask);
      }
      catch (Exception exc)
      {
        // Recover gracefully; affinity is an optimization, not a correctness requirement.
        Console.WriteLine($"Note: failure in AffinitizeSingleNUMANode for node {numaNode}: {exc.Message}");
      }
    }
  }

}

