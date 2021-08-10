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
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Security.Policy;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem.Linux;
using Ceres.Base.OperatingSystem.Windows;
using Microsoft.Extensions.Logging;

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Static helper methods for interfacing with the hardware.
  /// </summary>
  public static class HardwareManager
  {
    /// <summary>
    /// Maximum expected pages size across all supported OS.
    /// </summary>
    public const int PAGE_SIZE_MAX = 2048 * 1024;

    /// <summary>
    /// Maximum number of processors which are active for this process.
    /// </summary>
    public static int MaxAvailableProcessors { private set; get; } = System.Environment.ProcessorCount;

    public static void Initialize(int? numaNode)
    {
      if (numaNode.HasValue)
      {
        AffinitizeSingleNUMANode(numaNode.Value);
      }
    }


    public static void VerifyHardwareSoftwareCompatability()
    {
      string errorString = null;
      bool isWindows = RuntimeInformation.IsOSPlatform(OSPlatform.Windows);
      bool isLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux);
      if (!isWindows && !isLinux)
      {
        errorString = "Currently only Windows or Linux operating systems is supported.";
      }
      else if (isWindows && System.Environment.OSVersion.Version.Major < 6) // Note that Windows 7 is version 6
      {
        errorString = "Windows Version 7 or above required.";
      }
      else if (!Avx.IsSupported)
      {
        errorString = "AVX hardware support is required but not available on this processor.";
      }

      if (errorString != null)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Red, $"Fatal Error. {errorString}");
        System.Environment.Exit(-1);
      }
    }

    static bool haveAffinitizedSingle = false;


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
    public static long MemorySize
    {
      get
      {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
          return LinuxAPI.PhysicalMemorySize;
        }
        else
        {
          return (long)Win32.MemorySize;
        }
      }
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
    public static void AffinitizeSingleNUMANode(int numaNode)
    {
      try
      {
        // In non-NUMA situations better to limit number of processors, 
        // this sometimes dramatically improves performance.

        // The coreinfo.exe from sysinternals is useful for dumping topology info under Windows.

        // TODO: Someday it would be desirable to allow different
        //       logical calculations (e.g. chess tree searches)
        //       to be placed on distinct sockets, and not restrict
        //       all computation to a single socket.
        //       But then we'd have to abandon use of TPL and .NET thread pool
        //       completely and use our own versions which were processor constrained.

        int maxProcessors;

        // Default assumption to use all processors.
        int numProcessors = System.Environment.ProcessorCount;

        // TODO: extend this logic to also cover Linux.
        bool isKnownMultisocket = SoftwareManager.IsWindows && WindowsHardware.NumCPUSockets > 1;
        if (isKnownMultisocket)
        {
          // Only use a single socket to improve affinity.
          // CPUs in multisocket configurations are almost certainly at least 16 processors
          // and any possible small benefit from going over 16 would almost certainly be
          // overwhelemed by the increased latency.
          maxProcessors = System.Math.Max(1, System.Environment.ProcessorCount / WindowsHardware.NumCPUSockets);
        }
        else
        {
          // Use at most 32 processors, more are almost never going to be helpful.
          maxProcessors = System.Math.Min(32, System.Environment.ProcessorCount);
        }

        if (System.Environment.ProcessorCount > maxProcessors)
        {
          // TODO: Improve the mapping to NUMA nodes.
          //       They assumption here that the processors are consecutive
          //       within a NUMA node is not always correct.
          MaxAvailableProcessors = maxProcessors;
          long mask = ((long)1 << maxProcessors) - 1;

          Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)mask;
          haveAffinitizedSingle = true;
        }
      }
      catch (Exception exc)
      {
        // Therefore recover gracefully if failed for some reason.
        Console.WriteLine("Note: failure in call to AffinitizeSingleNUMANode.");
      }
    }
  }

}

