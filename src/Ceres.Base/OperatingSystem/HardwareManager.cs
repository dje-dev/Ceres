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
    /// Maximum number of processors which are active for this process.
    /// </summary>
    public static int MaxAvailableProcessors { private set; get; } = System.Environment.ProcessorCount;

    public static void Initialize(bool affinitizeSingleProcessor)
    {
      if (affinitizeSingleProcessor) AffinitizeSingleProcessor();
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
      else if (isWindows && System.Environment.OSVersion.Version.Major < 7)
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
      Console.WriteLine("NumCPUSockets: " + WindowsHardware.NumCPUSockets);
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
          return (long) Win32.MemorySize;
        }
      }
    }


    private static void AffinitizeSingleProcessor()
    {
      try
      {
        // In non-NUMA situations better to limit number of processors, 
        // this sometimes dramatically improves performance.

        // TODO: Someday it would be desirable to allow different
        //       logical calculations (e.g. chess tree searches)
        //       to be placed on distinct sockets, and not restrict
        //       all computation to a single socket.
        //       But then we'd have to abandon use of TPL and .NET thread pool
        //       completely and use our own versions which were processor constrained.

        // Use at most 16 processors, more are typically not needed/helpful
        const int MAX_PROCESSORS_TO_USE = 16;
        if (System.Environment.ProcessorCount > MAX_PROCESSORS_TO_USE)
        {
          MaxAvailableProcessors = MAX_PROCESSORS_TO_USE;
          int mask = (1 << MAX_PROCESSORS_TO_USE) - 1;

          Process.GetCurrentProcess().ProcessorAffinity = (IntPtr)mask;
          haveAffinitizedSingle = true;
        }
      }
      catch (Exception exc)
      {
        // Therefore recover gracefully if failed for some reason.
        Console.WriteLine("Note: failure in call to AffinitizeSingleProcessor.");
      }

    }


  }
}

