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

    public static int NumAffinitizedThreadsInProcess => System.Environment.ProcessorCount / (haveAffinitizedSingle ? 2 : 1);

    public static void Initialize(bool affinitizeSingleProcessor)
    {
      if (affinitizeSingleProcessor) AffinitizeSingleProcessor();
    }

    public static void VerifyHardwareSoftwareCompatability()
    {
      string errorString = null;
      if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
        && !RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
      {
        errorString = "Currently only Windows or Linux operating systems is supported.";
      }
      else if (System.Environment.OSVersion.Version.Major < 10)
      {
        errorString = "Windows Version 10 or above required.";
      }
      else if (!Avx2.IsSupported)
      {
        errorString = "AVX2 hardware support is required but not available on this processor.";
      }
      else if (!SoftwareManager.IsCUDAInstalled)
      {
        errorString = "GPU hardware with CUDA installation is required but not found.";
      }

      if (errorString != null)
      {
        ConsoleColor priorColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine($"Fatal Error. {errorString}");
        Console.WriteLine();
        Console.ForegroundColor = priorColor;
        System.Environment.Exit(-1);
      }
    }

    static bool haveAffinitizedSingle = false;

    private static void AffinitizeSingleProcessor()
    {
      // NOTE: Someday it would be desirable to allow different
      //       logical calculations (e.g. chess tree searches)
      //       to be placed on distinct sockets, and not restrict
      //       all computation to a single socket.
      //       But then we'd have to abandon use of TPL and .NET thread pool
      //       completely and use our own versions which were processor constrained.
      bool isMultisocket = WindowsHardware.NumCPUSockets > 1;
      if (isMultisocket)
      {
        // This dramatically improves performance (multithreading)
        Process Proc = Process.GetCurrentProcess();
        long AffinityMask = (long)Proc.ProcessorAffinity;
        // TODO: need to calculate correct mask. Currently below we only take half the bits (processors)
        var mask = (1 << (System.Environment.ProcessorCount / 2)) - 1;
        AffinityMask &= mask; 
        Proc.ProcessorAffinity = (IntPtr)AffinityMask;

        haveAffinitizedSingle = true;
      }
    }

  }
}

