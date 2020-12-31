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
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

#endregion

namespace Ceres.Base.OperatingSystem.Windows
{
  /// <summary>
  /// Interop with general Windows hardware information APIs.
  /// </summary>
  [SupportedOSPlatform("windows")]
  internal static class Win32SystemInfo
  {
    public const int MEM_LARGE_PAGES = 0x20000000; // flag to request from Windows


    [DllImport("kernel32.dll", SetLastError = true)]
    internal static extern void GetSystemInfo(ref SYSTEM_INFO Info);

    [StructLayout(LayoutKind.Sequential)]
    internal struct SYSTEM_INFO
    {
      internal ushort wProcessorArchitecture;
      internal ushort wReserved;
      internal uint dwPageSize;
      internal IntPtr lpMinimumApplicationAddress;
      internal IntPtr lpMaximumApplicationAddress;
      internal IntPtr dwActiveProcessorMask;
      internal uint dwNumberOfProcessors;
      internal uint dwProcessorType;
      internal uint dwAllocationGranularity;
      internal ushort wProcessorLevel;
      internal ushort wProcessorRevision;
    }

  }
}