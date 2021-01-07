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
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Runtime.Versioning;

#endregion

namespace Ceres.Base.OperatingSystem.Windows
{
  /// <summary>
  /// Static methods for interop with Windows API.
  /// </summary>
  [SupportedOSPlatform("windows")]
  public static class Win32
  {

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Auto)]
    private class MEMORYSTATUSEX
    {
      public uint dwLength;
      public uint dwMemoryLoad;
      public ulong ullTotalPhys;
      public ulong ullAvailPhys;
      public ulong ullTotalPageFile;
      public ulong ullAvailPageFile;
      public ulong ullTotalVirtual;
      public ulong ullAvailVirtual;
      public ulong ullAvailExtendedVirtual;

      public MEMORYSTATUSEX()
      {
        dwLength = (uint)Marshal.SizeOf(typeof(MEMORYSTATUSEX));
      }
    }


    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GlobalMemoryStatusEx([In, Out] MEMORYSTATUSEX lpBuffer);


    /// <summary>
    /// Returns physical memory size.
    /// </summary>
    [SupportedOSPlatform("windows")]
    public static ulong MemorySize
    {
      get
      {
        MEMORYSTATUSEX status = new MEMORYSTATUSEX();
        GlobalMemoryStatusEx(status);
        return status.ullTotalPhys;
      }
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr VirtualAlloc(IntPtr lpAddress, IntPtr dwSize, int flAllocationType, int flProtect);

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern IntPtr VirtualAllocExNuma(IntPtr hProcess, IntPtr lpAddress, IntPtr dwSize, int flAllocationType, int flProtect, int nndPreferred);

    public const int MEM_COMMIT = 0x1000;
    public const int MEM_RESERVE = 0x2000;

    public const int PAGE_NOACCESS = 0x01;
    public const int PAGE_READONLY = 0x02;
    public const int PAGE_READWRITE = 0x04;

    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern bool VirtualFree(IntPtr lpAddress, IntPtr dwSize, int dwFreeType);

    public const int MEM_DECOMMIT = 0x4000;
    public const int MEM_RELEASE = 0x8000;


    // length - basically length will be int (for 32-bit) and long (for 64-bit)
    [DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory", SetLastError = false)]
    public static extern void MoveMemory(IntPtr destination, IntPtr source, IntPtr length);

    // dwSize - basically size will be int (for 32-bit) and long (for 64-bit)
    [DllImport("kernel32.dll", EntryPoint = "RtlFillMemory", SetLastError = false)]
    public static extern void FillMemory(IntPtr lpAddress, IntPtr dwSize, byte fill);

    // dwSize - basically size will be int (for 32-bit) and long (for 64-bit)
    [DllImport("kernel32.dll", EntryPoint = "RtlZeroMemory", SetLastError = false)]
    public static extern void ZeroMemory(IntPtr lpAddress, IntPtr dwSize);

    public static void WriteCrashdumpFile(string dumpFileName)
    {
      using (FileStream crashDump = File.Create(dumpFileName))
      {
        bool success = MiniDumpWriteDump(
          Process.GetCurrentProcess().Handle,
          (uint)Process.GetCurrentProcess().Id,
          crashDump.SafeFileHandle,
          MINIDUMP_TYPE.MiniDumpWithFullMemory,
          IntPtr.Zero,
          IntPtr.Zero,
          IntPtr.Zero);
        if (!success)
          throw new Exception($"Minidump failed {new Win32Exception(Marshal.GetLastWin32Error()).ToString()}");
      }

    }

    [DllImport("DbgHelp.dll", SetLastError = true)]
    public extern static bool MiniDumpWriteDump(
        IntPtr hProcess,
        UInt32 ProcessId,
        SafeHandle hFile,
        MINIDUMP_TYPE DumpType,
        IntPtr ExceptionParam,
        IntPtr UserStreamParam,
        IntPtr CallbackParam);

    public enum MINIDUMP_TYPE
    {
      MiniDumpNormal = 0x00000000,
      MiniDumpWithDataSegs = 0x00000001,
      MiniDumpWithFullMemory = 0x00000002,
      MiniDumpWithHandleData = 0x00000004,
      MiniDumpFilterMemory = 0x00000008,
      MiniDumpScanMemory = 0x00000010,
      MiniDumpWithUnloadedModules = 0x00000020,
      MiniDumpWithIndirectlyReferencedMemory = 0x00000040,
      MiniDumpFilterModulePaths = 0x00000080,
      MiniDumpWithProcessThreadData = 0x00000100,
      MiniDumpWithPrivateReadWriteMemory = 0x00000200,
      MiniDumpWithoutOptionalData = 0x00000400,
      MiniDumpWithFullMemoryInfo = 0x00000800,
      MiniDumpWithThreadInfo = 0x00001000,
      MiniDumpWithCodeSegs = 0x00002000,
      MiniDumpWithoutManagedState = 0x00004000,
    }

  }


}
