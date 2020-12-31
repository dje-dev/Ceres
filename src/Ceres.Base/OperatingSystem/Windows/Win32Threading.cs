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
using Microsoft.Win32.SafeHandles;
using System.Runtime.Versioning;

#endregion

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.
// Originating from the FASTER library developed by Microsoft Research

namespace Ceres.Base.OperatingSystem.Windows
{
  /// <summary>
  /// Interop with WINAPI for file I/O, threading, and NUMA functions.
  /// </summary>
  [SupportedOSPlatform("windows")]
  public static unsafe class Win32Threading
  {
    #region Thread and NUMA functions
    [DllImport("kernel32.dll")]
    private static extern IntPtr GetCurrentThread();
    [DllImport("kernel32")]
    internal static extern uint GetCurrentThreadId();
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint GetCurrentProcessorNumber();
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint GetActiveProcessorCount(uint count);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern ushort GetActiveProcessorGroupCount();
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern int SetThreadGroupAffinity(IntPtr hThread, ref GROUP_AFFINITY GroupAffinity, ref GROUP_AFFINITY PreviousGroupAffinity);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern int GetThreadGroupAffinity(IntPtr hThread, ref GROUP_AFFINITY PreviousGroupAffinity);

    private static readonly uint ALL_PROCESSOR_GROUPS = 0xffff;

    [System.Runtime.InteropServices.StructLayoutAttribute(System.Runtime.InteropServices.LayoutKind.Sequential)]
    private struct GROUP_AFFINITY
    {
      public ulong Mask;
      public uint Group;
      public uint Reserved1;
      public uint Reserved2;
      public uint Reserved3;
    }

    //      foreach (ProcessThread p in Process.GetCurrentProcess().Threads)
    //        if (p.Id == GetCurrentThreadId())
    //          p.IdealProcessor = 1;

    /// <summary>
    /// Accepts thread id = 0, 1, 2, ... and sprays them round-robin
    /// across all cores (viewed as a flat space). On NUMA machines,
    /// this gives us [socket, core] ordering of affinitization. That is, 
    /// if there are N cores per socket, then thread indices of 0 to N-1 map
    /// to the range [socket 0, core 0] to [socket 0, core N-1].
    /// </summary>
    /// <param name="threadIdx">Index of thread (from 0 onwards)</param>
    public static void AffinitizeThreadRoundRobin(uint threadIdx)
    {
      uint nrOfProcessors = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
      ushort nrOfProcessorGroups = GetActiveProcessorGroupCount();
      uint nrOfProcsPerGroup = nrOfProcessors / nrOfProcessorGroups;

      GROUP_AFFINITY groupAffinityThread = default(GROUP_AFFINITY);
      GROUP_AFFINITY oldAffinityThread = default(GROUP_AFFINITY);

      IntPtr thread = GetCurrentThread();
      GetThreadGroupAffinity(thread, ref groupAffinityThread);

      threadIdx = threadIdx % nrOfProcessors;

      groupAffinityThread.Mask = (ulong)1L << ((int)(threadIdx % (int)nrOfProcsPerGroup));
      groupAffinityThread.Group = (uint)(threadIdx / nrOfProcsPerGroup);

      if (SetThreadGroupAffinity(thread, ref groupAffinityThread, ref oldAffinityThread) == 0)
      {
        throw new Exception("Unable to affinitize thread");
      }
    }

    /// <summary>
    /// Accepts thread id = 0, 1, 2, ... and sprays them round-robin
    /// across all cores (viewed as a flat space). On NUMA machines,
    /// this gives us [core, socket] ordering of affinitization. That is, 
    /// if there are N cores per socket, then thread indices of 0 to N-1 map
    /// to the range [socket 0, core 0] to [socket N-1, core 0].
    /// </summary>
    /// <param name="threadIdx">Index of thread (from 0 onwards)</param>
    /// <param name="nrOfProcessorGroups">Number of NUMA sockets</param>
    public static void AffinitizeThreadShardedNuma(uint threadIdx, ushort nrOfProcessorGroups)
    {
      uint nrOfProcessors = GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
      uint nrOfProcsPerGroup = nrOfProcessors / nrOfProcessorGroups;

      threadIdx = nrOfProcsPerGroup * (threadIdx % nrOfProcessorGroups) + (threadIdx / nrOfProcessorGroups);
      AffinitizeThreadRoundRobin(threadIdx);
      return;
    }
    #endregion

    #region Advanced file ops
    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool LookupPrivilegeValue(string lpSystemName, string lpName, ref LUID lpLuid);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr GetCurrentProcess();

    [DllImport("advapi32", SetLastError = true)]
    private static extern bool OpenProcessToken(IntPtr ProcessHandle, uint DesiredAccess, out IntPtr TokenHandle);

    [DllImport("advapi32.dll", SetLastError = true)]
    private static extern bool AdjustTokenPrivileges(IntPtr tokenhandle, int disableprivs, ref TOKEN_PRIVILEGES Newstate, int BufferLengthInBytes, int PreviousState, int ReturnLengthInBytes);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr hObject);

    [DllImport("Kernel32.dll", SetLastError = true)]
    private static extern bool DeviceIoControl(SafeFileHandle hDevice, uint IoControlCode, void* InBuffer, int nInBufferSize, IntPtr OutBuffer, int nOutBufferSize, ref uint pBytesReturned, IntPtr Overlapped);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetFilePointerEx(SafeFileHandle hFile, long liDistanceToMove, out long lpNewFilePointer, uint dwMoveMethod);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetFileValidData(SafeFileHandle hFile, long ValidDataLength);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern SafeFileHandle CreateFile(string filename, uint access, uint share, IntPtr securityAttributes, uint creationDisposition, uint flagsAndAttributes, IntPtr templateFile);

    #region Native structs

    [StructLayout(LayoutKind.Sequential)]
    private struct LUID
    {
      public uint lp;
      public int hp;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct LUID_AND_ATTRIBUTES
    {
      public LUID Luid;
      public uint Attributes;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct TOKEN_PRIVILEGES
    {
      public uint PrivilegeCount;
      public LUID_AND_ATTRIBUTES Privileges;
    }

    #endregion

    /// <summary>
    /// Enable privilege for process
    /// </summary>
    /// <returns></returns>
    public static bool EnableProcessPrivileges()
    {
#if DOTNETCORE
            if (!RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return false;
#endif

      TOKEN_PRIVILEGES token_privileges = default(TOKEN_PRIVILEGES);
      token_privileges.PrivilegeCount = 1;
      token_privileges.Privileges.Attributes = 0x2;

      if (!LookupPrivilegeValue(null, "SeManageVolumePrivilege",
          ref token_privileges.Privileges.Luid)) return false;

      if (!OpenProcessToken(GetCurrentProcess(), 0x20, out IntPtr token))
        return false;

      if (!AdjustTokenPrivileges(token, 0, ref token_privileges, 0, 0, 0))
      {
        CloseHandle(token);
        return false;
      }
      if (Marshal.GetLastWin32Error() != 0)
      {
        CloseHandle(token);
        return false;
      }
      CloseHandle(token);
      return true;
    }


    #endregion
  }
}