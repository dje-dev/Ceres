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
  /// Interop with Windows APIs for acess rights management.
  /// </summary>
  [SupportedOSPlatform("windows")]
  public static class Win32AcquirePrivilege
  {

    [DllImport("advapi32.dll", ExactSpelling = true, SetLastError = true)]
    internal static extern bool AdjustTokenPrivileges(IntPtr htok, bool disall, ref TokenPrivilige newst, int len, IntPtr prev, IntPtr relen);

    [DllImport("kernel32.dll", ExactSpelling = true)]
    internal static extern IntPtr GetCurrentProcess();

    [DllImport("advapi32.dll", ExactSpelling = true, SetLastError = true)]
    internal static extern bool OpenProcessToken(IntPtr h, int acc, ref IntPtr phtok);

    [DllImport("advapi32.dll", SetLastError = true)]
    internal static extern bool LookupPrivilegeValue(string host, string name, ref long pluid);

    [StructLayout(LayoutKind.Sequential, Pack = 1)]
    internal struct TokenPrivilige
    {
      public int Count;
      public long LUID;
      public int Attrib;
    }

    internal const int SE_PRIVILEGE_ENABLED = 0x00000002;
    internal const int TOKEN_ADJUST_PRIVILEGES = 0x00000020;
    internal const int TOKEN_QUERY = 0x00000008;

    internal const string LockMemoryPrivilegeName = "SeLockMemoryPrivilege";

    static bool haveAcquiredSeLockPrivilege = false;

    public static void VerifyCouldAcquireSeLockPrivilege()
    {
      if (!haveAcquiredSeLockPrivilege)
      {
        if (!AcquireSeLockMemoryPrivilege())
          throw new Exception("Could not acquire SeLockMemoryPrivilege (is current Windows account configured to allow this?");

        haveAcquiredSeLockPrivilege = true;
      }
    }

    public static bool AcquireSeLockMemoryPrivilege() => AcquirePrivilege("SeLockMemoryPrivilege");

    public static bool AcquirePrivilege(string privilegeName)
    {
      try
      {
        TokenPrivilige tp;
        tp.Count = 1;
        tp.LUID = 0;
        tp.Attrib = SE_PRIVILEGE_ENABLED;

        IntPtr hproc = GetCurrentProcess();

        IntPtr procToken = IntPtr.Zero;
        bool retVal = OpenProcessToken(hproc, TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, ref procToken);
        if (!retVal) throw new Exception($"OpenProcessToken err {retVal}");

        retVal = LookupPrivilegeValue(null, privilegeName, ref tp.LUID);
        if (!retVal) throw new Exception($"LookupPrivilegeValue err {retVal}");

        retVal = AdjustTokenPrivileges(procToken, false, ref tp, 0, IntPtr.Zero, IntPtr.Zero);
        if (!retVal) throw new Exception($"AdjustTokenPrivileges err {retVal}");

        return retVal;
      }
      catch (Exception )
      {
        return false;
      }

    }
  }
}
