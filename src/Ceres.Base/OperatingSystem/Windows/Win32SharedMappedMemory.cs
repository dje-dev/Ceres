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
  /// Interop with Windows memory mapped files.
  /// </summary>
  [SupportedOSPlatform("windows")]
  public class Win32SharedMappedMemory : IDisposable
  {
    enum FileProtection : uint
    {
      ReadOnly = 2,
      ReadWrite = 4,

      LargePages = 0x8000000,

      ReadWriteLargePages = ReadWrite | LargePages
    }

    enum FileRights : uint
    {
      Read = 4,
      Write = 2,
      LargePages = 0x20000000,

      ReadWrite = Read | Write,
      ReadWriteLargePages = ReadWrite | LargePages
    }

    static readonly IntPtr NullHandle = new IntPtr(-1);


    // length - basically length will be int (for 32-bit) and long (for 64-bit)
    [DllImport("kernel32.dll", EntryPoint = "RtlMoveMemory", SetLastError = false)]
    public static extern void MoveMemory(IntPtr destination, IntPtr source, IntPtr length);

    // dwSize - basically size will be int (for 32-bit) and long (for 64-bit)
    [DllImport("kernel32.dll", EntryPoint = "RtlFillMemory", SetLastError = false)]
    public static extern void FillMemory(IntPtr lpAddress, IntPtr dwSize, byte fill);

    // dwSize - basically size will be int (for 32-bit) and long (for 64-bit)
    [DllImport("kernel32.dll", EntryPoint = "RtlZeroMemory", SetLastError = false)]
    public static extern void ZeroMemory(IntPtr lpAddress, IntPtr dwSize);


    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr CreateFileMapping(IntPtr hFile,
                                            int secAttributes, uint allocflags,
                                            uint dwMaximumSizeHigh, uint dwMaximumSizeLow,
                                            string lpName);

    [DllImport("kernel32.dll", SetLastError = true)] static extern IntPtr OpenFileMapping(uint allocFlags, bool bInheritHandle, string lpName);

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr MapViewOfFile(IntPtr hFileMappingObject,
                                        int dwDesiredAccess,
                                        uint dwFileOffsetHigh, uint dwFileOffsetLow,
                                        uint dwNumberOfBytesToMap);
    [DllImport("Kernel32.dll")] static extern bool UnmapViewOfFile(IntPtr map);
    [DllImport("Kernel32.dll")] static extern int GetLargePageMinimum();
    [DllImport("kernel32.dll")] static extern int CloseHandle(IntPtr hObject);


    IntPtr fileHandle;
    IntPtr fileMap;

    public static int LargePageSize => GetLargePageMinimum();
    public IntPtr MemoryStartPtr => fileMap;


    public Win32SharedMappedMemory(string name, bool existing, uint sizeInBytes, bool largePages)
    {
      if (largePages)
      {
        ulong modulo = sizeInBytes % (ulong)GetLargePageMinimum();
        if (modulo != 0)
          throw new Exception("Must be a mulitple of large page minimum when allocating large pages");

        Win32AcquirePrivilege.VerifyCouldAcquireSeLockPrivilege();
      }

      if (existing)
      {
        const uint FLAGS_READ_WRITE = 6;
        fileHandle = OpenFileMapping(FLAGS_READ_WRITE, false, name);
      }
      else
      {
        fileHandle = CreateFileMapping(NullHandle, 0, (uint)(largePages ? FileProtection.ReadWriteLargePages
                                                                        : FileProtection.ReadWrite), 0, sizeInBytes, null);
      }

      if (fileHandle == IntPtr.Zero)
        throw new Exception("OpenFileMapping/CreateFileMapping error: " + Marshal.GetLastWin32Error());

      fileMap = MapViewOfFile(fileHandle, (int)FileRights.ReadWrite, 0, 0, sizeInBytes);

      if (fileMap == IntPtr.Zero)
      {
        string extraInfo = "";
        if (existing) 
          extraInfo = ", could not attach to existing memory under name " + name;
        else if (largePages)
          extraInfo = ", possibly could not create using large pages due to operating system memory fragmentation.";

        float sizeInMB = (float)sizeInBytes / (1024 * 1024);
        throw new Exception($"Memory allocation failure (attempting {sizeInMB,6:F3} megabytes). MapViewOfFile error { Marshal.GetLastWin32Error() } { extraInfo }");
      }
    }

    public void Dispose()
    {
      if (fileMap != IntPtr.Zero) UnmapViewOfFile(fileMap);
      if (fileHandle != IntPtr.Zero) CloseHandle(fileHandle);
      fileMap = fileHandle = IntPtr.Zero;
    }
  }

}