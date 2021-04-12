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

using System.Runtime.InteropServices;

#endregion

/// <summary>
/// Static interop for methods in Linux API.
/// Similar to: https://github.com/tmds/Tmds.LibC/tree/master/src/Sources/linux.common
/// </summary>
namespace Ceres.Base.OperatingSystem.Linux
{
  public unsafe static class LinuxAPI
  {
    public const string LIBC = "libc.so.6";

    public const int PROT_NONE = 0;
    public const int PROT_READ = 1;
    public const int PROT_WRITE = 2;

    public const int MAP_SHARED = 1;
    public const int MAP_PRIVATE = 2;
    public const int MAP_SHARED_VALIDATE = 3;
    public const int MAP_TYPE = 0;
    public const int MAP_FIXED = 0x10;
    public const int MAP_ANONYMOUS = 0x20;
    public const int MAP_NORESERVE = 0x4000;
    public const int MAP_HUGETLB = 0x40000;
    public const int MAP_FILE = 0;

    public const int MREMAP_MAYMOVE = 1;
    public const int MREMAP_FIXED = 2;

    public const int MFD_HUGETLB = 4;


    #region System configuration

    [DllImport(LIBC)] public static extern long sysconf(int name);


    /// <summary>
    /// Returns total amount of physical memory available.
    /// </summary>
    public static long PhysicalMemorySize
    {
      get
      {
        // API not reliable under Linux, so return 1TB.
        // TODO: improve.
        return (long)1024 * (long)1024 * (long)1024 * (long)1024;

        const int _SC_PAGESIZE = 8;
        const int _SC_PHYS_PAGES = 11;
        return sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGESIZE);
      }
    }

    #endregion

    //#define _SC_NPROCESSORS_CONF              9
    //#define _SC_NPROCESSORS_ONLN             10

    [DllImport(LIBC)] public static extern int getpid();

    #region Memory allocation 

    [DllImport(LIBC)] public static extern int munmap(void* addr, long length);

    [DllImport(LIBC)] public static extern int mprotect(void* addr, long len, int prot);

    [DllImport(LIBC)] public static extern int msync(void* addr, long len, int flags);

    [DllImport(LIBC)] public static extern int mlock(void* addr, long len);

    [DllImport(LIBC)] public static extern int mlock2(void* addr, long len, int flags);

    [DllImport(LIBC)] public static extern int munlock(void* addr, long len);

    [DllImport(LIBC)] public static extern int mlockall(int flags);

    [DllImport(LIBC)] public static extern int munlockall();

    [DllImport(LIBC)] public static extern void* mremap(void* old_address, long old_size, long new_size, int flags, void* new_address);

    [DllImport(LIBC)] public static extern void* mmap(void* addr, long len, int prot, int flags, int fildes, long off);

    [DllImport(LIBC)] public static extern int remap_file_pages(void* addr, long size, int prot, long pgoff, int flags);
    [DllImport(LIBC)] public static extern int memfd_create(byte* name, uint flags);
    [DllImport(LIBC)] public static extern int madvise(void* addr, long length, int advice);
    [DllImport(LIBC)] public static extern int mincore(void* addr, long length, byte* vec);

    #endregion

  }

}

