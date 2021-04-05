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
using System.IO;
using System.Runtime.InteropServices;

#endregion

namespace Ceres.Base.OperatingSystem
{
  /// <summary>
  /// Static helper methods for interfacing with the hardware.
  /// </summary>
  public static class SoftwareManager
  {
    //    public enum OSEnvironment {  Windows, Linux };

    /// <summary>
    /// Returns if running under Linux operating system.
    /// </summary>
    public static bool IsLinux => RuntimeInformation.IsOSPlatform(OSPlatform.Linux);


    /// <summary>
    /// Returns if running under WSL2 (Windows subsystem for Linux).
    /// </summary>
    public static bool IsWSL2
    {
      get
      {
        string fn = Path.Combine(Path.DirectorySeparatorChar.ToString(), "proc", "version");
        return IsLinux && File.Exists(fn) && File.ReadAllText(fn).Contains("microsoft");
      }
    }


    /// <summary>
    /// Returns if NVIDIA CUDA library is installed.
    /// </summary>
    public static bool IsCUDAInstalled => LibraryLoaded("NVCUDA.DLL", "libcuda");


    #region Private helpers

    [DllImport("kernel32.dll", SetLastError = true)]
    static extern IntPtr LoadLibrary(string dllToLoad);

    /// <summary>
    /// Returns if the library with a specified file could be successfully loaded.
    /// </summary>
    static bool LibraryLoaded(string libraryNameWindows, string libraryNameLinux)
    {
      if (IsLinux)
        LoadLibrary(libraryNameLinux);
      else
        LoadLibrary(libraryNameWindows);
      return Marshal.GetLastWin32Error() == 0;
    }

    #endregion

  }
}

