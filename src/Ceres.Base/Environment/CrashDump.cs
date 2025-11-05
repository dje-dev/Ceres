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

using System.Diagnostics;
using Microsoft.Diagnostics.NETCore.Client;

#endregion

namespace Ceres.Base.Environment;

public static class CrashDump
{
  /// <summary>
  /// Writes a process dump of the current process. Cross-platform.
  /// On Windows: .dmp (minidump). On Linux/macOS: native core file.
  /// </summary>
  public static void WriteDump(string outputPath,
                               DumpType dumpType = DumpType.Normal,
                               bool logProgress = false)
  {
    DiagnosticsClient client = new DiagnosticsClient(Process.GetCurrentProcess().Id);
    client.WriteDump(dumpType, outputPath, logDumpGeneration: logProgress);
  }
}
