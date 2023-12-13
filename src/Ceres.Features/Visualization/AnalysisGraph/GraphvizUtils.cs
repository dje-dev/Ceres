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
using System.IO;
using System.Text;

using Ceres.Base.Math;
using Ceres.Base.Misc;
using Ceres.Chess;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Features.Visualization.AnalysisGraph
{
  /// <summary>
  /// Collection of miscellaneous static helper methods relating to Graphviz and SVG.
  /// </summary>
  public static class GraphvizUtils
  {
    public static string Quoted(string str) => "\"" + str + "\"";
    public static string Bracketed(string str) => "{" + str + "}";

    static bool haveWarnedNoConfigEntry = false;


    /// <summary>
    /// Returns full path to DOT execuctable (part of graphviz that converts .DOT to .SVG).
    /// </summary>
    public static string DOT_EXE
    {
      get
      {
        string graphvizBinariesDir = CeresUserSettingsManager.Settings.DirGraphvizBinaries;
        if (graphvizBinariesDir == null)
        {
          if (!haveWarnedNoConfigEntry)
          {
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "WARNING: DirGraphvizBinaries is not set in CeresUserSettings.json." 
                                                             + "Assuming GraphViz has been installed and dot.exe is on the system path.");
            haveWarnedNoConfigEntry = true;
          }
          return "dot.exe"; // assume will be found because it is on the system path
        }
        else if (!Directory.Exists(graphvizBinariesDir))
        {
          throw new Exception("Directory referenced by DirGraphvizBinaries does not exist: " + graphvizBinariesDir);
        }
        else
        {
          string exePath1 = Path.Combine(graphvizBinariesDir, "DOT");
          string exePath2 = Path.Combine(graphvizBinariesDir, "DOT.EXE");
          if (!File.Exists(exePath1) && !File.Exists(exePath2))
          {
            throw new Exception("The DOT executable does not exist in directory referenced by DirGraphvizBinaries does not exist: " + graphvizBinariesDir);
          }
          else
          {
            return exePath1;
          }
        }
      }
    }

    /// <summary>
    /// Creates a unique empty temporary directory.
    /// </summary>
    /// <returns>
    /// Directory path.
    /// </returns>
    public static string CreateUniqueTempDirectory(string prefix)
    {
      var uniqueTempDir = Path.GetFullPath(Path.Combine(Path.GetTempPath(), prefix + "_" + Path.GetRandomFileName()));
      Directory.CreateDirectory(uniqueTempDir);
      return uniqueTempDir;
    }

    public static string ConvertedCRLF(string str) => str.ToString().Replace("\r\n", "&#13;&#10;");


    /// <summary>
    /// Palette of 10 colors used to represent evaluations (from bad to good, red to green).
    /// </summary>
    static readonly string[] colorsRDYlGn = new[]
    {
      "#a50026",
      "#d73027",
      "#f46d43",
      "#fdae61",
      "#fee08b",
      "#ffffbf",
      "#d9ef8b",
      "#a6d96a",
      "#66bd63",
      "#1a9850",
      "#006837"
    };

    /// <summary>
    /// Returns hexadecimal string code corresponding to a color
    /// appropriate for showing a position evaluation score.
    /// </summary>
    /// <param name="value"></param>
    /// <returns></returns>
    public static string ColorStr(float value)
    {
      value = 0.5f + (value * 0.5f);
      value = 10 * value;
      value = StatUtils.Bounded((int)value, 0, 10);
      return colorsRDYlGn[(int)value];
    }


    /// <summary>
    /// Writes SVG image representation of specified position
    /// to a file in a specified directory.
    /// </summary>
    /// <param name="pos"></param>
    /// <param name="hash"></param>
    /// <param name="index"></param>
    /// <param name="move"></param>
    /// <param name="tempDir"></param>
    /// <returns></returns>
    public static string WritePosToSVGFile(in Position pos, ulong hash, int index, Move move, string tempDir)
    {
      string posSVG = PositionToSVG.PosSVGString(pos, move);
      string posFN = Path.Combine(tempDir, "_pos" + hash + "_" + index + ".svg");
      File.Delete(posFN);
      File.WriteAllText(posFN, posSVG);
      return posFN;
    }


    public static string WritePositionsToSVGFile(bool showFromPerspectiveOfPlayerOnMove = true, params (Position pos, string label)[] positions)
    {
      string tempDir = Directory.CreateTempSubdirectory().FullName;

     const string HEADER_CORE =
@"<svg width=!500! height=!500! version = !1.1! id=!Capa_1! xmlns=!http://www.w3.org/2000/svg! xmlns:xlink=!http://www.w3.org/1999/xlink! x=!0px! y=!0px!
	 viewBox=!0 0 400 400! style=!enable-background:new 0 0 400 400;! xml:space=!preserve!>
";

    string svg = @"<html>
<body>
";

      foreach ((Position pos, string label) pos in positions)
      {
        svg += HEADER_CORE;// PositionToSVG.HEADER_CORE;
        bool shouldReverse = showFromPerspectiveOfPlayerOnMove &&  !positions[0].pos.IsWhite;
        svg += PositionToSVG.PosSVGString(shouldReverse ? pos.pos.Reversed : pos.pos, default, false);
        svg += PositionToSVG.FOOTER;
      }

      svg = svg.Replace("!", "\"");
      svg+= "</body></html>";
//      svg = svg.Replace("0 0 400 400", "0 0 1900 1900");

      string posFN = Path.Combine(tempDir, "_pos" + 0 + ".svg");
      File.Delete(posFN);
      File.WriteAllText(posFN, svg);
      Console.WriteLine(posFN);
      return posFN;
    }


    /// <summary>
    /// Converts text containing DOT graphviz language specification
    /// into an SVG representation using the DOT executable (part of graphviz).
    /// </summary>
    /// <param name="dot"></param>
    /// <param name="BASE_FN"></param>
    /// <param name="workDirectory"></param>
    /// <exception cref="Exception"></exception>
    public static void Convert(string dot, string BASE_FN, string workDirectory)
    {
      string FN_DOT = Path.Combine(workDirectory, $"{BASE_FN}.dot");
      string FN_SVG = Path.Combine(workDirectory, $"{BASE_FN}.svg");

      string dotFN = Path.Combine(workDirectory, FN_DOT);
      File.Delete(dotFN);
      File.WriteAllText(dotFN, dot);

      ProcessStartInfo startInfo = new ProcessStartInfo();

      string args = $"-Tsvg \"{FN_DOT}\" -o \"{FN_SVG}\"";
      startInfo.FileName = GraphvizUtils.DOT_EXE;
      startInfo.WorkingDirectory = workDirectory;
      startInfo.Arguments = args;
      startInfo.UseShellExecute = false;

      // turning this on can be useful for debugging
      // startInfo.CreateNoWindow = false;

      Process gotProcess = Process.Start(startInfo);
      gotProcess.WaitForExit();
      int exitCode = gotProcess.ExitCode;
      if (exitCode != 0)
      {
        Console.WriteLine(dot);
        throw new Exception($"Error in SVG convert {startInfo.FileName} {startInfo.WorkingDirectory} {startInfo.ArgumentList} Exit code " + gotProcess.ExitCode);
      }
    }

    public static void WriteArrowheadForFractionParent(StringBuilder sb, float nFractionParent)
    {
      if (nFractionParent > 0.5)
      {
        sb.Append($" arrowsize=5");
      }
      if (nFractionParent > 0.30)
      {
        sb.Append($" arrowsize=2.5");
      }
      else if (nFractionParent < 0.07)
      {
        sb.Append($" arrowsize=0.35");
      }
      else if (nFractionParent < 0.15)
      {
        sb.Append($" arrowsize=0.5");
      }
    }

  }
}
