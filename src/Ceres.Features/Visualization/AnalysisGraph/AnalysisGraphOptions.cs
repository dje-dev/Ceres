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

#endregion

namespace Ceres.Features.Visualization.AnalysisGraph
{
  /// <summary>
  /// Set of option controlling generation and appearance of analysis graphs.
  /// </summary>
  public record AnalysisGraphOptions
  {
    const int DEFAULT_DETAIL_LEVEL = 4;

    public int DetailLevel { init; get; } = DEFAULT_DETAIL_LEVEL;
    public bool ShowTranspositions { init; get; } = false;

    /// <summary>
    /// Parses options from specified string.
    /// </summary>
    /// <param name="options"></param>
    /// <returns></returns>
    public static AnalysisGraphOptions FromString(string options)
    {
      int detailLevel = DEFAULT_DETAIL_LEVEL;
      if (options != null)
      {
        // Check for transpositions flag.
        bool showTranspositions = options.ToUpper().Contains("T");

        // Check for detail analysis level flag (digit 0 thru 9).
        foreach (char c in options)
        {
          if (char.IsDigit(c))
          {
            detailLevel = int.Parse(c.ToString());
          }
        }

        AnalysisGraphOptions optionsObj = new AnalysisGraphOptions() 
        { 
          DetailLevel = detailLevel, 
          ShowTranspositions = showTranspositions 
        };
        return optionsObj;
      }
      else
      {
        return new AnalysisGraphOptions();
      }
    }
  }
}