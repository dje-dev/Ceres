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

using System.Collections.Generic;

#endregion


namespace Ceres.Features.Suites
{
  public class SuiteTestResult
  {
    public readonly SuiteTestDef Def;

    public SuiteTestResult(SuiteTestDef def)
    {
      Def = def;
    }

    public float AvgScoreLC0;

    public float AvgScore1;
    public float AvgScore2;

    public float AvgWScore1;
    public float AvgWScore2;

    public float TotalRuntimeLC0;
    public float TotalRuntime1;
    public float TotalRuntime2;

    public float TotalNodesLC0;
    public float TotalNodes1;
    public float TotalNodes2;

    public float[] FinalQ1;
    public float[] FinalQ2;

    public string SummaryLine
    {
      get
      {
        //return $"{Def.TestDescription,15}, {Def.SuiteID,35} {Def.MaxNumPositions} {Def.SearchLimitCeres1,25}  "
        //     + $"{TotalRuntimeLC0,7:F2} {TotalRuntime1,7:F2} {TotalRuntime2,7:F2}   "
        //     + $"{AvgScoreLC0,6:F2} {AvgScore1,6:F2} {AvgScore2,6:F2}";
        return null;
      }
    }

  }
}

