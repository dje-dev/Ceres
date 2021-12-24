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


#endregion

namespace Ceres.Features.EngineTests
{
  public record CompareEngineResultSummary(float RuntimeSeconds, float TimeEngine1, float TimeEngine2, int CountScored, int CountDifferentMoves, 
                                           float QDiffAvg, float QDiffSD, float QDiffZ, int CountEngine1MuchBetter, int CountEngine2MuchBetter)
  {

  }
}
