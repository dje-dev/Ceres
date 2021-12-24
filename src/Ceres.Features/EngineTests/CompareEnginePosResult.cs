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
  /// <summary>
  /// Information about a single position for which engines did not agree.
  /// </summary>
  /// <param name="GPUID"></param>
  /// <param name="CountScore"></param>
  /// <param name="FracDifferent"></param>
  /// <param name="Time1Secs"></param>
  /// <param name="Time2Secs"></param>
  /// <param name="N1"></param>
  /// <param name="N2"></param>
  /// <param name="CountMuchBetter"></param>
  /// <param name="CountMuchWorse"></param>
  /// <param name="ScoreBestMove1"></param>
  /// <param name="Diff1From2"></param>
  /// <param name="SFAgrees"></param>
  /// <param name="Move1"></param>
  /// <param name="Move2"></param>
  /// <param name="MoveSF"></param>
  /// <param name="FEN"></param>
  public record CompareEnginePosResult(int GPUID, int CountScore, float FracDifferent,
                                       float Time1Secs, float Time2Secs,
                                       int N1, int N2,
                                       int CountMuchBetter, int CountMuchWorse,
                                       float ScoreBestMove1, float Diff1From2,
                                       bool SFAgrees,
                                       string Move1, string Move2, string MoveSF, string FEN)
  {

  }
}
