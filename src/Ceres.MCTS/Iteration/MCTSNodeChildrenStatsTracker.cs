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
using Ceres.Base.DataTypes;

#endregion

namespace Ceres.MCTS.Iteration
{
  /// <summary>
  /// Data structure which tracks running statistics 
  /// relating to the the visits to it's children, specifically:
  ///   - the average fraction of visits going to each child, and
  ///   - the average V update being applied to Q
  /// </summary>
  public class MCTSNodeChildrenStatsTracker
  {
    /// <summary>
    /// The detailed specific history of visits is tracked
    /// if HISTORY_LENGTH > 0.
    /// </summary>
    const int HISTORY_LENGTH = 0;

    /// <summary>
    /// The explnential decay factor applied (per batch) when
    /// updating statistics related to visit counts.
    /// </summary>
    const float COUNTS_FRACTION_MOST_RECENT = 0.10f;

    /// <summary>
    /// The explnential decay factor applied (per visit) when
    /// updating statistics related to visit V values.
    /// </summary>
    const float V_FRACTION_MOST_RECENT = 0.005f;


    QueueFixedSize<short[]> history = HISTORY_LENGTH > 0 ? new(HISTORY_LENGTH) : null;

    /// <summary>
    /// Running average of visit frations.
    /// </summary>
    public float[] RunningFractionVisits;

    /// <summary>
    /// Running average of V values applied to update Q.
    /// </summary>
    public float[] RunningVValues;
    

    /// <summary>
    /// Updates internal statistics related to Q.
    /// </summary>
    /// <param name="index"></param>
    /// <param name="vValue"></param>
    /// <param name="numVisits"></param>
    internal void UpdateQValue(int index, float vValue, int numVisits)
    {
      if (RunningVValues == null)
      {
        RunningVValues = new float[64];
      }

      for (int i=0; i<numVisits;i++)
      {
        if (RunningVValues[index] == 0)
        {
          RunningVValues[index] = vValue;
        }
        else
        {
          RunningVValues[index] = (vValue * V_FRACTION_MOST_RECENT)
                                 + RunningVValues[index] * (1.0f - V_FRACTION_MOST_RECENT);
        }

      }
    }


    /// <summary>
    /// Updates internal statistics related to visit counts.
    /// </summary>
    /// <param name="visits"></param>
    /// <param name="numChildrenEligible"></param>
    /// <param name="numVisits"></param>
    internal void UpdateVisitCounts(Span<short> visits, int numChildrenEligible, int numVisits)
    {
      if (RunningFractionVisits == null)
      {
        RunningFractionVisits = new float[64];
      }

      // Put this in UpdateTopNodeInfo ??
      short[] historyValues = HISTORY_LENGTH > 0 ? new short[64] : null;

      for (int i=0; i<visits.Length; i++)
      {
        if (i < numChildrenEligible)
        {
          if (historyValues != null)
          {
            historyValues[i] = visits[i];
          }

          RunningFractionVisits[i] = ((float)visits[i] / numVisits) * COUNTS_FRACTION_MOST_RECENT
                             + RunningFractionVisits[i] * (1.0f - COUNTS_FRACTION_MOST_RECENT);
        }
        else
        {
          // Average in zero
          RunningFractionVisits[i] = RunningFractionVisits[i] * (1.0f - COUNTS_FRACTION_MOST_RECENT);
        }
      }

      history?.Enqueue(historyValues);
    }

  }
}
