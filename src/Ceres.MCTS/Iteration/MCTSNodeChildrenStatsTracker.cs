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
using Ceres.MCTS.MTCSNodes;

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
    /// The N of root node at time of last visit.
    /// </summary>
    public int[] LastRootN;


    #region Exponential decay scaling calculation

    const float LN_2 = 0.6931472f; // MathF.Log(2)
    static float LambdaForHalflife(float halflife) => LN_2 / halflife;

    // Calibrate exponential decay such that the EMWA value
    // has a halflife approximately equal to 5% of the current search length.
    const float HALFLIFE_SEARCH_FRAC = 0.10f;

    /// <summary>
    /// Computes the exponential decay factor applied (per visit) when
    /// updating statistics related to visit V values.
    /// </summary>
    /// <param name="N"></param>
    /// <returns></returns>
    public static float LambdaForN(int N) => LambdaForHalflife(N * HALFLIFE_SEARCH_FRAC);

    #endregion

    const int NUM_TRACK = 64;

    /// <summary>
    /// Updates internal statistics related to Q.
    /// </summary>
    /// <param name="rootN"></param>
    /// <param name="thisN"></param>
    /// <param name="index"></param>
    /// <param name="vValue"></param>
    /// <param name="numVisits"></param>
    internal void UpdateQValue(int rootN, int thisN, int index, float vValue, int numVisits)
    {
      if (RunningVValues == null)
      {
        RunningVValues = new float[NUM_TRACK];
        LastRootN = new int[NUM_TRACK];
      }

      LastRootN[index] = rootN;

      for (int i=0; i<numVisits;i++)
      {
        if (RunningVValues[index] == 0)
        {
          RunningVValues[index] = vValue;
        }
        else
        {
          float lambda = LambdaForN(thisN);
          RunningVValues[index] = (vValue * lambda)
                                 + RunningVValues[index] * (1.0f - lambda);
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


    public float[] Resamples;

    int resampleLastUpdateRootN = 0;

    /// <summary>
    /// Updates the resampled Q values at root if they are not current.
    /// </summary>
    /// <param name="root"></param>
    /// <exception cref="Exception"></exception>
    public void CheckUpdateResamples(MCTSNode root)
    {
      if (!root.IsRoot)
      {
        throw new Exception("Internal error; expected root");
      }

      if (Resamples == null)
      {
        Resamples = new float[64];
      }

      if (resampleLastUpdateRootN < root.N)
      {
        for (int i=0;i<root.NumChildrenExpanded;i++)
        {
          MCTSNode child = root.ChildAtIndex(i);
          if (child.N > 1000)
          {
            const int NUM_SAMPLES = 1024;
            float temp = root.Context.ParamsSearch.ResamplingMoveSelectionTemperature;
            float fracResample = root.Context.ParamsSearch.ResamplingMoveSelectionFractionMove;
            (float avg, float sd) = MCTSNodeResampling.GetResampledStats(child, NUM_SAMPLES, 1, temp);
            float qAvg = fracResample * avg + (1.0f - fracResample) * (float)child.Q;
            if (MathF.Abs((float)child.Q-avg) >  0.15f)
            {
              //  Console.WriteLine("resample big diff " + child.N + " " + child.Q + " " + avg);
            }
            Resamples[i] = qAvg;
          }
        }

        resampleLastUpdateRootN = root.N;
      }
    }
  }
}
s