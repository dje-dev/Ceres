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

using Ceres.Base.DataType;
using System;

#endregion

namespace Ceres.MCGS.Search.PUCT;

/// <summary>
/// Container for transient set of node statistics (gathered from children).
/// </summary>
internal class GatheredChildStats
{
  const int ALIGNMENT = 64; // For SIMD efficiency

  /// <summary>
  /// If the per-child uncertainty values (UV and UP) are gathered.
  ///
  /// Their only consumer is the value uncertainty adjustment experiment in PUCTSelector, which
  /// is currently disabled. The two are switched together from here so that when the experiment
  /// is off the gather is not paid for (it costs about 1ns per child per node visit, for values
  /// nothing reads), and so that when it is switched back on both buffers are actually filled.
  /// </summary>
  internal const bool GATHER_UNCERTAINTY = false;

  // N.B. If new summary fields are added here, be sure to
  //      also update the ResetSummaryFields method.

  /// <summary>
  /// Total number of visits in flight (from any selector)
  /// across all children.
  /// </summary>
  internal double SumNumInFlightAll;

  /// <summary>
  /// The total of all the policy probabilities across
  /// all children that have been visited (or are currently in flight).
  /// </summary>
  internal double SumPVisited;

  /// <summary>
  /// Sum of N across all children that have been visited.
  /// </summary>
  internal double SumNVisited;

  /// <summary>
  /// Sum of W (Q * N) across all children that have been visited.
  /// </summary>
  internal double SumWVisited;

  /// <summary>
  /// Sum of D (D * N) across all children that have been visited.
  /// </summary>
  internal double SumDVisited;

  internal SpanAligned<float> N;
  internal SpanAligned<float> NInFlightAdjusted; // weighted sum across all selectors (by 1 or collision fraction)
  internal SpanAligned<float> P;
  internal SpanAligned<float> W;
  internal SpanAligned<float> UV;
  internal SpanAligned<float> UP;
  internal SpanAligned<float> A;


  /// <summary>
  /// Constructor.
  /// </summary>
  internal GatheredChildStats()
  {
    N = new SpanAligned<float>(PUCTScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
    NInFlightAdjusted = new SpanAligned<float>(PUCTScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
    P = new SpanAligned<float>(PUCTScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
    W = new SpanAligned<float>(PUCTScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
    UV = new SpanAligned<float>(PUCTScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
    UP = new SpanAligned<float>(PUCTScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
    A = new SpanAligned<float>(PUCTScoreCalcVector.MAX_CHILDREN, ALIGNMENT);
  }


  /// <summary>
  /// Resets all summary fields to their default values.
  /// </summary>
  public void ResetSummaryFields()
  {
    SumNumInFlightAll = 0;
    SumPVisited = 0;
    SumNVisited = 0;
    SumWVisited = 0;
    SumDVisited = 0;
  }


  /// <summary>
  /// Tests if all values in these gathered stats 
  /// are close to the corresponding values in another set.
  /// </summary>
  /// <param name="otherStats"></param>
  /// <param name="allowableEpsilon"></param>
  public bool AllClose(GatheredChildStats otherStats, float allowableEpsilon, int maxIndex)
  {
    if (SumNumInFlightAll != otherStats.SumNumInFlightAll
     || Math.Abs(SumPVisited - otherStats.SumPVisited) > allowableEpsilon
     || Math.Abs(SumWVisited - otherStats.SumWVisited) > allowableEpsilon
     || Math.Abs(SumNVisited - otherStats.SumNVisited) > allowableEpsilon)
    {
      return false;
    }

    Span<float> sN = N.Span;
    Span<float> sInFlight = NInFlightAdjusted.Span;
    Span<float> sP = P.Span;
    Span<float> sA = A.Span;
    Span<float> sW = W.Span;
    Span<float> sUV = UV.Span;
    Span<float> sUP = UP.Span;

    Span<float> osN = otherStats.N.Span;
    Span<float> osInFlight = otherStats.NInFlightAdjusted.Span;
    Span<float> osP = otherStats.P.Span;
    Span<float> osA = otherStats.A.Span;
    Span<float> osW = otherStats.W.Span;
    Span<float> osUV = otherStats.UV.Span;
    Span<float> osUP = otherStats.UP.Span;

    for (int i = 0; i <= maxIndex; i++)
    {
      if (sN[i] != osN[i] ||
          Math.Abs(sInFlight[i] - osInFlight[i]) > allowableEpsilon ||
          Math.Abs(sP[i] - osP[i]) > allowableEpsilon ||
          Math.Abs(sA[i] - osA[i]) > allowableEpsilon ||
          Math.Abs(sW[i] - osW[i]) > allowableEpsilon ||
          Math.Abs(sUV[i] - osUV[i]) > allowableEpsilon ||
          Math.Abs(sUP[i] - osUP[i]) > allowableEpsilon)
      {
        return false;
      }
    }

    return true;
  }

  #region ToString

  /// <summary>
  /// Returns string representation of a Span<float>.
  /// </summary>
  /// <param name="name"></param>
  /// <param name="s"></param>
  /// <param name="maxItems"></param>
  /// <returns></returns>
  static string DumpSpan(string name, Span<float> s, int maxItems)
  {
    string str = $"{name}: ";
    for (int i = 0; i < s.Length && i < maxItems; i++)
    {
      str += $"{s[i],6:F3}";
    }
    return str;
  }


  /// <summary>
  /// Returns string representation of an Span<int>.
  /// </summary>
  /// <param name="name"></param>
  /// <param name="s"></param>
  /// <param name="maxItems"></param>
  /// <returns></returns>
  static string DumpSpanInt(string name, Span<int> s, int maxItems)
  {
    string str = $"{name}: ";
    for (int i = 0; i < s.Length && i < maxItems; i++)
    {
      str += $"{s[i],12:N0}";
    }
    return str;
  }


  /// <summary>
  /// Returns string representation of spans.
  /// </summary>
  /// <returns></returns>
  public string SpansStrings()
  {
    // find index of first element for which P is zero indicating end of moves (all actual moves will have nonzero P).
    int numNonZero = 0;
    for (int i = 0; i < N.Length; i++)
    {
      if (P.Span[i] == 0)
      {
        break;
      }
      else
      {
        numNonZero++;
      }
    }

    string s = DumpSpan("N", N.Span, numNonZero) + " " + System.Environment.NewLine;
    s += DumpSpan("InFlight", NInFlightAdjusted.Span, numNonZero) + " " + System.Environment.NewLine;
    s += DumpSpan("P", P.Span, numNonZero) + " " + System.Environment.NewLine;
    s += DumpSpan("W", W.Span, numNonZero) + " " + System.Environment.NewLine;
    s += DumpSpan("U", UV.Span, numNonZero);
    return s;
  }


  /// <summary>
  /// Returns string representation.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    string summary =
      $"SumNumInFlightAll: {SumNumInFlightAll,6:F3} " +
      $"SumPVisited: {SumPVisited,6:F3} " +
      $"SumNVisited: {SumNVisited,6:F3} " +
      $"SumWVisited: {SumWVisited,6:F3} " +
      $"SumDVisited: {SumDVisited,6:F3}";
    return summary + System.Environment.NewLine + SpansStrings();
  }

  #endregion
}
