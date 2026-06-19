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
using System.Collections.Generic;

using Ceres.Chess;

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

    public float AvgAbsQDifference;

    public float TotalRuntimeLC0;
    public float TotalRuntime1;
    public float TotalRuntime2;

    public float TotalNodesLC0;
    public float TotalNodes1;
    public float TotalNodes2;

    public float[] FinalQ1;
    public float[] FinalQ2;

    // --- Per-position averages (mirroring the console detail columns) ---

    /// <summary>Average search tree average-depth for engine 1 / 2.</summary>
    public float AvgDepth1;
    public float AvgDepth2;

    /// <summary>Average of the per-position maximum search depth for engine 1 / 2 (mean of per-position maxima, not the global maximum).</summary>
    public float MaxDepth1;
    public float MaxDepth2;

    /// <summary>Average visit-distribution entropy for engine 1 / 2.</summary>
    public float VisitEntropy1;
    public float VisitEntropy2;

    /// <summary>Average node-selection yield fraction for engine 1 / 2.</summary>
    public float YieldFrac1;
    public float YieldFrac2;

    /// <summary>Average percentage of total root visits placed on the correct move(s) for engine 1 / 2.</summary>
    public float CorrectMoveVisitFracPct1;
    public float CorrectMoveVisitFracPct2;

    /// <summary>Average root Q for engine 1 / 2 / external engine.</summary>
    public float AvgQ1;
    public float AvgQ2;
    public float AvgQLC0;

    // --- Totals (sums) over all positions ---

    /// <summary>Total number of NN batches evaluated by engine 1 / 2.</summary>
    public long TotalNNBatches1;
    public long TotalNNBatches2;

    /// <summary>Total number of NN position evaluations by engine 1 / 2.</summary>
    public long TotalNNEvals1;
    public long TotalNNEvals2;

    /// <summary>Total tablebase hits for engine 1 / 2.</summary>
    public long TotalTablebaseHits1;
    public long TotalTablebaseHits2;

    /// <summary>Total nodes when each engine first chose its eventual top move (over positions where both engines solved).</summary>
    public long TotalNodesWhenChoseTopN1;
    public long TotalNodesWhenChoseTopN2;

    // --- Head-to-head correct-move-visits (positions where both engines exposed root visit stats) ---

    /// <summary>Positions where engine 1 placed a (meaningfully, &gt; 3 points) larger fraction of its visits on the correct move(s).</summary>
    public int CountEngine1MoreCorrectVisits;

    /// <summary>Positions where engine 2 placed a (meaningfully, &gt; 3 points) larger fraction of its visits on the correct move(s).</summary>
    public int CountEngine2MoreCorrectVisits;

    /// <summary>Positions where the two engines were within 3 percentage points on correct-move visit fraction.</summary>
    public int CountCorrectVisitsEqual;

    /// <summary>Number of positions over which the correct-move-visit comparison was made (both engines had visit stats).</summary>
    public int CountCorrectVisitsCompared;

    /// <summary>Fraction of compared positions in which engine 1 placed more visits on the correct move(s) than engine 2.</summary>
    public float FracEngine1MoreCorrectVisits
      => CountCorrectVisitsCompared == 0 ? 0 : (float)CountEngine1MoreCorrectVisits / CountCorrectVisitsCompared;

    // --- Suite identity ---

    /// <summary>The suite ID (first argument of the SuiteTestDef constructor).</summary>
    public string ID;

    /// <summary>The EPD file name used as the source of test positions.</summary>
    public string EPDFileName;

    /// <summary>Number of positions in the test set actually run (after filtering/slicing).</summary>
    public int NumPositionsTested;

    /// <summary>The search limit used by engine 1.</summary>
    public SearchLimit SearchLimit1;

    /// <summary>The search limit used by engine 2 (null if no second engine).</summary>
    public SearchLimit SearchLimit2;

    /// <summary>Machine on which the suite was run.</summary>
    public string MachineName;

    /// <summary>Date/time the suite result was produced.</summary>
    public DateTime RunDateTime;

    // --- Performance summary metrics (engine 1 vs engine 2) ---

    /// <summary>Mean symmetric KL divergence between the two engines' empirical root-move visit distributions (NaN if unavailable).</summary>
    public float MeanPolicyKLD;

    /// <summary>Number of positions over which the policy KLD was computed (both engines exposed visit stats).</summary>
    public int CountPolicyKLDPositions;

    /// <summary>Average engine-reported evaluations per second for engine 1 / 2.</summary>
    public float AvgEPS1;
    public float AvgEPS2;

    /// <summary>Relative EPS of engine 1 vs engine 2 as a signed percentage ((EPS1/EPS2 - 1)*100).</summary>
    public float RelativeEPSPct;

    /// <summary>Difference in average solve score (engine 1 minus engine 2).</summary>
    public float SolveScoreDifference => AvgScore1 - AvgScore2;

    // --- Graded (regret-weighted) solution-quality comparison, engine 1 vs engine 2 ---

    /// <summary>Mean per-position graded-score difference (engine 1 minus engine 2), in 0-10 points.
    /// With weighted EPDs (ScoredMoves), this is a continuous solution-quality signal rather than binary solved/unsolved.</summary>
    public float ScoreDiffMean;

    /// <summary>Standard error of the mean graded-score difference.</summary>
    public float ScoreDiffStdErr;

    /// <summary>Number of positions contributing to the graded-score difference (both engines searched).</summary>
    public int ScoreDiffNumSamples;

    /// <summary>
    /// Z-score of the mean graded-score difference (ScoreDiffMean / ScoreDiffStdErr). The headline
    /// decision statistic: |z| greater than ~2 indicates engine 1 is statistically better (positive) or
    /// worse (negative) than engine 2 at finding the oracle-preferred move across the suite.
    /// </summary>
    public float ScoreDiffZ;

    /// <summary>Average graded regret for engine 1 (fraction of best-move credit forfeited; 0 = always best move).</summary>
    public float AvgRegret1 => 1.0f - AvgScore1 / 10.0f;

    /// <summary>Average graded regret for engine 2 (fraction of best-move credit forfeited; 0 = always best move).</summary>
    public float AvgRegret2 => 1.0f - AvgScore2 / 10.0f;

    // --- Nodes-to-solution speed comparison (engine 1 vs engine 2) ---

    /// <summary>Mean per-position difference in fraction-of-nodes-when-top-move-chosen (engine 1 minus engine 2);
    /// negative means engine 1 locked onto its move earlier (using a smaller fraction of its search) than engine 2.</summary>
    public float NodesToFindFasterMean;

    /// <summary>Standard error of <see cref="NodesToFindFasterMean"/>.</summary>
    public float NodesToFindFasterStdErr;

    /// <summary>Z-score of <see cref="NodesToFindFasterMean"/> (negative favors engine 1 finding its move faster).</summary>
    public float NodesToFindFasterZ;

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

