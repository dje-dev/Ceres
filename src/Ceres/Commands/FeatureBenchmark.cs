#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using directive

using System;
using Ceres.Base.Benchmarking;
using Ceres.Base.Environment;
using Ceres.Base.Math;
using Ceres.Base.Misc;
using Ceres.Base.OperatingSystem.NVML;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Boards;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;
using Ceres.Features;

#endregion

namespace Ceres.Commands
{
  public class FeatureBenchmark
  {
    /// <summary>
    /// Performs CPU and GPU benchmark and dumps summary results to Console.
    /// </summary>
    public static void DumpBenchmark()
    {
      Console.WriteLine();
      int cpuScore = DumpCPUBenchmark();
      (int countGPU, int gpuSumNPS) = DumpGPUBenchmark();
      Console.WriteLine();

      // Finally output a short summary of results and Ceres version
      // (useful single line of information for testers to include in posts).
      string gpuCountStr = countGPU == 1 ? "" : $" ({countGPU})";
      Console.WriteLine($"CPU Score={cpuScore}, NPS={gpuSumNPS}{gpuCountStr} "
                      + $"using {CeresUserSettingsManager.Settings.DefaultNetworkSpecString} "
                      + $"v{CeresVersion.VersionString} {GitInfo.VersionString}");
      Console.WriteLine();
    }


    /// <summary>
    /// Runs CPU benchmark and outputs summary results,
    /// with an overall statistic provided (index to 100 on a Intel Skylake 6142).
    /// </summary>
    /// <returns>Relative CPU index (baseline 100)</returns>
    static int DumpCPUBenchmark()
    {
      Console.WriteLine("-----------------------------------------------------------------------------------");
      Console.WriteLine("CPU BENCHMARK");

      Position ps = Position.StartPosition;
      EncodedPositionBoard zb = default;
      MGMove nmove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(new EncodedMove("e2e4"), MGChessPositionConverter.MGChessPositionFromFEN(ps.FEN));

      float ops1 = Benchmarking.DumpOperationTimeAndMemoryStats(() => MGPosition.FromPosition(ps), "MGPosition.FromPosition");
      float ops2 = Benchmarking.DumpOperationTimeAndMemoryStats(() => MGChessPositionConverter.MGChessPositionFromFEN(ps.FEN), "MGChessPositionFromFEN");
      float ops3 = Benchmarking.DumpOperationTimeAndMemoryStats(() => ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(nmove), "MGChessMoveToLZPositionMove");
      float ops4 = Benchmarking.DumpOperationTimeAndMemoryStats(() => EncodedBoardZobrist.ZobristHash(zb), "ZobristHash");

      // Performance metric is against a baseline system (Intel Skylake 6142)
      const float REFERENCE_BM1_OPS = 2160484;
      const float REFERENCE_BM2_OPS = 448074;
      const float REFERENCE_BM3_OPS = 157575582;
      const float REFERENCE_BM4_OPS = 112731351;

      float relative1 = ops1 / REFERENCE_BM1_OPS;
      float relative2 = ops2 / REFERENCE_BM2_OPS;
      float relative3 = ops3 / REFERENCE_BM3_OPS;
      float relative4 = ops4 / REFERENCE_BM4_OPS;

      float avg = StatUtils.Average(relative1, relative2, relative3, relative4);

      Console.WriteLine();
      Console.WriteLine($"CERES CPU BENCHMARK SCORE: {avg*100,4:F0}");

      return (int)MathF.Round(avg * 100, 0);
    }


    /// <summary>
    /// Dumps GPU information and runs benchmarks.
    /// </summary>
    /// <returns>Sum of NPS across all GPUs</returns>
    public static (int CountGPU, int SumNPS) DumpGPUBenchmark()
    {
      Console.WriteLine();
      Console.WriteLine("-----------------------------------------------------------------------------------");
      Console.WriteLine($"GPU BENCHMARK (benchmark net: {CeresUserSettingsManager.Settings.DefaultNetworkSpecString})");
      Console.WriteLine();
      Console.WriteLine(NVML.InfoDescriptionHeaderLine1 + "   NPS 1  NPS Batch");
      Console.WriteLine(NVML.InfoDescriptionHeaderLine2 + "   -----  ---------");

      int sumNPS = 0;
      int countGPU = 0;
      foreach (NVMLGPUInfo info in NVML.GetGPUsInfo())
      {
        NNEvaluatorDef evaluatorDef = NNEvaluatorDef.FromSpecification(CeresUserSettingsManager.Settings.DefaultNetworkSpecString,
                                                                       "GPU:" + info.ID.ToString());
        NNEvaluator evaluator = NNEvaluatorFactory.BuildEvaluator(evaluatorDef);
        (float npsSingletons, float npsBigBatch, _) = NNEvaluatorBenchmark.EstNPS(evaluator, false, 512, true, 3);

        Console.WriteLine(NVML.GetInfoDescriptionLine(info) + $"    {npsSingletons,6:N0} { npsBigBatch,10:N0}");

        countGPU++;
        sumNPS += (int)npsBigBatch;

        evaluator.Dispose();
      }

      return (countGPU, sumNPS);
    }


  }
}
