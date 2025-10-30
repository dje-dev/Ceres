﻿#region License notice

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
using System.Linq;
using System.Threading;

using Ceres.Base.Benchmarking;
using Ceres.Base.DataTypes;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.NNEvaluators.Specifications;
using Ceres.Chess.NNEvaluators.Specifications.Internal;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Commands
{
  /// <summary>
  /// Implementation of "backendbench" command which tests the 
  /// NN evaluator performance across a range of possible batch sizes.
  /// 
  /// Sample usages:
  ///   Ceres backendbench device=gpu:0,1,2,3
  ///   Ceres backendcompare network=mg-40b-swa-1595000 batchspec=[96..192]
  /// </summary>
  public record FeatureBenchmarkBackend
  {
    NNNetSpecificationString NetworkSpec;
    NNDevicesSpecificationString DeviceSpec;
    string batchSpec = null;


    public void ParseFields(string args)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args, null);

      NetworkSpec = keys.GetValueOrDefaultMapped("Network", CeresUserSettingsManager.Settings.DefaultNetworkSpecString, true, spec => new NNNetSpecificationString(spec));
      DeviceSpec = keys.GetValueOrDefaultMapped("Device", CeresUserSettingsManager.Settings.DefaultDeviceSpecString, true, spec => new NNDevicesSpecificationString(spec));
      batchSpec = keys.GetValueOrDefaultMapped("BatchSpec", null, false, spec=>spec);
    }


    const int SKIP = 2;
    const int TRIES = 10;

    public void ExecuteComparisonTest()
    {
      if(batchSpec == null)
      {
        throw new Exception("BatchSpec must be specified");
      }

      if(DeviceSpec.ComboType != NNEvaluatorDeviceComboType.Single)
      {
//        throw new Exception("BACKENDBENCH_COMPARE only supported with single devices");
      }

      //NNDevicesSpecificationString deviceSpec = new NNDevicesSpecificationString(modifiedDeviceString);
      NNEvaluatorDef evaluatorDef = new NNEvaluatorDef(NetworkSpec.ComboType, NetworkSpec.NetDefs,
                                                       DeviceSpec.ComboType, DeviceSpec.Devices, NetworkSpec.OptionsString, null);

      Console.WriteLine($"\r\nTESTING WITH BASELINE DEVICE SPECIFICTATION {NNDevicesSpecificationString.ToSpecificationString(DeviceSpec.ComboType, DeviceSpec.Devices)}");

      int maxBatchSize = Math.Min(CeresUserSettingsManager.Settings.MaxBatchSize, 2048);

      // Run base evaluator.
      var (evaluator1, _) = BackendBench(evaluatorDef, 16, maxBatchSize: maxBatchSize, show:false); // warmup
      var (evaluator2, before) = BackendBench(evaluatorDef, SKIP, TRIES, maxBatchSize: maxBatchSize);

      Thread.Sleep(3_000); // cooloff

      // Bulid modified evaluator.
      NNEvaluatorDeviceDef deviceDev = DeviceSpec.Devices[0].Item1;

      OptionsParserHelpers.ParseBatchSizeSpecification(batchSpec, out int ? maxTestBatchSize, out int ? optimalTestBatchSize, out string batchSizesFileName);
      if (batchSizesFileName != null)
      {
        throw new NotImplementedException("Use of batch file configuration file not yet supported.");
      }

      for (int i=0; i< DeviceSpec.Devices.Count;i++)
      {
        DeviceSpec.Devices[i].Item1.MaxBatchSize = maxTestBatchSize;
        DeviceSpec.Devices[i].Item1.OptimalBatchSize = optimalTestBatchSize;
      }
#if NOT
      string deviceSpecString = NNDevicesSpecificationString.ToSpecificationString(DeviceSpec.ComboType, DeviceSpec.Devices);
      if (DeviceSpec.Devices[0].Item1.MaxBatchSize is not null)
      {
        throw new Exception("Device specification already ")
      }
      string modifiedDeviceString = deviceSpecString + batchSpec;
#endif
      Console.WriteLine($"\r\n-------------------------------------------------------------------------------------------------------------------");
      Console.WriteLine($"TESTING WITH MODIFIED DEVICE SPECIFICTATION {NNDevicesSpecificationString.ToSpecificationString(DeviceSpec.ComboType, DeviceSpec.Devices)}");


      NNEvaluatorDef evaluatorDef1 = new NNEvaluatorDef(NetworkSpec.ComboType, NetworkSpec.NetDefs,
                                                        DeviceSpec.ComboType, DeviceSpec.Devices, NetworkSpec.OptionsString, null);

      // Run modified evaluator.
      var (evaluator3, _) = BackendBench(evaluatorDef1, 16, TRIES, maxBatchSize: maxBatchSize, show: false); // warmup
      var (evaluator4, after) = BackendBench(evaluatorDef1, SKIP, TRIES, maxBatchSize: maxBatchSize);

      // Output summary statistics.
      static float PctDiff(float v1, float v2) => 100.0f * ((v1 - v2) / v2);
      float sum = 0;
      for (int i = 0; i < Math.Min(before.Count, after.Count); i++)
      {
        float pctDiff = PctDiff(after[i].Item2, before[i].Item2);
        sum += pctDiff;
        Console.WriteLine(before[i].Item1 + ": " + MathF.Round(before[i].Item2, 0) + " --> " + MathF.Round(after[i].Item2, 0) + " " + $"{pctDiff,6:F2}%");
      }

      Console.WriteLine($"\r\nAverage Speedup Per Batch Size: {(sum / before.Count),6:F2}%");

      // NOTE: This is a hack. If we allow an NNBackendLC0_CUDA to destruct (finalizer)
      //       while operations on another NNBackendLC0_CUDA are active then a
      //       hard CUDA crash will results (cannot capture graph while other operations active).
      //       Therefore we hold onto references from above evaluators and only release here.
      evaluator1.Shutdown();
      evaluator2.Shutdown();
      evaluator3.Shutdown();
      evaluator4.Shutdown();

    }


    public (NNEvaluator, List<(int, float)>) ExecuteBenchmark(NNEvaluatorDef evaluatorDef)
    {
      if (evaluatorDef == null)
      {
        evaluatorDef = new NNEvaluatorDef(NetworkSpec.ComboType, NetworkSpec.NetDefs,
                                          DeviceSpec.ComboType, DeviceSpec.Devices, NetworkSpec.OptionsString, null);
      }
      return BackendBench(evaluatorDef);
    }


    /// <summary>
    /// Runs a benchmark of the NN backend (similar to the LC0 backendbench command).
    /// </summary>
    /// <param name="evaluatorDef"></param>
    /// <param name="extraSkipMultiplier"></param>
    /// <param name="numRunsPerBatchSize"></param>
    /// <param name="firstBatchSize"></param>
    /// <param name="maxBatchSize"></param>
    /// <param name="show"></param>
    /// <returns></returns>
    public static (NNEvaluator, List<(int,float)>) BackendBench(NNEvaluatorDef evaluatorDef, 
                                                                int extraSkipMultiplier = 1, int numRunsPerBatchSize = 5, 
                                                                int firstBatchSize = 1, int maxBatchSize = 4096, 
                                                                bool show = true)
    {
      if (show)
      {
        Console.WriteLine();
        Console.WriteLine($"Benchmark of neural network evaluator backend - {evaluatorDef}");
      }

      NNEvaluator evaluator = evaluatorDef.ToEvaluator();
      maxBatchSize = Math.Min(maxBatchSize, evaluator.MaxBatchSize);

      List<(int, float)> ret = new();
      InitPositionsBuffer(maxBatchSize);

      evaluator.Warmup();

      TestBatchSize(evaluator, 1, show:show);

      // Loop over a set of batch sizes.
      int lastBatchSize = firstBatchSize;
      while (true)
      {
        int skip = extraSkipMultiplier * (lastBatchSize < 256 ? 4 : (lastBatchSize < 512 ? 16 : 32));
        lastBatchSize += skip;
        lastBatchSize = (lastBatchSize / 4) * 4;

        if (lastBatchSize > maxBatchSize)
        {
          break;
        }
        else
        {
          float nps = TestBatchSize(evaluator, lastBatchSize, numRunsPerBatchSize, show);
          ret.Add((lastBatchSize, nps));  
        }
      }
      if (show)
      {
        Console.WriteLine();
      }
      evaluator.Shutdown();

      return (evaluator, ret);
    }

    static EncodedPositionWithHistory[] positions;

    static EncodedPositionBatchFlat testBatchBuffer;


    static void InitPositionsBuffer(int maxPositions)
    {
      positions = new EncodedPositionWithHistory[maxPositions];

      // Create test batch (a mix of two different positions, to assist in diagnostics).
      EncodedPositionWithHistory pos1 = EncodedPositionWithHistory.FromPosition(Position.StartPosition);
      EncodedPositionWithHistory pos2 = EncodedPositionWithHistory.FromFEN("1k1r2r1/1p3p2/p2pbb2/2q4p/1PP1P3/5N1P/P2QB1P1/3R1R1K b - - 0 30");
      for (int i = 0; i < maxPositions; i++)
      {
        positions[i] = i % 7 == 1 ? pos2 : pos1;
      }

      testBatchBuffer = new EncodedPositionBatchFlat(positions.AsSpan().Slice(0, maxPositions), maxPositions, true);
      (testBatchBuffer as IEncodedPositionBatchFlat).TrySetMoves();
    }

    private static float TestBatchSize(NNEvaluator evaluator, int batchSize, int numRunsPerBatchSize = 5, bool show = false)
    {
      // Prepare batch with test positions.
      IEncodedPositionBatchFlat testBatch = testBatchBuffer.GetSubBatchSlice(0, batchSize);

      // Run warmup steps if first time.
      if (batchSize == 1)
      {
        for (int i = 0; i < 5; i++)
        {
          Thread.Sleep(10);
          evaluator.EvaluateIntoBuffers(testBatch);
        }
      }

      // Keep track of all timings except the worst.
      TopN<float> bestTimings = new(numRunsPerBatchSize - 1, s => s);
      for (int runIndex = 0; runIndex < numRunsPerBatchSize; runIndex++)
      {
        TimingStats stats = new();
        using (new TimingBlock(stats, TimingBlock.LoggingType.None))
        {
          evaluator.EvaluateIntoBuffers(testBatch);
        }
        bestTimings.Add((float)stats.ElapsedTimeSecs);
      }

      float bestSeconds = bestTimings.Members.Min(f => f);
      float nps = batchSize / bestSeconds;
      if (show)
      {
        Console.WriteLine($"Benchmark batch size {batchSize} with inference time {1000.0f * bestSeconds,7:F3}ms - throughput {nps,7:N0} nps.");
      }
      return nps;
    }
  }
}
