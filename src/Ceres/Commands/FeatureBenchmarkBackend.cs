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
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Commands
{
  /// <summary>
  /// Implementation of "backendbench" command which tests the 
  /// NN evaluator performance across a range of possible batch sizes.
  /// </summary>
  public record FeatureBenchmarkBackend
  {
    NNNetSpecificationString NetworkSpec;
    NNDevicesSpecificationString DeviceSpec;


    internal void ParseFields(string args)
    {
      KeyValueSetParsed keys = new KeyValueSetParsed(args, null);

      NetworkSpec = keys.GetValueOrDefaultMapped<NNNetSpecificationString>("Network", CeresUserSettingsManager.Settings.DefaultNetworkSpecString, true, spec => new NNNetSpecificationString(spec));
      DeviceSpec = keys.GetValueOrDefaultMapped("Device", CeresUserSettingsManager.Settings.DefaultDeviceSpecString, true, spec => new NNDevicesSpecificationString(spec));
    }


    internal void Execute()
    {
      NNEvaluatorDef evaluatorDef = new NNEvaluatorDef(NetworkSpec.ComboType, NetworkSpec.NetDefs,
                                                       DeviceSpec.ComboType, DeviceSpec.Devices, null);
      BackendBench(evaluatorDef);
    }


    /// <summary>
    /// Runs a benchmark of the NN backend (similar to the LC0 backendbech command).
    /// </summary>
    /// <param name="netSpec"></param>
    /// <param name="gpuSpec"></param>
    public static void BackendBench(NNEvaluatorDef evaluatorDef, int extraSkipMultiplier = 1)
    {
      Console.WriteLine();
      Console.WriteLine($"Benchmark of neural network evaluator backend - {evaluatorDef}");

      NNEvaluator evaluator = evaluatorDef.ToEvaluator();
      int maxBatchSize = evaluator.MaxBatchSize;

      TestBatchSize(evaluator, 1);

      // Loop over a set of batch sizes.
      int lastBatchSize = extraSkipMultiplier;
      while (true)
      {
        int skip = extraSkipMultiplier * (lastBatchSize < 512 ? 4 : 8);
        lastBatchSize += skip;
        lastBatchSize = (lastBatchSize / 4) * 4;

        if (lastBatchSize > maxBatchSize)
        {
          break;
        }
        else
        {
          TestBatchSize(evaluator, lastBatchSize);
        }
      }
      Console.WriteLine();
      evaluator.Shutdown();
    }

    private static void TestBatchSize(NNEvaluator evaluator, int batchSize)
    {
      const int NUM_RUNS_PER_BATCH_SIZE = 5;

      // Prepare batch with test positions.
      EncodedPositionWithHistory[] positions = new EncodedPositionWithHistory[batchSize];
      Array.Fill(positions, EncodedPositionWithHistory.FromPosition(Position.StartPosition));

      EncodedPositionBatchFlat batch = new EncodedPositionBatchFlat(positions.AsSpan(), batchSize, true);

      // Run warmup steps if first time.
      if (batchSize == 1)
      {
        for (int i = 0; i < 5; i++)
        {
          Thread.Sleep(10);
          evaluator.EvaluateIntoBuffers(batch);
        }
      }

      // Keep track of all timings except the worst.
      TopN<float> bestTimings = new(NUM_RUNS_PER_BATCH_SIZE - 1, s => s);
      for (int runIndex = 0; runIndex < NUM_RUNS_PER_BATCH_SIZE; runIndex++)
      {
        TimingStats stats = new();
        using (new TimingBlock(stats, TimingBlock.LoggingType.None))
        {
          evaluator.EvaluateIntoBuffers(batch);
        }
        bestTimings.Add((float)stats.ElapsedTimeSecs);
      }

      float bestSeconds = bestTimings.Members.Min(f => f);
      float nps = batchSize / bestSeconds;
      Console.WriteLine($"Benchmark batch size {batchSize} with inference time {1000.0f * bestSeconds,7:F3}ms - throughput {nps,7:N0} nps.");
    }
  }
}
