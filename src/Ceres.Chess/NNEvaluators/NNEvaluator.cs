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
using System.Linq;

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Abstract base class for objects which can evaluate positions via neural network.
  /// </summary>
  public abstract class NNEvaluator : IDisposable
  {
    internal object PersistentID { set; get; }
    public bool IsPersistent => PersistentID != null;
    public int NumInstanceReferences { internal set; get; }
    public bool IsShutdown { private set; get; } = false;

    [Flags]
    public enum InputTypes 
    { 
      Undefined = 0, 
      Boards = 1, 
      Hashes = 2, 
      Moves = 4, 
      Positions = 8, 

      All =  Boards | Hashes | Moves | Positions
    };

    /// <summary>
    /// String description of underlying engine type.
    /// </summary>
    public string EngineType;

    /// <summary>
    /// String identifier of the underlying engine network.
    /// </summary>
    public string EngineNetworkID;

    /// <summary>
    /// Estimated performance characteristics.
    /// </summary>
    public NNEvaluatorPerformanceStats PerformanceStats;

    /// <summary>
    /// If the network returns policy moves in the same order
    /// as the legal MGMoveList.
    /// </summary>
    public virtual bool PolicyReturnedSameOrderMoveList => false;


    /// <summary>
    /// Performs quick benchmarks on evaluator to determine performance
    /// includng single positions and batchs and optionally estimated 
    /// batch size breaks (cut points beyond which speed drops due to batching effects).
    /// </summary>
    /// <param name="computeBreaks"></param>
    /// <param name="maxSeconds"></param>
    public virtual void CalcStatistics(bool computeBreaks, float maxSeconds = 1.0f)
    {
      (float npsSingletons, float npsBigBatch, int[] breaks) = NNEvaluatorBenchmark.EstNPS(this, computeBreaks);
      PerformanceStats = new NNEvaluatorPerformanceStats() { EvaluatorType = GetType(), SingletonNPS = npsSingletons, 
                                                             BigBatchNPS = npsBigBatch, Breaks = breaks };
    }

    public virtual float EstNPSBatch => PerformanceStats == null ? 30_000 : PerformanceStats.BigBatchNPS;
    public virtual float EstNPSSingleton => PerformanceStats == null ? 500 : PerformanceStats.SingletonNPS;

    /// <summary>
    /// Types of input(s) required by the evaluator.
    /// </summary>
    public virtual InputTypes InputsRequired => InputTypes.Boards;

    /// <summary>
    /// If the evaluator has a WDL (win/draw/loss) head.
    /// </summary>
    public abstract bool IsWDL { get; }

    /// <summary>
    /// If the evaluator has an M (moves left) head.
    /// </summary>
    public abstract bool HasM { get; }

    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public abstract int MaxBatchSize { get; }


    #region Static helpers

    /// Returns an NNEvaluator corresponding to speciifed strings with network and device specifications.
    /// </summary>
    /// <param name="netSpecificationString"></param>
    /// <param name="deviceSpecificationString"></param>
    /// <returns></returns>
    public static NNEvaluator FromSpecification(string netSpecificationString, string deviceSpecificationString)
      => NNEvaluatorDef.FromSpecification(netSpecificationString, deviceSpecificationString).ToEvaluator();

    #endregion

    #region Basic evaluator methods

    /// <summary>
    /// Worker method that evaluates batch of positions into the internal buffers.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public abstract IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false);


    public long NumBatchesEvaluated { private set; get; }

    public long NumPositionsEvaluated { private set; get; }

    /// <summary>
    /// Evaluates positions into internal buffers. 
    /// 
    /// Note that the batch returned is built over the local buffers 
    /// and may be overwritten upon next call to this method.
    /// 
    /// Therefore this method is intended only for low-level use.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public IPositionEvaluationBatch EvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      // Compute Moves if necessary
      if (InputsRequired.HasFlag(InputTypes.Moves))
      {
        positions.TrySetMoves();

        if (positions.Moves == null)
        {
          throw new Exception($"NNEvaluator requires Positions to be provided {this}");
        }
      }


      IPositionEvaluationBatch batch = DoEvaluateIntoBuffers(positions, retrieveSupplementalResults);

      NumBatchesEvaluated++;
      NumPositionsEvaluated += positions.NumPos;
      return batch;
    }

    object lockObj = new ();


    /// <summary>
    /// Evaluates batch of positions into newly allocated (and returned) result buffers.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public NNEvaluatorResult[] EvaluateBatch(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (positions.NumPos == 0)
      {
        throw new ArgumentException("Illegal input batch, empty.");
      }

      lock (lockObj)
      {
        // Evaluate into evaluators buffers
        IPositionEvaluationBatch bufferedResult = EvaluateIntoBuffers(positions, retrieveSupplementalResults);

        // Allocate copy buffers
        NNEvaluatorResult[] ret = new NNEvaluatorResult[bufferedResult.NumPos];

        // Extract values to copy buffers
        for (int i = 0; i < bufferedResult.NumPos; i++)
        {
          ExtractToNNEvaluatorResult(out ret[i], bufferedResult, i);
        }

        return ret;
      }
    }

    /// <summary>
    /// Internal worker to extract a single NNEvaluatorResult from an IPositionEvaluationBatch.
    /// </summary>
    /// <param name="result"></param>
    /// <param name="batch"></param>
    /// <param name="batchIndex"></param>
    private void ExtractToNNEvaluatorResult(out NNEvaluatorResult result, 
                                            IPositionEvaluationBatch batch, int batchIndex)
    {
      float w = batch.GetWinP(batchIndex);
      float l = IsWDL ? batch.GetLossP(batchIndex) : float.NaN;
      float m = HasM ? batch.GetM(batchIndex) : float.NaN;
      NNEvaluatorResultActivations activations = batch.GetActivations(batchIndex);
      (Memory<CompressedPolicyVector> policies, int index) policyRef = batch.GetPolicy(batchIndex);
      result = new NNEvaluatorResult(w, l, m, policyRef.policies.Span[policyRef.index], activations);
    }

    #endregion

    #region Helper methods

    /// <summary>
    /// Helper method to evaluates a single position.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public NNEvaluatorResult Evaluate(PositionWithHistory position, bool fillInMissingPlanes, bool retrieveSupplementalResults = false)
    {
      // TODO: someday we might be able to relax the InputTypes.All below
      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(1, NNEvaluator.InputTypes.All);
      builder.Add(position, fillInMissingPlanes);

      NNEvaluatorResult[] result = EvaluateBatch(builder.GetBatch(), retrieveSupplementalResults);
      return result[0];
    }


    /// <summary>
    /// Helper method to evaluates set of PositionWithHistory.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public NNEvaluatorResult[] Evaluate(IEnumerable<PositionWithHistory> positions, bool fillInMissingPlanes, bool retrieveSupplementalResults = false)
    {
      PositionWithHistory[] positionsAll = positions.ToArray();

      // TODO: someday we might be able to relax the InputTypes.All below
      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(positionsAll.Length, NNEvaluator.InputTypes.All);
      foreach (PositionWithHistory position in positionsAll)
      {
        builder.Add(position, fillInMissingPlanes);
      }

      return EvaluateBatch(builder.GetBatch(), retrieveSupplementalResults);
    }


    /// <summary>
    /// Helper method to evaluates a single position.
    /// </summary>
    /// <param name="encodedPosition"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public NNEvaluatorResult Evaluate(in Position position, bool fillInMissingPlanes, bool retrieveSupplementalResults = false)
    {
      // TODO: someday we might be able to relax the InputTypes.All below
      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(1, NNEvaluator.InputTypes.All);
      builder.Add(in position, fillInMissingPlanes);

      NNEvaluatorResult[] result = EvaluateBatch(builder.GetBatch(), retrieveSupplementalResults);
      return result[0];
    }

    /// <summary>
    /// Helper method to evaluates a single position.
    /// </summary>
    /// <param name="encodedPosition"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public IPositionEvaluationBatch Evaluate(in EncodedPositionWithHistory encodedPosition, bool retrieveSupplementalResults = false)
    {
      return Evaluate(new EncodedPositionWithHistory[] { encodedPosition }, 1, retrieveSupplementalResults);
    }


    /// <summary>
    /// Helper method to evaluates batch originating from array of EncodedPosition.
    /// </summary>
    /// <param name="encodedPositions"></param>
    /// <param name="numPositions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public IPositionEvaluationBatch Evaluate(EncodedPositionWithHistory[] encodedPositions, int numPositions, bool retrieveSupplementalResults = false)
    {
      EncodedPositionBatchFlat batch;
      if (InputsRequired > InputTypes.Boards)
      {
        EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(numPositions, InputsRequired | InputTypes.Positions);
        for (int i = 0; i < numPositions; i++)
        {
          builder.Add(in encodedPositions[i]);
        }
        batch = builder.GetBatch();
      }
      else
      {
        bool setPositions = InputsRequired.HasFlag(InputTypes.Positions);
        batch = new EncodedPositionBatchFlat(encodedPositions, numPositions, setPositions);
      }

      return EvaluateIntoBuffers(batch, retrieveSupplementalResults);
    }


    /// <summary>
    /// Helper method to evaluates batch originating from array of EncodedTrainingPosition.
    /// </summary>
    /// <param name="encodedPositions"></param>
    /// <param name="numPositions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public IPositionEvaluationBatch Evaluate(Span<EncodedTrainingPosition> encodedPositions, int numPositions, bool retrieveSupplementalResults = false)
    {
      EncodedPositionBatchFlat batch;
      if (InputsRequired > InputTypes.Boards)
      {
        EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(numPositions, InputsRequired);
        for (int i = 0; i < numPositions; i++)
        {
          // Unmirror before adding.
          builder.Add(encodedPositions[i].PositionWithBoardsMirrored.Mirrored);
        }

        batch = builder.GetBatch();
      }
      else
      {
        bool setPositions = InputsRequired.HasFlag(InputTypes.Positions);
        batch = new EncodedPositionBatchFlat(encodedPositions, numPositions, EncodedPositionType.PositionOnly, setPositions);
      }

      return EvaluateIntoBuffers(batch, retrieveSupplementalResults);
    }


    /// <summary>
    /// Evaluates all positions in an oversized batch (that cannot be evaluated all at once).
    /// The batch is broken into sub-batches, evaluated, and each sub-batch is passed to a provided delete.
    /// </summary>
    /// <param name="bigBatch"></param>
    /// <param name="processor"></param>
    public void EvaluateOversizedBatch(EncodedPositionBatchFlat bigBatch, Action<(int, NNEvaluatorResult[])> processor)
    {
      int subBatchSize = MaxBatchSize;
      if (bigBatch.NumPos % subBatchSize != 0) throw new Exception("Size not divisible by " + subBatchSize);

      // Process each sub-batch.
      for (int i = 0; i < bigBatch.NumPos / subBatchSize; i++)
      {
        // Extract a slice of manageable size.
        EncodedPositionBatchFlatSlice slice = new EncodedPositionBatchFlatSlice(bigBatch, i * subBatchSize, subBatchSize);

        // Evaluate with the neural network.
        NNEvaluatorResult[] result = EvaluateBatch(slice);

        // Passs sub-batch to delegate.
        processor((i * subBatchSize, result));
      }
    }


    /// <summary>
    /// If this evaluator produces the same output as another specified evaluator.
    /// 
    /// TODO: implement this on more subclasses.
    /// </summary>
    /// <param name="evaluator"></param>
    /// <returns></returns>
    public virtual bool IsEquivalentTo(NNEvaluator evaluator)
    {
      // Default assumption is false (unless same object);
      // subclasses may possibly implement smarter logic.
      return object.ReferenceEquals(this, evaluator);
    }

    #endregion

    #region Shutdown

    public void Dispose()
    {
      Release();
    }

    public void Release()
    {
      lock (this)
      {
        bool shouldShutdown = true;
        if (IsPersistent)
        {
          NumInstanceReferences--;
          shouldShutdown = NumInstanceReferences == 0;
        }

        if (shouldShutdown)
        {
          DoShutdown();
          IsShutdown = true;
        }
      }
    }

    protected abstract void DoShutdown();

    /// <summary>
    /// Shuts down the evaluator, releasing all associated resources.
    /// </summary>
    public void Shutdown()
    {
      if (NumInstanceReferences > 0)
      {
        throw new Exception("Cannot shutdown until all instances call Release()");
      }
      else
      {
        DoShutdown();

        if (IsPersistent)
        {
          NNEvaluatorFactory.DeletePersistent(this);
        }

        IsShutdown = true;
      }
    }

    #endregion
  }
}
