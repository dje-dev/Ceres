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
using System.Threading.Tasks;
using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.Positions;

#endregion

namespace Ceres.Chess.NNEvaluators
{
  /// <summary>
  /// Abstract base class for objects which can evaluate positions via neural network.
  /// </summary>
  public abstract class NNEvaluator
  {
    [Flags]
    public enum InputTypes
    {
      Undefined = 0,
      Boards = 1,
      Hashes = 2,
      Moves = 4,
      Positions = 8,
      LastMovePlies = 16,

      All = Boards | Hashes | Moves | Positions,
      AllWithLastMovePlies = Boards | Hashes | Moves | Positions | LastMovePlies
    };

    /// <summary>
    /// Miscellaneous information about the evaluator.
    /// </summary>
    public record EvaluatorInfo
    {
      /// <summary>
      /// Number of network parameters used during inference.
      /// </summary>
      public readonly long NumParameters;

      public EvaluatorInfo(long numParameters)
      {
        NumParameters = numParameters;
      }
    }


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



    internal object PersistentID { set; get; }
    public bool IsPersistent => PersistentID != null;
    public int NumInstanceReferences { internal set; get; }
    public bool IsShutdown { private set; get; } = false;

    /// <summary>
    /// If the underlying execution engine is threadsafe, 
    /// i.e. can support concurrent execution from 
    /// </summary>
    public virtual bool SupportsParallelExecution => true; 

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
    /// If action head is present in the network.
    /// </summary>
    public abstract bool HasAction { get; }

    /// <summary>
    /// If the evaluator has an UV (uncertainty of V) head.
    /// </summary>
    public abstract bool HasUncertaintyV { get; }

    /// <summary>
    /// If the evaluator has an secondary value head.
    /// </summary>
    public abstract bool HasValueSecondary { get; }

    /// <summary>
    /// The maximum number of positions that can be evaluated in a single batch.
    /// </summary>
    public abstract int MaxBatchSize { get; }

    /// <summary>
    /// When true and playing using SearchLimit of BestValueMove, engine using this evaluator 
    /// will slightly adjust evaluation when repetitions are nonzero to prefer repetitions/draws
    /// when seemingly losing and disfavor when seemingly winning.
    /// This feature can compensate for lack of history consideration by the neural network.
    /// </summary>
    public virtual bool UseBestValueMoveUseRepetitionHeuristic { get; set; } = false;

    /// <summary>
    /// Miscellaneous information about the evaluator.
    /// </summary>
    public virtual EvaluatorInfo Info => null;


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


    /// <summary>
    /// Optional worker method which evaluates batch of positions which are already converted into native format needed by evaluator.
    /// TODO: Reconsider this design, it is not very flexible because it requires network encoding specific logic in the evaluator.
    /// </summary>
    /// <param name="positionsNativeInput"></param>
    /// <param name="usesSecondaryInputs"></param>
    /// <param name="numPositions"></param>
    /// <param name="retrieveSupplementalResults"></param>?
    /// <returns></returns>
    /// <exception cref="NotImplementedException"></exception>
    public virtual IPositionEvaluationBatch DoEvaluateNativeIntoBuffers(object positionsNativeInput, bool usesSecondaryInputs,
                                                                        int numPositions, Func<int, int, bool> posMoveIsLegal,
                                                                        bool retrieveSupplementalResults = false)
    {
      throw new NotImplementedException();
    }

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
      SetMovesIfNeeded(positions);

      IPositionEvaluationBatch batch = DoEvaluateIntoBuffers(positions, retrieveSupplementalResults);

#if DEBUG
      if (!positions.PositionsBuffer.IsEmpty)
      {
        foreach (var pos in positions.PositionsBuffer.Span.Slice(0, positions.NumPos))
        {
          pos.Validate();
        }
      }
#endif

      NumBatchesEvaluated++;
      NumPositionsEvaluated += positions.NumPos;
      return batch;
    }


    private void SetMovesIfNeeded(IEncodedPositionBatchFlat positions)
    {
      // Compute Moves if necessary
      if (InputsRequired.HasFlag(InputTypes.Moves))
      {
        positions.TrySetMoves();

        if (positions.Moves.IsEmpty)
        {
          throw new Exception($"NNEvaluator requires Positions to be provided {this}");
        }
      }
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
    public void ExtractToNNEvaluatorResult(out NNEvaluatorResult result, IPositionEvaluationBatch batch, int batchIndex)
    {
      float w = batch.GetWinP(batchIndex);
      float l = IsWDL ? batch.GetLossP(batchIndex) : float.NaN;

      float w2 = HasValueSecondary ? batch.GetWin2P(batchIndex) : float.NaN;
      float l2 = HasValueSecondary && IsWDL ? batch.GetLoss2P(batchIndex) : float.NaN;

      float m = HasM ? batch.GetM(batchIndex) : float.NaN;
      float uncertaintyV = HasUncertaintyV ? batch.GetUncertaintyV(batchIndex) : float.NaN;
      
      NNEvaluatorResultActivations activations = batch.GetActivations(batchIndex);
      
      (Memory<CompressedPolicyVector> policies, int index) policyRef = batch.GetPolicy(batchIndex);

      FP16 extraStat0 = batch.GetExtraStat0(batchIndex);
      FP16 extraStat1 = batch.GetExtraStat1(batchIndex);

      // Possibly extract action values.
      ActionValues actionValues = default;
      if (HasAction)
      {
        ref readonly CompressedPolicyVector policy = ref policyRef.policies.Span[policyRef.index];
        int actionIndex = 0;
        foreach (var move in policy.ProbabilitySummary())
        {
          (float aW, float aD, float aL) = batch.GetA(batchIndex, move.Move.IndexNeuralNet);
          actionValues[actionIndex++] = ((FP16)aW, (FP16)aL);
        }
      }

      result = new NNEvaluatorResult(w, l, w2, l2, m, uncertaintyV, policyRef.policies.Span[policyRef.index], actionValues, activations, extraStat0, extraStat1);
    }

    #endregion

    #region Helper methods

    /// <summary>
    /// Helper method to evaluates a single position.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="fillInMissingPlanes">if history planes should be filled in if incomplete (typically necessary)</param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <param name="extraInputs">optional set of additional inputs to be set within the encoded batch</param>
    /// <returns></returns>
    public NNEvaluatorResult Evaluate(PositionWithHistory position, 
                                      bool fillInMissingPlanes = true, 
                                      bool retrieveSupplementalResults = false,
                                      InputTypes extraInputs = InputTypes.Undefined)
    {
      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(1, InputsRequired | extraInputs);
      builder.Add(position, fillInMissingPlanes);

      NNEvaluatorResult[] result = EvaluateBatch(builder.GetBatch(), retrieveSupplementalResults);
      return result[0];
    }


    /// <summary>
    /// Helper method to evaluates set of PositionWithHistory.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="fillInMissingPlanes">if history planes should be filled in if incomplete (typically necessary)</param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public NNEvaluatorResult[] Evaluate(IEnumerable<PositionWithHistory> positions, bool fillInMissingPlanes = true, bool retrieveSupplementalResults = false)
    {
      if (InputsRequired.HasFlag(InputTypes.LastMovePlies))
      {
        // TODO: it should be possible to extract some of the history
        //       from the positions into the LastMovePlies to support this case.
        //       Probably leverage existing method SetMoveSinceFromPositions to do this.
        throw new NotImplementedException();
      }

      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(positions.Count(), InputsRequired);
      foreach (PositionWithHistory position in positions)
      {
        builder.Add(position, fillInMissingPlanes);
      }

      return EvaluateBatch(builder.GetBatch(), retrieveSupplementalResults);
    }


    /// <summary>
    /// Helper method to evaluates a single position.
    /// </summary>
    /// <param name="position"></param>
    /// <param name="fillInMissingPlanes">if history planes should be filled in if incomplete (typically necessary)</param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public NNEvaluatorResult Evaluate(in Position position, bool fillInMissingPlanes = true, bool retrieveSupplementalResults = false)
    {
      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(1, InputsRequired | InputTypes.Positions);
      builder.Add(in position, fillInMissingPlanes);

      NNEvaluatorResult[] result = EvaluateBatch(builder.GetBatch(), retrieveSupplementalResults);
      return result[0];
    }


    /// <summary>
    /// Helper method to evaluates a single position.
    /// </summary>
    /// <param name="encodedPosition"></param>
    /// <param name="fillInHistory"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public IPositionEvaluationBatch Evaluate(in EncodedPositionWithHistory encodedPosition, bool fillInHistory, bool retrieveSupplementalResults)
    {
      return Evaluate(new EncodedPositionWithHistory[] { encodedPosition }, 1, fillInHistory, retrieveSupplementalResults);
    }


    /// <summary>
    /// Helper method to evaluates batch originating from array of EncodedPosition.
    /// </summary>
    /// <param name="encodedPositions"></param>
    /// <param name="numPositions"></param>
    /// <param name="fillInHistoryPlanes"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public IPositionEvaluationBatch Evaluate(EncodedPositionWithHistory[] encodedPositions, int numPositions, bool fillInHistoryPlanes, bool retrieveSupplementalResults)
    {
      if (InputsRequired.HasFlag(InputTypes.LastMovePlies))
      {
        // TODO: it should be possible to extract some of the history
        //       from the encodedPositions into the LastMovePlies to support this case.
        //       Probably leverage existing method SetMoveSinceFromPositions to do this.
        throw new NotImplementedException();
      }

      EncodedPositionBatchFlat batch;
      if (InputsRequired > InputTypes.Boards)
      {
        EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(numPositions, InputsRequired | InputTypes.Positions);
        for (int i = 0; i < numPositions; i++)
        {
          builder.Add(in encodedPositions[i], fillInHistoryPlanes);
        }
        batch = builder.GetBatch();
      }
      else
      {
        bool setPositions = InputsRequired.HasFlag(InputTypes.Positions);
        batch = new EncodedPositionBatchFlat(encodedPositions, numPositions, setPositions, fillInHistoryPlanes);        
      }

      if (EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS)
      {
        batch.PositionsBuffer = encodedPositions;
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
    public IPositionEvaluationBatch Evaluate(Span<EncodedTrainingPosition> encodedPositions, int numPositions, bool fillInHistoryPlanes, bool retrieveSupplementalResults = false)
    {
      throw new Exception("Method needs retesting. Also the fillInHistoryPlanes is not completely implemented below.");

      if (InputsRequired.HasFlag(InputTypes.LastMovePlies))
      {
        // TODO: it should be possible to extract some of the history
        //       from the encodedPositions into the LastMovePlies to support this case.
        //       Probably leverage existing method SetMoveSinceFromPositions to do this.
        throw new NotImplementedException();
      }

      EncodedPositionBatchFlat batch;

      if (InputsRequired > InputTypes.Boards)
      {
        EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(numPositions, InputsRequired);
        for (int i = 0; i < numPositions; i++)
        {
          // Unmirror before adding.
          builder.Add(encodedPositions[i].PositionWithBoards, fillInHistoryPlanes);
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
    /// The batch is broken into sub-batches, evaluated, and each item in the sub-batch is passed to a provided delegate.
    /// </summary>
    public void EvaluateOversizedBatch(EncodedPositionBatchFlat bigBatch, Action<int, Memory<NNEvaluatorResult>> processor)
    {
      bool needsToBeSplit = bigBatch.NumPos > MaxBatchSize;

      int numProcessed = 0;
      int numToProcess = bigBatch.NumPos;

      // Repeatedly process sub-batches no larger than the MaxBatchSize.
      while (numProcessed < numToProcess)
      {
        int numRemaining = numToProcess - numProcessed;
        int numThisBatch = Math.Min(MaxBatchSize, numRemaining);

        // Extract a slice of manageable size.
        IEncodedPositionBatchFlat slice = needsToBeSplit ? new EncodedPositionBatchFlatSlice(bigBatch, numProcessed, numThisBatch)
                                                         : bigBatch;

        // Evaluate with the neural network.
        // TODO: for efficiency reasons could we use EvaluateBatchIntoBuffers instead?
        NNEvaluatorResult[] result = EvaluateBatch(slice);

        // Pass sub-batch to delegate.
        processor(numProcessed, new Memory<NNEvaluatorResult>(result, 0, numThisBatch));

        numProcessed += numThisBatch;
      }
    }


    /// <summary>
    /// Evaluates all positions in an oversized batch (that cannot be evaluated all at once).
    /// The batch is broken into sub-batches, evaluated, and each sub-batch is passed to a provided delegate.
    /// </summary>
    /// <param name="bigBatch"></param>
    /// <param name="processor"></param>
    public void EvaluateOversizedBatch(EncodedPositionBatchFlat bigBatch, Action<int, NNEvaluatorResult> processor)
    {
      int numProcessed = 0;
      EvaluateOversizedBatch(bigBatch, (int index, Memory<NNEvaluatorResult> results) =>
      {
        Span<NNEvaluatorResult> resultsSpan = results.Span;
        for (int i = 0; i < results.Length; i++)
        {
          processor(numProcessed++, resultsSpan[i]);
        }
      });
    }


    /// <summary>
    /// Performs quick benchmarks on evaluator to determine performance
    /// includng single positions and batches and optionally estimated 
    /// batch size breaks (cut points beyond which speed drops due to batching effects).
    /// </summary>
    /// <param name="computeBreaks"></param>
    /// <param name="maxSeconds"></param>
    public virtual void CalcStatistics(bool computeBreaks, float maxSeconds = 1.0f)
    {
      (float npsSingletons, float npsBigBatch, int[] breaks) = NNEvaluatorBenchmark.EstNPS(this, computeBreaks);
      PerformanceStats = new NNEvaluatorPerformanceStats()
      {
        EvaluatorType = GetType(),
        SingletonNPS = npsSingletons,
        BigBatchNPS = npsBigBatch,
        Breaks = breaks
      };
    }


    #region Optional Async support

    /// <summary>
    /// Starts a task to begin evalation of specified positions on device.
    /// </summary>
    /// <param name="positions">positions to be evaluated, or null indicating the evaluator has already been configured with the positions</param>
    /// <param name="numPositions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    public Task LaunchEvaluateBatchAsync(IEncodedPositionBatchFlat positions, int numPositions, bool retrieveSupplementalResults = false)
    {
      if (positions != null && positions.NumPos != numPositions)
      {
        throw new ArgumentException("numPositions wrong size");
      }

      if (positions != null)
      {
        SetMovesIfNeeded(positions);
      }

      Task task = DoLaunchEvaluateBatchAsync(positions, retrieveSupplementalResults);

      NumBatchesEvaluated++;
      NumPositionsEvaluated += numPositions;
      return task;
    }

    protected virtual Task DoLaunchEvaluateBatchAsync(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      throw new NotImplementedException();
    }

    public virtual IPositionEvaluationBatch GetLastAsyncBatchResult(IEncodedPositionBatchFlat positions, short[] numMoves, short[] moveIndices, bool retrieveSupplementalResults, bool returnResultsCopy)
    {
      throw new NotImplementedException();
    }

    #endregion


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
