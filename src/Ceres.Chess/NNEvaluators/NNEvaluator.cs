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
using System.Diagnostics;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.DataTypes;

using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
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
    /// <summary>
    /// Currently hardcoded value for the per-square dimension of the prior state information, 
    /// if the network has a state output. Linked to TPGRecord.SIZE_STATE_PER_SQUARE.
    /// </summary>
    public const int SIZE_STATE_PER_SQUARE = 4;

    [Flags]
    public enum InputTypes
    {
      Undefined = 0,
      Boards = 1,
      Hashes = 2,
      Moves = 4,
      Positions = 8,
      LastMovePlies = 16,
      State = 32,

      All = Boards | Hashes | Moves | Positions | State,
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
    /// String identifier of the underlying engine network.
    /// </summary>
    public string EngineNetworkID;

    public NNEvaluatorOptions options = new NNEvaluatorOptions();

    /// <summary>
    /// Optional options relating to evaluator (e.g. output head postprocessing).
    /// </summary>
    public NNEvaluatorOptions Options
    {
      get => options;
      set
      {
        if (value == null)
        {
          throw new ArgumentNullException(nameof(value));
        }
        else
        {
          options = value;
        }
      }
    }

    /// <summary>
    /// Optional short identification string.
    /// </summary>
    public string ShortID;

    /// <summary>
    /// Optional description string.
    /// </summary>
    public string Description;

    /// <summary>
    /// Estimated performance characteristics.
    /// </summary>
    public NNEvaluatorPerformanceStats PerformanceStats;

    /// <summary>
    /// Optional buffers synchronization object.
    /// If not null, the base evaluator will acquire this before DoEvaluateIntoBuffers.
    /// The client is thereafter responsible for releasing the lock when the internal buffers
    /// are no longer needed.
    /// </summary>
    public SemaphoreSlim BuffersLock;

    /// <summary>
    /// If the network returns policy moves in the same order
    /// as the legal MGMoveList.
    /// </summary>
    public virtual bool PolicyReturnedSameOrderMoveList => false;

    /// <summary>
    /// Optional list of head overrides for the evaluator.
    /// </summary>
    public NNEvaluatorHeadOverride[] HeadOverrides;

    /// <summary>
    /// Optional Action that is invoked with:
    ///   - object (typically a tree object) ]
    ///   - bool "searchDone":
    ///     - if true, then is the call after a search is about to be performed
    ///     - if false, then the call is before a search is performed
    /// </summary>
    public Action<object, bool> RetrainFunc;

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
    /// If the evaluator has an UP (uncertainty of policy) head.
    /// </summary>
    public abstract bool HasUncertaintyP { get; }

    /// <summary>
    /// If the evaluator has an secondary value head.
    /// </summary>
    public abstract bool HasValueSecondary { get; }

    /// <summary>
    /// Optional contextual information to be potentially used 
    /// as supplemental input for the evaluation of children.
    /// </summary>
    public virtual bool HasState { get; protected set; } = false;

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
    /// If history planes should be zeroed out before evaluation.
    /// </summary>
    public bool ZeroHistoryPlanes = false;

    /// <summary>
    /// Miscellaneous information about the evaluator.
    /// </summary>
    public virtual EvaluatorInfo Info => null;

    /// <summary>
    /// If the raw neural network outputs should be retained.
    /// Note that this may be memory intensive.
    /// </summary>
    public virtual bool RetainRawOutputs { get; set; } = false;

    /// <summary>
    /// Array of names of raw neural network outputs (if RetainRawOutputs is true).
    /// </summary>
    public string[] RawNetworkOutputNames;



    #region Static helpers

    /// Returns an NNEvaluator corresponding to specified strings with network and device specifications.
    /// </summary>
    /// <param name="netSpecificationString"></param>
    /// <param name="deviceSpecificationString"></param>
    /// <param name="evaluatorOptions"></param>
    /// <returns></returns>
    public static NNEvaluator FromSpecification(string netSpecificationString,
                                                string deviceSpecificationString,
                                                NNEvaluatorOptions evaluatorOptions = null)
      => NNEvaluatorDef.FromSpecification(netSpecificationString, deviceSpecificationString, evaluatorOptions).ToEvaluator();

    #endregion


    #region Basic evaluator methods

    /// <summary>
    /// Worker method that evaluates batch of positions into the internal buffers.
    /// </summary>
    /// <param name="positions"></param>
    /// <param name="retrieveSupplementalResults"></param>
    /// <returns></returns>
    protected abstract IPositionEvaluationBatch DoEvaluateIntoBuffers(IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false);


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

      if (ZeroHistoryPlanes)
      {
        positions.ZeroHistoryPlanes();
      }

      // Acquire the buffers lock, if any.
      BuffersLock?.Wait();

      // Do the actual subclass evaluation.
      IPositionEvaluationBatch batchEvaluation = DoEvaluateIntoBuffers(positions, retrieveSupplementalResults);

      NumBatchesEvaluated++;
      NumPositionsEvaluated += positions.NumPos;

      if (Options.PVExtensionDepth > 0)
      {
        PositionEvaluationBatch parentEvaluations = batchEvaluation as PositionEvaluationBatch;

        //        Func<int, float> priorityScore = index => parentEvaluations.UncertaintyV.Span[index];
        const float V_CUTOFF = 0.9f;
        Func<int, float> priorityScore = index => MathF.Abs((float)parentEvaluations.GetV(index)) < V_CUTOFF ? 1 : -1;

        if (extensionEvaluator == null)
        {
          extensionEvaluator = NNEvaluator.FromSpecification("~BT4_FP16_TRT", "GPU:0", Options with { PVExtensionDepth = 0 });
        }
        NNEvaluator extensionEvaluatorToUse = extensionEvaluator;

        ChildBatchEvaluator evaluator = new ChildBatchEvaluator(extensionEvaluatorToUse, parentEvaluations, positions, priorityScore);

        // Recursively evaluate the child batches.
        // This call will update the parentEvaluations with improvements propagated from deeper levels.
        int numToEvaluate = int.MaxValue;// Math.Max(1, parentEvaluations.NumPos / 2);
        const float MIN_PRIORITY = 1;
        const float FRACTION_CHILD = 0.666f;
        evaluator.EvaluateRecursive(Options.PVExtensionDepth, numToEvaluate, MIN_PRIORITY, FRACTION_CHILD);
      }

      return batchEvaluation;
    }
    NNEvaluator extensionEvaluator;


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

    readonly object lockObj = new();


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
    /// Extracts and returns a single NNEvaluatorResult 
    /// from an IPositionEvaluationBatch (at a specified batch index).
    /// </summary>
    /// <param name="batch"></param>
    /// <param name="batchIndex"></param>
    /// <returns></returns>
    public NNEvaluatorResult ResultFromBatch(IPositionEvaluationBatch batch, int batchIndex)
    {

      ExtractToNNEvaluatorResult(out NNEvaluatorResult resultBoard, batch, batchIndex);
      return resultBoard;
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

      float w1 = batch.GetWin1P(batchIndex);
      float l1 = IsWDL ? batch.GetLoss1P(batchIndex) : float.NaN;

      float w2 = HasValueSecondary ? batch.GetWin2P(batchIndex) : float.NaN;
      float l2 = HasValueSecondary && IsWDL ? batch.GetLoss2P(batchIndex) : float.NaN;

      float m = HasM ? batch.GetM(batchIndex) : float.NaN;
      float uncertaintyV = HasUncertaintyV ? batch.GetUncertaintyV(batchIndex) : float.NaN;
      float uncertaintyP = HasUncertaintyP ? batch.GetUncertaintyP(batchIndex) : float.NaN;

      NNEvaluatorResultActivations activations = batch.GetActivations(batchIndex);
      Half[] stateInfo = HasState ? batch.GetState(batchIndex) : null;

      (Memory<CompressedPolicyVector> policies, int index) policyRef = batch.GetPolicy(batchIndex);
      (Memory<CompressedActionVector> actions, int index) actionRef = HasAction ? batch.GetAction(batchIndex) : default;

      FP16 extraStat0 = batch.GetExtraStat0(batchIndex);
      FP16 extraStat1 = batch.GetExtraStat1(batchIndex);

      // Note that support for rawNetworkOutputs is currently incomplete.
      // TODO: Improve this, add to the interface, and implement in implementors.
      FP16[][] rawNetworkOutputs = null;
      if (batch is PositionEvaluationBatch)
      {
        rawNetworkOutputs = new FP16[batch.NumPos][];
        PositionEvaluationBatch peb = (PositionEvaluationBatch)batch;
        if (!peb.RawNetworkOutputs.IsEmpty && peb.RawNetworkOutputs.Length > batchIndex)
        {
          rawNetworkOutputs = peb.RawNetworkOutputs.Span[batchIndex];
        }
      }

      result = new NNEvaluatorResult(w, l, w1, l1, w2, l2, m, uncertaintyV, uncertaintyP,
                                     policyRef.policies.Span[policyRef.index],
                                     HasAction ? actionRef.actions.Span[actionRef.index] : default,
                                     activations, stateInfo, extraStat0, extraStat1,
                                     rawNetworkOutputs, RawNetworkOutputNames);
    }

    #endregion

    #region Helper methods

    /// <summary>
    /// Performs any initialization to prepare evaluator for delay-free execution.
    /// </summary>
    public virtual void Warmup()
    {

    }


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
                                      InputTypes extraInputs = InputTypes.Undefined,
                                      Half[] state = null)
    {
      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(1, InputsRequired | extraInputs);
      builder.Add(position, fillInMissingPlanes, state);

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
    public NNEvaluatorResult[] Evaluate(IEnumerable<PositionWithHistory> positions,
                                        bool fillInMissingPlanes = true,
                                        bool retrieveSupplementalResults = false)
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
    public NNEvaluatorResult Evaluate(in Position position,
                                      bool fillInMissingPlanes = true,
                                      bool retrieveSupplementalResults = false,
                                      Half[] state = null)
    {
      InputTypes types = InputsRequired | InputTypes.Positions;
      if (state != null)
      {
        types |= InputTypes.State;
      }

      EncodedPositionBatchBuilder builder = new EncodedPositionBatchBuilder(1, types);
      builder.Add(in position, fillInMissingPlanes, state);

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
    public IPositionEvaluationBatch Evaluate(in EncodedPositionWithHistory encodedPosition,
                                             bool fillInHistory,
                                             bool retrieveSupplementalResults)
    {
      return Evaluate([encodedPosition], 1, fillInHistory, retrieveSupplementalResults);
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
    /// <param name="bigBatch"></param>
    /// <param name="processor"></param>
    /// <param name="overrideMaxBatchSize"></param>
    public void EvaluateOversizedBatch(EncodedPositionBatchFlat bigBatch,
                                       Action<int, Memory<NNEvaluatorResult>> processor,
                                       int? overrideMaxBatchSize = null)
    {
      int batchSizeToUse = overrideMaxBatchSize ?? MaxBatchSize;
      bool needsToBeSplit = bigBatch.NumPos > batchSizeToUse;

      int numProcessed = 0;
      int numToProcess = bigBatch.NumPos;

      // Repeatedly process sub-batches no larger than the specified maximum batch size.
      while (numProcessed < numToProcess)
      {
        int numRemaining = numToProcess - numProcessed;
        int numThisBatch = Math.Min(batchSizeToUse, numRemaining);

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

      if (ZeroHistoryPlanes)
      {
        positions.ZeroHistoryPlanes();
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


    public class ChildBatchEvaluator
    {
      NNEvaluator baseEvaluator;

      private readonly PositionEvaluationBatch _parentEvaluations;
      private readonly IEncodedPositionBatchFlat _positions;
      private readonly Func<int, float> _priorityScore;
      private List<int> _selectedIndices;
      private EncodedPositionBatchBuilder _batchBuilder;
      private IPositionEvaluationBatch _childEvaluations;

      public ChildBatchEvaluator(NNEvaluator baseEvaluator,
                                 PositionEvaluationBatch parentEvaluations,
                                 IEncodedPositionBatchFlat positions,
                                 Func<int, float> priorityScore)
      {
        this.baseEvaluator = baseEvaluator;
        _parentEvaluations = parentEvaluations;
        _positions = positions;
        _priorityScore = priorityScore;
        _selectedIndices = new List<int>();
        // EncodedPositionBatchFlat.RETAIN_POSITIONS_INTERNALS = true;
      }

      /// <summary>
      /// Prepares a child batch from the parent's positions. Only those indices that have a
      /// priority score above minPriorityScore are considered, and if there are too many,
      /// only the top maxPositions (by descending priority) are selected.
      /// </summary>
      public int BuildChildBatch(int maxPositions = int.MaxValue, float minPriorityScore = float.MinValue)
      {
        // Gather candidate indices based on the priority function.
        List<(int index, float score)> candidates = new();
        for (int i = 0; i < _parentEvaluations.NumPos; i++)
        {
          if (_parentEvaluations.Policies.Span[i].Count > 0)
          {
            float score = _priorityScore(i);
            if (score >= minPriorityScore)
            {
              candidates.Add((i, score));
            }
          }
        }

        // Select top candidates by priority (if there are more than maxPositions)
        _selectedIndices = candidates
            .OrderByDescending(c => c.score)
            .Take(maxPositions)
            .Select(c => c.index)
            .ToList();

        // Build the child batch for the selected indices.
        _batchBuilder = new EncodedPositionBatchBuilder(_positions.NumPos, InputTypes.All);
        foreach (int i in _selectedIndices)
        {
          // Convert the parent's position to a PositionWithHistory.
          PositionWithHistory parentPWH = _positions.PositionsBuffer.Span[i].ToPositionWithHistory(8);
          CompressedPolicyVector childPolicy = _parentEvaluations.Policies.Span[i];

          (EncodedMove Move, float Probability) thisPolicyMove = childPolicy.PolicyInfoAtIndex(0);

          if (false)
          {
            if (childPolicy.Count > 1)
            {
              (EncodedMove Move, float Probability) thisPolicyMove1 = childPolicy.PolicyInfoAtIndex(1);
              float diff = thisPolicyMove.Probability - thisPolicyMove1.Probability;
              if (diff < 0.05f)
              {
                thisPolicyMove = thisPolicyMove1;
              }
            }
          }

          // Convert the encoded move into an MGMove.
          MGMove thisMove = ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(thisPolicyMove.Move, parentPWH.FinalPosMG);

          // Determine the child position after the move.
          MGPosition childPos = parentPWH.FinalPosMG;
          childPos.MakeMove(thisMove);

          // Construct a child PositionWithHistory.
          PositionWithHistory childPWH = new PositionWithHistory(parentPWH);
          childPWH.AppendMove(thisMove);
          childPWH.ForceFinalPosMG(childPos);

          // Add the computed child position to the batch.
          _batchBuilder.Add(childPWH);
        }

        return _selectedIndices.Count;
      }

      /// <summary>
      /// Evaluates the built child batch.
      /// </summary>
      public void EvaluateChildBatch()
      {
        // Evaluate the child batch using an evaluation function.
        // (Assumes DoEvaluateIntoBuffers is accessible in the context.)
        EncodedPositionBatchFlat batch = _batchBuilder.GetBatch();
        if (batch.NumPos == 0)
        {
          throw new Exception("Child batch is empty.");
        }
        _childEvaluations = baseEvaluator.DoEvaluateIntoBuffers(batch, false);
      }


      /// <summary>
      /// Applies the child evaluations back to the parent evaluations.
      /// Only updates positions that were selected for child evaluation.
      /// </summary>
      /// <param name="fractionChild">Fraction to blend the child evaluation with the parent's.</param>
      public void ApplyChildBatch(float fractionChild)
      {
        // Iterate through the child evaluations.
        for (int j = 0; j < _selectedIndices.Count; j++)
        {
          int parentIndex = _selectedIndices[j];

          _parentEvaluations.W.Span[parentIndex] = (FP16)((1.0f - fractionChild) * _parentEvaluations.W.Span[parentIndex]
              + fractionChild * _childEvaluations.GetLossP(j)); // Using LossP due to perspective change.
          _parentEvaluations.L.Span[parentIndex] = (FP16)((1.0f - fractionChild) * _parentEvaluations.L.Span[parentIndex]
              + fractionChild * _childEvaluations.GetWinP(j)); // Using WinP due to perspective change.
        }
      }

      /// <summary>
      /// Recursively evaluates child batches down to the specified depth.
      /// For depth = 1, this is equivalent to a single-level evaluation.
      /// For depth > 1, the evaluator recursively builds, evaluates, and updates the child batches.
      /// </summary>
      /// <param name="depth">Depth level for recursive evaluation (>= 1)</param>
      /// <param name="maxPositions">Optional limit on positions per batch</param>
      /// <param name="minPriorityScore">Optional minimum priority score to consider</param>
      /// <param name="fractionChild">Fraction to blend the child evaluation with the parent's</param>
      public void EvaluateRecursive(int depth, int maxPositions = int.MaxValue, float minPriorityScore = float.MinValue, float fractionChild = 0.5f)
      {
        // Build and evaluate the child batch at the current level.
        int numChildren = BuildChildBatch(maxPositions, minPriorityScore);
        if (numChildren > 0)
        {
          EvaluateChildBatch();

          // If deeper recursion is requested, recursively evaluate the child batch.
          if (depth > 1)
          {
            IEncodedPositionBatchFlat childPositions = _batchBuilder.GetBatch();
            if (_childEvaluations is PositionEvaluationBatch childEvaluations)
            {
              ChildBatchEvaluator deeperEvaluator = new ChildBatchEvaluator(baseEvaluator, childEvaluations, childPositions, _priorityScore);

              // Continue evaluation recursively (all nodes).
              const int MAX_POSITIONS_RECURSIVE = int.MaxValue;
              const float MIN_PRIORITY_SCORE_RECURSIVE = float.MaxValue;
              deeperEvaluator.EvaluateRecursive(depth - 1, MAX_POSITIONS_RECURSIVE, MIN_PRIORITY_SCORE_RECURSIVE, fractionChild);
            }

            // Otherwise, if the cast fails, we simply proceed with applying the evaluations.
          }

          // Finally, update the parent evaluations with the (possibly recursively updated) child evaluations.
          ApplyChildBatch(fractionChild);
        }
      }
    }


  }
}
