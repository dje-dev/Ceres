#region License notice

/*
  This file is part of the Ceres project at https://github.com/dje-dev/ceres.
  Copyright (C) 2020- by David Elliott and the Ceres Authors.

  Ceres is free software under the terms of the GNU General Public License v3.0.
  You should have received a copy of the GNU General Public License
  along with Ceres. If not, see <http://www.gnu.org/licenses/>.
*/

#endregion

#region Using Directives

using System;
using System.Diagnostics;
using System.Buffers;

using Ceres.Base.DataTypes;

using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.MoveGen;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.LC0.Batches;

using Ceres.MCTS.MTCSNodes;
using Ceres.MCTS.Iteration;
using Ceres.MCTS.Environment;

#endregion

namespace Ceres.MCTS.Evaluators
{
  /// <summary>
  /// Base class for an evaluator which performs inference using a neural network.
  /// </summary>
  public sealed class LeafEvaluatorNN : LeafEvaluatorBase
  {
    /// <summary>
    ///
    /// </summary>
    public NNEvaluatorDef EvaluatorDef;

    /// <summary>
    /// If retrieved results should be saved to the cache
    /// </summary>
    public bool SaveToCache { get; }

    /// <summary>
    /// If the NN evaluation should run with lower priority
    /// </summary>
    public bool LowPriority { get; }

    /// <summary>
    /// Reusable batch for NN evaluations
    /// </summary>
    public EncodedPositionBatchFlat Batch { private set; get; }

    public enum LocationType { Local, Remote };

    /// <summary>
    /// If the secondary evaluator should be used for evaluation.
    /// </summary>
    public bool EvaluateUsingSecondaryEvaluator = false;

    NNEvaluator localEvaluator;
    NNEvaluator localEvaluatorSecondary;

    public readonly PositionEvalCache Cache;


    /// <summary>
    /// Optional Func that dynamically determines the index of possibly 
    /// multiple valuators which should be used for the current batch.
    /// TODO: possibly remove this functionality, possibly subsumed.
    /// </summary>
    public readonly Func<MCTSIterator, int> BatchEvaluatorIndexDynamicSelector;


    /// <summary>
    /// Constructor for a NN evaluator (either local or remote) with specified parameters.
    /// </summary>
    /// <param name="paramsNN"></param>
    /// <param name="saveToCache"></param>
    /// <param name="instanceID"></param>
    /// <param name="lowPriority"></param>
    public LeafEvaluatorNN(NNEvaluatorDef evaluatorDef, NNEvaluator evaluator,
                           bool saveToCache,
                           bool lowPriority,
                           PositionEvalCache cache,
                           Func<MCTSIterator, int> batchEvaluatorIndexDynamicSelector,
                           NNEvaluator evaluatorSecondary)
    {
      rawPosArray = posArrayPool.Rent(evaluator.MaxBatchSize);

      EvaluatorDef = evaluatorDef;
      SaveToCache = saveToCache;
      LowPriority = lowPriority;
      Cache = cache;
      BatchEvaluatorIndexDynamicSelector = batchEvaluatorIndexDynamicSelector;

      if (evaluatorDef.Location == NNEvaluatorDef.LocationType.Local)
      {
        localEvaluator = evaluator;// isEvaluator1 ? Params.Evaluator1 : Params.Evaluator2;
        localEvaluatorSecondary = evaluatorSecondary;
      }
      else
      {
        throw new NotImplementedException();
      }

      // TODO: auto-estimate performance
#if SOMEDAY
      for (int i = 0; i < 10; i++)
      {
//        using (new TimingBlock("benchmark"))
        {
            float[] splits = WFEvalNetBenchmark.GetBigBatchNPSFractions(((WFEvalNetCompound)localEvaluator).Evaluators);
            Console.WriteLine(splits[0] + " " + splits[1] + " " + splits[2] + " " + splits[3]);
          (float estNPSSingletons, float estNPSBigBatch) = WFEvalNetBenchmark.EstNPS(localEvaluator);
          Console.WriteLine(estNPSSingletons + " " + estNPSBigBatch);
        }
      }
#endif
    }


    /// <summary>
    /// Verifies that Batch is allocated and has sufficient size.
    /// The batch is grown incrementally if/as needed starting from a modest size.
    /// </summary>
    /// <param name="batchSize"></param>
    void VerifyBatchAllocated(int batchSize)
    {    
      int currentSize = Batch == null ? 0 : Batch.MaxBatchSize;
      if (batchSize > currentSize)
      {
        // Double the current size to make room
        const int INITAL_BATCH_SIZE = 128;
        int newSize = currentSize == 0 ? Math.Max(INITAL_BATCH_SIZE, batchSize) 
                                       : Math.Max(batchSize, currentSize * 2);

        Batch = new EncodedPositionBatchFlat(EncodedPositionType.PositionOnly, newSize);
      }
    }


    protected override LeafEvaluationResult DoTryEvaluate(MCTSNode node) => default;

    EncodedPositionWithHistory[] rawPosArray;

    // TO DO: make uses elsewhere of ArrayPool use the shared
    static ArrayPool<EncodedPositionWithHistory> posArrayPool = ArrayPool<EncodedPositionWithHistory>.Shared;

    readonly object shutdownLockObj = new();
    public void Shutdown()
    {
      if (rawPosArray != null)
      {
        lock (shutdownLockObj)
        {
          if (rawPosArray != null)
          {
            posArrayPool.Return(rawPosArray);
            rawPosArray = null;
            Batch?.Shutdown();
          }
        }
      }
    }

    ~LeafEvaluatorNN()
    {
      Shutdown();
    }


    /// <summary>
    /// Sets the Batch field with set of positions coming from a specified Span<MCTSNode>.
    /// </summary>
    /// <param name="context"></param>
    /// <param name="nodes"></param>
    /// <returns></returns>
    int SetBatch(MCTSIterator context, Span<MCTSNode> nodes)
    {
      if (nodes.Length > 0)
      {
        VerifyBatchAllocated(nodes.Length);

        for (int i = 0; i < nodes.Length; i++)
        {
          nodes[i].Annotate();
          nodes[i].Annotation.CalcRawPosition(nodes[i], ref rawPosArray[i]);

          if (EvaluatorDef.PositionTransform == NNEvaluatorDef.PositionTransformType.Mirror)
          {
            rawPosArray[i] = rawPosArray[i].Mirrored;
          }
        }

        if (EvaluatorDef.Location == NNEvaluatorDef.LocationType.Local)
        {
          const bool SET_POSITIONS = false; // we assume this is already done (if needed)
          Batch.Set(rawPosArray, nodes.Length, SET_POSITIONS);
        }

        if (BatchEvaluatorIndexDynamicSelector != null)
        {
          Batch.PreferredEvaluatorIndex = (short)BatchEvaluatorIndexDynamicSelector(context);
        }
      }

      return nodes.Length;
    }

    public enum EvalResultTarget { PrimaryEvalResult, SecondaryEvalResult };


    /// <summary>
    /// 
    /// TODO: this method similar to one below, try to unify them
    /// </summary>
    /// <param name="nodes"></param>
    /// <param name="results"></param>
    void RetrieveResults(Span<MCTSNode> nodes, IPositionEvaluationBatch results, EvalResultTarget resultTarget, bool markSecondaryNN)
    {
      //if (EvaluateUsingSecondaryEvaluator) Console.WriteLine("early batch " + nodes.Length);

      for (int i = 0; i < nodes.Length; i++)
      {
        MCTSNode node = nodes[i];

        if (markSecondaryNN)
        {
          node.StructRef.SecondaryNN = true;
        }

        FP16 winP;
        FP16 lossP;
        FP16 rawM = results.GetM(i);

        // Copy WinP
        FP16 rawWinP = results.GetWinP(i);
        Debug.Assert(!float.IsNaN(rawWinP));

        // Copy LossP
        FP16 rawLossP = results.GetLossP(i);
        Debug.Assert(!float.IsNaN(rawLossP));

        // Assign win and loss probabilities
        // If they look like non-WDL result, try to rewrite them
        // in equivalent way that avoids negative probabilities
        if (rawLossP == 0 && rawWinP < 0)
        {
          winP = 0;
          lossP = -rawWinP;
        }
        else
        {
          winP = rawWinP;
          lossP = rawLossP;
        }

        LeafEvaluationResult evalResult = new LeafEvaluationResult(GameResult.Unknown, winP, lossP, rawM);
        evalResult.PolicyInArray = results.GetPolicy(i);

        // Copy policy
        if (resultTarget == EvalResultTarget.PrimaryEvalResult)
        {
          node.EvalResult = evalResult;
        }
        else if (resultTarget == EvalResultTarget.SecondaryEvalResult)
        {
          // Currently primary and secondary eval results share same memory
          node.EvalResult = evalResult;
        }
        else
        {
          throw new Exception("Internal error: unexpected EvalResultTarget");
        }

        // Save back to cache
        if (SaveToCache) Cache.Store(node.StructRef.ZobristHash,
                                     GameResult.Unknown, rawWinP, rawLossP, rawM,
                                     in node.EvalResult.PolicyRef);
      }
    }


    void RunLocal(Span<MCTSNode> nodes, EvalResultTarget resultTarget)
    {
      const bool RETRIEVE_SUPPLEMENTAL = false;

      NNEvaluator evaluator = localEvaluator;
      bool useSecondary = EvaluateUsingSecondaryEvaluator;
      if (useSecondary)
      {
        evaluator = localEvaluatorSecondary;
        MCTSManager.NumSecondaryEvaluations += nodes.Length;
        MCTSManager.NumSecondaryBatches++;
      }

      IPositionEvaluationBatch result;
      if (evaluator.InputsRequired > NNEvaluator.InputTypes.Boards)
      {
        bool hasPositions = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Positions);
        bool hasHashes = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Hashes);
        bool hasMoves = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Moves);

        if (hasPositions && Batch.Positions == null)
        {
          Batch.Positions = new MGPosition[Batch.MaxBatchSize];
        }

        if (hasHashes && Batch.PositionHashes == null)
        {
          Batch.PositionHashes = new ulong[Batch.MaxBatchSize];
        }

        if (hasMoves && Batch.Moves == null)
        {
          Batch.Moves = new MGMoveList[Batch.MaxBatchSize];
        }

        for (int i = 0; i < nodes.Length; i++)
        {
          MCTSNode node = nodes[i];

          if (hasPositions)
          {
            Batch.Positions[i] = node.Annotation.PosMG;
          }

          if (hasHashes)
          {
            Batch.PositionHashes[i] = node.StructRef.ZobristHash;
          }

          if (hasMoves)
          {
            Batch.Moves[i] = node.Annotation.Moves;
          }
        }
      }

      // Note that we call EvaluateBatchIntoBuffers instead of EvaluateBatch for performance reasons
      // (we immediately extract from buffers in RetrieveResults below)
      result = evaluator.EvaluateIntoBuffers(Batch, RETRIEVE_SUPPLEMENTAL);
      Debug.Assert(!FP16.IsNaN(result.GetWinP(0)) && !FP16.IsNaN(result.GetLossP(0)));

      bool markSecondaryNN = EvaluateUsingSecondaryEvaluator
                          || resultTarget == EvalResultTarget.SecondaryEvalResult
                          || Batch.PositionsUseSecondaryEvaluator;
      RetrieveResults(nodes, result, resultTarget, markSecondaryNN);
    }


    public void BatchGenerate(MCTSIterator context, Span<MCTSNode> nodes, EvalResultTarget resultTarget)
    {
      try
      {
        if (EvaluatorDef.Location == NNEvaluatorDef.LocationType.Remote)
        {
          throw new NotImplementedException();
          //SetBatch(context, nodes);
          //RunRemote(nodes, resultTarget);
        }
        else
        {
          SetBatch(context, nodes);
          RunLocal(nodes, resultTarget);
        }
      }
      catch (Exception exc)
      {
        Console.WriteLine("Error in NodeEvaluatorNN " + exc);
        throw exc;
      }
    }


    public void WaitDone()
    {
      throw new NotImplementedException();
    }

  }
}
