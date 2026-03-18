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
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.DataTypes;
using Ceres.Base.Threading;
using Ceres.Chess;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.PositionEvalCaching;
using Ceres.Chess.Positions;
using Ceres.MCGS.Graphs;
using Ceres.MCGS.Graphs.GNodes;
using Ceres.MCGS.Search.Coordination;
using Ceres.MCGS.Search.Params;
using Ceres.MCGS.Search.PathEvaluators;
using Ceres.MCGS.Search.Paths;

#endregion

namespace Ceres.MCGS.Search.Phases.Evaluation;

/// <summary>
/// Base class for an selection terminator which performs inference using a neural network.
/// </summary>
public sealed class MCGSEvaluatorNeuralNet : IDisposable
{
  public static bool DEBUGGING_PLY_SINCE_LAST_MOVE = false;


  /// <summary>
  ///
  /// </summary>
  public NNEvaluatorDef EvaluatorDef { init; get; }


  /// <summary>
  /// If the NN evaluation should run with lower priority
  /// </summary>
  public bool LowPriority { get; }


  /// <summary>
  /// Optional temperature to be applied to value WDL.
  /// </summary>
  public float ValueTemperature { get; set; }



  /// <summary>
  /// Reusable batch for NN evaluations
  /// </summary>
  public EncodedPositionBatchFlat Batch { private set; get; }

  public enum LocationType { Local, Remote };

  public readonly bool EnableState;

  public readonly bool FillInHistory = false;

  public NNEvaluator Evaluator { init; get; }

  public readonly PositionEvalCache Cache;

  public readonly bool EngineIsWhite;

  /// <summary>
  /// Optional Func that dynamically determines the index of possibly 
  /// multiple valuators which should be used for the current batch.
  /// TODO: possibly remove this functionality, possibly subsumed.
  /// </summary>
  public Func<object /*MCTSIterator*/, int> BatchEvaluatorIndexDynamicSelector { init; get; }

  // State for deferred result retrieval
  private MCGSEngine deferredEngine;
  private ListBounded<MCGSPath> deferredPaths;
  private IPositionEvaluationBatch deferredResult;
  private (NNEvaluatorResult[][] evalResults, MGMoveList[] moveLists) deferredActualEvalsAllPositionsAllMoves;


  /// <summary>
  /// Constructor for a NN evaluator (either local or remote) with specified parameters.
  /// </summary>
  /// <param name="evaluatorDef"></param>
  /// <param name="evaluator"></param>
  /// <param name="fillInHistory"></param>
  /// <param name="maxBatchSize"></param>
  /// <param name="lowPriority"></param>
  /// <param name="valueTemperature"></param>
  /// <param name="cache"></param>
  /// <param name="batchEvaluatorIndexDynamicSelector"></param>
  public MCGSEvaluatorNeuralNet(NNEvaluatorDef evaluatorDef,
                                NNEvaluator evaluator,
                                ItemsInBucketsAllocator nnDeviceAllocator,
                                bool fillInHistory,
                                int maxBatchSize,
                                bool lowPriority,
                                float valueTemperature,
                                bool enableState,
                                PositionEvalCache cache,
                                Func<object, int> batchEvaluatorIndexDynamicSelector,
                                bool engineIsWhite)
  {
    rawPosArray = posArrayPool.Rent(maxBatchSize);

    EvaluatorDef = evaluatorDef;
    FillInHistory = fillInHistory;
    LowPriority = lowPriority;
    EnableState = enableState;
    ValueTemperature = valueTemperature;
    Cache = cache;
    BatchEvaluatorIndexDynamicSelector = batchEvaluatorIndexDynamicSelector;
    EngineIsWhite = engineIsWhite;

    if (evaluatorDef.Location == NNEvaluatorDef.LocationType.Local)
    {
      Evaluator = evaluator;

      // Ask for rented policy buffers, whic greatly reduces memory allocations
      // but requires us to call Dispose on returned batches when done.
      evaluator.UseRentedPolicyBuffer = true;
    }
    else
    {
      throw new NotImplementedException();
    }

    if (nnDeviceAllocator != null)
    {
      (Evaluator as NNEvaluatorSplit).SetAllocator(nnDeviceAllocator);
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
      const int INITAL_BATCH_SIZE = 16;
      int newSize = currentSize == 0 ? Math.Max(INITAL_BATCH_SIZE, batchSize)
                                     : Math.Max(batchSize, currentSize * 2);

      Batch = new EncodedPositionBatchFlat(EncodedPositionType.PositionOnly, newSize);
    }
  }



  EncodedPositionWithHistory[] rawPosArray;

  // TO DO: make uses elsewhere of ArrayPool use the shared
  static readonly ArrayPool<EncodedPositionWithHistory> posArrayPool = ArrayPool<EncodedPositionWithHistory>.Shared;

  readonly Lock shutdownLockObj = new();
  private bool disposed = false;

  /// <summary>
  /// Shuts down the evaluator, releasing resources.
  /// </summary>
  public void Shutdown()
  {
    Dispose();
  }

  /// <summary>
  /// Disposes the evaluator, releasing resources.
  /// </summary>
  public void Dispose()
  {
    if (disposed)
    {
      return;
    }

    lock (shutdownLockObj)
    {
      if (!disposed)
      {
        if (rawPosArray != null)
        {
          posArrayPool.Return(rawPosArray);
          rawPosArray = null;
        }

        Batch?.Shutdown();
        disposed = true;
      }
    }

    GC.SuppressFinalize(this);
  }

  ~MCGSEvaluatorNeuralNet()
  {
    Dispose();
  }

  public static long CountTotalBatches = 0;
  public static long CountTotalPositions = 0;


  /// <summary>
  /// Sets the Batch field with set of positions coming from a list of MCGSPath.
  /// </summary>
  /// <param name="paths"></param>
  /// <returns></returns>
  /// <exception cref="Exception"></exception>
  /// <exception cref="NotImplementedException"></exception>
  int SetBatch(MCGSEngine engine, ListBounded<MCGSPath> paths)
  {
    if (paths.Count > Evaluator.MaxBatchSize)
    {
      throw new Exception("Batch size exceeds maximum size of evaluator.");
    }

    if (paths.Count > 0)
    {
      CountTotalBatches++;
      CountTotalPositions += paths.Count;

      VerifyBatchAllocated(paths.Count);      

      if (engine.NeedsPlySinceLastMove && (Batch.LastMovePlies == null || Batch.LastMovePlies.Length < paths.Count * 64))
      {
        Batch.LastMovePlies = new byte[Batch.MaxBatchSize * 64];
      }

      if (EvaluatorDef.PositionTransform == NNEvaluatorDef.PositionTransformType.Mirror)
      {
        throw new NotImplementedException("Mirroring temporarily disabled.");
        //rawPosArray[i] = rawPosArray[i].Mirrored;
      }

      const int NUM_ITEMS_PER_THREAD = 48;
      ParallelOptions parallelOptions = ParallelUtils.ParallelOptions(paths.Count, NUM_ITEMS_PER_THREAD);

      Parallel.For(0, paths.Count, parallelOptions, delegate (int i)
      {
        SetEncodedBoardPositionFromPath(paths[i], ref rawPosArray[i], FillInHistory);

        if (engine.NeedsPlySinceLastMove)
        {
          if (MCGSEvaluatorNeuralNet.DEBUGGING_PLY_SINCE_LAST_MOVE)
          {
            Console.WriteLine("<PLY_SINCE_DEBUG> Dumping ply since last move for path " + i);
//            paths[i].DumpLocalVisits();
            foreach (Position pos in paths[i].LeafPositionWithHistory.Positions)
            {
              Console.WriteLine(pos);
            }
            paths[i].PlySinceLastMove.Dump(paths[i].LeafPositionWithHistory);
          }
          ValidatePlySinceLastMoveIncremental(paths[i]);
          Span<byte> targetSlice = new Span<byte>(Batch.LastMovePlies, i * 64, 64);
          ReadOnlySpan<byte> srcPlySince = paths[i].PlySinceLastMove.SquarePlySince;

          // PlySince is maintained in absolute coordinates (A1=0).
          // The NN expects values in the side-to-move's perspective.
          // For Black-to-move positions, flip with ^56.
          if (paths[i].LeafVisitRef.ChildPosition.BlackToMove)
          {
            for (int s = 0; s < 64; s++)
            {
              targetSlice[s] = srcPlySince[s ^ 56];
            }
          }
          else
          {
            srcPlySince.CopyTo(targetSlice);
          }
        }
      });


      if (EvaluatorDef.Location == NNEvaluatorDef.LocationType.Local)
      {
        const bool FILL_EMPTY_PLANES = false; // planes assumed already filled out
        Debug.Assert(!rawPosArray[0].HistoryPositionIsEmpty(EncodedPositionBatchFlat.NUM_HISTORY_POSITIONS - 1));

        const bool SET_POSITIONS = false; // we assume this is already done (if needed)
        Batch.Set(rawPosArray, paths.Count, SET_POSITIONS, fillInHistoryPlanes: FILL_EMPTY_PLANES);
        //          Batch.States = states;
      }

      if (BatchEvaluatorIndexDynamicSelector != null)
      {
        throw new NotImplementedException();
        //Batch.PreferredEvaluatorIndex = (short)BatchEvaluatorIndexDynamicSelector(context);
      }
    }

    return paths.Count;
  }


  /// <summary>
  /// Retrieves the results from a NN evaluation batch and applies them to the specified paths.
  /// </summary>
  /// <param name="engine"></param>
  /// <param name="paths"></param>
  /// <param name="results"></param>
  /// <param name="actualEvalsAllPositionsAllMoves">optional (research mode) NN evaluations of all moves from all positions</param>
  void RetrieveResults(MCGSEngine engine, ListBounded<MCGSPath> paths,
                       IPositionEvaluationBatch results,
                       (NNEvaluatorResult[][] evalResults, MGMoveList[] moveLists) actualEvalsAllPositionsAllMoves = default)
  {
    Parallel.For(0,
      paths.Count,
      new ParallelOptions() { MaxDegreeOfParallelism = ParallelUtils.CalcMaxParallelism(paths.Count, 24) },
      i =>
      {
        // TODO: make this cleaner/faster
        MCGSPath path = paths[i];
        GNode node = path.IsRootInitializationPath ? path.Engine.SearchRootNode
                                                   : path.LeafVisitRef.ParentChildEdge.ChildNode;

        FP16 winP;
        FP16 lossP;
        FP16 rawM = results.HasM ? results.GetM(i) : FP16.NaN;
        FP16 rawUncertaintyV = results.HasUncertaintyV ? FP16.Max(0, (FP16)(float)results.GetUncertaintyV(i)) : FP16.NaN;
        FP16 rawUncertaintyP = results.HasUncertaintyP ? FP16.Max(0, (FP16)(float)results.GetUncertaintyP(i)) : FP16.NaN;

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

        if (ValueTemperature != 1)
        {
          // TODO: Consider removing this.
          //       It is inefficient, and seems to overlap with functionality now in NNEvaluator.
          float temperature = ValueTemperature;
          (float winPRaw, float drawPRaw, float lossPRaw) = (winP, Math.Max(0, 1 - winP - lossP), lossP);
          (float winPRawLogit, float drawPRawLogit, float lossPRawLogit) = (MathF.Log(winPRaw) / temperature, MathF.Log(drawPRaw) / temperature, MathF.Log(lossPRaw) / temperature);
          (float winPAdj, float drawPAdj, float lossPAdj) = (MathF.Exp(winPRawLogit), MathF.Exp(drawPRawLogit), MathF.Exp(lossPRawLogit));
          float sum = winPAdj + drawPAdj + lossPAdj;

#if NOT
        (float winPRaw, float lossPRaw) = (winP, lossP);
        (float winPRawLogit, float lossPRawLogit) = (MathF.Log(winPRaw) / temperature,  MathF.Log(lossPRaw) / temperature);
        (float winPAdj, float lossPAdj) = (MathF.Exp(winPRawLogit),  MathF.Exp(lossPRawLogit));
        float sum = winPAdj + lossPAdj;
#endif

          winP = (FP16)(winPAdj / sum);
          lossP = (FP16)(lossPAdj / sum);
        }


        (Memory<CompressedPolicyVector> policies, int index) policyInfo;
        Memory<CompressedActionVector> actionArray = default;
        Half[] state = default;

        policyInfo = results.GetPolicy(i);

        if (results.HasAction)
        {
          actionArray = results.GetAction(i).actions;

          // If we have actualEvalsAllPositionsAllMoves,
          // use these actual values replace values in actionArray
          // with true (lookahead) values. Used only for testing/research.
          if (actualEvalsAllPositionsAllMoves != default)
          {
            CompressedActionVector av = new();
            for (int m = 0; m < actualEvalsAllPositionsAllMoves.evalResults[i].Length; m++)
            {
              // The moves coming back are in arbitrary order.
              // But we need to place the action value into the action array,
              // which in a generally different order (same as policy).
              // Therefore need to lookup the index of the move in the target policy array.
              MGMove thisMGMove = actualEvalsAllPositionsAllMoves.moveLists[i].MovesArray[m];
              EncodedMove thisEncodedMove = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(thisMGMove);
              int indexThisMoveInPolicyVector = policyInfo.policies.Span[i].IndexOfMove(thisEncodedMove);
              Debug.Assert(indexThisMoveInPolicyVector >= 0, "Move not found in policy vector.");

              // Note that W and L are reversed to invert this to be
              // from player to move perspective (consistent with value head convention).
              av[indexThisMoveInPolicyVector] = ((Half)actualEvalsAllPositionsAllMoves.evalResults[i][m].L,
                                                 (Half)actualEvalsAllPositionsAllMoves.evalResults[i][m].W);
            }
            actionArray.Span[i] = av;
          }
        }

        if (results.HasState && EnableState)
        {
          state = results.GetState(i);
        }

        // Compute FortressP if ply-bin move outputs are available.
        FP16 fortressP = FP16.NaN;
        if (results.HasPlyBinOutputs)
        {
          ReadOnlySpan<Half> captureProbsSpan = results.GetPlyBinCaptureProbs(i);
          if (!captureProbsSpan.IsEmpty)
          {
            ref MCGSPathVisit leafVisit = ref path.LeafVisitRef;
            float fortressPFloat = NNEvaluatorResult.ComputeFortressP(captureProbsSpan, in leafVisit.ChildPosition, !node.IsWhite);
            fortressP = (FP16)fortressPFloat;
          }
        }

        // TODO: make side determination faster
        // (short)policyInfo.index policyInfo.policies, actionArray, 
        SelectTerminationInfo evalResult = new(node.IsWhite ? SideType.White : SideType.Black,
                                               MCGSPathTerminationReason.PendingNeuralNetEval, GameResult.Unknown,
                                               winP, lossP, rawM, rawUncertaintyV, rawUncertaintyP, state, fortressP);

        GNode leafNode = path.IsRootInitializationPath ? engine.SearchRootNode : path.LeafNode;
        Debug.Assert(!leafNode.IsEvaluated);
        ref MCGSPathVisit refLeafVisit = ref path.LeafVisitRef;

        using (new NodeLockBlock(leafNode))
        {
          engine.ApplyNodeEvaluationValues(leafNode, in refLeafVisit.ChildPosition, refLeafVisit.Moves,
                                           policyInfo.index,
                                           policyInfo.policies, actionArray,
                                           in evalResult);

          if ((engine.Manager.ParamsSearch.MoveOrderingPhase == ParamsSearch.MoveOrderingPhaseEnum.NodeInitialization
           || engine.Manager.ParamsSearch.MoveOrderingPhase == ParamsSearch.MoveOrderingPhaseEnum.NodeInitializationAndChildSelect)
        && leafNode.NumPolicyMoves > 1)
          {
            if (leafNode.NumPolicyMoves > 2)
            {
              const int MAX_LOOK_RIGHT = 5;
              leafNode.CheckMoveOrderRearrangeAtIndex(in refLeafVisit.ChildPosition, 1, 2 + MAX_LOOK_RIGHT, MCGSParamsFixed.MOVE_ORDERING_MIN_RATIO_POLICY);
            }
            leafNode.CheckMoveOrderRearrangeAtIndex(in refLeafVisit.ChildPosition, 0, 1, MCGSParamsFixed.MOVE_ORDERING_MIN_RATIO_POLICY);
          }

#if NOT
        const bool DUMP = false;
        if (DUMP && leafNode.HashStandalone.GetHashCode() % 1999 == 233)
        {
          bool haveOutput = false;
          for (int ix = 0; ix < leafNode.NumPolicyMoves; ix++)
          {
            GNode lookupBestNode = leafNode.MaxSiblingNodeAfterChildMove( in refLeafVisit.ChildPosition, ix);
            if (!lookupBestNode.IsNull)
            {
              if (!haveOutput)
              {
                Console.WriteLine("\r\n#" + leafNode.Index + " " + + leafNode.V 
                                + " Best sibling nodes for children:" + " " + refLeafVisit.ChildPosition.ToPosition.FEN);
                haveOutput = true;
              }
              Console.WriteLine(ix + " " + Math.Round(100*leafNode.EdgeHeadersSpan[ix].P )+ "% " + lookupBestNode);
            }
          }
          if (!haveOutput) Console.WriteLine("none");
        }
#endif
        }

        paths[i].TerminationInfo = evalResult;
      });
  }


  void RunLocal(MCGSEngine engine, ListBounded<MCGSPath> paths, bool deferRetrieveResults = false)
  {
    const bool RETRIEVE_SUPPLEMENTAL = false;

    NNEvaluator evaluator = Evaluator;


    IPositionEvaluationBatch result;
    if (evaluator.InputsRequired > NNEvaluator.InputTypes.Boards)
    {
      bool hasPositions = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Positions);
      bool hasHashes = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Hashes);
      bool hasMoves = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.Moves);
      bool hasLastMovePlies = evaluator.InputsRequired.HasFlag(NNEvaluator.InputTypes.LastMovePlies);
      bool hasState = evaluator.HasState;

      hasLastMovePlies = engine.NeedsPlySinceLastMove;

      if (hasPositions && Batch.Positions == null)
      {
        Batch.Positions = new MGPosition[Batch.MaxBatchSize];
      }

      if (hasHashes && Batch.PositionHashes == null)
      {
        Batch.PositionHashes = new ulong[Batch.MaxBatchSize];
      }

      if (hasLastMovePlies && Batch.LastMovePlies == null)
      {
        Batch.LastMovePlies = new byte[Batch.MaxBatchSize * 64];
      }

      if (hasMoves && Batch.Moves == null)
      {
        Batch.Moves = new MGMoveList[Batch.MaxBatchSize];
      }

      // Get the iterator handy (will be same for all paths).
      MCGSIterator loop = paths[0].Iterator;

      for (int i = 0; i < paths.Count; i++)
      {
        GNode node = paths[i].LeafVisitRef.ChildNode;
        ref MCGSPathVisit leafPathNode = ref paths[i].LeafVisitRef;

        if (hasPositions)
        {
          Batch.Positions[i] = leafPathNode.ChildPosition;
        }

        if (hasState && EnableState && !node.IsGraphRoot)
        {
          Half[] priorStateInfo = null;
          throw new NotImplementedException();
          //          if (Batch.States != null) Batch.States[i] = null; // default assumption


          //if (FP16.IsNaN(node.NodeRef.ActionV))
          //{
          //  throw new Exception("Action must be enabled when using prior state feature");
          //}

          // Because the state information is used during training only to
          // propagate information from a parent to its optimal (or nearly optimal) child,
          // only apply the state if this is the case.
          const float THRESHOLD_POLICY_USE_STATE = 0.03f;// 0.03f;
          throw new NotImplementedException();
#if NOT
            if (float.IsNaN((float)node.NodeRef.ActionV)) throw new Exception("Action value required if state enabled.");

            float actionDiff =  MathF.Abs((float)node.NodeRef.ActionV - node.VisitsFromNodes.FirstOrDefault().NodeRef.V);
            float policyDiff = MathF.Abs((float)node.VisitsFromNodes.FirstOrDefault()[0].P - node.NodeRef.P);
//if (!node.IsRoot) Console.WriteLine(actionDiff + " " + (actionDiff < THRESHOLD_ACTION_USE_HISTORY));
//            if (actionDiff < THRESHOLD_ACTION_USE_HISTORY)
            if (policyDiff <= THRESHOLD_POLICY_USE_STATE) // was 0.01
            {
              // TODO: is there a way to pick a non-arbitrary parent (possibly one of many)
              GPosition aParent = node.VisitsFromNodes.FirstOrDefault();
              priorStateInfo = node.Graph.Store.AllStateVectors[aParent.Index.Index];
              if (DEBUG_MODE) Console.WriteLine(priorStateInfo[0] + " ~~~~ BATCH USE STATE " + " " + node);

              if (false)
              {
                NNEvaluatorResult priorEval = localEvaluator.Evaluate(paths[i].VisitedNodes[paths[i].NumNodesInPath - 2].Pos);
                Half[] priorState = priorEval.PriorState;
                Console.WriteLine("corel statexx" + " "+  StatUtils.Correlation(MemoryMarshal.Cast<Half, FP16>(priorState),
                                                        MemoryMarshal.Cast<Half, FP16>(priorStateInfo)));
              }
            }
            else
            {
              if (DEBUG_MODE) Console.WriteLine("~~~~ DO NOT SET STATE " + node);
            }
#endif
          if (Batch.States == null)
          {
            Batch.States = new Half[Batch.MaxBatchSize][];
          }

          if (priorStateInfo == null)
          {
            if (Batch.States[i] != null)
            {
              Array.Clear(Batch.States[i]);
            }
          }
          else
          {
            // TODO: It seems making a copy here is essential because buffer is reused
            //       Not doing this results in terrible performance. Verify.
            Half[] copy = new Half[priorStateInfo.Length];
            Array.Copy(priorStateInfo, copy, priorStateInfo.Length);
            Batch.States[i] = copy;

            // Batch.States[i] = priorStateInfo *** don't do this (see above)
          }
        }

        if (hasHashes)
        {
          Batch.PositionHashes[i] = node.NodeRef.HashStandalone.Hash;
        }

        if (hasMoves)
        {
          Batch.Moves[i] = leafPathNode.Moves;
        }

        if (hasLastMovePlies)
        {
          // Ply-since-last-move data is populated in SetBatch.
        }
      }
    }

    // Note that we call EvaluateBatchIntoBuffers instead of EvaluateBatch for performance reasons
    // (we immediately extract from buffers in RetrieveResults below)
    Batch.EngineIsWhite = EngineIsWhite;
    result = evaluator.EvaluateIntoBuffers(Batch, RETRIEVE_SUPPLEMENTAL);
    Debug.Assert(!FP16.IsNaN(result.GetWinP(0)) && !FP16.IsNaN(result.GetLossP(0)));

    const bool LOOKAHEAD_COMPUTE_TRUE_ACTION_VALUES = false; // Only for research/testing purposes. Very slow!
    NNEvaluatorResult[][] preEvalsAllPositionsAllMoves = null;
    MGMoveList[] moveLists = null;
    if (LOOKAHEAD_COMPUTE_TRUE_ACTION_VALUES)
    {
      // Evaluate all moves from all positions (slow).
      preEvalsAllPositionsAllMoves = new NNEvaluatorResult[Batch.MaxBatchSize][];
      moveLists = new MGMoveList[Batch.MaxBatchSize];
      for (int i = 0; i < Batch.NumPos; i++)
      {
        MCGSManager.BestValueMove(evaluator, new PositionWithHistory(Batch.Positions[i].ToPosition),
                                  ref moveLists[i], ref preEvalsAllPositionsAllMoves[i], true, false);
      }
    }

    if (deferRetrieveResults)
    {
      // Store state for deferred retrieval
      deferredEngine = engine;
      deferredPaths = paths;
      deferredResult = result;
      deferredActualEvalsAllPositionsAllMoves = (preEvalsAllPositionsAllMoves, moveLists);
    }
    else
    {
      RetrieveResults(engine, paths, result, (preEvalsAllPositionsAllMoves, moveLists));
      result.Dispose();
    }
  }

  /// <summary>
  /// Retrieves results from a previously deferred evaluation.
  /// Must be called after RunLocal with deferRetrieveResults=true.
  /// </summary>
  public void RetrieveDeferredResults()
  {
    if (deferredEngine == null || deferredPaths == null || deferredResult == null)
    {
      throw new InvalidOperationException("No deferred results available. Call RunLocal with deferRetrieveResults=true first.");
    }

    RetrieveResults(deferredEngine, deferredPaths, deferredResult, deferredActualEvalsAllPositionsAllMoves);

    // Clear deferred state
    deferredEngine = null;
    deferredPaths = null;
    deferredResult?.Dispose();
    deferredResult = null;
    deferredActualEvalsAllPositionsAllMoves = default;
  }

  public static long TOTAL_NUM_NN_EVALS = 0;

  public void BatchGenerate(MCGSEngine engine, ListBounded<MCGSPath> paths, bool deferRetrieveResults = false)
  {
    Interlocked.Add(ref TOTAL_NUM_NN_EVALS, paths.Count);

#if DEBUG
    CheckDuplicatedLeafs(paths);
#endif

    Debug.Assert(EvaluatorDef.Location != NNEvaluatorDef.LocationType.Remote);

    SetBatch(engine, paths);
    RunLocal(engine, paths, deferRetrieveResults);
  }


  private void CheckDuplicatedLeafs(ListBounded<MCGSPath> paths)
  {
    HashSet<GNode> seenPositions = new();
    foreach (MCGSPath path in paths)
    {
      if (!path.IsRootInitializationPath
        && path.TerminationReason == MCGSPathTerminationReason.PendingNeuralNetEval)
      {
        if (seenPositions.Contains(path.LeafNode))
        {
          throw new Exception("Duplicate leaf node in batch. " + path.LeafNode);
        }
        seenPositions.Add(path.LeafNode);
      }
    }
  }


  public void WaitDone()
  {
    throw new NotImplementedException();
  }



  [ThreadStatic]
  static Position[] positionsBuffer;

  [ThreadStatic]
  static MGPosition[] positionsBufferMG;

  /// <summary>
  /// Initializes a specified EncodedPosition to reflect the a specified node's position.
  /// </summary>
  /// <param name="path"></param>
  /// <param name="boardsHistory"></param>
  public unsafe void SetEncodedBoardPositionFromPath(MCGSPath path, ref EncodedPositionWithHistory boardsHistory, bool historyFillIn)
  {
    PositionWithHistory RootPreHistory = path.Graph.Store.PositionHistory;

    // Make sure the ThreadStatic temporary buffer used for positions is populated.
    positionsBufferMG ??= new MGPosition[EncodedPositionBatchFlat.NUM_HISTORY_POSITIONS];

    // An alternate design of writing a local method that returns IEnumerable<Position>
    // and just yield returning the Positions in the right order was considered
    // to possibly improve speed.
    // However ultimately the SetFromSequentialPositions method uses ref semantics
    // into the Position[] to avoid copies, so this alternate design would 
    // ultimately be a wash because the copy of Position would merely be moved 
    // from one place to another.

    // Last position in prehistory should be same as root position in path (also tree).
    // TODO: Someday remove the ".FEN" so we compare full positions, including make sure repetition count same).

    Debug.Assert(RootPreHistory.Count >= 1);

    int numUsablePositionsPath = path.IsRootInitializationPath ? 0 : path.NumVisitsInPath;
    int numUsablePositionsSearchRootToGraphRoot = path.Engine.SearchRootPathFromGraphRoot == null
                                                    ? 0
                                                    : path.Engine.SearchRootPathFromGraphRoot.Length;
    int numUsablePreHistoryPositions = RootPreHistory.Count;

    int numUsablePositions = Math.Min(EncodedPositionBatchFlat.NUM_HISTORY_POSITIONS, numUsablePositionsPath
                                                                                    + numUsablePositionsSearchRootToGraphRoot
                                                                                    + numUsablePreHistoryPositions);

    Span<MGPosition> positionsPopulatedMG = new(positionsBufferMG, 0, numUsablePositions);
    int nextTargetIndex = numUsablePositions - 1;

    // Populate first using sequential positions from the path in reverse order.
    if (numUsablePositionsPath > 0)
    {
      foreach (MCGSPathVisitMember visitPair in path.PathVisitsLastBackedUpToRoot)
      {
        if (nextTargetIndex < 0)
        {
          break;
        }

        Debug.Assert(visitPair.PathVisitRef.ChildPosition != default);

        positionsPopulatedMG[nextTargetIndex--] = visitPair.PathVisitRef.ChildPosition;
      }
    }

    // .............................................
    // Process all nodes (if any) from the search root node up to the graph root node
    for (int i = numUsablePositionsSearchRootToGraphRoot - 1; i >= 0; i--)
    {
      if (nextTargetIndex < 0)
      {
        break;
      }

      ref readonly GraphRootToSearchRootNodeInfo nodeFromGraphRootToSearchRoot = ref path.Engine.SearchRootPathFromGraphRoot[i];
      // TODO: restore the second stronger assertion once the "casling reset move 50" problem is fixed
      //       (castling should not reset, one of the paths below false resets it).
      // problem if cycle found, disable: Debug.Assert(nodeFromGraphRootToSearchRoot.ChildPosMG.ToPosition.PiecesEqual(nodeFromGraphRootToSearchRoot.ChildNode.CalcPosition().ToPosition));
      //Debug.Assert(nodeFromGraphRootToSearchRoot.ChildNode.CalcPosition() == nodeFromGraphRootToSearchRoot.ChildPosMG);

      positionsPopulatedMG[nextTargetIndex--] = nodeFromGraphRootToSearchRoot.ChildPosMG;
    }

    // .............................................

    // Fill in the rest of the planes from prehistory (if available).
    for (int i = path.Graph.Store.HistoryHashes.PriorPositionsMG.Length - 1; i >= 0; i--)
    {
      if (nextTargetIndex < 0)
      {
        break;
      }
      positionsPopulatedMG[nextTargetIndex--] = path.Graph.Store.HistoryHashes.PriorPositionsMG[i];
    }

#if DEBUG
    // Validate that the history positions are all valid consecutive positions.
    for (int i = 1; i < numUsablePositions; i++)
    {
      if (positionsPopulatedMG[i].B % 27 == 4) // run this test only infrequently because slow
      {
        bool ok = MoveBetweenPositions(positionsPopulatedMG[i - 1].ToPosition, positionsPopulatedMG[i].ToPosition) != default;
        if (!ok)
        {
          Console.WriteLine("Error: positions not consecutive: " + positionsPopulatedMG[i - 1].ToPosition.FEN + " -> " + positionsPopulatedMG[i].ToPosition.FEN);
          for (int j = 0; j < numUsablePositions; j++)
          {
            Console.WriteLine(positionsPopulatedMG[j].ToPosition.FEN);
          }
          Debug.Assert(false);
        }
      }
    }
#endif

    // Set repetition count
    // TODO: The span collected above is only of length 8 (max)
    //       instead we should look back over path to starting position.
    //      PositionRepetitionCalc.SetRepetitionsCount(positionsPopulated);
    //  fix   Span<MGPosition> rootPreHistoryNotAlreadyUsed = new Span<Position>(RootPreHistory.Positions, firstHistoryPosToUse, numHistoryPositionsToUse);
    //  fix      Span<MGPosition> positionsPopulatedNotFromPreHistory = positionsPopulated.Slice(numHistoryPositionsToUse);
    //  fix     SetRepetitionsCount(rootPreHistoryNotAlreadyUsed, positionsPopulatedNotFromPreHistory);

    // Fill in boards history with the gathered positions
    boardsHistory.SetFromSequentialPositions(positionsPopulatedMG, historyFillIn);
  }




  /// <summary>
  /// Validates that the incrementally maintained PlySinceLastMove matches
  /// a from-scratch recomputation by walking the path root-to-leaf.
  /// </summary>
  [Conditional("DEBUG")]
  static void ValidatePlySinceLastMoveIncremental(MCGSPath path)
  {
    if (path.PlySinceLastMove == null || path.IsRootInitializationPath)
    {
      return;
    }

    byte[] rootPlySince = path.Engine.SearchRootPlySinceLastMove;

    // Cannot validate depth-1 paths: the root visit's move is not captured
    // in the PathVisitsLeafToRoot enumeration (IsRoot skips recording the move),
    // so the general recomputation below would exit at numMoves <= 0.
    if (path.NumVisitsInPath <= 1)
    {
      return;
    }

    // Collect positions and encoded moves from leaf to root.
    // Every visit (including root) has a valid ParentChildEdge set by the select phase.
    const int MAX_DEPTH = 256;
    Span<MGPosition> positions = stackalloc MGPosition[MAX_DEPTH + 1];
    Span<Chess.EncodedPositions.Basic.EncodedMove> moves = stackalloc Chess.EncodedPositions.Basic.EncodedMove[MAX_DEPTH];
    int numMoves = 0;

    foreach (MCGSPathVisitMember visitMember in path.PathVisitsLeafToRoot)
    {
      if (numMoves >= MAX_DEPTH)
      {
        break;
      }

      ref readonly MCGSPathVisit visit = ref visitMember.PathVisitRef;
      positions[numMoves] = visit.ChildPosition;
      moves[numMoves] = visit.ParentChildEdge.Move;
      numMoves++;
    }

    if (numMoves <= 0)
    {
      return;
    }

    // The parent position for the root visit is the search root position.
    positions[numMoves] = path.Engine.SearchRootPosMG;

    // Replay root-to-leaf using the same algorithm as the incremental code.
    byte[] recomputed = new byte[64];
    byte[] temp = new byte[64];
    rootPlySince.AsSpan().CopyTo(recomputed);

    for (int k = numMoves - 1; k >= 0; k--)
    {
      MGPosition parentPos = positions[k + 1];
      MGMove mgMove = Chess.MoveGen.Converters.ConverterMGMoveEncodedMove.EncodedMoveToMGChessMove(moves[k], in parentPos);
      PlySinceLastMoveArray.ApplyMoveWithSwap(ref recomputed, ref temp, in mgMove);
    }

    for (int s = 0; s < 64; s++)
    {
      Debug.Assert(path.PlySinceLastMove.SquarePlySince[s] == recomputed[s],
        $"PlySince mismatch at square {s}: incremental={path.PlySinceLastMove.SquarePlySince[s]}, recomputed={recomputed[s]}, path={path}");
    }

    if (DEBUGGING_PLY_SINCE_LAST_MOVE)
    {
      Console.WriteLine("<PLY_SINCE_DEBUG> MCGSEvaluatorNeuralNet.ValidatePlySinceLastMoveIncremental passes!");
    }
  }


  /// <summary>
  /// Updates the repetition count for each position in the specified span based on prior positions.
  /// </summary>
  static void SetRepetitionsCount(ReadOnlySpan<Position> priorPositions, Span<Position> positionsToUpdate)
  {
    // TODO: For efficiency, this hash set could be computed once and stored in the Graph node store.
    HashSet<PosHash64> hashesOfPositionsSeen = [];
    foreach (Position pos in priorPositions)
    {
      PosHash64 hash = MGPositionHashing.Hash64(pos.ToMGPosition);
      hashesOfPositionsSeen.Add(hash);
    }

    for (int i = 0; i < positionsToUpdate.Length; i++)
    {
      // TODO: Should we also be calling EqualAsRepetition to confirm the hash hit?
      //if (thisPos.EqualAsRepetition(in positionsToUpdate[k]))

      PosHash64 thisHash = MGPositionHashing.Hash64(positionsToUpdate[i].ToMGPosition);
      bool positionAlreadyExisted = hashesOfPositionsSeen.Contains(thisHash);
      positionsToUpdate[i].MiscInfo.SetRepetitionCount(positionAlreadyExisted ? 1 : 0);

      hashesOfPositionsSeen.Add(thisHash);
    }
  }


  // TODO: 1. Centralize somewhere in Ceres.Chess
  //       2. Leverage this in the existing methods MoveBetweenHistory in TPGRecord and EncodedTrainingPosition classes
  public static MGMove MoveBetweenPositions(in Position posFirst, in Position posSecond)
  {
    MGMoveList moves = new();
    MGMoveGen.GenerateMoves(posFirst.ToMGPosition, moves);

    // Iterate thru legal moves to find a move
    // such that after applying it to priorPos, we get curPos.
    for (int i = 0; i < moves.NumMovesUsed; i++)
    {
      Position newPos = posFirst.AfterMove(MGMoveConverter.ToMove(moves.MovesArray[i]));
      if (newPos.PiecesEqual(posSecond))
      {
        return moves.MovesArray[i];
      }
    }

    return default;
  }


  /// <summary>
  /// Returns a string description of this evaluator.
  /// </summary>
  /// <returns></returns>
  public override string ToString()
  {
    return $"<MCGSEvaluatorNeuralNet Def={EvaluatorDef}, VTemp={ValueTemperature}>";
  }
}
