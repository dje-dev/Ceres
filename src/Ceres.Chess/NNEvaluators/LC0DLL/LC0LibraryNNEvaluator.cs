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
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading;
using System.Threading.Tasks;

using Ceres.Base.DataTypes;
using Ceres.Base.OperatingSystem;
using Ceres.Base.Threading;
using Ceres.Chess.EncodedPositions.Basic;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.UserSettings;
using static Ceres.Chess.NNEvaluators.LC0DLL.LCO_Interop;

#endregion

namespace Ceres.Chess.NNEvaluators.Internals
{
  /// <summary>
  /// Interface code to process neural network evaluations by 
  /// calling set of functions via interop in the LC0 library.
  /// </summary>
  public class LC0LibraryNNEvaluator : IDisposable
  {
    /// <summary>
    /// Fixed address at which input buffer is located (pinned).
    /// </summary>
    IntPtr ptrBlockInItems;

    /// <summary>
    /// Fixed address at which output buffer is located (pinned).
    /// </summary>
    IntPtr ptrBlockOutItems;

    internal Span<CeresTransferBlockInItem> ItemsIn => ItemsInStruct;
    internal Span<CeresTransferBlockOutItem> ItemsOut => ItemsOutStruct;

    unsafe private Span<CeresTransferBlockInItem> ItemsInSpan => new Span<CeresTransferBlockInItem>((void*)ptrBlockInItems, MAX_POSITIONS_PER_BATCH);

    unsafe private Span<CeresTransferBlockOutItem> ItemsOutSpan => new Span<CeresTransferBlockOutItem>((void*)ptrBlockOutItems, MAX_POSITIONS_PER_BATCH);

    /// <summary>
    /// The shared data structure used to pass inputs to the LC0 evaluator.
    /// </summary>
    private CeresTransferBlockInItem[] ItemsInStruct;

    /// <summary>
    /// The shared data structure used to receive outputs from the LC0 evaluator.
    /// </summary>
    private CeresTransferBlockOutItem[] ItemsOutStruct;


    /// <summary>
    /// Each evaluator is associated with a unique session ID,
    /// drawn from this static pool .
    /// </summary>
    static IDPool sessionIDPool = new IDPool("LC0DllNNEvaluator", MAX_SESSIONS);


    #region Constructor/Destructor

    /// <summary>
    /// The session ID associated with this evaluator (shared with the library).
    /// </summary>
    public readonly int SessionID;

    /// <summary>
    /// The neural network loaded by this evaluator.
    /// </summary>
    public readonly string NetworkFilename;

    /// <summary>
    /// The ID of the GPU with which this evaluator is associated.
    /// </summary>
    public readonly int GPUID;


    /// <summary>
    /// Constructor that initializes an evaluator to use a specified GPU with a specified network file.
    /// </summary>
    /// <param name="networkFilename"></param>
    /// <param name="gpuID"></param>
    public LC0LibraryNNEvaluator(string networkFilename, int gpuID)
    {
      if (gpuID < 0 || gpuID >= NNEvaluatorStats.MAX_GPUS) throw new ArgumentOutOfRangeException(nameof(gpuID));

      CheckLibraryDirectoryOnPath();

      SessionID = sessionIDPool.GetFreeID();
      GPUID = gpuID;
      NetworkFilename = networkFilename;

      AllocErrorCode allocError = Alloc(SessionID, networkFilename, gpuID);
      if (allocError != AllocErrorCode.NO_ERROR)
        throw new Exception($"Error returned by LC0 library in call to Alloc: {allocError}");

      // Allocate arrays pinned so we can pass the pointer to the DLL
      // TODO: would it be better to used fixed each time we call out to DLL?
      ItemsInStruct = GC.AllocateArray<CeresTransferBlockInItem>(MAX_POSITIONS_PER_BATCH, pinned:true);
      ItemsOutStruct = GC.AllocateArray<CeresTransferBlockOutItem>(MAX_POSITIONS_PER_BATCH, pinned:true);

      unsafe
      {
        fixed (CeresTransferBlockInItem* inputs = &ItemsInStruct[0])
        fixed (CeresTransferBlockOutItem* outputs = &ItemsOutStruct[0])
        {
          ptrBlockInItems = (IntPtr)inputs;
          ptrBlockOutItems = (IntPtr)outputs;
        }
      }
    }


    #endregion

    #region Request processing

    readonly object lockObj = new ();

    /// <summary>
    /// Processes a single request by calling Compute interop function.
    /// </summary>
    /// <param name="numPos"></param>
    unsafe void ProcessRequest(int numPos)
    {
      if (numPos > MAX_POSITIONS_PER_BATCH)
      {
        throw new ArgumentOutOfRangeException($"Internal error: numPos {numPos} exceeds limit of {MAX_POSITIONS_PER_BATCH}");
      }

      // Insert an invalid value for output so we can detect if not properly processed
      ItemsOut[0].Q = float.NaN;

      // Execute the request
      Compute(SessionID, numPos, (CeresTransferBlockInItem*)ptrBlockInItems, (CeresTransferBlockOutItem*)ptrBlockOutItems);

      if (float.IsNaN(ItemsOut[0].Q))
      {
        throw new Exception($"Internal error: LC0 DLL failed to process request");
      }

      NNEvaluatorStats.UpdateStatsForBatch(GPUID, numPos);
    }


    /// <summary>
    /// Coordinates evaluation of a batch, encoding positions in
    /// data structures expected by the LC0 functions
    /// </summary>
    /// <param name="batch"></param>
    /// <param name="pos"></param>
    internal unsafe void EvaluateNN(IEncodedPositionBatchFlat batch, Span<MGPosition> pos)
    {
      if (batch.NumPos > MAX_POSITIONS_PER_BATCH)
        throw new ArgumentOutOfRangeException($"batch.NumPos is too large, max {MAX_POSITIONS_PER_BATCH} versus actual {batch.NumPos}");

      lock (lockObj)
      {
        ParallelOptions parallelOptions = ParallelUtils.ParallelOptions(batch.NumPos, 192);
        Parallel.For(0, batch.NumPos, parallelOptions, delegate (int i)
        {
          ref CeresTransferBlockInItem refItem = ref ItemsIn[i];

          // Determine legal move list
          MGMoveList movesLegal = batch.Moves[i];

          // Note that rarely there might be more legal moves than we can fit in our buffer;
          // in this case we just silently ignore some
          // TODO: consider if this could cause missing good moves, if we could prioritize somehow
          if (movesLegal.NumMovesUsed > CeresTransferBlockIn.MAX_MOVES) Console.WriteLine("Warning: move overflow");

          int numMoves = Math.Min(CeresTransferBlockIn.MAX_MOVES, movesLegal.NumMovesUsed);

          ItemsIn[i].NumMoves = numMoves;
          for (int m = 0; m < numMoves; m++)
          {
            EncodedMove moveVal = ConverterMGMoveEncodedMove.MGChessMoveToEncodedMove(movesLegal.MovesArray[m]);
            refItem.Moves[m] = (short)moveVal.IndexNeuralNet;
          }

          int baseOffset = i * CeresTransferBlockIn.NUM_PLANES_PER_POSITION;

          Span<byte> values = batch.PosPlaneValues;
          Span<ulong> masks = batch.PosPlaneBitmaps;

          if (VERBOSE_DEBUG) Console.WriteLine("POSITION DUMP");
          for (int j = 0; j < CeresTransferBlockIn.NUM_PLANES_PER_POSITION; j++)
          {
            int offset = baseOffset + j;
            refItem.Masks[j] = masks[offset];
            refItem.Values[j] = values[offset];
            if (VERBOSE_DEBUG) Console.WriteLine("(" + (j + "," + masks[j]) + "," + values[j] + "),");
          }
        });

        ProcessRequest(batch.NumPos);
      }
    }

    public static bool VERBOSE_DEBUG = false;

    #endregion

    /// <summary>
    /// Returns string representation.
    /// </summary>
    /// <returns></returns>
    public override string ToString()
    {
      return $"<LC0DllNNEvaluator Session:{SessionID} WeightsFile:{NetworkFilename} GPU:{GPUID}>";
    }

    #region Disposal

    bool isDisposed = false;

    ~LC0LibraryNNEvaluator()
    {
      Dispose();
    }

    public void Dispose()
    {
      if (!isDisposed)
      {
        Free(SessionID);
        sessionIDPool.ReleaseID(SessionID);
      }

      isDisposed = true;
    }

    #endregion
  }
}
