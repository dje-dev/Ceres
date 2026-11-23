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
using System.Buffers.Binary;
using System.IO;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
using Ceres.Chess.MoveGen.Converters;
using Ceres.Chess.NetEvaluation.Batch;

#endregion

namespace Ceres.Chess.NNEvaluators.Remote
{
  /// <summary>
  /// Flags indicating which optional fields are present in a serialized batch.
  /// </summary>
  [Flags]
  public enum NNRemoteBatchFlags : byte
  {
    None = 0,
    HasPositions = 1 << 0,
    HasHashes = 1 << 1,
    HasStates = 1 << 2,
    HasLastMovePlies = 1 << 3,
    HasPositionsBuffer = 1 << 4,
    HasCompactHistories = 1 << 5,
    HasPlanes = 1 << 6,
  }


  /// <summary>
  /// Flags indicating which output heads are present in a serialized result.
  /// </summary>
  [Flags]
  public enum NNRemoteResultFlags : ushort
  {
    None = 0,
    IsWDL = 1 << 0,
    HasM = 1 << 1,
    HasUncertaintyV = 1 << 2,
    HasUncertaintyP = 1 << 3,
    HasAction = 1 << 4,
    HasValueSecondary = 1 << 5,
    HasState = 1 << 6,
    HasPlyBinOutputs = 1 << 7,
    HasPunimOutputs = 1 << 8,
    HasPolicySecondary = 1 << 9,

    /// <summary>
    /// Policies are written with only their used move slots (a count, then that many move indices and
    /// probabilities) rather than as full fixed-size vectors. Set per result message by servers
    /// for clients which announced CLIENT_CAPABILITY_COMPACT_POLICIES; never part of the capabilities.
    /// </summary>
    CompactPolicies = 1 << 10,
  }


  /// <summary>
  /// Binary serialization/deserialization for NN evaluation batches (inputs and outputs).
  /// Uses MemoryMarshal for zero-copy handling of blittable arrays.
  /// </summary>
  public static class NNRemoteSerializer
  {
    const int PLANES_PER_POS = EncodedPositionBatchFlat.TOTAL_NUM_PLANES_ALL_HISTORIES; // 112


    #region Input batch serialization

    /// <summary>
    /// Calculates the maximum buffer size needed to serialize a batch of given size.
    /// </summary>
    public static int MaxSerializedInputSize(int numPositions)
    {
      // Header (4+1) + planes bitmaps + plane values + positions + hashes + lastMovePlies + states overhead
      return 5
           + numPositions * PLANES_PER_POS * sizeof(ulong)  // PosPlaneBitmaps
           + numPositions * PLANES_PER_POS                   // PosPlaneValues
           + numPositions * Unsafe.SizeOf<MGPosition>()      // Positions (optional)
           + numPositions * sizeof(ulong)                    // Hashes (optional)
           + numPositions * 64                               // LastMovePlies (optional)
           + numPositions * 256 * sizeof(ushort) + numPositions * 4  // States (optional, generous)
           + numPositions * Unsafe.SizeOf<MGPositionHistoryCompact>() // CompactHistories (optional)
           + 1024;                                           // margin
    }


    /// <summary>
    /// Serializes an IEncodedPositionBatchFlat into a byte buffer.
    /// Returns the number of bytes written.
    /// </summary>
    public static int SerializeBatchInput(IEncodedPositionBatchFlat batch,
                                           NNEvaluator.InputTypes requiredInputs,
                                           Span<byte> buffer)
    {
      int numPos = batch.NumPos;
      int offset = 0;

      // Write number of positions.
      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), numPos);
      offset += 4;

      // Determine which optional fields to include.
      // Always send Positions if the server requires Moves (server needs Positions to recompute Moves via TrySetMoves).
      NNRemoteBatchFlags flags = NNRemoteBatchFlags.None;
      bool hasPositions = (requiredInputs.HasFlag(NNEvaluator.InputTypes.Positions)
                           || requiredInputs.HasFlag(NNEvaluator.InputTypes.Moves))
                          && !batch.Positions.IsEmpty;
      bool hasHashes = requiredInputs.HasFlag(NNEvaluator.InputTypes.Hashes)
                       && !batch.PositionHashes.IsEmpty;
      bool hasStates = requiredInputs.HasFlag(NNEvaluator.InputTypes.State)
                       && !batch.States.IsEmpty;
      bool hasLastMovePlies = requiredInputs.HasFlag(NNEvaluator.InputTypes.LastMovePlies)
                              && !batch.LastMovePlies.IsEmpty;
      // Wire flags are content-driven (v4): ship whatever representation the batch actually
      // carries. The client fed its batch through EvaluateIntoBuffers, so hooks 1/2 already
      // normalized it. Prefer compact records (the smaller representation): an MCGS client ships
      // compact even to an LC0 server, which materializes planes server-side. A legacy plane
      // producer with no compact records ships its planes.
      bool hasCompactHistories = batch.CompactHistoriesPopulated && !batch.CompactHistories.IsEmpty;
      bool hasPlanes = batch.PlanesPopulated && !hasCompactHistories;

      if (hasPositions) flags |= NNRemoteBatchFlags.HasPositions;
      if (hasHashes) flags |= NNRemoteBatchFlags.HasHashes;
      if (hasStates) flags |= NNRemoteBatchFlags.HasStates;
      if (hasLastMovePlies) flags |= NNRemoteBatchFlags.HasLastMovePlies;
      if (hasCompactHistories) flags |= NNRemoteBatchFlags.HasCompactHistories;
      if (hasPlanes) flags |= NNRemoteBatchFlags.HasPlanes;

      buffer[offset++] = (byte)flags;

      // Conditional: LC0 planes (bitmaps + values), only for servers requiring Boards.
      if (hasPlanes)
      {
        // PosPlaneBitmaps: N * 112 ulongs
        int bitmapCount = numPos * PLANES_PER_POS;
        ReadOnlySpan<ulong> bitmaps = batch.PosPlaneBitmaps.Span.Slice(0, bitmapCount);
        ReadOnlySpan<byte> bitmapBytes = MemoryMarshal.AsBytes(bitmaps);
        bitmapBytes.CopyTo(buffer.Slice(offset));
        offset += bitmapBytes.Length;

        // PosPlaneValues: N * 112 bytes
        int valuesCount = numPos * PLANES_PER_POS;
        ReadOnlySpan<byte> values = batch.PosPlaneValues.Span.Slice(0, valuesCount);
        values.CopyTo(buffer.Slice(offset));
        offset += valuesCount;
      }

      // Conditional: Positions
      if (hasPositions)
      {
        ReadOnlySpan<MGPosition> positions = batch.Positions.Span.Slice(0, numPos);
        ReadOnlySpan<byte> posBytes = MemoryMarshal.AsBytes(positions);
        posBytes.CopyTo(buffer.Slice(offset));
        offset += posBytes.Length;
      }

      // Conditional: Hashes
      if (hasHashes)
      {
        ReadOnlySpan<ulong> hashes = batch.PositionHashes.Span.Slice(0, numPos);
        ReadOnlySpan<byte> hashBytes = MemoryMarshal.AsBytes(hashes);
        hashBytes.CopyTo(buffer.Slice(offset));
        offset += hashBytes.Length;
      }

      // Conditional: LastMovePlies (64 bytes per position)
      if (hasLastMovePlies)
      {
        ReadOnlySpan<byte> plies = batch.LastMovePlies.Span.Slice(0, numPos * 64);
        plies.CopyTo(buffer.Slice(offset));
        offset += plies.Length;
      }

      // Conditional: States (variable-size Half[] per position)
      if (hasStates)
      {
        ReadOnlySpan<Half[]> states = batch.States.Span.Slice(0, numPos);
        for (int i = 0; i < numPos; i++)
        {
          Half[] state = states[i];
          int stateLen = state?.Length ?? 0;
          BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), stateLen);
          offset += 4;
          if (stateLen > 0)
          {
            ReadOnlySpan<byte> stateBytes = MemoryMarshal.AsBytes(state.AsSpan());
            stateBytes.CopyTo(buffer.Slice(offset));
            offset += stateBytes.Length;
          }
        }
      }

      // Conditional: CompactHistories (the canonical history representation for Ceres/TPG servers)
      if (hasCompactHistories)
      {
        ReadOnlySpan<MGPositionHistoryCompact> histories = batch.CompactHistories.Span.Slice(0, numPos);
        ReadOnlySpan<byte> historyBytes = MemoryMarshal.AsBytes(histories);
        historyBytes.CopyTo(buffer.Slice(offset));
        offset += historyBytes.Length;
      }

      return offset;
    }


    /// <summary>
    /// Deserializes a batch input from a byte buffer into an EncodedPositionBatchFlat.
    /// </summary>
    public static EncodedPositionBatchFlat DeserializeBatchInput(ReadOnlySpan<byte> buffer)
    {
      int offset = 0;

      int numPos = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
      offset += 4;

      NNRemoteBatchFlags flags = (NNRemoteBatchFlags)buffer[offset++];

      // Create the batch with sufficient capacity.
      EncodedPositionBatchFlat batch = new EncodedPositionBatchFlat(
        EncodedPositionType.PositionOnly, numPos);
      batch.NumPos = numPos;

      // Conditional: LC0 planes. Only present when the client actually shipped planes; the plane
      // arrays are otherwise never allocated (compact-only batch). Allocate lazily here.
      if (flags.HasFlag(NNRemoteBatchFlags.HasPlanes))
      {
        batch.EnsurePlaneArraysAllocated();

        // PosPlaneBitmaps
        int bitmapCount = numPos * PLANES_PER_POS;
        int bitmapByteCount = bitmapCount * sizeof(ulong);
        ReadOnlySpan<byte> bitmapBytes = buffer.Slice(offset, bitmapByteCount);
        MemoryMarshal.Cast<byte, ulong>(bitmapBytes).CopyTo(batch.PosPlaneBitmaps.AsSpan());
        offset += bitmapByteCount;

        // PosPlaneValues
        int valuesCount = numPos * PLANES_PER_POS;
        buffer.Slice(offset, valuesCount).CopyTo(batch.PosPlaneValues.AsSpan());
        offset += valuesCount;

        batch.PlanesPopulated = true;
      }

      // Conditional: Positions
      if (flags.HasFlag(NNRemoteBatchFlags.HasPositions))
      {
        int posSize = numPos * Unsafe.SizeOf<MGPosition>();
        if (batch.Positions == null || batch.Positions.Length < numPos)
        {
          batch.Positions = new MGPosition[numPos];
        }
        MemoryMarshal.Cast<byte, MGPosition>(buffer.Slice(offset, posSize))
          .CopyTo(batch.Positions.AsSpan());
        offset += posSize;
      }

      // Conditional: Hashes
      if (flags.HasFlag(NNRemoteBatchFlags.HasHashes))
      {
        int hashSize = numPos * sizeof(ulong);
        if (batch.PositionHashes == null || batch.PositionHashes.Length < numPos)
        {
          batch.PositionHashes = new ulong[numPos];
        }
        MemoryMarshal.Cast<byte, ulong>(buffer.Slice(offset, hashSize))
          .CopyTo(batch.PositionHashes.AsSpan());
        offset += hashSize;
      }

      // Conditional: LastMovePlies
      if (flags.HasFlag(NNRemoteBatchFlags.HasLastMovePlies))
      {
        int pliesSize = numPos * 64;
        if (batch.LastMovePlies == null || batch.LastMovePlies.Length < pliesSize)
        {
          batch.LastMovePlies = new byte[pliesSize];
        }
        buffer.Slice(offset, pliesSize).CopyTo(batch.LastMovePlies.AsSpan());
        offset += pliesSize;
      }

      // Conditional: States
      if (flags.HasFlag(NNRemoteBatchFlags.HasStates))
      {
        if (batch.States == null || batch.States.Length < numPos)
        {
          batch.States = new Half[numPos][];
        }
        for (int i = 0; i < numPos; i++)
        {
          int stateLen = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
          offset += 4;
          if (stateLen > 0)
          {
            Half[] state = new Half[stateLen];
            MemoryMarshal.Cast<byte, Half>(buffer.Slice(offset, stateLen * sizeof(ushort)))
              .CopyTo(state);
            offset += stateLen * sizeof(ushort);
            batch.States[i] = state;
          }
        }
      }

      // The legacy PositionsBuffer never appears on the wire from v3+ clients (they always ship
      // CompactHistories). Reject it loudly rather than silently mis-parsing the stream.
      if (flags.HasFlag(NNRemoteBatchFlags.HasPositionsBuffer))
      {
        throw new NotSupportedException("HasPositionsBuffer is no longer supported on the wire "
                                      + "(v3+ clients ship CompactHistories instead).");
      }

      // Conditional: CompactHistories (the canonical history representation for Ceres/TPG servers)
      if (flags.HasFlag(NNRemoteBatchFlags.HasCompactHistories))
      {
        int historiesSize = numPos * Unsafe.SizeOf<MGPositionHistoryCompact>();
        if (batch.CompactHistories == null || batch.CompactHistories.Length < numPos)
        {
          batch.CompactHistories = new MGPositionHistoryCompact[numPos];
        }
        MemoryMarshal.Cast<byte, MGPositionHistoryCompact>(buffer.Slice(offset, historiesSize))
          .CopyTo(batch.CompactHistories.AsSpan());
        offset += historiesSize;

        batch.CompactHistoriesPopulated = true;
      }

      return batch;
    }

    #endregion


    #region Output batch serialization

    /// <summary>
    /// Builds the result flags byte from evaluator capability booleans.
    /// </summary>
    public static NNRemoteResultFlags BuildResultFlags(NNEvaluator evaluator)
    {
      NNRemoteResultFlags flags = NNRemoteResultFlags.None;
      if (evaluator.IsWDL) flags |= NNRemoteResultFlags.IsWDL;
      if (evaluator.HasM) flags |= NNRemoteResultFlags.HasM;
      if (evaluator.HasUncertaintyV) flags |= NNRemoteResultFlags.HasUncertaintyV;
      if (evaluator.HasUncertaintyP) flags |= NNRemoteResultFlags.HasUncertaintyP;
      if (evaluator.HasAction) flags |= NNRemoteResultFlags.HasAction;
      if (evaluator.HasValueSecondary) flags |= NNRemoteResultFlags.HasValueSecondary;
      if (evaluator.HasState) flags |= NNRemoteResultFlags.HasState;
      if (evaluator.HasPolicySecondary) flags |= NNRemoteResultFlags.HasPolicySecondary;
      return flags;
    }


    /// <summary>
    /// Calculates the maximum buffer size needed to serialize a result batch.
    /// </summary>
    public static int MaxSerializedResultSize(int numPositions, NNRemoteResultFlags flags)
    {
      int size = 6; // numPos(4) + flags(2)
      size += numPositions * Unsafe.SizeOf<CompressedPolicyVector>(); // policies
      size += numPositions * sizeof(ushort); // W
      if (flags.HasFlag(NNRemoteResultFlags.IsWDL)) size += numPositions * sizeof(ushort); // L
      if (flags.HasFlag(NNRemoteResultFlags.HasM)) size += numPositions * sizeof(ushort);
      if (flags.HasFlag(NNRemoteResultFlags.HasUncertaintyV)) size += numPositions * sizeof(ushort);
      if (flags.HasFlag(NNRemoteResultFlags.HasUncertaintyP)) size += numPositions * sizeof(ushort);
      if (flags.HasFlag(NNRemoteResultFlags.HasValueSecondary)) size += numPositions * 2 * sizeof(ushort);
      if (flags.HasFlag(NNRemoteResultFlags.HasAction)) size += numPositions * Unsafe.SizeOf<CompressedActionVector>();
      if (flags.HasFlag(NNRemoteResultFlags.HasState)) size += numPositions * (4 + 256 * sizeof(ushort));
      size += numPositions * 2 * sizeof(ushort); // ExtraStat0 + ExtraStat1
      if (flags.HasFlag(NNRemoteResultFlags.HasPlyBinOutputs)) size += numPositions * 512 * 2 * sizeof(ushort); // move + capture
      if (flags.HasFlag(NNRemoteResultFlags.HasPunimOutputs)) size += numPositions * 8 * 2 * sizeof(ushort); // self + opponent
      if (flags.HasFlag(NNRemoteResultFlags.HasPolicySecondary)) size += numPositions * Unsafe.SizeOf<CompressedPolicyVector>();
      return size + 256; // margin
    }


    /// <summary>
    /// Buffer size sufficient to serialize the given result batch: MaxSerializedResultSize for the flags
    /// actually written (including the ply-bin/PUNIM outputs SerializeBatchResult adds at runtime), with the
    /// variable-length State outputs sized from their actual lengths. Without this, nets emitting such
    /// outputs overflow the handshake-sized buffer once the batch is large enough ("Destination is too short").
    /// </summary>
    public static int RequiredResultSize(IPositionEvaluationBatch batch, NNRemoteResultFlags flags)
    {
      int numPos = batch.NumPos;

      // Include the optional outputs that SerializeBatchResult adds at runtime when present in the batch
      // (ply-bin: 2 x 512 values per position, PUNIM: 2 x 8). These are not part of the handshake flags,
      // so a buffer sized from those flags alone overflows for nets that emit them (e.g. C3 nets).
      if (batch is PositionEvaluationBatch pebRuntime)
      {
        if (pebRuntime.HasPlyBinOutputs && !pebRuntime.PlyBinMoveProbs.IsEmpty) flags |= NNRemoteResultFlags.HasPlyBinOutputs;
        if (pebRuntime.HasPunimOutputs && !pebRuntime.PunimSelfProbs.IsEmpty) flags |= NNRemoteResultFlags.HasPunimOutputs;
      }

      int size = MaxSerializedResultSize(numPos, flags & ~NNRemoteResultFlags.HasState);
      if (flags.HasFlag(NNRemoteResultFlags.HasState))
      {
        size += numPos * 4;
        if (batch is PositionEvaluationBatch peb && !peb.States.IsEmpty)
        {
          ReadOnlySpan<Half[]> states = peb.States.Span.Slice(0, numPos);
          for (int i = 0; i < numPos; i++)
          {
            size += (states[i]?.Length ?? 0) * sizeof(ushort);
          }
        }
      }
      return size;
    }


    /// <summary>
    /// Buffer size sufficient to serialize the given input batch (variable-length input States sized from
    /// their actual lengths rather than the 256 elements assumed by MaxSerializedInputSize).
    /// </summary>
    public static int RequiredInputSize(IEncodedPositionBatchFlat batch, NNEvaluator.InputTypes requiredInputs)
    {
      int numPos = batch.NumPos;
      int size = MaxSerializedInputSize(numPos);
      if (requiredInputs.HasFlag(NNEvaluator.InputTypes.State) && !batch.States.IsEmpty)
      {
        size -= numPos * 256 * sizeof(ushort); // replace the generic allowance with actual lengths
        ReadOnlySpan<Half[]> states = batch.States.Span.Slice(0, numPos);
        for (int i = 0; i < numPos; i++)
        {
          size += (states[i]?.Length ?? 0) * sizeof(ushort);
        }
        size = Math.Max(size, MaxSerializedInputSize(numPos));
      }
      return size;
    }


    /// <summary>
    /// Serializes an IPositionEvaluationBatch into a byte buffer.
    /// Returns the number of bytes written.
    /// </summary>
    public static int SerializeBatchResult(IPositionEvaluationBatch batch,
                                            NNRemoteResultFlags resultFlags,
                                            Span<byte> buffer)
    {
      int numPos = batch.NumPos;
      int offset = 0;

      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), numPos);
      offset += 4;
      BinaryPrimitives.WriteUInt16LittleEndian(buffer.Slice(offset), (ushort)resultFlags);
      offset += 2;

      PositionEvaluationBatch peb = batch as PositionEvaluationBatch;

      // Set runtime flags for optional outputs that may be present in this specific batch.
      if (peb.HasPlyBinOutputs && !peb.PlyBinMoveProbs.IsEmpty) resultFlags |= NNRemoteResultFlags.HasPlyBinOutputs;
      if (peb.HasPunimOutputs && !peb.PunimSelfProbs.IsEmpty) resultFlags |= NNRemoteResultFlags.HasPunimOutputs;

      // Rewrite flags with runtime additions.
      BinaryPrimitives.WriteUInt16LittleEndian(buffer.Slice(4), (ushort)resultFlags);
      if (peb == null)
      {
        throw new InvalidOperationException("SerializeBatchResult requires a PositionEvaluationBatch.");
      }

      // Policies (CompressedPolicyVector[])
      bool compactPolicies = resultFlags.HasFlag(NNRemoteResultFlags.CompactPolicies);
      offset += compactPolicies ? WriteCompactPolicies(peb.Policies.Span.Slice(0, numPos), buffer.Slice(offset))
                                : WriteBlittableSpan(peb.Policies.Span.Slice(0, numPos), buffer.Slice(offset));

      // W (FP16[])
      offset += WriteFP16Span(peb.W.Span, numPos, buffer.Slice(offset));

      // L (if WDL)
      if (resultFlags.HasFlag(NNRemoteResultFlags.IsWDL))
      {
        offset += WriteFP16Span(peb.L.Span, numPos, buffer.Slice(offset));
      }

      // M
      if (resultFlags.HasFlag(NNRemoteResultFlags.HasM))
      {
        offset += WriteFP16Span(peb.M.Span, numPos, buffer.Slice(offset));
      }

      // UncertaintyV
      if (resultFlags.HasFlag(NNRemoteResultFlags.HasUncertaintyV))
      {
        offset += WriteFP16Span(peb.UncertaintyV.Span, numPos, buffer.Slice(offset));
      }

      // UncertaintyP
      if (resultFlags.HasFlag(NNRemoteResultFlags.HasUncertaintyP))
      {
        offset += WriteFP16Span(peb.UncertaintyP.Span, numPos, buffer.Slice(offset));
      }

      // W2, L2
      if (resultFlags.HasFlag(NNRemoteResultFlags.HasValueSecondary))
      {
        offset += WriteFP16Span(peb.W2.Span, numPos, buffer.Slice(offset));
        if (resultFlags.HasFlag(NNRemoteResultFlags.IsWDL))
        {
          offset += WriteFP16Span(peb.L2.Span, numPos, buffer.Slice(offset));
        }
      }

      // Actions (CompressedActionVector[])
      if (resultFlags.HasFlag(NNRemoteResultFlags.HasAction) && !peb.Actions.IsEmpty)
      {
        offset += WriteBlittableSpan(peb.Actions.Span.Slice(0, numPos), buffer.Slice(offset));
      }

      // States
      if (resultFlags.HasFlag(NNRemoteResultFlags.HasState) && !peb.States.IsEmpty)
      {
        ReadOnlySpan<Half[]> states = peb.States.Span.Slice(0, numPos);
        for (int i = 0; i < numPos; i++)
        {
          Half[] state = states[i];
          int stateLen = state?.Length ?? 0;
          BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), stateLen);
          offset += 4;
          if (stateLen > 0)
          {
            ReadOnlySpan<byte> stateBytes = MemoryMarshal.AsBytes(state.AsSpan());
            stateBytes.CopyTo(buffer.Slice(offset));
            offset += stateBytes.Length;
          }
        }
      }

      // ExtraStat0, ExtraStat1
      if (!peb.ExtraStat0.IsEmpty)
      {
        offset += WriteFP16Span(peb.ExtraStat0.Span, numPos, buffer.Slice(offset));
      }
      else
      {
        // Write zeros
        buffer.Slice(offset, numPos * sizeof(ushort)).Clear();
        offset += numPos * sizeof(ushort);
      }

      if (!peb.ExtraStat1.IsEmpty)
      {
        offset += WriteFP16Span(peb.ExtraStat1.Span, numPos, buffer.Slice(offset));
      }
      else
      {
        buffer.Slice(offset, numPos * sizeof(ushort)).Clear();
        offset += numPos * sizeof(ushort);
      }

      // PlyBin outputs (flat: numPos * 512 Half values each)
      if (resultFlags.HasFlag(NNRemoteResultFlags.HasPlyBinOutputs))
      {
        offset += WriteHalfSpan(peb.PlyBinMoveProbs.Span, numPos * 512, buffer.Slice(offset));
        offset += WriteHalfSpan(peb.PlyBinCaptureProbs.Span, numPos * 512, buffer.Slice(offset));
      }

      // PUNIM outputs (flat: numPos * 8 Half values each)
      if (resultFlags.HasFlag(NNRemoteResultFlags.HasPunimOutputs))
      {
        offset += WriteHalfSpan(peb.PunimSelfProbs.Span, numPos * 8, buffer.Slice(offset));
        offset += WriteHalfSpan(peb.PunimOpponentProbs.Span, numPos * 8, buffer.Slice(offset));
      }

      // Policy2 (secondary policy head)
      if (resultFlags.HasFlag(NNRemoteResultFlags.HasPolicySecondary) && !peb.Policies2.IsEmpty)
      {
        offset += compactPolicies ? WriteCompactPolicies(peb.Policies2.Span.Slice(0, numPos), buffer.Slice(offset))
                                  : WriteBlittableSpan(peb.Policies2.Span.Slice(0, numPos), buffer.Slice(offset));
      }

      return offset;
    }


    /// <summary>
    /// Deserializes a PositionEvaluationBatch from a byte buffer.
    /// </summary>
    public static PositionEvaluationBatch DeserializeBatchResult(ReadOnlySpan<byte> buffer)
    {
      int offset = 0;

      int numPos = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
      offset += 4;
      NNRemoteResultFlags flags = (NNRemoteResultFlags)BinaryPrimitives.ReadUInt16LittleEndian(buffer.Slice(offset));
      offset += 2;

      bool isWDL = flags.HasFlag(NNRemoteResultFlags.IsWDL);
      bool hasM = flags.HasFlag(NNRemoteResultFlags.HasM);
      bool hasUncV = flags.HasFlag(NNRemoteResultFlags.HasUncertaintyV);
      bool hasUncP = flags.HasFlag(NNRemoteResultFlags.HasUncertaintyP);
      bool hasAction = flags.HasFlag(NNRemoteResultFlags.HasAction);
      bool hasValueSecondary = flags.HasFlag(NNRemoteResultFlags.HasValueSecondary);
      bool hasState = flags.HasFlag(NNRemoteResultFlags.HasState);
      bool hasPlyBin = flags.HasFlag(NNRemoteResultFlags.HasPlyBinOutputs);
      bool hasPunim = flags.HasFlag(NNRemoteResultFlags.HasPunimOutputs);
      bool hasPolicySecondary = flags.HasFlag(NNRemoteResultFlags.HasPolicySecondary);

      // Policies
      bool compactPolicies = flags.HasFlag(NNRemoteResultFlags.CompactPolicies);
      CompressedPolicyVector[] policies = new CompressedPolicyVector[numPos];
      offset += compactPolicies ? ReadCompactPolicies(buffer.Slice(offset), policies.AsSpan())
                                : ReadBlittableSpan(buffer.Slice(offset), policies.AsSpan());

      // W
      FP16[] w = new FP16[numPos];
      offset += ReadFP16Span(buffer.Slice(offset), w, numPos);

      // L
      FP16[] l = isWDL ? new FP16[numPos] : Array.Empty<FP16>();
      if (isWDL) offset += ReadFP16Span(buffer.Slice(offset), l, numPos);

      // M
      FP16[] m = hasM ? new FP16[numPos] : Array.Empty<FP16>();
      if (hasM) offset += ReadFP16Span(buffer.Slice(offset), m, numPos);

      // UncertaintyV
      FP16[] uncV = hasUncV ? new FP16[numPos] : Array.Empty<FP16>();
      if (hasUncV) offset += ReadFP16Span(buffer.Slice(offset), uncV, numPos);

      // UncertaintyP
      FP16[] uncP = hasUncP ? new FP16[numPos] : Array.Empty<FP16>();
      if (hasUncP) offset += ReadFP16Span(buffer.Slice(offset), uncP, numPos);

      // W2, L2
      FP16[] w2 = hasValueSecondary ? new FP16[numPos] : Array.Empty<FP16>();
      FP16[] l2 = (hasValueSecondary && isWDL) ? new FP16[numPos] : Array.Empty<FP16>();
      if (hasValueSecondary)
      {
        offset += ReadFP16Span(buffer.Slice(offset), w2, numPos);
        if (isWDL) offset += ReadFP16Span(buffer.Slice(offset), l2, numPos);
      }

      // Actions
      CompressedActionVector[] actions = hasAction ? new CompressedActionVector[numPos] : null;
      if (hasAction)
      {
        offset += ReadBlittableSpan(buffer.Slice(offset), actions.AsSpan());
      }

      // States
      Half[][] states = null;
      if (hasState)
      {
        states = new Half[numPos][];
        for (int i = 0; i < numPos; i++)
        {
          int stateLen = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
          offset += 4;
          if (stateLen > 0)
          {
            states[i] = new Half[stateLen];
            MemoryMarshal.Cast<byte, Half>(buffer.Slice(offset, stateLen * sizeof(ushort)))
              .CopyTo(states[i]);
            offset += stateLen * sizeof(ushort);
          }
        }
      }

      // ExtraStat0, ExtraStat1
      FP16[] extraStat0 = new FP16[numPos];
      offset += ReadFP16Span(buffer.Slice(offset), extraStat0, numPos);
      FP16[] extraStat1 = new FP16[numPos];
      offset += ReadFP16Span(buffer.Slice(offset), extraStat1, numPos);

      // PlyBin outputs
      Half[] plyBinMove = null, plyBinCapture = null;
      if (hasPlyBin)
      {
        plyBinMove = new Half[numPos * 512];
        offset += ReadHalfSpan(buffer.Slice(offset), plyBinMove, numPos * 512);
        plyBinCapture = new Half[numPos * 512];
        offset += ReadHalfSpan(buffer.Slice(offset), plyBinCapture, numPos * 512);
      }

      // PUNIM outputs
      Half[] punimSelf = null, punimOpponent = null;
      if (hasPunim)
      {
        punimSelf = new Half[numPos * 8];
        offset += ReadHalfSpan(buffer.Slice(offset), punimSelf, numPos * 8);
        punimOpponent = new Half[numPos * 8];
        offset += ReadHalfSpan(buffer.Slice(offset), punimOpponent, numPos * 8);
      }

      // Policy2 (secondary policy head)
      CompressedPolicyVector[] policies2 = null;
      if (hasPolicySecondary)
      {
        policies2 = new CompressedPolicyVector[numPos];
        offset += compactPolicies ? ReadCompactPolicies(buffer.Slice(offset), policies2.AsSpan())
                                  : ReadBlittableSpan(buffer.Slice(offset), policies2.AsSpan());
      }

      // Construct the result batch using the Memory<>-based constructor.
      return new PositionEvaluationBatch(
        isWDL, hasM, hasUncV, hasUncP, hasAction,
        hasValueSecondary, hasState, numPos,
        policies,
        hasAction ? (Memory<CompressedActionVector>)actions : default,
        w, l,
        w2, l2,
        m, uncV, uncP,
        hasState ? (Memory<Half[]>)states : default,
        default, // activations
        null,    // stats
        extraStat0,
        extraStat1,
        makeCopy: false,
        plyBinMoveProbs: hasPlyBin ? (Memory<Half>)plyBinMove : default,
        plyBinCaptureProbs: hasPlyBin ? (Memory<Half>)plyBinCapture : default,
        punimSelfProbs: hasPunim ? (Memory<Half>)punimSelf : default,
        punimOpponentProbs: hasPunim ? (Memory<Half>)punimOpponent : default,
        hasPolicySecondary: hasPolicySecondary,
        policies2: hasPolicySecondary ? (Memory<CompressedPolicyVector>)policies2 : default);
    }

    #endregion


    #region Handshake serialization

    /// <summary>
    /// Serializes the client handshake message.
    /// </summary>
    public static int SerializeHandshake(Span<byte> buffer,
                                          string networkSpec, string deviceSpec,
                                          string optionsString, int maxBatchSize,
                                          bool useCompression, bool acceptCompactPolicies = true, int streamIndex = 0)
    {
      int offset = 0;
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), networkSpec);
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), deviceSpec);
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), optionsString);
      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), maxBatchSize);
      offset += 4;
      // Capability bits (an older server reads this byte as "compression on" whenever it is nonzero;
      // the client copes with either, since each message header says whether its payload is compressed).
      byte capabilities = 0;
      if (useCompression) capabilities |= NNRemoteProtocol.CLIENT_CAPABILITY_COMPRESSION;
      if (acceptCompactPolicies) capabilities |= NNRemoteProtocol.CLIENT_CAPABILITY_COMPACT_POLICIES;
      buffer[offset++] = capabilities;
      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), NNRemoteProtocol.PROTOCOL_VERSION);
      offset += 4;

      // Optional trailing field (ignored by older servers, which then share one evaluator among all streams):
      // index of the concurrent stream on whose behalf the client evaluates (see NNDeviceSlots). The server
      // makes it part of its evaluator cache key so that streams placed on the same device(s) do not share an evaluator.
      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), streamIndex);
      offset += 4;
      return offset;
    }


    /// <summary>
    /// Deserializes the client handshake message.
    /// </summary>
    public static (string networkSpec, string deviceSpec, string optionsString,
                    int maxBatchSize, bool useCompression, int protocolVersion, bool acceptsCompactPolicies, int streamIndex)
      DeserializeHandshake(ReadOnlySpan<byte> buffer)
    {
      int offset = 0;

      var (networkSpec, n1) = NNRemoteProtocol.ReadString(buffer.Slice(offset));
      offset += n1;
      var (deviceSpec, n2) = NNRemoteProtocol.ReadString(buffer.Slice(offset));
      offset += n2;
      var (optionsString, n3) = NNRemoteProtocol.ReadString(buffer.Slice(offset));
      offset += n3;
      int maxBatchSize = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
      offset += 4;
      byte capabilities = buffer[offset++];
      bool useCompression = (capabilities & NNRemoteProtocol.CLIENT_CAPABILITY_COMPRESSION) != 0;
      bool acceptsCompactPolicies = (capabilities & NNRemoteProtocol.CLIENT_CAPABILITY_COMPACT_POLICIES) != 0;
      int protocolVersion = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
      offset += 4;

      // Optional trailing field (absent in handshakes from older clients, in which case stream 0 is assumed).
      int streamIndex = 0;
      if (buffer.Length >= offset + 4)
      {
        streamIndex = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
        offset += 4;
      }

      return (networkSpec, deviceSpec, optionsString, maxBatchSize, useCompression, protocolVersion, acceptsCompactPolicies, streamIndex);
    }


    /// <summary>
    /// Serializes the server handshake acknowledgment.
    /// </summary>
    public static int SerializeHandshakeAck(Span<byte> buffer,
                                             bool success,
                                             NNRemoteResultFlags resultFlags,
                                             int maxBatchSize,
                                             NNEvaluator.InputTypes inputsRequired,
                                             string engineNetworkID,
                                             string serverInfo,
                                             int numDevices = 0,
                                             long networkFileSizeBytes = 0)
    {
      int offset = 0;
      buffer[offset++] = success ? (byte)1 : (byte)0;
      BinaryPrimitives.WriteUInt16LittleEndian(buffer.Slice(offset), (ushort)resultFlags);
      offset += 2;
      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), maxBatchSize);
      offset += 4;
      BinaryPrimitives.WriteUInt32LittleEndian(buffer.Slice(offset), (uint)inputsRequired);
      offset += 4;
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), engineNetworkID);
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), serverInfo);

      // Optional trailing fields (ignored by older clients, absent from older servers):
      // number of compute devices behind the server's evaluator and size of its network file,
      // which the client's search uses to size batches exactly as it would for a local evaluator.
      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), numDevices);
      offset += 4;
      BinaryPrimitives.WriteInt64LittleEndian(buffer.Slice(offset), networkFileSizeBytes);
      offset += 8;
      return offset;
    }


    /// <summary>
    /// Deserializes the server handshake acknowledgment.
    /// </summary>
    public static (bool success, NNRemoteResultFlags resultFlags, int maxBatchSize,
                    NNEvaluator.InputTypes inputsRequired, string engineNetworkID, string serverInfo,
                    int numDevices, long networkFileSizeBytes)
      DeserializeHandshakeAck(ReadOnlySpan<byte> buffer)
    {
      int offset = 0;
      bool success = buffer[offset++] != 0;
      NNRemoteResultFlags resultFlags = (NNRemoteResultFlags)BinaryPrimitives.ReadUInt16LittleEndian(buffer.Slice(offset));
      offset += 2;
      int maxBatchSize = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
      offset += 4;
      NNEvaluator.InputTypes inputsRequired = (NNEvaluator.InputTypes)
        BinaryPrimitives.ReadUInt32LittleEndian(buffer.Slice(offset));
      offset += 4;
      var (engineNetworkID, n1) = NNRemoteProtocol.ReadString(buffer.Slice(offset));
      offset += n1;
      var (serverInfo, n2) = NNRemoteProtocol.ReadString(buffer.Slice(offset));
      offset += n2;

      // Optional trailing fields (absent when the server predates them): 0 means unknown.
      int numDevices = 0;
      long networkFileSizeBytes = 0;
      if (buffer.Length >= offset + 12)
      {
        numDevices = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
        offset += 4;
        networkFileSizeBytes = BinaryPrimitives.ReadInt64LittleEndian(buffer.Slice(offset));
        offset += 8;
      }

      return (success, resultFlags, maxBatchSize, inputsRequired, engineNetworkID, serverInfo, numDevices, networkFileSizeBytes);
    }


    /// <summary>
    /// Serializes an error message.
    /// </summary>
    public static int SerializeError(Span<byte> buffer, string exceptionType, string message)
    {
      int offset = 0;
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), exceptionType);
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), message);
      return offset;
    }


    /// <summary>
    /// Deserializes an error message.
    /// </summary>
    public static (string exceptionType, string message) DeserializeError(ReadOnlySpan<byte> buffer)
    {
      int offset = 0;
      var (exceptionType, n1) = NNRemoteProtocol.ReadString(buffer.Slice(offset));
      offset += n1;
      var (message, n2) = NNRemoteProtocol.ReadString(buffer.Slice(offset));
      return (exceptionType, message);
    }

    #endregion


    #region Compact policies

    // Byte offsets of the fields of CompressedPolicyVector (determined from the actual layout).
    static readonly int policySideOffset;
    static readonly int policyIndicesOffset;
    static readonly int policyProbabilitiesOffset;

    static NNRemoteSerializer()
    {
      CompressedPolicyVector sample = default;
      ref byte start = ref Unsafe.As<CompressedPolicyVector, byte>(ref sample);
      policySideOffset = (int)Unsafe.ByteOffset(ref start, ref Unsafe.As<SideType, byte>(ref Unsafe.AsRef(in sample.Side)));
      policyIndicesOffset = (int)Unsafe.ByteOffset(ref start,
        ref Unsafe.As<ushort, byte>(ref MemoryMarshal.GetReference(sample.MoveIndicesSpan)));
      policyProbabilitiesOffset = (int)Unsafe.ByteOffset(ref start,
        ref Unsafe.As<ushort, byte>(ref MemoryMarshal.GetReference(sample.ProbabilitiesSpan)));
    }


    /// <summary>
    /// Writes policies with only their used move slots. Per position:
    ///   [1 byte] number of slots n written (those before the terminator, at most NUM_MOVE_SLOTS)
    ///   [1 byte] side
    ///   [n * 2 bytes] move indices, [n * 2 bytes] probabilities
    /// A position typically has about 30 legal moves, so this is well under half the size of the
    /// full 80-slot vector at negligible CPU cost (it never exceeds the size of the full vector).
    /// Returns bytes written.
    /// </summary>
    static int WriteCompactPolicies(ReadOnlySpan<CompressedPolicyVector> policies, Span<byte> dest)
    {
      const int NUM_SLOTS = CompressedPolicyVector.NUM_MOVE_SLOTS;
      int policySize = Unsafe.SizeOf<CompressedPolicyVector>();
      ReadOnlySpan<byte> raw = MemoryMarshal.AsBytes(policies);

      int offset = 0;
      for (int i = 0; i < policies.Length; i++)
      {
        ReadOnlySpan<byte> policy = raw.Slice(i * policySize, policySize);
        ReadOnlySpan<ushort> indices = MemoryMarshal.Cast<byte, ushort>(policy.Slice(policyIndicesOffset, NUM_SLOTS * sizeof(ushort)));

        int numUsed = indices.IndexOf(CompressedPolicyVector.SPECIAL_VALUE_SENTINEL_TERMINATOR);
        if (numUsed < 0)
        {
          numUsed = NUM_SLOTS;
        }

        dest[offset++] = (byte)numUsed;
        dest[offset++] = policy[policySideOffset];

        int numBytes = numUsed * sizeof(ushort);
        policy.Slice(policyIndicesOffset, numBytes).CopyTo(dest.Slice(offset));
        offset += numBytes;
        policy.Slice(policyProbabilitiesOffset, numBytes).CopyTo(dest.Slice(offset));
        offset += numBytes;
      }
      return offset;
    }


    /// <summary>
    /// Reads policies written by WriteCompactPolicies into dest (which must be zero initialized).
    /// Returns bytes consumed.
    /// </summary>
    static int ReadCompactPolicies(ReadOnlySpan<byte> src, Span<CompressedPolicyVector> dest)
    {
      const int NUM_SLOTS = CompressedPolicyVector.NUM_MOVE_SLOTS;
      int policySize = Unsafe.SizeOf<CompressedPolicyVector>();
      Span<byte> raw = MemoryMarshal.AsBytes(dest);

      int offset = 0;
      for (int i = 0; i < dest.Length; i++)
      {
        Span<byte> policy = raw.Slice(i * policySize, policySize);

        int numUsed = src[offset++];
        if (numUsed > NUM_SLOTS)
        {
          throw new InvalidDataException($"Compact policy has {numUsed} move slots (maximum {NUM_SLOTS}).");
        }
        policy[policySideOffset] = src[offset++];

        int numBytes = numUsed * sizeof(ushort);
        src.Slice(offset, numBytes).CopyTo(policy.Slice(policyIndicesOffset));
        offset += numBytes;
        src.Slice(offset, numBytes).CopyTo(policy.Slice(policyProbabilitiesOffset));
        offset += numBytes;

        if (numUsed < NUM_SLOTS)
        {
          BinaryPrimitives.WriteUInt16LittleEndian(policy.Slice(policyIndicesOffset + numBytes),
                                                   CompressedPolicyVector.SPECIAL_VALUE_SENTINEL_TERMINATOR);
        }
      }
      return offset;
    }

    #endregion


    #region Helper methods

    /// <summary>
    /// Writes a span of blittable structs as raw bytes. Returns bytes written.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int WriteBlittableSpan<T>(ReadOnlySpan<T> src, Span<byte> dest) where T : struct
    {
      ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(src);
      bytes.CopyTo(dest);
      return bytes.Length;
    }

    /// <summary>
    /// Reads raw bytes into a span of blittable structs. Returns bytes consumed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int ReadBlittableSpan<T>(ReadOnlySpan<byte> src, Span<T> dest) where T : struct
    {
      int byteCount = dest.Length * Unsafe.SizeOf<T>();
      MemoryMarshal.Cast<byte, T>(src.Slice(0, byteCount)).CopyTo(dest);
      return byteCount;
    }

    /// <summary>
    /// Writes FP16 values as raw bytes. Returns bytes written.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int WriteFP16Span(ReadOnlySpan<FP16> src, int count, Span<byte> dest)
    {
      ReadOnlySpan<FP16> slice = src.Slice(0, count);
      ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(slice);
      bytes.CopyTo(dest);
      return bytes.Length;
    }

    /// <summary>
    /// Reads raw bytes into FP16 values. Returns bytes consumed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int ReadFP16Span(ReadOnlySpan<byte> src, FP16[] dest, int count)
    {
      int byteCount = count * sizeof(ushort);
      MemoryMarshal.Cast<byte, FP16>(src.Slice(0, byteCount)).CopyTo(dest);
      return byteCount;
    }

    /// <summary>
    /// Writes Half values as raw bytes. Returns bytes written.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int WriteHalfSpan(ReadOnlySpan<Half> src, int count, Span<byte> dest)
    {
      ReadOnlySpan<Half> slice = src.Slice(0, count);
      ReadOnlySpan<byte> bytes = MemoryMarshal.AsBytes(slice);
      bytes.CopyTo(dest);
      return bytes.Length;
    }

    /// <summary>
    /// Reads raw bytes into Half values. Returns bytes consumed.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    static int ReadHalfSpan(ReadOnlySpan<byte> src, Half[] dest, int count)
    {
      int byteCount = count * sizeof(ushort);
      MemoryMarshal.Cast<byte, Half>(src.Slice(0, byteCount)).CopyTo(dest);
      return byteCount;
    }

    #endregion
  }
}
