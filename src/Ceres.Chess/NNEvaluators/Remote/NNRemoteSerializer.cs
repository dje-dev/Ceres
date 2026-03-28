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
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

using Ceres.Base.DataTypes;
using Ceres.Chess.EncodedPositions;
using Ceres.Chess.LC0.Batches;
using Ceres.Chess.MoveGen;
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
  }


  /// <summary>
  /// Flags indicating which output heads are present in a serialized result.
  /// </summary>
  [Flags]
  public enum NNRemoteResultFlags : byte
  {
    None = 0,
    IsWDL = 1 << 0,
    HasM = 1 << 1,
    HasUncertaintyV = 1 << 2,
    HasUncertaintyP = 1 << 3,
    HasAction = 1 << 4,
    HasValueSecondary = 1 << 5,
    HasState = 1 << 6,
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
           + numPositions * Unsafe.SizeOf<EncodedPositionWithHistory>() // PositionsBuffer (optional)
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
      // PositionsBuffer (EncodedPositionWithHistory[]) is needed by Ceres/TPG networks.
      bool hasPositionsBuffer = !batch.PositionsBuffer.IsEmpty;

      if (hasPositions) flags |= NNRemoteBatchFlags.HasPositions;
      if (hasHashes) flags |= NNRemoteBatchFlags.HasHashes;
      if (hasStates) flags |= NNRemoteBatchFlags.HasStates;
      if (hasLastMovePlies) flags |= NNRemoteBatchFlags.HasLastMovePlies;
      if (hasPositionsBuffer) flags |= NNRemoteBatchFlags.HasPositionsBuffer;

      buffer[offset++] = (byte)flags;

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

      // Conditional: PositionsBuffer (EncodedPositionWithHistory[] - needed by Ceres/TPG)
      if (hasPositionsBuffer)
      {
        ReadOnlySpan<EncodedPositionWithHistory> posBuffer = batch.PositionsBuffer.Span.Slice(0, numPos);
        ReadOnlySpan<byte> posBufferBytes = MemoryMarshal.AsBytes(posBuffer);
        posBufferBytes.CopyTo(buffer.Slice(offset));
        offset += posBufferBytes.Length;
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

      // Conditional: PositionsBuffer (EncodedPositionWithHistory[] - needed by Ceres/TPG)
      if (flags.HasFlag(NNRemoteBatchFlags.HasPositionsBuffer))
      {
        int posBufferSize = numPos * Unsafe.SizeOf<EncodedPositionWithHistory>();
        if (batch.PositionsBuffer == null || batch.PositionsBuffer.Length < numPos)
        {
          batch.PositionsBuffer = new EncodedPositionWithHistory[numPos];
        }
        MemoryMarshal.Cast<byte, EncodedPositionWithHistory>(buffer.Slice(offset, posBufferSize))
          .CopyTo(batch.PositionsBuffer.AsSpan());
        offset += posBufferSize;
      }

      return batch;
    }

    #endregion


    #region Output batch serialization

    /// <summary>
    /// Builds the result flags byte from evaluator capability booleans.
    /// </summary>
    public static NNRemoteResultFlags BuildResultFlags(bool isWDL, bool hasM,
                                                        bool hasUncertaintyV, bool hasUncertaintyP,
                                                        bool hasAction, bool hasValueSecondary,
                                                        bool hasState)
    {
      NNRemoteResultFlags flags = NNRemoteResultFlags.None;
      if (isWDL) flags |= NNRemoteResultFlags.IsWDL;
      if (hasM) flags |= NNRemoteResultFlags.HasM;
      if (hasUncertaintyV) flags |= NNRemoteResultFlags.HasUncertaintyV;
      if (hasUncertaintyP) flags |= NNRemoteResultFlags.HasUncertaintyP;
      if (hasAction) flags |= NNRemoteResultFlags.HasAction;
      if (hasValueSecondary) flags |= NNRemoteResultFlags.HasValueSecondary;
      if (hasState) flags |= NNRemoteResultFlags.HasState;
      return flags;
    }


    /// <summary>
    /// Calculates the maximum buffer size needed to serialize a result batch.
    /// </summary>
    public static int MaxSerializedResultSize(int numPositions, NNRemoteResultFlags flags)
    {
      int size = 5; // numPos + flags
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
      return size + 256; // margin
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
      buffer[offset++] = (byte)resultFlags;

      PositionEvaluationBatch peb = batch as PositionEvaluationBatch;
      if (peb == null)
      {
        throw new InvalidOperationException("SerializeBatchResult requires a PositionEvaluationBatch.");
      }

      // Policies (CompressedPolicyVector[])
      offset += WriteBlittableSpan(peb.Policies.Span.Slice(0, numPos), buffer.Slice(offset));

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
      NNRemoteResultFlags flags = (NNRemoteResultFlags)buffer[offset++];

      bool isWDL = flags.HasFlag(NNRemoteResultFlags.IsWDL);
      bool hasM = flags.HasFlag(NNRemoteResultFlags.HasM);
      bool hasUncV = flags.HasFlag(NNRemoteResultFlags.HasUncertaintyV);
      bool hasUncP = flags.HasFlag(NNRemoteResultFlags.HasUncertaintyP);
      bool hasAction = flags.HasFlag(NNRemoteResultFlags.HasAction);
      bool hasValueSecondary = flags.HasFlag(NNRemoteResultFlags.HasValueSecondary);
      bool hasState = flags.HasFlag(NNRemoteResultFlags.HasState);

      // Policies
      CompressedPolicyVector[] policies = new CompressedPolicyVector[numPos];
      offset += ReadBlittableSpan(buffer.Slice(offset), policies.AsSpan());

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
        makeCopy: false);
    }

    #endregion


    #region Handshake serialization

    /// <summary>
    /// Serializes the client handshake message.
    /// </summary>
    public static int SerializeHandshake(Span<byte> buffer,
                                          string networkSpec, string deviceSpec,
                                          string optionsString, int maxBatchSize,
                                          bool useCompression)
    {
      int offset = 0;
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), networkSpec);
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), deviceSpec);
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), optionsString);
      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), maxBatchSize);
      offset += 4;
      buffer[offset++] = useCompression ? (byte)1 : (byte)0;
      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), NNRemoteProtocol.PROTOCOL_VERSION);
      offset += 4;
      return offset;
    }


    /// <summary>
    /// Deserializes the client handshake message.
    /// </summary>
    public static (string networkSpec, string deviceSpec, string optionsString,
                    int maxBatchSize, bool useCompression, int protocolVersion)
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
      bool useCompression = buffer[offset++] != 0;
      int protocolVersion = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
      offset += 4;

      return (networkSpec, deviceSpec, optionsString, maxBatchSize, useCompression, protocolVersion);
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
                                             string serverInfo)
    {
      int offset = 0;
      buffer[offset++] = success ? (byte)1 : (byte)0;
      buffer[offset++] = (byte)resultFlags;
      BinaryPrimitives.WriteInt32LittleEndian(buffer.Slice(offset), maxBatchSize);
      offset += 4;
      BinaryPrimitives.WriteUInt32LittleEndian(buffer.Slice(offset), (uint)inputsRequired);
      offset += 4;
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), engineNetworkID);
      offset += NNRemoteProtocol.WriteString(buffer.Slice(offset), serverInfo);
      return offset;
    }


    /// <summary>
    /// Deserializes the server handshake acknowledgment.
    /// </summary>
    public static (bool success, NNRemoteResultFlags resultFlags, int maxBatchSize,
                    NNEvaluator.InputTypes inputsRequired, string engineNetworkID, string serverInfo)
      DeserializeHandshakeAck(ReadOnlySpan<byte> buffer)
    {
      int offset = 0;
      bool success = buffer[offset++] != 0;
      NNRemoteResultFlags resultFlags = (NNRemoteResultFlags)buffer[offset++];
      int maxBatchSize = BinaryPrimitives.ReadInt32LittleEndian(buffer.Slice(offset));
      offset += 4;
      NNEvaluator.InputTypes inputsRequired = (NNEvaluator.InputTypes)
        BinaryPrimitives.ReadUInt32LittleEndian(buffer.Slice(offset));
      offset += 4;
      var (engineNetworkID, n1) = NNRemoteProtocol.ReadString(buffer.Slice(offset));
      offset += n1;
      var (serverInfo, n2) = NNRemoteProtocol.ReadString(buffer.Slice(offset));
      offset += n2;

      return (success, resultFlags, maxBatchSize, inputsRequired, engineNetworkID, serverInfo);
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

    #endregion
  }
}
