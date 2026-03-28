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
using System.Net.Sockets;

using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.EncodedPositions;

#endregion

namespace Ceres.Chess.NNEvaluators.Remote
{
  /// <summary>
  /// NNEvaluator subclass that transparently forwards evaluation requests
  /// to a remote NNRemoteServer over TCP.
  ///
  /// Usage: specify "@hostname" in the device string, e.g. "GPU:0,1@spark2-fast"
  /// </summary>
  public class NNEvaluatorRemote : NNEvaluator
  {
    #region Configuration

    /// <summary>
    /// Hostname of the remote evaluation server.
    /// </summary>
    readonly string hostname;

    /// <summary>
    /// TCP port of the remote evaluation server.
    /// </summary>
    readonly int port;

    /// <summary>
    /// Network specification string to send to the server (e.g. "LC0:42767").
    /// </summary>
    readonly string networkSpec;

    /// <summary>
    /// Device specification string for the server-local devices (e.g. "GPU:0,1").
    /// </summary>
    readonly string deviceSpec;

    /// <summary>
    /// Optional evaluator options string.
    /// </summary>
    readonly string optionsString;

    /// <summary>
    /// Whether to use Zstd compression for payloads above the threshold.
    /// </summary>
    readonly bool useCompression;

    /// <summary>
    /// Maximum batch size supported.
    /// </summary>
    readonly int requestedMaxBatchSize;

    #endregion


    #region Connection state

    TcpClient tcpClient;
    NetworkStream stream;
    bool isConnected;

    #endregion


    #region Cached capabilities (from handshake)

    bool cachedIsWDL;
    bool cachedHasM;
    bool cachedHasAction;
    bool cachedHasUncertaintyV;
    bool cachedHasUncertaintyP;
    bool cachedHasValueSecondary;
    bool cachedHasState;
    bool cachedHasPolicySecondary;
    int cachedMaxBatchSize;
    InputTypes cachedInputsRequired;
    NNRemoteResultFlags cachedResultFlags;

    #endregion


    #region Reusable buffers

    byte[] sendBuffer;
    byte[] receiveBuffer;
    byte[] compressBuffer;
    byte[] decompressBuffer;

    #endregion


    #region NNEvaluator abstract property overrides

    public override bool IsWDL => cachedIsWDL;
    public override bool HasM => cachedHasM;
    public override bool HasAction => cachedHasAction;
    public override bool HasUncertaintyV => cachedHasUncertaintyV;
    public override bool HasUncertaintyP => cachedHasUncertaintyP;
    public override bool HasValueSecondary => cachedHasValueSecondary;
    public override bool HasPolicySecondary => cachedHasPolicySecondary;
    public override int MaxBatchSize => cachedMaxBatchSize;
    public override InputTypes InputsRequired => cachedInputsRequired;

    #endregion


    /// <summary>
    /// Constructor. Connects to the remote server and performs handshake.
    /// </summary>
    public NNEvaluatorRemote(string hostname, int port,
                              string networkSpec, string deviceSpec,
                              string optionsString = null,
                              bool useCompression = true,
                              int maxBatchSize = 1024)
    {
      this.hostname = hostname;
      this.port = port;
      this.networkSpec = networkSpec;
      this.deviceSpec = deviceSpec;
      this.optionsString = optionsString;
      this.useCompression = useCompression;
      this.requestedMaxBatchSize = maxBatchSize;

      // Pre-allocate buffers.
      sendBuffer = new byte[NNRemoteSerializer.MaxSerializedInputSize(maxBatchSize)];
      receiveBuffer = new byte[NNRemoteSerializer.MaxSerializedResultSize(maxBatchSize,
        (NNRemoteResultFlags)0x7F)]; // all flags set for max size
      compressBuffer = useCompression ? new byte[ZstdBlockCompress.CompressBound(sendBuffer.Length)] : null;
      decompressBuffer = useCompression ? new byte[receiveBuffer.Length] : null;

      Connect();
    }


    /// <summary>
    /// Establishes TCP connection and performs handshake with the remote server.
    /// </summary>
    void Connect()
    {
      try
      {
        tcpClient = new TcpClient();
        tcpClient.NoDelay = true;
        tcpClient.SendBufferSize = 1024 * 1024;
        tcpClient.ReceiveBufferSize = 1024 * 1024;
        tcpClient.Connect(hostname, port);
        stream = tcpClient.GetStream();
        stream.ReadTimeout = NNRemoteProtocol.DEFAULT_READ_TIMEOUT_MS;
      }
      catch (Exception ex)
      {
        throw new NNRemoteConnectionException(
          $"Failed to connect to remote NN server at {hostname}:{port}", ex);
      }

      PerformHandshake();
      isConnected = true;
    }


    /// <summary>
    /// Sends handshake and processes acknowledgment.
    /// </summary>
    void PerformHandshake()
    {
      // Serialize and send handshake.
      byte[] hsBuffer = new byte[4096];
      int hsLen = NNRemoteSerializer.SerializeHandshake(
        hsBuffer, networkSpec, deviceSpec, optionsString,
        requestedMaxBatchSize, useCompression);

      NNRemoteProtocol.WriteMessage(stream, NNRemoteMessageType.Handshake,
        hsBuffer.AsSpan(0, hsLen), false, null);

      // Read response.
      var (type, payloadLen) = NNRemoteProtocol.ReadMessage(stream, ref receiveBuffer, ref decompressBuffer);

      if (type == NNRemoteMessageType.Error)
      {
        var (exType, exMsg) = NNRemoteSerializer.DeserializeError(receiveBuffer.AsSpan(0, payloadLen));
        throw new NNRemoteEvaluationException(exType, exMsg);
      }

      if (type != NNRemoteMessageType.HandshakeAck)
      {
        throw new NNRemoteConnectionException(
          $"Expected HandshakeAck, received {type}");
      }

      var (success, resultFlags, maxBatch, inputsReq, engineNetID, serverInfo) =
        NNRemoteSerializer.DeserializeHandshakeAck(receiveBuffer.AsSpan(0, payloadLen));

      if (!success)
      {
        throw new NNRemoteConnectionException(
          $"Remote server at {hostname}:{port} rejected handshake. Info: {serverInfo}");
      }

      // Cache capabilities.
      cachedResultFlags = resultFlags;
      cachedIsWDL = resultFlags.HasFlag(NNRemoteResultFlags.IsWDL);
      cachedHasM = resultFlags.HasFlag(NNRemoteResultFlags.HasM);
      cachedHasAction = resultFlags.HasFlag(NNRemoteResultFlags.HasAction);
      cachedHasUncertaintyV = resultFlags.HasFlag(NNRemoteResultFlags.HasUncertaintyV);
      cachedHasUncertaintyP = resultFlags.HasFlag(NNRemoteResultFlags.HasUncertaintyP);
      cachedHasValueSecondary = resultFlags.HasFlag(NNRemoteResultFlags.HasValueSecondary);
      cachedHasState = resultFlags.HasFlag(NNRemoteResultFlags.HasState);
      cachedHasPolicySecondary = resultFlags.HasFlag(NNRemoteResultFlags.HasPolicySecondary);
      cachedMaxBatchSize = maxBatch;
      // If the server requires Moves, the client must also provide Positions
      // (needed by TrySetMoves on the server to recompute moves).
      cachedInputsRequired = inputsReq;
      if (cachedInputsRequired.HasFlag(InputTypes.Moves))
      {
        cachedInputsRequired |= InputTypes.Positions;
      }

      EngineNetworkID = engineNetID;
      Description = $"Remote({hostname}:{port})";

      // Ceres/TPG networks need PositionsBuffer populated in batches.
      // Set the global flag so batch constructors retain position internals.
      EncodedPositionBatchFlat.RETAIN_POSITION_INTERNALS = true;
    }


    /// <summary>
    /// Core evaluation: serializes positions, sends to server, receives and deserializes results.
    /// </summary>
    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(
      IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      if (!isConnected)
      {
        throw new NNRemoteConnectionException("Not connected to remote server.");
      }

      try
      {
        // Serialize the input batch.
        int serializedLen = NNRemoteSerializer.SerializeBatchInput(
          positions, cachedInputsRequired, sendBuffer);

        // Send as EvalRequest.
        NNRemoteProtocol.WriteMessage(stream, NNRemoteMessageType.EvalRequest,
          sendBuffer.AsSpan(0, serializedLen), useCompression, compressBuffer);

        // Read response.
        var (type, payloadLen) = NNRemoteProtocol.ReadMessage(
          stream, ref receiveBuffer, ref decompressBuffer);

        if (type == NNRemoteMessageType.Error)
        {
          var (exType, exMsg) = NNRemoteSerializer.DeserializeError(
            receiveBuffer.AsSpan(0, payloadLen));
          throw new NNRemoteEvaluationException(exType, exMsg);
        }

        if (type != NNRemoteMessageType.EvalResponse)
        {
          throw new NNRemoteConnectionException(
            $"Expected EvalResponse, received {type}");
        }

        // Deserialize the result.
        return NNRemoteSerializer.DeserializeBatchResult(receiveBuffer.AsSpan(0, payloadLen));
      }
      catch (NNRemoteEvaluationException)
      {
        throw; // Re-throw remote evaluation errors as-is.
      }
      catch (NNRemoteConnectionException)
      {
        throw;
      }
      catch (Exception ex)
      {
        isConnected = false;
        throw new NNRemoteConnectionException(
          $"Communication error with remote server at {hostname}:{port}", ex);
      }
    }


    /// <summary>
    /// Sends disconnect message and closes the TCP connection.
    /// </summary>
    protected override void DoShutdown()
    {
      if (isConnected && stream != null)
      {
        try
        {
          NNRemoteProtocol.WriteMessage(stream, NNRemoteMessageType.Disconnect,
            ReadOnlySpan<byte>.Empty, false, null);
        }
        catch
        {
          // Best effort disconnect.
        }
      }

      isConnected = false;
      stream?.Dispose();
      tcpClient?.Dispose();
      stream = null;
      tcpClient = null;
    }


    public override string ToString()
    {
      return $"NNEvaluatorRemote({hostname}:{port}, net={networkSpec}, dev={deviceSpec})";
    }
  }
}
