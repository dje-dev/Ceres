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
using System.Diagnostics;
using System.Net;
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
    /// Whether Zstd compression of requests was asked for by the creator of this evaluator.
    /// </summary>
    readonly bool compressionRequested;

    /// <summary>
    /// Whether requests on the current connection are Zstd compressed (above the size threshold):
    /// as requested, unless disabled globally (NNRemoteProtocol.CompressionEnabled)
    /// or the server is on this machine.
    /// </summary>
    bool useCompression;

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
    int cachedNumDevices = 1;
    EvaluatorInfo cachedInfo;

    #endregion


    #region Performance diagnostics

    // Phases of one remote evaluation as seen by the client (see NNRemotePerfStats).
    const int PERF_GAP = 0, PERF_SERIALIZE = 1, PERF_COMPRESS = 2, PERF_SEND = 3, PERF_WAIT = 4,
              PERF_RECEIVE = 5, PERF_DECOMPRESS = 6, PERF_DESERIALIZE = 7;
    NNRemotePerfStats perf;
    long perfLastEnd;

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

    /// <summary>
    /// Number of compute devices behind the remote evaluator (as reported by the server,
    /// otherwise the number of devices named in the device specification).
    /// </summary>
    public override int NumDevices => cachedNumDevices;

    /// <summary>
    /// Information about the remote evaluator (network file size as reported by the server), or null if unknown.
    /// </summary>
    public override EvaluatorInfo Info => cachedInfo;

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
      this.compressionRequested = useCompression && NNRemoteProtocol.CompressionEnabled;
      this.requestedMaxBatchSize = maxBatchSize;

      // Pre-allocate buffers (a decompression buffer is allocated on demand if a server compresses unasked).
      sendBuffer = new byte[NNRemoteSerializer.MaxSerializedInputSize(maxBatchSize)];
      receiveBuffer = new byte[NNRemoteSerializer.MaxSerializedResultSize(maxBatchSize,
        (NNRemoteResultFlags)0x7F)]; // all flags set for max size
      compressBuffer = compressionRequested ? new byte[ZstdBlockCompress.CompressBound(sendBuffer.Length)] : null;
      decompressBuffer = compressionRequested ? new byte[receiveBuffer.Length] : null;

      perf = NNRemotePerfStats.CreateIfEnabled($"client {hostname}:{port} {deviceSpec}",
               "gap", "serialize", "compress", "send", "wait", "receive", "decompress", "deserialize");

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

        // Compression only costs time when the server is on this machine.
        useCompression = compressionRequested && !IsConnectedToLocalMachine(tcpClient.Client);
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
    /// Returns if the socket is connected to a server on this machine
    /// (via a loopback address, or via one of the machine's own addresses).
    /// </summary>
    static bool IsConnectedToLocalMachine(Socket socket)
    {
      if (socket.RemoteEndPoint is not IPEndPoint remote || socket.LocalEndPoint is not IPEndPoint local)
      {
        return false;
      }

      // A dual-mode socket reports IPv4 addresses in IPv4-mapped IPv6 form,
      // which IPAddress.IsLoopback recognizes only for 127.0.0.1 itself.
      IPAddress remoteAddress = remote.Address.IsIPv4MappedToIPv6 ? remote.Address.MapToIPv4() : remote.Address;
      IPAddress localAddress = local.Address.IsIPv4MappedToIPv6 ? local.Address.MapToIPv4() : local.Address;
      return IPAddress.IsLoopback(remoteAddress) || remoteAddress.Equals(localAddress);
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
        requestedMaxBatchSize, useCompression, NNRemoteProtocol.CompactPoliciesEnabled);

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

      (bool success, NNRemoteResultFlags resultFlags, int maxBatch, InputTypes inputsReq, string engineNetID,
       string serverInfo, int serverNumDevices, long serverNetFileSize) =
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
      // Never exceed the batch size our send/receive buffers were sized for
      // (a multi-GPU server evaluator can report a larger maximum than the client requested).
      cachedMaxBatchSize = Math.Min(maxBatch, requestedMaxBatchSize);
      // If the server requires Moves, the client must also provide Positions
      // (needed by TrySetMoves on the server to recompute moves).
      cachedInputsRequired = inputsReq;
      if (cachedInputsRequired.HasFlag(InputTypes.Moves))
      {
        cachedInputsRequired |= InputTypes.Positions;
      }

      // Plane materialization happens SERVER-side. Masking Boards out of the client's required
      // inputs prevents the choke-point hook 2 from materializing planes on the client for an LC0
      // server: the client ships the compact records instead (smaller wire), and the server
      // materializes. Legacy plane producers still ship their existing planes (content-driven).
      cachedInputsRequired &= ~InputTypes.Boards;

      EngineNetworkID = engineNetID;
      Description = $"Remote({hostname}:{port})";

      // Devices and network size behind the server's evaluator. The search sizes its batches from these,
      // exactly as for a local evaluator (a server predating these fields reports neither; the device
      // count is then taken from the device specification).
      cachedNumDevices = serverNumDevices > 0 ? serverNumDevices : CountDevicesInSpecification(deviceSpec);
      cachedInfo = serverNetFileSize > 0 ? new EvaluatorInfo(0, serverNetFileSize) : null;

      // Ceres/TPG networks (which advertise CompactHistories) need position histories in the
      // batches they receive. The client feeds batches through EvaluateIntoBuffers, so the
      // choke-point hook 1 derives the compact records from planes (for producers that supply
      // only planes) before serialization ships them - no retained PositionsBuffer needed.
    }


    /// <summary>
    /// Returns the number of devices named in a device specification such as "GPU:0,1,2#TensorRTNative".
    /// </summary>
    static int CountDevicesInSpecification(string deviceSpecification)
    {
      if (string.IsNullOrEmpty(deviceSpecification))
      {
        return 1;
      }

      int start = deviceSpecification.IndexOf(':') + 1;
      int end = deviceSpecification.IndexOfAny(['#', '@', '/'], start);
      string indices = end < 0 ? deviceSpecification.Substring(start) : deviceSpecification.Substring(start, end - start);
      return Math.Max(1, indices.Split(',', StringSplitOptions.RemoveEmptyEntries).Length);
    }


    /// <summary>
    /// Maximum number of reconnection attempts (with exponential backoff starting at 0.5s, ~30s total)
    /// before a lost connection is reported as an error.
    /// </summary>
    const int MAX_RECONNECT_ATTEMPTS = 6;


    /// <summary>
    /// Core evaluation: serializes positions, sends to server, receives and deserializes results.
    ///
    /// If the connection was lost (e.g. the server closed it after its idle timeout, a network glitch,
    /// or a server restart), the evaluator transparently reconnects (with backoff) and resends the batch
    /// once. Evaluation is idempotent and the server reuses its cached evaluator, so this is safe and fast.
    /// </summary>
    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(
      IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      for (int attempt = 0; ; attempt++)
      {
        if (!isConnected)
        {
          Reconnect();
        }

        try
        {
          return EvaluateOnce(positions);
        }
        catch (NNRemoteConnectionException ex) when (attempt == 0)
        {
          Console.Error.WriteLine($"[NNEvaluatorRemote] connection to {hostname}:{port} lost ({ex.InnerException?.Message ?? ex.Message}); reconnecting.");
          CloseConnection();
        }
      }
    }


    /// <summary>
    /// Closes the current connection (if any) and reconnects with exponential backoff.
    /// </summary>
    void Reconnect()
    {
      CloseConnection();
      int delayMS = 500;
      for (int i = 1; ; i++)
      {
        try
        {
          Connect();
          Console.Error.WriteLine($"[NNEvaluatorRemote] reconnected to {hostname}:{port} (attempt {i}).");
          return;
        }
        catch (Exception ex) when (ex is NNRemoteConnectionException || ex is System.IO.IOException || ex is SocketException)
        {
          CloseConnection();
          if (i >= MAX_RECONNECT_ATTEMPTS)
          {
            throw new NNRemoteConnectionException($"Unable to reconnect to remote NN server at {hostname}:{port} after {i} attempts.", ex);
          }
          System.Threading.Thread.Sleep(delayMS);
          delayMS *= 2;
        }
      }
    }


    /// <summary>
    /// Closes the TCP connection without sending a disconnect message.
    /// </summary>
    void CloseConnection()
    {
      isConnected = false;
      try { stream?.Dispose(); } catch { }
      try { tcpClient?.Dispose(); } catch { }
      stream = null;
      tcpClient = null;
    }


    /// <summary>
    /// Sends one batch and receives its result over the current connection.
    /// Throws NNRemoteConnectionException on any communication failure.
    /// </summary>
    IPositionEvaluationBatch EvaluateOnce(IEncodedPositionBatchFlat positions)
    {
      try
      {
        long t0 = Stopwatch.GetTimestamp();

        // Grow the send buffer if this batch needs more room than the initial estimate
        // (e.g. large input State vectors).
        int requiredLen = NNRemoteSerializer.RequiredInputSize(positions, cachedInputsRequired);
        if (sendBuffer.Length < requiredLen)
        {
          sendBuffer = new byte[requiredLen];
          if (useCompression)
          {
            compressBuffer = new byte[ZstdBlockCompress.CompressBound(requiredLen)];
          }
        }

        // Serialize the input batch.
        int serializedLen = NNRemoteSerializer.SerializeBatchInput(
          positions, cachedInputsRequired, sendBuffer);

        long t1 = Stopwatch.GetTimestamp();

        // Send as EvalRequest.
        NNRemoteProtocol.WriteMessage(stream, NNRemoteMessageType.EvalRequest,
          sendBuffer.AsSpan(0, serializedLen), useCompression, compressBuffer);
        long t2 = Stopwatch.GetTimestamp();
        long compressTicks = NNRemoteProtocol.PerfCodecTicks;
        int wireOut = NNRemoteProtocol.PerfWireBytes;

        // Read response.
        var (type, payloadLen) = NNRemoteProtocol.ReadMessage(
          stream, ref receiveBuffer, ref decompressBuffer);
        long t3 = Stopwatch.GetTimestamp();

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
        IPositionEvaluationBatch result = NNRemoteSerializer.DeserializeBatchResult(receiveBuffer.AsSpan(0, payloadLen));

        if (perf != null)
        {
          long t4 = Stopwatch.GetTimestamp();
          long tHeader = NNRemoteProtocol.PerfHeaderTimestamp;
          long decompressTicks = NNRemoteProtocol.PerfCodecTicks;
          if (perfLastEnd != 0) perf.Add(PERF_GAP, t0 - perfLastEnd);
          perf.Add(PERF_SERIALIZE, t1 - t0);
          perf.Add(PERF_COMPRESS, compressTicks);
          perf.Add(PERF_SEND, t2 - t1 - compressTicks);
          perf.Add(PERF_WAIT, tHeader - t2);
          perf.Add(PERF_RECEIVE, t3 - tHeader - decompressTicks);
          perf.Add(PERF_DECOMPRESS, decompressTicks);
          perf.Add(PERF_DESERIALIZE, t4 - t3);
          perf.AddBytes(serializedLen, wireOut, payloadLen, NNRemoteProtocol.PerfWireBytes);
          perfLastEnd = t4;
          perf.EndBatch(positions.NumPos);
        }

        return result;
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
