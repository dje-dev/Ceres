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
using System.Collections.Concurrent;
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

    /// <summary>
    /// Index of the concurrent stream on whose behalf this evaluator runs (see NNDeviceSlots).
    /// Evaluators having different stream indices get separate evaluator instances on the server.
    /// </summary>
    readonly int streamIndex;

    #endregion


    #region Reconnection policy and statistics (process wide)

    /// <summary>
    /// How long (in seconds) an evaluator keeps trying to reconnect to its server
    /// before a lost connection is reported as an error.
    ///
    /// While reconnecting the evaluation call simply blocks, so a search stalls rather than fails.
    /// Tournaments and suites which run on remote device slots raise this value so that
    /// a restarted server or a network outage does not terminate the affected workers.
    /// </summary>
    public static int ReconnectPatienceSeconds = DEFAULT_RECONNECT_PATIENCE_SECONDS;

    /// <summary>
    /// Default value of ReconnectPatienceSeconds.
    /// </summary>
    public const int DEFAULT_RECONNECT_PATIENCE_SECONDS = 30;

    /// <summary>
    /// Total time (in milliseconds) lost so far to broken connections, per stream
    /// (keyed by server, device indices on that server and stream index, see StreamKey).
    /// </summary>
    static readonly ConcurrentDictionary<string, long> lostMillisecondsByStream = new();

    /// <summary>
    /// Returns the key identifying a stream: the evaluators of one concurrent worker
    /// on given device(s) of a given server.
    /// </summary>
    static string StreamKey(string hostname, int port, string deviceIndices, int streamIndex)
      => $"{hostname.ToLowerInvariant()}:{port}|{deviceIndices}|{streamIndex}";

    /// <summary>
    /// Extracts the device indices (e.g. "0,1") from a device specification string such as "GPU:0,1#TensorRTNative".
    /// </summary>
    static string DeviceIndicesOf(string deviceSpec)
    {
      string indices = deviceSpec ?? "";
      int colonIndex = indices.IndexOf(':');
      if (colonIndex >= 0)
      {
        indices = indices.Substring(colonIndex + 1);
      }

      int engineIndex = indices.IndexOf('#');
      if (engineIndex >= 0)
      {
        indices = indices.Substring(0, engineIndex);
      }

      return indices.Replace(" ", "");
    }

    /// <summary>
    /// Returns the total time (in milliseconds) that the evaluators of one stream of this process
    /// (those evaluating on the specified devices of the specified server with the specified stream index)
    /// have so far spent stalled because their connection was lost: the time from the start of each
    /// evaluation call which hit a broken connection until that call completed after reconnecting.
    ///
    /// A caller can sample this before and after a time-controlled search or game to decide
    /// if its timing was distorted by a connection problem.
    /// </summary>
    public static long LostMilliseconds(string hostname, int port, int[] deviceIndices, int streamIndex)
      => lostMillisecondsByStream.TryGetValue(StreamKey(hostname, port, string.Join(",", deviceIndices), streamIndex), out long ms) ? ms : 0;


    /// <summary>
    /// Checks if a server is listening at a specified host and port (without creating an evaluator there).
    /// </summary>
    /// <param name="hostname"></param>
    /// <param name="port"></param>
    /// <param name="timeoutMS">maximum time to wait for the connection</param>
    /// <param name="error">description of the failure, if any</param>
    /// <returns>true if a connection could be established</returns>
    public static bool TryProbeServer(string hostname, int port, int timeoutMS, out string error)
    {
      error = null;
      try
      {
        using TcpClient probe = new TcpClient();
        if (!probe.ConnectAsync(hostname, port).Wait(timeoutMS))
        {
          error = $"no response within {timeoutMS} ms";
          return false;
        }

        // Tell the server this was only a probe so that it closes the connection quietly.
        NNRemoteProtocol.WriteMessage(probe.GetStream(), NNRemoteMessageType.Disconnect,
                                      ReadOnlySpan<byte>.Empty, false, null);
        return true;
      }
      catch (Exception ex)
      {
        Exception inner = ex is AggregateException agg ? agg.GetBaseException() : ex;
        error = inner.Message;
        return false;
      }
    }

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
                              int maxBatchSize = 1024,
                              int streamIndex = 0)
    {
      this.streamIndex = streamIndex;
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

        // Enable keepalive probes so that a server which vanishes without closing the connection
        // (power loss, network partition) is noticed within about half a minute, rather than
        // only when the (deliberately long) read timeout expires.
        try
        {
          tcpClient.Client.SetSocketOption(SocketOptionLevel.Socket, SocketOptionName.KeepAlive, true);
          tcpClient.Client.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveTime, 15);
          tcpClient.Client.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveInterval, 5);
          tcpClient.Client.SetSocketOption(SocketOptionLevel.Tcp, SocketOptionName.TcpKeepAliveRetryCount, 3);
        }
        catch (Exception)
        {
          // Keepalive tuning is an optimization only (not supported on all platforms).
        }

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
        requestedMaxBatchSize, useCompression, NNRemoteProtocol.CompactPoliciesEnabled, streamIndex);

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
    /// Maximum number of times a batch is resent after its connection broke.
    /// (Each resend is preceded by a reconnection, which itself retries for up to ReconnectPatienceSeconds.)
    /// </summary>
    const int MAX_RESENDS_PER_BATCH = 3;


    /// <summary>
    /// Core evaluation: serializes positions, sends to server, receives and deserializes results.
    ///
    /// If the connection was lost (e.g. the server closed it after its idle timeout, a network glitch,
    /// or a server restart), the evaluator transparently reconnects (with backoff, for up to
    /// ReconnectPatienceSeconds) and resends the batch. Evaluation is idempotent and the server reuses
    /// its cached evaluator, so this is safe and fast. The time lost is recorded (see LostMilliseconds).
    /// </summary>
    protected override IPositionEvaluationBatch DoEvaluateIntoBuffers(
      IEncodedPositionBatchFlat positions, bool retrieveSupplementalResults = false)
    {
      long startTimestamp = Stopwatch.GetTimestamp();
      bool connectionWasLost = false;

      try
      {
        for (int attempt = 0; ; attempt++)
        {
          if (!isConnected)
          {
            connectionWasLost = true;
            Reconnect();
          }

          try
          {
            return EvaluateOnce(positions);
          }
          catch (NNRemoteConnectionException ex) when (attempt < MAX_RESENDS_PER_BATCH)
          {
            Console.Error.WriteLine($"[NNEvaluatorRemote] connection to {hostname}:{port} lost ({ex.InnerException?.Message ?? ex.Message}); reconnecting.");
            CloseConnection();
          }
        }
      }
      finally
      {
        if (connectionWasLost)
        {
          // Record how long this call was stalled by the broken connection
          // (the whole call, since a batch on a healthy connection takes negligible time by comparison).
          long lostMS = (long)Stopwatch.GetElapsedTime(startTimestamp).TotalMilliseconds;
          lostMillisecondsByStream.AddOrUpdate(StreamKey(hostname, port, DeviceIndicesOf(deviceSpec), streamIndex),
                                               lostMS, (_, prior) => prior + lostMS);
        }
      }
    }


    /// <summary>
    /// Closes the current connection (if any) and reconnects, retrying with exponential backoff
    /// (starting at 0.5 seconds, at most 5 seconds between attempts) for up to ReconnectPatienceSeconds.
    /// </summary>
    void Reconnect()
    {
      CloseConnection();

      long startTimestamp = Stopwatch.GetTimestamp();
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

          double elapsedSeconds = Stopwatch.GetElapsedTime(startTimestamp).TotalSeconds;
          if (elapsedSeconds + delayMS / 1000.0 > ReconnectPatienceSeconds)
          {
            throw new NNRemoteConnectionException($"Unable to reconnect to remote NN server at {hostname}:{port} "
                                                + $"after {i} attempts over {elapsedSeconds:F0} seconds.", ex);
          }

          if (i == 4)
          {
            Console.Error.WriteLine($"[NNEvaluatorRemote] still unable to reach {hostname}:{port}; "
                                  + $"will keep trying for up to {ReconnectPatienceSeconds} seconds in total.");
          }

          System.Threading.Thread.Sleep(delayMS);
          delayMS = Math.Min(delayMS * 2, 5000);
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
      string streamStr = streamIndex != 0 ? $", stream={streamIndex}" : "";
      return $"NNEvaluatorRemote({hostname}:{port}, net={networkSpec}, dev={deviceSpec}{streamStr})";
    }
  }
}
