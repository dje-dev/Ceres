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
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Threading;

using Ceres.Chess.LC0.Batches;
using Ceres.Chess.NetEvaluation.Batch;
using Ceres.Chess.NNEvaluators.Defs;
using Ceres.Chess.UserSettings;

#endregion

namespace Ceres.Chess.NNEvaluators.Remote
{
  /// <summary>
  /// TCP server that accepts connections from NNEvaluatorRemote clients
  /// and performs neural network evaluation on their behalf using local GPUs.
  ///
  /// Evaluators are cached and shared across client sessions (up to 16).
  /// Inference on each evaluator is serialized via a per-evaluator lock.
  /// A background task disposes evaluators unused for 1 hour.
  ///
  /// Usage: Ceres SERVER net=LC0:42767 device=GPU:0,1 [port=50055]
  /// </summary>
  public class NNRemoteServer
  {
    /// <summary>
    /// Maximum number of cached evaluators.
    /// </summary>
    const int MAX_CACHED_EVALUATORS = 16;

    /// <summary>
    /// Time after which an unused evaluator is disposed.
    /// </summary>
    static readonly TimeSpan EVALUATOR_EXPIRY = TimeSpan.FromHours(1);

    /// <summary>
    /// Interval for the cleanup sweep of expired evaluators.
    /// </summary>
    const int CLEANUP_INTERVAL_MS = 60_000; // 1 minute


    /// <summary>
    /// TCP port to listen on.
    /// </summary>
    public readonly int Port;

    /// <summary>
    /// Maximum concurrent client connections.
    /// </summary>
    public readonly int MaxClients;

    /// <summary>
    /// Optional default network specification (used if client doesn't provide one).
    /// </summary>
    public readonly string DefaultNetworkSpec;

    /// <summary>
    /// Optional default device specification (used if client doesn't provide one).
    /// </summary>
    public readonly string DefaultDeviceSpec;


    TcpListener listener;
    volatile bool isRunning;
    int activeClients;

    // Aggregate statistics.
    long totalBatches;
    long totalPositions;
    long statsBatches;
    long statsPositions;
    DateTime lastStatsTime;
    Timer statsTimer;
    Timer cleanupTimer;


    #region Evaluator cache

    /// <summary>
    /// A cached evaluator entry with metadata for sharing and expiry.
    /// </summary>
    class CachedEvaluator
    {
      public readonly NNEvaluator Evaluator;
      public readonly NNRemoteResultFlags ResultFlags;
      public readonly string Key;
      public readonly object InferenceLock = new();

      /// <summary>
      /// Full path to the network file (if resolved), for staleness detection.
      /// </summary>
      public readonly string NetworkFilePath;

      /// <summary>
      /// Last write time of the network file when the evaluator was built.
      /// </summary>
      public readonly DateTime NetworkFileTimeUtc;

      public int ActiveUsers;
      public DateTime LastUsedUtc;

      public CachedEvaluator(NNEvaluator evaluator, NNRemoteResultFlags resultFlags,
                              string key, string networkFilePath)
      {
        Evaluator = evaluator;
        ResultFlags = resultFlags;
        Key = key;
        LastUsedUtc = DateTime.UtcNow;

        NetworkFilePath = networkFilePath;
        if (networkFilePath != null && File.Exists(networkFilePath))
        {
          NetworkFileTimeUtc = File.GetLastWriteTimeUtc(networkFilePath);
        }
      }

      /// <summary>
      /// Returns true if the underlying network file has been modified since this evaluator was built.
      /// </summary>
      public bool IsNetworkFileStale
      {
        get
        {
          if (NetworkFilePath == null || !File.Exists(NetworkFilePath))
          {
            return false; // Can't determine; assume not stale.
          }
          return File.GetLastWriteTimeUtc(NetworkFilePath) != NetworkFileTimeUtc;
        }
      }
    }

    /// <summary>
    /// Cache of evaluators keyed by "netSpec|deviceSpec|optionsStr".
    /// </summary>
    readonly Dictionary<string, CachedEvaluator> evaluatorCache = new();
    readonly object cacheLock = new();

    #endregion


    /// <summary>
    /// Constructor.
    /// </summary>
    public NNRemoteServer(int port = NNRemoteProtocol.DEFAULT_PORT,
                           int maxClients = 4,
                           string defaultNetworkSpec = null,
                           string defaultDeviceSpec = null)
    {
      Port = port;
      MaxClients = maxClients;
      DefaultNetworkSpec = defaultNetworkSpec;
      DefaultDeviceSpec = defaultDeviceSpec;
    }


    /// <summary>
    /// Starts the server. Blocks until Stop() is called or the process is terminated.
    /// </summary>
    public void Start()
    {
      PrintBanner();

      listener = new TcpListener(IPAddress.Any, Port);
      listener.Start();
      isRunning = true;

      // Start the 30-second statistics timer.
      lastStatsTime = DateTime.UtcNow;
      statsTimer = new Timer(_ => LogStatistics(), null, 30_000, 30_000);

      // Start the background cleanup timer for expired evaluators.
      cleanupTimer = new Timer(_ => CleanupExpiredEvaluators(), null, CLEANUP_INTERVAL_MS, CLEANUP_INTERVAL_MS);

      Console.CancelKeyPress += (s, e) =>
      {
        e.Cancel = true;
        Stop();
      };

      Console.WriteLine("[SERVER] Listening for connections...");
      Console.WriteLine();

      while (isRunning)
      {
        try
        {
          TcpClient client = listener.AcceptTcpClient();
          if (Interlocked.Increment(ref activeClients) > MaxClients)
          {
            Interlocked.Decrement(ref activeClients);
            Console.WriteLine($"[SERVER] Rejected connection from {client.Client.RemoteEndPoint} (max clients reached)");
            client.Close();
            continue;
          }

          Thread clientThread = new Thread(() => HandleClient(client))
          {
            IsBackground = true,
            Name = $"NNRemoteServer-Client-{client.Client.RemoteEndPoint}",
            Priority = ThreadPriority.AboveNormal
          };
          clientThread.Start();
        }
        catch (SocketException) when (!isRunning)
        {
          break; // Normal shutdown.
        }
        catch (Exception ex)
        {
          Console.WriteLine($"[SERVER] Accept error: {ex.Message}");
        }
      }
    }


    /// <summary>
    /// Stops the server and closes all connections.
    /// </summary>
    public void Stop()
    {
      isRunning = false;
      statsTimer?.Dispose();
      cleanupTimer?.Dispose();
      listener?.Stop();

      // Shut down all cached evaluators.
      lock (cacheLock)
      {
        foreach (var entry in evaluatorCache.Values)
        {
          try { entry.Evaluator.Shutdown(); } catch { }
        }
        evaluatorCache.Clear();
      }

      Console.WriteLine("[SERVER] Shutting down.");
    }


    #region Evaluator cache management

    /// <summary>
    /// Builds the cache key for an evaluator configuration.
    /// </summary>
    static string MakeCacheKey(string netSpec, string deviceSpec, string optionsStr)
    {
      return $"{netSpec}|{deviceSpec}|{optionsStr ?? ""}";
    }


    /// <summary>
    /// Gets or creates a cached evaluator for the given configuration.
    /// Increments ActiveUsers on success.
    /// </summary>
    CachedEvaluator AcquireEvaluator(string netSpec, string deviceSpec, string optionsStr,
                                      string clientEndpoint)
    {
      string key = MakeCacheKey(netSpec, deviceSpec, optionsStr);

      lock (cacheLock)
      {
        if (evaluatorCache.TryGetValue(key, out CachedEvaluator cached))
        {
          // Check if the underlying network file has changed since we built this evaluator.
          if (cached.IsNetworkFileStale)
          {
            Console.WriteLine($"[SERVER] Network file changed for {key}, invalidating cached evaluator.");
            evaluatorCache.Remove(key);
            if (cached.ActiveUsers == 0)
            {
              try { cached.Evaluator.Shutdown(); } catch { }
            }
            // Fall through to rebuild below.
          }
          else
          {
            cached.ActiveUsers++;
            cached.LastUsedUtc = DateTime.UtcNow;
            Console.WriteLine($"[SERVER] Client {clientEndpoint}: reusing cached evaluator for {key} (users={cached.ActiveUsers})");
            return cached;
          }
        }

        // Check if we've hit the cache limit.
        if (evaluatorCache.Count >= MAX_CACHED_EVALUATORS)
        {
          // Evict the least recently used evaluator that has no active users.
          var evictCandidate = evaluatorCache.Values
            .Where(e => e.ActiveUsers == 0)
            .OrderBy(e => e.LastUsedUtc)
            .FirstOrDefault();

          if (evictCandidate != null)
          {
            Console.WriteLine($"[SERVER] Evicting cached evaluator: {evictCandidate.Key}");
            evaluatorCache.Remove(evictCandidate.Key);
            try { evictCandidate.Evaluator.Shutdown(); } catch { }
          }
          else
          {
            throw new InvalidOperationException(
              $"Evaluator cache is full ({MAX_CACHED_EVALUATORS}) and all evaluators are in use.");
          }
        }

        // Build a new evaluator.
        Console.WriteLine($"[SERVER] Client {clientEndpoint}: building new evaluator for {key}");
        NNEvaluatorDef evalDef = NNEvaluatorDef.FromSpecification(netSpec, deviceSpec);
        if (optionsStr != null) evalDef.OptionsString = optionsStr;
        NNEvaluator evaluator = NNEvaluatorFactory.BuildEvaluator(evalDef);

        NNRemoteResultFlags resultFlags = NNRemoteSerializer.BuildResultFlags(evaluator);

        // Try to resolve the network file path for staleness detection.
        string networkFilePath = ResolveNetworkFilePath(netSpec);

        CachedEvaluator entry = new CachedEvaluator(evaluator, resultFlags, key, networkFilePath);
        entry.ActiveUsers = 1;
        evaluatorCache[key] = entry;
        return entry;
      }
    }


    /// <summary>
    /// Attempts to resolve the full file path for a network specification.
    /// Returns null if the path cannot be determined.
    /// </summary>
    static string ResolveNetworkFilePath(string netSpec)
    {
      try
      {
        // Strip type prefix (e.g., "LC0:", "Ceres:", "ONNX_ORT:").
        string netID = netSpec;
        int colonIndex = netSpec.IndexOf(':');
        if (colonIndex >= 0)
        {
          netID = netSpec.Substring(colonIndex + 1);
        }

        // Strip alias prefix.
        if (netID.StartsWith("~"))
        {
          return null; // Alias -- can't easily resolve to a file.
        }

        // Try direct path.
        if (File.Exists(netID)) return Path.GetFullPath(netID);

        // Try with .onnx extension.
        string withOnnx = netID.EndsWith(".onnx", StringComparison.OrdinalIgnoreCase)
                          ? netID : netID + ".onnx";
        if (File.Exists(withOnnx)) return Path.GetFullPath(withOnnx);

        // Try Ceres networks directory.
        string ceresDir = CeresUserSettingsManager.Settings.DirCeresNetworks;
        if (!string.IsNullOrEmpty(ceresDir))
        {
          string ceresPath = Path.Combine(ceresDir, withOnnx);
          if (File.Exists(ceresPath)) return ceresPath;
        }

        // Try LC0 networks directory.
        string lc0Dir = CeresUserSettingsManager.Settings.DirLC0Networks;
        if (!string.IsNullOrEmpty(lc0Dir))
        {
          string lc0Path = Path.Combine(lc0Dir, netID);
          if (File.Exists(lc0Path)) return lc0Path;
          string lc0PathOnnx = Path.Combine(lc0Dir, withOnnx);
          if (File.Exists(lc0PathOnnx)) return lc0PathOnnx;
        }
      }
      catch
      {
        // Best effort -- ignore errors.
      }
      return null;
    }


    /// <summary>
    /// Releases a cached evaluator (decrements ActiveUsers).
    /// The evaluator is NOT disposed -- it remains in the cache for future reuse.
    /// </summary>
    void ReleaseEvaluator(CachedEvaluator cached)
    {
      if (cached == null) return;

      lock (cacheLock)
      {
        cached.ActiveUsers--;
        cached.LastUsedUtc = DateTime.UtcNow;
      }
    }


    /// <summary>
    /// Background task: disposes evaluators that are not in use and have not been
    /// used within the expiry period.
    /// </summary>
    void CleanupExpiredEvaluators()
    {
      List<CachedEvaluator> toRemove = new();

      lock (cacheLock)
      {
        DateTime cutoff = DateTime.UtcNow - EVALUATOR_EXPIRY;

        foreach (var entry in evaluatorCache.Values)
        {
          if (entry.ActiveUsers == 0 && entry.LastUsedUtc < cutoff)
          {
            toRemove.Add(entry);
          }
        }

        foreach (var entry in toRemove)
        {
          evaluatorCache.Remove(entry.Key);
        }
      }

      // Shut down outside the lock to avoid holding it during potentially slow operations.
      foreach (var entry in toRemove)
      {
        Console.WriteLine($"[SERVER] Expiring unused evaluator: {entry.Key} (last used {(DateTime.UtcNow - entry.LastUsedUtc).TotalMinutes:F0} min ago)");
        try { entry.Evaluator.Shutdown(); } catch { }
      }
    }

    #endregion


    /// <summary>
    /// Handles a single client connection: handshake, evaluation loop, cleanup.
    /// </summary>
    void HandleClient(TcpClient tcpClient)
    {
      string clientEndpoint = tcpClient.Client.RemoteEndPoint?.ToString() ?? "unknown";
      CachedEvaluator cachedEval = null;
      long clientBatches = 0;
      long clientPositions = 0;
      bool useCompression = false;

      try
      {
        tcpClient.NoDelay = true;
        tcpClient.SendBufferSize = 1024 * 1024;
        tcpClient.ReceiveBufferSize = 1024 * 1024;
        NetworkStream stream = tcpClient.GetStream();
        stream.ReadTimeout = 300_000; // 5 minute idle timeout.

        byte[] readBuffer = new byte[1024 * 1024];
        byte[] writeBuffer = new byte[1024 * 1024];
        byte[] compressBuffer = null;
        byte[] decompressBuffer = null;

        // -- Handshake --
        var (msgType, payloadLen) = NNRemoteProtocol.ReadMessage(stream, ref readBuffer, ref decompressBuffer);
        if (msgType != NNRemoteMessageType.Handshake)
        {
          Console.WriteLine($"[SERVER] Client {clientEndpoint}: expected Handshake, got {msgType}. Disconnecting.");
          return;
        }

        var (networkSpec, deviceSpec, optionsStr, maxBatchSize, clientUseCompression, protocolVersion) =
          NNRemoteSerializer.DeserializeHandshake(readBuffer.AsSpan(0, payloadLen));

        // Validate protocol version.
        if (protocolVersion != NNRemoteProtocol.PROTOCOL_VERSION)
        {
          SendError(stream, writeBuffer, compressBuffer, false,
            "InvalidOperationException",
            $"Protocol version mismatch: client={protocolVersion}, server={NNRemoteProtocol.PROTOCOL_VERSION}");
          return;
        }

        // Use defaults if client didn't specify.
        string netSpec = networkSpec ?? DefaultNetworkSpec;
        string devSpec = deviceSpec ?? DefaultDeviceSpec;

        if (netSpec == null || devSpec == null)
        {
          SendError(stream, writeBuffer, compressBuffer, false,
            "ArgumentException",
            "Network and device specifications are required.");
          return;
        }

        useCompression = clientUseCompression;
        if (useCompression)
        {
          compressBuffer = new byte[ZstdBlockCompress.CompressBound(writeBuffer.Length)];
          decompressBuffer = new byte[readBuffer.Length];
        }

        Console.WriteLine($"[SERVER] Client {clientEndpoint} connected: net={netSpec}, device={devSpec}, maxBatch={maxBatchSize}, compression={useCompression}");

        // Acquire (or reuse) a cached evaluator.
        try
        {
          cachedEval = AcquireEvaluator(netSpec, devSpec, optionsStr, clientEndpoint);
        }
        catch (Exception ex)
        {
          Console.WriteLine($"[SERVER] Client {clientEndpoint}: evaluator build failed: {ex.Message}");
          SendError(stream, writeBuffer, compressBuffer, useCompression,
            ex.GetType().Name, ex.Message);
          return;
        }

        NNEvaluator evaluator = cachedEval.Evaluator;
        NNRemoteResultFlags resultFlags = cachedEval.ResultFlags;

        // Send HandshakeAck.
        int ackLen = NNRemoteSerializer.SerializeHandshakeAck(writeBuffer,
          true, resultFlags, evaluator.MaxBatchSize,
          evaluator.InputsRequired, evaluator.EngineNetworkID,
          $"Ceres NNRemoteServer on {Environment.MachineName}");
        NNRemoteProtocol.WriteMessage(stream, NNRemoteMessageType.HandshakeAck,
          writeBuffer.AsSpan(0, ackLen), false, null);

        Console.WriteLine($"[SERVER] Client {clientEndpoint}: evaluator ready, capabilities: WDL={evaluator.IsWDL}, M={evaluator.HasM}, Action={evaluator.HasAction}, State={evaluator.HasState}");

        // Resize buffers for actual max batch size.
        int maxBatch = Math.Min(maxBatchSize, evaluator.MaxBatchSize);
        int maxInputSize = NNRemoteSerializer.MaxSerializedInputSize(maxBatch);
        int maxResultSize = NNRemoteSerializer.MaxSerializedResultSize(maxBatch, resultFlags);
        if (readBuffer.Length < maxInputSize) readBuffer = new byte[maxInputSize];
        if (writeBuffer.Length < maxResultSize) writeBuffer = new byte[maxResultSize];
        if (useCompression)
        {
          compressBuffer = new byte[ZstdBlockCompress.CompressBound(maxResultSize)];
          if (decompressBuffer.Length < maxInputSize) decompressBuffer = new byte[maxInputSize];
        }

        // -- Evaluation loop --
        while (isRunning)
        {
          var (reqType, reqLen) = NNRemoteProtocol.ReadMessage(stream, ref readBuffer, ref decompressBuffer);

          switch (reqType)
          {
            case NNRemoteMessageType.EvalRequest:
              try
              {
                // Deserialize input batch.
                EncodedPositionBatchFlat batch =
                  NNRemoteSerializer.DeserializeBatchInput(readBuffer.AsSpan(0, reqLen));

                // Recompute moves from positions if needed.
                ((IEncodedPositionBatchFlat)batch).TrySetMoves();

                IPositionEvaluationBatch result;

                // Serialize inference on this evaluator (GPU is not thread-safe).
                lock (cachedEval.InferenceLock)
                {
                  result = evaluator.EvaluateIntoBuffers(batch);

                  // Serialize result while still holding the lock
                  // (result references evaluator's internal buffers).
                  int resultLen = NNRemoteSerializer.SerializeBatchResult(
                    result, resultFlags, writeBuffer);

                  // Send response (can happen outside the lock since writeBuffer is per-client).
                  NNRemoteProtocol.WriteMessage(stream, NNRemoteMessageType.EvalResponse,
                    writeBuffer.AsSpan(0, resultLen), useCompression, compressBuffer);
                }

                // Update statistics.
                clientBatches++;
                clientPositions += batch.NumPos;
                Interlocked.Add(ref totalBatches, 1);
                Interlocked.Add(ref totalPositions, batch.NumPos);
                Interlocked.Add(ref statsBatches, 1);
                Interlocked.Add(ref statsPositions, batch.NumPos);
              }
              catch (Exception ex)
              {
                Console.WriteLine($"[SERVER] Client {clientEndpoint}: eval error: {ex.Message}");
                SendError(stream, writeBuffer, compressBuffer, useCompression,
                  ex.GetType().Name, ex.Message);
              }
              break;

            case NNRemoteMessageType.Ping:
              NNRemoteProtocol.WriteMessage(stream, NNRemoteMessageType.Pong,
                ReadOnlySpan<byte>.Empty, false, null);
              break;

            case NNRemoteMessageType.Disconnect:
              Console.WriteLine($"[SERVER] Client {clientEndpoint} disconnected gracefully after {clientBatches:N0} batches, {clientPositions:N0} positions.");
              return;

            default:
              Console.WriteLine($"[SERVER] Client {clientEndpoint}: unexpected message type {reqType}.");
              return;
          }
        }
      }
      catch (System.IO.IOException)
      {
        Console.WriteLine($"[SERVER] Client {clientEndpoint} connection lost after {clientBatches:N0} batches, {clientPositions:N0} positions.");
      }
      catch (Exception ex)
      {
        Console.WriteLine($"[SERVER] Client {clientEndpoint} error: {ex.Message}");
      }
      finally
      {
        ReleaseEvaluator(cachedEval);
        tcpClient?.Close();
        Interlocked.Decrement(ref activeClients);
      }
    }


    /// <summary>
    /// Sends an error message to the client.
    /// </summary>
    static void SendError(NetworkStream stream, byte[] buffer, byte[] compressBuffer,
                           bool useCompression, string exceptionType, string message)
    {
      int len = NNRemoteSerializer.SerializeError(buffer, exceptionType, message);
      NNRemoteProtocol.WriteMessage(stream, NNRemoteMessageType.Error,
        buffer.AsSpan(0, len), false, null); // Errors are never compressed.
    }


    /// <summary>
    /// Logs aggregate statistics if there was activity since the last report.
    /// </summary>
    void LogStatistics()
    {
      long batchesSinceLast = Interlocked.Exchange(ref statsBatches, 0);
      long positionsSinceLast = Interlocked.Exchange(ref statsPositions, 0);

      if (batchesSinceLast == 0)
      {
        return; // No activity, don't spam the console.
      }

      DateTime now = DateTime.UtcNow;
      double seconds = (now - lastStatsTime).TotalSeconds;
      lastStatsTime = now;

      double posPerSec = seconds > 0 ? positionsSinceLast / seconds : 0;
      double avgBatch = batchesSinceLast > 0 ? (double)positionsSinceLast / batchesSinceLast : 0;

      int cachedCount;
      lock (cacheLock) { cachedCount = evaluatorCache.Count; }

      Console.WriteLine($"[SERVER] 30s stats: {activeClients} client(s), {cachedCount} cached evaluator(s), " +
                         $"{batchesSinceLast:N0} batches, {positionsSinceLast:N0} positions, " +
                         $"avg batch={avgBatch:F0}, {posPerSec:N0} pos/sec " +
                         $"(total: {totalBatches:N0} batches, {totalPositions:N0} positions)");
    }


    /// <summary>
    /// Prints the server startup banner.
    /// </summary>
    void PrintBanner()
    {
      Console.WriteLine("============================================================");
      Console.WriteLine("Ceres Neural Network Evaluation Server v1");
      Console.WriteLine("------------------------------------------------------------");
      if (DefaultNetworkSpec != null)
        Console.WriteLine($"Network:     {DefaultNetworkSpec}");
      if (DefaultDeviceSpec != null)
        Console.WriteLine($"Device:      {DefaultDeviceSpec}");
      Console.WriteLine($"Port:        {Port}");
      Console.WriteLine($"Max clients: {MaxClients}");
      Console.WriteLine($"Host:        {Environment.MachineName}");
      Console.WriteLine($"Max cached:  {MAX_CACHED_EVALUATORS}");
      Console.WriteLine("------------------------------------------------------------");
    }
  }
}
