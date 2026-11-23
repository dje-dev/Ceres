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
  /// Evaluators are cached and shared across client sessions (up to MaxCachedEvaluators),
  /// keyed by network, device, options and the client's stream index. Clients which belong to
  /// different concurrent streams (see NNDeviceSlots) therefore get separate evaluators,
  /// even on the same device(s). Each evaluator may in turn have several instances (see InstancesPerEvaluator).
  /// Inference on each evaluator instance is serialized via a per-instance lock, held only for the evaluation
  /// itself (input preparation, compression and sending are done outside it).
  /// A background task disposes evaluators unused for 1 hour.
  ///
  /// Usage: Ceres SERVER net=LC0:42767 device=GPU:0,1 [port=50055]
  /// </summary>
  public class NNRemoteServer
  {
    /// <summary>
    /// Default maximum number of concurrent client connections.
    /// A tournament stream typically opens 4 connections (2 players, each with 2 evaluators),
    /// so the default accommodates on the order of 16 concurrent streams per server.
    /// </summary>
    public const int DEFAULT_MAX_CLIENTS = 64;

    /// <summary>
    /// Default maximum number of cached evaluators
    /// (one is needed per distinct network on each device set, for each concurrent stream).
    /// </summary>
    public const int DEFAULT_MAX_CACHED_EVALUATORS = 64;

    /// <summary>
    /// Time after which an unused evaluator is disposed.
    /// </summary>
    static readonly TimeSpan EVALUATOR_EXPIRY = TimeSpan.FromHours(1);

    /// <summary>
    /// A result with compact policies is Zstd compressed (for clients which asked for compression)
    /// only if it is larger than this many bytes per position.
    /// </summary>
    const int COMPACT_RESULT_COMPRESSION_MIN_BYTES_PER_POSITION = 512;

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
    /// Maximum number of cached evaluators.
    /// </summary>
    public readonly int MaxCachedEvaluators;

    /// <summary>
    /// Optional default network specification (used if client doesn't provide one).
    /// </summary>
    public readonly string DefaultNetworkSpec;

    /// <summary>
    /// Optional default device specification (used if client doesn't provide one).
    /// </summary>
    public readonly string DefaultDeviceSpec;


    /// <summary>
    /// Maximum number of instances built of each evaluator.
    ///
    /// With one instance (the default) all clients of an evaluator take turns on it, so the device(s) execute
    /// one batch at a time. With two, the two connections of a client which keeps two batches in flight (as
    /// the MCGS search does) each evaluate on an instance of their own, and their batches execute concurrently.
    ///
    /// Whether that is faster depends on the device and on what else the client uses. Measured with a
    /// 512x15 network: on a GB300 in a split with another server it raised the total by about 6%, but with
    /// two instances on every server of the split the batches synchronize (both evaluate, then both wait for
    /// the search) and the total fell by about 3%. Further instances share the first one's loaded network
    /// but need their own execution contexts in device memory.
    /// </summary>
    public readonly int InstancesPerEvaluator;

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

      /// <summary>
      /// The instances of this evaluator (the first is Evaluator itself; see NNRemoteServer.InstancesPerEvaluator).
      /// </summary>
      public readonly List<EvaluatorInstance> Instances = new();

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
        Instances.Add(new EvaluatorInstance(evaluator));

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
    /// One instance of a cached evaluator. A client connection evaluates on one instance; connections
    /// sharing an instance take turns (the evaluator and its buffers are not thread-safe).
    /// </summary>
    class EvaluatorInstance
    {
      public readonly NNEvaluator Evaluator;
      public readonly object InferenceLock = new();
      public int ActiveUsers;

      public EvaluatorInstance(NNEvaluator evaluator)
      {
        Evaluator = evaluator;
      }
    }

    /// <summary>
    /// Cache of evaluators keyed by "netSpec|deviceSpec|optionsStr|stream".
    /// </summary>
    readonly Dictionary<string, CachedEvaluator> evaluatorCache = new();
    readonly object cacheLock = new();

    #endregion


    /// <summary>
    /// Constructor.
    /// </summary>
    public NNRemoteServer(int port = NNRemoteProtocol.DEFAULT_PORT,
                           int maxClients = DEFAULT_MAX_CLIENTS,
                           string defaultNetworkSpec = null,
                           string defaultDeviceSpec = null,
                           int instancesPerEvaluator = 1,
                           int maxCachedEvaluators = DEFAULT_MAX_CACHED_EVALUATORS)
    {
      Port = port;
      MaxClients = maxClients;
      MaxCachedEvaluators = maxCachedEvaluators;
      DefaultNetworkSpec = defaultNetworkSpec;
      DefaultDeviceSpec = defaultDeviceSpec;
      InstancesPerEvaluator = Math.Max(1, instancesPerEvaluator);
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
            Console.WriteLine($"[SERVER] Rejected connection from {client.Client.RemoteEndPoint} "
                            + $"(maximum of {MaxClients} clients reached; restart the server with a larger MAXCLIENTS value to allow more)");
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
          ShutdownInstances(entry);
        }
        evaluatorCache.Clear();
      }

      Console.WriteLine("[SERVER] Shutting down.");
    }


    #region Evaluator cache management

    /// <summary>
    /// Builds the cache key for an evaluator configuration.
    /// </summary>
    static string MakeCacheKey(string netSpec, string deviceSpec, string optionsStr, int streamIndex)
    {
      // The stream index is part of the key so that concurrent streams deliberately placed on the
      // same device(s) each get their own evaluator (rather than queueing for a single shared one).
      return $"{netSpec}|{deviceSpec}|{optionsStr ?? ""}|s{streamIndex}";
    }


    /// <summary>
    /// Gets or creates a cached evaluator for the given configuration.
    /// Increments ActiveUsers on success.
    /// </summary>
    CachedEvaluator AcquireEvaluator(string netSpec, string deviceSpec, string optionsStr,
                                      int streamIndex, string clientEndpoint)
    {
      string key = MakeCacheKey(netSpec, deviceSpec, optionsStr, streamIndex);

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
              ShutdownInstances(cached);
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
        if (evaluatorCache.Count >= MaxCachedEvaluators)
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
            ShutdownInstances(evictCandidate);
          }
          else
          {
            throw new InvalidOperationException(
              $"Evaluator cache is full ({MaxCachedEvaluators}) and all evaluators are in use; "
            + "restart the server with a larger MAXEVALUATORS value to allow more.");
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
    /// Shuts down all instances of a cached evaluator.
    /// </summary>
    static void ShutdownInstances(CachedEvaluator cached)
    {
      foreach (EvaluatorInstance instance in cached.Instances)
      {
        try { instance.Evaluator.Shutdown(); } catch { }
      }
    }


    /// <summary>
    /// Releases a cached evaluator (decrements ActiveUsers).
    /// The evaluator is NOT disposed -- it remains in the cache for future reuse.
    /// </summary>
    void ReleaseEvaluator(CachedEvaluator cached, EvaluatorInstance instance)
    {
      if (cached == null) return;

      lock (cacheLock)
      {
        cached.ActiveUsers--;
        cached.LastUsedUtc = DateTime.UtcNow;
        if (instance != null)
        {
          instance.ActiveUsers--;
        }
      }
    }


    /// <summary>
    /// Chooses the instance of a cached evaluator which a new client connection will evaluate on:
    /// the least used one, or a newly built one if all are in use and fewer than InstancesPerEvaluator exist.
    /// </summary>
    EvaluatorInstance AcquireInstance(CachedEvaluator cached, string netSpec, string deviceSpec, string optionsStr,
                                      string clientEndpoint)
    {
      lock (cacheLock)
      {
        EvaluatorInstance best = cached.Instances.MinBy(i => i.ActiveUsers);
        if (best.ActiveUsers > 0 && cached.Instances.Count < InstancesPerEvaluator)
        {
          // Build another instance over the same devices, sharing the first instance's loaded network.
          Console.WriteLine($"[SERVER] Client {clientEndpoint}: building evaluator instance {cached.Instances.Count + 1} for {cached.Key}");
          NNEvaluatorDef evalDef = NNEvaluatorDef.FromSpecification(netSpec, deviceSpec);
          if (optionsStr != null) evalDef.OptionsString = optionsStr;
          best = new EvaluatorInstance(NNEvaluatorFactory.BuildEvaluator(evalDef, cached.Evaluator));
          cached.Instances.Add(best);
        }

        best.ActiveUsers++;
        return best;
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
        ShutdownInstances(entry);
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
      EvaluatorInstance instance = null;
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
        if (msgType == NNRemoteMessageType.Disconnect)
        {
          // A client only checking that the server is reachable (see NNEvaluatorRemote.TryProbeServer).
          return;
        }

        if (msgType != NNRemoteMessageType.Handshake)
        {
          Console.WriteLine($"[SERVER] Client {clientEndpoint}: expected Handshake, got {msgType}. Disconnecting.");
          return;
        }

        (string networkSpec, string deviceSpec, string optionsStr, int maxBatchSize, bool clientUseCompression,
         int protocolVersion, bool clientAcceptsCompactPolicies, int streamIndex) =
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

        // Results for clients which can read them carry only the used move slots of each policy
        // (smaller than what Zstd makes of the full vectors, at almost no CPU cost).
        bool compactPolicies = clientAcceptsCompactPolicies;
        if (useCompression)
        {
          compressBuffer = new byte[ZstdBlockCompress.CompressBound(writeBuffer.Length)];
          decompressBuffer = new byte[readBuffer.Length];
        }

        Console.WriteLine($"[SERVER] Client {clientEndpoint} connected: net={netSpec}, device={devSpec}, stream={streamIndex}, maxBatch={maxBatchSize}, compression={useCompression}, compactPolicies={compactPolicies}");

        // Acquire (or reuse) a cached evaluator.
        try
        {
          cachedEval = AcquireEvaluator(netSpec, devSpec, optionsStr, streamIndex, clientEndpoint);
          instance = AcquireInstance(cachedEval, netSpec, devSpec, optionsStr, clientEndpoint);
        }
        catch (Exception ex)
        {
          Console.WriteLine($"[SERVER] Client {clientEndpoint}: evaluator build failed: {ex.Message}");
          SendError(stream, writeBuffer, compressBuffer, useCompression,
            ex.GetType().Name, ex.Message);
          return;
        }

        NNEvaluator evaluator = instance.Evaluator;
        NNRemoteResultFlags resultFlags = cachedEval.ResultFlags;

        // Send HandshakeAck.
        long networkFileSizeBytes = 0;
        try { networkFileSizeBytes = evaluator.Info?.NetworkFileSizeBytes ?? 0; } catch { }
        int ackLen = NNRemoteSerializer.SerializeHandshakeAck(writeBuffer,
          true, resultFlags, evaluator.MaxBatchSize,
          evaluator.InputsRequired, evaluator.EngineNetworkID,
          $"Ceres NNRemoteServer on {Environment.MachineName}",
          evaluator.NumDevices, networkFileSizeBytes);
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

        // Flags written with each result (the evaluator's capabilities plus the per-connection encoding).
        NNRemoteResultFlags sendFlags = compactPolicies ? resultFlags | NNRemoteResultFlags.CompactPolicies : resultFlags;

        // Optional per-phase timing (CERES_REMOTE_PERF=1).
        const int PERF_IDLE = 0, PERF_RECEIVE = 1, PERF_DECOMPRESS = 2, PERF_DESERIALIZE = 3, PERF_PREPARE = 4,
                  PERF_LOCKWAIT = 5, PERF_EVALUATE = 6, PERF_SERIALIZE = 7, PERF_COMPRESS = 8, PERF_SEND = 9;
        NNRemotePerfStats perf = NNRemotePerfStats.CreateIfEnabled($"server {Environment.MachineName} {devSpec} <- {clientEndpoint}",
          "idle", "receive", "decompress", "deserialize", "prepare", "lockwait", "evaluate", "serialize", "compress", "send");
        long perfLastEnd = 0;

        // -- Evaluation loop --
        while (isRunning)
        {
          var (reqType, reqLen) = NNRemoteProtocol.ReadMessage(stream, ref readBuffer, ref decompressBuffer);
          long tRead = Stopwatch.GetTimestamp();
          long tHeader = NNRemoteProtocol.PerfHeaderTimestamp;
          long decompressTicks = NNRemoteProtocol.PerfCodecTicks;
          int wireIn = NNRemoteProtocol.PerfWireBytes;

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
                long tDeserialized = Stopwatch.GetTimestamp();

                // CPU-side input preparation (such as materializing planes from the compact records) touches
                // neither the evaluator's buffers nor the device, so it is done before taking the evaluator
                // lock: it then overlaps the evaluation of another client's batch instead of delaying it.
                evaluator.PrepareBatchInputs(batch);
                long tPrepared = Stopwatch.GetTimestamp();

                int resultLen;
                long tLocked, tEvaluated, tSerialized;

                // Serialize inference on this evaluator (GPU is not thread-safe). Only the evaluation and the
                // copy of its result are done under the lock: a client such as the MCGS search keeps two batches
                // in flight on two connections, and everything else done here would hold up the other batch.
                lock (instance.InferenceLock)
                {
                  tLocked = Stopwatch.GetTimestamp();
                  result = evaluator.EvaluateIntoBuffers(batch);
                  tEvaluated = Stopwatch.GetTimestamp();

                  // Serialize the result while still holding the lock (it references the evaluator's
                  // internal buffers). Grow the buffer if this result needs more room than the initial
                  // estimate (e.g. networks with large State outputs).
                  int requiredLen = NNRemoteSerializer.RequiredResultSize(result, resultFlags);
                  if (writeBuffer.Length < requiredLen)
                  {
                    writeBuffer = new byte[requiredLen];
                    if (useCompression)
                    {
                      compressBuffer = new byte[ZstdBlockCompress.CompressBound(requiredLen)];
                    }
                  }

                  resultLen = NNRemoteSerializer.SerializeBatchResult(result, sendFlags, writeBuffer);
                  tSerialized = Stopwatch.GetTimestamp();
                }

                // A compact result holding little beyond policies and values is not worth compressing (it is
                // dense, and Zstd costs more time than it saves on the wire). One which also carries bulky outputs
                // such as ply-bin distributions still compresses several-fold, so that is compressed as before.
                bool compressResult = useCompression
                                   && (!compactPolicies || resultLen > (long)batch.NumPos * COMPACT_RESULT_COMPRESSION_MIN_BYTES_PER_POSITION);

                // The result now lives in this client's own write buffer, so it is compressed and sent
                // after the evaluator has been released to the next batch.
                NNRemoteProtocol.WriteMessage(stream, NNRemoteMessageType.EvalResponse,
                  writeBuffer.AsSpan(0, resultLen), compressResult, compressBuffer);

                if (perf != null)
                {
                  long tSent = Stopwatch.GetTimestamp();
                  long compressTicks = NNRemoteProtocol.PerfCodecTicks;
                  if (perfLastEnd != 0) perf.Add(PERF_IDLE, tHeader - perfLastEnd);
                  perf.Add(PERF_RECEIVE, tRead - tHeader - decompressTicks);
                  perf.Add(PERF_DECOMPRESS, decompressTicks);
                  perf.Add(PERF_DESERIALIZE, tDeserialized - tRead);
                  perf.Add(PERF_PREPARE, tPrepared - tDeserialized);
                  perf.Add(PERF_LOCKWAIT, tLocked - tPrepared);
                  perf.Add(PERF_EVALUATE, tEvaluated - tLocked);
                  perf.Add(PERF_SERIALIZE, tSerialized - tEvaluated);
                  perf.Add(PERF_COMPRESS, compressTicks);
                  perf.Add(PERF_SEND, tSent - tSerialized - compressTicks);
                  perf.AddBytes(resultLen, NNRemoteProtocol.PerfWireBytes, reqLen, wireIn);
                  perfLastEnd = tSent;
                  perf.EndBatch(batch.NumPos);
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
                // Show the full exception (with stack trace) here on the server,
                // since the client is sent only the exception type and message.
                Console.WriteLine($"[SERVER] Client {clientEndpoint}: eval error: {ex}");
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
        ReleaseEvaluator(cachedEval, instance);
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
      Console.WriteLine($"Max cached:  {MaxCachedEvaluators}");
      Console.WriteLine($"Instances:   {InstancesPerEvaluator} per evaluator");
      Console.WriteLine("------------------------------------------------------------");
    }
  }
}
