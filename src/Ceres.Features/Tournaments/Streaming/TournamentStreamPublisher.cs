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
using System.IO;
using System.Linq;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

#endregion

namespace Ceres.Features.Tournaments.Streaming
{
  /// <summary>
  /// Implements ITournamentObserver by publishing live tournament events over a TCP socket
  /// using newline-delimited JSON (NDJSON, the CELT/1 protocol). Subscribers connect and request
  /// either the tournament-global feed (tournament info + every completed game result) or a single
  /// game thread's live feed (game start + moves + game end). On subscribe a snapshot of current
  /// state is sent, followed by live deltas.
  ///
  /// All observer methods are wrapped so they never throw or block back into the tournament worker
  /// threads: events are serialized and enqueued to per-subscriber bounded queues, with blocking
  /// socket I/O performed on dedicated writer tasks. When no subscriber is connected the cost of an
  /// event is a quick serialize + dictionary scan under a short lock.
  /// </summary>
  public sealed class TournamentStreamPublisher : ITournamentObserver, IDisposable
  {
    public const string PROTOCOL = "celt/1.0";

    readonly int port;
    readonly int threadCount;

    readonly object gate = new();
    readonly Dictionary<int, ThreadLiveState> threads = new();
    readonly List<GameEndDTO> results = new();
    readonly Dictionary<Guid, StreamSubscriber> subscribers = new();
    readonly List<Action> onFirstClientCallbacks = new();
    bool anyClientConnected;
    TournamentMetaDTO meta;
    long globalSeq;

    TcpListener listener;
    bool enabled;
    bool disposed;

    static readonly JsonSerializerOptions JsonOpts = new()
    {
      PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
      DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
      // Tolerate non-finite floats: serialize NaN/Infinity as named tokens rather than throwing
      // (a throw here would silently drop a move frame). DtoMappers also sanitizes these to finite
      // values; this is belt-and-suspenders so a stray non-finite value can never drop a frame.
      NumberHandling = JsonNumberHandling.AllowNamedFloatingPointLiterals
    };


    public TournamentStreamPublisher(int port, int threadCount)
    {
      this.port = port;
      this.threadCount = threadCount;
      StartListener();
    }


    void StartListener()
    {
      try
      {
        listener = new TcpListener(IPAddress.Any, port);
        listener.Start();
        enabled = true;
        _ = Task.Run(AcceptLoopAsync);
        Console.WriteLine($"Live tournament stream listening on TCP port {port} (protocol {PROTOCOL}).");
      }
      catch (Exception exc)
      {
        enabled = false;
        Console.WriteLine($"Live tournament streaming disabled (could not listen on port {port}): {exc.Message}");
      }
    }


    async Task AcceptLoopAsync()
    {
      while (!disposed)
      {
        TcpClient client;
        try
        {
          client = await listener.AcceptTcpClientAsync();
        }
        catch
        {
          break;
        }
        _ = HandleClientAsync(client);
      }
    }


    async Task HandleClientAsync(TcpClient client)
    {
      NetworkStream stream = null;
      try
      {
        client.NoDelay = true;
        stream = client.GetStream();

        // Greet the client.
        await WriteLineDirectAsync(stream, Serialize(new HelloDTO
        {
          Type = "hello",
          Protocol = PROTOCOL,
          ProtocolsSupported = new List<string> { PROTOCOL },
          ThreadCount = threadCount
        }));

        // Read the single request line (subscribe or directory).
        string requestLine = await ReadLineRawAsync(stream, TimeSpan.FromSeconds(10));
        if (requestLine == null)
        {
          stream.Dispose();
          client.Close();
          return;
        }

        SubscribeRequest req;
        try
        {
          req = JsonSerializer.Deserialize<SubscribeRequest>(requestLine, JsonOpts);
        }
        catch
        {
          stream.Dispose();
          client.Close();
          return;
        }

        if (req != null && string.Equals(req.Type, "directory", StringComparison.OrdinalIgnoreCase))
        {
          await WriteLineDirectAsync(stream, Serialize(BuildDirectory()));
          stream.Dispose();
          client.Close();
          return;
        }

        // Treat anything else as a subscribe.
        string scope = (req?.Scope == "thread") ? "thread" : (req?.Type == "subscribe" ? (req.Scope ?? "thread") : "thread");
        if (req != null && string.Equals(req.Type, "subscribe", StringComparison.OrdinalIgnoreCase) && req.Scope == "global")
        {
          scope = "global";
        }
        int threadId = req?.ThreadId ?? 0;

        StreamSubscriber sub = new(client, stream, scope, threadId, RemoveSubscriber);

        // Register and send the snapshot atomically with respect to live broadcasts.
        Action[] firstClientCallbacks = null;
        lock (gate)
        {
          subscribers[sub.Id] = sub;
          SendSnapshotLocked(sub);

          // First client to ever connect: capture any registered on-first-client callbacks to run
          // (outside the lock). These enable optional higher-overhead gathering (e.g. verbose move
          // stats) only once someone is actually watching.
          if (!anyClientConnected)
          {
            anyClientConnected = true;
            firstClientCallbacks = onFirstClientCallbacks.ToArray();
            onFirstClientCallbacks.Clear();
          }
        }

        if (firstClientCallbacks != null)
        {
          foreach (Action cb in firstClientCallbacks)
          {
            try { cb(); } catch { }
          }
        }
      }
      catch
      {
        try { stream?.Dispose(); } catch { }
        try { client?.Close(); } catch { }
      }
    }


    // Must be called holding 'gate'.
    void SendSnapshotLocked(StreamSubscriber sub)
    {
      if (sub.Scope == "global")
      {
        sub.Enqueue(Serialize(new SubscribedDTO
        {
          Type = "subscribed",
          Scope = "global",
          Protocol = PROTOCOL,
          CurrentSeq = globalSeq,
          SnapshotFollows = true
        }));
        if (meta != null)
        {
          sub.Enqueue(Serialize(meta));
        }
        foreach (GameEndDTO r in results)
        {
          sub.Enqueue(Serialize(r));
        }
      }
      else
      {
        threads.TryGetValue(sub.ThreadId, out ThreadLiveState ts);
        sub.Enqueue(Serialize(new SubscribedDTO
        {
          Type = "subscribed",
          Scope = "thread",
          ThreadId = sub.ThreadId,
          Protocol = PROTOCOL,
          CurrentSeq = ts?.Seq ?? 0,
          SnapshotFollows = true
        }));
        if (meta != null)
        {
          sub.Enqueue(Serialize(meta));
        }
        if (ts != null)
        {
          if (ts.CurrentGame != null)
          {
            sub.Enqueue(Serialize(ts.CurrentGame));
          }
          foreach (MoveDTO m in ts.Moves)
          {
            sub.Enqueue(Serialize(m));
          }
          if (ts.Interim != null && !ts.Finished)
          {
            sub.Enqueue(Serialize(ts.Interim));
          }
          if (ts.LastEnd != null)
          {
            sub.Enqueue(Serialize(ts.LastEnd));
          }
        }
      }
    }


    DirectoryDTO BuildDirectory()
    {
      DirectoryDTO dir = new() { Type = "directoryResponse" };
      lock (gate)
      {
        dir.TournamentName = meta?.Name;
        foreach (ThreadLiveState ts in threads.Values.OrderBy(t => t.ThreadId))
        {
          dir.Threads.Add(new DirThreadDTO
          {
            ThreadId = ts.ThreadId,
            White = ts.CurrentGame?.WhiteName,
            Black = ts.CurrentGame?.BlackName,
            GameNr = ts.CurrentGame?.GameSequenceNum ?? 0,
            State = ts.Finished ? "finished" : (ts.CurrentGame != null ? "playing" : "betweenGames"),
            LatestSeq = ts.Seq
          });
        }

        // If no games have started yet, advertise the configured thread ids so a client can
        // still address (e.g. publish a URL for) each thread.
        if (dir.Threads.Count == 0)
        {
          for (int i = 0; i < Math.Max(1, threadCount); i++)
          {
            dir.Threads.Add(new DirThreadDTO { ThreadId = i, State = "betweenGames" });
          }
        }
      }
      return dir;
    }


    void RemoveSubscriber(StreamSubscriber sub)
    {
      lock (gate)
      {
        subscribers.Remove(sub.Id);
      }
    }


    ThreadLiveState GetThreadLocked(int threadIndex)
    {
      if (!threads.TryGetValue(threadIndex, out ThreadLiveState ts))
      {
        ts = new ThreadLiveState(threadIndex);
        threads[threadIndex] = ts;
      }
      return ts;
    }


    void BroadcastLocked(string scope, int threadId, string line)
    {
      foreach (StreamSubscriber sub in subscribers.Values)
      {
        if (sub.Matches(scope, threadId))
        {
          sub.Enqueue(line);
        }
      }
    }


    #region ITournamentObserver

    public void OnTournamentStart(TournamentMetaDTO m)
    {
      if (!enabled)
      {
        return;
      }
      try
      {
        string line;
        lock (gate)
        {
          // A new tournament beginning on a reused publisher must not retain the prior tournament's
          // completed games, otherwise the global standings / game count accumulate across tournaments.
          results.Clear();
          meta = m;
          line = Serialize(m);
          BroadcastLocked("global", -1, line);
        }
      }
      catch { }
    }


    public void OnGameStart(int threadIndex, GameStartDTO dto)
    {
      if (!enabled)
      {
        return;
      }
      try
      {
        lock (gate)
        {
          ThreadLiveState ts = GetThreadLocked(threadIndex);
          dto.ThreadId = threadIndex;
          dto.Seq = ++ts.Seq;
          ts.CurrentGame = dto;
          ts.Moves.Clear();
          ts.Interim = null;
          ts.LastEnd = null;
          ts.Finished = false;
          BroadcastLocked("thread", threadIndex, Serialize(dto));
        }
      }
      catch { }
    }


    public void OnMove(int threadIndex, MoveDTO dto)
    {
      if (!enabled)
      {
        return;
      }
      try
      {
        lock (gate)
        {
          ThreadLiveState ts = GetThreadLocked(threadIndex);
          dto.ThreadId = threadIndex;
          dto.Seq = ++ts.Seq;
          ts.Moves.Add(dto);
          ts.Interim = null; // the completed move supersedes any pending interim snapshot
          BroadcastLocked("thread", threadIndex, Serialize(dto));
        }
      }
      catch { }
    }


    public void OnInterim(int threadIndex, InterimDTO dto)
    {
      if (!enabled || dto == null)
      {
        return;
      }
      try
      {
        lock (gate)
        {
          ThreadLiveState ts = GetThreadLocked(threadIndex);
          dto.ThreadId = threadIndex;
          dto.Seq = ++ts.Seq;
          ts.Interim = dto; // transient: replaced by the next interim, cleared by the real move
          BroadcastLocked("thread", threadIndex, Serialize(dto));
        }
      }
      catch { }
    }


    public bool WantsInterim(int threadIndex)
    {
      if (!enabled)
      {
        return false;
      }
      lock (gate)
      {
        foreach (StreamSubscriber sub in subscribers.Values)
        {
          if (sub.Matches("thread", threadIndex))
          {
            return true;
          }
        }
      }
      return false;
    }


    public void OnGameEnd(int threadIndex, GameEndDTO dto)
    {
      if (!enabled)
      {
        return;
      }
      try
      {
        lock (gate)
        {
          ThreadLiveState ts = GetThreadLocked(threadIndex);
          dto.Type = "gameEnd";
          dto.ThreadId = threadIndex;
          dto.Seq = ++ts.Seq;
          ts.Interim = null;
          ts.LastEnd = dto;
          ts.Finished = true;
          BroadcastLocked("thread", threadIndex, Serialize(dto));

          // Tournament-global result (drives standings/crosstable across all threads).
          GameEndDTO global = CloneAsResult(dto);
          global.Seq = ++globalSeq;
          results.Add(global);
          BroadcastLocked("global", -1, Serialize(global));
        }
      }
      catch { }
    }


    public void OnTournamentEnd()
    {
      if (!enabled)
      {
        return;
      }
      try
      {
        lock (gate)
        {
          BroadcastLocked("global", -1, Serialize(new TournamentEndDTO
          {
            Type = "tournamentEnd",
            Name = meta?.Name,
            Reason = "completed"
          }));
        }
      }
      catch { }
    }


    public void RegisterOnFirstClient(Action onFirstClient)
    {
      if (onFirstClient == null)
      {
        return;
      }

      bool runNow;
      lock (gate)
      {
        // If a client is already connected, run immediately; otherwise defer until the first connects.
        runNow = anyClientConnected;
        if (!runNow)
        {
          onFirstClientCallbacks.Add(onFirstClient);
        }
      }

      if (runNow)
      {
        try { onFirstClient(); } catch { }
      }
    }

    #endregion


    static GameEndDTO CloneAsResult(GameEndDTO d) => new()
    {
      Type = "gameResult",
      ThreadId = d.ThreadId,   // keep producing thread so consumers can pair games (each thread plays a pair back-to-back)
      WhiteName = d.WhiteName,
      BlackName = d.BlackName,
      Result = d.Result,
      Reason = d.Reason,
      Moves = d.Moves,
      PlyCount = d.PlyCount,
      GameTimeMs = d.GameTimeMs,
      OpeningIndex = d.OpeningIndex
    };


    static string Serialize(WireMsg msg) => JsonSerializer.Serialize(msg, msg.GetType(), JsonOpts);


    static async Task WriteLineDirectAsync(NetworkStream stream, string line)
    {
      byte[] bytes = Encoding.UTF8.GetBytes(line + "\n");
      await stream.WriteAsync(bytes);
      await stream.FlushAsync();
    }


    /// <summary>
    /// Reads a single newline-terminated line directly from the socket (UTF-8), with a timeout.
    /// Returns null on timeout, EOF or error.
    /// </summary>
    static async Task<string> ReadLineRawAsync(NetworkStream stream, TimeSpan timeout)
    {
      using CancellationTokenSource cts = new(timeout);
      MemoryStream buffer = new();
      byte[] one = new byte[1];
      try
      {
        while (true)
        {
          int n = await stream.ReadAsync(one.AsMemory(0, 1), cts.Token);
          if (n == 0)
          {
            return buffer.Length == 0 ? null : Encoding.UTF8.GetString(buffer.ToArray());
          }
          if (one[0] == (byte)'\n')
          {
            string s = Encoding.UTF8.GetString(buffer.ToArray()).TrimEnd('\r');
            return s;
          }
          buffer.WriteByte(one[0]);
          if (buffer.Length > 64 * 1024)
          {
            return null; // request line implausibly long
          }
        }
      }
      catch
      {
        return null;
      }
    }


    public void Dispose()
    {
      disposed = true;
      try { listener?.Stop(); } catch { }
      lock (gate)
      {
        foreach (StreamSubscriber sub in subscribers.Values.ToArray())
        {
          sub.Close();
        }
        subscribers.Clear();
      }
    }
  }
}
