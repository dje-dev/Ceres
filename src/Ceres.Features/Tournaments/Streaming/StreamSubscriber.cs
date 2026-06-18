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
using System.Text;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

#endregion

namespace Ceres.Features.Tournaments.Streaming
{
  /// <summary>
  /// Represents a single connected subscriber. Owns a bounded queue of pending NDJSON lines
  /// and a dedicated writer task that drains them to the socket. The tournament worker threads
  /// only ever enqueue (a fast, non-blocking operation); all blocking socket I/O happens on the
  /// writer task, so a slow or dead client can never stall the tournament. If the queue overflows
  /// (a client that cannot keep up) the subscriber is dropped rather than blocking producers.
  /// </summary>
  internal sealed class StreamSubscriber
  {
    public readonly Guid Id = Guid.NewGuid();

    /// <summary>"global" (tournament-wide results) or "thread" (one game thread's live feed).</summary>
    public readonly string Scope;

    /// <summary>Subscribed thread id (only meaningful when Scope == "thread").</summary>
    public readonly int ThreadId;

    const int QUEUE_CAPACITY = 20_000;

    readonly TcpClient client;
    readonly NetworkStream stream;
    readonly Channel<string> channel;
    readonly CancellationTokenSource cts = new();
    readonly Action<StreamSubscriber> onClosed;
    int closed;

    public StreamSubscriber(TcpClient client, NetworkStream stream, string scope, int threadId,
                            Action<StreamSubscriber> onClosed)
    {
      this.client = client;
      this.stream = stream;
      Scope = scope;
      ThreadId = threadId;
      this.onClosed = onClosed;

      channel = Channel.CreateBounded<string>(new BoundedChannelOptions(QUEUE_CAPACITY)
      {
        SingleReader = true,
        SingleWriter = false,
        FullMode = BoundedChannelFullMode.Wait
      });

      _ = Task.Run(WriterLoopAsync);
    }

    public bool Matches(string scope, int threadId)
      => Scope == scope && (scope != "thread" || ThreadId == threadId);

    /// <summary>
    /// Enqueues a line for delivery. Returns false (and drops the subscriber) if the queue is full.
    /// Never blocks.
    /// </summary>
    public void Enqueue(string line)
    {
      if (Volatile.Read(ref closed) != 0)
      {
        return;
      }
      if (!channel.Writer.TryWrite(line))
      {
        // Client cannot keep up; drop it rather than stall producers.
        Close();
      }
    }

    async Task WriterLoopAsync()
    {
      try
      {
        await foreach (string line in channel.Reader.ReadAllAsync(cts.Token))
        {
          byte[] bytes = Encoding.UTF8.GetBytes(line + "\n");
          await stream.WriteAsync(bytes, cts.Token);
          await stream.FlushAsync(cts.Token);
        }
      }
      catch
      {
        // Socket error or cancellation: fall through to cleanup.
      }
      finally
      {
        Close();
      }
    }

    public void Close()
    {
      if (Interlocked.Exchange(ref closed, 1) != 0)
      {
        return;
      }

      try { channel.Writer.TryComplete(); } catch { }
      try { cts.Cancel(); } catch { }
      try { stream?.Dispose(); } catch { }
      try { client?.Close(); } catch { }
      try { onClosed?.Invoke(this); } catch { }
    }
  }
}
