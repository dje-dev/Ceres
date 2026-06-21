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
using System.Net.Http;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

#endregion

namespace Ceres.Features.TCEC;

/// <summary>
/// Retrieval of the current chess position from the TCEC live broadcast feed.
///
/// This is the headless fetch core: it performs the HTTP GET against the TCEC
/// live.json endpoint and parses the payload into an immutable TCECLiveSnapshot
/// (which exposes the most-recent position as a Ceres.Chess PositionWithHistory).
/// It contains no console/UI or settings dependencies; presentation and the
/// engine-driven monitoring loop live in TCECMonitor.
/// </summary>
public static class TCECLiveFeed
{
  /// <summary>
  /// Default TCEC live game state endpoint (JSON).
  /// </summary>
  public const string DEFAULT_URL = "https://tcec-chess.com/live.json";

  const int MAX_BACKOFF_SECONDS = 30;

  static readonly HttpClient httpClient = new HttpClient
  {
    Timeout = TimeSpan.FromSeconds(15),
  };

  /// <summary>
  /// Performs a single fetch+parse of the TCEC live feed.
  /// Returns null (and reports via onWarn) on any transient network/JSON error,
  /// allowing the caller to retry; the snapshot's FromJson is what sets the
  /// global MGPositionConstants.IsChess960 flag from the game variant.
  /// </summary>
  public static TCECLiveSnapshot? FetchOnce(CancellationToken ct,
                                            string apiUrl = DEFAULT_URL,
                                            Action<string> onWarn = null)
  {
    try
    {
      string body = httpClient.GetStringAsync(apiUrl, ct).GetAwaiter().GetResult();
      if (string.IsNullOrWhiteSpace(body))
      {
        onWarn?.Invoke("empty response");
        return null;
      }
      return TCECLiveSnapshot.FromJson(body);
    }
    catch (OperationCanceledException) when (ct.IsCancellationRequested)
    {
      throw;
    }
    catch (TaskCanceledException ex)
    {
      onWarn?.Invoke("timeout: " + ex.Message);
      return null;
    }
    catch (HttpRequestException ex)
    {
      onWarn?.Invoke("http: " + ex.Message);
      return null;
    }
    catch (JsonException ex)
    {
      onWarn?.Invoke("json: " + ex.Message);
      return null;
    }
    catch (IOException ex)
    {
      onWarn?.Invoke("io: " + ex.Message);
      return null;
    }
    catch (Exception ex)
    {
      onWarn?.Invoke("unexpected " + ex.GetType().Name + ": " + ex.Message);
      return null;
    }
  }

  /// <summary>
  /// Fetches the latest TCEC snapshot, retrying with exponential backoff until a
  /// valid payload is obtained or cancellation is requested. Returns null only if
  /// cancelled before any successful fetch.
  /// </summary>
  public static TCECLiveSnapshot? FetchLatestSnapshot(CancellationToken ct,
                                                      string apiUrl = DEFAULT_URL,
                                                      Action<string> onWarn = null)
  {
    int backoffSec = 2;
    while (!ct.IsCancellationRequested)
    {
      TCECLiveSnapshot? snap = FetchOnce(ct, apiUrl, msg =>
      {
        onWarn?.Invoke(msg + "  (retry in " + backoffSec + "s)");
      });

      if (snap != null)
      {
        return snap;
      }

      if (SleepInterruptible(backoffSec, ct))
      {
        break;
      }
      backoffSec = Math.Min(backoffSec * 2, MAX_BACKOFF_SECONDS);
    }
    return null;
  }

  /// <summary>
  /// Sleeps up to the given number of seconds, returning true if cancellation
  /// was requested during the wait.
  /// </summary>
  static bool SleepInterruptible(int seconds, CancellationToken ct)
  {
    try
    {
      return ct.WaitHandle.WaitOne(TimeSpan.FromSeconds(seconds));
    }
    catch
    {
      return ct.IsCancellationRequested;
    }
  }
}
