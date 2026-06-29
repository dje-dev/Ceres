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
using System.Net;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Net.Http;
using System.Net.Http.Headers;
using System.IO.Compression;
using System.Collections.Generic;
using System.Text.Json.Serialization;

using Spectre.Console;

using Ceres.Base.Misc;

#endregion

namespace Ceres.Features.NetPublishing
{
  /// <summary>
  /// Automates publishing a Ceres neural-net (.onnx) as a GitHub Release on the
  /// CeresNets repository, producing an artifact that the existing
  /// <see cref="CeresNetDownloader"/> can consume (a single {TAG}.zip asset whose
  /// only entry is {TAG}.onnx, with TAG uppercased).
  ///
  /// Driven by a small JSON config file (see publish-example.json). The GitHub
  /// Personal Access Token is read from the environment variable
  /// CERESNETS_GITHUB_TOKEN (fallback GITHUB_TOKEN); it must have Contents:write
  /// permission on the target repository. No interactive browser login is needed.
  ///
  /// The upload is resilient over slow/flaky links: the release is created as a
  /// draft, the asset is uploaded with retry+backoff (and any stale/partial asset
  /// deleted first), the upload is verified by size, and only then is the release
  /// published. Because GitHub's release-asset upload is a single non-resumable
  /// POST, "recover from a timeout" works by simply re-running the tool: it finds
  /// the existing draft, deletes any partial asset, and retries.
  /// </summary>
  public sealed class CeresNetGitHubUploader
  {
    const string API = "https://api.github.com";
    const string UPLOADS = "https://uploads.github.com";

    static readonly JsonSerializerOptions JSON_OPTS = new JsonSerializerOptions
    {
      AllowTrailingCommas = true,
      PropertyNameCaseInsensitive = true,
      ReadCommentHandling = JsonCommentHandling.Skip,
    };

    readonly UploaderConfig cfg;
    readonly string token;

    // Infinite timeout: a ~300MB upload over ~200KB/s ADSL takes ~25 minutes,
    // far beyond HttpClient's 100s default. Small JSON calls are bounded by an
    // explicit per-call CancellationToken instead (see Send).
    readonly HttpClient client = new HttpClient { Timeout = Timeout.InfiniteTimeSpan };


    CeresNetGitHubUploader(UploaderConfig config, string token)
    {
      this.cfg = config;
      this.token = token;
    }


    #region Public entry points

    /// <summary>
    /// Primary entry point: reads the JSON config at the given path and runs the
    /// full create-release / upload-asset / publish flow.
    /// </summary>
    public static void Run(string jsonConfigPath)
    {
      if (string.IsNullOrWhiteSpace(jsonConfigPath))
      {
        throw new ArgumentException("PUBLISHNET requires Config=<path to JSON config file>.");
      }
      if (!File.Exists(jsonConfigPath))
      {
        throw new FileNotFoundException($"Config file not found: {jsonConfigPath}");
      }

      string json = File.ReadAllText(jsonConfigPath);
      UploaderConfig config = JsonSerializer.Deserialize<UploaderConfig>(json, JSON_OPTS)
                              ?? throw new Exception($"Could not parse config file {jsonConfigPath}.");
      config.Validate();

      new CeresNetGitHubUploader(config, ResolveToken()).Execute();
    }


    /// <summary>
    /// Dev convenience (mirrors CeresNetDownloader.Test()): publishes a hardcoded
    /// example net. Adjust the values to your own net before using.
    /// </summary>
    public static void Test()
    {
      UploaderConfig config = new UploaderConfig
      {
        Owner = "dje-dev",
        Repo = "CeresNets",
        NetID = "C1-640-34-LEPNED-I8",
        Title = "Ceres net C1-640-34-LEPNED-I8",
        Body = "Lepned tune of Ceres net C1-640-34-I8",
        SourceOnnxPath = "/mnt/deve/cout/nets/lepdev_c1_640_34_KL30_body32_PONLY_aug_2500up_folded_trt.onnx",
        Draft = false,
        Prerelease = false,
      };
      config.Validate();
      new CeresNetGitHubUploader(config, ResolveToken()).Execute();
    }

    #endregion


    #region Orchestration

    void Execute()
    {
      IAnsiConsole console = AnsiConsole.Console;

      ConsoleUtils.WriteLineColored(ConsoleColor.White, $"Publishing CeresNet to {cfg.Owner}/{cfg.Repo}");
      Console.WriteLine($"  tag/release : {cfg.Tag}");
      Console.WriteLine($"  asset       : {cfg.AssetName}");
      Console.WriteLine($"  inner entry : {cfg.InnerName}");
      Console.WriteLine($"  source onnx : {cfg.SourceOnnxPath}");
      Console.WriteLine($"  final state : {(cfg.Draft ? "DRAFT (you publish manually)" : "PUBLISHED (auto after verify)")}");

      string tempDir = Directory.CreateTempSubdirectory().FullName;
      try
      {
        string zipPath = BuildZip(tempDir);
        long zipSize = new FileInfo(zipPath).Length;
        ConsoleUtils.WriteLineColored(ConsoleColor.Gray, $"Built {cfg.AssetName}: {zipSize:N0} bytes.");

        ReleaseInfo release = FindOrCreateRelease();
        UploadAssetWithRetry(release, zipPath, zipSize, console);

        if (!cfg.Draft)
        {
          PublishRelease(release.Id);
          if (cfg.VerifyRoundTrip)
          {
            VerifyRoundTrip();
          }
          // Construct the canonical published URL from owner/repo/tag. (release.HtmlUrl was
          // captured at draft-creation time and is an "untagged-..." URL until the release is published.)
          ConsoleUtils.WriteLineColored(ConsoleColor.Green, $"Done. Release published: https://github.com/{cfg.Owner}/{cfg.Repo}/releases/tag/{cfg.Tag}");
          Console.WriteLine($"Downloadable at: https://github.com/{cfg.Owner}/{cfg.Repo}/releases/download/{cfg.Tag}/{cfg.AssetName}");
        }
        else
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Green, $"Done. Asset uploaded to DRAFT release: {release.HtmlUrl}");
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "Release is still a draft — publish it in the GitHub UI to make the net downloadable.");
        }
      }
      finally
      {
        try
        {
          if (Directory.Exists(tempDir))
          {
            foreach (string f in Directory.GetFiles(tempDir))
            {
              File.Delete(f);
            }
            Directory.Delete(tempDir);
          }
        }
        catch
        {
          // best-effort cleanup
        }
      }
    }

    #endregion


    #region Zip build

    /// <summary>
    /// Builds {TAG}.zip in tempDir containing exactly one entry named {TAG}.onnx
    /// (the inner name CeresNetDownloader expects). Uses SmallestSize to minimize
    /// upload bytes over the slow link, at the cost of extra local CPU time.
    /// </summary>
    string BuildZip(string tempDir)
    {
      if (!File.Exists(cfg.SourceOnnxPath))
      {
        throw new FileNotFoundException($"Source onnx not found: {cfg.SourceOnnxPath}");
      }

      string zipPath = Path.Combine(tempDir, cfg.AssetName);
      ConsoleUtils.WriteLineColored(ConsoleColor.Gray, $"Compressing {cfg.SourceOnnxPath} -> {cfg.AssetName} (entry {cfg.InnerName}); this can take a while...");

      using (ZipArchive zip = ZipFile.Open(zipPath, ZipArchiveMode.Create))
      {
        zip.CreateEntryFromFile(cfg.SourceOnnxPath, cfg.InnerName, CompressionLevel.SmallestSize);
      }
      return zipPath;
    }

    #endregion


    #region GitHub REST: release create / publish

    ReleaseInfo FindOrCreateRelease()
    {
      string getUrl = $"{API}/repos/{cfg.Owner}/{cfg.Repo}/releases/tags/{Uri.EscapeDataString(cfg.Tag)}";
      using (HttpRequestMessage req = NewRequest(HttpMethod.Get, getUrl))
      using (HttpResponseMessage resp = Send(req, TimeSpan.FromSeconds(30)))
      {
        if (resp.StatusCode == HttpStatusCode.OK)
        {
          ReleaseInfo existing = ParseJson<ReleaseInfo>(resp);
          ConsoleUtils.WriteLineColored(ConsoleColor.Gray, $"Found existing release for tag {cfg.Tag} (id {existing.Id}, draft={existing.Draft}).");
          return existing;
        }
        if (resp.StatusCode != HttpStatusCode.NotFound)
        {
          ThrowHttp(resp, getUrl);
        }
      }

      // Not found: create. Always create as a draft so a partial upload is never
      // publicly visible; cfg.Draft decides whether we publish at the very end.
      string createUrl = $"{API}/repos/{cfg.Owner}/{cfg.Repo}/releases";
      Dictionary<string, object> payload = new Dictionary<string, object>
      {
        ["tag_name"] = cfg.Tag,
        ["name"] = string.IsNullOrEmpty(cfg.Title) ? cfg.Tag : cfg.Title,
        ["body"] = cfg.Body ?? "",
        ["draft"] = true,
        ["prerelease"] = cfg.Prerelease,
      };
      if (!string.IsNullOrWhiteSpace(cfg.TargetCommitish))
      {
        payload["target_commitish"] = cfg.TargetCommitish;
      }

      using (HttpRequestMessage req = NewRequest(HttpMethod.Post, createUrl))
      {
        req.Content = JsonBody(payload);
        using HttpResponseMessage resp = Send(req, TimeSpan.FromSeconds(30));
        if (!resp.IsSuccessStatusCode)
        {
          ThrowHttp(resp, createUrl);
        }
        ReleaseInfo created = ParseJson<ReleaseInfo>(resp);
        ConsoleUtils.WriteLineColored(ConsoleColor.Gray, $"Created draft release for tag {cfg.Tag} (id {created.Id}).");
        return created;
      }
    }


    void PublishRelease(long releaseId)
    {
      string url = $"{API}/repos/{cfg.Owner}/{cfg.Repo}/releases/{releaseId}";
      using HttpRequestMessage req = NewRequest(HttpMethod.Patch, url);
      req.Content = JsonBody(new Dictionary<string, object> { ["draft"] = false });
      using HttpResponseMessage resp = Send(req, TimeSpan.FromSeconds(30));
      if (!resp.IsSuccessStatusCode)
      {
        ThrowHttp(resp, url);
      }
      ConsoleUtils.WriteLineColored(ConsoleColor.Gray, $"Published release {cfg.Tag}.");
    }

    #endregion


    #region GitHub REST: assets

    List<AssetInfo> ListAssets(long releaseId)
    {
      string url = $"{API}/repos/{cfg.Owner}/{cfg.Repo}/releases/{releaseId}/assets?per_page=100";
      using HttpRequestMessage req = NewRequest(HttpMethod.Get, url);
      using HttpResponseMessage resp = Send(req, TimeSpan.FromSeconds(30));
      if (!resp.IsSuccessStatusCode)
      {
        ThrowHttp(resp, url);
      }
      return ParseJson<List<AssetInfo>>(resp) ?? new List<AssetInfo>();
    }


    void DeleteAsset(long assetId)
    {
      string url = $"{API}/repos/{cfg.Owner}/{cfg.Repo}/releases/assets/{assetId}";
      using HttpRequestMessage req = NewRequest(HttpMethod.Delete, url);
      using HttpResponseMessage resp = Send(req, TimeSpan.FromSeconds(30));
      if (!resp.IsSuccessStatusCode && resp.StatusCode != HttpStatusCode.NotFound)
      {
        ThrowHttp(resp, url);
      }
    }


    AssetInfo FindAsset(long releaseId)
    {
      return ListAssets(releaseId).Find(a => string.Equals(a.Name, cfg.AssetName, StringComparison.Ordinal));
    }


    void UploadAssetWithRetry(ReleaseInfo release, string localZipPath, long localZipSize, IAnsiConsole console)
    {
      // Idempotent fast path: already fully uploaded with the right size.
      AssetInfo existing = FindAsset(release.Id);
      if (existing != null && existing.State == "uploaded" && existing.Size == localZipSize)
      {
        ConsoleUtils.WriteLineColored(ConsoleColor.Green, $"Asset {cfg.AssetName} already uploaded ({localZipSize:N0} bytes); skipping upload.");
        return;
      }

      int attempt = 0;
      while (true)
      {
        attempt++;
        try
        {
          // Delete any pre-existing (possibly partial/stale) asset first — GitHub
          // rejects a duplicate asset name with 422. This is also how a dropped
          // upload from a prior run is "resumed": the partial is removed and resent.
          AssetInfo current = FindAsset(release.Id);
          if (current != null)
          {
            if (current.State == "uploaded" && current.Size == localZipSize)
            {
              return; // finished in a prior attempt
            }
            ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Removing existing asset {cfg.AssetName} (state={current.State}, size={current.Size:N0}) before (re)upload.");
            DeleteAsset(current.Id);
          }

          UploadAssetOnce(release, localZipPath, localZipSize, console);

          // Verify the asset landed completely.
          AssetInfo after = FindAsset(release.Id);
          if (after == null || after.State != "uploaded" || after.Size != localZipSize)
          {
            throw new IOException($"Post-upload verification failed for {cfg.AssetName} "
                                + $"(state={after?.State ?? "missing"}, size={after?.Size ?? -1}, expected {localZipSize}).");
          }

          ConsoleUtils.WriteLineColored(ConsoleColor.Green, $"Uploaded and verified {cfg.AssetName} ({localZipSize:N0} bytes).");
          return;
        }
        catch (Exception ex) when (IsRetryable(ex) && attempt < cfg.MaxUploadAttempts)
        {
          int delaySec = Math.Min(60, 5 * (1 << (attempt - 1)));
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Upload attempt {attempt}/{cfg.MaxUploadAttempts} failed: {ex.Message}");
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Retrying in {delaySec}s (re-running the tool later also resumes safely)...");
          Thread.Sleep(delaySec * 1000);
        }
      }
    }


    void UploadAssetOnce(ReleaseInfo release, string localZipPath, long localZipSize, IAnsiConsole console)
    {
      string uploadUrl = BuildUploadUrl(release);

      void DoPost(ProgressTask task)
      {
        using FileStream fs = new FileStream(localZipPath, FileMode.Open, FileAccess.Read, FileShare.Read, 1 << 16, useAsync: false);
        using Stream body = task == null ? (Stream)fs : new ProgressReadStream(fs, task);
        using HttpRequestMessage req = NewRequest(HttpMethod.Post, uploadUrl);

        StreamContent content = new StreamContent(body, 1 << 16);
        content.Headers.ContentType = new MediaTypeHeaderValue("application/zip");
        content.Headers.ContentLength = localZipSize; // fixed length avoids chunked transfer-encoding
        req.Content = content;

        // No per-call timeout here: the client timeout is infinite for the upload.
        using HttpResponseMessage resp = client.Send(req, HttpCompletionOption.ResponseHeadersRead);
        if (!resp.IsSuccessStatusCode)
        {
          ThrowHttp(resp, uploadUrl);
        }
      }

      if (console == null)
      {
        DoPost(null);
      }
      else
      {
        console.Progress()
               .AutoClear(false)
               .Start(ctx =>
               {
                 float sizeMB = MathF.Round(localZipSize / (1024f * 1024f), 1);
                 ProgressTask task = ctx.AddTask($"[green]Uploading {cfg.AssetName} ({sizeMB:F1} mb)[/]",
                                                 new ProgressTaskSettings { MaxValue = localZipSize });
                 DoPost(task);
               });
      }
    }


    string BuildUploadUrl(ReleaseInfo release)
    {
      // upload_url is a URI template, e.g. ".../assets{?name,label}". Strip the template part.
      string baseUrl;
      string template = release.UploadUrl;
      if (!string.IsNullOrEmpty(template))
      {
        int brace = template.IndexOf('{');
        baseUrl = brace >= 0 ? template.Substring(0, brace) : template;
      }
      else
      {
        baseUrl = $"{UPLOADS}/repos/{cfg.Owner}/{cfg.Repo}/releases/{release.Id}/assets";
      }
      return $"{baseUrl}?name={Uri.EscapeDataString(cfg.AssetName)}";
    }

    #endregion


    #region Round-trip verification (optional)

    /// <summary>
    /// Optional: after publishing, download the net back via the same path
    /// CeresNetDownloader uses and confirm the extracted onnx size matches.
    /// </summary>
    void VerifyRoundTrip()
    {
      ConsoleUtils.WriteLineColored(ConsoleColor.Gray, "Round-trip verifying via CeresNetDownloader...");
      string dir = Directory.CreateTempSubdirectory().FullName;
      try
      {
        CeresNetDownloader downloader = new CeresNetDownloader();
        (bool _, string fullNetworkPath) = downloader.DownloadCeresNetIfNeeded(cfg.Tag, dir, false);

        if (fullNetworkPath == null || !File.Exists(fullNetworkPath))
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, "Round-trip download did not produce a file.");
          return;
        }

        long got = new FileInfo(fullNetworkPath).Length;
        long src = new FileInfo(cfg.SourceOnnxPath).Length;
        if (got != src)
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Yellow, $"Round-trip size mismatch: downloaded {got:N0} vs source {src:N0} bytes.");
        }
        else
        {
          ConsoleUtils.WriteLineColored(ConsoleColor.Green, $"Round-trip verified: downloaded {cfg.Tag}.onnx matches source size ({got:N0} bytes).");
        }
      }
      finally
      {
        try
        {
          foreach (string f in Directory.GetFiles(dir))
          {
            File.Delete(f);
          }
          Directory.Delete(dir);
        }
        catch
        {
          // best-effort
        }
      }
    }

    #endregion


    #region HTTP helpers

    static string ResolveToken()
    {
      string t = Environment.GetEnvironmentVariable("CERESNETS_GITHUB_TOKEN");
      if (string.IsNullOrWhiteSpace(t))
      {
        t = Environment.GetEnvironmentVariable("GITHUB_TOKEN");
      }
      if (string.IsNullOrWhiteSpace(t))
      {
        throw new Exception("No GitHub token found. Set environment variable CERESNETS_GITHUB_TOKEN "
                          + "(or GITHUB_TOKEN) to a Personal Access Token with Contents:write permission "
                          + "(fine-grained) or the 'repo' scope (classic) on the target repository.");
      }
      return t.Trim();
    }


    HttpRequestMessage NewRequest(HttpMethod method, string url)
    {
      HttpRequestMessage req = new HttpRequestMessage(method, url);
      // Set per-request (not on DefaultRequestHeaders) so the shared client never pins the token.
      req.Headers.Authorization = new AuthenticationHeaderValue("Bearer", token);
      req.Headers.Accept.Add(new MediaTypeWithQualityHeaderValue("application/vnd.github+json"));
      req.Headers.Add("X-GitHub-Api-Version", "2022-11-28");
      req.Headers.UserAgent.ParseAdd("Ceres-NetUploader");
      return req;
    }


    HttpResponseMessage Send(HttpRequestMessage req, TimeSpan timeout)
    {
      using CancellationTokenSource cts = new CancellationTokenSource(timeout);
      return client.Send(req, HttpCompletionOption.ResponseHeadersRead, cts.Token);
    }


    static HttpContent JsonBody(object payload)
    {
      string json = JsonSerializer.Serialize(payload);
      return new StringContent(json, Encoding.UTF8, "application/json");
    }


    static T ParseJson<T>(HttpResponseMessage resp)
    {
      string body = resp.Content.ReadAsStringAsync().GetAwaiter().GetResult();
      return JsonSerializer.Deserialize<T>(body, JSON_OPTS);
    }


    static bool IsRetryable(Exception ex)
    {
      if (ex is GitHubApiException g)
      {
        return g.Retryable;
      }
      // Network-level failures and timeouts are all worth retrying.
      return ex is HttpRequestException
          || ex is IOException
          || ex is System.Threading.Tasks.TaskCanceledException
          || ex is OperationCanceledException;
    }


    static void ThrowHttp(HttpResponseMessage resp, string url)
    {
      string body = "";
      try
      {
        body = resp.Content.ReadAsStringAsync().GetAwaiter().GetResult();
      }
      catch
      {
        // ignore body-read failures
      }
      if (body != null && body.Length > 600)
      {
        body = body.Substring(0, 600) + "...";
      }

      int code = (int)resp.StatusCode;
      bool retryable = code == 429 || code >= 500;

      string hint = code switch
      {
        401 => " (token missing/invalid — check CERESNETS_GITHUB_TOKEN)",
        403 => " (token lacks Contents:write on the repo, or you are rate-limited)",
        404 => " (owner/repo not found, or the token cannot see it)",
        422 => " (validation error — e.g. an asset with this name already exists)",
        _ => "",
      };

      throw new GitHubApiException($"HTTP {code} {resp.ReasonPhrase}{hint} for {url}. Body: {body}", resp.StatusCode, retryable);
    }

    #endregion


    #region Nested types

    /// <summary>
    /// User-supplied configuration (deserialized from the JSON config file).
    /// The token is NOT part of this — it comes from the environment.
    /// </summary>
    public sealed class UploaderConfig
    {
      public string Owner { get; set; } = "dje-dev";
      public string Repo { get; set; } = "CeresNets";

      /// <summary>Net identifier, e.g. "C1-640-34-LEPNED-I8". Uppercased to form the tag.</summary>
      public string NetID { get; set; }

      /// <summary>Release title (GitHub release "name").</summary>
      public string Title { get; set; }

      /// <summary>Release description (GitHub release "body").</summary>
      public string Body { get; set; }

      /// <summary>Local path to the source .onnx file to publish.</summary>
      public string SourceOnnxPath { get; set; }

      /// <summary>Desired final state: true = leave as draft (publish manually); false = auto-publish after verify.</summary>
      public bool Draft { get; set; } = false;

      public bool Prerelease { get; set; } = false;

      /// <summary>Optional branch/commit the tag should point at (null = repo default branch).</summary>
      public string TargetCommitish { get; set; }

      public int MaxUploadAttempts { get; set; } = 6;

      /// <summary>If true and publishing, re-download via CeresNetDownloader to confirm consumability (extra download).</summary>
      public bool VerifyRoundTrip { get; set; } = false;

      // Derived (case-sensitive, uppercased) names that CeresNetDownloader requires.
      [JsonIgnore] public string Tag => NetID.ToUpperInvariant();
      [JsonIgnore] public string AssetName => Tag + ".zip";
      [JsonIgnore] public string InnerName => Tag + ".onnx";

      public void Validate()
      {
        if (string.IsNullOrWhiteSpace(NetID))
        {
          throw new Exception("Config: 'netID' is required.");
        }
        if (string.IsNullOrWhiteSpace(SourceOnnxPath))
        {
          throw new Exception("Config: 'sourceOnnxPath' is required.");
        }
        if (string.IsNullOrWhiteSpace(Owner))
        {
          Owner = "dje-dev";
        }
        if (string.IsNullOrWhiteSpace(Repo))
        {
          Repo = "CeresNets";
        }
        if (MaxUploadAttempts <= 0)
        {
          MaxUploadAttempts = 6;
        }
      }
    }


    sealed class ReleaseInfo
    {
      [JsonPropertyName("id")] public long Id { get; set; }
      [JsonPropertyName("upload_url")] public string UploadUrl { get; set; }
      [JsonPropertyName("html_url")] public string HtmlUrl { get; set; }
      [JsonPropertyName("draft")] public bool Draft { get; set; }
      [JsonPropertyName("tag_name")] public string TagName { get; set; }
    }


    sealed class AssetInfo
    {
      [JsonPropertyName("id")] public long Id { get; set; }
      [JsonPropertyName("name")] public string Name { get; set; }
      [JsonPropertyName("state")] public string State { get; set; }
      [JsonPropertyName("size")] public long Size { get; set; }
    }


    sealed class GitHubApiException : Exception
    {
      public HttpStatusCode Status { get; }
      public bool Retryable { get; }

      public GitHubApiException(string message, HttpStatusCode status, bool retryable) : base(message)
      {
        Status = status;
        Retryable = retryable;
      }
    }


    /// <summary>
    /// Read-only pass-through stream that advances a Spectre ProgressTask as bytes
    /// are pulled off the underlying file by the HTTP stack, so the progress bar
    /// tracks bytes actually sent over the wire.
    /// </summary>
    sealed class ProgressReadStream : Stream
    {
      readonly Stream inner;
      readonly ProgressTask task;

      public ProgressReadStream(Stream inner, ProgressTask task)
      {
        this.inner = inner;
        this.task = task;
      }

      public override int Read(byte[] buffer, int offset, int count)
      {
        int n = inner.Read(buffer, offset, count);
        if (n > 0)
        {
          task.Increment(n);
        }
        return n;
      }

      public override int Read(Span<byte> buffer)
      {
        int n = inner.Read(buffer);
        if (n > 0)
        {
          task.Increment(n);
        }
        return n;
      }

      public override bool CanRead => true;
      public override bool CanSeek => false;
      public override bool CanWrite => false;
      public override long Length => inner.Length;
      public override long Position { get => inner.Position; set => throw new NotSupportedException(); }
      public override void Flush() { }
      public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
      public override void SetLength(long value) => throw new NotSupportedException();
      public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    #endregion
  }
}
